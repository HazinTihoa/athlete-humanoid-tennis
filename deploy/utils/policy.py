##
#
# Policy class for handling policy-related operations, 
# such as loading, inferencing, and getting properties.
#
##

import numpy as np
import onnx
import onnxruntime as ort

# torch is only required for .pt policies; ONNX deployment (e.g. onboard the
# Jetson, where no aarch64 torch wheel is available) does not need it.
try:
    import torch
except ImportError:
    torch = None


############################################################################
# HELPERS
############################################################################

# get the input and output dimensions of a torch policy
def get_policy_io_size_torch(policy, max_input_size=1024):

    # set the policy to eval mode
    policy.eval()

    # input size search
    input_size = None
    for input_size in range(1, max_input_size + 1):
        try:
            dummy = torch.zeros(1, input_size)
            with torch.no_grad():
                policy(dummy)
            break
        except Exception:
            continue
    
    # query the output size
    with torch.no_grad():
        output = policy(torch.zeros(1, input_size))
    output_size = output.shape[-1]

    return input_size, output_size

# get the input and output dimensions of an onnx policy
def get_policy_io_size_onnx(policy):

    # input size from first input tensor
    input_shape = policy.graph.input[0].type.tensor_type.shape
    input_size = input_shape.dim[-1].dim_value

    # output size from first output tensor
    output_shape = policy.graph.output[0].type.tensor_type.shape
    output_size = output_shape.dim[-1].dim_value

    return input_size, output_size


# parse a CSV string into a list of floats
def parse_float_csv(s):
    return np.array([float(x) for x in s.split(",") if x.strip()], dtype=np.float32)

# parse a CSV string into a list of strings
def parse_str_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]

# load metadata embedded in an ONNX model
def load_policy_metadata(onnx_model):
    metadata = {}
    for prop in onnx_model.metadata_props:
        value = prop.value

        # try parsing as floats first
        try:
            parsed = parse_float_csv(value)
            if len(parsed) > 0:
                metadata[prop.key] = parsed
                continue
        except ValueError:
            pass

        # try parsing as string list (if it contains commas)
        if "," in value:
            metadata[prop.key] = parse_str_csv(value)
        else:
            metadata[prop.key] = value.strip()

    return metadata


# inference with a torch policy
def policy_inference_torch(policy, input):

    # convert to torch tensor and add batch dimension
    input_tensor = torch.from_numpy(input).unsqueeze(0)

    # forward pass (no_grad disables autograd for faster inference)
    with torch.no_grad():
        action = policy(input_tensor).numpy().squeeze()

    return action

# inference with an onnx policy
def policy_inference_onnx(session, input, **extra_inputs):

    # build input feed starting with the primary observation
    input_name = session.get_inputs()[0].name
    input_feed = {input_name: input.reshape(1, -1).astype(np.float32)}

    # fill in any additional required inputs (e.g., time_step)
    for inp in session.get_inputs()[1:]:
        if inp.name in extra_inputs:
            val = np.array(extra_inputs[inp.name], dtype=np.float32).reshape(1, -1)
        else:
            shape = [max(d, 1) for d in inp.shape]
            val = np.zeros(shape, dtype=np.float32)
        input_feed[inp.name] = val

    # forward pass
    action = session.run(None, input_feed)[0].squeeze()

    return action


############################################################################
# POLICY CLASS
############################################################################

class Policy:
    """
    Generic control policy class that can handle both PyTorch and ONNX policies.
    """

    def __init__(self, policy_path):

        # load the policy
        self._load_policy(policy_path)

        # compute important properties of the policy
        self._get_policy_properties()

        
    # load a policy given the path
    def _load_policy(self, policy_path):

        # torch file
        if "pt" in policy_path.lower():
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            self._policy_type = "torch"

        # onnx file
        elif "onnx" in policy_path.lower():
            self.policy = onnx.load(policy_path)
            self._onnx_session = ort.InferenceSession(
                self.policy.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            self._policy_type = "onnx"

            # load embedded metadata if available
            if self.policy.metadata_props:
                self.metadata = load_policy_metadata(self.policy)
        # incompatible file
        else:
            raise ValueError("Unsupported policy format. Please use .pt or .onnx files.")


    # get important properties of the policy
    def _get_policy_properties(self):
        # I/O names and sizes
        self.input_size = None
        self.output_size = None
        self.inputs = []
        self.outputs = []
        if self._policy_type == "torch":
            self.input_size, self.output_size = get_policy_io_size_torch(self.policy)
        elif self._policy_type == "onnx":
            self.input_size, self.output_size = get_policy_io_size_onnx(self.policy)
            self.inputs = [{"name": inp.name, "shape": inp.shape} for inp in self._onnx_session.get_inputs()]
            self.outputs = [{"name": out.name, "shape": out.shape} for out in self._onnx_session.get_outputs()]
            self.input_sizes = [inp.shape[-1] for inp in self._onnx_session.get_inputs()]

    # inference the policy given an input
    def inference(self, input, **extra_inputs):
        if self._policy_type == "torch":
            return policy_inference_torch(self.policy, input)
        elif self._policy_type == "onnx":
            return policy_inference_onnx(self._onnx_session, input, **extra_inputs)


############################################################################
# TEST POLICY
############################################################################

def main(args=None):
    
    import os
    ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")

    # specify the policy name
    # policy_name = "g1_23dof_vel.onnx"
    # policy_name = "g1_29dof_vel.onnx"
    # policy_name = "g1_29dof_mimic_pelvis.onnx"
    # policy_name = "g1_29dof_mimic_torso.onnx"
    policy_name = "g1_29dof_mimic_squat.onnx"

    # load the policy
    policy_path = ROOT_DIR + "/policy/" + policy_name
    policy = Policy(policy_path)
    print(f"Policy loaded from [{policy_path}]")
    print(f"    Type: {policy._policy_type}")
    print(f"    Input size: {policy.input_size}")
    print(f"    Output size: {policy.output_size}")
    print(f"    Inputs: {policy.inputs}")
    print(f"    Outputs: {policy.outputs}")

    # print metadata if available
    if hasattr(policy, 'metadata'):
        for key, value in policy.metadata.items():
            print(f"{key}: {value}")

    # test inference with a zero input
    obs = np.zeros(policy.input_size, dtype=np.float32)
    action = policy.inference(obs)
    print(f"    Test action shape: {action.shape}")
    print(f"    Test action: {action}")


if __name__ == "__main__":
    main()