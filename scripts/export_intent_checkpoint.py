"""Export an exact Intent checkpoint without constructing a training environment."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import mjlab  # Initialize task registration before importing the RL package.
import onnx
import onnxruntime as ort
import torch
import yaml
from tensordict import TensorDict

from athlete.goal_cond_tracking.rl.intent_transformer import IntentTransformerActor


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--metadata-onnx", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    config = yaml.load((args.checkpoint.parent / "params/agent.yaml").read_text(), Loader=ConfigLoader)
    cfg = dict(config["actor"])
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        cfg.pop(key, None)
    dims = {
        "student": state["obs_normalizer._mean"].shape[-1],
        cfg["state_history_group"]: cfg["state_history_steps"] * cfg["state_token_dim"],
        cfg["ball_history_group"]: cfg["ball_history_steps"] * cfg["ball_token_dim"],
        cfg["reference_future_group"]: cfg["reference_steps"] * cfg["reference_token_dim"],
    }
    obs = TensorDict({k: torch.zeros(1, n) for k, n in dims.items()}, batch_size=[1])
    actor = IntentTransformerActor(obs, {"actor": ["student"]}, "actor", 29, **cfg)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    wrapper = actor.as_onnx(verbose=False).eval()
    output = args.checkpoint.with_suffix(".onnx")
    if output.exists():
        raise FileExistsError(output)
    torch.onnx.export(wrapper, wrapper.get_dummy_inputs(), str(output),
                      input_names=wrapper.input_names, output_names=wrapper.output_names,
                      opset_version=18, dynamo=False)
    source = onnx.load(args.metadata_onnx)
    model = onnx.load(output)
    metadata = {p.key: p.value for p in source.metadata_props}
    metadata["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    metadata["checkpoint_iteration"] = str(checkpoint["iter"])
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    torch.manual_seed(42)
    max_error = 0.0
    for _ in range(8):
        x = torch.randn(1, wrapper.input_size) * .2
        with torch.no_grad():
            expected = wrapper(x).numpy()
        actual = session.run(["actions"], {"obs": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    print(f"EXPORTED {output} input={wrapper.input_size} action=29 max_error={max_error:.8g}")


if __name__ == "__main__":
    main()
