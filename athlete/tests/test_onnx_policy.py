import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from athlete.scripts.onnx_policy import OnnxPlayPolicy


class _ValueInfo:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape


def _fake_onnxruntime(output_dim: int = 3):
    module = types.ModuleType("onnxruntime")

    class FakeSession:
        latest: "FakeSession | None" = None

        def __init__(self, path: str, providers: list[str]) -> None:
            self.path = path
            self.providers = providers
            self.feeds: dict[str, np.ndarray] | None = None
            FakeSession.latest = self

        def get_inputs(self):
            return [
                _ValueInfo("obs", [1, 4]),
                _ValueInfo("which_motion", [1, 1]),
                _ValueInfo("time_step", [1, 1]),
            ]

        def get_outputs(self):
            return [_ValueInfo("actions", [1, output_dim])]

        def get_providers(self):
            return self.providers

        def run(self, output_names: list[str], feeds: dict[str, np.ndarray]):
            self.feeds = feeds
            assert output_names == ["actions"]
            return [np.arange(output_dim, dtype=np.float32).reshape(1, -1)]

    module.get_available_providers = lambda: ["CPUExecutionProvider"]
    module.InferenceSession = FakeSession
    return module, FakeSession


def _environment(num_actions: int = 3):
    motion = SimpleNamespace(
        which_motion=torch.tensor([7]),
        time_steps=torch.tensor([11]),
    )
    command_manager = SimpleNamespace(
        active_terms=("motion",),
        get_term=lambda name: motion,
    )
    env = SimpleNamespace(
        num_envs=1,
        num_actions=num_actions,
        device="cpu",
        unwrapped=SimpleNamespace(command_manager=command_manager),
    )
    return env


class OnnxPlayPolicyTest(unittest.TestCase):
    def test_feeds_actor_and_motion_state_and_returns_torch_actions(self) -> None:
        fake_ort, fake_session = _fake_onnxruntime()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.touch()
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                policy = OnnxPlayPolicy(path, _environment())
                actions = policy({"actor": torch.tensor([[1.0, 2.0, 3.0, 4.0]])})

        np.testing.assert_array_equal(actions.numpy(), [[0.0, 1.0, 2.0]])
        session = fake_session.latest
        assert session is not None and session.feeds is not None
        np.testing.assert_array_equal(session.feeds["obs"], [[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(session.feeds["which_motion"], [[7.0]])
        np.testing.assert_array_equal(session.feeds["time_step"], [[11.0]])
        self.assertEqual(policy.providers, ("CPUExecutionProvider",))

    def test_rejects_action_dimension_mismatch(self) -> None:
        fake_ort, _ = _fake_onnxruntime(output_dim=4)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.touch()
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                with self.assertRaisesRegex(ValueError, "outputs 4 actions"):
                    OnnxPlayPolicy(path, _environment(num_actions=3))

    def test_student_observation_and_zero_padded_environment_action(self) -> None:
        fake_ort, fake_session = _fake_onnxruntime(output_dim=2)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "student.onnx"
            path.touch()
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                policy = OnnxPlayPolicy(
                    path,
                    _environment(num_actions=3),
                    observation_group="student",
                    zero_pad_actions=1,
                )
                actions = policy(
                    {"student": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
                )

        np.testing.assert_array_equal(actions.numpy(), [[0.0, 1.0, 0.0]])
        self.assertEqual(policy.obs_groups, ("student",))
        session = fake_session.latest
        assert session is not None and session.feeds is not None
        np.testing.assert_array_equal(
            session.feeds["obs"], [[1.0, 2.0, 3.0, 4.0]]
        )


if __name__ == "__main__":
    unittest.main()
