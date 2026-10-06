from __future__ import annotations

import numpy as np

from athlete.motion_sets.time_resampling import (
    discrete_contact_times,
    generate_time_warp_dataset,
    resample_motion_arrays,
)


def _synthetic_motion() -> dict[str, np.ndarray]:
    frames = 16
    fps = 10
    time = np.arange(frames, dtype=np.float32) / fps
    joint_pos = time[:, None]
    body_pos = np.zeros((frames, 1, 3), dtype=np.float32)
    body_pos[:, 0, 0] = time
    body_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    return {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": np.ones_like(joint_pos),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros_like(body_pos),
        "body_ang_vel_w": np.zeros_like(body_pos),
    }


def test_discrete_contact_times_stay_on_grid_and_below_nominal() -> None:
    times = discrete_contact_times(1.0, 148 / 50, 0.1)

    assert len(times) == 20
    assert times[0] == 1.0
    assert times[-1] == 2.9
    np.testing.assert_allclose(np.diff(times), 0.1)


def test_resampling_preserves_start_strike_and_end() -> None:
    source = _synthetic_motion()
    resampled, speed_scale = resample_motion_arrays(
        source, source_strike_frame=10, target_strike_frame=5
    )

    assert speed_scale == 2.0
    np.testing.assert_allclose(resampled["joint_pos"][0], source["joint_pos"][0])
    np.testing.assert_allclose(resampled["joint_pos"][5], source["joint_pos"][10])
    np.testing.assert_allclose(resampled["joint_pos"][-1], source["joint_pos"][-1])
    # The final source frame is appended exactly; its residual interval may be
    # shorter than one full warped step, so only the uniform section is 2x.
    np.testing.assert_allclose(resampled["joint_vel"][:-2], 2.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.linalg.norm(resampled["body_quat_w"], axis=-1), 1.0, atol=1.0e-6
    )


def test_dataset_generation_keeps_sources_shorter_than_minimum(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _synthetic_motion()
    for index, strike_frame in enumerate((8, 10)):
        path = source_dir / f"ep_{index:04d}.npz"
        np.savez_compressed(path, **source)
        path.with_suffix(".json").write_text(
            '{"strike_frame": %d, "clip": "fh_test", "ball_local": [0, 0, 0]}\n'
            % strike_frame,
            encoding="utf-8",
        )

    manifest = generate_time_warp_dataset(
        str(source_dir / "ep_*.npz"),
        tmp_path / "output",
        min_contact_time_s=1.0,
        interval_s=0.2,
    )

    assert manifest["source_count"] == 2
    assert manifest["variant_count"] == 2
    assert manifest["short_sources_kept_at_nominal_time"] == ["ep_0000.npz"]
    assert (tmp_path / "output" / "ep_0000_t0800ms.npz").exists()
    assert (tmp_path / "output" / "ep_0001_t1000ms.npz").exists()
    for sidecar in (tmp_path / "output").glob("ep_*.json"):
        assert __import__("json").loads(sidecar.read_text())["sampling_weight"] == 1.0
