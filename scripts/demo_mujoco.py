"""Local Student ONNX demo using the training environment, without ROS."""

from __future__ import annotations

import argparse
from pathlib import Path

from athlete.scripts.play_student_onnx import StudentOnnxPlayConfig, run_student_onnx


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-policy-file", type=Path,
                        default=root / "deploy/policies/gpu3_2_model45000/model_45000.onnx",
                        help="Student ONNX model (default: included GPU3-2 model_45000).")
    parser.add_argument("--headless", action="store_true", help="Run without a window.")
    parser.add_argument("--duration-s", type=float, default=None, help="Stop after this many simulated seconds (headless default: 20).")
    parser.add_argument("--no-realtime", action="store_true", help="Run as fast as possible.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual", action="store_true", help="Open a separate court plot; click a robot-side first-bounce target to launch.")
    args = parser.parse_args()
    if args.headless and args.manual:
        parser.error("--manual requires a window")
    data = root / "artifacts/rollout_1000epis_relabelled"
    missing = [f"ep_{i:04d}{suffix}" for i in range(200) for suffix in (".npz", ".json") if not (data / f"ep_{i:04d}{suffix}").is_file()]
    if missing:
        parser.error(
            f"Reference data missing under {data} (first missing file: {missing[0]}). "
            "Download ep_0000 through ep_0199 (NPZ + JSON) from "
            "https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K"
        )
    policy = args.onnx_policy_file.expanduser().resolve()
    if not policy.is_file():
        parser.error(f"ONNX policy not found: {policy}")
    print(f"[DEMO POLICY] {policy}", flush=True)
    run_student_onnx(StudentOnnxPlayConfig(
        onnx_policy_file=policy,
        motion_config=root / "athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml",
        court_preset="small",
        landing_target_x=5.0,
        landing_target_y=0.0,
        landing_target_observation_frame="current_root",
        sweet_spot_velocity_mode="relative_joint",
        seed=args.seed,
        headless=args.headless,
        duration_s=args.duration_s if args.duration_s is not None else (20.0 if args.headless else None),
        realtime=not args.no_realtime,
        manual_mode=args.manual,
        manual_mouse_select=args.manual,
    ))


if __name__ == "__main__":
    main()
