"""Real-robot entry point for the shared TPPO Student ROS controller."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_control_node import run_student_control_node


def main() -> None:
    parser = argparse.ArgumentParser(description="TPPO Student hardware controller.")
    parser.add_argument("--config", required=True, help="Deployment YAML path or name.")
    parser.add_argument(
        "--enable-command",
        action="store_true",
        help="Publish commands to the Unitree hardware node. Default is dry-run.",
    )
    parser.add_argument(
        "--shadow-command",
        action="store_true",
        help="Publish the 145-D command while the hardware node remains read-only.",
    )
    args = parser.parse_args()
    if args.enable_command and args.shadow_command:
        parser.error("--enable-command and --shadow-command are mutually exclusive")
    if args.enable_command and os.environ.get("TPPO_ENABLE_REAL_ROBOT") != "YES":
        parser.error(
            "--enable-command requires: export TPPO_ENABLE_REAL_ROBOT=YES"
        )
    run_student_control_node(
        backend="hardware",
        config_path=args.config,
        publish_commands=args.enable_command or args.shadow_command,
    )


if __name__ == "__main__":
    main()
