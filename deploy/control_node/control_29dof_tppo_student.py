"""Backend-selectable entry point for the shared TPPO Student controller."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_control_node import run_student_control_node


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared TPPO Student controller.")
    parser.add_argument("--backend", choices=("simulation", "hardware"), required=True)
    parser.add_argument("--config", required=True, help="Deployment YAML path or name.")
    parser.add_argument("--enable-command", action="store_true")
    parser.add_argument("--shadow-command", action="store_true")
    args = parser.parse_args()
    if args.enable_command and args.shadow_command:
        parser.error("--enable-command and --shadow-command are mutually exclusive")
    if args.backend == "simulation" and (args.enable_command or args.shadow_command):
        parser.error("simulation backend does not accept hardware command flags")
    if (
        args.backend == "hardware"
        and args.enable_command
        and os.environ.get("TPPO_ENABLE_REAL_ROBOT") != "YES"
    ):
        parser.error("--enable-command requires: export TPPO_ENABLE_REAL_ROBOT=YES")
    run_student_control_node(
        backend=args.backend,
        config_path=args.config,
        publish_commands=(
            args.backend == "simulation"
            or args.enable_command
            or args.shadow_command
        ),
    )


if __name__ == "__main__":
    main()
