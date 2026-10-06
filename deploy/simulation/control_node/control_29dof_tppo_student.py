"""Simulation entry point for the shared TPPO Student ROS controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOY_DIR))

from utils.tppo_control_node import run_student_control_node


def main() -> None:
    parser = argparse.ArgumentParser(description="TPPO Student simulation controller.")
    parser.add_argument("--config", required=True, help="Deployment YAML path or name.")
    args = parser.parse_args()
    run_student_control_node(
        backend="simulation",
        config_path=args.config,
        publish_commands=True,
    )


if __name__ == "__main__":
    main()
