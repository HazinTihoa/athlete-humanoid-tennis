#!/usr/bin/env bash
# Source this file so the ROS and Python environment remains active in the caller.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source deploy/setup_deploy.sh" >&2
  exit 1
fi

DEPLOY_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_VENV="$DEPLOY_ROOT_DIR/deploy/.venv"
ROS_SETUP=/opt/ros/jazzy/setup.bash

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS 2 Jazzy was not found at $ROS_SETUP" >&2
  return 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]] && command -v deactivate >/dev/null 2>&1; then
  deactivate
fi

source "$ROS_SETUP"

if [[ -f "$DEPLOY_VENV/bin/activate" ]]; then
  source "$DEPLOY_VENV/bin/activate"
else
  echo "Deploy venv is not installed: $DEPLOY_VENV" >&2
  echo "Create it with: bash deploy/setup_jazzy_env.sh" >&2
fi

export DEPLOY_ROOT_DIR
cd "$DEPLOY_ROOT_DIR/deploy"
echo "Deploy environment: ROS 2 $ROS_DISTRO, Python $(python3 --version 2>&1)"
