# Source this file from zsh to load the local ROS 2 Jazzy deployment environment.
if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
  print -u2 'Usage: source deploy/setup_deploy.zsh'
  return 1
fi

typeset deploy_script="${(%):-%N}"
export DEPLOY_ROOT_DIR="${deploy_script:A:h:h}"
export TPPO_RECORDING_DIR="${TPPO_RECORDING_DIR:-$DEPLOY_ROOT_DIR/deploy/recordings}"
typeset deploy_venv="$DEPLOY_ROOT_DIR/deploy/.venv"

if [[ ! -f /opt/ros/jazzy/setup.zsh ]]; then
  print -u2 'ROS 2 Jazzy was not found at /opt/ros/jazzy/setup.zsh'
  return 1
fi

if (( $+functions[deactivate] )); then
  deactivate
fi

source /opt/ros/jazzy/setup.zsh

typeset optitrack_ws="${OPTITRACK_COMPANION_WS:-$HOME/optitrack_companion_ws}"
if [[ -f "$optitrack_ws/install/setup.zsh" ]]; then
  source "$optitrack_ws/install/setup.zsh"
fi

if [[ -f "$deploy_venv/bin/activate" ]]; then
  source "$deploy_venv/bin/activate"
else
  print -u2 "Deploy venv is not installed: $deploy_venv"
  print -u2 'Create it with: bash deploy/setup_jazzy_env.sh'
fi

cd "$DEPLOY_ROOT_DIR/deploy"
print "Deploy environment: ROS 2 $ROS_DISTRO, Python $(python3 --version 2>&1)"
