#!/usr/bin/env bash
set -eo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$DEPLOY_DIR/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage:
  bash run_tppo.sh sim CONFIG [SIM_OPTIONS] [--record]
  bash run_tppo.sh real CONFIG NETWORK [readonly|shadow|enable] [--record]

Examples:
  bash run_tppo.sh sim g1_tppo_student_intentref.yaml
  bash run_tppo.sh real g1_tppo_student_intentref.yaml enp3s0 shadow --record
EOF
}

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ "$1" == "-h" || "$1" == "--help" ]] && exit 0
  exit 2
fi

BACKEND="$1"
CONFIG="$2"
shift 2

RECORD_ENABLED=false
REMAINING_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--record" ]]; then
    RECORD_ENABLED=true
  else
    REMAINING_ARGS+=("$arg")
  fi
done
set -- "${REMAINING_ARGS[@]}"

if [[ "$BACKEND" != "sim" && "$BACKEND" != "real" ]]; then
  echo "Backend must be sim or real." >&2
  usage >&2
  exit 2
fi

if [[ ! -x "$PYTHON" || "${ROS_DISTRO:-}" != "jazzy" ]]; then
  echo "Run: source deploy/setup_deploy.zsh" >&2
  exit 1
fi

if [[ "$BACKEND" == "sim" ]]; then
  SIM_ARGS=("$@")
else
  if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Real mode requires NETWORK and optionally readonly, shadow, or enable." >&2
    usage >&2
    exit 2
  fi
  NETWORK="$1"
  COMMAND_MODE="${2:-readonly}"
  if [[ "$COMMAND_MODE" != "readonly" && "$COMMAND_MODE" != "shadow" && "$COMMAND_MODE" != "enable" ]]; then
    echo "Real command mode must be readonly, shadow, or enable." >&2
    exit 2
  fi
  if [[ "$COMMAND_MODE" == "enable" && "${TPPO_ENABLE_REAL_ROBOT:-}" != "YES" ]]; then
    echo "Refusing to arm: export TPPO_ENABLE_REAL_ROBOT=YES first." >&2
    exit 1
  fi
fi

pids=()

start_background() {
  "$@" &
  pids+=("$!")
}

start_background_tty() {
  "$@" </dev/tty &
  pids+=("$!")
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do
    kill -CONT "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 0.5
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$DEPLOY_DIR"

if [[ "$RECORD_ENABLED" == true ]]; then
  RECORDING_ROOT="${TPPO_RECORDING_DIR:-$DEPLOY_DIR/recordings}"
  CONFIG_NAME="$(basename "$CONFIG")"
  CONFIG_NAME="${CONFIG_NAME%.yaml}"
  BAG_PATH="$RECORDING_ROOT/$(date +%Y-%m-%d_%H-%M-%S)_${BACKEND}_${CONFIG_NAME}"
  RECORD_TOPICS=(
    /rigid_bodies
    /g1_pelvis/pose
    /ball/pose
    /ball/velocity
    /ball/track_epoch
    /mocap_policy_bridge/status
    /deploy_robot/pelvis_imu_state
    /deploy_robot/joint_state
    /deploy_robot/hardware_time
    /deploy_robot/simulation_time
    /deploy_robot/control_time
    /deploy_robot/fsm_time
    /deploy_robot/fsm_state
    /deploy_robot/fsm_request
    /deploy_robot/root_calibration_request
    /deploy_robot/student_observation
    /deploy_robot/student_action
    /deploy_robot/command
  )
  mkdir -p "$RECORDING_ROOT"
  echo "ROS bag recording: $BAG_PATH"
  start_background ros2 bag record \
    --output "$BAG_PATH" \
    --storage mcap \
    --disable-keyboard-controls \
    --topics "${RECORD_TOPICS[@]}"
fi

if [[ "$BACKEND" == "sim" ]]; then
  CONTROLLER_ARGS=(
    --backend simulation
    --config "$CONFIG"
  )
  if [[ -t 0 ]]; then
    start_background_tty "$PYTHON" control_node/control_29dof_tppo_student.py \
      "${CONTROLLER_ARGS[@]}"
  else
    start_background "$PYTHON" control_node/control_29dof_tppo_student.py \
      "${CONTROLLER_ARGS[@]}"
  fi

  start_background "$PYTHON" simulation/simulation_node/simulation_tppo_student.py \
    --config "$CONFIG" \
    "${SIM_ARGS[@]}"
else
  HARDWARE_ARGS=(
    --network "$NETWORK"
    --config "$CONFIG"
    --no-prompt
    --unitree-remote-fsm
  )
  CONTROLLER_ARGS=(
    --backend hardware
    --config "$CONFIG"
  )

  start_background "$PYTHON" estimation/mocap/natnet_policy_bridge.py \
    --config "$CONFIG"

  if [[ "$COMMAND_MODE" == "readonly" ]]; then
    HARDWARE_ARGS+=(--read-only)
  elif [[ "$COMMAND_MODE" == "shadow" ]]; then
    HARDWARE_ARGS+=(--read-only)
    CONTROLLER_ARGS+=(--shadow-command)
  else
    CONTROLLER_ARGS+=(--enable-command)
  fi

  start_background "$PYTHON" hardware/hardware_node/hardware_athlete.py \
    "${HARDWARE_ARGS[@]}"
  if [[ -t 0 ]]; then
    start_background_tty "$PYTHON" control_node/control_29dof_tppo_student.py \
      "${CONTROLLER_ARGS[@]}"
  else
    start_background "$PYTHON" control_node/control_29dof_tppo_student.py \
      "${CONTROLLER_ARGS[@]}"
  fi
fi

# Stop the complete deployment as soon as either backend or controller exits.
wait -n "${pids[@]}"
