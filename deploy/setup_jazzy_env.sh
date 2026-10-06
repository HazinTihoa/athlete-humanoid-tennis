#!/usr/bin/env bash
# ROS setup scripts intentionally read optional unset tracing variables.
set -eo pipefail

DEPLOY_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP=/opt/ros/jazzy/setup.bash
VENV_PATH="$DEPLOY_ROOT_DIR/deploy/.venv"
UNITREE_SOURCE="$DEPLOY_ROOT_DIR/submodules/unitree_sdk2_wrapper"
UNITREE_BUILD="$DEPLOY_ROOT_DIR/deploy/.build/unitree_sdk2_wrapper"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS 2 Jazzy was not found at $ROS_SETUP" >&2
  exit 1
fi

source "$ROS_SETUP"

if [[ ! -x "$VENV_PATH/bin/python" ]] || ! "$VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1; then
  python3 -m venv --clear "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -r "$DEPLOY_ROOT_DIR/deploy/requirements-jazzy.txt"

if ! "$VENV_PATH/bin/python" -c "import unitree_interface" >/dev/null 2>&1; then
  if [[ ! -f "$UNITREE_SOURCE/CMakeLists.txt" ]]; then
    echo "Missing initialized submodule: $UNITREE_SOURCE" >&2
    echo "Run: git submodule update --init submodules/unitree_sdk2_wrapper" >&2
    exit 1
  fi
  cmake \
    -S "$UNITREE_SOURCE" \
    -B "$UNITREE_BUILD" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDING=ON \
    -DGENERATE_STUBS=OFF \
    -DPYTHON_EXECUTABLE="$VENV_PATH/bin/python"
  cmake --build "$UNITREE_BUILD" --parallel "$(nproc)"
  UNITREE_MODULE="$(find "$UNITREE_BUILD" -name 'unitree_interface*.so' -print -quit)"
  if [[ -z "$UNITREE_MODULE" ]]; then
    echo "Unitree Python build completed without producing unitree_interface.so" >&2
    exit 1
  fi
  SITE_PACKAGES="$(
    "$VENV_PATH/bin/python" -c \
      'import site; print(site.getsitepackages()[0])'
  )"
  install -m 755 "$UNITREE_MODULE" "$SITE_PACKAGES/"
  install -m 644 \
    "$UNITREE_SOURCE/python_binding/unitree_interface.pyi" \
    "$SITE_PACKAGES/unitree_interface.pyi"
fi

"$VENV_PATH/bin/python" - <<'PY'
import rclpy
import mujoco
import onnxruntime
import unitree_interface

print(f"rclpy: {rclpy.__file__}")
print(f"mujoco: {mujoco.__version__}")
print(f"onnxruntime: {onnxruntime.__version__}")
print(f"unitree_interface: {unitree_interface.__file__}")
PY

echo "Deployment venv ready: $VENV_PATH"
