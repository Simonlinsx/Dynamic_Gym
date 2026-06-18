#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal

DEFAULT_REVO2_URDF="/data1/linsixu/simtoolreal/assets/generated/revo2_description/urdf/revo2_right_hand.urdf"
if [ -z "${BRAINCO_REVO2_URDF:-}" ]; then
  if [ -f "$DEFAULT_REVO2_URDF" ]; then
    BRAINCO_REVO2_URDF="$DEFAULT_REVO2_URDF"
  else
    echo "Set BRAINCO_REVO2_URDF=/path/to/revo2_right_hand.urdf first."
    echo "Official source: https://github.com/BrainCoTech/revo2_description"
    exit 1
  fi
fi
if grep -Eq 'name="left_|left_hand|revo2_left' "$BRAINCO_REVO2_URDF"; then
  echo "Refusing to build Franka+Revo2 asset from a left-hand URDF: $BRAINCO_REVO2_URDF"
  echo "This project branch is standardized on the BrainCo Revo2 right hand."
  exit 1
fi
if ! grep -Eq 'name="right_|right_hand|revo2_right' "$BRAINCO_REVO2_URDF"; then
  echo "Could not verify right-hand link names in: $BRAINCO_REVO2_URDF"
  echo "Use the official revo2_right_hand.urdf or set BRAINCO_REVO2_URDF to a right-hand URDF."
  exit 1
fi

OUTPUT_DIR="${BRAINCO_REVO2_OUTPUT_DIR:-/data1/linsixu/simtoolreal/assets/generated/franka_brainco_revo2_right}"
OUTPUT_NAME="${BRAINCO_REVO2_OUTPUT_NAME:-franka_brainco_revo2_right.urdf}"
HAND_PREFIX="${BRAINCO_REVO2_HAND_PREFIX:-revo2_}"
HAND_PACKAGE_ROOT="${BRAINCO_REVO2_PACKAGE_ROOT:-}"
# Match the Franka+Inspire right-hand convention at the panda_hand flange.
# The raw Revo2 right-hand URDF points its four fingers along flange +Z, but its
# thumb side / index-to-pinky direction is opposite to the tuned Inspire-right
# asset unless the hand root is yawed by pi around local Z.
MOUNT_XYZ="${BRAINCO_REVO2_MOUNT_XYZ:-0 0 0.0584}"
MOUNT_RPY="${BRAINCO_REVO2_MOUNT_RPY:-0 0 3.14159265359}"
ROBOT_NAME="${BRAINCO_REVO2_ROBOT_NAME:-franka_brainco_revo2_right}"
MOUNT_JOINT_NAME="${BRAINCO_REVO2_MOUNT_JOINT_NAME:-revo2_mount_joint}"
HAND_VISUAL_COLOR="${BRAINCO_REVO2_HAND_VISUAL_COLOR:-0.04 0.10 0.22 1.0}"
HAND_TOUCH_VISUAL_COLOR="${BRAINCO_REVO2_HAND_TOUCH_VISUAL_COLOR:-0.18 0.18 0.20 1.0}"
FRANKA_HAND_ADAPTER_STYLE="${BRAINCO_REVO2_FRANKA_HAND_ADAPTER_STYLE:-inspire_cylinders}"
STRIP_FRANKA_HAND_GEOMETRY="${BRAINCO_REVO2_STRIP_FRANKA_HAND_GEOMETRY:-1}"
STRIP_FRANKA_WRIST_CAMERA_GEOMETRY="${BRAINCO_REVO2_STRIP_FRANKA_WRIST_CAMERA_GEOMETRY:-1}"

HAND_PACKAGE_ROOT_ARGS=()
if [ -n "$HAND_PACKAGE_ROOT" ]; then
  HAND_PACKAGE_ROOT_ARGS=(--hand-package-root "$HAND_PACKAGE_ROOT")
fi

STRIP_GEOMETRY_ARGS=()
if [ "$STRIP_FRANKA_HAND_GEOMETRY" != "0" ]; then
  STRIP_GEOMETRY_ARGS+=(--strip-franka-hand-geometry)
fi
if [ "$STRIP_FRANKA_WRIST_CAMERA_GEOMETRY" != "0" ]; then
  STRIP_GEOMETRY_ARGS+=(--strip-franka-wrist-camera-geometry)
fi

python scripts/build_franka_hand_asset.py \
  --hand-urdf "$BRAINCO_REVO2_URDF" \
  "${HAND_PACKAGE_ROOT_ARGS[@]}" \
  "${STRIP_GEOMETRY_ARGS[@]}" \
  --franka-hand-adapter-style "$FRANKA_HAND_ADAPTER_STYLE" \
  --output-dir "$OUTPUT_DIR" \
  --output-name "$OUTPUT_NAME" \
  --robot-name "$ROBOT_NAME" \
  --hand-prefix "$HAND_PREFIX" \
  --mount-joint-name "$MOUNT_JOINT_NAME" \
  --mount-xyz $MOUNT_XYZ \
  --mount-rpy $MOUNT_RPY \
  --hand-visual-color $HAND_VISUAL_COLOR \
  --hand-touch-visual-color $HAND_TOUCH_VISUAL_COLOR \
  --force

python scripts/inspect_robot_asset_urdf.py "$OUTPUT_DIR/$OUTPUT_NAME" \
  --hand-prefix "$HAND_PREFIX"
