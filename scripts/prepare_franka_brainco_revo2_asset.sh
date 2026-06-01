#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal

if [ -z "${BRAINCO_REVO2_URDF:-}" ]; then
  echo "Set BRAINCO_REVO2_URDF=/path/to/revo2_right.urdf first."
  exit 1
fi

python scripts/build_franka_hand_asset.py \
  --hand-urdf "$BRAINCO_REVO2_URDF" \
  --output-dir /data1/linsixu/simtoolreal/assets/generated/franka_brainco_revo2_right \
  --output-name franka_brainco_revo2_right.urdf \
  --robot-name franka_brainco_revo2_right \
  --hand-prefix revo2_ \
  --mount-joint-name revo2_mount_joint \
  --force
