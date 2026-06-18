#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal

echo "[preview] Franka + BrainCo Revo2 Falling Baton"
TASK="${TASK:-SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch}" \
NUM_ENVS="${NUM_ENVS:-8}" \
STEPS="${STEPS:-160}" \
FPS="${FPS:-30}" \
OUT_DIR="${OUT_DIR:-preview_videos}" \
bash scripts/preview_dg_franka_brainco_revo2_env.sh

cat <<'MSG'

[preview] Baton Insert is currently a passive receive/catch task specification,
not a runnable Isaac Gym task. It is no longer a slot-insertion target in this
handoff version.
See:
  docs/handoff/franka_brainco_revo2_aerial_env/env_specs/baton_insert.yaml

MSG
