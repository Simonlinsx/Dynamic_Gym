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

[preview] Aerial Object Catch is currently a passive airborne receive/catch
task specification, not a runnable Isaac Gym task. It replaces the earlier
misleading Baton Insert name and is not a slot-insertion target in this handoff
version.
See:
  docs/handoff/franka_brainco_revo2_aerial_env/env_specs/aerial_object_catch.yaml

MSG
