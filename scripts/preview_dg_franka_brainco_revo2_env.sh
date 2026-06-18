#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal

if [ ! -f assets/generated/franka_brainco_revo2_right/franka_brainco_revo2_right.urdf ]; then
  bash scripts/prepare_franka_brainco_revo2_asset.sh
fi

source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal

python scripts/preview_v33_franka_inspire_env.py \
  --task "${TASK:-SimToolRealDynamicGraspV60FrankaBrainCoRevo2RgbdTemporalBootstrapStudent}" \
  --num-envs "${NUM_ENVS:-20}" \
  --steps "${STEPS:-96}" \
  --fps "${FPS:-30}" \
  --out-dir "${OUT_DIR:-preview_videos}" \
  --sim-device "${SIM_DEVICE:-cuda:0}" \
  --rl-device "${RL_DEVICE:-cuda:0}" \
  --graphics-device-id "${GRAPHICS_DEVICE_ID:-0}"
