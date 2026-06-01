#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal
source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal

export CUDA_VISIBLE_DEVICES=6

exec python isaacgymenvs/launch_training.py \
  --task SimToolRealDynamicGrasp \
  --custom-experiment-name dynamic_grasp_v6 \
  --num-envs 12288 \
  --wandb-entity simonlsx \
  --wandb-project simtoolreal \
  --wandb-group dynamic_grasp \
  --capture-video True \
  --capture-video-freq 3000 \
  --capture-video-len 240
