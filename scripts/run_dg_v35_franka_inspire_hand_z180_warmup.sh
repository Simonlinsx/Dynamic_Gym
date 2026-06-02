#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal
mkdir -p train_logs
exec > >(tee /data1/linsixu/simtoolreal/train_logs/dg_v35_franka_inspire_hand_z180_warmup.log) 2>&1

source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python isaacgymenvs/launch_training.py \
  --task SimToolRealDynamicGraspV35FrankaInspireHandZ180WarmupDominoPointNet \
  --custom-experiment-name dynamic_grasp_franka_inspire_hand_z180_warmup_v35 \
  --num-envs 6144 \
  --wandb-entity simonlsx \
  --wandb-project simtoolreal \
  --wandb-group dynamic_grasp_franka_inspire \
  --wandb-tags v35 franka inspire hand_z180 legacy_action warmup domino8 gpu5 \
  --capture-video True \
  --capture-video-freq 3000 \
  --capture-video-len 240
