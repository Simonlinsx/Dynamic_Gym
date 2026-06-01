#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal
source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

python isaacgymenvs/launch_training.py \
  --task SimToolRealDynamicGraspPointCloud \
  --custom-experiment-name dynamic_grasp_pointcloud_v1 \
  --num-envs "${NUM_ENVS:-8192}" \
  --num-blocks "${NUM_BLOCKS:-4}" \
  --wandb-entity simonlsx \
  --wandb-project simtoolreal \
  --wandb-group dynamic_grasp_pointcloud \
  --capture-video True \
  --capture-video-freq 3000 \
  --capture-video-len 240
