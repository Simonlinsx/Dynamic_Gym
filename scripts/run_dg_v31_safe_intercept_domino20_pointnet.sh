#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal
mkdir -p train_logs
exec > >(tee /data1/linsixu/simtoolreal/train_logs/dg_v31_safe_intercept_domino20_pointnet.log) 2>&1

source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

python isaacgymenvs/launch_training.py \
  --task SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV31SafeInterceptDomino20PointNet \
  --custom-experiment-name dynamic_grasp_domino20_safe_intercept_v31 \
  --num-envs 12288 \
  --wandb-entity simonlsx \
  --wandb-project simtoolreal \
  --wandb-group dynamic_grasp_domino20_safe_intercept \
  --capture-video True \
  --capture-video-freq 3000 \
  --capture-video-len 240
