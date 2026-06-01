#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal
source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

wandb_activate_args=()
if [[ "${WANDB_ACTIVATE:-True}" == "False" || "${WANDB_ACTIVATE:-True}" == "false" || "${WANDB_ACTIVATE:-True}" == "0" ]]; then
  wandb_activate_args+=(--no-wandb-activate)
fi

python isaacgymenvs/launch_training.py \
  --task SimToolRealDynamicGraspSim2RealPointCloudFast \
  --custom-experiment-name "${EXPERIMENT_NAME:-dynamic_grasp_sim2real_pointcloud_fast_v1}" \
  --seed "${SEED:-0}" \
  --num-envs "${NUM_ENVS:-8192}" \
  --num-blocks "${NUM_BLOCKS:-4}" \
  --wandb-entity "${WANDB_ENTITY:-simonlsx}" \
  --wandb-project "${WANDB_PROJECT:-simtoolreal}" \
  --wandb-group "${WANDB_GROUP:-dynamic_grasp_sim2real_fast}" \
  "${wandb_activate_args[@]}" \
  --capture-video True \
  --capture-video-freq 3000 \
  --capture-video-len 240
