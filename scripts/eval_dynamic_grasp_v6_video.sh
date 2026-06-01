#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data1/linsixu/simtoolreal}
CONDA_SH=${CONDA_SH:-/data1/linsixu/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-simtoolreal}

RUN_DIR=${RUN_DIR:-/data1/linsixu/simtoolreal/train_dir/simtoolreal/dynamic_grasp/dynamic_grasp_v6_2026-05-17_17-57-34}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/runs/00_dynamic_grasp_v6_2026-05-17_17-57-34/best/model.pth}
EVAL_NAME=${EVAL_NAME:-dynamic_grasp_v6_eval_$(date +%Y-%m-%d_%H-%M-%S)}
EVAL_DIR=${EVAL_DIR:-$REPO_DIR/eval_videos/$EVAL_NAME}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-256}
EVAL_GAMES_NUM=${EVAL_GAMES_NUM:-1024}
EVAL_VIDEO_LEN=${EVAL_VIDEO_LEN:-120}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$EVAL_DIR"

python -m isaacgymenvs.train \
  ++task.env.useSparseReward=False \
  task=SimToolRealDynamicGrasp \
  experiment=00_dynamic_grasp_v6_eval \
  test=True \
  checkpoint="$CHECKPOINT" \
  headless=True \
  wandb_activate=False \
  hydra.run.dir="$EVAL_DIR" \
  task.env.numEnvs="$EVAL_NUM_ENVS" \
  train.params.config.minibatch_size=98304 \
  train.params.config.good_reset_boundary=0 \
  task.env.goodResetBoundary=0 \
  train.params.config.use_others_experience=lf \
  train.params.config.off_policy_ratio=1.0 \
  train.params.config.expl_type=mixed_expl_learn_param \
  train.params.config.expl_reward_type=entropy \
  +train.params.config.expl_reward_coef=0.0 \
  train.params.config.expl_coef_block_size=2048 \
  train.params.config.expl_reward_coef_scale=0.005 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  train.params.config.player.deterministic=True \
  train.params.config.player.games_num="$EVAL_GAMES_NUM" \
  train.params.config.player.print_stats=False \
  task.env.evalStats=True \
  task.env.forceConsecutiveNearGoalSteps=True \
  task.env.objectScaleNoiseMultiplierRange='[0.9,1.1]' \
  task.env.capture_video=True \
  task.env.capture_video_freq=1 \
  task.env.capture_video_len="$EVAL_VIDEO_LEN" \
  task.env.enableCameraSensors=True
