#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data1/linsixu/simtoolreal}
CONDA_SH=${CONDA_SH:-/data1/linsixu/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-simtoolreal}

RUN_NAME=${RUN_NAME:-dynamic_grasp_domino20_safe_intercept_bridge_v311_2026-05-29_11-49-49}
TRAIN_GROUP=${TRAIN_GROUP:-dynamic_grasp_domino20_safe_intercept_bridge}
TASK=${TASK:-SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV311SafeInterceptBridgeDomino20PointNet}
RUN_DIR=${RUN_DIR:-$REPO_DIR/train_dir/simtoolreal/$TRAIN_GROUP/$RUN_NAME}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/runs/00_$RUN_NAME/nn/00_$RUN_NAME.pth}

EVAL_TIMESTAMP=${EVAL_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}
EVAL_NAME=${EVAL_NAME:-${RUN_NAME}_det_video_${EVAL_TIMESTAMP}}
EVAL_DIR=${EVAL_DIR:-$REPO_DIR/eval_videos/$EVAL_NAME}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-100}
EVAL_GAMES_NUM=${EVAL_GAMES_NUM:-512}
EVAL_VIDEO_LEN=${EVAL_VIDEO_LEN:-240}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$EVAL_DIR"
SNAPSHOT="$EVAL_DIR/checkpoint_eval_snapshot.pth"
cp -f "$CHECKPOINT" "$SNAPSHOT"

echo "Deterministic DOMINO20 video rollout"
echo "  task:       $TASK"
echo "  checkpoint: $CHECKPOINT"
echo "  snapshot:   $SNAPSHOT"
echo "  eval dir:   $EVAL_DIR"
echo "  num envs:   $EVAL_NUM_ENVS"
echo "  cuda:       $CUDA_VISIBLE_DEVICES"

python -m isaacgymenvs.train \
  ++task.env.useSparseReward=False \
  task="$TASK" \
  experiment=00_dg_v311_domino20_det_video_eval \
  test=True \
  checkpoint="$SNAPSHOT" \
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
  +train.params.config.expl_num_blocks=6 \
  train.params.config.expl_reward_coef_scale=0.005 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  train.params.config.player.deterministic=True \
  train.params.config.player.games_num="$EVAL_GAMES_NUM" \
  train.params.config.player.print_stats=False \
  task.env.evalStats=True \
  task.env.forceConsecutiveNearGoalSteps=True \
  task.env.objectScaleNoiseMultiplierRange='[1.0,1.0]' \
  task.env.capture_video=True \
  task.env.capture_video_freq=1 \
  task.env.capture_video_len="$EVAL_VIDEO_LEN" \
  task.env.enableCameraSensors=True
