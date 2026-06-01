#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data1/linsixu/simtoolreal}
CONDA_SH=${CONDA_SH:-/data1/linsixu/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-simtoolreal}

VARIANT=${VARIANT:-with_vel}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-512}
EVAL_GAMES_NUM=${EVAL_GAMES_NUM:-2048}
EVAL_VIDEO_LEN=${EVAL_VIDEO_LEN:-180}
EXPL_NUM_BLOCKS=${EXPL_NUM_BLOCKS:-4}
DETERMINISTIC=${DETERMINISTIC:-True}
EVAL_TIMESTAMP=${EVAL_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}

case "$VARIANT" in
  no_vel)
    TASK=SimToolRealDynamicGraspSim2RealPointCloudFast
    RUN_NAME=fast_no_object_vel_v1_2026-05-19_06-04-12
    TRAIN_GROUP=dynamic_grasp_sim2real_fast
    EXPERIMENT=00_fast_no_object_vel_eval
    ;;
  with_vel)
    TASK=SimToolRealDynamicGraspSim2RealPointCloudObjectVelFast
    RUN_NAME=fast_with_object_vel_v1_2026-05-19_06-04-15
    TRAIN_GROUP=dynamic_grasp_sim2real_fast
    EXPERIMENT=00_fast_with_object_vel_eval
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use no_vel or with_vel." >&2
    exit 2
    ;;
esac

RUN_DIR=${RUN_DIR:-$REPO_DIR/train_dir/simtoolreal/$TRAIN_GROUP/$RUN_NAME}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/runs/00_$RUN_NAME/best/model.pth}
EVAL_NAME=${EVAL_NAME:-${RUN_NAME}_eval_${EVAL_TIMESTAMP}}
EVAL_DIR=${EVAL_DIR:-$REPO_DIR/eval_videos/$EVAL_NAME}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$EVAL_DIR"

echo "Evaluating $VARIANT"
echo "  task:       $TASK"
echo "  checkpoint: $CHECKPOINT"
echo "  eval dir:   $EVAL_DIR"
echo "  cuda:       $CUDA_VISIBLE_DEVICES"

python -m isaacgymenvs.train \
  ++task.env.useSparseReward=False \
  task="$TASK" \
  experiment="$EXPERIMENT" \
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
  +train.params.config.expl_num_blocks="$EXPL_NUM_BLOCKS" \
  train.params.config.expl_reward_coef_scale=0.005 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  train.params.config.player.deterministic="$DETERMINISTIC" \
  train.params.config.player.games_num="$EVAL_GAMES_NUM" \
  train.params.config.player.print_stats=False \
  task.env.evalStats=True \
  task.env.forceConsecutiveNearGoalSteps=True \
  task.env.objectScaleNoiseMultiplierRange='[0.9,1.1]' \
  task.env.capture_video=True \
  task.env.capture_video_freq=1 \
  task.env.capture_video_len="$EVAL_VIDEO_LEN" \
  task.env.enableCameraSensors=True
