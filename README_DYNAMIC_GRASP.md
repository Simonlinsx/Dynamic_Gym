# Dynamic Tabletop Dexterous Grasping Extension

This fork adds a dynamic tabletop grasping research track on top of the original
SimToolReal codebase. The target task is to intercept a moving object on a table,
close the dexterous hand around it, lift it, and keep a stable hold.

The current implementation is useful for simulation research and ablations. It is
not a ready-to-deploy real-robot stack yet. In particular, the current embodied
hand is still the original SimToolReal/Shadow-style hand, while the intended real
hardware direction is Linker Hand or BrainCo Revo2 with a Franka arm.

## What Was Added

- Dynamic tabletop object motion and bounce logic in `SimToolReal`.
- A larger table asset for moving-object grasping.
- RGB-D/object-mask style object point cloud observations.
- Sim2real-oriented point cloud noise, dropout, partial-view, centroid velocity,
  and tracking-confidence features.
- A privileged `object_vel_rel_palm` ablation path.
- A lightweight PointNet encoder in `rl_games` for flattened object point clouds.
- DOMINO/RoboTwin 20-object asset pool support.
- W&B/video-friendly launch wrappers.
- Deterministic evaluation scripts for per-object success.
- Several reward versions exploring fast tracking, lift/hold stability,
  anti-scoop terms, true-grasp requirements, and safer pregrasp interception.

## Recommended Implementations To Keep

### 1. Current Safer Baseline: v31.1 Safe Intercept Bridge

Use this as the current main research baseline when studying safer, smoother,
pregrasp-first interception.

Task config:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV311SafeInterceptBridgeDomino20PointNet.yaml
```

Train config:

```text
isaacgymenvs/cfg/train/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV311SafeInterceptBridgeDomino20PointNetPPO.yaml
```

Launch script:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_dg_v311_safe_intercept_bridge_domino20_pointnet.sh
```

What it does:

- Keeps v31's smoother, lower-impact arm motion.
- Rewards holding a pregrasp pose ahead of the object.
- Adds a bridge from pregrasp readiness to controlled contact, enclosure, and
  true grasp.
- Keeps anti-scoop / palm-only lift penalties.

Known result:

```text
deterministic DOMINO20 eval: 22 / 100 = 0.22
```

This is lower than v28's numeric success, but the behavior is more aligned with
the desired real-robot style: less rushing and less scooping.

Local eval summary:

```text
eval_videos/dynamic_grasp_domino20_safe_intercept_bridge_v311_2026-05-29_11-49-49_det100_per_object_2026-05-31_04-29-42/eval_summaries/per_object_eval_stats.json
```

### 2. Highest Numeric DOMINO20 Baseline: v28

Use this as a high-success reference and regression baseline, but do not treat it
as the final behavior target.

Task config:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV28Domino20PointNet.yaml
```

Launch script:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_dg_v28_domino20_pointnet.sh
```

Known result:

```text
deterministic DOMINO20 eval: 38 / 100 = 0.38
```

Important caveat:

v28 often succeeds by scooping or lifting with the palm, especially on diverse
objects. It is valuable because it shows that the environment/reward can produce
successful lifts, but it is not the safest sim2real candidate.

### 3. Anti-Scoop / True-Grasp Line: v29 and v30

Use this line to study true fingertip grasp constraints.

Configs:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV29Domino20PointNet.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV30Domino20PointNet.yaml
```

Launch scripts:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_dg_v29_domino20_true_grasp_pointnet.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/run_dg_v30_domino20_true_grasp_pointnet.sh
```

What changed:

- Success requires true fingertip grasp.
- Lift rewards are gated by grasp quality.
- Opposing contact and post-lift hold rewards are stronger.
- Palm-only and scoop lifts are penalized.

v30 is a useful middle point: it is less permissive than v28 and less conservative
than v31/v31.1.

### 4. Point Cloud Sim2Real + Object Velocity Ablation: v2/v26

Use this line to compare real-observation-compatible inputs against the privileged
`object_vel_rel_palm` signal.

Representative configs:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloud.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVel.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV26PointNet.yaml
```

Key idea:

- Actor gets robot state, palm state, fingertips relative to palm, object
  pointcloud relative to palm, pointcloud centroid, pointcloud velocity proxy,
  and tracking confidence.
- The object velocity variant appends simulator `object_vel_rel_palm` for
  ablation.
- PointNet compresses the 128 x 3 point cloud before the policy MLP.

v26 is the last useful pre-DOMINO reward baseline with extra post-lift hold
shaping and a leaky stable counter.

### 5. Affordance-Conditioned Line: v32

Use this as the first checkpoint after the GitHub archive. It keeps v31.1 as the
parent config and adds a future affordance target:

```text
future_affordance = current_object_side_affordance + object_velocity * lead_time
```

Task config:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV32AffordanceDomino20PointNet.yaml
```

Launch script:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_dg_v32_affordance_domino20_pointnet.sh
```

What it changes:

- The intercept/pregrasp reward target can be `future_affordance_pos` instead of
  the future object center.
- Actor observation appends:

```text
affordance_pos_rel_palm
future_affordance_pos_rel_palm
affordance_axis_rel_palm
affordance_confidence
```

- The current affordance source is a heuristic future-side prior. It is designed
  as a drop-in placeholder for a later DOMINO/AnyDex/RGB-D affordance predictor.
- Embodiment metadata is explicit but still uses the legacy Shadow/Sharpa action
  interface. Non-legacy Linker/Revo2 actions are intentionally not enabled until
  a real adapter is implemented.

## Main Observation Variants

The sim2real-oriented point cloud actor observation is:

```text
joint_pos
joint_vel
prev_action_targets
palm_pos
palm_rot
palm_vel
fingertip_pos_rel_palm
object_pointcloud_rel_palm
object_pointcloud_centroid_rel_palm
object_pointcloud_vel_rel_palm
object_pointcloud_tracking_confidence
```

The object-velocity ablation appends:

```text
object_vel_rel_palm
```

`object_vel_rel_palm` is not directly available on the real robot unless a visual
tracker estimates it. Keep it for ablation, but do not rely on it as the final
deployment observation.

## Main Action Space

The policy still controls the original SimToolReal robot action vector: arm and
hand target deltas/targets as used by the base environment. This is one of the
main remaining sim2real gaps. For Linker Hand or BrainCo Revo2 + Franka, the
next version should introduce an embodiment adapter:

```text
policy action -> wrist / palm command + low-dimensional grasp synergy
              -> Shadow-style hand adapter
              -> Linker Hand adapter
              -> BrainCo Revo2 adapter
```

## Training

Activate the environment:

```bash
source /data1/linsixu/miniconda3/etc/profile.d/conda.sh
conda activate simtoolreal
```

Recommended current baseline:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_dg_v311_safe_intercept_bridge_domino20_pointnet.sh
```

Equivalent direct command:

```bash
CUDA_VISIBLE_DEVICES=6 python isaacgymenvs/launch_training.py \
  --task SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV311SafeInterceptBridgeDomino20PointNet \
  --custom-experiment-name dynamic_grasp_domino20_safe_intercept_bridge_v311 \
  --num-envs 12288 \
  --wandb-entity simonlsx \
  --wandb-project simtoolreal \
  --wandb-group dynamic_grasp_domino20_safe_intercept_bridge \
  --capture-video True \
  --capture-video-freq 3000 \
  --capture-video-len 240
```

`num_envs` must be divisible by `num_blocks` in `launch_training.py`.

## Evaluation

Deterministic per-object eval for v31.1:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/eval_dg_v311_domino20_det100_per_object.sh
```

This runs the protocol:

```text
20 objects x 5 envs = 100 deterministic episodes
```

Deterministic video rollout for v31.1:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/eval_dg_v311_domino20_det_video.sh
```

v28 per-object eval:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/eval_dg_v28_domino20_det100_per_object.sh
```

v32 per-object eval:

```bash
RUN_NAME=<dynamic_grasp_domino20_affordance_v32_RUN_TIMESTAMP> \
CUDA_VISIBLE_DEVICES=7 bash scripts/eval_dg_v32_domino20_det100_per_object.sh
```

v32 deterministic video rollout:

```bash
RUN_NAME=<dynamic_grasp_domino20_affordance_v32_RUN_TIMESTAMP> \
CUDA_VISIBLE_DEVICES=7 bash scripts/eval_dg_v32_domino20_det_video.sh
```

## Local Experimental Artifacts

These paths are intentionally ignored by git. Keep them locally or upload them to
external storage if needed.

v31.1 training checkpoint:

```text
train_dir/simtoolreal/dynamic_grasp_domino20_safe_intercept_bridge/dynamic_grasp_domino20_safe_intercept_bridge_v311_2026-05-29_11-49-49/runs/00_dynamic_grasp_domino20_safe_intercept_bridge_v311_2026-05-29_11-49-49/nn/00_dynamic_grasp_domino20_safe_intercept_bridge_v311_2026-05-29_11-49-49.pth
```

v31.1 deterministic eval summary:

```text
eval_videos/dynamic_grasp_domino20_safe_intercept_bridge_v311_2026-05-29_11-49-49_det100_per_object_2026-05-31_04-29-42/eval_summaries/per_object_eval_stats.json
```

v28 deterministic eval summary:

```text
eval_videos/dynamic_grasp_domino20_v28_det100_per_object_2026-05-27_04-10-00/eval_summaries/per_object_eval_stats.json
```

## Known Limitations

- The current hand embodiment is not Linker Hand or BrainCo Revo2.
- The current arm embodiment is not yet constrained as a final Franka deployment
  controller.
- DOMINO20 success is still object-dependent. Flat, long, small, and asymmetric
  objects remain weak.
- v28 has higher numeric success but relies too much on scoop-like behavior.
- v31.1 is safer and smoother but under-rewards decisive final closure/lift for
  difficult objects.
- `object_vel_rel_palm` is privileged unless replaced by a real visual tracker.
- The point cloud is simulated from object mask/depth assumptions; the real mask
  adapter and temporal filter still need to be integrated.

## Recommended Next Step

v32 adds the first affordance-conditioned hook. The next useful version should
replace the heuristic future-side target with a learned or perception-derived
affordance estimate:

```text
RGB-D/object mask history
  -> object pointcloud / future flow / affordance prediction
  -> future affordance position + approach axis + confidence
  -> policy observation and reward target
```

DOMINO already has useful components for this direction under:

```text
/data1/linsixu/DOMINO/dynamic_affordance_data
/data1/linsixu/DOMINO/object_future_flow
```

For deployment-oriented work, add an embodiment adapter before committing to a
specific hand:

```text
Franka + Linker Hand
Franka + BrainCo Revo2
```
