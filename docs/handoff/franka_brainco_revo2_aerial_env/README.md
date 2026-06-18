# Franka + BrainCo Revo2 Aerial Environment Handoff

This folder is the simulation-only handoff for the Franka + BrainCo Revo2
right-hand embodiment and the first two aerial manipulation tasks.

It is meant for environment setup, debugging, and task construction. It does
not include PPO training configs, checkpoints, `train_dir`, W&B logs, or any
experiment output.

## Scope

Included:

- Franka + BrainCo Revo2 right-hand combined URDF and mesh assets.
- Scripts used to rebuild, inspect, and preview the combined robot asset.
- Isaac Gym task code needed by the current SimToolReal environments.
- Task YAMLs for the BrainCo Revo2 dynamic grasp and falling baton variants.
- DextoolBench marker and screwdriver object assets.
- Clean v2 grasp affordance labels for marker and screwdriver objects.
- Environment specs for the two aerial tasks.

Not included:

- Training launchers and train YAMLs.
- RL checkpoints.
- `train_dir`, videos, W&B run state, or tensorboard logs.
- A complete Aerial Object Catch implementation. That task is currently a
  passive airborne receive/catch design spec and integration target.

## Task Status

### Falling Baton

Status: implemented as an Isaac Gym task variant.

Reference task:

```bash
SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch
```

Behavior:

- No table.
- One elongated object is spawned above and in front of the hand.
- The robot actively moves toward the predicted catch point.
- The task intuition is a stick-catching game: predict the falling baton,
  intercept near the middle/handle affordance, close the hand, and hold.
- Object falls under gravity with configurable initial linear velocity and
  random angular velocity.
- Spawn XY is constrained to the front catch workspace and can be constrained
  relative to the home catch center.
- Object-arm collision can be filtered so the falling object does not
  immediately hit the Franka forearm.
- Current object pool includes:
  - `marker/sharpie_marker`
  - `marker/staples_marker`
  - `screwdriver/long_screwdriver`
  - `screwdriver/short_screwdriver`
- Affordance labels use `grasp_affordance_clean_v2.npz`; the middle/handle is
  treated as graspable and the ends/tool regions as non-preferred.
- Illustration: `docs/project_page/assets/images/falling_baton_game_schematic.png`.

### Aerial Object Catch

Status: environment spec/template only.

Target behavior:

- No table and no slot fixture in this v1 setting.
- A rod-like object is passively received from an airborne toss/drop, a
  low-speed handoff, or guided free-fall into a receive zone.
- The hand starts open near the receive pose, absorbs object motion, closes on
  the safe middle region, and stabilizes the object.
- Unlike Falling Baton, this task should not require large active chasing or
  interception before contact.
- This task is named Aerial Object Catch to avoid the earlier misleading
  "Baton Insert" wording. The reference setting is
  [Catch It! Learning to Catch in Flight with Mobile Dexterous Hands](https://arxiv.org/pdf/2409.10319),
  adapted here to fixed-base Franka + BrainCo Revo2 instead of a mobile base.

Implementation work still needed:

- Add receive-zone reset logic and object trajectory generation.
- Add low pre-contact palm-motion and low-impact contact measurements.
- Add success/reset logic for passive receive and stable catch.
- Add preview scripts and debug overlays for receive zone, incoming velocity,
  rod axis, and stable catch state.

## Quick Preview

From the SimToolReal repo root:

```bash
conda activate simtoolreal

TASK=SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch \
NUM_ENVS=8 \
STEPS=160 \
OUT_DIR=preview_videos \
bash scripts/preview_dg_franka_brainco_revo2_env.sh
```

For a scripted sanity check:

```bash
conda activate simtoolreal

python scripts/run_revo2_scripted_grasp_sanity.py \
  --task SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch \
  --num-envs 4 \
  --out-dir preview_videos
```

## Important Embodiment Conventions

- Use the BrainCo Revo2 right hand only.
- The combined asset path is:

```text
assets/generated/franka_brainco_revo2_right/franka_brainco_revo2_right.urdf
```

- The Revo2 hand is mounted at the Panda hand flange with:

```text
mount_xyz = [0.0, 0.0, 0.0584]
mount_rpy = [0.0, 0.0, pi]
```

- The Franka hand visual/collision geometry is stripped from the combined
  asset, leaving the Revo2 mounted at the flange adapter.
- The current physical-hand action mode exposes 11 Revo2 joints. A coupled mode
  exists in the repo, but the handoff default is the physical 11-DOF interface.

## Affordance Labels

Use the clean v2 binary labels:

```text
assets/affordance_labels/**/grasp_affordance_clean_v2.npz
```

Fields:

```text
grasp_label = 1   positive grasp region
grasp_label = 0   conservative negative region
grasp_label = -1  ignore / uncertain
```

Training or supervised losses should only use valid labels:

```python
valid = grasp_label >= 0
```

For the aerial baton tasks, treat these labels as a weak environment prior and
debug signal, not as the final success definition.

## Package Usage

The generated tarball is an overlay. Extract it into a compatible SimToolReal
repo checkout:

```bash
tar -xzf franka_brainco_revo2_aerial_env_*.tar.gz \
  --strip-components=1 \
  -C /path/to/simtoolreal
cd /path/to/simtoolreal
conda activate simtoolreal
bash scripts/preview_franka_brainco_revo2_aerial_envs.sh
```

If the combined URDF needs to be regenerated from an official BrainCo Revo2
right-hand URDF:

```bash
BRAINCO_REVO2_URDF=/path/to/revo2_right_hand.urdf \
bash scripts/prepare_franka_brainco_revo2_asset.sh
```

## Files To Start From

- `env_specs/franka_brainco_revo2_common.yaml`
- `env_specs/falling_baton.yaml`
- `env_specs/aerial_object_catch.yaml`
- `scripts/preview_franka_brainco_revo2_aerial_envs.sh`
- `scripts/prepare_franka_brainco_revo2_asset.sh`
- `scripts/preview_dg_franka_brainco_revo2_env.sh`
- `isaacgymenvs/cfg/task/SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch.yaml`
- `isaacgymenvs/tasks/simtoolreal/env.py`
