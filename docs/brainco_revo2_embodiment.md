# BrainCo Revo2 Embodiment Hook

This repo now keeps three hand-family paths separate:

- Sharpa legacy configs keep using the original KUKA + Sharpa asset.
- Franka + Inspire configs keep using `assets/embodiments/franka-inspire-z180`.
- Franka + BrainCo Revo2 configs use generated assets under `assets/generated/`.

The generated BrainCo asset is intentionally not checked in. The official Revo2
description package is public at:

```text
https://github.com/BrainCoTech/revo2_description
```

On this machine it is downloaded under:

```text
/data1/linsixu/simtoolreal/assets/generated/revo2_description
```

To download it on a fresh checkout:

```bash
bash scripts/download_revo2_description.sh
```

## Generate The Combined Asset

```bash
BRAINCO_REVO2_URDF=/path/to/revo2_right_hand.urdf \
  bash scripts/prepare_franka_brainco_revo2_asset.sh
```

If `assets/generated/revo2_description/urdf/revo2_right_hand.urdf` exists, the
prepare script uses it by default.

Optional overrides:

```bash
BRAINCO_REVO2_HAND_PREFIX=revo2_ \
BRAINCO_REVO2_MOUNT_XYZ="0 0 0.0584" \
BRAINCO_REVO2_MOUNT_RPY="0 0 3.14159265359" \
BRAINCO_REVO2_FRANKA_HAND_ADAPTER_STYLE=inspire_cylinders \
BRAINCO_REVO2_OUTPUT_DIR=/data1/linsixu/simtoolreal/assets/generated/franka_brainco_revo2_right \
BRAINCO_REVO2_OUTPUT_NAME=franka_brainco_revo2_right.urdf \
  bash scripts/prepare_franka_brainco_revo2_asset.sh
```

The default mount follows the Franka+Inspire-z180 convention: use the Panda
finger/flange offset and replace the bulky Panda hand with the same simple
cylindrical adapter used by the Inspire-z180 asset. Revo2 uses local `+z` as
its finger direction and local `+x` as the palmar/touch-pad side, so the default
Revo2 mount yaw is `pi` to make the back of the hand face the same side as the
Inspire-z180 asset.

The official right-hand URDF exposes 11 non-fixed joints in simulation, including
five mimic distal joints. The real control interface is usually six semantic
commands, so sim2real deployment will still need an action adapter/coupling map.
The current configs use the prefixed URDF names:

```text
palmBodyName: revo2_right_base_link
fingertips:
  revo2_right_thumb_tip_link
  revo2_right_index_tip_link
  revo2_right_middle_tip_link
  revo2_right_ring_tip_link
  revo2_right_pinky_tip_link
```

## Current Configs

Teacher-style privileged point cloud:

```bash
bash scripts/run_dg_v59_franka_brainco_revo2_priv_pointcloud_bootstrap_no_fake_affordance.sh
```

Deployable RGB-D temporal point cloud student baseline:

```bash
bash scripts/run_dg_v60_franka_brainco_revo2_rgbd_temporal_bootstrap_student.sh
```

Environment preview after generating the asset:

```bash
bash scripts/preview_dg_franka_brainco_revo2_env.sh
```

Important detail: the BrainCo placeholder uses `armDofs=7` and `handDofs=11`,
so `object_pointcloud_rel_palm` starts at observation index `82`, not the Inspire
index `85`. The V59/V60 train configs override PointNet `start_idx` accordingly.

## Distillation Note

Do not use the existing Inspire V53 teacher checkpoint directly with BrainCo
unless the action dimensions and action semantics are made compatible. For a real
BrainCo teacher-student run, train V59 first, then build a BrainCo-only
distillation config that points to that BrainCo teacher checkpoint.
