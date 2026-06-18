# Handoff Manifest

This manifest describes the intended contents of the packaged environment
overlay.

## Robot Assets

- `assets/generated/franka_brainco_revo2_right/`
  - Combined Franka + BrainCo Revo2 right-hand URDF.
  - Franka reference files.
  - Official Revo2 right-hand URDF and meshes copied into the generated asset.
  - Left-hand Revo2 files are pruned from the generated handoff package to
    avoid ambiguity. The project convention here is right-hand only.

## Object Assets

- `assets/urdf/dextoolbench/marker/`
- `assets/urdf/dextoolbench/screwdriver/`

The two aerial handoff tasks do not require a table. Falling Baton is the
active catch setting; Baton Insert is currently treated as a passive
receive/catch setting, not a slot-insertion setting.

## Affordance Labels

- `assets/affordance_labels/dextoolbench/marker/`
- `assets/affordance_labels/dextoolbench/screwdriver/`
- `assets/affordance_labels/analysis/*clean_v2*`
- `assets/affordance_labels/visualizations/*clean_v2*`

Preferred label file:

```text
grasp_affordance_clean_v2.npz
```

## Environment Code

- `isaacgymenvs/tasks/simtoolreal/env.py`
- `isaacgymenvs/cfg/task/`

Only task/environment configs are packaged. Training configs are intentionally
left out.

## Scripts

- `scripts/build_franka_hand_asset.py`
- `scripts/prepare_franka_brainco_revo2_asset.sh`
- `scripts/inspect_robot_asset_urdf.py`
- `scripts/preview_v33_franka_inspire_env.py`
- `scripts/preview_dg_franka_brainco_revo2_env.sh`
- `scripts/preview_franka_brainco_revo2_aerial_envs.sh`
- `scripts/run_revo2_scripted_grasp_sanity.py`

## Handoff Docs

- `docs/brainco_revo2_embodiment.md`
- `docs/handoff/franka_brainco_revo2_aerial_env/README.md`
- `docs/handoff/franka_brainco_revo2_aerial_env/env_specs/*.yaml`
