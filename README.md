# Dynamic Gym: Franka + BrainCo Revo2 Dynamic Dexterous Benchmark

Dynamic Gym is a dynamic dexterous manipulation benchmark built on top of
SimToolReal. This branch is focused on simulation environment construction for
Franka + BrainCo Revo2, especially the aerial baton-style tasks that Bill will
continue developing.

The original SimToolReal codebase is still the foundation for Isaac Gym scene
creation, assets, robot control, and RL utilities, but the entry point of this
fork is now the dynamic-object benchmark rather than the original static
DexToolBench policy release.

## Current Scope

This branch is the simulation/environment handoff branch:

```text
sim-env/franka-revo2-aerial-v1
```

It contains:

- Franka + BrainCo Revo2 right-hand simulation assets.
- Revo2 right-hand mounting, alignment, preview, and sanity-check scripts.
- A runnable Falling Baton reference environment.
- A Baton Insert / passive receiving task specification.
- Marker and screwdriver object assets.
- Clean v2 grasp affordance labels for the baton-like objects.
- A static project page summarizing the benchmark, task settings, pipeline, and
  affordance labeling.

It intentionally does not treat training logs, W&B runs, checkpoints, or
historical PPO launch scripts as the primary product of this branch.

## Benchmark Tasks

### Aerial Tasks

| Task | Motion source | Affordance | Goal |
| --- | --- | --- | --- |
| Falling Baton | Free fall with random angular velocity | Middle / handle graspable; ends negative | Actively intercept, grasp, and hold the object |
| Baton Insert | Low-speed handoff or guided free fall into a receive zone | Safe middle region | Passively receive, absorb impact, close on the object, and stabilize it |

### Tabletop Tasks

| Task | Motion source | Affordance | Goal |
| --- | --- | --- | --- |
| Rolling Marker | Ramp or tabletop rolling | Marker tip negative; body graspable | Capture, align, and place into a holder |
| Conveyor Tool | Conveyor or moving cart | Handle graspable; functional end negative | Grasp by handle and correct pose for downstream use |

The two tabletop tasks are part of the planned benchmark direction. The current
handoff package emphasizes the two aerial tasks.

## Project Layout

```text
assets/
  generated/franka_brainco_revo2_right/    # Combined Franka + Revo2 right-hand asset
  urdf/dextoolbench/marker/                # Baton-like marker objects
  urdf/dextoolbench/screwdriver/           # Baton-like screwdriver objects
  affordance_labels/                       # Clean v2 grasp affordance labels and visualizations

docs/
  handoff/franka_brainco_revo2_aerial_env/ # Simulation-only handoff docs and task specs
  project_page/                            # Static website for the benchmark summary
  brainco_revo2_embodiment.md              # Revo2 embodiment notes

isaacgymenvs/
  cfg/task/                                # Task/environment configs
  tasks/simtoolreal/env.py                 # Main Isaac Gym environment implementation

scripts/
  prepare_franka_brainco_revo2_asset.sh    # Build combined Franka + Revo2 asset
  preview_franka_brainco_revo2_aerial_envs.sh
  preview_dg_franka_brainco_revo2_env.sh
  run_revo2_scripted_grasp_sanity.py
  package_franka_brainco_revo2_aerial_env.sh
```

## Environment Setup

Isaac Gym Preview 4 requires Python 3.8. The most robust route on the lab
server is a conda environment.

### 1. Create the Conda Environment

```bash
conda create -n simtoolreal python=3.8 -y
conda activate simtoolreal

python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

Use the CUDA wheel that matches the machine. For CUDA 11.8:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If the cluster already has a known-good PyTorch/Isaac Gym combination, prefer
that version.

### 3. Install Isaac Gym Preview 4

Download Isaac Gym Preview 4 from NVIDIA and install the Python package:

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz -C /path/to/isaacgym_preview4
python -m pip install -e /path/to/isaacgym_preview4/isaacgym/python
```

Quick import check:

```bash
python - <<'PY'
import isaacgym
print("Isaac Gym import OK")
PY
```

### 4. Install This Repository

From the repository root:

```bash
python -m pip install -e .
python -m pip install -e rl_games
```

Some local utilities also expect:

```bash
python -m pip install tyro wandb imageio[ffmpeg]
```

### 5. Prepare the Franka + Revo2 Asset

The branch already includes the generated right-hand combined asset. If it needs
to be rebuilt from an official BrainCo Revo2 right-hand URDF:

```bash
BRAINCO_REVO2_URDF=/path/to/revo2_right_hand.urdf \
bash scripts/prepare_franka_brainco_revo2_asset.sh
```

The project convention is BrainCo Revo2 right hand only.

## Quick Start: Preview the Aerial Environment

Run a headless preview of the Falling Baton environment:

```bash
conda activate simtoolreal

TASK=SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch \
NUM_ENVS=8 \
STEPS=160 \
OUT_DIR=preview_videos \
bash scripts/preview_dg_franka_brainco_revo2_env.sh
```

Or use the handoff wrapper:

```bash
bash scripts/preview_franka_brainco_revo2_aerial_envs.sh
```

The preview videos/images are written to:

```text
preview_videos/
```

## Sanity Check the Robot Asset

```bash
python scripts/run_revo2_scripted_grasp_sanity.py \
  --task SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch \
  --num-envs 4 \
  --out-dir preview_videos
```

This is useful for checking gross hand alignment, collision behavior, and basic
Revo2 joint motion before training or environment development.

## Handoff Package

To create a simulation-only overlay package for another checkout:

```bash
bash scripts/package_franka_brainco_revo2_aerial_env.sh
```

The package is written to:

```text
handoff_packages/franka_brainco_revo2_aerial_env_<timestamp>.tar.gz
```

Extract it into another SimToolReal-compatible checkout with:

```bash
tar -xzf franka_brainco_revo2_aerial_env_<timestamp>.tar.gz \
  --strip-components=1 \
  -C /path/to/simtoolreal
```

See:

```text
docs/handoff/franka_brainco_revo2_aerial_env/README.md
```

## Static Project Page

The benchmark summary page lives at:

```text
docs/project_page/
```

Serve it on the lab server:

```bash
cd docs/project_page
python -m http.server 8124 --bind 0.0.0.0
```

If direct browser access to the server is blocked, use SSH port forwarding from
your local machine:

```bash
ssh -N -L 8124:127.0.0.1:8124 linsixu@10.26.1.172
```

Then open:

```text
http://127.0.0.1:8124/
```

## Training Notes

This branch is organized around environment construction. Training support still
exists in the repository, but new training experiments should be treated as a
separate research track and should not be mixed into this environment handoff
branch without a clear reason.

For training, the main entry point remains:

```bash
python isaacgymenvs/launch_training.py --help
```

Typical logs are written under:

```text
train_dir/simtoolreal/
```

Do not commit `train_dir`, W&B run directories, checkpoints, or generated
evaluation videos to this branch.

## Affordance Labels

The recommended labels are:

```text
assets/affordance_labels/**/grasp_affordance_clean_v2.npz
```

Label convention:

```text
grasp_label = 1   positive grasp region
grasp_label = 0   conservative negative region
grasp_label = -1  ignore / uncertain
```

For supervised losses or analysis:

```python
valid = grasp_label >= 0
```

For RL and environment shaping, use the labels as weak priors/debug signals, not
as the final task success criterion.

## Attribution

This fork builds on:

```text
SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
https://simtoolreal.github.io/
https://github.com/tylerlum/simtoolreal
```

The original repository provides the Isaac Gym framework, DexToolBench assets,
deployment utilities, and RL infrastructure that Dynamic Gym extends for
dynamic-object dexterous manipulation.
