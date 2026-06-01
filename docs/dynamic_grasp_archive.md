# Dynamic Grasp Archive Notes

Archive date: 2026-06-01

Base branch and commit when this archive note was written:

```text
branch: main
commit: 6809a978753e950913a7588bbeaef07d16f10b56
```

The worktree intentionally contains uncommitted dynamic-grasp development files.
This note records what should be kept for a GitHub snapshot and what should stay
local.

## Core Files To Keep

Tracked files with dynamic-grasp changes:

```text
isaacgymenvs/launch_training.py
isaacgymenvs/tasks/simtoolreal/env.py
isaacgymenvs/utils/wandb_utils.py
rl_games/rl_games/algos_torch/network_builder.py
rl_games/rl_games/algos_torch/players.py
rl_games/rl_games/algos_torch/torch_ext.py
rl_games/rl_games/common/player.py
```

Important untracked source/config files to add:

```text
assets/urdf/table_dynamic_grasp.urdf
isaacgymenvs/cfg/task/SimToolRealDynamicGrasp*.yaml
isaacgymenvs/cfg/train/SimToolRealDynamicGrasp*.yaml
scripts/*.sh
scripts/capture_actual_sim2real_pointcloud.py
scripts/visualize_sim2real_pointcloud.py
README_DYNAMIC_GRASP.md
docs/dynamic_grasp_archive.md
```

Do not add local experiment outputs:

```text
train_dir/
eval_videos/
eval_logs/
train_logs/
outputs/
nohup*.log
nohup.out
*.pth
*.mp4
*.pkl
*.tfevents*
```

## Selected Version History

### v6: Early Dynamic Object-State Baseline

Useful as a sanity check for dynamic object motion and the original state-based
observation path. Not sim2real-oriented because it uses clean simulator object
state.

Launch scripts:

```text
scripts/launch_dynamic_grasp_v6.sh
scripts/launch_dynamic_grasp_v6_logged.sh
```

### v2/v26: Sim2Real Point Cloud + PointNet Line

Useful as the cleanest pre-DOMINO sim2real observation baseline.

Key files:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloud.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVel.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV26PointNet.yaml
isaacgymenvs/cfg/train/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV26PointNetPPO.yaml
```

### v28: DOMINO20 High-Success Baseline

Highest deterministic DOMINO20 success observed among archived per-object evals.

```text
100 deterministic episodes: 38 successes, success_rate = 0.38
```

Keep as a numeric baseline, but document the scooping behavior.

Key files:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV28Domino20PointNet.yaml
isaacgymenvs/cfg/train/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV28Domino20PointNetPPO.yaml
scripts/run_dg_v28_domino20_pointnet.sh
scripts/eval_dg_v28_domino20_det100_per_object.sh
```

### v29/v30: True-Grasp / Anti-Scoop Line

Important because it made success depend on true fingertip grasp and penalized
palm-only lifting.

Key files:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV29Domino20PointNet.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV30Domino20PointNet.yaml
scripts/run_dg_v29_domino20_true_grasp_pointnet.sh
scripts/run_dg_v30_domino20_true_grasp_pointnet.sh
```

### v31/v31.1: Safe Intercept Line

Most aligned with the intended real-robot behavior: smooth interception,
pregrasp hold, controlled contact, and lower-impact movement.

v31 was too conservative. v31.1 added reward bridge terms so the policy could
transition from pregrasp readiness into contact, enclosure, and lift.

Known v31.1 eval:

```text
100 deterministic episodes: 22 successes, success_rate = 0.22
```

Key files:

```text
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV31SafeInterceptDomino20PointNet.yaml
isaacgymenvs/cfg/task/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV311SafeInterceptBridgeDomino20PointNet.yaml
isaacgymenvs/cfg/train/SimToolRealDynamicGraspSim2RealPointCloudObjectVelFastRewardV311SafeInterceptBridgeDomino20PointNetPPO.yaml
scripts/run_dg_v31_safe_intercept_domino20_pointnet.sh
scripts/run_dg_v311_safe_intercept_bridge_domino20_pointnet.sh
scripts/eval_dg_v311_domino20_det100_per_object.sh
scripts/eval_dg_v311_domino20_det_video.sh
```

## Suggested GitHub Staging

Use a targeted `git add`; avoid `git add .` until after checking ignored files.

```bash
git add .gitignore README.md README_DYNAMIC_GRASP.md docs/dynamic_grasp_archive.md
git add assets/urdf/table_dynamic_grasp.urdf
git add isaacgymenvs/launch_training.py isaacgymenvs/tasks/simtoolreal/env.py isaacgymenvs/utils/wandb_utils.py
git add rl_games/rl_games/algos_torch/network_builder.py rl_games/rl_games/algos_torch/players.py
git add rl_games/rl_games/algos_torch/torch_ext.py rl_games/rl_games/common/player.py
git add isaacgymenvs/cfg/task/SimToolRealDynamicGrasp*.yaml
git add isaacgymenvs/cfg/train/SimToolRealDynamicGrasp*.yaml
git add scripts/
```

Then check:

```bash
git status --short
git diff --cached --stat
```

The GitHub snapshot should not include training outputs, checkpoint files,
TensorBoard events, W&B run directories, videos, or local PDFs.

