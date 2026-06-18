# Dynamic Gym：Franka + BrainCo Revo2 动态灵巧操作基准

Dynamic Gym 是基于 SimToolReal 扩展出来的动态物体灵巧操作 benchmark。
当前分支主要用于交接和搭建仿真环境，重点是 Franka + BrainCo Revo2
右手，以及空中棒状物体相关任务。

原始 SimToolReal 仍然是本项目的底层基础，负责 Isaac Gym 环境、资产加载、
机器人控制接口和 RL 工具链。但这个 fork 的主入口已经从原始静态
DexToolBench 工具操作，转向动态物体抓取、承接和后续 sim2real 研究。

## 当前分支定位

当前分支：

```text
sim-env/franka-revo2-aerial-v1
```

这个分支的目标是提供一个相对干净的 simulation/environment handoff，
方便其他人基于它继续搭建任务环境。

包含内容：

- Franka + BrainCo Revo2 右手仿真资产。
- Revo2 右手 mounting、对齐、预览和 sanity check 脚本。
- 已经可以预览和继续开发的 Falling Baton 环境。
- Baton Insert / 被动承接任务的环境规格说明。
- marker / screwdriver 等棒状物体资产。
- clean v2 grasp affordance labels。
- benchmark 网页，用于展示背景、任务设置、pipeline 和 affordance 标注。

不建议把以下内容混进这个分支：

- `train_dir`
- W&B run 目录
- checkpoints
- 大量历史 PPO launch/eval 脚本
- 训练视频和评估视频输出

## 任务设置

### 空中任务

| 任务 | 动态来源 | Affordance 设计 | 最终目标 |
| --- | --- | --- | --- |
| Falling Baton 主动抓取 | 自由下落 + 随机角速度 | 中部 / handle 可抓，两端不可抓 | 主动预测并拦截下落物，抓住后稳定保持 |
| Baton Insert 被动接住 | 低速 handoff 或引导式下落到接收区域 | 指定 safe grasp region | 手在接收区域等待，被动承接物体，吸收冲击并稳定抓住 |

这里的 `Baton Insert` 目前不是插入 slot 的任务，而是先定义为被动接住 /
被动承接任务。后续如果要扩展成插入任务，可以在此基础上继续加入 slot actor、
轴向对齐、插入深度和插入成功判定。

### 桌面任务

| 任务 | 动态来源 | Affordance 设计 | 最终目标 |
| --- | --- | --- | --- |
| Rolling Marker | 斜坡滚动 / 桌面滚动 | 笔尖不可抓，body 可抓 | 捕获滚动物体，轴向对齐，并放入 holder |
| Conveyor Tool | 传送带 / 小车移动 | handle 可抓，functional end 不可抓 | 从 handle 抓取工具，并修正姿态用于后续操作 |

桌面任务是后续 benchmark 的计划方向。当前 handoff 主要聚焦两个空中任务。

## 代码结构

```text
assets/
  generated/franka_brainco_revo2_right/    # Franka + Revo2 right hand 合并资产
  urdf/dextoolbench/marker/                # marker / baton-like 物体
  urdf/dextoolbench/screwdriver/           # screwdriver / baton-like 物体
  affordance_labels/                       # clean v2 grasp affordance labels 和可视化

docs/
  handoff/franka_brainco_revo2_aerial_env/ # simulation-only 交接文档和任务规格
  project_page/                            # benchmark 静态网页
  brainco_revo2_embodiment.md              # Revo2 接入和对齐说明

isaacgymenvs/
  cfg/task/                                # 任务 / 环境配置
  tasks/simtoolreal/env.py                 # 主要 Isaac Gym 环境实现

scripts/
  prepare_franka_brainco_revo2_asset.sh    # 生成 Franka + Revo2 合并 URDF
  preview_franka_brainco_revo2_aerial_envs.sh
  preview_dg_franka_brainco_revo2_env.sh
  run_revo2_scripted_grasp_sanity.py
  package_franka_brainco_revo2_aerial_env.sh
```

## 环境安装

Isaac Gym Preview 4 需要 Python 3.8。建议使用 conda 环境。

### 1. 创建 conda 环境

```bash
conda create -n simtoolreal python=3.8 -y
conda activate simtoolreal

python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 PyTorch

根据机器 CUDA 版本选择对应 wheel。例如 CUDA 11.8：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

如果服务器上已经有验证过的 PyTorch + Isaac Gym 组合，优先使用已有版本。

### 3. 安装 Isaac Gym Preview 4

从 NVIDIA 下载 Isaac Gym Preview 4，然后安装 Python package：

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz -C /path/to/isaacgym_preview4
python -m pip install -e /path/to/isaacgym_preview4/isaacgym/python
```

检查是否安装成功：

```bash
python - <<'PY'
import isaacgym
print("Isaac Gym import OK")
PY
```

### 4. 安装本仓库

在仓库根目录执行：

```bash
python -m pip install -e .
python -m pip install -e rl_games
```

一些本地脚本还会用到：

```bash
python -m pip install tyro wandb imageio[ffmpeg]
```

### 5. 准备 Franka + Revo2 资产

当前分支已经包含生成好的右手合并资产：

```text
assets/generated/franka_brainco_revo2_right/franka_brainco_revo2_right.urdf
```

如果需要从官方 BrainCo Revo2 右手 URDF 重新生成：

```bash
BRAINCO_REVO2_URDF=/path/to/revo2_right_hand.urdf \
bash scripts/prepare_franka_brainco_revo2_asset.sh
```

注意：当前项目统一使用 BrainCo Revo2 右手。

## 快速预览空中任务环境

预览 Falling Baton 环境：

```bash
conda activate simtoolreal

TASK=SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch \
NUM_ENVS=8 \
STEPS=160 \
OUT_DIR=preview_videos \
bash scripts/preview_dg_franka_brainco_revo2_env.sh
```

也可以直接使用 handoff wrapper：

```bash
bash scripts/preview_franka_brainco_revo2_aerial_envs.sh
```

输出会保存在：

```text
preview_videos/
```

## Revo2 资产 sanity check

可以用 scripted sanity check 检查手的朝向、基础碰撞、关节运动和环境加载：

```bash
python scripts/run_revo2_scripted_grasp_sanity.py \
  --task SimToolRealFallingBatonV88FrankaBrainCoRevo2PrivPointCloudPhysicalCatch \
  --num-envs 4 \
  --out-dir preview_videos
```

这个检查不是训练，只是帮助确认仿真资产和环境没有明显穿模、错向或加载错误。

## 交接包

如果需要把当前仿真环境打包成 overlay 交给其他 checkout：

```bash
bash scripts/package_franka_brainco_revo2_aerial_env.sh
```

生成路径类似：

```text
handoff_packages/franka_brainco_revo2_aerial_env_<timestamp>.tar.gz
```

在另一份 SimToolReal-compatible checkout 中解压：

```bash
tar -xzf franka_brainco_revo2_aerial_env_<timestamp>.tar.gz \
  --strip-components=1 \
  -C /path/to/simtoolreal
```

详细说明见：

```text
docs/handoff/franka_brainco_revo2_aerial_env/README.md
```

## 项目网页

benchmark 网页在：

```text
docs/project_page/
```

在服务器上启动：

```bash
cd docs/project_page
python -m http.server 8124 --bind 0.0.0.0
```

如果本地浏览器无法直接访问服务器端口，可以在本地机器开 SSH tunnel：

```bash
ssh -N -L 8124:127.0.0.1:8124 linsixu@10.26.1.172
```

然后本地浏览器打开：

```text
http://127.0.0.1:8124/
```

## 训练说明

当前分支主要面向环境搭建和仿真交接。训练代码仍然保留在仓库里，但新的训练实验
最好单独放在训练分支，不要和 simulation handoff 分支混在一起。

训练入口仍然是：

```bash
python isaacgymenvs/launch_training.py --help
```

训练日志默认在：

```text
train_dir/simtoolreal/
```

不要把 `train_dir`、W&B run、checkpoint、eval videos 等生成结果提交到这个分支。

## Affordance Labels

推荐使用 clean v2 标签：

```text
assets/affordance_labels/**/grasp_affordance_clean_v2.npz
```

标签约定：

```text
grasp_label = 1   positive grasp region
grasp_label = 0   conservative negative region
grasp_label = -1  ignore / uncertain
```

训练监督或分析时只使用有效标签：

```python
valid = grasp_label >= 0
```

在 RL 环境里，这些 affordance labels 更适合作为 weak prior / debug signal，
不要直接当成最终成功标准。

## 对外协作者快速上手

如果只是想快速确认环境能不能跑，可以按下面顺序：

```bash
git clone -b sim-env/franka-revo2-aerial-v1 git@github.com:Simonlinsx/Dynamic_Gym.git
cd Dynamic_Gym

conda create -n simtoolreal python=3.8 -y
conda activate simtoolreal

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -e rl_games

# 需要先安装 Isaac Gym Preview 4
bash scripts/preview_franka_brainco_revo2_aerial_envs.sh
```

如果环境已经在服务器上配置好，只需要：

```bash
conda activate simtoolreal
bash scripts/preview_franka_brainco_revo2_aerial_envs.sh
```

## 致谢和来源

本项目基于 SimToolReal：

```text
SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
https://simtoolreal.github.io/
https://github.com/tylerlum/simtoolreal
```

原始 SimToolReal 提供了 Isaac Gym 环境框架、DexToolBench assets、部署工具和
RL 基础设施。Dynamic Gym 在此基础上扩展动态物体灵巧操作任务。
