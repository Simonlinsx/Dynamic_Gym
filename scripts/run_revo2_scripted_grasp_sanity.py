#!/usr/bin/env python
"""Run a scripted Franka + BrainCo Revo2 grasp/lift sanity check.

This does not train a policy. It loads the same SimToolReal task, places each
object near the current Revo2 palm, closes the hand with joint targets, then
raises the Franka target a little. The output is a short video and a JSON file
with contact/lift metrics so we can separate embodiment/control issues from RL.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
from isaacgym import gymapi

from hydra import compose, initialize_config_dir
import imageio.v2 as imageio
import isaacgymenvs
from omegaconf import open_dict
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="SimToolRealDynamicGraspV65FrankaBrainCoRevo2PrivPointCloudLiftEscape",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--out-dir", default="preview_videos")
    parser.add_argument("--output-prefix", default="brainco_revo2_scripted_grasp_sanity")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=540)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--default-arm-dof-pos",
        type=float,
        nargs=7,
        default=None,
        help="Optional 7-DOF Franka default arm pose override.",
    )
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--close-steps", type=int, default=110)
    parser.add_argument("--lift-steps", type=int, default=110)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--probe-close-steps", type=int, default=100)
    parser.add_argument(
        "--placement-mode",
        choices=["open_axis", "closed_center", "closed_center_on_table"],
        default="closed_center_on_table",
    )
    parser.add_argument(
        "--object-offset",
        type=float,
        default=0.20,
        help="Distance from palm along the finger axis for object placement.",
    )
    parser.add_argument("--close-fraction", type=float, default=0.92)
    parser.add_argument("--lift-joint2-delta", type=float, default=-0.28)
    parser.add_argument("--lift-joint4-delta", type=float, default=0.28)
    parser.add_argument("--lift-joint6-delta", type=float, default=0.10)
    parser.add_argument("--lifted-threshold", type=float, default=0.06)
    return parser.parse_args()


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-8)


def _render_frame(env) -> torch.Tensor:
    env.enable_viewer_sync = True
    env.gym.render_all_camera_sensors(env.sim)
    color_image = env.gym.get_camera_image(
        env.sim,
        env.envs[env.index_to_view],
        env.camera_handle,
        gymapi.IMAGE_COLOR,
    )
    if color_image.size == 0:
        raise RuntimeError("camera image is empty")
    return color_image.reshape(
        env.camera_properties.height,
        env.camera_properties.width,
        4,
    )[..., :3].copy()


def _mean_float(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu().item())


def _per_env_float(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().float().cpu().tolist()]


def _collect_metrics(env, placed_z: torch.Tensor, lifted_threshold: float) -> Dict:
    env.populate_sim_buffers()
    object_lift = env.object_pos[:, 2] - placed_z
    lifted = object_lift > lifted_threshold
    object_palm_dist = torch.norm(env.object_pos - env.palm_center_pos, dim=-1)
    true_grasp_metrics = env._dynamic_true_grasp_metrics(object_palm_dist, lifted)
    fingertip_dist = torch.norm(
        env.fingertip_pos_offset - env.object_pos[:, None, :], dim=-1
    )

    metrics = {
        "mean_object_lift": _mean_float(object_lift),
        "max_object_lift": float(object_lift.detach().max().cpu().item()),
        "lifted_fraction": _mean_float(lifted),
        "object_palm_dist": _mean_float(object_palm_dist),
        "closest_fingertip_dist": _mean_float(fingertip_dist.min(dim=-1).values),
        "mean_fingertip_dist": _mean_float(fingertip_dist.mean(dim=-1)),
        "finger_contact_count": _mean_float(
            true_grasp_metrics["finger_contact_count"]
        ),
        "non_thumb_contact_count": _mean_float(
            true_grasp_metrics["non_thumb_contact_count"]
        ),
        "thumb_contact_fraction": _mean_float(
            true_grasp_metrics["thumb_contact"].float()
        ),
        "opposing_contact_fraction": _mean_float(
            true_grasp_metrics["opposing_contact"].float()
        ),
        "true_grasp_fraction": _mean_float(
            true_grasp_metrics["true_grasp"].float()
        ),
        "grasp_quality": _mean_float(true_grasp_metrics["grasp_quality"]),
        "per_env_object_lift": _per_env_float(object_lift),
        "per_env_true_grasp": _per_env_float(
            true_grasp_metrics["true_grasp"].float()
        ),
        "per_env_grasp_quality": _per_env_float(
            true_grasp_metrics["grasp_quality"]
        ),
        "per_env_thumb_contact": _per_env_float(
            true_grasp_metrics["thumb_contact"].float()
        ),
        "per_env_non_thumb_contact_count": _per_env_float(
            true_grasp_metrics["non_thumb_contact_count"]
        ),
    }
    return metrics


def _step(env, targets: torch.Tensor) -> None:
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.rl_device)
    env.step(actions, joint_pos_targets=targets)


def _hard_reset_env(env) -> None:
    env.reset_idx(torch.arange(env.num_envs, device=env.device), tensor_reset=True)
    env.set_actor_root_state_tensor_indexed()
    env.set_dof_state_tensor_indexed()
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.populate_sim_buffers()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mp4_path = out_dir / f"{args.output_prefix}_{stamp}.mp4"
    png_path = out_dir / f"{args.output_prefix}_{stamp}.png"
    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"

    cfg_dir = Path(__file__).resolve().parents[1] / "isaacgymenvs" / "cfg"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task={args.task}",
                "headless=True",
                "capture_video=False",
                "wandb_activate=False",
            ],
        )

    with open_dict(cfg):
        cfg.task.env.numEnvs = args.num_envs
        cfg.task.env.videoCameraWidth = args.camera_width
        cfg.task.env.videoCameraHeight = args.camera_height
        cfg.task.env.dynamicGraspInitialSpeedRange = [0.0, 0.0]
        cfg.task.env.dynamicGraspSpeedCurriculum = False
        cfg.task.env.forceScale = 0.0
        cfg.task.env.torqueScale = 0.0
        cfg.task.env.useActionDelay = False
        cfg.task.env.resetDofPosRandomIntervalArm = 0.0
        cfg.task.env.resetDofPosRandomIntervalFingers = 0.0
        cfg.task.env.resetDofVelRandomInterval = 0.0
        if args.default_arm_dof_pos is not None:
            cfg.task.env.defaultArmDofPos = list(args.default_arm_dof_pos)
        cfg.seed = 0
        cfg.sim_device = args.sim_device
        cfg.rl_device = args.rl_device
        cfg.graphics_device_id = args.graphics_device_id
        cfg.headless = True
        cfg.capture_video = False
        cfg.force_render = False

    env = isaacgymenvs.make(
        seed=0,
        task=cfg.task.name,
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        force_render=False,
        cfg=cfg,
    )
    env.cfg["env"]["capture_video"] = False
    _hard_reset_env(env)

    object_labels = getattr(env, "object_asset_labels", None)
    if object_labels is None:
        object_labels = [f"env_{i}" for i in range(env.num_envs)]
    object_labels = [str(object_labels[int(i)]) for i in env.object_asset_indices.cpu()]

    lower = env.arm_hand_dof_lower_limits[: env.num_hand_arm_dofs]
    upper = env.arm_hand_dof_upper_limits[: env.num_hand_arm_dofs]
    start_targets = env.cur_targets[:, : env.num_hand_arm_dofs].clone()
    open_targets = start_targets.clone()
    close_targets = start_targets.clone()
    close_hand = lower[env.num_arm_dofs :] + float(args.close_fraction) * (
        upper[env.num_arm_dofs :] - lower[env.num_arm_dofs :]
    )
    close_targets[:, env.num_arm_dofs :] = close_hand.unsqueeze(0)
    lift_targets = close_targets.clone()
    lift_targets[:, 1] += float(args.lift_joint2_delta)
    lift_targets[:, 3] += float(args.lift_joint4_delta)
    lift_targets[:, 5] += float(args.lift_joint6_delta)
    lift_targets = torch.max(torch.min(lift_targets, upper.unsqueeze(0)), lower.unsqueeze(0))

    open_palm = env.palm_center_pos.clone()
    open_tips = env.fingertip_pos_offset.clone()
    for _ in range(args.probe_close_steps):
        _step(env, close_targets)
    closed_palm = env.palm_center_pos.clone()
    closed_tips = env.fingertip_pos_offset.clone()
    closed_thumb = closed_tips[:, env.thumb_fingertip_idx, :]
    closed_non_thumb = closed_tips[:, env.non_thumb_fingertip_indices, :].mean(dim=1)
    closed_center = 0.5 * (closed_thumb + closed_non_thumb)
    closed_gap = torch.norm(closed_thumb - closed_non_thumb, dim=-1)

    _hard_reset_env(env)
    start_targets = env.cur_targets[:, : env.num_hand_arm_dofs].clone()
    open_targets = start_targets.clone()
    close_targets[:, : env.num_arm_dofs] = start_targets[:, : env.num_arm_dofs]
    lift_targets[:, : env.num_arm_dofs] = start_targets[:, : env.num_arm_dofs]
    lift_targets[:, 1] += float(args.lift_joint2_delta)
    lift_targets[:, 3] += float(args.lift_joint4_delta)
    lift_targets[:, 5] += float(args.lift_joint6_delta)
    lift_targets = torch.max(torch.min(lift_targets, upper.unsqueeze(0)), lower.unsqueeze(0))

    palm = env.palm_center_pos
    non_thumb = env.fingertip_pos_offset[:, env.non_thumb_fingertip_indices, :]
    finger_axis = _normalize(non_thumb.mean(dim=1) - palm)
    if args.placement_mode == "open_axis":
        place_pos = palm + float(args.object_offset) * finger_axis
    elif args.placement_mode == "closed_center":
        place_pos = closed_center.clone()
    else:
        place_pos = closed_center.clone()
        place_pos[:, 2] = env.object_init_state[:, 2]
    placed_z = place_pos[:, 2].clone()

    obj_indices = env.object_indices.to(torch.int32)
    env.root_state_tensor[obj_indices, 0:3] = place_pos
    env.root_state_tensor[obj_indices, 7:13] = 0.0
    env.object_init_state[:, 0:3] = place_pos
    env.object_init_state[:, 7:13] = 0.0
    env.dynamic_grasp_object_xy_velocity[:] = 0.0
    env.dynamic_grasp_object_yaw_rate[:] = 0.0
    env.deferred_set_actor_root_state_tensor_indexed([obj_indices])
    env.set_actor_root_state_tensor_indexed()
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.populate_sim_buffers()

    frames = []
    metrics_by_phase: Dict[str, Dict] = {}
    stage_plan = [
        ("settle_open", args.settle_steps, open_targets, open_targets),
        ("close", args.close_steps, open_targets, close_targets),
        ("lift", args.lift_steps, close_targets, lift_targets),
        ("hold", args.hold_steps, lift_targets, lift_targets),
    ]

    for phase_name, num_steps, phase_start, phase_end in stage_plan:
        for step_idx in range(num_steps):
            if num_steps <= 1:
                alpha = 1.0
            else:
                alpha = float(step_idx + 1) / float(num_steps)
            targets = phase_start + alpha * (phase_end - phase_start)
            _step(env, targets)
            if not args.no_video:
                frames.append(_render_frame(env))
        metrics_by_phase[phase_name] = _collect_metrics(
            env, placed_z, args.lifted_threshold
        )
        print(f"phase={phase_name} metrics={metrics_by_phase[phase_name]}", flush=True)

    summary = {
        "task": args.task,
        "num_envs": env.num_envs,
        "object_labels": object_labels,
        "hand_dof_names": list(env.gym.get_actor_dof_names(env.envs[0], 0))[
            env.num_arm_dofs : env.num_hand_arm_dofs
        ],
        "hand_lower_limits": _per_env_float(lower[env.num_arm_dofs :]),
        "hand_upper_limits": _per_env_float(upper[env.num_arm_dofs :]),
        "object_offset": args.object_offset,
        "placement_mode": args.placement_mode,
        "default_arm_dof_pos": (
            list(args.default_arm_dof_pos)
            if args.default_arm_dof_pos is not None
            else list(cfg.task.env.defaultArmDofPos)
        ),
        "close_fraction": args.close_fraction,
        "probe": {
            "open_palm_z_mean": _mean_float(open_palm[:, 2]),
            "closed_palm_z_mean": _mean_float(closed_palm[:, 2]),
            "closed_center_z_mean": _mean_float(closed_center[:, 2]),
            "closed_gap_mean": _mean_float(closed_gap),
            "closed_gap_per_env": _per_env_float(closed_gap),
            "closed_center_table_z_delta_mean": _mean_float(
                closed_center[:, 2] - env.object_init_state[:, 2]
            ),
        },
        "lift_joint_delta": {
            "panda_joint2": args.lift_joint2_delta,
            "panda_joint4": args.lift_joint4_delta,
            "panda_joint6": args.lift_joint6_delta,
        },
        "metrics_by_phase": metrics_by_phase,
        "mp4": None if args.no_video else str(mp4_path),
        "png": None if args.no_video else str(png_path),
    }

    if not args.no_video:
        imageio.imwrite(png_path, frames[-1])
        imageio.mimsave(mp4_path, frames, fps=args.fps)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_video:
        print(f"saved_png {png_path}", flush=True)
        print(f"saved_mp4 {mp4_path}", flush=True)
    print(f"saved_json {json_path}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
