#!/usr/bin/env python
"""Save a short camera preview for the v33 Franka + Inspire environment."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
from isaacgym import gymapi

import isaacgymenvs
import imageio.v2 as imageio
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="SimToolRealDynamicGraspV33FrankaInspireAffordanceDomino20PointNet",
    )
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", default="preview_videos")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=None,
        help="Optional camera xyz position in the selected env frame.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=None,
        help="Optional camera look-at xyz target in the selected env frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    png_path = out_dir / f"franka_inspire_v33_env_preview_{stamp}.png"
    mp4_path = out_dir / f"franka_inspire_v33_env_preview_{stamp}.mp4"

    env = isaacgymenvs.make(
        seed=0,
        task=args.task,
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        force_render=False,
    )
    env.cfg["env"]["capture_video"] = False
    if args.camera_pos is not None or args.camera_target is not None:
        if args.camera_pos is None or args.camera_target is None:
            raise ValueError("--camera-pos and --camera-target must be set together")
        env.gym.set_camera_location(
            env.camera_handle,
            env.envs[env.index_to_view],
            gymapi.Vec3(*args.camera_pos),
            gymapi.Vec3(*args.camera_target),
        )

    print(f"created {args.task}", flush=True)
    print(
        "interface "
        f"{env.policy_action_interface} dofs "
        f"{env.num_arm_dofs}+{env.num_hand_dofs}={env.num_hand_arm_dofs} "
        f"actions {env.num_actions}",
        flush=True,
    )
    print(f"palm {env.palm_body_name} fingertips {env.fingertips}", flush=True)
    print(
        f"camera {env.camera_properties.width}x{env.camera_properties.height} "
        f"env {env.index_to_view}",
        flush=True,
    )

    frames = []
    for step in range(args.steps):
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.rl_device)
        if step >= args.steps * 3 // 8:
            actions[:, env.num_arm_dofs : env.num_hand_arm_dofs] = 0.75
        if step >= args.steps * 2 // 3:
            actions[:, env.num_arm_dofs : env.num_hand_arm_dofs] = -0.25

        env.step(actions)
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
        frame = color_image.reshape(
            env.camera_properties.height,
            env.camera_properties.width,
            4,
        )[..., :3].copy()
        frames.append(frame)

    imageio.imwrite(png_path, frames[0])
    imageio.mimsave(mp4_path, frames, fps=args.fps)
    print(f"saved_png {png_path}", flush=True)
    print(f"saved_mp4 {mp4_path}", flush=True)

    # Isaac Gym sometimes keeps background resources alive after headless camera
    # runs; exit directly after files are flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
