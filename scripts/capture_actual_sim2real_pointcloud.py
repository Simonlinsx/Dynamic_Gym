#!/usr/bin/env python
import argparse
from pathlib import Path

# Import Isaac Gym before torch.
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict

import isaacgymenvs
from isaacgymenvs.utils.torch_jit_utils import quat_rotate_inverse


def obs_offsets(env):
    offsets = {}
    start = 0
    for name in env.obs_list:
        width = env.obs_type_size_dict[name]
        offsets[name] = (start, start + width)
        start += width
    return offsets


def slice_obs(obs, offsets, name):
    start, end = offsets[name]
    return obs[:, start:end]


def plot_clouds(
    output_png,
    cloud,
    centroid,
    velocity,
    confidence,
    env_ids,
    centered=False,
):
    num_plots = min(len(env_ids), cloud.shape[0])
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        "Actual actor obs_buf pointcloud from SimToolRealDynamicGraspSim2RealPointCloud",
        fontsize=13,
    )

    for plot_idx in range(num_plots):
        env_id = env_ids[plot_idx]
        points = cloud[plot_idx]
        center = centroid[plot_idx]
        vel = velocity[plot_idx]
        conf = confidence[plot_idx, 0]
        if centered:
            points = points - center[None, :]
            center = np.zeros_like(center)

        ax = fig.add_subplot(2, 2, plot_idx + 1, projection="3d")
        dist = np.linalg.norm(points - center[None, :], axis=-1)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=22,
            c=dist,
            cmap="viridis",
            alpha=0.88,
        )
        ax.scatter([center[0]], [center[1]], [center[2]], s=80, c="black", marker="x")
        ax.quiver(
            center[0],
            center[1],
            center[2],
            vel[0],
            vel[1],
            vel[2],
            length=0.08,
            normalize=False,
            color="red",
            linewidth=2.0,
        )

        ax.set_title(
            f"env {env_id} | confidence {conf:.2f} | vel [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]",
            fontsize=9,
        )
        ax.set_xlabel("palm x")
        ax.set_ylabel("palm y")
        ax.set_zlabel("palm z")
        all_pts = np.concatenate([points, center[None, :]], axis=0)
        span = np.max(np.abs(all_pts))
        lim = max(0.10, float(span) * 1.15)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-55)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SimToolRealDynamicGraspSim2RealPointCloud")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--action-mode", choices=["zero", "random"], default="zero")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/actual_sim2real_pointcloud_obs"),
    )
    args = parser.parse_args()

    with initialize(config_path="../isaacgymenvs/cfg", version_base="1.1"):
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
        cfg.seed = args.seed
        cfg.sim_device = args.sim_device
        cfg.rl_device = args.rl_device
        cfg.graphics_device_id = args.graphics_device_id
        cfg.headless = True
        cfg.capture_video = False
        cfg.force_render = False

    env = isaacgymenvs.make(
        seed=args.seed,
        task=cfg.task.name,
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,
        cfg=cfg,
    )

    obs = None
    for _ in range(args.steps):
        if args.action_mode == "zero":
            actions = torch.zeros(
                (env.num_envs, env.num_actions),
                dtype=torch.float32,
                device=env.rl_device,
            )
        else:
            actions = torch.rand(
                (env.num_envs, env.num_actions),
                dtype=torch.float32,
                device=env.rl_device,
            )
            actions = actions * 2.0 - 1.0
        obs, _, _, _ = env.step(actions)

    actor_obs = obs["obs"].detach().cpu()
    offsets = obs_offsets(env)
    cloud_flat = slice_obs(actor_obs, offsets, "object_pointcloud_rel_palm")
    centroid = slice_obs(actor_obs, offsets, "object_pointcloud_centroid_rel_palm")
    velocity = slice_obs(actor_obs, offsets, "object_pointcloud_vel_rel_palm")
    confidence = slice_obs(actor_obs, offsets, "object_pointcloud_tracking_confidence")

    with torch.no_grad():
        palm_rot = env._palm_state[:, 3:7]
        gt_center_rel_palm = quat_rotate_inverse(
            palm_rot, env.object_pos - env.palm_center_pos
        )
        gt_vel_rel_palm = quat_rotate_inverse(
            palm_rot, env.object_state[:, 7:10] - env._palm_state[:, 7:10]
        )

    num_points = env.object_pointcloud_num_points
    cloud = cloud_flat.reshape(args.num_envs, num_points, 3).numpy()
    centroid_np = centroid.numpy()
    velocity_np = velocity.numpy()
    confidence_np = confidence.numpy()
    gt_center_rel_palm_np = gt_center_rel_palm.detach().cpu().numpy()
    gt_vel_rel_palm_np = gt_vel_rel_palm.detach().cpu().numpy()

    centroid_to_gt_center = np.linalg.norm(
        centroid_np - gt_center_rel_palm_np, axis=-1
    )
    visual_vel_error = np.linalg.norm(velocity_np - gt_vel_rel_palm_np, axis=-1)
    point_radius = np.linalg.norm(cloud - centroid_np[:, None, :], axis=-1)

    env_ids = list(range(min(4, args.num_envs)))
    output_png = args.output_prefix.with_suffix(".png")
    output_centered_png = args.output_prefix.with_name(
        args.output_prefix.name + "_centered"
    ).with_suffix(".png")
    output_npz = args.output_prefix.with_suffix(".npz")
    plot_clouds(output_png, cloud[env_ids], centroid_np[env_ids], velocity_np[env_ids], confidence_np[env_ids], env_ids)
    plot_clouds(
        output_centered_png,
        cloud[env_ids],
        centroid_np[env_ids],
        velocity_np[env_ids],
        confidence_np[env_ids],
        env_ids,
        centered=True,
    )

    np.savez_compressed(
        output_npz,
        task=args.task,
        steps=args.steps,
        action_mode=args.action_mode,
        obs_list=np.array(env.obs_list),
        object_pointcloud_rel_palm=cloud,
        object_pointcloud_centroid_rel_palm=centroid_np,
        object_pointcloud_vel_rel_palm=velocity_np,
        object_pointcloud_tracking_confidence=confidence_np,
        gt_object_center_rel_palm=gt_center_rel_palm_np,
        gt_object_vel_rel_palm=gt_vel_rel_palm_np,
        centroid_to_gt_center=centroid_to_gt_center,
        visual_vel_error=visual_vel_error,
        point_radius=point_radius,
        actor_obs=actor_obs.numpy(),
    )

    print(f"saved_png={output_png.resolve()}")
    print(f"saved_centered_png={output_centered_png.resolve()}")
    print(f"saved_npz={output_npz.resolve()}")
    print(f"cloud_shape={cloud.shape}")
    print(
        "confidence_min_mean_max="
        f"{confidence_np.min():.4f},{confidence_np.mean():.4f},{confidence_np.max():.4f}"
    )
    print(
        "centroid_to_gt_center_median_mean_max="
        f"{np.median(centroid_to_gt_center):.4f},"
        f"{centroid_to_gt_center.mean():.4f},"
        f"{centroid_to_gt_center.max():.4f}"
    )
    print(
        "visual_vel_error_median_mean_max="
        f"{np.median(visual_vel_error):.4f},"
        f"{visual_vel_error.mean():.4f},"
        f"{visual_vel_error.max():.4f}"
    )
    print(
        "point_radius_median_mean_max="
        f"{np.median(point_radius):.4f},"
        f"{point_radius.mean():.4f},"
        f"{point_radius.max():.4f}"
    )


if __name__ == "__main__":
    main()
