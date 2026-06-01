#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def unit_surface(num_points: int, seed: int):
    rng = np.random.default_rng(seed)
    faces = rng.integers(0, 6, size=num_points)
    uv = rng.uniform(-0.5, 0.5, size=(num_points, 2)).astype(np.float32)
    points = np.zeros((num_points, 3), dtype=np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)

    for point_idx, face in enumerate(faces):
        axis = face // 2
        sign = 0.5 if face % 2 == 0 else -0.5
        free_axes = [idx for idx in range(3) if idx != axis]
        points[point_idx, axis] = sign
        points[point_idx, free_axes[0]] = uv[point_idx, 0]
        points[point_idx, free_axes[1]] = uv[point_idx, 1]
        normals[point_idx, axis] = 1.0 if sign > 0.0 else -1.0

    points -= points.mean(axis=0, keepdims=True)
    return points, normals


def euler_xyz(roll: float, pitch: float, yaw: float):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def box_corners(size, rot, pos):
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return signs * (size / 2.0) @ rot.T + pos


def draw_box(ax, corners, color="black", alpha=0.45):
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            alpha=alpha,
            linewidth=0.8,
        )


def set_axes(ax, title, lim=0.12):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1, 1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/object_pointcloud_sim2real_preview.png"),
    )
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed + 7)

    num_points = 128
    candidate_multiplier = 6
    candidate_points = num_points * candidate_multiplier
    object_size = np.array([0.141, 0.03025, 0.0271], dtype=np.float32)
    camera_pos = np.array([0.45, -0.75, 0.78], dtype=np.float32)
    camera_pos = camera_pos + rng.normal(0.0, 0.015, size=3)
    object_pos = np.zeros(3, dtype=np.float32)
    object_rot = euler_xyz(np.deg2rad(10), np.deg2rad(-5), np.deg2rad(28))

    unit_points, unit_normals = unit_surface(candidate_points, args.seed)
    full_points = unit_points * object_size
    world_points = full_points @ object_rot.T + object_pos
    world_normals = unit_normals @ object_rot.T

    point_to_camera = camera_pos[None, :] - world_points
    point_to_camera_unit = point_to_camera / np.maximum(
        np.linalg.norm(point_to_camera, axis=-1, keepdims=True), 1e-6
    )
    visible_scores = np.sum(world_normals * point_to_camera_unit, axis=-1)
    visible_scores = visible_scores + 0.01 * rng.normal(size=visible_scores.shape)
    selected_ids = np.argsort(visible_scores)[-num_points:][::-1]
    selected_points = world_points[selected_ids].copy()

    camera_ray = selected_points - camera_pos[None, :]
    camera_ray_unit = camera_ray / np.maximum(
        np.linalg.norm(camera_ray, axis=-1, keepdims=True), 1e-6
    )
    selected_points += camera_ray_unit * rng.normal(0.0, 0.006, size=(num_points, 1))

    centroid = selected_points.mean(axis=0)
    final_points = selected_points + rng.normal(0.0, 0.004, size=selected_points.shape)

    keep_mask = rng.random((num_points, 1)) > 0.25
    dropped_mask = ~keep_mask[:, 0]
    final_points = np.where(keep_mask, final_points, centroid[None, :])

    outlier_mask = rng.random((num_points, 1)) < 0.03
    outlier_offsets = (rng.random((num_points, 3)) * 2.0 - 1.0) * 0.08
    final_points = np.where(outlier_mask, centroid[None, :] + outlier_offsets, final_points)
    outlier_mask = outlier_mask[:, 0]

    order = np.argsort(rng.random(num_points))
    final_points = final_points[order]
    dropped_mask = dropped_mask[order]
    outlier_mask = outlier_mask[order]
    regular_mask = ~(dropped_mask | outlier_mask)

    visible_conf = np.mean(visible_scores > 0.0)
    keep_conf = np.mean(keep_mask)
    outlier_conf = 1.0 - np.mean(outlier_mask)
    tracking_confidence = visible_conf * keep_conf * outlier_conf

    corners = box_corners(object_size, object_rot, object_pos)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(
        "SimToolRealDynamicGraspSim2RealPointCloud observation preview",
        fontsize=14,
    )

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], s=8, c="0.75")
    draw_box(ax1, corners)
    set_axes(ax1, "candidate object surface samples (768)")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(
        selected_points[:, 0],
        selected_points[:, 1],
        selected_points[:, 2],
        s=22,
        c=visible_scores[selected_ids],
        cmap="viridis",
    )
    ax2.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], s=90, c="red", marker="^")
    draw_box(ax2, corners)
    set_axes(ax2, "camera-visible partial crop (128)")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.scatter(
        final_points[regular_mask, 0],
        final_points[regular_mask, 1],
        final_points[regular_mask, 2],
        s=24,
        c="#1f77b4",
        label="kept noisy points",
    )
    ax3.scatter(
        final_points[dropped_mask, 0],
        final_points[dropped_mask, 1],
        final_points[dropped_mask, 2],
        s=34,
        c="0.35",
        label="dropout -> centroid",
    )
    ax3.scatter(
        final_points[outlier_mask, 0],
        final_points[outlier_mask, 1],
        final_points[outlier_mask, 2],
        s=42,
        c="#ff7f0e",
        label="mask/depth outliers",
    )
    ax3.scatter([centroid[0]], [centroid[1]], [centroid[2]], s=70, c="black", marker="x")
    draw_box(ax3, corners)
    set_axes(ax3, "final actor observation pointcloud (128)")
    ax3.legend(loc="upper left", fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(
        final_points[regular_mask, 0],
        final_points[regular_mask, 1],
        s=24,
        c="#1f77b4",
        alpha=0.8,
    )
    ax4.scatter(
        final_points[dropped_mask, 0],
        final_points[dropped_mask, 1],
        s=34,
        c="0.35",
        alpha=0.8,
    )
    ax4.scatter(
        final_points[outlier_mask, 0],
        final_points[outlier_mask, 1],
        s=42,
        c="#ff7f0e",
        alpha=0.9,
    )
    ax4.scatter([centroid[0]], [centroid[1]], s=80, c="black", marker="x")
    ax4.set_title("top-down final observation", fontsize=10)
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect("equal", adjustable="box")
    ax4.grid(True, alpha=0.25)
    ax4.text(
        0.03,
        0.97,
        (
            f"points: {num_points}\n"
            f"visible confidence: {visible_conf:.2f}\n"
            f"kept after dropout: {keep_conf:.2f}\n"
            f"tracking confidence: {tracking_confidence:.2f}\n"
            f"dropout points: {int(dropped_mask.sum())}\n"
            f"outliers: {int(outlier_mask.sum())}"
        ),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
