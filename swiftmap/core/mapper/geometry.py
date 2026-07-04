# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic camera geometry helpers shared by the mapper backends.

Some backbones (e.g. VGGT) predict world points directly with a dedicated point
head; others (e.g. VGGT-Omega) only predict per-pixel depth, so the point cloud
has to be recovered by unprojecting depth through the camera intrinsics and
extrinsics. This module owns that math so it lives in SwiftMap rather than being
duplicated per backbone.
"""

import numpy as np


def unproject_depth_map_to_point_map(depth_map: np.ndarray,
                                     extrinsic: np.ndarray,
                                     intrinsic: np.ndarray) -> np.ndarray:
    """Unproject per-frame depth maps into world-space 3D points.

    Args:
        depth_map: (S, H, W, 1) or (S, H, W) depth per frame.
        extrinsic: (S, 3, 4) world-to-camera matrices [R | t].
        intrinsic: (S, 3, 3) pinhole intrinsics.

    Returns:
        (S, H, W, 3) world-space points, matching the layout the scene exporter
        expects for ``world_points_from_depth``.
    """
    depth = depth_map[..., 0] if depth_map.ndim == 4 else depth_map
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    # Pixel -> camera-space ray * depth.
    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    # Camera -> world: world = R^T @ (cam - t).
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def camera_poses_from_extrinsics(extrinsic: np.ndarray):
    """Return (positions, rotations) in world coordinates from (S,3,4) extrinsics.

    Camera position: ``-R^T @ t``; camera rotation: ``R^T``.
    """
    positions, rotations = [], []
    for ext in extrinsic:
        R = ext[:3, :3]
        t = ext[:3, 3]
        positions.append(-R.T @ t)
        rotations.append(R.T)
    return np.array(positions), np.array(rotations)
