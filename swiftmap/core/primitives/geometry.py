# Copyright (C) 2024 Carnegie Mellon University

"""Shared point-cloud primitives (pure numpy/trimesh): color flattening, confidence
masks, trimesh point clouds and scenes, PLY writing, and the scene-alignment
transforms. A leaf that ``types``, ``map`` and the pipeline build on; single-consumer
geometry lives with its consumer instead."""

import numpy as np

_CONF_EPSILON = 1e-6


def flatten_colors(images: np.ndarray) -> np.ndarray:
    """(S,3,H,W) or (S,H,W,3) float[0,1] images -> (N,3) uint8 RGB."""
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    return (images.reshape(-1, 3) * 255).astype(np.uint8)


def confidence_mask(conf, percentile) -> np.ndarray:
    """Boolean keep-mask over a flat confidence array: ``conf >= P``-th percentile
    and strictly positive. ``percentile`` in [0, 100]; 0/None keeps all positive."""
    conf = np.asarray(conf, dtype=float).reshape(-1)
    if not percentile:
        return conf > _CONF_EPSILON
    thr = np.percentile(conf, float(percentile))
    return (conf >= thr) & (conf > _CONF_EPSILON)


def voxel_merge(points, colors, conf, voxel_size: float):
    """Collapse points sharing a ``voxel_size`` grid cell into one, confidence-weighted.

    Position and color become the confidence-weighted mean of the cell; the merged
    confidence is the cell's max. ``voxel_size <= 0`` returns the input unchanged.
    """
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0 or len(points) == 0:
        return points, np.asarray(colors), np.asarray(conf, dtype=float)

    w = np.asarray(conf, dtype=float)
    vidx = np.floor(points / voxel_size).astype(np.int64)
    vidx -= vidx.min(axis=0)
    dims = vidx.max(axis=0) + 1
    if dims.prod() >= np.iinfo(np.int64).max:
        raise OverflowError("voxel grid too large for a linear key; raise the voxel size")
    key = (vidx[:, 0] * dims[1] + vidx[:, 1]) * dims[2] + vidx[:, 2]
    _, inv = np.unique(key, return_inverse=True)

    g = inv.max() + 1
    wsum = np.bincount(inv, weights=w, minlength=g)
    wsafe = np.where(wsum > 0, wsum, 1.0)
    pos = np.empty((g, 3))
    col = np.empty((g, 3))
    cols_f = np.asarray(colors, dtype=float)
    for k in range(3):
        pos[:, k] = np.bincount(inv, weights=w * points[:, k], minlength=g) / wsafe
        col[:, k] = np.bincount(inv, weights=w * cols_f[:, k], minlength=g) / wsafe
    mconf = np.zeros(g)
    np.maximum.at(mconf, inv, w)
    return pos, np.clip(col, 0, 255).astype(np.uint8), mconf


# ---------------------------------------------------------------------- meshing
def pointcloud(points, colors):
    """trimesh PointCloud from (N,3) points + (N,3) uint8 RGB (alpha filled)."""
    import trimesh
    rgba = np.hstack([np.asarray(colors, np.uint8),
                      np.full((len(points), 1), 255, np.uint8)])
    return trimesh.PointCloud(vertices=np.asarray(points), colors=rgba)


def camera_frustums(frames, size: float, rgba):
    """One Trimesh of camera-frustum pyramids for ``frames`` (in their coordinate frame)."""
    import trimesh
    corners = np.array([[-0.5, -0.375, 1.0], [0.5, -0.375, 1.0],
                        [0.5, 0.375, 1.0], [-0.5, 0.375, 1.0]])
    verts = np.zeros((len(frames) * 5, 3))
    faces = np.zeros((len(frames) * 4, 3), dtype=np.int64)
    for k, fr in enumerate(frames):
        r = np.asarray(fr["rotation_matrix"], float)
        c = np.asarray(fr["camera_position_world"], float)
        base = (r.T @ (size * corners).T).T + c
        vb = 5 * k
        verts[vb] = c
        verts[vb + 1:vb + 5] = base
        for i in range(4):
            faces[4 * k + i] = [vb, vb + 1 + i, vb + 1 + (i + 1) % 4]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(np.asarray(rgba, np.uint8), (len(faces), 1))
    return mesh


def pointcloud_scene(points, colors, frames=None,
                     frustum_rgba=(20, 20, 20, 255), frustum_scale: float = 0.01):
    """trimesh.Scene of a point cloud plus optional camera frustums, in the world frame.

    Frustum size scales with the cloud's diagonal (clamped to [1, 5] m). The single
    world-coordinate scene builder shared by ``Map`` write and render.
    """
    import trimesh
    points = np.asarray(points)
    scene = trimesh.Scene()
    scene.add_geometry(pointcloud(points, colors), geom_name="points")
    if frames and len(points):
        size = float(np.clip(np.linalg.norm(points.max(0) - points.min(0)) * frustum_scale, 1.0, 5.0))
        scene.add_geometry(camera_frustums(frames, size, frustum_rgba), geom_name="cameras")
    return scene


def write_ply(path, points, colors, confidence=None, comments=None) -> str:
    """Write an ASCII PLY of xyz + rgb(+alpha) (+per-point confidence). ``colors`` is
    (N,3) or (N,4) uint8; ``comments`` is an optional list of header comment lines."""
    points = np.asarray(points)
    colors = np.asarray(colors, np.uint8)
    alpha = colors.shape[1] == 4
    conf = None if confidence is None else np.asarray(confidence, dtype=float).reshape(-1)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        for line in (comments or []):
            f.write(f"comment {line}\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if alpha:
            f.write("property uchar alpha\n")
        if conf is not None:
            f.write("property float confidence\n")
        f.write("end_header\n")
        for i in range(len(points)):
            x, y, z = points[i]
            row = f"{x:.6f} {y:.6f} {z:.6f} " + " ".join(str(int(v)) for v in colors[i])
            if conf is not None:
                row += f" {conf[i]:.6f}"
            f.write(row + "\n")
    return path


# ------------------------------------------------------------- scene alignment
def opengl_conversion_matrix() -> np.ndarray:
    """4x4 matrix flipping the y and z axes (trimesh <-> OpenGL viewer convention)."""
    m = np.identity(4)
    m[1, 1] = -1
    m[2, 2] = -1
    return m


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """Apply a 4x4 ``transformation`` to (..., D) ``points``."""
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    t = transformation.swapaxes(-1, -2)
    points = points @ t[..., :-1, :] + t[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def apply_scene_alignment(scene, extrinsics_matrices: np.ndarray):
    """Align a trimesh scene to the first camera's view (OpenGL flip + 180deg about y)."""
    from scipy.spatial.transform import Rotation
    align = np.eye(4)
    align[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    scene.apply_transform(np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix() @ align)
    return scene
