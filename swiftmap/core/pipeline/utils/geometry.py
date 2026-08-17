# Copyright (C) 2024 Carnegie Mellon University

"""Drawing helpers (pure numpy/trimesh): trimesh point clouds and scenes, camera
frustums, and PLY writing. Used only by the pipeline."""

import numpy as np

_CONF_EPSILON = 1e-6


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
