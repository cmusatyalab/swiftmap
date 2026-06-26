# Copyright (C) 2024 Carnegie Mellon University
"""
Viser viewer for the confidence-difference Next Flight Navigation planner.

Renders:
  - the reconstruction point cloud (with an adjustable confidence-percentile filter),
  - the low-confidence "to-improve" band (red),
  - the fitted per-cluster surface normals (yellow),
  - the suggested next viewpoints (blue frustums, with a line to their target),
  - the existing camera poses (green).
"""

import numpy as np
import viser
import viser.transforms as viser_tf
from typing import Dict, Optional, Tuple
import time


def visualize_nfn_with_viser(
    predictions: Dict,
    plan: Dict,
    port: int = 7867,
    conf_threshold: float = 50.0,
    background_mode: bool = False,
) -> Optional[viser.ViserServer]:
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(control_layout="collapsible")

    world_points = np.asarray(predictions["world_points"]).reshape(-1, 3)
    conf_flat = np.asarray(predictions["world_points_conf"]).reshape(-1)
    images = predictions.get("images")
    if images is not None:
        colors = (np.asarray(images).transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    else:
        colors = np.full((len(world_points), 3), 128, np.uint8)

    finite = np.isfinite(world_points).all(1) & np.isfinite(conf_flat)
    valid_conf = conf_flat[finite]

    def filtered(thr_pct: float):
        t = np.percentile(valid_conf, thr_pct) if len(valid_conf) else 0.0
        m = finite & (conf_flat >= t)
        return world_points[m], colors[m]

    diag = float(plan.get("statistics", {}).get("scene_diagonal", 1.0)) or 1.0

    # Base reconstruction point cloud
    base_point_size = diag * 0.0015
    pts0, col0 = filtered(conf_threshold)
    cloud = server.scene.add_point_cloud("/nfn/cloud", points=pts0, colors=col0,
                                         point_size=base_point_size)
    print(f"[NFN Visualization] Base cloud: {len(pts0)}/{len(world_points)} points")

    # "To-improve" band points (red)
    enhance = np.asarray(plan.get("enhance_points", np.empty((0, 3))))
    if len(enhance):
        enhance = enhance[np.isfinite(enhance).all(1)]
    if len(enhance) > 60000:
        enhance = enhance[np.random.choice(len(enhance), 60000, replace=False)]
    enhance_point_size = diag * 0.0012
    enhance_handle = None
    if len(enhance):
        enhance_handle = server.scene.add_point_cloud(
            "/nfn/to_improve", points=enhance,
            colors=np.tile(np.array([[255, 50, 0]], np.uint8), (len(enhance), 1)),
            point_size=enhance_point_size)

    # Existing cameras (green)
    ext = predictions.get("extrinsic")
    if ext is not None:
        _add_existing_cameras(server, np.asarray(ext), scale=diag * 0.015)

    # Cluster normals (yellow arrows) + suggested viewpoints (blue frustums + target lines)
    for c in plan.get("clusters", []):
        start = np.asarray(c["centroid"])
        end = start + np.asarray(c["normal"]) * diag * 0.05
        _add_line(server, f"/nfn/normals/n{c['id']}", start, end, (255, 220, 0), diag)

    for i, vp in enumerate(plan.get("viewpoints", [])):
        pos = np.asarray(vp["camera_position"])
        tgt = np.asarray(vp["target"])
        wxyz = viser_tf.SO3.from_matrix(np.asarray(vp["camera_rotation"])).wxyz
        server.scene.add_camera_frustum(
            f"/nfn/viewpoints/v{i}", fov=np.deg2rad(50.0), aspect=1.0, scale=diag * 0.012,
            color=(0, 80, 255), position=pos, wxyz=wxyz)
        _add_line(server, f"/nfn/viewpoints/line{i}", pos, tgt, (0, 80, 255), diag)

    # ---- GUI ----
    thr = server.gui.add_slider("Confidence threshold (%)", min=0, max=100, step=1,
                                initial_value=conf_threshold)

    @thr.on_update
    def _update_cloud(_):
        p, c = filtered(thr.value)
        cloud.points, cloud.colors = p, c

    # Point-size control (multiplier of the scene-relative default).
    point_size = server.gui.add_slider("Point size (x)", min=0.1, max=5.0, step=0.1,
                                       initial_value=1.0)

    @point_size.on_update
    def _update_point_size(_):
        cloud.point_size = base_point_size * point_size.value
        if enhance_handle is not None:
            enhance_handle.point_size = enhance_point_size * point_size.value

    if enhance_handle is not None:
        toggle = server.gui.add_checkbox("Show to-improve points", True)

        @toggle.on_update
        def _toggle(_):
            enhance_handle.visible = toggle.value

    th, st = plan.get("thresholds", {}), plan.get("statistics", {})
    server.gui.add_markdown(
        f"**NFN — confidence-difference plan**\n\n"
        f"- Band: P{th.get('low_percentile', 60):.0f}–P{th.get('high_percentile', 80):.0f} "
        f"(conf {th.get('p_low', 0):.2f}–{th.get('p_high', 0):.2f})\n"
        f"- To-improve points: {st.get('num_enhance_points', 0)}\n"
        f"- Clusters: {st.get('num_clusters', 0)}\n"
        f"- Suggested viewpoints: {plan.get('num_viewpoints', 0)}\n\n"
        f"🔴 to-improve · 🟡 normal · 🔵 next viewpoint · 🟢 existing camera"
    )

    print(f"[NFN Visualization] Viser server ready at http://localhost:{port}")

    if background_mode:
        # Viser serves in its own threads; hand the server back so the caller can stop it.
        return server
    print("[NFN Visualization] Press Ctrl+C to exit")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[NFN Visualization] Shutting down...")
    return server


def _add_existing_cameras(server: viser.ViserServer, extrinsics: np.ndarray, scale: float):
    print(f"[NFN Visualization] Adding {len(extrinsics)} existing camera poses")
    for i, ext in enumerate(extrinsics):
        R, t = ext[:3, :3], ext[:3, 3]
        world_pos = -R.T @ t
        wxyz = viser_tf.SO3.from_matrix(R.T).wxyz
        server.scene.add_camera_frustum(
            f"/nfn/existing_cameras/c{i}", fov=np.deg2rad(60.0), aspect=1.0, scale=scale,
            color=(0, 200, 0), position=world_pos, wxyz=wxyz)


def _add_line(server: viser.ViserServer, name: str, start: np.ndarray, end: np.ndarray,
              color: Tuple[int, int, int], diag: float):
    """Draw a line segment as a thin cylinder."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return
    direction = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        wxyz = np.array([1.0, 0, 0, 0]) if np.dot(z_axis, direction) > 0 else np.array([0.0, 1, 0, 0])
    else:
        axis = axis / axis_norm
        half = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)) / 2
        wxyz = np.array([np.cos(half), *(axis * np.sin(half))])
    server.scene.add_mesh_simple(
        name=name,
        vertices=_cylinder_vertices(length, diag * 0.0008),
        faces=_cylinder_faces(),
        color=color,
        position=(start + end) / 2,
        wxyz=wxyz,
    )


def _cylinder_vertices(height: float, radius: float, segments: int = 8) -> np.ndarray:
    angles = 2 * np.pi * np.arange(segments) / segments
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    bottom = np.column_stack([ring, np.full(segments, -height / 2)])
    top = np.column_stack([ring, np.full(segments, height / 2)])
    return np.vstack([bottom, top]).astype(np.float32)


def _cylinder_faces(segments: int = 8) -> np.ndarray:
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + i])
        faces.append([j, segments + j, segments + i])
    return np.array(faces, dtype=np.uint32)
