# Copyright (C) 2024 Carnegie Mellon University

"""Render a map: point-cloud + camera GLBs written into the map's own dir."""

import os

import numpy as np

from swiftmap.core.database.map import Map
from swiftmap.core.pipeline.renderer.confidence import generate_confidence_point_cloud
from swiftmap.core.primitives import geometry


def render(m: Map, conf_level: float) -> dict:
    """Reconstruction + confidence GLBs, both regenerated at ``conf_level`` (percentile)."""
    if not m.exists():
        return {"error": f"Unknown map '{m.tag}'."}
    return {"success": True, "map_tag": m.tag,
            "scene_glb": _render_scene(m, conf_level),
            "confidence_glb": _render_confidence(m, conf_level)}


def _view_path(m: Map, kind: str, conf_level) -> str:
    return os.path.join(m.path, f"{kind}_view_c{int(round(float(conf_level)))}.glb")


def _render_scene(m: Map, conf_level):
    """Point cloud + camera frustums in world coords -- same path for raw and merged."""
    try:
        data = m.load(conf_thres=0.0)
        keep = geometry.confidence_mask(data.conf, conf_level)
        scene = geometry.pointcloud_scene(data.points[keep], data.colors[keep], data.frames)
        path = _view_path(m, "reconstruction", conf_level)
        scene.export(path)
        return path
    except Exception as e:
        print(f"[map] reconstruction render failed: {e}")
        return _fallback(m, "scene.glb")


def _render_confidence(m: Map, conf_level):
    try:
        if m.is_merged:
            z = np.load(os.path.join(m.path, "merged_points.npz"))
            wp, conf = z["points"], z["conf"]
        else:
            preds = m.predictions
            wp, conf = preds.get("world_points"), preds.get("world_points_conf")
            if wp is None or conf is None:
                return None
        scene, _, _ = generate_confidence_point_cloud(
            wp, conf, conf_threshold=float(conf_level) / 100.0, max_points=500000, save_ply=False)
        path = _view_path(m, "confidence", conf_level)
        scene.export(path)
        return path
    except Exception as e:
        print(f"[map] confidence render failed: {e}")
        return None


# ----------------------------------------------------------------------- segment


def _fallback(m: Map, name):
    p = os.path.join(m.path, name)
    return p if os.path.exists(p) else None


def write_previews(m: Map) -> None:
    """The map's default scene.glb + confidence_map.glb/.ply, from its stored cloud."""
    z = np.load(os.path.join(m.path, "merged_points.npz"))
    pts, cols, conf = z["points"], z["colors"], z["conf"]
    geometry.pointcloud_scene(pts, cols, m.frames).export(os.path.join(m.path, "scene.glb"))
    try:
        scene, _, _ = generate_confidence_point_cloud(
            pts, conf, conf_threshold=0.0, max_points=500000, save_ply=True,
            ply_path=os.path.join(m.path, "confidence_map.ply"))
        scene.export(os.path.join(m.path, "confidence_map.glb"))
    except Exception as e:
        print(f"[map] confidence map export failed: {e}")
