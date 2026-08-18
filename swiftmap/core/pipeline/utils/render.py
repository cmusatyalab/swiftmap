# Copyright (C) 2024 Carnegie Mellon University

"""Render a map into viewable GLBs, written beside its data. Called through the session."""

import os

import numpy as np

from swiftmap.core.database.map import Map
from swiftmap.core.pipeline.utils.confidence import generate_confidence_point_cloud
from swiftmap.core.pipeline.utils import geometry
from swiftmap.core.database import cloud as arrays


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
        pts, cols, conf, _, frames = m.load(conf_thres=0.0)
        keep = arrays.confidence_mask(conf, conf_level)
        scene = geometry.pointcloud_scene(pts[keep], cols[keep], frames)
        path = _view_path(m, "reconstruction", conf_level)
        scene.export(path)
        return path
    except Exception as e:
        print(f"[map] reconstruction render failed: {e}")
        return None


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
