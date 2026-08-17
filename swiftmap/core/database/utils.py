# Copyright (C) 2024 Carnegie Mellon University

"""Derived artifacts written into a map dir.

A ``Map`` owns its data (cloud, transform, poses); everything here *derives* files from
it and drops them beside that data: rendered GLB views, segmentation results, and the
NFN plan. Kept out of ``Map`` so the map stays a store record.
"""

import glob
import json
import os

import numpy as np

from swiftmap.core.database.map import Map
from swiftmap.core.pipeline.next_flight_planner import kml
from swiftmap.core.primitives import geometry


# ------------------------------------------------------------------------ render
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
    if m.is_merged:
        return _render_merged_scene(m, conf_level)
    try:
        from swiftmap.core.pipeline.reconstructor.scene_export import predictions_to_glb
        preds = predictions(m)
        has_images = bool(glob.glob(os.path.join(m.path, "images", "*")))
        scene = predictions_to_glb(
            predictions=preds, conf_thres=float(conf_level), filter_by_frames="all",
            mask_black_bg=False, mask_white_bg=False, show_cam=True,
            mask_sky=has_images, mask_dynamic=False, target_dir=m.path)
        path = _view_path(m, "reconstruction", conf_level)
        scene.export(path)
        return path
    except Exception as e:
        print(f"[map] reconstruction render failed: {e}")
        return _fallback(m, "scene.glb")


def _render_merged_scene(m: Map, conf_level):
    try:
        z = np.load(os.path.join(m.path, "merged_points.npz"))
        keep = geometry.confidence_mask(z["conf"], conf_level)
        scene = geometry.pointcloud_scene(z["points"][keep], z["colors"][keep], m.frames)
        path = _view_path(m, "reconstruction", conf_level)
        scene.export(path)
        return path
    except Exception as e:
        print(f"[map] merged reconstruction render failed: {e}")
        return _fallback(m, "scene.glb")


def _render_confidence(m: Map, conf_level):
    try:
        from swiftmap.core.pipeline.reconstructor.confidence_mapping import generate_confidence_point_cloud
        if m.is_merged:
            z = np.load(os.path.join(m.path, "merged_points.npz"))
            wp, conf = z["points"], z["conf"]
        else:
            preds = predictions(m)
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
def segment(m: Map, query: str, segmenter, conf_threshold: float = 60.0) -> dict:
    """Segment ``query`` on ``m``. A merged map has no per-frame images, so it returns
    its inherited segmentation instead of running a new query."""
    if not m.exists():
        return {"error": f"Unknown map '{m.tag}'."}
    if m.is_merged:
        return _inherited(m, query)

    from swiftmap.core.pipeline.segmentor import lift
    query = (query or "").strip()
    if not query:
        return {"error": "Enter a segmentation query (e.g. 'person')."}
    preds = predictions(m)
    if "world_points" not in preds or "images" not in preds:
        return {"error": f"Map '{m.tag}' has no reconstruction to segment."}

    masks = segmenter.segment(lift.frame_images(preds), query)
    if masks is None:
        return {"error": "Segmentation model failed to initialize."}
    glb = lift.export_highlight_glb(preds, masks, query, m.path, conf_thres=conf_threshold)
    pts, _ = lift.masks_to_points(preds, masks, conf_thres=conf_threshold)

    wp = np.asarray(preds["world_points"]).reshape(-1, 3)
    wp = wp[np.isfinite(wp).all(1)]
    diag = float(np.linalg.norm(wp.max(0) - wp.min(0))) if len(wp) else 1.0

    gt = m.transform
    items = []
    for i, ob in enumerate(lift.cluster_objects(pts, diag)):
        item = {"id": i, "position": np.asarray(ob["centroid"], float).tolist(),
                "num_points": int(ob["num_points"]), "radius": float(ob["radius"])}
        if gt is not None:
            item["position_gps"] = np.asarray(gt.to_lla(ob["centroid"]), float).tolist()
        items.append(item)

    _write_segmented(m, query, conf_threshold, gt is not None, items)
    print(f"[map] segmented '{query}' on {m.tag}: {len(pts)} pts -> {len(items)} object(s)")
    return {"success": True, "map_tag": m.tag, "query": query, "glb_path": glb,
            "conf_threshold": float(conf_threshold), "num_points": int(len(pts)),
            "num_objects": len(items), "gps_aligned": gt is not None, "objects": items}


def _inherited(m: Map, query):
    """The segmentation a merged map carries, matching ``query`` when possible."""
    safe = _safe(query)
    glbs = sorted(glob.glob(os.path.join(m.path, "segmented_*.glb")))
    if not glbs:
        return {"error": f"'{m.tag}' is a merged map (no per-frame images to segment) and "
                         "has no inherited segmentation."}
    match = [g for g in glbs if safe and os.path.basename(g) == f"segmented_{safe}.glb"]
    glb = match[0] if match else glbs[0]
    base = os.path.basename(glb)[len("segmented_"):-len(".glb")]
    meta = {}
    jp = os.path.join(m.path, f"segmented_{base}.json")
    if os.path.isfile(jp):
        with open(jp) as f:
            meta = json.load(f)
    objects = meta.get("objects", [])
    return {"success": True, "map_tag": m.tag, "query": meta.get("query", base),
            "glb_path": glb, "inherited": True,
            "conf_threshold": float(meta.get("conf_threshold", 0.0)), "num_points": 0,
            "num_objects": meta.get("num_objects", len(objects)),
            "gps_aligned": meta.get("gps_aligned", True), "objects": objects}


def _write_segmented(m: Map, query, conf, gps_aligned, items):
    safe = _safe(query) or "query"
    m._dump(f"segmented_{safe}.json",
            {"map_tag": m.tag, "query": query, "conf_threshold": float(conf),
             "gps_aligned": gps_aligned, "num_objects": len(items), "objects": items})
    gps_items = [it for it in items if "position_gps" in it]
    if gps_items:
        kml.write_kml(gps_items, os.path.join(m.path, f"segmented_{safe}.kml"),
                      gps_key="position_gps", doc_name=f"{m.tag}: {query}")


# --------------------------------------------------------------------- NFN plan
def write_nfn_plan(plan, gps_transform, target_dir, segmented=None, seg_query=None) -> str:
    """Write next_flight_viewpoints.json (+ transform.json + KML when GPS-aligned)."""
    viewpoints = _viewpoints_payload(plan)
    out = {"num_viewpoints": len(viewpoints), "thresholds": plan.get("thresholds", {}),
           "gps_aligned": gps_transform is not None, "viewpoints": viewpoints}
    if segmented:
        out["segmented_objects"] = {"query": seg_query, "num_objects": len(segmented),
                                    "objects": segmented}
    path = _dump(target_dir, "next_flight_viewpoints.json", out)
    if gps_transform is not None:
        _dump(target_dir, "transform.json", gps_transform.cfg)
        kml.write_kml(viewpoints, os.path.join(target_dir, "next_flight_viewpoints.kml"),
                      gps_key="target_gps", doc_name="nfn_pts")
        kml.write_polygon_kml(viewpoints, os.path.join(target_dir, "next_flight_area.kml"),
                              gps_key="target_gps", doc_name="nfn_area")
    return path


def write_segmented_objects(items, seg_query, conf_threshold, gps_transform, target_dir) -> str:
    """Write segmented_objects.json (+ KML when any object has GPS)."""
    path = _dump(target_dir, "segmented_objects.json",
                 {"query": seg_query, "conf_threshold": conf_threshold,
                  "gps_aligned": gps_transform is not None,
                  "num_objects": len(items), "objects": items})
    gps_items = [s for s in items if "position_gps" in s]
    if gps_items:
        kml.write_kml(gps_items, os.path.join(target_dir, "segmented_objects.kml"),
                      gps_key="position_gps", doc_name=f"SwiftMap Segmented: {seg_query}")
    return path


def _viewpoints_payload(plan) -> list:
    """Per-viewpoint records from an NFN plan (GPS keys copied when tagged)."""
    out = []
    for i, vp in enumerate(plan.get("viewpoints", [])):
        item = {"id": i, "cluster_id": int(vp.get("cluster_id", -1)),
                "position": np.asarray(vp["camera_position"], float).tolist(),
                "look_dir": np.asarray(vp["camera_rotation"], float)[:, 2].tolist(),
                "target": np.asarray(vp["target"], float).tolist(),
                "score": float(vp.get("score", 0.0))}
        for k in ("camera_position_gps", "target_gps"):
            if k in vp:
                item["position_gps" if k == "camera_position_gps" else k] = vp[k]
        out.append(item)
    return out


# ----------------------------------------------------------------------- shared
def predictions(m: Map) -> dict:
    """The raw prediction arrays of a non-merged map."""
    npz = np.load(os.path.join(m.path, "predictions.npz"), allow_pickle=True)
    return {k: npz[k] for k in npz.files if k != "metadata"}


def _fallback(m: Map, name):
    p = os.path.join(m.path, name)
    return p if os.path.exists(p) else None


def _safe(text) -> str:
    return "".join(c if c.isalnum() else "_" for c in (text or "").strip())


def _dump(target_dir, name, obj) -> str:
    path = os.path.join(target_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
