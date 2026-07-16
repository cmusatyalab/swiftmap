# Copyright (C) 2024 Carnegie Mellon University

"""Area tags + the decoupled, on-demand segmentation service.

Each time the mission loop hits the keyframe cap it writes a run into an
*area* directory named by a tag (``{site}_{YYYYMMDD_HHMMSS}``) and drops an
``area.json`` describing it (GPS center + geohash when aligned). Segmentation is
**not** part of that loop — it is a separate, request-driven step: given an area
tag and a text query, ``segment_area`` reloads that area's ``predictions.npz`` /
``transform.json`` from disk and runs the segmenter, independent of the live
session. This lets a client segment any past area at any time.
"""

import glob
import json
import os
from datetime import datetime

import numpy as np

_GEOHASH_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int = 9) -> str:
    """Standard geohash of a lat/lon — a compact area id (nearby areas share a prefix)."""
    lat_int, lon_int = [-90.0, 90.0], [-180.0, 180.0]
    out, bits, bit, ch, even = [], [16, 8, 4, 2, 1], 0, 0, True
    while len(out) < precision:
        if even:
            mid = sum(lon_int) / 2
            if lon > mid:
                ch |= bits[bit]; lon_int[0] = mid
            else:
                lon_int[1] = mid
        else:
            mid = sum(lat_int) / 2
            if lat > mid:
                ch |= bits[bit]; lat_int[0] = mid
            else:
                lat_int[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_GEOHASH_B32[ch]); bit, ch = 0, 0
    return "".join(out)


def make_area_tag(site: str, when: datetime) -> str:
    """Unique, sortable, human-referenceable tag: ``{site}_{YYYYMMDD_HHMMSS}``."""
    safe = "".join(c if c.isalnum() else "-" for c in (site or "area")).strip("-") or "area"
    return f"{safe}_{when.strftime('%Y%m%d_%H%M%S')}"


def write_area_metadata(area_dir, tag, site, created, num_keyframes, preds, gps_transform):
    """Write ``area.json`` (identity + GPS center/geohash when aligned)."""
    meta = {
        "area_tag": tag, "site": site,
        "created": created.isoformat(timespec="seconds") if isinstance(created, datetime) else created,
        "num_keyframes": int(num_keyframes),
        "gps_aligned": gps_transform is not None,
    }
    try:
        cams = np.asarray(preds.get("camera_positions")) if preds else None
        if gps_transform is not None and cams is not None and len(cams):
            lla = np.asarray(gps_transform.to_lla(cams.mean(0)), dtype=float).tolist()
            meta["center_gps"] = lla
            meta["geohash"] = geohash_encode(lla[0], lla[1])
    except Exception as e:
        print(f"[areas] metadata GPS center failed: {e}")
    with open(os.path.join(area_dir, "area.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def export_model_input_images(preds, area_dir, subdir="model_input"):
    """Save the (reduced-resolution) frames actually fed to the model."""
    import cv2
    from swiftmap.core.segmentation import lift
    dst = os.path.join(area_dir, subdir)
    os.makedirs(dst, exist_ok=True)
    for i, frame in enumerate(lift.frame_images(preds)):
        cv2.imwrite(os.path.join(dst, f"frame_{i:06d}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return dst


def list_areas(output_dir):
    """All areas under ``output_dir`` (dirs with an ``area.json``), newest first."""
    metas = []
    for p in glob.glob(os.path.join(output_dir, "*", "area.json")):
        try:
            with open(p) as f:
                metas.append(json.load(f))
        except Exception:
            pass
    metas.sort(key=lambda m: m.get("created", ""), reverse=True)
    return metas


def resolve_area_dir(output_dir, tag):
    """Directory for ``tag`` if it holds an ``area.json``, else None."""
    d = os.path.join(output_dir, tag)
    return d if os.path.isfile(os.path.join(d, "area.json")) else None


def load_predictions(area_dir):
    """Load an area's saved prediction arrays (everything but the metadata blob)."""
    npz = np.load(os.path.join(area_dir, "predictions.npz"), allow_pickle=True)
    return {k: npz[k] for k in npz.files if k != "metadata"}


def load_gps_transform(area_dir):
    """Rebuild the area's GpsTransform from ``transform.json``, or None."""
    p = os.path.join(area_dir, "transform.json")
    if not os.path.isfile(p):
        return None
    from swiftmap.core.geo_transform.geo import GpsTransform
    with open(p) as f:
        return GpsTransform(json.load(f))


def render_area(output_dir, tag, conf_level=60.0):
    """Reconstruction + confidence GLBs, both regenerated at ``conf_level`` (percentile)."""
    area_dir = resolve_area_dir(output_dir, tag)
    if area_dir is None:
        return {"error": f"Unknown area '{tag}'."}
    return {"success": True, "area_tag": tag,
            "scene_glb": _render_reconstruction(area_dir, conf_level),
            "confidence_glb": _render_confidence(area_dir, conf_level)}


def _render_reconstruction(area_dir, conf_level):
    try:
        preds = load_predictions(area_dir)
        from swiftmap.core.mapper.scene_export import predictions_to_glb
        has_images = bool(glob.glob(os.path.join(area_dir, "images", "*")))
        scene = predictions_to_glb(
            predictions=preds, conf_thres=float(conf_level), filter_by_frames="all",
            mask_black_bg=False, mask_white_bg=False, show_cam=True,
            mask_sky=has_images, mask_dynamic=False, target_dir=area_dir)
        path = os.path.join(area_dir, f"reconstruction_view_c{int(round(float(conf_level)))}.glb")
        scene.export(path)
        return path
    except Exception as e:
        print(f"[areas] reconstruction render failed: {e}")
        fallback = os.path.join(area_dir, "scene.glb")
        return fallback if os.path.exists(fallback) else None


def _render_confidence(area_dir, conf_level):
    try:
        preds = load_predictions(area_dir)
        wp, conf = preds.get("world_points"), preds.get("world_points_conf")
        if wp is None or conf is None:
            return None
        from swiftmap.core.mapper.confidence_mapping import generate_confidence_point_cloud
        scene, _, _ = generate_confidence_point_cloud(
            wp, conf, conf_threshold=float(conf_level) / 100.0, save_ply=False)
        path = os.path.join(area_dir, f"confidence_view_c{int(round(float(conf_level)))}.glb")
        scene.export(path)
        return path
    except Exception as e:
        print(f"[areas] confidence render failed: {e}")
        return None


def segment_area(output_dir, tag, query, segmenter, conf_threshold=60.0):
    """Segment a stored area on demand: reload it from disk, run the segmenter,
    lift masks to 3D, cluster objects, GPS-tag them, and export into the area dir.

    Fully decoupled from the live mission loop. Returns a result dict.
    """
    from swiftmap.core.segmentation import lift

    area_dir = resolve_area_dir(output_dir, tag)
    if area_dir is None:
        return {"error": f"Unknown area '{tag}'."}
    query = (query or "").strip()
    if not query:
        return {"error": "Enter a segmentation query (e.g. 'person')."}

    preds = load_predictions(area_dir)
    if "world_points" not in preds or "images" not in preds:
        return {"error": f"Area '{tag}' has no reconstruction to segment."}

    images = lift.frame_images(preds)
    masks = segmenter.segment(images, query)
    if masks is None:
        return {"error": "Segmentation model failed to initialize."}

    glb = lift.export_highlight_glb(preds, masks, query, area_dir, conf_thres=conf_threshold)
    pts, _ = lift.masks_to_points(preds, masks, conf_thres=conf_threshold)

    wp = np.asarray(preds["world_points"]).reshape(-1, 3)
    wp = wp[np.isfinite(wp).all(1)]
    diag = float(np.linalg.norm(wp.max(0) - wp.min(0))) if len(wp) else 1.0
    objects = lift.cluster_objects(pts, diag)

    gt = load_gps_transform(area_dir)
    items = []
    for i, o in enumerate(objects):
        item = {"id": i, "position": np.asarray(o["centroid"], float).tolist(),
                "num_points": int(o["num_points"]), "radius": float(o["radius"])}
        if gt is not None:
            item["position_gps"] = np.asarray(gt.to_lla(o["centroid"]), float).tolist()
        items.append(item)

    _write_segmented(area_dir, tag, query, conf_threshold, gt is not None, items)
    print(f"[areas] segmented '{query}' on {tag}: {len(pts)} pts -> {len(items)} object(s)")
    return {"success": True, "area_tag": tag, "query": query, "glb_path": glb,
            "conf_threshold": float(conf_threshold), "num_points": int(len(pts)),
            "num_objects": len(items), "gps_aligned": gt is not None, "objects": items}


def _write_segmented(area_dir, tag, query, conf, gps_aligned, items):
    safe = "".join(c if c.isalnum() else "_" for c in query.strip()) or "query"
    out = {"area_tag": tag, "query": query, "conf_threshold": float(conf),
           "gps_aligned": gps_aligned, "num_objects": len(items), "objects": items}
    with open(os.path.join(area_dir, f"segmented_{safe}.json"), "w") as f:
        json.dump(out, f, indent=2)
    gps_items = [it for it in items if "position_gps" in it]
    if gps_items:
        from swiftmap.core.nfn import kml
        kml.write_kml(gps_items, os.path.join(area_dir, f"segmented_{safe}.kml"),
                      gps_key="position_gps", doc_name=f"{tag}: {query}")
