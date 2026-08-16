# Copyright (C) 2024 Carnegie Mellon University

"""Lift per-frame 2D masks into 3D, then export and cluster them.

The reconstruction's ``world_points`` is a per-pixel ``(S, H, W, 3)`` grid
pixel-aligned with the frames the segmenter ran on, so a ``(S, H, W)`` boolean
mask indexes 3D points directly — ``world_points[mask]``. No camera projection is
involved; that direct index *is* the 3D lift.

This module:
  * ``frame_images``      recover RGB uint8 frames from ``predictions["images"]``
  * ``masks_to_points``   lift masked world points (+ colors), confidence-cut
  * ``export_highlight_glb``  a scene GLB with the queried points colored red
  * ``cluster_objects``   v1 spatial clustering -> one centroid per object
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from swiftmap.core import constants
from swiftmap.core.primitives import geometry

RED = np.array([255, 0, 0], dtype=np.uint8)
# Cap on segmented points fed to clustering (subsampled if exceeded) to bound cost.
_MAX_CLUSTER_POINTS = 30_000


def frame_images(predictions: Dict[str, Any]) -> List[np.ndarray]:
    """RGB uint8 (H, W, 3) frames from ``predictions["images"]`` (NCHW or NHWC)."""
    images = np.asarray(predictions["images"])
    if images.ndim == 4 and images.shape[1] == 3:      # (S, 3, H, W)
        images = np.transpose(images, (0, 2, 3, 1))     # -> (S, H, W, 3)
    if images.dtype != np.uint8:
        images = (np.clip(images, 0.0, 1.0) * 255).astype(np.uint8)
    return [np.ascontiguousarray(images[i]) for i in range(images.shape[0])]


def _confidence_keep(predictions: Dict[str, Any], conf_thres: Optional[float]) -> np.ndarray:
    """Per-pixel (S,H,W) confidence keep-mask (percentile cut via geometry)."""
    conf = predictions.get("world_points_conf")
    if conf is None:
        return np.ones(np.asarray(predictions["world_points"]).shape[:-1], dtype=bool)
    conf = np.asarray(conf)
    return geometry.confidence_mask(conf, conf_thres).reshape(conf.shape)


def masks_to_points(predictions: Dict[str, Any], masks: np.ndarray,
                    conf_thres: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Select (points (M,3), colors (M,3)) for masked, finite, confident points.

    ``conf_thres`` (percentile, matching the Processing-Control slider) applies
    the same confidence cut the reconstruction viewer uses, so segmented points
    are consistent with what is shown.
    """
    wp = np.asarray(predictions["world_points"])        # (S, H, W, 3)
    valid = (masks.astype(bool)
             & np.isfinite(wp).all(-1)
             & _confidence_keep(predictions, conf_thres))
    pts = wp[valid]
    colors = geometry.flatten_colors(predictions["images"]).reshape(wp.shape[:-1] + (3,))
    cols = colors[valid]
    return pts, cols


def _extrinsics_4x4(predictions: Dict[str, Any]) -> Optional[np.ndarray]:
    if "extrinsic" not in predictions:
        return None
    ext = np.asarray(predictions["extrinsic"])          # (S, 3, 4)
    m = np.zeros((ext.shape[0], 4, 4))
    m[:, :3, :4] = ext
    m[:, 3, 3] = 1.0
    return m


def export_highlight_glb(predictions: Dict[str, Any], masks: np.ndarray,
                         query: str, target_dir: str,
                         conf_thres: Optional[float] = None) -> Optional[str]:
    """Write ``segmented_{query}.glb``: the scene cloud with masked points red.

    Applies the same confidence cut (``conf_thres``) as the reconstruction viewer
    to both the background and the highlighted points, and the same camera-based
    scene alignment, so it matches the left panel.
    """
    wp = np.asarray(predictions["world_points"])
    flat_pts = wp.reshape(-1, 3)
    flat_cols = geometry.flatten_colors(predictions["images"])   # (N, 3) uint8
    seg = masks.astype(bool).reshape(-1)
    keep = np.isfinite(flat_pts).all(-1) & _confidence_keep(predictions, conf_thres).reshape(-1)

    # Keep the full confidence-passing background (no decimation) so the cloud
    # matches the reconstruction viewer exactly — only the coloring differs.
    seg_idx = np.where(seg & keep)[0]
    bg_idx = np.where(~seg & keep)[0]
    if len(seg_idx) == 0:
        print(f"No 3D points matched query '{query}' — nothing to highlight.")

    keep = np.concatenate([bg_idx, seg_idx])
    pts = flat_pts[keep]
    cols = flat_cols[keep].copy()
    cols[len(bg_idx):] = RED                             # segmented points -> red

    scene = trimesh.Scene()
    scene.add_geometry(geometry.pointcloud(pts, cols))
    ext = _extrinsics_4x4(predictions)
    if ext is not None:
        scene = geometry.apply_scene_alignment(scene, ext)

    safe = "".join(c if c.isalnum() else "_" for c in query.strip()) or "query"
    path = os.path.join(target_dir, f"segmented_{safe}.glb")
    scene.export(path)
    print(f"Segmented highlight GLB saved: {path} ({len(seg_idx)} red / {len(pts)} pts)")
    return path if os.path.exists(path) else None


def cluster_objects(points: np.ndarray,
                    scene_diag: float,
                    eps_fraction: Optional[float] = None,
                    min_points: Optional[int] = None) -> List[Dict[str, Any]]:
    """v1 spatial clustering: single-linkage connected components at radius eps.

    Groups segmented 3D points into objects by spatial proximity (eps scaled to
    the scene) and returns one entry per object::

        {"centroid": (3,), "num_points": int, "radius": float}

    Note (v1 limitation): objects standing closer than eps merge into one
    cluster. Instance separation for close objects needs SAM 3 tracking (v2).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    eps_fraction = eps_fraction if eps_fraction is not None else constants.SEG_CLUSTER_EPS_FRACTION
    min_points = min_points if min_points is not None else constants.SEG_CLUSTER_MIN_POINTS

    pts = np.asarray(points, dtype=float)
    if len(pts) < max(min_points, 1):
        return []
    if len(pts) > _MAX_CLUSTER_POINTS:                   # deterministic subsample
        pts = pts[:: (len(pts) // _MAX_CLUSTER_POINTS) + 1]

    eps = max(scene_diag * eps_fraction, 1e-9)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(eps, output_type="ndarray")

    n = len(pts)
    if len(pairs) == 0:
        labels = np.arange(n)                            # every point isolated
    else:
        data = np.ones(len(pairs))
        graph = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)

    objects = []
    for lbl in np.unique(labels):
        cpts = pts[labels == lbl]
        if len(cpts) < min_points:
            continue
        centroid = cpts.mean(0)
        radius = float(np.linalg.norm(cpts - centroid, axis=1).max())
        objects.append({"centroid": centroid, "num_points": int(len(cpts)), "radius": radius})
    # Largest (most-supported) objects first.
    objects.sort(key=lambda o: o["num_points"], reverse=True)
    return objects
