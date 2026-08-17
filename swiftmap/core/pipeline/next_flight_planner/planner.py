# Copyright (C) 2024 Carnegie Mellon University
"""
Next Flight Navigation (NFN) — confidence-difference flight planner.

Idea: a VGGT reconstruction comes with a per-point confidence. Points that pass a
*loose* confidence cut (e.g. the 60th percentile) but fail a *strict* one (e.g. the
80th percentile) are the "marginally mapped" regions -- good enough to show up, but not
yet reliable. Those are exactly the places worth re-flying.

Pipeline:
  1. Keep the points whose confidence falls in the [P_low, P_high) band.
  2. Estimate the ground plane (PCA) and tile the band points into fixed-footprint cells
     on that plane. Each well-populated cell is one local cluster -- so a large weak area
     becomes several actionable viewpoints instead of one giant blob.
  3. Fit a surface normal per cluster (PCA), oriented toward the observed side.
  4. Place a suggested viewpoint at a fixed stand-off along the normal, looking back at
     the cluster center.

All distances are scale-relative (VGGT world coordinates are unitless), derived from the
scene's bounding-box diagonal, so the planner works regardless of the scene's scale.
"""

import json
import os

import numpy as np

from swiftmap.core.primitives import kml
from swiftmap.core.primitives.types import Reconstruction
from typing import Dict, Optional


class NextFlightPlanner:
    def __init__(self,
                 low_percentile: float = 60.0,
                 high_percentile: float = 80.0,
                 cell_fraction: float = 0.1,
                 standoff_fraction: float = 0.12,
                 tilt_deg: float = 35.0,
                 min_cluster_points: int = 30,
                 max_viewpoints: int = 20):
        """
        Args:
            low_percentile:  loose confidence cut (the "60%" map).
            high_percentile: strict confidence cut (the "80" map).
            cell_fraction:   side length of each planning cell, as a fraction of the scene
                             diagonal (the footprint a single suggested viewpoint covers).
            standoff_fraction: how far from the surface to place a viewpoint, as a
                             fraction of the scene diagonal.
            tilt_deg:        oblique angle of suggested viewpoints away from the surface
                             normal (0 = straight-down nadir). Each cluster gets a
                             different azimuth so the plan covers varied viewing angles.
            min_cluster_points: discard cells with fewer band points than this.
            max_viewpoints:  cap on the number of suggested viewpoints (weakest cells first).
        """
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.cell_fraction = cell_fraction
        self.standoff_fraction = standoff_fraction
        self.tilt_deg = tilt_deg
        self.min_cluster_points = min_cluster_points
        self.max_viewpoints = max_viewpoints

    def plan(self, predictions: Dict) -> Dict:
        recon = Reconstruction.wrap(predictions)
        wp = recon.world_points.reshape(-1, 3)
        conf = recon.world_points_conf.reshape(-1)
        valid = np.isfinite(wp).all(1) & np.isfinite(conf)
        wp, conf = wp[valid], conf[valid]
        if len(wp) == 0:
            return self._empty("No valid points")

        # Scene scale (VGGT world coords are unitless -> everything is scale-relative)
        diag = float(np.linalg.norm(wp.max(0) - wp.min(0))) + 1e-9

        # Confidence band between the loose and strict thresholds = "to-improve" points
        p_low = float(np.percentile(conf, self.low_percentile))
        p_high = float(np.percentile(conf, self.high_percentile))
        if p_low >= p_high:
            # Confidence is (near-)uniform across all points, so there is no
            # low-but-not-lowest band to target. VGGT was weakly confident everywhere
            # -> refuse to plan and let the UI explain it.
            return self._empty(
                "VGGT confidence is saturated (near-uniform across all points), so there "
                "are no low-confidence regions to target. The reconstruction is weakly "
                "constrained — capture more overlapping views and try again.",
                np.empty((0, 3)), p_low, p_high, diag)
        band = (conf >= p_low) & (conf < p_high)
        enhance_pts = wp[band]
        if len(enhance_pts) < self.min_cluster_points:
            return self._empty("Not enough band points to plan", enhance_pts, p_low, p_high, diag)

        # Ground plane (PCA): up = least-variance axis; (basis_u, basis_v) span the plane
        scene_centroid = wp.mean(0)
        _, eigvecs = np.linalg.eigh(np.cov((wp - scene_centroid).T))
        up = eigvecs[:, 0]
        basis_v, basis_u = eigvecs[:, 1], eigvecs[:, 2]
        cams = self._camera_positions(recon)
        if cams is not None and len(cams) and np.dot(up, cams.mean(0) - scene_centroid) < 0:
            up = -up

        # Tile the band points into fixed-footprint cells on the ground plane
        rel = enhance_pts - scene_centroid
        u = rel @ basis_u
        v = rel @ basis_v
        cell = max(diag * self.cell_fraction, 1e-9)
        cells = np.stack([np.floor(u / cell), np.floor(v / cell)], axis=1).astype(int)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)

        standoff = diag * self.standoff_fraction
        tilt = np.deg2rad(self.tilt_deg)
        golden = np.pi * (3.0 - np.sqrt(5.0))  # spreads azimuths evenly across clusters
        clusters, viewpoints = [], []
        for gi in range(inverse.max() + 1):
            cpts = enhance_pts[inverse == gi]
            if len(cpts) < self.min_cluster_points:
                continue

            centroid = cpts.mean(0)
            normal = self._principal_normal(cpts - centroid)
            if np.dot(normal, up) < 0:
                normal = -normal
            radius = float(np.linalg.norm(cpts - centroid, axis=1).max())

            # Oblique stand-off: tilt away from the normal at a per-cluster azimuth so the
            # suggested views cover varied directions (not all straight-down nadir).
            k = len(clusters)
            phi = k * golden
            horiz = np.cos(phi) * basis_u + np.sin(phi) * basis_v
            view_dir = np.cos(tilt) * up + np.sin(tilt) * horiz
            view_dir /= (np.linalg.norm(view_dir) + 1e-9)
            position = centroid + view_dir * standoff
            rotation = self._look_at(position, centroid, up)

            clusters.append({
                "id": k, "centroid": centroid, "normal": normal,
                "num_points": int(len(cpts)), "radius": radius,
            })
            viewpoints.append({
                "cluster_id": k,
                "camera_position": position,
                "camera_rotation": rotation,
                "target": centroid,
                "score": float(len(cpts)),
            })

        # Prioritize the cells with the most points to improve
        order = np.argsort([-v["score"] for v in viewpoints])[:self.max_viewpoints]
        viewpoints = [viewpoints[i] for i in order]

        return {
            "enhance_points": enhance_pts,
            "clusters": clusters,
            "viewpoints": viewpoints,
            "num_viewpoints": len(viewpoints),
            "thresholds": {
                "low_percentile": self.low_percentile, "high_percentile": self.high_percentile,
                "p_low": p_low, "p_high": p_high,
            },
            "statistics": {
                "num_enhance_points": int(len(enhance_pts)),
                "num_clusters": len(clusters),
                "scene_diagonal": diag, "cell_size": cell, "standoff": standoff,
            },
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _principal_normal(centered: np.ndarray) -> np.ndarray:
        """Surface normal = direction of least variance (smallest PCA eigenvector)."""
        if len(centered) < 3:
            return np.array([0.0, 0.0, 1.0])
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        return eigvecs[:, 0]

    @staticmethod
    def _camera_positions(predictions: Dict) -> Optional[np.ndarray]:
        ext = predictions.get("extrinsic")
        if ext is None:
            return None
        ext = np.asarray(ext)
        return np.array([(-e[:3, :3].T @ e[:3, 3]) for e in ext])

    @staticmethod
    def _look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Camera-to-world rotation (OpenCV axes: +Z forward) looking from position at target."""
        z = target - position
        z = z / (np.linalg.norm(z) + 1e-9)
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-9)
        y = np.cross(z, x)
        return np.column_stack([x, y, z]).astype(np.float32)

    def _empty(self, msg, enhance_pts=None, p_low=0.0, p_high=0.0, diag=0.0) -> Dict:
        print(f"[NFN] {msg}")
        return {
            "enhance_points": enhance_pts if enhance_pts is not None else np.empty((0, 3)),
            "clusters": [], "viewpoints": [], "num_viewpoints": 0,
            "thresholds": {
                "low_percentile": self.low_percentile, "high_percentile": self.high_percentile,
                "p_low": p_low, "p_high": p_high,
            },
            "statistics": {"message": msg, "scene_diagonal": diag},
        }


def write_plan(plan, gps_transform, target_dir, segmented=None, seg_query=None) -> str:
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
