# Copyright (C) 2024 Carnegie Mellon University

"""The ``Map``: one stored map directory (tag + path) in the map database, holding a
raw ``predictions.npz`` or a merged ``merged_points.npz`` plus transform/camera/map json
and rendered artifacts. Wraps ``load`` (-> MapData), ``render``, ``segment``, ``write``,
and segmentation inheritance. Geometry/merge live in ``geometry``/``types``."""

import glob
import json
import os
import shutil
from datetime import datetime
from typing import List, Optional

import numpy as np

from swiftmap.core.primitives import geometry
from swiftmap.core.pipeline.next_flight_planner import kml
from swiftmap.core.primitives.types import Georeference, MapData, Reconstruction

# write()/render()/segment() import their pipeline modules (confidence_mapping,
# scene_export, lift) on demand: those are the heavy render/segmentation deps, needed
# only on those paths. Discovery (list/get/metadata) and load() pull none of them.

_GEOHASH_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_META_NAMES = ("map.json", "area.json")  # new name, then legacy


def _meta_path(dir_path: str) -> Optional[str]:
    """A map dir's metadata file: new ``map.json`` or legacy ``area.json``, else None."""
    for name in _META_NAMES:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            return p
    return None


def geohash_encode(lat: float, lon: float, precision: int = 9) -> str:
    """Standard geohash of a lat/lon (nearby maps share a prefix)."""
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


class Map:
    """An on-disk map directory (tag + path) with load/render/segment/merge I/O."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.tag = os.path.basename(self.path.rstrip("/"))
        self._meta = None

    def __repr__(self):
        return f"Map({self.tag!r}{', merged' if self.is_merged else ''})"

    # --------------------------------------------------------------- discovery
    @classmethod
    def list(cls, root: str) -> List["Map"]:
        """All maps under ``root`` (dirs holding an ``map.json``), newest first."""
        found = {}
        for name in _META_NAMES:
            for p in glob.glob(os.path.join(root, "*", name)):
                d = os.path.dirname(p)
                if d in found:
                    continue
                try:
                    with open(p) as f:
                        found[d] = json.load(f).get("created", "")
                except Exception:
                    pass
        return [cls(d) for d, _ in sorted(found.items(), key=lambda t: t[1], reverse=True)]

    @classmethod
    def get(cls, root: str, tag: str) -> Optional["Map"]:
        """The map named ``tag`` under ``root`` if it exists, else None."""
        a = cls(os.path.join(root, tag))
        return a if a.exists() else None

    @staticmethod
    def tag_for(site: str, when: datetime) -> str:
        """Unique, sortable tag ``{site}_{YYYYMMDD_HHMMSS}``."""
        safe = "".join(c if c.isalnum() else "-" for c in (site or "map")).strip("-") or "map"
        return f"{safe}_{when.strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------- state
    def exists(self) -> bool:
        return _meta_path(self.path) is not None

    @property
    def is_merged(self) -> bool:
        return os.path.isfile(os.path.join(self.path, "merged_points.npz"))

    @property
    def metadata(self) -> dict:
        if self._meta is None:
            p = _meta_path(self.path)
            self._meta = json.load(open(p)) if p else {}
        return self._meta

    @property
    def transform(self) -> Optional[Georeference]:
        p = os.path.join(self.path, "transform.json")
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return Georeference(json.load(f))

    @property
    def frames(self) -> list:
        p = os.path.join(self.path, "camera_poses.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f).get("frames", [])
        return []

    # ---------------------------------------------------------------- geometry
    def load(self, conf_thres: float = 50.0) -> MapData:
        """Load this map's geometry into a MapData (merged flat, or raw predictions).

        A merged map's cloud is already confidence-filtered and loaded verbatim; a raw
        ``predictions.npz`` gets the ``conf_thres`` percentile cut.
        """
        gt = self.transform
        if gt is None:
            raise FileNotFoundError(f"{self.tag} has no transform.json (not GPS-aligned).")
        frames = self.frames

        flat = os.path.join(self.path, "merged_points.npz")
        if os.path.isfile(flat):
            z = np.load(flat)
            return MapData(z["points"], z["colors"], z["conf"], gt, frames)

        npz = np.load(os.path.join(self.path, "predictions.npz"), allow_pickle=True)
        recon = Reconstruction({k: npz[k] for k in npz.files if k != "metadata"})
        return recon.to_mapdata(gt, frames=frames, conf_percentile=conf_thres)

    # ------------------------------------------------------------- persistence
    @classmethod
    def write(cls, mapdata: MapData, root: str, site: str, source_maps,
              tag: str = None, created: datetime = None, inherit_from: "Map" = None,
              voxel_size: float = 0.1) -> "Map":
        """Persist a merged ``MapData`` as a new map directory and return it.

        Writes merged_points.npz (source of truth for re-merge), transform.json,
        camera_poses.json, scene.glb (+ camera frustums), confidence_map.glb/.ply, and
        map.json; inherits ``inherit_from``'s segmentation artifacts if given.
        """
        from swiftmap.core.pipeline.reconstructor.confidence_mapping import generate_confidence_point_cloud

        created = created or datetime.now()
        tag = tag or cls.tag_for(site, created)
        out = os.path.join(root, tag)
        os.makedirs(out, exist_ok=True)

        pts = np.asarray(mapdata.points)
        cols = np.asarray(mapdata.colors, np.uint8)
        conf = np.asarray(mapdata.conf, dtype=float)
        frames = mapdata.frames
        o = mapdata.origin

        np.savez(os.path.join(out, "merged_points.npz"), points=pts, colors=cols, conf=conf)

        tf = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
              "lat0": o[0], "lon0": o[1], "alt0": o[2], "mode": "merged",
              "num_points": int(len(pts)), "source_maps": list(source_maps)}
        with open(os.path.join(out, "transform.json"), "w") as f:
            json.dump(tf, f, indent=2)
        with open(os.path.join(out, "camera_poses.json"), "w") as f:
            json.dump({"metadata": {"description": "Merged camera poses (SwiftMap GPS merge)",
                                    "frame": "ENU meters about origin (lat0, lon0, alt0)",
                                    "source_maps": list(source_maps), "num_keyframes": len(frames)},
                       "frames": frames}, f, indent=2)

        geometry.pointcloud_scene(pts, cols, frames).export(os.path.join(out, "scene.glb"))

        try:
            cscene, _, _ = generate_confidence_point_cloud(
                pts, conf, conf_threshold=0.0, max_points=500000, save_ply=True,
                ply_path=os.path.join(out, "confidence_map.ply"))
            cscene.export(os.path.join(out, "confidence_map.glb"))
        except Exception as e:
            print(f"[map] confidence map export failed: {e}")

        center = Georeference(tf).to_lla(pts.mean(axis=0)) if len(pts) else np.array(o)
        meta = {"map_tag": tag, "site": site, "created": created.isoformat(timespec="seconds"),
                "num_keyframes": len(frames), "num_points": int(len(pts)), "gps_aligned": True,
                "center_gps": [float(center[0]), float(center[1]), float(center[2])],
                "geohash": geohash_encode(float(center[0]), float(center[1])),
                "source_maps": list(source_maps)}
        with open(os.path.join(out, "map.json"), "w") as f:
            json.dump(meta, f, indent=2)

        mp = cls(out)
        inherited = mp.inherit_segmentation_from(inherit_from) if inherit_from else 0
        print(f"[map] wrote '{tag}': {len(pts):,} pts, {len(frames)} cameras"
              + (f", inherited {inherited} segmentation file(s)" if inherited else ""))
        return mp

    def merge(self, other: "Map", *, conf_thres: float = 50.0, voxel_size: float = 0.1,
              site: str = None, tag: str = None, created: datetime = None,
              delete_sources: bool = True) -> "Map":
        """Grow this map by merging ``other`` into it, and return the new merged map.

        The clouds are GPS co-registered about *this* map's origin (so coordinates never
        drift), near-duplicates collapsed by ``voxel_size``; this map's segmentation is
        inherited. Both source dirs are deleted by default.
        """
        merged = MapData.merge([self.load(conf_thres), other.load(conf_thres)],
                               voxel_size=voxel_size)  # origin defaults to this map's
        new = Map.write(merged, os.path.dirname(self.path),
                        site or self.metadata.get("site") or "map",
                        [self.tag, other.tag], tag=tag, created=created,
                        inherit_from=self, voxel_size=voxel_size)
        if delete_sources:
            for src in (self, other):
                if src.path != new.path:
                    src.delete()
        return new

    def inherit_segmentation_from(self, other: "Map") -> int:
        """Copy segmentation artifacts (segmented_*) from ``other`` into this map."""
        n = 0
        for p in glob.glob(os.path.join(other.path, "segmented_*")):
            shutil.copy2(p, os.path.join(self.path, os.path.basename(p)))
            n += 1
        return n

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

    # ----------------------------------------------------------------- render
    def render(self, conf_level: float) -> dict:
        """Reconstruction + confidence GLBs, both regenerated at ``conf_level`` (percentile)."""
        if not self.exists():
            return {"error": f"Unknown map '{self.tag}'."}
        return {"success": True, "map_tag": self.tag,
                "scene_glb": self._render_scene(conf_level),
                "confidence_glb": self._render_confidence(conf_level)}

    def _render_scene(self, conf_level):
        if self.is_merged:
            return self._render_merged_scene(conf_level)
        try:
            from swiftmap.core.pipeline.reconstructor.scene_export import predictions_to_glb
            preds = self._predictions()
            has_images = bool(glob.glob(os.path.join(self.path, "images", "*")))
            scene = predictions_to_glb(
                predictions=preds, conf_thres=float(conf_level), filter_by_frames="all",
                mask_black_bg=False, mask_white_bg=False, show_cam=True,
                mask_sky=has_images, mask_dynamic=False, target_dir=self.path)
            path = os.path.join(self.path, f"reconstruction_view_c{int(round(float(conf_level)))}.glb")
            scene.export(path)
            return path
        except Exception as e:
            print(f"[map] reconstruction render failed: {e}")
            return self._fallback("scene.glb")

    def _render_merged_scene(self, conf_level):
        try:
            z = np.load(os.path.join(self.path, "merged_points.npz"))
            keep = geometry.confidence_mask(z["conf"], conf_level)
            scene = geometry.pointcloud_scene(z["points"][keep], z["colors"][keep], self.frames)
            path = os.path.join(self.path, f"reconstruction_view_c{int(round(float(conf_level)))}.glb")
            scene.export(path)
            return path
        except Exception as e:
            print(f"[map] merged reconstruction render failed: {e}")
            return self._fallback("scene.glb")

    def _render_confidence(self, conf_level):
        try:
            from swiftmap.core.pipeline.reconstructor.confidence_mapping import generate_confidence_point_cloud
            if self.is_merged:
                z = np.load(os.path.join(self.path, "merged_points.npz"))
                wp, conf = z["points"], z["conf"]
            else:
                preds = self._predictions()
                wp, conf = preds.get("world_points"), preds.get("world_points_conf")
                if wp is None or conf is None:
                    return None
            scene, _, _ = generate_confidence_point_cloud(
                wp, conf, conf_threshold=float(conf_level) / 100.0, max_points=500000, save_ply=False)
            path = os.path.join(self.path, f"confidence_view_c{int(round(float(conf_level)))}.glb")
            scene.export(path)
            return path
        except Exception as e:
            print(f"[map] confidence render failed: {e}")
            return None

    # -------------------------------------------------------------- segment
    def segment(self, query: str, segmenter, conf_threshold: float = 60.0) -> dict:
        """Segment ``query`` on this map. A merged map has no per-frame images, so it
        returns its inherited segmentation instead of running a new query."""
        if not self.exists():
            return {"error": f"Unknown map '{self.tag}'."}
        if self.is_merged:
            return self._inherited_segmentation(query)

        from swiftmap.core.pipeline.segmentor import lift
        query = (query or "").strip()
        if not query:
            return {"error": "Enter a segmentation query (e.g. 'person')."}
        preds = self._predictions()
        if "world_points" not in preds or "images" not in preds:
            return {"error": f"Map '{self.tag}' has no reconstruction to segment."}

        masks = segmenter.segment(lift.frame_images(preds), query)
        if masks is None:
            return {"error": "Segmentation model failed to initialize."}
        glb = lift.export_highlight_glb(preds, masks, query, self.path, conf_thres=conf_threshold)
        pts, _ = lift.masks_to_points(preds, masks, conf_thres=conf_threshold)

        wp = np.asarray(preds["world_points"]).reshape(-1, 3)
        wp = wp[np.isfinite(wp).all(1)]
        diag = float(np.linalg.norm(wp.max(0) - wp.min(0))) if len(wp) else 1.0
        objects = lift.cluster_objects(pts, diag)

        gt = self.transform
        items = []
        for i, ob in enumerate(objects):
            item = {"id": i, "position": np.asarray(ob["centroid"], float).tolist(),
                    "num_points": int(ob["num_points"]), "radius": float(ob["radius"])}
            if gt is not None:
                item["position_gps"] = np.asarray(gt.to_lla(ob["centroid"]), float).tolist()
            items.append(item)

        self._write_segmented(query, conf_threshold, gt is not None, items)
        print(f"[map] segmented '{query}' on {self.tag}: {len(pts)} pts -> {len(items)} object(s)")
        return {"success": True, "map_tag": self.tag, "query": query, "glb_path": glb,
                "conf_threshold": float(conf_threshold), "num_points": int(len(pts)),
                "num_objects": len(items), "gps_aligned": gt is not None, "objects": items}

    def _inherited_segmentation(self, query):
        safe = "".join(c if c.isalnum() else "_" for c in (query or "").strip())
        glbs = sorted(glob.glob(os.path.join(self.path, "segmented_*.glb")))
        if not glbs:
            return {"error": f"'{self.tag}' is a merged map (no per-frame images to segment) and "
                             "has no inherited segmentation."}
        match = [g for g in glbs if safe and os.path.basename(g) == f"segmented_{safe}.glb"]
        glb = match[0] if match else glbs[0]
        base = os.path.basename(glb)[len("segmented_"):-len(".glb")]
        meta = {}
        jp = os.path.join(self.path, f"segmented_{base}.json")
        if os.path.isfile(jp):
            with open(jp) as f:
                meta = json.load(f)
        objects = meta.get("objects", [])
        return {"success": True, "map_tag": self.tag, "query": meta.get("query", base),
                "glb_path": glb, "inherited": True,
                "conf_threshold": float(meta.get("conf_threshold", 0.0)), "num_points": 0,
                "num_objects": meta.get("num_objects", len(objects)),
                "gps_aligned": meta.get("gps_aligned", True), "objects": objects}

    def _write_segmented(self, query, conf, gps_aligned, items):
        safe = "".join(c if c.isalnum() else "_" for c in query.strip()) or "query"
        out = {"map_tag": self.tag, "query": query, "conf_threshold": float(conf),
               "gps_aligned": gps_aligned, "num_objects": len(items), "objects": items}
        with open(os.path.join(self.path, f"segmented_{safe}.json"), "w") as f:
            json.dump(out, f, indent=2)
        gps_items = [it for it in items if "position_gps" in it]
        if gps_items:
            kml.write_kml(gps_items, os.path.join(self.path, f"segmented_{safe}.kml"),
                          gps_key="position_gps", doc_name=f"{self.tag}: {query}")

    # ------------------------------------------------------------------ helpers
    def _predictions(self) -> dict:
        npz = np.load(os.path.join(self.path, "predictions.npz"), allow_pickle=True)
        return {k: npz[k] for k in npz.files if k != "metadata"}

    def _fallback(self, name):
        p = os.path.join(self.path, name)
        return p if os.path.exists(p) else None



def _viewpoints_payload(plan) -> list:
    """Per-viewpoint records from an NFN plan (GPS keys copied when tagged)."""
    out = []
    for i, vp in enumerate(plan.get("viewpoints", [])):
        item = {"id": i, "cluster_id": int(vp.get("cluster_id", -1)),
                "position": np.asarray(vp["camera_position"], float).tolist(),
                "look_dir": np.asarray(vp["camera_rotation"], float)[:, 2].tolist(),
                "target": np.asarray(vp["target"], float).tolist(),
                "score": float(vp.get("score", 0.0))}
        if "camera_position_gps" in vp:
            item["position_gps"] = vp["camera_position_gps"]
        if "target_gps" in vp:
            item["target_gps"] = vp["target_gps"]
        out.append(item)
    return out


def write_nfn_plan(plan, gps_transform, target_dir, segmented=None, seg_query=None) -> str:
    """Write next_flight_viewpoints.json (+ transform.json + KML when GPS-aligned)."""
    viewpoints = _viewpoints_payload(plan)
    out = {"num_viewpoints": len(viewpoints), "thresholds": plan.get("thresholds", {}),
           "gps_aligned": gps_transform is not None, "viewpoints": viewpoints}
    if segmented:
        out["segmented_objects"] = {"query": seg_query, "num_objects": len(segmented), "objects": segmented}
    path = os.path.join(target_dir, "next_flight_viewpoints.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    if gps_transform is not None:
        with open(os.path.join(target_dir, "transform.json"), "w") as f:
            json.dump(gps_transform.cfg, f, indent=2)
        kml.write_kml(viewpoints, os.path.join(target_dir, "next_flight_viewpoints.kml"),
                      gps_key="target_gps", doc_name="nfn_pts")
        kml.write_polygon_kml(viewpoints, os.path.join(target_dir, "next_flight_area.kml"),
                              gps_key="target_gps", doc_name="nfn_area")
    return path


def write_segmented_objects(items, seg_query, conf_threshold, gps_transform, target_dir) -> str:
    """Write segmented_objects.json (+ KML when any object has GPS)."""
    out = {"query": seg_query, "conf_threshold": conf_threshold,
           "gps_aligned": gps_transform is not None, "num_objects": len(items), "objects": items}
    path = os.path.join(target_dir, "segmented_objects.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    gps_items = [s for s in items if "position_gps" in s]
    if gps_items:
        kml.write_kml(gps_items, os.path.join(target_dir, "segmented_objects.kml"),
                      gps_key="position_gps", doc_name=f"SwiftMap Segmented: {seg_query}")
    return path
