# Copyright (C) 2024 Carnegie Mellon University

"""The ``Map``: one stored map directory in the database.

Holds a raw ``predictions.npz`` (a reconstructed batch) or a merged ``merged_points.npz``
plus transform/camera/map json. It is the store record -- identity, metadata, ``load``
(-> MapData) and ``write`` (MapData -> dir); previews and segmentation come from the pipeline."""

import glob
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from swiftmap.core.primitives.types import Georeference, MapData, Reconstruction

# write() imports confidence_mapping on demand: it is a heavy dep needed only there.

_GEOHASH_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_META_NAMES = ("map.json",)  # new name


def _meta_path(dir_path: str) -> Optional[str]:
    """A map dir's metadata file: new ``map.json``, else None."""
    for name in _META_NAMES:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            return p
    return None


@dataclass
class MapMetaData:
    """What ``map.json`` holds: a map's identity and summary."""

    map_tag: str
    site: str = "map"
    created: str = ""
    num_keyframes: int = 0
    num_points: int = 0
    gps_aligned: bool = False
    center_gps: Optional[list] = None
    geohash: Optional[str] = None
    source_maps: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "MapMetaData":
        with open(path) as f:
            raw = json.load(f)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        known.setdefault("map_tag", os.path.basename(os.path.dirname(path)))
        return cls(**known)

    def with_center(self, center) -> "MapMetaData":
        """Set the GPS center (and its geohash) from a [lat, lon, alt]."""
        if center is None:
            return self
        self.center_gps = [float(center[0]), float(center[1]), float(center[2])]
        self.geohash = geohash_encode(self.center_gps[0], self.center_gps[1])
        self.gps_aligned = True
        return self

    def asdict(self) -> dict:
        return asdict(self)


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
    def metadata(self) -> MapMetaData:
        if self._meta is None:
            p = _meta_path(self.path)
            self._meta = MapMetaData.load(p) if p else MapMetaData(map_tag=self.tag)
        return self._meta

    def save_metadata(self, meta: MapMetaData) -> MapMetaData:
        """Persist ``meta`` as this map's ``map.json``."""
        self._meta = meta
        self._dump("map.json", meta.asdict())
        return meta

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

    @property
    def predictions(self) -> dict:
        """The raw prediction arrays of a non-merged map."""
        npz = np.load(os.path.join(self.path, "predictions.npz"), allow_pickle=True)
        return {k: npz[k] for k in npz.files if k != "metadata"}

    # ---------------------------------------------------------------- geometry
    def load(self, conf_thres: float = 50.0) -> MapData:
        """Load this map's geometry into a MapData (merged flat, or raw predictions).

        A merged map's cloud is already confidence-filtered and loaded verbatim; a raw
        ``predictions.npz`` gets the ``conf_thres`` percentile cut.
        """
        gt = self.transform or Georeference(
            {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
             "lat0": 0.0, "lon0": 0.0, "alt0": 0.0})  # not GPS-aligned yet: local frame
        frames = self.frames

        flat = os.path.join(self.path, "merged_points.npz")
        if os.path.isfile(flat):
            z = np.load(flat)
            return MapData(z["points"], z["colors"], z["conf"], gt, frames)

        npz = np.load(os.path.join(self.path, "predictions.npz"), allow_pickle=True)
        recon = Reconstruction({k: npz[k] for k in npz.files if k != "metadata"})
        return recon.to_mapdata(gt, frames=frames, conf_percentile=conf_thres)

    # ------------------------------------------------------------- persistence
    def write(self, mapdata: MapData, source_maps=(), site: str = None,
              created: datetime = None) -> "Map":
        """Persist ``mapdata`` into this dir: cloud, transform, poses, map.json.

        Data only -- previews come from ``pipeline.renderer.write_previews``."""
        created = created or datetime.now()
        site = site or self.metadata.site or "map"
        os.makedirs(self.path, exist_ok=True)

        pts = np.asarray(mapdata.points)
        cols = np.asarray(mapdata.colors, np.uint8)
        conf = np.asarray(mapdata.conf, dtype=float)
        frames, o = mapdata.frames, mapdata.origin
        sources = list(source_maps)

        np.savez(os.path.join(self.path, "merged_points.npz"), points=pts, colors=cols, conf=conf)

        tf = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
              "lat0": o[0], "lon0": o[1], "alt0": o[2], "mode": "merged",
              "num_points": int(len(pts)), "source_maps": sources}
        self._dump("transform.json", tf)
        self._dump("camera_poses.json",
                   {"metadata": {"frame": "ENU meters about origin (lat0, lon0, alt0)",
                                 "source_maps": sources, "num_keyframes": len(frames)},
                    "frames": frames})

        center = Georeference(tf).to_lla(pts.mean(axis=0)) if len(pts) else np.array(o)
        self.save_metadata(MapMetaData(
            map_tag=self.tag, site=site, created=created.isoformat(timespec="seconds"),
            num_keyframes=len(frames), num_points=int(len(pts)),
            source_maps=sources).with_center(center))
        print(f"[map] wrote '{self.tag}': {len(pts):,} pts, {len(frames)} cameras")
        return self

    def stamp_metadata(self, site: str, created: datetime = None, source_maps=()) -> dict:
        """Write ``map.json`` for an already-populated dir (a freshly reconstructed batch).

        The center is the mean camera position in GPS; ``num_points`` is left out since a
        raw map's cloud is only known once ``load()`` filters it.
        """
        created = created or datetime.now()
        gt, frames = self.transform, self.frames
        center = None
        if gt is not None:
            cams = np.asarray([f["camera_position_world"] for f in frames], float) if frames else None
            center = np.asarray(gt.to_lla(cams.mean(axis=0)), float) if cams is not None \
                else np.array([gt.lat0, gt.lon0, gt.alt0])
        return self.save_metadata(MapMetaData(
            map_tag=self.tag, site=site, created=created.isoformat(timespec="seconds"),
            num_keyframes=len(frames), source_maps=list(source_maps)).with_center(center))

    def _dump(self, name: str, obj):
        with open(os.path.join(self.path, name), "w") as f:
            json.dump(obj, f, indent=2)

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)
