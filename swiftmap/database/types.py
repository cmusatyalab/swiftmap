from typing import List, Optional

import numpy as np

# ================================================================== shared

class GPS:
    """
    A class representing a GPS coordinate with latitude and longitude.
    """

    def __init__(self, latitude: float, longitude: float, altitude: Optional[float] = None):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def __repr__(self):
        return f"GPS(latitude={self.latitude}, longitude={self.longitude}, altitude={self.altitude})"

class CameraPose:
    """One keyframe's world pose: camera centre and camera-to-world rotation."""

    def __init__(self, position: np.ndarray, rotation: np.ndarray):
        self.position = position      # (3,)
        self.rotation = rotation      # (3, 3)

    def __repr__(self):
        return f"CameraPose(position={np.round(self.position, 3).tolist()})"

    @classmethod
    def from_extrinsic(cls, extrinsic: np.ndarray) -> "CameraPose":
        """Build from one (3, 4) world-to-camera extrinsic: centre ``-R^T t``, rotation ``R^T``."""
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        return cls(-R.T @ t, R.T)


class PointCloud:

    _CONF_EPSILON = 1e-6

    def __init__(self,
                 world_points: Optional[np.ndarray] = None,               # (S, H, W, 3)
                 world_points_conf: Optional[np.ndarray] = None,          # (S, H, W)
                 world_points_from_depth: Optional[np.ndarray] = None,    # (S, H, W, 3)
                 depth_conf: Optional[np.ndarray] = None,                 # (S, H, W)
                 images: Optional[np.ndarray] = None,                     # (S, 3, H, W)
                 extrinsic: Optional[np.ndarray] = None,                  # (S, 3, 4)
                 intrinsic: Optional[np.ndarray] = None,                  # (S, 3, 3)
                 cameras: Optional[List[CameraPose]] = None,
                 metadata: Optional[dict] = None):
        # ---- reconstructor
        self.world_points = world_points
        self.world_points_conf = world_points_conf
        self.world_points_from_depth = world_points_from_depth
        self.depth_conf = depth_conf
        self.images = images
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.cameras: List[CameraPose] = cameras or []
        self.metadata = metadata or {}

        # ---- segmentor
        self.segmented_worldpoints: List["SegmentedTarget"] = []

        self.scene = None                # trimesh.Scene -- scene.glb
        self.confidence_scene = None     # trimesh.Scene -- scene_confidence.glb
        self.segmented_scene = None      # trimesh.Scene -- seg_scene.glb
        self.camera_poses = None         # dict -- camera_poses.json payload
        self.model_input = None          # [(filename, jpeg bytes)] -- model_input/

    def __repr__(self):
        return f"PointCloud(metadata={self.metadata})"

    def flatten_points(self) -> np.ndarray:
        """World points as (N, 3), out of the per-pixel grid."""
        return self.world_points.reshape(-1, 3)

    def flatten_conf(self) -> np.ndarray:
        """Per-point confidence as (N,), aligned with flatten_points()."""
        return self.world_points_conf.reshape(-1).astype(float)

    def flatten_colors(self) -> np.ndarray:
        """Per-point RGB as (N, 3) uint8, aligned with flatten_points()."""
        images = self.images
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        return (images.reshape(-1, 3) * 255).astype(np.uint8)

    def confidence_mask(self, percentile: float = 0.0,
                        conf: Optional[np.ndarray] = None) -> np.ndarray:
        """Keep-mask over flatten_points(): finite points at/above the confidence percentile.

        ``conf`` overrides the stored confidence (e.g. a sky-masked copy)."""
        conf = self.flatten_conf() if conf is None else conf.reshape(-1).astype(float)
        keep = np.isfinite(self.flatten_points()).all(1) & (conf > self._CONF_EPSILON)
        if percentile:
            keep &= conf >= np.percentile(conf, float(percentile))
        return keep

    def camera_centers(self) -> np.ndarray:
        """Camera positions as (S, 3)."""
        return np.array([c.position for c in self.cameras], dtype=float).reshape(-1, 3)

    def add_segmentation(self, query: str, points: np.ndarray) -> "SegmentedTarget":
        """Append one query's lifted points, giving it a hue no earlier target has used."""
        import colorsys
        self.segmented_worldpoints = [t for t in self.segmented_worldpoints if t.query != query]
        hue = (len(self.segmented_worldpoints) * 0.6180339887) % 1.0     # golden-angle spacing
        color = np.array([int(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.9, 1.0)],
                         dtype=np.uint8)
        target = SegmentedTarget(query, points, color)
        self.segmented_worldpoints.append(target)
        return target

    def to_las(self, georef: "Georeference", percentile: float = 0.0,
               conf: Optional[np.ndarray] = None):
        """LAS 1.4 (format 7) point cloud in the georeference's UTM zone: XYZ + RGB + intensity.

        Confidence becomes LAS intensity, rescaled to uint16. ``conf`` overrides the
        stored confidence (e.g. a sky-masked copy), matching confidence_mask().
        """
        import laspy
        from pyproj import CRS

        keep = self.confidence_mask(percentile, conf)
        xyz = georef.to_utm(self.flatten_points()[keep])
        rgb = self.flatten_colors()[keep]
        c = (self.flatten_conf() if conf is None else conf.reshape(-1).astype(float))[keep]

        header = laspy.LasHeader(version="1.4", point_format=7)
        header.offsets = np.floor(xyz.min(axis=0))   # keeps the int32 payload small
        header.scales = np.array([0.001, 0.001, 0.001])
        header.add_crs(CRS.from_epsg(georef.utm_epsg))

        las = laspy.LasData(header)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.red, las.green, las.blue = (rgb[:, i].astype(np.uint16) * 257 for i in range(3))
        lo, hi = float(c.min()), float(c.max())
        las.intensity = (((c - lo) / (hi - lo) if hi > lo else np.ones_like(c)) * 65535
                         ).astype(np.uint16)
        return las

# ========================================================= reconstructor

# ========================================================= gps transformer


class Georeference:
    """Similarity transform from local reconstruction coords to ENU metres about an origin."""

    def __init__(self, scale: float, rotation, translation, origin: GPS):
        self.scale = float(scale)
        self.R = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.t = np.asarray(translation, dtype=float).reshape(3)
        self.origin = origin
        self.lla0 = (origin.latitude, origin.longitude, origin.altitude or 0.0)

    def __repr__(self):
        return f"Georeference(scale={self.scale:.4f}, origin={self.origin})"

    def to_enu(self, points, origin=None) -> np.ndarray:
        """Local points -> ENU metres; re-based on ``origin`` (lat, lon, alt) if given."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        enu = self.scale * (pts @ self.R.T) + self.t
        if origin is not None and tuple(origin) != self.lla0:
            import pymap3d
            lat, lon, alt = pymap3d.enu2geodetic(enu[:, 0], enu[:, 1], enu[:, 2], *self.lla0)
            e, n, u = pymap3d.geodetic2enu(lat, lon, alt, *origin)
            enu = np.column_stack([e, n, u])
        return enu

    def to_lla(self, points) -> np.ndarray:
        """Local points -> (N, 3) [lat, lon, alt]."""
        return self.enu_to_lla(self.to_enu(points))

    def enu_to_lla(self, enu) -> np.ndarray:
        """ENU metres (about this origin) -> (N, 3) [lat, lon, alt]."""
        import pymap3d
        enu = np.asarray(enu, dtype=float).reshape(-1, 3)
        lat, lon, alt = pymap3d.enu2geodetic(enu[:, 0], enu[:, 1], enu[:, 2], *self.lla0)
        return np.column_stack([lat, lon, alt])

    def to_json(self) -> dict:
        """transform.json payload."""
        return {"scale": self.scale,
                "rotation": self.R.tolist(),
                "translation": self.t.tolist(),
                "origin": {"latitude": self.lla0[0], "longitude": self.lla0[1],
                           "altitude": self.lla0[2]}}

    @property
    def utm_epsg(self) -> int:
        """EPSG code of the WGS84/UTM zone containing this origin."""
        zone = int((self.lla0[1] + 180.0) / 6.0) + 1
        return (32600 if self.lla0[0] >= 0 else 32700) + zone

    def to_utm(self, points) -> np.ndarray:
        """Local points -> (N, 3) [easting, northing, altitude] in this origin's UTM zone."""
        from pyproj import CRS, Transformer
        lla = self.to_lla(points)
        fwd = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(self.utm_epsg),
                                   always_xy=True)
        east, north = fwd.transform(lla[:, 1], lla[:, 0])
        return np.column_stack([east, north, lla[:, 2]])


# ===================================================== next flight planner


class Cluster:
    """A weakly-mapped patch of ground worth re-flying, in ENU metres."""

    def __init__(self, id: int, centroid: np.ndarray, normal: np.ndarray,
                 num_points: int, radius: float):
        self.id = id
        self.centroid = centroid      # (3,)
        self.normal = normal          # (3,) surface normal, oriented up
        self.num_points = num_points
        self.radius = radius

    def __repr__(self):
        return f"Cluster(id={self.id}, num_points={self.num_points}, radius={self.radius:.1f}m)"


class Viewpoint:
    """One suggested camera pose over a Cluster, GPS-tagged once georeferenced."""

    def __init__(self, cluster_id: int, pose: CameraPose, target: np.ndarray,
                 azimuth_deg: float, score: float):
        self.cluster_id = cluster_id
        self.pose = pose              # camera position/rotation in ENU metres
        self.target = target          # (3,) the cluster centroid it looks at
        self.azimuth_deg = azimuth_deg
        self.score = score
        self.position_gps: Optional[GPS] = None
        self.target_gps: Optional[GPS] = None

    def __repr__(self):
        return f"Viewpoint(cluster_id={self.cluster_id}, azimuth={self.azimuth_deg:.0f}deg)"

    @property
    def look_dir(self) -> np.ndarray:
        """Unit viewing direction (OpenCV axes: +Z forward)."""
        return self.pose.rotation[:, 2]

    def to_json(self, id: int) -> dict:
        """One next_flight_viewpoints.json entry."""
        item = {
            "id": id,
            "cluster_id": self.cluster_id,
            "position": self.pose.position.tolist(),
            "look_dir": self.look_dir.tolist(),
            "target": np.asarray(self.target, dtype=float).tolist(),
            "azimuth_deg": self.azimuth_deg,
            "score": self.score,
        }
        for key, gps in (("position_gps", self.position_gps), ("target_gps", self.target_gps)):
            if gps is not None:
                item[key] = [gps.latitude, gps.longitude, gps.altitude]
        return item


class FlightPlan:
    """A planned next flight: viewpoints in local coords, GPS waypoints once aligned."""

    def __init__(self, viewpoints: Optional[List[Viewpoint]] = None,
                 clusters: Optional[List[Cluster]] = None,
                 thresholds=None, statistics=None,
                 waypoints: Optional[List[GPS]] = None,
                 altitude: Optional[float] = None, speed: Optional[float] = None):
        self.viewpoints = viewpoints or []
        self.clusters = clusters or []
        self.thresholds = thresholds or {}
        self.statistics = statistics or {}
        self.waypoints = waypoints or []
        self.altitude = altitude
        self.speed = speed

    def __repr__(self):
        return (f"FlightPlan(num_viewpoints={len(self.viewpoints)}, "
                f"num_waypoints={len(self.waypoints)})")

    def to_json(self) -> dict:
        """next_flight_viewpoints.json payload."""
        return {
            "num_viewpoints": len(self.viewpoints),
            "gps_aligned": bool(self.waypoints),
            "thresholds": self.thresholds,
            "statistics": self.statistics,
            "viewpoints": [vp.to_json(i) for i, vp in enumerate(self.viewpoints)],
        }


    def to_kml(self, doc_name: str = "flight_plan") -> Optional[str]:
        """Single-layer placemark KML string of this plan's waypoints.

        None if there are no waypoints.
        """
        if not self.waypoints:
            return None
        placemarks = [
            f"    <Placemark>\n"
            f"      <name>v{i}</name>\n"
            f"      <styleUrl>#icon-1899-0288D1-nodesc</styleUrl>\n"
            f"      <Point><coordinates>{wp.longitude:.13f},{wp.latitude:.13f},"
            f"{wp.altitude or 0}</coordinates></Point>\n"
            f"    </Placemark>"
            for i, wp in enumerate(self.waypoints)
        ]
        kml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            '  <Document>\n'
            f'    <name>{doc_name}</name>\n'
            f'{self._STYLE}\n'
            f'{chr(10).join(placemarks)}\n'
            '  </Document>\n'
            '</kml>\n'
        )
        return kml

    def to_polygon_kml(self, doc_name: str = "nfn_area",
                       layer_name: str = "Untitled layer",
                       placemark_name: str = "nfn_target_area") -> Optional[str]:
        """Polygon KML whose ring vertices are the viewpoints' target GPS.

        Reads as an area rather than pins. None unless at least 3 targets are
        GPS-tagged; the ring is ordered around its centroid and closed.
        """
        latlons = [(vp.target_gps.latitude, vp.target_gps.longitude)
                   for vp in self.viewpoints if vp.target_gps is not None]
        if len(latlons) < 3:
            return None

        ring = self._order_ring(latlons)
        ring.append(ring[0])  # close the ring
        coords = "\n".join(f"                {lon:.7f},{lat:.7f},0" for lat, lon in ring)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            '  <Document>\n'
            f'    <name>{doc_name}</name>\n'
            '    <description/>\n'
            f'{self._POLY_STYLE}\n'
            '    <Folder>\n'
            f'      <name>{layer_name}</name>\n'
            '      <Placemark>\n'
            f'        <name>{placemark_name}</name>\n'
            '        <styleUrl>#poly-000000-1200-77-nodesc</styleUrl>\n'
            '        <Polygon>\n'
            '          <outerBoundaryIs>\n'
            '            <LinearRing>\n'
            '              <tessellate>1</tessellate>\n'
            '              <coordinates>\n'
            f'{coords}\n'
            '              </coordinates>\n'
            '            </LinearRing>\n'
            '          </outerBoundaryIs>\n'
            '        </Polygon>\n'
            '      </Placemark>\n'
            '    </Folder>\n'
            '  </Document>\n'
            '</kml>\n'
        )

    @staticmethod
    def _order_ring(latlons: List[tuple]) -> List[tuple]:
        """Sort (lat, lon) by polar angle about the centroid: a non-self-intersecting ring."""
        clat = sum(p[0] for p in latlons) / len(latlons)
        clon = sum(p[1] for p in latlons) / len(latlons)
        return sorted(latlons, key=lambda p: np.arctan2(p[0] - clat, p[1] - clon))

    _STYLE = """    <Style id="icon-1899-0288D1-nodesc-normal">
    <IconStyle><color>ffd18802</color><scale>1</scale>
        <Icon><href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href></Icon>
        <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/></IconStyle>
    <LabelStyle><scale>1</scale></LabelStyle>
    <BalloonStyle><text><![CDATA[<h3>$[name]</h3>]]></text></BalloonStyle>
    </Style>
    <Style id="icon-1899-0288D1-nodesc-highlight">
    <IconStyle><color>ffd18802</color><scale>1</scale>
        <Icon><href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href></Icon>
        <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/></IconStyle>
    <LabelStyle><scale>1</scale></LabelStyle>
    <BalloonStyle><text><![CDATA[<h3>$[name]</h3>]]></text></BalloonStyle>
    </Style>
    <StyleMap id="icon-1899-0288D1-nodesc">
    <Pair><key>normal</key><styleUrl>#icon-1899-0288D1-nodesc-normal</styleUrl></Pair>
    <Pair><key>highlight</key><styleUrl>#icon-1899-0288D1-nodesc-highlight</styleUrl></Pair>
    </StyleMap>"""

    _POLY_STYLE = """    <Style id="poly-000000-1200-77-nodesc-normal">
      <LineStyle><color>ff000000</color><width>1.2</width></LineStyle>
      <PolyStyle><color>4d000000</color><fill>1</fill><outline>1</outline></PolyStyle>
      <BalloonStyle><text><![CDATA[<h3>$[name]</h3>]]></text></BalloonStyle>
    </Style>
    <Style id="poly-000000-1200-77-nodesc-highlight">
      <LineStyle><color>ff000000</color><width>1.8</width></LineStyle>
      <PolyStyle><color>4d000000</color><fill>1</fill><outline>1</outline></PolyStyle>
      <BalloonStyle><text><![CDATA[<h3>$[name]</h3>]]></text></BalloonStyle>
    </Style>
    <StyleMap id="poly-000000-1200-77-nodesc">
      <Pair><key>normal</key><styleUrl>#poly-000000-1200-77-nodesc-normal</styleUrl></Pair>
      <Pair><key>highlight</key><styleUrl>#poly-000000-1200-77-nodesc-highlight</styleUrl></Pair>
    </StyleMap>"""


# ================================================================ segmentor


class SegmentedTarget:
    """One text query lifted to 3D: the world points whose pixels matched it."""

    def __init__(self, query: str, points: np.ndarray, color: np.ndarray):
        self.query = query
        self.points = points          # (M, 3) local reconstruction coords
        self.color = color            # (3,) uint8, this target's scene colour

    def __repr__(self):
        return f"SegmentedTarget(query={self.query!r}, num_points={len(self.points)})"
