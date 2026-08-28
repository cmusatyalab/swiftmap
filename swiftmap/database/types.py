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

        # ---- gps transformer
        self.geo_scene = None            # trimesh.Scene -- scene_geo.glb (ENU metres)

        # ---- segmentor
        self.segmented_worldpoints: List["SegmentedTarget"] = []

        self.scene = None                # trimesh.Scene -- scene.glb
        self.confidence_scene = None     # trimesh.Scene -- scene_confidence.glb
        self.segmented_scene = None      # trimesh.Scene -- seg_scene.glb
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

    def to_geojson(self, georef: "Georeference") -> dict:
        """scene_geo.geojson: where scene_geo.glb's local (0, 0, 0) sits on Earth.

        glTF carries no coordinate system, so the mesh ships with its origin beside it.
        """
        return {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [georef.lla0[1], georef.lla0[0], georef.lla0[2]]},
            "properties": {"layer": "scene_origin",
                           "model": "scene_geo.glb",
                           "description": "model local (0,0,0); the first keyframe's GPS fix",
                           "altitude_mode": "absolute",
                           "scale": georef.scale}}]}

    def cameras_to_kml(self, georef: "Georeference") -> Optional[str]:
        """camera_poses.kml: a placemark where each keyframe was shot."""
        if not self.cameras:
            return None
        placemarks = "\n".join(
            f"    <Placemark><name>frame_{i:03d}</name>"
            f"<Point><altitudeMode>absolute</altitudeMode>"
            f"<coordinates>{lon:.9f},{lat:.9f},{alt:.3f}</coordinates></Point></Placemark>"
            for i, (lat, lon, alt) in enumerate(georef.to_lla(self.camera_centers())))
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n'
                '    <name>camera_poses</name>\n'
                f'{placemarks}\n  </Document>\n</kml>\n')

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


    def to_kml(self, doc_name: str = "next_flight") -> Optional[str]:
        """One KML holding the planned waypoints and the area they enclose.

        The polygon's ring is the waypoints themselves, at the same altitude, so the
        two layers sit on top of each other. None if there are no waypoints.
        """
        if not self.waypoints:
            return None

        pins = "\n".join(
            f"    <Placemark>\n"
            f"      <name>v{i}</name>\n"
            f"      <styleUrl>#icon-1899-0288D1-nodesc</styleUrl>\n"
            f"      <Point><altitudeMode>absolute</altitudeMode>"
            f"<coordinates>{wp.longitude:.9f},{wp.latitude:.9f},"
            f"{wp.altitude or 0:.3f}</coordinates></Point>\n"
            f"    </Placemark>"
            for i, wp in enumerate(self.waypoints))

        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
                '  <Document>\n'
                f'    <name>{doc_name}</name>\n'
                f'{self._STYLE}\n'
                f'{self._POLY_STYLE}\n'
                f'{self._area_placemark()}'
                f'{pins}\n'
                '  </Document>\n'
                '</kml>\n')

    def _area_placemark(self) -> str:
        """The polygon enclosing the waypoints; empty until there are three of them."""
        if len(self.waypoints) < 3:
            return ""
        ring = self._order_ring([(wp.latitude, wp.longitude, wp.altitude or 0.0)
                                 for wp in self.waypoints])
        ring.append(ring[0])  # close the ring
        coords = "\n".join(f"          {lon:.9f},{lat:.9f},{alt:.3f}"
                            for lat, lon, alt in ring)
        return ('    <Placemark>\n'
                '      <name>nfn_flight_area</name>\n'
                '      <styleUrl>#poly-000000-1200-77-nodesc</styleUrl>\n'
                '      <Polygon><altitudeMode>absolute</altitudeMode>\n'
                '        <outerBoundaryIs><LinearRing><coordinates>\n'
                f'{coords}\n'
                '        </coordinates></LinearRing></outerBoundaryIs>\n'
                '      </Polygon>\n'
                '    </Placemark>\n')

    @staticmethod
    def _order_ring(latlons: List[tuple]) -> List[tuple]:
        """Sort (lat, lon, alt) by polar angle about the centroid: a simple, closed ring."""
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
