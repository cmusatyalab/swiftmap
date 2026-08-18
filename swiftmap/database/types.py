from typing import List, Optional

import numpy as np
import trimesh

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



class Georeference:
    """
    A class representing a georeference with scale, rotation, and translation.
    """

    def __init__(self, scale: float, rotation: List[List[float]], translation: List[float]):
        self.scale = scale
        self.rotation = rotation
        self.translation = translation

    def __repr__(self):
        return f"Georeference(scale={self.scale}, rotation={self.rotation}, translation={self.translation})"



class PointCloud:

    _CONF_EPSILON = 1e-6

    def __init__(self, world_points=None, world_points_conf=None,
                 world_points_from_depth=None, depth_conf=None,
                 images=None, extrinsic=None, intrinsic=None,
                 camera_positions=None, camera_rotations=None, metadata=None):
        self.world_points = world_points
        self.world_points_conf = world_points_conf
        self.world_points_from_depth = world_points_from_depth
        self.depth_conf = depth_conf
        self.images = images
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.camera_positions = camera_positions
        self.camera_rotations = camera_rotations
        self.metadata = metadata or {}

        self.scene = None                # trimesh.Scene -- scene.glb
        self.confidence_scene = None     # trimesh.Scene -- confidence_map.glb
        self.confidence_stats = None     # dict -- point counts/coverage from the above
        self.camera_poses = None         # dict -- camera_poses.json payload

    def __repr__(self):
        return f"PointCloud(metadata={self.metadata})"

    def flatten_colors(self) -> np.ndarray:
        images = np.asarray(self.images)
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        return (images.reshape(-1, 3) * 255).astype(np.uint8)



class FlightPlan:
    """
    A class representing a flight plan with waypoints and other relevant information.
    """

    def __init__(self, waypoints: List[GPS], altitude: float, speed: float):
        self.waypoints = waypoints
        self.altitude = altitude
        self.speed = speed

    def __repr__(self):
        return f"FlightPlan(waypoints={self.waypoints}, altitude={self.altitude}, speed={self.speed})"
    
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