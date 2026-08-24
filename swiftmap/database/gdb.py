# Copyright (C) 2024 Carnegie Mellon University

"""Esri File Geodatabase export: a Map's derived vector layers.

A .gdb is a vector/table container, so it holds the layers a GIS analyst clicks on --
the planned waypoints, the target area, the flown camera positions -- in WGS84 (the
same coordinates as the KMLs). The dense cloud has no geodatabase geometry type; it
ships beside this as ``scene.laz``.
"""

import shutil
from typing import Any, Dict, List, Optional
from swiftmap.database.map import Map
import numpy as np

_CRS = "EPSG:4326"


def _layers(map: Map) -> Dict[str, tuple]:
    """Build {layer_name: (geometries, fields, field_data, geometry_type)} for one Map."""
    import shapely

    layers = {}
    plan = map.get_flight_plan()
    georef = map.get_georeference()

    if plan is not None and plan.waypoints:
        pts = [shapely.Point(vp.position_gps.longitude, vp.position_gps.latitude,
                             vp.position_gps.altitude or 0.0) for vp in plan.viewpoints]
        layers["waypoints"] = (
            pts,
            ["vp_id", "cluster_id", "azimuth_deg", "score"],
            [np.arange(len(pts), dtype=np.int32),
             np.array([vp.cluster_id for vp in plan.viewpoints], dtype=np.int32),
             np.array([vp.azimuth_deg for vp in plan.viewpoints]),
             np.array([vp.score for vp in plan.viewpoints])],
            "Point Z",
        )

        ring = [(vp.target_gps.latitude, vp.target_gps.longitude)
                for vp in plan.viewpoints if vp.target_gps is not None]
        if len(ring) >= 3:
            ring = plan._order_ring(ring)
            layers["target_area"] = (
                [shapely.Polygon([(lon, lat) for lat, lon in ring + [ring[0]]])],
                ["name", "num_viewpoints"],
                [np.array(["nfn_target_area"]), np.array([len(plan.viewpoints)], dtype=np.int32)],
                "Polygon",
            )

    pt = map.get_pointcloud()
    if pt is not None and pt.cameras and georef is not None:
        lla = georef.to_lla(pt.camera_centers())
        names = map.keyframe_images or [f"keyframe_{i}" for i in range(len(lla))]
        layers["camera_poses"] = (
            [shapely.Point(lon, lat, alt) for lat, lon, alt in lla],
            ["frame", "image_name"],
            [np.arange(len(lla), dtype=np.int32), np.array(names[:len(lla)], dtype=object)],
            "Point Z",
        )

    return layers


def write_gdb(map, path: str) -> Optional[str]:
    """Write a Map's vector layers as a File Geodatabase; None if there is nothing to write."""
    import shapely
    from pyogrio.raw import write

    layers = _layers(map)
    if not layers:
        return None

    shutil.rmtree(path, ignore_errors=True)  # OpenFileGDB appends otherwise
    for name, (geoms, fields, field_data, geom_type) in layers.items():
        write(path, geometry=shapely.to_wkb(np.array(geoms), flavor="iso"),
              field_data=field_data, fields=fields, layer=name,
              driver="OpenFileGDB", crs=_CRS, geometry_type=geom_type)
    return path
