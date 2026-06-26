# Copyright (C) 2024 Carnegie Mellon University
"""
Geo-alignment: align a VGGT (local, unitless) reconstruction to GPS.

Solves a similarity transform (scale/rotation/translation) from the keyframe camera
trajectory to a GPS trace (Umeyama + optional ICP), then converts local points to
lat/lon/alt. See geo.py.
"""
from swiftmap.core.geo_transform.geo import (
    calibrate,
    umeyama,
    icp_sim3,
    GpsTransform,
    load_gps_csv,
)

__all__ = ["calibrate", "umeyama", "icp_sim3", "GpsTransform", "load_gps_csv"]
