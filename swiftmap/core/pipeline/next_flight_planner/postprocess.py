# Copyright (C) 2024 Carnegie Mellon University

"""Builds the FlightPlan artifacts; Map.write2disk() exports them."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from swiftmap.database.map import Map
from swiftmap.database.types import FlightPlan, GPS, Viewpoint


def generate_flight_plan(map: Map, plan: Dict[str, Any]) -> FlightPlan:
    """Turn the planner's ENU geometry into a FlightPlan with GPS-tagged waypoints."""
    georef = map.get_georeference()
    viewpoints: List[Viewpoint] = plan.get("viewpoints", [])
    waypoints: List[GPS] = []
    if georef is not None:
        for vp in viewpoints:
            vp.position_gps = GPS(*georef.enu_to_lla(vp.pose.position)[0])
            vp.target_gps = GPS(*georef.enu_to_lla(vp.target)[0])
            waypoints.append(vp.position_gps)

    flight_plan = FlightPlan(viewpoints=viewpoints,
                             clusters=plan.get("clusters", []),
                             thresholds=plan.get("thresholds", {}),
                             statistics=plan.get("statistics", {}),
                             waypoints=waypoints)
    map.update_flight_plan(flight_plan)
    return flight_plan


# ------------------------------------------------------------------ polygon (area) KML
# Ring vertices are the NFN targets, so the plan reads as an area, not pins.
_POLY_STYLE = """    <Style id="poly-000000-1200-77-nodesc-normal">
      <LineStyle>
        <color>ff000000</color>
        <width>1.2</width>
      </LineStyle>
      <PolyStyle>
        <color>4d000000</color>
        <fill>1</fill>
        <outline>1</outline>
      </PolyStyle>
      <BalloonStyle>
        <text><![CDATA[<h3>$[name]</h3>]]></text>
      </BalloonStyle>
    </Style>
    <Style id="poly-000000-1200-77-nodesc-highlight">
      <LineStyle>
        <color>ff000000</color>
        <width>1.8</width>
      </LineStyle>
      <PolyStyle>
        <color>4d000000</color>
        <fill>1</fill>
        <outline>1</outline>
      </PolyStyle>
      <BalloonStyle>
        <text><![CDATA[<h3>$[name]</h3>]]></text>
      </BalloonStyle>
    </Style>
    <StyleMap id="poly-000000-1200-77-nodesc">
      <Pair>
        <key>normal</key>
        <styleUrl>#poly-000000-1200-77-nodesc-normal</styleUrl>
      </Pair>
      <Pair>
        <key>highlight</key>
        <styleUrl>#poly-000000-1200-77-nodesc-highlight</styleUrl>
      </Pair>
    </StyleMap>"""


def _order_ring(latlons: List[tuple]) -> List[tuple]:
    """Order (lat, lon) points into a simple (non-self-intersecting) ring.

    Sort by polar angle around the centroid: sorting around an interior point
    always yields a star-shaped, non-self-intersecting polygon.
    """
    clat = sum(p[0] for p in latlons) / len(latlons)
    clon = sum(p[1] for p in latlons) / len(latlons)
    return sorted(latlons, key=lambda p: math.atan2(p[0] - clat, p[1] - clon))


def _polygon_to_kml(viewpoints: List[Dict[str, Any]],
                    gps_key: str, doc_name: str,
                    layer_name: str = "Untitled layer",
                    placemark_name: str = "nfn_target_area") -> Optional[str]:
    """Build a polygon KML whose ring vertices are the viewpoints' target GPS.

    Needs at least 3 GPS-tagged targets to form a polygon; returns None otherwise.
    Vertices are ordered around their centroid and the ring is closed (first point
    repeated at the end). Coordinates are KML order (lon, lat, alt) with alt zeroed.
    """
    latlons = [(float(vp[gps_key][0]), float(vp[gps_key][1]))
               for vp in viewpoints if vp.get(gps_key)]
    if len(latlons) < 3:
        return None

    ring = _order_ring(latlons)
    ring.append(ring[0])  # close the ring
    coords = "\n".join(f"                {lon:.7f},{lat:.7f},0" for lat, lon in ring)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>{doc_name}</name>\n'
        '    <description/>\n'
        f'{_POLY_STYLE}\n'
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


def _write_polygon_kml(viewpoints: List[Dict[str, Any]], path: str,
                       gps_key: str = "target_gps",
                       doc_name: str = "nfn_area") -> Optional[str]:
    """Write a polygon (area) KML from viewpoints' target GPS. Returns path or None."""
    kml = _polygon_to_kml(viewpoints, gps_key=gps_key, doc_name=doc_name)
    if kml is None:
        return None
    with open(path, "w") as f:
        f.write(kml)
    return path
