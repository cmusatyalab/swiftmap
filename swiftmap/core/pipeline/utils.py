# Copyright (C) 2024 Carnegie Mellon University

"""The two helpers shared across pipeline stages (reconstructor and
segmentor both build point clouds; next_flight_planner and segmentor both export KML):
a bare trimesh point cloud, and single-layer placemark KML for Google My Maps.
"""

from typing import Any, Dict, List, Optional

import numpy as np

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


def pointcloud(points, colors):
    """trimesh PointCloud from (N,3) points + (N,3) uint8 RGB (alpha filled)."""
    import trimesh
    rgba = np.hstack([np.asarray(colors, np.uint8),
                      np.full((len(points), 1), 255, np.uint8)])
    return trimesh.PointCloud(vertices=np.asarray(points), colors=rgba)


def write_kml(viewpoints: List[Dict[str, Any]], path: str,
              gps_key: str = "target_gps",
              doc_name: str = "nfn_pts") -> Optional[str]:
    """Write viewpoints to a single-layer placemark KML file.

    Returns the path, or None if nothing written (no viewpoint had ``gps_key``).
    """
    kml = _viewpoints_to_kml(viewpoints, gps_key=gps_key, doc_name=doc_name)
    if kml is None:
        return None
    with open(path, "w") as f:
        f.write(kml)
    return path


def _viewpoints_to_kml(viewpoints: List[Dict[str, Any]],
                       gps_key: str, doc_name: str) -> Optional[str]:
    """Build a single-layer KML string from viewpoints' GPS."""
    placemarks = []
    for vp in viewpoints:
        gps = vp.get(gps_key)
        if not gps:
            continue
        lat, lon = float(gps[0]), float(gps[1])
        vid = vp.get("id", len(placemarks))
        placemarks.append(
            f"    <Placemark>\n"
            f"      <name>v{vid}</name>\n"
            f"      <styleUrl>#icon-1899-0288D1-nodesc</styleUrl>\n"
            f"      <Point><coordinates>{lon:.13f},{lat:.13f},0</coordinates></Point>\n"
            f"    </Placemark>"
        )
    if not placemarks:
        return None
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>{doc_name}</name>\n'
        f'{_STYLE}\n'
        f'{chr(10).join(placemarks)}\n'
        '  </Document>\n'
        '</kml>\n'
    )
