# Copyright (C) 2024 Carnegie Mellon University
"""
Export NFN viewpoints as KML for Google My Maps.

Produces a single-layer KML (one Document, point placemarks, no Folders) so it
imports as a single layer -- avoids My Maps' per-map layer limit. Coordinates are
written in KML order (lon, lat, alt) with altitude zeroed (2D maps ignore it, and
the GPS altitude is the least-reliable axis).
"""

from typing import Any, Dict, List, Optional

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


def viewpoints_to_kml(viewpoints: List[Dict[str, Any]],
                      gps_key: str = "target_gps",
                      doc_name: str = "SwiftMap NFN Targets") -> Optional[str]:
    """Build a single-layer KML string from viewpoints' GPS.

    Args:
        viewpoints: list of viewpoint dicts (each with an ``id`` and a GPS field).
        gps_key: which GPS to plot -- "target_gps" (ground patch) or "position_gps".
        doc_name: KML document/layer name.

    Returns the KML string, or None if no viewpoint has the requested GPS.
    """
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


def write_kml(viewpoints: List[Dict[str, Any]], path: str,
              gps_key: str = "target_gps",
              doc_name: str = "SwiftMap NFN Targets") -> Optional[str]:
    """Write viewpoints to a KML file. Returns the path, or None if nothing written."""
    kml = viewpoints_to_kml(viewpoints, gps_key=gps_key, doc_name=doc_name)
    if kml is None:
        return None
    with open(path, "w") as f:
        f.write(kml)
    return path


# --- Polygon (area) KML -----------------------------------------------------
# Google My Maps polygon layer whose ring vertices are the NFN target points, so
# the plan reads as a coverage *area* rather than scattered pins.
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
    import math
    clat = sum(p[0] for p in latlons) / len(latlons)
    clon = sum(p[1] for p in latlons) / len(latlons)
    return sorted(latlons, key=lambda p: math.atan2(p[0] - clat, p[1] - clon))


def polygon_to_kml(viewpoints: List[Dict[str, Any]],
                   gps_key: str = "target_gps",
                   doc_name: str = "SwiftMap NFN Area",
                   layer_name: str = "Untitled layer",
                   placemark_name: str = "NFN Coverage") -> Optional[str]:
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


def write_polygon_kml(viewpoints: List[Dict[str, Any]], path: str,
                      gps_key: str = "target_gps",
                      doc_name: str = "SwiftMap NFN Area") -> Optional[str]:
    """Write a polygon (area) KML from viewpoints' target GPS. Returns path or None."""
    kml = polygon_to_kml(viewpoints, gps_key=gps_key, doc_name=doc_name)
    if kml is None:
        return None
    with open(path, "w") as f:
        f.write(kml)
    return path
