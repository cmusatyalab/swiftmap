# Copyright (C) 2024 Carnegie Mellon University

"""Builds the georeferenced scene; Map.write2disk() exports it.

The reconstruction's own coordinates are unitless and aligned to the first camera, so
a viewer that places models on the Earth draws them at the wrong size and heading. This
rebuilds the surface in ENU metres about the georeference origin, rotated into glTF's
Y-up convention. glTF carries no coordinate system, so the origin ships alongside it as
GeoJSON (see ``Georeference.to_geojson``) for whoever has to place the model.
"""

from typing import Any, Dict, Optional

import numpy as np
import trimesh

from swiftmap.database.map import Map

_MESH_STRIDE = 1        # every Nth pixel; faces scale with 1/stride^2
_MESH_EDGE_RATIO = 5.0  # drop faces whose longest edge exceeds this x the median

# ENU (east, north, up) -> glTF's Y-up axes: x=-east, y=up, z=north.
_ENU_TO_YUP = np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])


def generate_geo_scene(map: Map, params: Dict[str, Any]) -> Optional[trimesh.Scene]:
    """Rebuild the surface in metres about the GPS origin, attached as pt.geo_scene."""
    pt = map.get_pointcloud()
    georef = map.get_georeference()
    if pt is None or pt.world_points is None or georef is None:
        return None

    stride = max(int(params.get("mesh_stride", _MESH_STRIDE)), 1)
    verts, faces, colors = _triangulate_grid(pt, stride, params.get("conf_threshold", 0.0))
    mesh = trimesh.Trimesh(vertices=georef.to_enu(verts) @ _ENU_TO_YUP.T, faces=faces,
                           vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.visual.material = _surface_material()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="mesh")
    pt.geo_scene = scene
    print(f"Geo scene generated: {len(mesh.faces)} triangles, {np.round(mesh.extents, 1)} m "
          f"about {georef.origin}")
    return scene


def _triangulate_grid(pt, stride: int = 1, percentile: float = 0.0):
    """(vertices, faces, colors) from the per-pixel grid: each 2x2 quad is two triangles."""
    S, H0, W0 = pt.world_points.shape[:3]
    grid = pt.world_points[:, ::stride, ::stride]
    S, H, W = grid.shape[:3]
    verts = grid.reshape(-1, 3)
    colors = pt.flatten_colors().reshape(S, H0, W0, 3)[:, ::stride, ::stride].reshape(-1, 3)
    keep = pt.confidence_mask(percentile).reshape(S, H0, W0)[:, ::stride, ::stride].reshape(-1)

    idx = np.arange(S * H * W).reshape(S, H, W)
    a, b = idx[:, :-1, :-1], idx[:, :-1, 1:]
    c, d = idx[:, 1:, 1:], idx[:, 1:, :-1]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                            np.stack([a, c, d], -1).reshape(-1, 3)])
    faces = faces[keep[faces].all(1)]

    edges = np.maximum.reduce([np.linalg.norm(verts[faces[:, i]] - verts[faces[:, j]], axis=1)
                               for i, j in ((0, 1), (1, 2), (2, 0))])
    return verts, faces[edges < _MESH_EDGE_RATIO * np.median(edges)], colors

def _surface_material():
    """Non-metallic, two-sided PBR material; trimesh's defaults dim and cull the surface."""
    from trimesh.visual.material import PBRMaterial
    return PBRMaterial(baseColorFactor=[255, 255, 255, 255],
                       metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
