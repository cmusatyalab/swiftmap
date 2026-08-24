# Copyright (C) 2024 Carnegie Mellon University

"""Builds seg_scene.glb; Map.write2disk() exports it."""

from typing import Any, Dict, Optional

import numpy as np
import trimesh

from swiftmap.core.pipeline.reconstructor.postprocess import _align_scene
from swiftmap.database.map import Map

_BACKGROUND_DIM = 0.35   # segmented targets should read against a muted scene


def generate_seg_scene(map: Map, params: Dict[str, Any]) -> Optional[trimesh.Scene]:
    """Build the dimmed scene with every segmented target in its own colour and node name."""
    pt = map.get_pointcloud()
    if pt is None or not pt.segmented_worldpoints:
        return None

    keep = pt.confidence_mask(params.get("conf_threshold", 0.0))
    background = (pt.flatten_colors()[keep] * _BACKGROUND_DIM).astype(np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=pt.flatten_points()[keep], colors=background),
                       geom_name="scene")
    for target in pt.segmented_worldpoints:
        if not len(target.points):
            continue
        colors = np.tile(target.color, (len(target.points), 1))
        scene.add_geometry(trimesh.PointCloud(vertices=target.points, colors=colors),
                           geom_name=target.query)   # node name is the viewer's label

    _align_scene(scene, pt.extrinsic)
    pt.segmented_scene = scene
    return scene
