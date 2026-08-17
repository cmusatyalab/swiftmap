# Copyright (C) 2024 Carnegie Mellon University

"""The ``Site``: a database's one growing map."""

import glob
import os
from datetime import datetime

from swiftmap.core.database.map import Map
from swiftmap.core.primitives.types import MapData


class Site(Map):
    """A ``Map`` that grows: each stored map is merged into it, in place."""

    def __init__(self, path: str, name: str = "map"):
        super().__init__(path)
        self.name = name

    def __repr__(self):
        return f"Site({self.name!r}, {len(self.sources)} map(s))"

    @property
    def sources(self) -> list:
        """Tags of the maps merged in so far."""
        return list(self.metadata.source_maps)

    def grow(self, new_map: Map, conf_thres: float = 50.0, voxel_size: float = 0.1,
             created: datetime = None) -> "Site":
        """Merge ``new_map`` into the site (origin pinned to the site) and rewrite its data.

        Previews are a pipeline concern: the caller runs ``renderer.write_previews``."""
        parts = [self.load()] if self.exists() else []
        parts.append(new_map.load(conf_thres))
        merged = MapData.merge(parts, voxel_size=voxel_size)
        o = merged.origin
        print(f"[site] grow with '{new_map.tag}': {sum(len(p) for p in parts):,} -> "
              f"{len(merged):,} pts (voxel {voxel_size:g} m, conf>={conf_thres:g}p, "
              f"origin {o[0]:.6f},{o[1]:.6f},{o[2]:.1f})")
        for stale in glob.glob(os.path.join(self.path, "*_view_c*.glb")):
            os.remove(stale)
        return self.write(merged, self.sources + [new_map.tag], site=self.name, created=created)
