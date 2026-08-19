# Copyright (C) 2024 Carnegie Mellon University

"""The ``Database``: the result dir, reached through the session.

    <root>/maps/   one dir per reconstructed batch (images, sky masks, GLB/PLY, NFN,
                   transform.json, map.json)
    <root>/site/   the ``Site`` -- same layout, grown by merging each stored map in

Storage only: create a map, grow the site, list/resolve tags. Rendering and
segmentation are pipeline stages that take a ``Map``.
"""

import os
from datetime import datetime
from typing import List, Optional

from swiftmap.database.map import Map
from swiftmap.database.site import Site

MAPS_DIRNAME = "maps"
SITE_DIRNAME = "site"


class Database:
    """The result dir: every stored map under ``maps/``, plus the growing ``site/``."""

    def __init__(self, root: str, site: str):
        self.root = root
        self.site_name = site
        self.maps_dir = os.path.join(root, MAPS_DIRNAME)
        self.site_dir = os.path.join(root, SITE_DIRNAME)
        os.makedirs(self.maps_dir, exist_ok=True)
        self.site = Site()

    def get_maps(self) -> List[Map]:
        """Stored maps, newest first."""

    def get_site(self) -> Site:
        """The growing site map (tag ``site``)."""
        return self.site

    def create_site(self, created: datetime = None) -> Site:
        """Create the site map under ``site/`` for the first time."""

    def create_map(self, created: datetime = None) -> Map:
        """Create an empty map under ``maps/`` for a run to write into."""
        created = created or datetime.now()
        tag = f"map_{created.strftime('%Y%m%d_%H%M%S_%f')}"
        path = os.path.join(self.maps_dir, tag)
        os.makedirs(path, exist_ok=True)
        return Map(name=tag, path=path)

    def grow_site(self, new_map: Map, conf_thres: float = 50.0, voxel_size: float = 0.1,
                  created: datetime = None) -> Site:
        """Grow the site with a stored map."""
        return self.site.grow(new_map, conf_thres=conf_thres, voxel_size=voxel_size,
                              created=created)
