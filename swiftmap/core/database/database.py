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

from swiftmap.core.database.map import Map
from swiftmap.core.database.site import Site

MAPS_DIRNAME = "maps"
SITE_DIRNAME = "site"


class Database:
    """The result dir: every stored map under ``maps/``, plus the growing ``site/``."""

    def get_maps(self) -> List[Map]:
        """Stored maps, newest first."""

    def get_site(self) -> Site:
        """The growing site map (tag ``site``)."""

    def create_site(self, created: datetime = None) -> Site:
        """Create the site map under ``site/`` for the first time."""

    def create_map(self, created: datetime = None) -> Map:
        """Create an empty map under ``maps/`` for a run to write into."""

    def grow_site(self, new_map: Map, conf_thres: float = 50.0, voxel_size: float = 0.1,
                  created: datetime = None) -> Site:
        """Grow the site with a stored map."""
        return self.site.grow(new_map, conf_thres=conf_thres, voxel_size=voxel_size,
                              created=created)
