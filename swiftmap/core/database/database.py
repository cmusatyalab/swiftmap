# Copyright (C) 2024 Carnegie Mellon University

"""The ``Database``: the result dir, and the interface the server and frontend use.

    <root>/maps/   one dir per reconstructed batch (images, sky masks, GLB/PLY, NFN,
                   transform.json, map.json)
    <root>/site/   the ``Site`` -- same layout, grown by merging each stored map in

Storage only: store a batch, grow the site, list/resolve tags. Rendering and
segmentation are pipeline stages that take a ``Map``.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional

from swiftmap.core.database.map import Map
from swiftmap.core.database.site import Site

MAPS_DIRNAME = "maps"
SITE_DIRNAME = "site"


class Database:
    """The result dir: every stored map under ``maps/``, plus the growing ``site/``."""

    def __init__(self, root: str, site_name: str = "map"):
        self.root = os.path.abspath(root)
        self.site = Site(os.path.join(self.root, SITE_DIRNAME), site_name)

    def __repr__(self):
        return f"Database({self.root!r}, {len(self.maps())} map(s))"

    @property
    def maps_dir(self) -> str:
        return os.path.join(self.root, MAPS_DIRNAME)

    def maps(self) -> List[Map]:
        """Stored maps, newest first."""
        return Map.list(self.maps_dir)

    def tags(self) -> List[str]:
        """The site (when it exists) first, then every stored map."""
        return ([self.site.tag] if self.site.exists() else []) + [m.tag for m in self.maps()]

    def get(self, tag: str) -> Optional[Map]:
        """The site (tag ``site``) or a stored map by tag."""
        if tag == self.site.tag:
            return self.site if self.site.exists() else None
        return Map.get(self.maps_dir, tag)

    def store(self, batch_dir: str, created: datetime = None) -> Map:
        """Move a reconstructed batch into ``maps/`` and stamp it as a stored map."""
        created = created or datetime.now()
        os.makedirs(self.maps_dir, exist_ok=True)
        stored = Map(os.path.join(self.maps_dir, Map.tag_for("map", created)))
        shutil.move(batch_dir, stored.path)
        stored.stamp_metadata(self.site.name, created)
        print(f"[db] stored map '{stored.tag}' ({len(stored.frames)} cameras)")
        return stored

    def grow(self, new_map: Map, conf_thres: float = 50.0, voxel_size: float = 0.1,
             created: datetime = None) -> Site:
        """Grow the site with a stored map."""
        return self.site.grow(new_map, conf_thres=conf_thres, voxel_size=voxel_size,
                              created=created)

    def delete(self, tag: str) -> bool:
        """Delete a stored map (the site is never deleted here)."""
        m = Map.get(self.maps_dir, tag)
        if m is None:
            return False
        m.delete()
        return True
