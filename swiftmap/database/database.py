# Copyright (C) 2024 Carnegie Mellon University

"""The ``Database``: the result dir, reached through the session.

    <root>/maps/   one dir per reconstructed batch (images, sky masks, GLB/PLY, NFN,
                   transform.json, map.json); each NFN round mapped after it, in the
                   same layout, under its rounds/<k>/
    <root>/site/   the ``Site`` -- same layout, grown by merging each stored map in,
                   as its latest reconstructed round once its rounds are done

Storage only: create a map, grow the site, list/resolve tags. Rendering and
segmentation are pipeline stages that take a ``Map``.
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from swiftmap import constants
from swiftmap.database.map import Map
from swiftmap.database.site import Site

MAPS_DIRNAME = "maps"
SITE_DIRNAME = "site"
ROUNDS_DIRNAME = "rounds"


class Database:
    """The result dir: every stored map under ``maps/``, plus the growing ``site/``."""

    def __init__(self, root: str):
        self.root = root
        self.maps_dir = os.path.join(root, MAPS_DIRNAME)
        self.site_dir = os.path.join(root, SITE_DIRNAME)
        os.makedirs(self.maps_dir, exist_ok=True)
        self.site = Site(self.site_dir)
        self.maps: Dict[str, Map] = {}     # map id -> its latest round, the id everything is passed by
        self.rounds: Dict[str, List[Map]] = {}     # map id -> every round, the batch's first
        self._load_maps()
        self._load_site()

    def _load_site(self):
        """Pick up the site an earlier run left; needs the maps loaded to resolve its ids."""
        self.site.load(self.maps)

    def get_site(self) -> Site:
        """The growing site map, merged from every stored map."""
        return self.site

    def grow_site(self, map_id: str,
                  conf_threshold: float = constants.DEFAULT_CONF_THRESHOLD) -> bool:
        """Merge a stored map, as its latest reconstructed round, into the site; False if it
        has nothing to contribute."""
        map_ = next((m for m in reversed(self.get_rounds(map_id))
                     if m.get_pointcloud() is not None), None)
        return False if map_ is None else self.site.grow(map_, conf_threshold)
    
    def del_site(self) -> bool:
        """Delete the site from disk and start it over."""
        shutil.rmtree(self.site_dir, ignore_errors=True)
        self.site = Site(self.site_dir)
        return True

    def _load_maps(self):
        """Register the maps already under ``maps/``, so ids survive a restart."""
        if not os.path.isdir(self.maps_dir):
            return
        for tag in sorted(os.listdir(self.maps_dir)):
            path = os.path.join(self.maps_dir, tag)
            if not os.path.isdir(path):
                continue
            rounds_dir = os.path.join(path, ROUNDS_DIRNAME)
            rounds = sorted(int(k) for k in os.listdir(rounds_dir) if k.isdigit()) \
                if os.path.isdir(rounds_dir) else []
            self.rounds[tag] = [Map(name=tag, path=path).load()] + [
                Map(name=tag, path=os.path.join(rounds_dir, str(k))).load() for k in rounds]
            self.maps[tag] = self.rounds[tag][-1]

    def get_maps(self) -> List[Map]:
        """Stored maps, newest first."""
        return list(self.maps.values())[::-1]

    def get_map(self, map_id: str) -> Optional[Map]:
        """The latest round of the stored map with this id, or None if there is no such map."""
        return self.maps.get(map_id)

    def get_rounds(self, map_id: str) -> List[Map]:
        """Every round of the stored map with this id, the batch's first."""
        return self.rounds.get(map_id, [])
    
    def del_map(self, map_id: str) -> bool:
        """Delete a stored map from the registry, the site and disk; False if there is no such map."""
        map_ = self.maps.pop(map_id, None)
        if map_ is None:
            return False
        if self.site.drop(map_):
            self.site.write2disk()     # the merged cloud on disk no longer includes it
        shutil.rmtree(self.rounds.pop(map_id)[0].path, ignore_errors=True)
        return True

    def create_map(self, created: datetime = None) -> Map:
        """Create an empty map under ``maps/`` for a run to write into."""
        created = created or datetime.now()
        tag = f"map_{created.strftime('%Y%m%d_%H%M%S_%f')}"
        path = os.path.join(self.maps_dir, tag)
        os.makedirs(path, exist_ok=True)
        self.maps[tag] = Map(name=tag, path=path)
        self.rounds[tag] = [self.maps[tag]]
        return self.maps[tag]

    def create_round(self, map_id: str) -> Map:
        """Create an empty next round of a stored map, under its rounds/, as its latest."""
        rounds = self.rounds[map_id]
        path = os.path.join(rounds[0].path, ROUNDS_DIRNAME, str(len(rounds)))
        os.makedirs(path, exist_ok=True)
        rounds.append(Map(name=map_id, path=path))
        self.maps[map_id] = rounds[-1]
        return rounds[-1]

