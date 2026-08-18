# Copyright (C) 2024 Carnegie Mellon University
"""The map database: on-disk store of reconstructed and merged maps.

``Database`` is a results root holding ``maps/`` (every generated map, one per
reconstructed batch) and ``site/`` (the ``Site`` -- the one growing map, the merge of the
stored maps). ``Map`` is one stored map directory. Rendering/segmentation live in the pipeline.
"""

from swiftmap.database.map import Map
from swiftmap.database.site import Site
from swiftmap.database.database import Database

__all__ = ["Database", "Map", "Site"]
