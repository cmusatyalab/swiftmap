# Copyright (C) 2024 Carnegie Mellon University
"""The area database: on-disk store of reconstructed/merged maps.

``Map`` is one stored area directory; ``session`` writes and merges results here per
the request. ``write_nfn_plan``/``write_segmented_objects`` serialize pipeline outputs
into a map directory.
"""

from swiftmap.core.database.map import Map, write_nfn_plan, write_segmented_objects

__all__ = ["Map", "write_nfn_plan", "write_segmented_objects"]
