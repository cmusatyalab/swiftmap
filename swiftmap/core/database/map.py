# Copyright (C) 2024 Carnegie Mellon University

"""The ``Map``: one stored map directory in the database.

Holds a raw ``predictions.npz`` (a reconstructed batch) or a merged ``merged_points.npz``
plus transform/camera/map json. It is the store record -- identity, metadata, ``load``
(-> arrays) and ``write`` (arrays -> dir); previews and segmentation come from the pipeline."""


class Map:
    """An on-disk map directory (tag + path) with load/render/segment/merge I/O."""

