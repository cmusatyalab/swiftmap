# Copyright (C) 2024 Carnegie Mellon University
"""Segmentation backends. Importing this package registers each segmenter."""
from swiftmap.core.semantic.backends.sam3 import SAM3Segmenter

__all__ = ["SAM3Segmenter"]
