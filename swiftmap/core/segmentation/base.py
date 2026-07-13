# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic text-promptable segmentation interface.

A ``BaseSegmenter`` takes the frames a reconstruction backbone actually ran on
(VGGT's internal, resized images — pixel-aligned with its world-point map) plus a
text query like ``"person"`` and returns a per-frame boolean mask. Because the
masks live on the same (H, W) grid as ``world_points``, turning a mask into 3D
points is a direct index (see ``segmentation.lift``) — no camera projection.

Concrete backends (e.g. SAM 3) implement ``initialize_model`` and
``_segment_image``; the base class handles lazy init and the per-frame loop.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch


class BaseSegmenter(ABC):
    """Base class for text-promptable segmentation backends."""

    #: stable key, set by the ``@register_segmenter`` decorator.
    name: str = "base"

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_initialized = False
        print(f"{self.name} segmenter initialized on device: {self.device}")

    @abstractmethod
    def initialize_model(self) -> bool:
        """Load weights onto ``self.device``; set ``self.is_initialized``."""

    @abstractmethod
    def _segment_image(self, image_rgb: np.ndarray, query: str) -> np.ndarray:
        """Segment one RGB uint8 image (H, W, 3) for ``query``.

        Returns a single ``(H, W)`` boolean mask — the union of all detected
        instances of the concept (v1 is semantic; per-instance handling is a
        future extension).
        """

    def segment(self, images_rgb: List[np.ndarray], query: str) -> Optional[np.ndarray]:
        """Segment a list of RGB uint8 frames; return ``(S, H, W)`` bool masks.

        Frames must share (H, W) — they are VGGT's fixed-size internal images.
        Returns None if the model can't initialize.
        """
        if not self.is_initialized and not self.initialize_model():
            return None
        if not images_rgb:
            return None

        masks = []
        for i, img in enumerate(images_rgb):
            try:
                m = self._segment_image(img, query)
            except Exception as e:
                print(f"[{self.name}] frame {i} segmentation failed: {e}")
                m = np.zeros(img.shape[:2], dtype=bool)
            masks.append(np.asarray(m, dtype=bool))
        return np.stack(masks, axis=0)
