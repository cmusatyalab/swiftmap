# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic text-promptable segmentation interface.

A ``BaseSegmenter`` takes the frames a reconstruction backbone actually ran on
(VGGT's internal, resized images — pixel-aligned with its world-point map) plus a
text query like ``"person"`` and returns a per-frame boolean mask. Because the
masks live on the same (H, W) grid as ``world_points``, turning a mask into 3D
points is a direct index -- no camera projection.

Concrete backends (e.g. SAM 3) implement ``initialize_model`` and
``_segment_image``; the base class handles lazy init and the per-frame loop.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from swiftmap import constants
from swiftmap.core.pipeline.segmentor import postprocess
from swiftmap.database.map import Map


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

    # ---------------------------------------------------------- orchestration
    def run(self, map: Map, processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Segment one query over the Map's frames and lift the matching pixels into 3D."""
        params = {"query": "", "conf_threshold": constants.DEFAULT_CONF_THRESHOLD}
        if processing_params:
            params.update(processing_params)

        query = (params["query"] or "").strip()
        if not query:
            return {"error": "Enter a segmentation query (e.g. 'person')."}
        pt = map.get_pointcloud()
        if pt is None or pt.world_points is None or pt.images is None:
            return {"error": "No reconstruction to segment"}

        segment_start = time.time()
        masks = self._segment(self._reshape_images(pt.images), query)
        if masks is None:
            return {"error": f"{self.name} failed to initialize"}

        # world_points is pixel-aligned with the frames, so the mask indexes 3D directly.
        keep = masks.reshape(-1) & pt.confidence_mask(params["conf_threshold"])
        target = pt.add_segmentation(query, pt.flatten_points()[keep])
        postprocess.generate_seg_scene(map, params)

        segment_time = time.time() - segment_start
        print(f"[{self.name}] '{query}': {len(target.points)} points from "
              f"{int(masks.sum())} pixels in {segment_time:.2f}s")
        return {"success": True, "query": query, "num_points": int(len(target.points)),
                "targets": [t.query for t in pt.segmented_worldpoints],
                "timing": {"segmentation": segment_time}}

    @staticmethod
    def _reshape_images(images: np.ndarray) -> List[np.ndarray]:
        """VGGT's internal frames as RGB uint8 (H, W, 3) -- the grid world_points lives on."""
        images = np.asarray(images)
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        if images.dtype != np.uint8:
            images = (np.clip(images, 0.0, 1.0) * 255).astype(np.uint8)
        return [np.ascontiguousarray(img) for img in images]

    def _segment(self, images_rgb: List[np.ndarray], query: str) -> Optional[np.ndarray]:
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
