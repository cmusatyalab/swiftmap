# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic text-promptable segmentation interface.

A ``BaseSegmenter`` takes a text query like ``"person"`` and runs it over the frames
every map in the site was reconstructed from (VGGT's internal, resized images --
pixel-aligned with its world-point map), returning one mask per detected instance.
Because the masks live on the same (H, W) grid as ``world_points``, turning a mask
into 3D points is a direct index -- no camera projection. Each instance mask is
lifted into the site's shared ENU frame by its map's Georeference, giving a fragment,
and fragments of the same object across views and maps are associated into instances
(see ``instances``). Targets accumulate on the Site across queries.

Concrete backends (e.g. SAM 3) implement ``initialize_model`` and
``_segment_image``; the base class handles lazy init and the per-frame loop.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from swiftmap import constants
from swiftmap.core.pipeline.segmentor.instances import Fragment, associate
from swiftmap.database.site import Site

Instances = List[Tuple[np.ndarray, float]]


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
    def _segment_image(self, image_rgb: np.ndarray, query: str) -> Instances:
        """Segment one RGB uint8 image (H, W, 3) for ``query``.

        Returns each detected instance of the concept as its ``(H, W)`` boolean mask
        and detection score; empty when nothing matches.
        """

    # ---------------------------------------------------------- orchestration
    def run(self, site: Site, processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Segment one query over every map's frames and lift the matches into the site."""
        params = {"query": "", "conf_threshold": constants.DEFAULT_CONF_THRESHOLD}
        if processing_params:
            params.update(processing_params)

        query = (params["query"] or "").strip()
        if not query:
            return {"error": "Enter a segmentation query (e.g. 'person')."}
        if site.origin is None or not site.maps:
            return {"error": "No site to segment"}

        segment_start = time.time()
        lla0 = (site.origin.latitude, site.origin.longitude, site.origin.altitude or 0.0)
        fragments = []
        for batch, map_ in enumerate(site.maps):
            pt, georef = map_.get_pointcloud(), map_.get_georeference()
            if pt is None or pt.world_points is None or pt.images is None or georef is None:
                continue
            frames = self._segment(self._reshape_images(pt.images), query)
            if frames is None:
                return {"error": f"{self.name} failed to initialize"}
            reliable =pt.confidence_mask(params["conf_threshold"]).reshape(len(frames), -1)
            points = pt.world_points.reshape(len(frames), -1, 3)
            conf = pt.world_points_conf.reshape(len(frames), -1)
            for frame, found in enumerate(frames):
                for mask, score in found:
                    keep = mask.reshape(-1) & reliable[frame]
                    if keep.any():
                        fragments.append(Fragment(georef.to_enu(points[frame][keep], lla0),
                                                  conf[frame][keep], score, batch, frame))

        target = site.add_segmentation(query, associate(fragments, site))
        segment_time = time.time() - segment_start
        print(f"[{self.name}] '{query}': {len(target.instances)} instance(s) from "
              f"{len(fragments)} fragment(s) across {len(site.maps)} map(s) "
              f"in {segment_time:.2f}s")
        centres = site.origin.enu_to_lla([inst.centroid for inst in target.instances])
        return {"success": True, "query": query, "num_fragments": len(fragments),
                "instances": [{"latitude": lat, "longitude": lon, "altitude": alt,
                               "support": inst.support, "score": inst.score,
                               "points": len(inst.points)}
                              for inst, (lat, lon, alt) in zip(target.instances, centres)],
                "targets": [t.query for t in site.segments],
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

    def masks(self, map_, query: str) -> Optional[np.ndarray]:
        """``(S, H, W)`` masks of ``query`` over one map's frames, each the union of its
        instances; None if the map has no frames or the model can't initialize."""
        pt = map_.get_pointcloud()
        if pt is None or pt.images is None:
            return None
        images = self._reshape_images(pt.images)
        frames = self._segment(images, query)
        if frames is None:
            return None
        masks = np.zeros((len(images), *images[0].shape[:2]), dtype=bool)
        for mask, found in zip(masks, frames):
            for m, _ in found:
                mask |= m
        return masks

    def _segment(self, images_rgb: List[np.ndarray], query: str) -> Optional[List[Instances]]:
        """Segment a list of RGB uint8 frames; per frame, the instances found.

        Frames must share (H, W) — they are VGGT's fixed-size internal images. A frame
        that fails contributes no instances. Returns None if the model can't initialize.
        """
        if not self.is_initialized and not self.initialize_model():
            return None

        frames = []
        for i, img in enumerate(images_rgb):
            try:
                found = [(np.asarray(m, dtype=bool), float(s))
                         for m, s in self._segment_image(img, query)]
            except Exception as e:
                print(f"[{self.name}] frame {i} segmentation failed: {e}")
                found = []
            frames.append(found)
        return frames
