# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic reconstructor interface.

A concrete backbone implements just the four model-specific steps:
    initialize_model()      load weights
    _load_and_preprocess()  images on device, sized for this model
    _infer()                forward pass -> raw prediction dict
    _decode_predictions()   raw dict -> normalized numpy schema (see postprocess)
"""

import gc
import time
from abc import ABC, abstractmethod

from typing import Any, Dict, List, Optional

import torch

from swiftmap.database.map import Map
from swiftmap.database.types import PointCloud
from swiftmap.core.pipeline.reconstructor import postprocess
from swiftmap import constants


class BaseReconstructor(ABC):
    """Base class for reconstruction backbones. Registered via ``register_reconstructor``."""

    #: stable key, set by the ``@register_reconstructor`` decorator.
    name: str = "base"

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_initialized = False

        self.default_params = {
            "mask_sky": False,
            "mask_dynamic": False,
            "conf_threshold": constants.DEFAULT_CONF_THRESHOLD,
            "mask_black_bg": False,
            "mask_white_bg": False,
            "show_cam": True,
        }

        print(f"{self.name} reconstructor initialized on device: {self.device}")

    # ---------------------------------------------------------------- backbone API
    @abstractmethod
    def initialize_model(self) -> bool:
        """Load model weights onto ``self.device``; set ``self.is_initialized``."""

    @abstractmethod
    def _load_and_preprocess(self, keyframe_paths: List[str]) -> torch.Tensor:
        """Load + preprocess keyframes into a model-ready image tensor on device."""

    @abstractmethod
    def _infer(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the forward pass; return the raw prediction dict."""

    @abstractmethod
    def _decode_predictions(self, predictions: Dict[str, torch.Tensor],
                            images: torch.Tensor,
                            keyframe_paths: List[str]) -> PointCloud:
        """Convert raw predictions to the normalized numpy schema."""

    # ---------------------------------------------------------- orchestration
    def run(self, map: Map, processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full backbone -> 3D content -> confidence pipeline."""
        keyframe_paths = map.get_keyframe_paths()

        if not self.is_initialized:
            if not self.initialize_model():
                return {"success": False, "error": "Model initialization failed"}
        if not keyframe_paths:
            return {"success": False, "error": "No keyframes provided"}

        params = self.default_params.copy()
        if processing_params:
            params.update(processing_params)
        params["backbone"] = self.name

        try:
            print(f"[{self.name}] Processing {len(keyframe_paths)} keyframes...")
            print(f"Parameters: {params}")
            processing_start = time.time()

            preprocess_start = time.time()
            images = self._load_and_preprocess(keyframe_paths)
            preprocess_time = time.time() - preprocess_start
            print(f"Loaded {len(keyframe_paths)} keyframes: {tuple(images.shape)}")

            inference_start = time.time()
            with torch.no_grad():
                raw_predictions = self._infer(images)
            inference_time = time.time() - inference_start

            postprocess_start = time.time()
            reconstruction = self._decode_predictions(raw_predictions, images, keyframe_paths)
            postprocess_time = time.time() - postprocess_start

            # add the point cloud to the map
            map.update_reconstruction(reconstruction)

            generation_start = time.time()

            # update the 3d scene result in the maps reconstruction field
            postprocess.generate_3d_scene(map, params)
            generation_time = time.time() - generation_start


            # update the confidence map result in the maps reconstruction field
            conf_generate_start = time.time()
            postprocess.generate_confidence_scene(map, params)
            conf_generate_time = time.time() - conf_generate_start

            # update model_input
            postprocess.generate_model_input(map)

            total_processing_time = time.time() - processing_start


            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # logging the time
            print(f"[{self.name}] Preprocessing time: {preprocess_time:.2f}s")
            print(f"[{self.name}] Inference time: {inference_time:.2f}s")
            print(f"[{self.name}] Postprocessing time: {postprocess_time:.2f}s")
            print(f"[{self.name}] 3D scene generation time: {generation_time:.2f}s")
            print(f"[{self.name}] Confidence map generation time: {conf_generate_time:.2f}s")
            print(f"[{self.name}] processing completed in {total_processing_time:.2f}s")

            return {
                "success": True,
                "backbone": self.name,
                "keyframe_count": len(keyframe_paths),
                "processing_params": params,
                "timing": {
                    "total_processing": total_processing_time,
                    "preprocessing": preprocess_time,
                    "inference": inference_time,
                    "postprocessing": postprocess_time,
                    "3d_generation": generation_time,
                    "confidence_generation": conf_generate_time,
                },
            }

        except Exception as e:
            print(f"Error during {self.name} processing: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}