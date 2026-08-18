# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic reconstructor interface.

``BaseReconstructor`` defines the contract SwiftMap depends on and implements
everything that does not depend on the specific reconstruction model:
orchestration (``run``), which delegates all output -- scene.glb,
confidence_map.glb, camera_poses.json -- to ``postprocess``.

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

from swiftmap.core.pipeline.reconstructor import postprocess


class BaseReconstructor(ABC):
    """Base class for reconstruction backbones. Registered via ``register_reconstructor``."""

    #: stable key, set by the ``@register_reconstructor`` decorator.
    name: str = "base"

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_initialized = False

        self.default_params = {
            "mask_sky": True,
            "mask_dynamic": False,
            "conf_threshold": 60.0,
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
                            keyframe_paths: List[str]) -> Dict[str, Any]:
        """Convert raw predictions to the normalized numpy schema."""

    # ---------------------------------------------------------- orchestration
    def run(self, keyframe_paths: List[str],
           processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full backbone -> 3D content -> confidence pipeline."""
        if not self.is_initialized:
            if not self.initialize_model():
                return {"success": False, "error": "Model initialization failed"}
        if not keyframe_paths:
            return {"success": False, "error": "No keyframes provided"}

        params = self.default_params.copy()
        if processing_params:
            params.update(processing_params)

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
            processed = self._decode_predictions(raw_predictions, images, keyframe_paths)
            postprocess_time = time.time() - postprocess_start

            generation_start = time.time()
            scene_results = postprocess.generate_3d_scene(processed, params)
            generation_time = time.time() - generation_start

            confidence_results: Dict[str, Any] = {}
            if scene_results.get("target_directory"):
                target_dir = scene_results["target_directory"]
                try:
                    conf_start = time.time()
                    confidence_results = postprocess.generate_confidence_scene(
                        processed, params, target_dir)
                    confidence_results["generation_time"] = time.time() - conf_start
                except Exception as e:
                    print(f"Warning: Confidence mapping generation failed: {e}")

                try:
                    scene_results["camera_poses_path"] = postprocess.generate_camera_poses(
                        processed, target_dir, self.name)
                except Exception as e:
                    print(f"Warning: Camera pose export failed: {e}")

            total_processing_time = time.time() - processing_start

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[{self.name}] processing completed in {total_processing_time:.2f}s")
            return {
                "success": True,
                "backbone": self.name,
                "keyframe_count": len(keyframe_paths),
                "processing_params": params,
                "predictions": processed,
                "scene_results": scene_results,
                "confidence_results": confidence_results,
                "timing": {
                    "total_processing": total_processing_time,
                    "preprocessing": preprocess_time,
                    "inference": inference_time,
                    "postprocessing": postprocess_time,
                    "3d_generation": generation_time,
                },
            }
        except Exception as e:
            print(f"Error during {self.name} processing: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}