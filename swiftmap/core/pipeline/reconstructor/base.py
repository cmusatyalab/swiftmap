# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic mapper interface.

``BaseMapper`` defines the contract SwiftMap depends on and implements
everything that does not depend on the specific reconstruction model:
orchestration (``process_keyframes``), 3D/confidence export (delegated to
``postprocess``), result storage, and camera-pose serialization.

A concrete backbone implements just the four model-specific steps:
    initialize_model()      load weights
    _load_and_preprocess()  images on device, sized for this model
    _infer()                forward pass -> raw prediction dict
    _decode_predictions()   raw dict -> normalized numpy schema (see postprocess)
"""

import gc
import json
import os
import time
from abc import ABC, abstractmethod

from swiftmap.core.primitives.types import Reconstruction
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from swiftmap.core.pipeline.reconstructor import postprocess


class BaseMapper(ABC):
    """Base class for reconstruction backbones. Registered via ``register_mapper``."""

    #: stable key, set by the ``@register_mapper`` decorator.
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

        # Latest-run storage (read back by the session / UI).
        self.latest_predictions = None
        self.latest_scene = None
        self.latest_confidence_scene = None
        self.latest_scene_results: Dict[str, Any] = {}
        self.latest_confidence_results: Dict[str, Any] = {}
        self.latest_keyframes_info = None

        print(f"{self.name} mapper initialized on device: {self.device}")

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
        """Convert raw predictions to the normalized numpy schema.

        Must populate ``extrinsic``, ``intrinsic``, a point field
        (``world_points`` and/or ``world_points_from_depth``), ``images``,
        ``camera_positions`` and a ``metadata`` dict.
        """

    # ---------------------------------------------------------- orchestration
    def process_keyframes(self, keyframe_paths: List[str],
                          processing_params: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
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
            scene_results = postprocess.generate_3d_content(processed, params)
            generation_time = time.time() - generation_start

            confidence_results: Dict[str, Any] = {}
            if scene_results.get("target_directory"):
                try:
                    conf_start = time.time()
                    confidence_results = postprocess.generate_confidence_mapping(
                        processed, params, scene_results["target_directory"])
                    confidence_results["generation_time"] = time.time() - conf_start
                except Exception as e:
                    print(f"Warning: Confidence mapping generation failed: {e}")

            total_processing_time = time.time() - processing_start

            self.latest_predictions = Reconstruction(processed)
            self.latest_scene = scene_results.get("scene")
            self.latest_confidence_scene = confidence_results.get("scene")
            self.latest_scene_results = scene_results
            self.latest_confidence_results = confidence_results
            self.latest_keyframes_info = {
                "keyframe_paths": keyframe_paths,
                "processing_params": params,
                "processing_time": total_processing_time,
                "backbone": self.name,
            }

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

    # ------------------------------------------------------------------- shared
    def _generate_confidence_mapping(self, predictions: Dict[str, Any],
                                     params: Dict[str, Any],
                                     target_dir: str) -> Dict[str, Any]:
        """Kept as a method for callers that regenerate the confidence map."""
        return postprocess.generate_confidence_mapping(predictions, params, target_dir)

    def get_latest_results(self) -> Dict[str, Any]:
        if self.latest_predictions is None:
            return {"error": "No processing results available"}
        return {
            "predictions": self.latest_predictions,
            "scene": self.latest_scene,
            "confidence_scene": self.latest_confidence_scene,
            "scene_results": self.latest_scene_results,
            "confidence_results": self.latest_confidence_results,
            "keyframes_info": self.latest_keyframes_info,
        }

    def save_camera_poses_json(self, output_path: str) -> bool:
        if self.latest_predictions is None:
            print("No predictions available to save")
            return False
        try:
            predictions = self.latest_predictions
            keyframe_paths = self.latest_keyframes_info["keyframe_paths"]
            extrinsic = predictions["extrinsic"]
            intrinsic = predictions["intrinsic"]

            poses_data = {
                "metadata": {
                    "description": "Camera poses from SwiftMap Mapping",
                    "backbone": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "num_keyframes": len(keyframe_paths),
                    "processing_params": self.latest_keyframes_info["processing_params"],
                },
                "frames": [],
            }
            for i, (ext, intr) in enumerate(zip(extrinsic, intrinsic)):
                image_name = (os.path.basename(keyframe_paths[i])
                              if i < len(keyframe_paths) else f"Keyframe_{i}")
                R = ext[:3, :3]
                t = ext[:3, 3]
                world_pos = -R.T @ t
                poses_data["frames"].append({
                    "image_name": image_name,
                    "camera_position_world": world_pos.tolist(),
                    "rotation_matrix": R.tolist(),
                    "translation_vector": t.tolist(),
                    "intrinsic_matrix": intr.tolist(),
                    "extrinsic_matrix": ext.tolist(),
                })
            with open(output_path, "w") as f:
                json.dump(poses_data, f, indent=2)
            print(f"Camera poses saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving camera poses: {e}")
            return False

    def clear_results(self):
        self.latest_predictions = None
        self.latest_scene = None
        self.latest_confidence_scene = None
        self.latest_scene_results = {}
        self.latest_confidence_results = {}
        self.latest_keyframes_info = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Results cleared")
