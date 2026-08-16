# Copyright (C) 2024 Carnegie Mellon University

"""VGGT backbone adapter.

Wraps facebook/VGGT-1B behind ``BaseMapper``. VGGT predicts world points
directly (dedicated point head) and encodes camera pose as ``pose_enc`` decoded
with ``pose_encoding_to_extri_intri``. The heavy ``vggt`` package is imported
lazily inside methods so selecting a different backbone never imports it.
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from swiftmap.core import constants
from swiftmap.core.pipeline.reconstructor.base import BaseMapper
from swiftmap.core.pipeline.reconstructor.postprocess import camera_poses_from_extrinsics
from swiftmap.core.pipeline.reconstructor.registry import register_mapper


@register_mapper(
    "vggt",
    label="VGGT-1B",
    description="Visual Geometry Grounded Transformer (facebook/VGGT-1B). "
                "Predicts world points directly via a point head.",
)
class VGGTMapper(BaseMapper):
    """VGGT reconstruction backbone."""

    def initialize_model(self) -> bool:
        try:
            print("Initializing VGGT model...")
            from vggt.models.vggt import VGGT
            self.model = VGGT()
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(constants.VGGT_MODEL_URL))
            self.model.eval()
            self.model = self.model.to(self.device)
            self.is_initialized = True
            print("VGGT model initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing VGGT model: {e}")
            self.is_initialized = False
            return False

    def _load_and_preprocess(self, keyframe_paths: List[str]) -> torch.Tensor:
        from vggt.utils.load_fn import load_and_preprocess_images
        return load_and_preprocess_images(keyframe_paths).to(self.device)

    def _infer(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            return self.model(images)

    def _decode_predictions(self, predictions: Dict[str, torch.Tensor],
                            images: torch.Tensor,
                            keyframe_paths: List[str]) -> Dict[str, Any]:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        processed: Dict[str, Any] = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.cpu().numpy().squeeze(0)
            else:
                processed[key] = value

        positions, rotations = camera_poses_from_extrinsics(processed["extrinsic"])
        processed["camera_positions"] = positions
        processed["camera_rotations"] = rotations
        processed["metadata"] = {
            "keyframe_paths": keyframe_paths,
            "num_keyframes": len(keyframe_paths),
            "processing_timestamp": datetime.now(),
            "input_image_shape": images.shape,
        }
        return processed
