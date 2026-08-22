# Copyright (C) 2024 Carnegie Mellon University

"""VGGT backbone adapter."""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from swiftmap import constants
from swiftmap.core.pipeline.reconstructor.base import BaseReconstructor
from swiftmap.core.pipeline.reconstructor.registry import register_reconstructor
from swiftmap.database.types import CameraPose, PointCloud


@register_reconstructor(
    "vggt",
    label="VGGT-1B",
    description="Visual Geometry Grounded Transformer (facebook/VGGT-1B). "
                "Predicts world points directly via a point head.",
)
class VGGTReconstructor(BaseReconstructor):
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
                            keyframe_paths: List[str]) -> PointCloud:
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

        return PointCloud(
            world_points=processed.get("world_points"),
            world_points_conf=processed.get("world_points_conf"),
            world_points_from_depth=processed.get("world_points_from_depth"),
            depth_conf=processed.get("depth_conf"),
            images=processed.get("images"),
            extrinsic=processed["extrinsic"],
            intrinsic=processed["intrinsic"],
            cameras=[CameraPose.from_extrinsic(e) for e in processed["extrinsic"]],
            metadata={
                "keyframe_paths": keyframe_paths,
                "num_keyframes": len(keyframe_paths),
                "processing_timestamp": datetime.now(),
                "input_image_shape": images.shape,
            },
        )
