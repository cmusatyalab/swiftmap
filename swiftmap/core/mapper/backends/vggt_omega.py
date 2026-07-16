# Copyright (C) 2024 Carnegie Mellon University

"""VGGT-Omega backbone adapter.

Wraps VGGT-Omega behind ``BaseMapper``. Unlike VGGT, Omega:
  * loads from a local checkpoint (.pt) via ``torch.load`` + ``load_state_dict``
    (no hub URL) — see ``constants.VGGT_OMEGA_CHECKPOINT``;
  * preprocesses at a configurable ``image_resolution`` (patch size 16);
  * decodes pose with ``encoding_to_camera`` (not ``pose_encoding_to_extri_intri``);
  * predicts depth only (no point head), so world points are recovered by
    unprojecting depth (``geometry.unproject_depth_map_to_point_map``) into
    ``world_points_from_depth``.

The ``vggt_omega`` package (import name ``vggt_omega``, distinct from ``vggt``)
is imported lazily so it is only loaded when this backbone is selected.
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from swiftmap.core import constants
from swiftmap.core.mapper.base import BaseMapper
from swiftmap.core.mapper.geometry import (
    camera_poses_from_extrinsics, unproject_depth_map_to_point_map)
from swiftmap.core.mapper.registry import register_mapper


@register_mapper(
    "vggt_omega",
    label="VGGT-Omega",
    description="VGGT-Omega (depth-based). Loads from a local checkpoint; "
                "world points are unprojected from predicted depth.",
)
class VGGTOmegaMapper(BaseMapper):
    """VGGT-Omega reconstruction backbone."""

    def __init__(self, device=None,
                 checkpoint_path: str = None,
                 image_resolution: int = None):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path or constants.VGGT_OMEGA_CHECKPOINT
        self.image_resolution = image_resolution or constants.VGGT_OMEGA_IMAGE_RESOLUTION

    def initialize_model(self) -> bool:
        try:
            import os
            from vggt_omega.models import VGGTOmega

            if not os.path.isfile(self.checkpoint_path):
                raise FileNotFoundError(
                    f"VGGT-Omega checkpoint not found: {self.checkpoint_path}. "
                    f"Set the VGGT_OMEGA_CHECKPOINT env var or pass checkpoint_path.")

            print(f"Initializing VGGT-Omega from {self.checkpoint_path} ...")
            model = VGGTOmega().eval()
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            self.model = model.to(self.device)
            self.is_initialized = True
            print("VGGT-Omega model initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing VGGT-Omega model: {e}")
            self.is_initialized = False
            return False

    def _load_and_preprocess(self, keyframe_paths: List[str]) -> torch.Tensor:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        return load_and_preprocess_images(
            keyframe_paths, image_resolution=self.image_resolution).to(self.device)

    def _infer(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            return self.model(images)

    def _decode_predictions(self, predictions: Dict[str, torch.Tensor],
                            images: torch.Tensor,
                            keyframe_paths: List[str]) -> Dict[str, Any]:
        from vggt_omega.utils.pose_enc import encoding_to_camera

        image_hw = predictions["images"].shape[-2:] if "images" in predictions else images.shape[-2:]
        extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], image_hw)
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # To numpy, squeezing the leading batch dim (batch size 1), matching the
        # Omega reference pipeline.
        processed: Dict[str, Any] = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                arr = value.detach().float().cpu().numpy()
                processed[key] = arr[0] if arr.shape[0] == 1 else arr
            else:
                processed[key] = value

        # Omega has no point head: recover world points from depth and expose
        # them under the standard keys so the exporter, confidence map, and NFN
        # planner all work uniformly (same schema as VGGT's point head). Keep the
        # explicit *_from_depth alias too for clarity/provenance.
        world_points = unproject_depth_map_to_point_map(
            processed["depth"], processed["extrinsic"], processed["intrinsic"])
        processed["world_points_from_depth"] = world_points
        processed["world_points"] = world_points
        if "depth_conf" in processed:
            processed["world_points_conf"] = processed["depth_conf"]

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
