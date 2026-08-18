# Copyright (C) 2024 Carnegie Mellon University

"""VGGT-Omega backbone adapter."""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from swiftmap.core import constants
from swiftmap.core.pipeline.reconstructor.base import BaseReconstructor
from swiftmap.core.pipeline.reconstructor.pose import camera_poses_from_extrinsics
from swiftmap.core.pipeline.reconstructor.registry import register_reconstructor
from swiftmap.database.types import PointCloud


def _unproject_depth_map_to_point_map(depth_map: np.ndarray,
                                      extrinsic: np.ndarray,
                                      intrinsic: np.ndarray) -> np.ndarray:
    """Unproject per-frame depth maps into world-space 3D points.

    depth_map (S,H,W,1)|(S,H,W); extrinsic (S,3,4) world-to-camera [R|t]; intrinsic
    (S,3,3). Returns (S,H,W,3) world points (the ``world_points_from_depth`` layout).
    """
    depth = depth_map[..., 0] if depth_map.ndim == 4 else depth_map
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)),
                     camera_points - translation[:, None, None, :])


@register_reconstructor(
    "vggt_omega",
    label="VGGT-Omega",
    description="VGGT-Omega (depth-based). Loads from a local checkpoint; "
                "world points are unprojected from predicted depth.",
)
class VGGTOmegaReconstructor(BaseReconstructor):
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
                            keyframe_paths: List[str]) -> PointCloud:
        from vggt_omega.utils.pose_enc import encoding_to_camera

        image_hw = predictions["images"].shape[-2:] if "images" in predictions else images.shape[-2:]
        extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], image_hw)
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # To numpy, squeezing the leading batch dim, as the Omega reference does.
        processed: Dict[str, Any] = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                arr = value.detach().float().cpu().numpy()
                processed[key] = arr[0] if arr.shape[0] == 1 else arr
            else:
                processed[key] = value

        # Omega has no point head: unproject depth into world_points_from_depth;
        # PointCloud.points_and_conf() falls back to it when world_points is unset.
        world_points_from_depth = _unproject_depth_map_to_point_map(
            processed["depth"], processed["extrinsic"], processed["intrinsic"])

        positions, rotations = camera_poses_from_extrinsics(processed["extrinsic"])
        return PointCloud(
            world_points_from_depth=world_points_from_depth,
            depth_conf=processed.get("depth_conf"),
            images=processed.get("images"),
            extrinsic=processed["extrinsic"],
            intrinsic=processed["intrinsic"],
            camera_positions=positions,
            camera_rotations=rotations,
            metadata={
                "keyframe_paths": keyframe_paths,
                "num_keyframes": len(keyframe_paths),
                "processing_timestamp": datetime.now(),
                "input_image_shape": images.shape,
            },
        )
