# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic post-processing shared by all mapper backends.

Once a backbone has produced a *normalized* prediction dict (numpy arrays with
``extrinsic``/``intrinsic``, a point field, ``images``, ``depth`` ...), the 3D
export, PLY/GLB generation, and confidence-map generation are identical
regardless of which model produced them. That shared logic lives here so each
backend only implements load / preprocess / infer / decode.

Normalized prediction schema (keys consumed downstream):
    images                 (S, 3, H, W)
    extrinsic              (S, 3, 4)      world-to-camera
    intrinsic              (S, 3, 3)
    world_points           (S, H, W, 3)   optional (dedicated point head)
    world_points_conf      (S, H, W)      optional, pairs with world_points
    world_points_from_depth(S, H, W, 3)   optional (unprojected depth)
    depth / depth_conf     (S, H, W, 1)/(S, H, W)
    camera_positions       (S, 3)
    metadata               dict
"""

import os
import shutil
from typing import Any, Dict

import numpy as np
import torch

from swiftmap.core.pipeline.reconstructor.scene_export import predictions_to_glb

try:
    from swiftmap.core.pipeline.reconstructor.confidence_mapping import generate_confidence_point_cloud
    CONFIDENCE_MAPPING_AVAILABLE = True
except ImportError:  # pragma: no cover - optional viz deps
    CONFIDENCE_MAPPING_AVAILABLE = False
    print("Warning: Confidence mapping utilities not available")


def camera_poses_from_extrinsics(extrinsic: np.ndarray):
    """(positions, rotations) in world coords from (S,3,4) extrinsics: ``-R^T t``, ``R^T``."""
    positions, rotations = [], []
    for ext in extrinsic:
        R = ext[:3, :3]
        t = ext[:3, 3]
        positions.append(-R.T @ t)
        rotations.append(R.T)
    return np.array(positions), np.array(rotations)


def ensure_cpu_tensors(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert any torch tensors to CPU numpy arrays."""
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            result[key] = ensure_cpu_tensors(value)
        elif isinstance(value, list):
            result[key] = [
                item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _point_conf_keys(predictions: Dict[str, Any]):
    """Pick the (points, confidence) key pair present for this backbone.

    Prefers a dedicated point head (``world_points``); falls back to points
    unprojected from depth (``world_points_from_depth`` + ``depth_conf``).
    Returns (points, confidence) arrays or (None, None) if neither is present.
    """
    if "world_points" in predictions:
        pts = predictions["world_points"]
        conf = predictions.get("world_points_conf", np.ones(pts.shape[:-1]))
        return pts, conf
    if "world_points_from_depth" in predictions:
        pts = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf")
        if conf is None:
            conf = np.ones(pts.shape[:-1])
        return pts, conf
    return None, None


def generate_3d_content(predictions: Dict[str, Any],
                        params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the preview GLB + predictions.npz into the run dir.

    Backbone-agnostic: works off the normalized prediction schema. The scene
    exporter itself decides whether to use ``world_points`` or fall back to
    ``world_points_from_depth``.
    """
    from datetime import datetime
    try:
        print("Generating 3D content...")

        # Run dir: caller-provided name (e.g. an area tag) or the default timestamp.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = params.get("output_name") or f"input_stream_{timestamp}"
        target_dir_images = os.path.join(target_dir, "images")
        os.makedirs(target_dir_images, exist_ok=True)
        print(f"Created target directory: {target_dir}")

        keyframe_paths = predictions["metadata"]["keyframe_paths"]
        image_paths_in_target = []
        for i, keyframe_path in enumerate(keyframe_paths):
            dst_path = os.path.join(target_dir_images, f"frame_{i:06d}.jpg")
            shutil.copy2(keyframe_path, dst_path)
            image_paths_in_target.append(dst_path)
        print(f"Copied {len(keyframe_paths)} keyframes to {target_dir_images}")

        scene = predictions_to_glb(
            predictions=predictions,
            conf_thres=params["conf_threshold"],
            filter_by_frames="all",
            mask_black_bg=params["mask_black_bg"],
            mask_white_bg=params["mask_white_bg"],
            show_cam=params["show_cam"],
            mask_sky=params["mask_sky"],
            mask_dynamic=params["mask_dynamic"],
            target_dir=target_dir,
        )

        glb_path = os.path.join(target_dir, "scene.glb")
        scene.export(glb_path)
        print(f"GLB scene generated: {glb_path}")

        predictions_for_save = ensure_cpu_tensors(predictions)
        prediction_save_path = os.path.join(target_dir, "predictions.npz")
        np.savez(prediction_save_path, **predictions_for_save)
        print(f"Predictions saved: {prediction_save_path}")

        return {
            "glb_path": glb_path if os.path.exists(glb_path) else None,
            "scene": scene,
            "target_directory": target_dir,
            "images_directory": target_dir_images,
            "copied_images": image_paths_in_target,
        }
    except Exception as e:
        print(f"Error generating 3D content: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def generate_confidence_mapping(predictions: Dict[str, Any],
                                params: Dict[str, Any],
                                target_dir: str) -> Dict[str, Any]:
    """Generate the map-quality (confidence) point cloud for NFN visualization.

    Uses whichever point/confidence pair the backbone produced (dedicated point
    head or unprojected depth), so both point-head and depth-only backbones get
    a confidence map.
    """
    if not CONFIDENCE_MAPPING_AVAILABLE:
        return {"error": "Confidence mapping utilities not available"}
    try:
        print("Generating confidence mapping...")

        world_points, world_points_conf = _point_conf_keys(predictions)
        if world_points is None:
            return {"error": "No world points or depth available in predictions"}

        conf_threshold_normalized = params["conf_threshold"] / 100.0
        target_ply_path = os.path.join(target_dir, "confidence_map.ply")
        scene, stats, ply_path = generate_confidence_point_cloud(
            world_points=world_points,
            confidence=world_points_conf,
            conf_threshold=conf_threshold_normalized,
            save_ply=True,
            ply_path=target_ply_path,
        )
        if ply_path:
            print(f"Confidence map PLY saved: {ply_path}")

        conf_glb_path = os.path.join(target_dir, "confidence_map.glb")
        if scene:
            scene.export(conf_glb_path)
            print(f"Confidence map GLB saved: {conf_glb_path}")

        results = {
            "scene": scene,
            "glb_path": conf_glb_path if os.path.exists(conf_glb_path) else None,
            "ply_path": target_ply_path if target_ply_path and os.path.exists(target_ply_path) else None,
            "statistics": stats,
            "threshold_used": conf_threshold_normalized,
        }
        print(f"Confidence mapping generated: {stats.get('high_conf_points', 0)}/"
              f"{stats.get('total_points', 0)} high-confidence points")
        return results
    except Exception as e:
        print(f"Error generating confidence mapping: {e}")
        return {"error": str(e)}
