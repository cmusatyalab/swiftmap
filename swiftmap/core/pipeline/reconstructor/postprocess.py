# Copyright (C) 2024 Carnegie Mellon University

"""Backbone-agnostic post-processing: scene.glb, confidence_map.glb, camera_poses.json.

Public API: ``generate_3d_scene``, ``generate_confidence_scene``, ``generate_camera_poses``
(used by ``BaseReconstructor``). Everything else here is a private helper. Extrinsics ->
world-pose math lives in ``pipeline.reconstructor.pose`` (shared with the backend
adapters, which need it independently of this module).
"""

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import trimesh

from swiftmap.core import constants
from swiftmap.core.database import cloud as arrays
from swiftmap.core.pipeline import utils as pipeline_utils
from swiftmap.core.pipeline.reconstructor.pose import camera_poses_from_extrinsics

_SKY_MASK_THRESHOLD = 0.1   # skyseg value above this counts as sky (zeroed)
_MAX_CONFIDENCE_POINTS = 50000  # subsample cap for the confidence scene


# =================================================================== generate_3d_scene
def generate_3d_scene(predictions: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate scene.glb, the run's preview scene, into the run dir."""
    try:
        print("Generating 3D content...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = params.get("output_name") or f"input_stream_{timestamp}"
        os.makedirs(target_dir, exist_ok=True)

        scene = _preview_scene(predictions, params, target_dir)
        glb_path = os.path.join(target_dir, "scene.glb")
        scene.export(glb_path)
        print(f"GLB scene generated: {glb_path}")
        return {"glb_path": glb_path, "scene": scene, "target_directory": target_dir}
    except Exception as e:
        print(f"Error generating 3D content: {e}")
        return {"error": str(e)}


def _preview_scene(predictions, params, target_dir):
    """Confidence-filtered cloud + camera frustums, in local coords."""
    pts, conf = _point_conf_keys(predictions)
    if params.get("mask_sky"):
        conf = _apply_sky_mask(np.array(conf, dtype=float), target_dir)
    xyz, _, keep = _flatten_valid(pts, conf, params["conf_threshold"])
    cols = arrays.flatten_colors(predictions["images"])

    frames = []
    if params.get("show_cam") and "extrinsic" in predictions:
        positions, rotations = camera_poses_from_extrinsics(predictions["extrinsic"])
        frames = [{"camera_position_world": p.tolist(), "rotation_matrix": R.tolist()}
                  for p, R in zip(positions, rotations)]
    return _pointcloud_scene(xyz[keep], cols[keep], frames)


def _pointcloud_scene(points, colors, frames=None,
                      frustum_rgba=(20, 20, 20, 255), frustum_scale: float = 0.01):
    """trimesh.Scene of a point cloud plus optional camera frustums, in the world frame.

    Frustum size scales with the cloud's diagonal (clamped to [1, 5] m).
    """
    points = np.asarray(points)
    scene = trimesh.Scene()
    scene.add_geometry(pipeline_utils.pointcloud(points, colors), geom_name="points")
    if frames and len(points):
        size = float(np.clip(np.linalg.norm(points.max(0) - points.min(0)) * frustum_scale, 1.0, 5.0))
        scene.add_geometry(_camera_frustums(frames, size, frustum_rgba), geom_name="cameras")
    return scene


def _camera_frustums(frames, size: float, rgba):
    """One Trimesh of camera-frustum pyramids for ``frames`` (in their coordinate frame)."""
    corners = np.array([[-0.5, -0.375, 1.0], [0.5, -0.375, 1.0],
                        [0.5, 0.375, 1.0], [-0.5, 0.375, 1.0]])
    verts = np.zeros((len(frames) * 5, 3))
    faces = np.zeros((len(frames) * 4, 3), dtype=np.int64)
    for k, fr in enumerate(frames):
        r = np.asarray(fr["rotation_matrix"], float)
        c = np.asarray(fr["camera_position_world"], float)
        base = (r.T @ (size * corners).T).T + c
        vb = 5 * k
        verts[vb] = c
        verts[vb + 1:vb + 5] = base
        for i in range(4):
            faces[4 * k + i] = [vb, vb + 1 + i, vb + 1 + (i + 1) % 4]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(np.asarray(rgba, np.uint8), (len(faces), 1))
    return mesh


# ============================================================ generate_confidence_scene
def generate_confidence_scene(predictions: Dict[str, Any], params: Dict[str, Any],
                              target_dir: str) -> Dict[str, Any]:
    """Generate confidence_map.glb, the map-quality point cloud, for NFN visualization."""
    try:
        print("Generating confidence mapping...")
        pts, conf = _point_conf_keys(predictions)
        if pts is None:
            return {"error": "No world points or depth available in predictions"}

        xyz, conf, keep = _flatten_valid(pts, conf, params["conf_threshold"])
        xyz, conf = xyz[keep], conf[keep]
        stats = {
            "total_points": int(keep.size),
            "high_conf_points": len(xyz),
            "coverage_ratio": (len(xyz) / keep.size) if keep.size else 0.0,
            "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
            "confidence_std": float(np.std(conf)) if len(conf) else 0.0,
        }
        if len(xyz) == 0:
            return {"scene": None, "glb_path": None, "statistics": stats}

        if len(xyz) > _MAX_CONFIDENCE_POINTS:
            idx = np.random.choice(len(xyz), _MAX_CONFIDENCE_POINTS, replace=False)
            xyz, conf = xyz[idx], conf[idx]

        xyz = xyz.copy()
        xyz[:, 1:] *= -1  # flip Y/Z so the cloud is right-side up in the viewer
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.points.PointCloud(vertices=xyz, colors=_confidence_to_colors(conf)),
                          node_name="confidence_points")

        glb_path = os.path.join(target_dir, "confidence_map.glb")
        scene.export(glb_path)
        print(f"Confidence map GLB saved: {glb_path}")
        print(f"Confidence mapping generated: {stats['high_conf_points']}/{stats['total_points']} "
              f"high-confidence points")
        return {"scene": scene, "glb_path": glb_path, "statistics": stats}
    except Exception as e:
        print(f"Error generating confidence mapping: {e}")
        return {"error": str(e)}


def _confidence_to_colors(confidence: np.ndarray) -> np.ndarray:
    """Confidence -> RGBA, red (low) to green (high), normalised over the given values."""
    lo, hi = np.min(confidence), np.max(confidence)
    norm = (confidence - lo) / (hi - lo) if hi > lo else np.ones_like(confidence)
    colors = np.zeros((len(norm), 4), dtype=np.uint8)
    colors[:, 0] = ((1.0 - norm) * 255).astype(np.uint8)
    colors[:, 1] = (norm * 255).astype(np.uint8)
    colors[:, 3] = 255
    return colors


# =============================================================== generate_camera_poses
def generate_camera_poses(predictions: Dict[str, Any], target_dir: str,
                          backbone: str) -> Optional[str]:
    """Write camera_poses.json (per-keyframe pose + intrinsics) into the run dir."""
    if "extrinsic" not in predictions or "intrinsic" not in predictions:
        return None
    extrinsic = np.asarray(predictions["extrinsic"])
    intrinsic = np.asarray(predictions["intrinsic"])
    keyframe_paths = predictions.get("metadata", {}).get("keyframe_paths", [])
    positions, rotations = camera_poses_from_extrinsics(extrinsic)

    poses_data = {
        "metadata": {
            "description": "Camera poses from SwiftMap Mapping",
            "backbone": backbone,
            "timestamp": datetime.now().isoformat(),
            "num_keyframes": len(extrinsic),
        },
        "frames": [],
    }
    for i, (ext, intr, pos, rot) in enumerate(zip(extrinsic, intrinsic, positions, rotations)):
        image_name = (os.path.basename(keyframe_paths[i])
                      if i < len(keyframe_paths) else f"keyframe_{i}")
        poses_data["frames"].append({
            "image_name": image_name,
            "camera_position_world": pos.tolist(),
            "rotation_matrix": rot.tolist(),
            "translation_vector": ext[:3, 3].tolist(),
            "intrinsic_matrix": intr.tolist(),
            "extrinsic_matrix": ext.tolist(),
        })

    poses_path = os.path.join(target_dir, "camera_poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses_data, f, indent=2)
    print(f"Camera poses saved: {poses_path}")
    return poses_path


# ============================================================================ shared
def _point_conf_keys(predictions: Dict[str, Any]):
    """Pick the (points, confidence) key pair present for this backbone.

    Prefers a dedicated point head (``world_points``); falls back to points
    unprojected from depth (``world_points_from_depth`` + ``depth_conf``).
    """
    if "world_points" in predictions:
        pts = predictions["world_points"]
        return pts, predictions.get("world_points_conf", np.ones(pts.shape[:-1]))
    if "world_points_from_depth" in predictions:
        pts = predictions["world_points_from_depth"]
        return pts, predictions.get("depth_conf", np.ones(pts.shape[:-1]))
    return None, None


def _flatten_valid(pts, conf, percentile: float):
    """Flatten (..., 3)/(...) point+confidence arrays; mask to finite points at/above
    ``percentile`` confidence. Shared by the 3D scene and the confidence scene."""
    xyz = np.asarray(pts).reshape(-1, 3)
    conf = np.asarray(conf, dtype=float).reshape(-1)
    keep = np.isfinite(xyz).all(1) & arrays.confidence_mask(conf, percentile)
    return xyz, conf, keep


def _apply_sky_mask(conf, target_dir):
    """Zero confidence on sky pixels for each frame (skyseg.onnx). No-op if ``target_dir``
    is None or has no saved images to segment."""
    if target_dir is None:
        return conf
    import cv2
    import onnxruntime

    images_dir = os.path.join(target_dir, "images")
    image_list = sorted(os.listdir(images_dir)) if os.path.isdir(images_dir) else []
    if not image_list:
        return conf
    os.makedirs(os.path.join(target_dir, "sky_masks"), exist_ok=True)
    _, H, W = conf.shape

    if not os.path.exists(constants.SKYSEG_ONNX_PATH):
        print(f"Downloading skyseg.onnx -> {constants.SKYSEG_ONNX_PATH}")
        os.makedirs(os.path.dirname(constants.SKYSEG_ONNX_PATH), exist_ok=True)
        _download_skyseg(constants.SKYSEG_ONNX_URL, constants.SKYSEG_ONNX_PATH)

    session = None
    masks = []
    for name in image_list:
        mask_path = os.path.join(target_dir, "sky_masks", name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            if session is None:
                session = onnxruntime.InferenceSession(constants.SKYSEG_ONNX_PATH)
            mask = _segment_sky(os.path.join(images_dir, name), session, mask_path)
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H))
        masks.append(mask)

    binary = (np.array(masks) > _SKY_MASK_THRESHOLD).astype(np.float32)
    return conf * binary


def _segment_sky(image_path, session, mask_path) -> np.ndarray:
    """Binary mask (255 = non-sky) for one image; also written to ``mask_path``."""
    import cv2
    image = cv2.imread(image_path)
    result = cv2.resize(_run_skyseg(session, (320, 320), image), (image.shape[1], image.shape[0]))
    out = np.zeros_like(result)
    out[result < 32] = 255  # model emits low values for sky
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, out)
    return out


def _run_skyseg(session, input_size, image) -> np.ndarray:
    """Run skyseg inference; returns a uint8 [0,255] segmentation map."""
    import cv2
    x = cv2.cvtColor(cv2.resize(copy.deepcopy(image), input_size), cv2.COLOR_BGR2RGB)
    x = (np.asarray(x, np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1).reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    out = np.asarray(session.run([session.get_outputs()[0].name],
                                 {session.get_inputs()[0].name: x})).squeeze()
    out = (out - out.min()) / (out.max() - out.min()) * 255
    return out.astype("uint8")


def _download_skyseg(url, filename):
    """Download ``url`` to ``filename``, following a single redirect."""
    import requests
    response = requests.get(url, allow_redirects=False)
    response.raise_for_status()
    if response.status_code == 302:
        response = requests.get(response.headers["Location"], stream=True)
        response.raise_for_status()
    else:
        print(f"Unexpected status code: {response.status_code}")
        return
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")
