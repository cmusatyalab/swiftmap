# Copyright (C) 2024 Carnegie Mellon University

"""Builds scene.glb / confidence_map.glb / camera_poses.json content; Map.write2disk() exports it."""

import copy
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import trimesh

from swiftmap.core import constants
from swiftmap.database.map import Map
from swiftmap.database.types import PointCloud
from swiftmap.core.pipeline.reconstructor.pose import camera_poses_from_extrinsics

_SKY_MASK_THRESHOLD = 0.1
_MAX_CONFIDENCE_POINTS = 50000
_CONF_EPSILON = 1e-6


def generate_3d_scene(map: Map, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the confidence-filtered preview scene and attach it as pt.scene."""
    try:
        print("Generating 3D content...")
        pt = map.get_pointcloud()
        pts, conf = pt.world_points, pt.world_points_conf

        if params.get("mask_sky"):
            conf = _apply_sky_mask(map)
        xyz, conf, keep = _flatten_valid(params["conf_threshold"], pts, conf)
        xyz, conf = xyz[keep], conf[keep]
        cols = pt.flatten_colors()[keep]
        xyz[:, 1:] *= -1

        frames = []
        if params.get("show_cam") and pt.extrinsic is not None:
            positions, rotations = camera_poses_from_extrinsics(pt.extrinsic)
            for p, R in zip(positions, rotations):
                p[1:] *= -1
                R[1:, :] *= -1
                frames.append({"camera_position_world": p.tolist(), "rotation_matrix": R.tolist()})

        scene = trimesh.Scene()
        geometry = trimesh.PointCloud(vertices=xyz, colors=cols)
        scene.add_geometry(geometry, geom_name="points")
        if frames and len(xyz):
            size = float(np.clip(np.linalg.norm(xyz.max(0) - xyz.min(0)) * 0.01, 1.0, 5.0))
            scene.add_geometry(_camera_frustums(frames, size, (20, 20, 20, 255)), geom_name="cameras")
        pt.scene = scene

        print(f"3D scene generated: {len(xyz)} points")
        return {"success": True}
    except Exception as e:
        print(f"Error generating 3D content: {e}")
        return {"error": str(e)}

def _camera_frustums(frames, size: float, rgba):
    """One Trimesh of camera-frustum pyramids for frames."""
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


def generate_confidence_scene(map: Map, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the map-quality confidence point cloud and attach it as pt.confidence_scene."""
    try:
        print("Generating confidence mapping...")
        pt = map.get_pointcloud()
        pts, conf = pt.world_points, pt.world_points_conf
        if pts is None:
            return {"error": "No world points or depth available in predictions"}

        xyz, conf, keep = _flatten_valid(params["conf_threshold"], pts, conf)
        xyz, conf = xyz[keep], conf[keep]
        stats = {
            "total_points": int(keep.size),
            "high_conf_points": len(xyz),
            "coverage_ratio": (len(xyz) / keep.size) if keep.size else 0.0,
            "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
            "confidence_std": float(np.std(conf)) if len(conf) else 0.0,
        }
        pt.confidence_stats = stats
        if len(xyz) == 0:
            pt.confidence_scene = None
            return {"statistics": stats}

        if len(xyz) > _MAX_CONFIDENCE_POINTS:
            idx = np.random.choice(len(xyz), _MAX_CONFIDENCE_POINTS, replace=False)
            xyz, conf = xyz[idx], conf[idx]

        xyz = xyz.copy()
        xyz[:, 1:] *= -1
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.points.PointCloud(vertices=xyz, colors=_confidence_to_colors(conf)),
                          node_name="confidence_points")
        pt.confidence_scene = scene

        print(f"Confidence mapping generated: {stats['high_conf_points']}/{stats['total_points']} "
              f"high-confidence points")
        return {"statistics": stats}
    except Exception as e:
        print(f"Error generating confidence mapping: {e}")
        return {"error": str(e)}


def _confidence_to_colors(confidence: np.ndarray) -> np.ndarray:
    """Confidence -> RGBA, red (low) to green (high)."""
    lo, hi = np.min(confidence), np.max(confidence)
    norm = (confidence - lo) / (hi - lo) if hi > lo else np.ones_like(confidence)
    colors = np.zeros((len(norm), 4), dtype=np.uint8)
    colors[:, 0] = ((1.0 - norm) * 255).astype(np.uint8)
    colors[:, 1] = (norm * 255).astype(np.uint8)
    colors[:, 3] = 255
    return colors


def generate_camera_poses(map: Map, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the camera_poses.json payload and attach it as pt.camera_poses."""
    pt = map.get_pointcloud()
    if pt is None or pt.extrinsic is None or pt.intrinsic is None:
        return None

    extrinsic = np.asarray(pt.extrinsic)
    intrinsic = np.asarray(pt.intrinsic)
    keyframe_paths = map.get_keyframe_paths()
    positions, rotations = camera_poses_from_extrinsics(extrinsic)

    poses_data = {
        "metadata": {
            "description": "Camera poses from SwiftMap Mapping",
            "backbone": params.get("backbone"),
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

    pt.camera_poses = poses_data
    return poses_data


def generate_model_input(map: Map) -> Optional[list]:
    """Encode the model-resolution input frames as JPEGs and attach as pt.model_input."""
    import cv2
    pt = map.get_pointcloud()
    if pt is None or pt.images is None:
        return None

    images = np.asarray(pt.images)
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = (np.clip(images, 0.0, 1.0) * 255).astype(np.uint8)

    encoded = []
    for i, img in enumerate(images):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if ok:
            encoded.append((f"frame_{i:06d}.jpg", buf.tobytes()))
    pt.model_input = encoded
    return encoded


def _flatten_valid(percentile: float, pts, conf):
    """Flatten points/conf and mask to finite points at/above the confidence percentile."""
    xyz = np.asarray(pts).reshape(-1, 3)
    conf = np.asarray(conf, dtype=float).reshape(-1)

    keep = np.isfinite(xyz).all(1) & (conf > _CONF_EPSILON)
    if percentile:
        thr = np.percentile(conf, float(percentile))
        keep &= conf >= thr
    return xyz, conf, keep


def _apply_sky_mask(map: Map) -> np.ndarray:
    """Zero confidence on sky pixels for each frame (skyseg.onnx)."""
    pt = map.get_pointcloud()
    conf = np.array(pt.world_points_conf, dtype=float)
    target_dir = map.path

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
    """Binary mask (255 = non-sky) for one image; also written to mask_path."""
    import cv2
    image = cv2.imread(image_path)
    result = cv2.resize(_run_skyseg(session, (320, 320), image), (image.shape[1], image.shape[0]))
    out = np.zeros_like(result)
    out[result < 32] = 255
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
    """Download url to filename, following a single redirect."""
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
