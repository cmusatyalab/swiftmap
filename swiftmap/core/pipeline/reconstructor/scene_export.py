# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Live-preview scene export: raw backbone predictions -> a GLB scene / PLY, in the
VGGT viewer convention (OpenGL flip + align to the first camera). Stored areas render
through ``geometry.pointcloud_scene`` instead; this path is the mapper's own preview."""

import trimesh
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation

from swiftmap.core.primitives import geometry
from swiftmap.core.pipeline.reconstructor import sky_mask

# Point-cloud filtering thresholds.
_BLACK_BG_SUM = 16           # drop pixels whose RGB channel sum is below this
_WHITE_BG_LEVEL = 240        # drop pixels with every channel above this
_SCENE_SCALE_PERCENTILES = (5, 95)


def _selected_frame_index(filter_by_frames):
    """Parse an 'idx:...' frame filter into an int index, or None for 'all'."""
    if filter_by_frames in ("all", "All"):
        return None
    try:
        return int(filter_by_frames.split(":")[0])
    except (ValueError, IndexError):
        return None


def _select_points_and_conf(predictions, prediction_mode):
    """Pick (world_points, confidence) per the prediction mode. Shared by the GLB
    and PLY export paths."""
    if "Pointmap" in prediction_mode and "world_points" not in predictions:
        print("Warning: world_points not found in predictions, falling back to depth-based points")
    if "Pointmap" in prediction_mode and "world_points" in predictions:
        pts = predictions["world_points"]
        conf = predictions.get("world_points_conf", np.ones_like(pts[..., 0]))
    else:
        pts = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf", np.ones_like(pts[..., 0]))
    return pts, conf


def _apply_dynamic_filter(predictions, conf, target_dir):
    """Zero confidence on dynamic objects via the semantic filter, if available.
    No-op when target_dir is None. Returns the updated confidence array."""
    if target_dir is None:
        return conf
    try:
        from utils.semantic_filter import filter_dynamic_objects_from_predictions
        predictions_copy = predictions.copy()
        predictions_copy["world_points_conf"] = conf
        print("[Dynamic Filter] Applying semantic filtering to remove dynamic objects...")
        filtered = filter_dynamic_objects_from_predictions(predictions_copy, target_dir)
        return filtered["world_points_conf"]
    except ImportError as e:
        print(f"[Dynamic Filter] Warning: Could not import semantic filter: {e}")
    except Exception as e:
        print(f"[Dynamic Filter] Warning: Semantic filtering failed: {e}")
    return conf


def _confidence_mask(conf, conf_thres, colors_rgb, mask_black_bg, mask_white_bg):
    """``geometry.confidence_mask`` plus optional black/white background rejection."""
    mask = geometry.confidence_mask(conf, conf_thres)
    if mask_black_bg:
        mask = mask & (colors_rgb.sum(axis=1) >= _BLACK_BG_SUM)
    if mask_white_bg:
        white = ((colors_rgb[:, 0] > _WHITE_BG_LEVEL) &
                 (colors_rgb[:, 1] > _WHITE_BG_LEVEL) &
                 (colors_rgb[:, 2] > _WHITE_BG_LEVEL))
        mask = mask & ~white
    return mask


def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    mask_dynamic=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
) -> trimesh.Scene:
    """Convert backbone predictions to a GLB scene (point cloud + camera cones).

    predictions: dict with world_points (S,H,W,3), world_points_conf (S,H,W),
    images (S,H,W,3), extrinsic (S,3,4). ``conf_thres`` is the confidence percentile;
    ``mask_sky``/``mask_dynamic`` zero sky/dynamic points when ``target_dir`` is set.
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")
    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")
    selected_frame_idx = _selected_frame_index(filter_by_frames)

    pred_world_points, pred_world_points_conf = _select_points_and_conf(predictions, prediction_mode)
    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    if mask_sky:
        pred_world_points_conf = sky_mask.apply_sky_mask(pred_world_points_conf, target_dir)
    if mask_dynamic:
        pred_world_points_conf = _apply_dynamic_filter(predictions, pred_world_points_conf, target_dir)

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    colors_rgb = geometry.flatten_colors(images)
    conf = pred_world_points_conf.reshape(-1)
    conf_mask = _confidence_mask(conf, conf_thres, colors_rgb, mask_black_bg, mask_white_bg)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        lower_percentile = np.percentile(vertices_3d, _SCENE_SCALE_PERCENTILES[0], axis=0)
        upper_percentile = np.percentile(vertices_3d, _SCENE_SCALE_PERCENTILES[1], axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    scene_3d = trimesh.Scene()
    scene_3d.add_geometry(geometry.pointcloud(vertices_3d, colors_rgb))

    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        for i in range(num_cameras):
            camera_to_world = np.linalg.inv(extrinsics_matrices[i])
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    scene_3d = geometry.apply_scene_alignment(scene_3d, extrinsics_matrices)
    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray,
                                face_colors: tuple, scene_scale: float):
    """Add a camera cone (4-sided) at ``transform`` (camera-to-world) to ``scene``."""
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    complete_transform = transform @ geometry.opengl_conversion_matrix() @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate([
        camera_cone_shape.vertices,
        0.95 * camera_cone_shape.vertices,
        geometry.transform_points(slight_rotation, camera_cone_shape.vertices),
    ])
    vertices_transformed = geometry.transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)

    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """Faces of the pseudo-camera mesh built from a cone (for ``integrate_camera_into_scene``)."""
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend([
            (v1, v2, v2_offset), (v1, v1_offset, v3), (v3_offset, v2, v3),
            (v1, v2, v2_offset_2), (v1, v1_offset_2, v3), (v3_offset_2, v2, v3),
        ])
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def extract_point_cloud_data(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    mask_sky=False,
    mask_dynamic=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
):
    """(vertices_3d, colors_rgb) filtered exactly as ``predictions_to_glb``, for PLY export."""
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")
    if conf_thres is None:
        conf_thres = 10.0

    print("Extracting point cloud data for PLY export...")
    selected_frame_idx = _selected_frame_index(filter_by_frames)

    pred_world_points, pred_world_points_conf = _select_points_and_conf(predictions, prediction_mode)
    images = predictions["images"]

    if mask_sky:
        pred_world_points_conf = sky_mask.apply_sky_mask(pred_world_points_conf, target_dir)
    if mask_dynamic:
        pred_world_points_conf = _apply_dynamic_filter(predictions, pred_world_points_conf, target_dir)

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    colors_rgb = geometry.flatten_colors(images)
    conf = pred_world_points_conf.reshape(-1)
    conf_mask = _confidence_mask(conf, conf_thres, colors_rgb, mask_black_bg, mask_white_bg)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])

    print(f"Extracted {len(vertices_3d)} points for PLY export")
    return vertices_3d, colors_rgb


def save_point_cloud_ply(vertices, colors, filename):
    """Write an xyz+rgb PLY (delegates to ``geometry.write_ply``)."""
    return geometry.write_ply(filename, vertices, colors)


def export_point_cloud_to_ply(
    predictions,
    ply_filename,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    mask_sky=False,
    mask_dynamic=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
):
    """Extract point-cloud data for PLY export (same filtering as GLB export)."""
    return extract_point_cloud_data(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        mask_dynamic=mask_dynamic,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
