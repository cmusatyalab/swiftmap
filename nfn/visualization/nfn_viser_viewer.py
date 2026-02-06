"""
NFN Viser Visualization

3D visualization of VGGT reconstruction with NFN analysis overlay:
- Existing point cloud reconstruction
- Coverage gap regions (wireframe boxes)
- Suggested camera viewpoints (camera frustums)
- Interactive controls for gap detection parameters

Based on demo_viser.py architecture.
"""

import numpy as np
import viser
import viser.transforms as viser_tf
from typing import Dict, List, Optional, Tuple
import threading
import time


def visualize_nfn_with_viser(
    predictions: Dict,
    gap_results: Dict,
    viewpoint_results: Dict,
    port: int = 8080,
    conf_threshold: float = 50.0,
    background_mode: bool = False
) -> Optional[viser.ViserServer]:
    """
    Visualize VGGT reconstruction with NFN analysis using Viser.

    Args:
        predictions: VGGT predictions containing world_points, confidence, etc.
        gap_results: Gap detection results from CoverageGapDetector
        viewpoint_results: Viewpoint planning results from ViewpointPlanner
        port: Viser server port
        conf_threshold: Initial confidence threshold (percentage)
        background_mode: Run in background thread

    Returns:
        Viser server instance (or None if background mode)
    """
    print(f"[NFN Visualization] Starting Viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Extract prediction data
    world_points = predictions["world_points"]  # (S, H, W, 3)
    confidence = predictions["world_points_conf"]  # (S, H, W)
    images = predictions.get("images")  # Optional: (S, 3, H, W)
    extrinsics = predictions.get("extrinsic")  # (S, 3, 4)

    # Prepare point cloud data
    S, H, W, _ = world_points.shape
    points = world_points.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)

    # Prepare colors
    if images is not None:
        colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
        colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    else:
        # Default gray color
        colors_flat = np.full((len(points), 3), 128, dtype=np.uint8)

    # Apply initial confidence filter
    # conf_threshold is already a percentage (0-100), use it directly as percentile
    percentile = conf_threshold  # Already in range [0, 100]
    valid_conf = conf_flat[np.isfinite(conf_flat)]
    if len(valid_conf) > 0:
        actual_threshold = np.percentile(valid_conf, percentile)
    else:
        actual_threshold = 0.0
    mask = (conf_flat >= actual_threshold) & np.isfinite(points).all(axis=1)

    filtered_points = points[mask]
    filtered_colors = colors_flat[mask]

    print(f"[NFN Visualization] Displaying {len(filtered_points)}/{len(points)} points")

    # Add main point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name="/nfn/point_cloud",
        points=filtered_points,
        colors=filtered_colors,
        point_size=0.01
    )

    # Add existing camera poses (if available)
    if extrinsics is not None:
        _add_existing_cameras(server, extrinsics)

    # Add gap visualizations
    if gap_results and gap_results.get("gap_count", 0) > 0:
        _add_gap_visualizations(server, gap_results)

    # Add suggested viewpoints
    if viewpoint_results and viewpoint_results.get("num_viewpoints", 0) > 0:
        _add_suggested_viewpoints(server, viewpoint_results)

    # Add GUI controls
    _add_gui_controls(
        server,
        points,
        colors_flat,
        conf_flat,
        point_cloud_handle,
        conf_threshold
    )

    # Display statistics
    _display_statistics(server, predictions, gap_results, viewpoint_results)

    print(f"[NFN Visualization] Viser server ready at http://localhost:{port}")

    if background_mode:
        # Run in background thread
        def run_server():
            while True:
                time.sleep(0.1)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return None
    else:
        # Block until user closes
        print("[NFN Visualization] Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[NFN Visualization] Shutting down...")

        return server


def _add_existing_cameras(server: viser.ViserServer, extrinsics: np.ndarray):
    """
    Add existing camera poses to the scene.
    """
    print(f"[NFN Visualization] Adding {len(extrinsics)} existing camera poses")

    for i, ext in enumerate(extrinsics):
        R = ext[:3, :3]
        t = ext[:3, 3]

        # Camera position in world coordinates
        world_pos = -R.T @ t

        # Camera rotation (wxyz quaternion)
        rotation_mat = R.T  # world-to-camera rotation
        wxyz = viser_tf.SO3.from_matrix(rotation_mat).wxyz

        # Add camera frustum (scale matches demo_viser.py)
        server.scene.add_camera_frustum(
            name=f"/nfn/existing_cameras/camera_{i}",
            fov=60.0,
            aspect=1.0,
            scale=0.05,  # Match demo_viser.py scale
            color=(0, 255, 0),  # Green for existing cameras
            position=world_pos,
            wxyz=wxyz
        )


def _add_gap_visualizations(server: viser.ViserServer, gap_results: Dict):
    """
    Add gap region visualizations as wireframe boxes.
    """
    gap_regions = gap_results.get("gap_regions", [])

    print(f"[NFN Visualization] Adding {len(gap_regions)} gap regions")

    for i, (gap_min, gap_max) in enumerate(gap_regions):
        gap_center = (gap_min + gap_max) / 2
        gap_size = gap_max - gap_min

        # Add wireframe box for gap region
        _add_wireframe_box(
            server,
            name=f"/nfn/gaps/gap_{i}",
            center=gap_center,
            size=gap_size,
            color=(255, 0, 0)  # Red for gaps
        )

        # Add label at gap center
        # Note: add_label() doesn't support color parameter in current viser version
        server.scene.add_label(
            name=f"/nfn/gaps/label_{i}",
            text=f"Gap {i}",
            position=gap_center
        )


def _add_suggested_viewpoints(server: viser.ViserServer, viewpoint_results: Dict):
    """
    Add suggested camera viewpoints as camera frustums.
    """
    viewpoints = viewpoint_results.get("viewpoints", [])

    print(f"[NFN Visualization] Adding {len(viewpoints)} suggested viewpoints")

    for i, vp in enumerate(viewpoints):
        camera_pos = vp["camera_position"]
        camera_rot = vp["camera_rotation"]

        # Convert rotation matrix to quaternion
        wxyz = viser_tf.SO3.from_matrix(camera_rot).wxyz

        # Add camera frustum (slightly larger than existing cameras for visibility)
        server.scene.add_camera_frustum(
            name=f"/nfn/suggested_viewpoints/camera_{i}",
            fov=60.0,
            aspect=1.0,
            scale=0.1,  # 2x larger than existing cameras (0.05) for distinction
            color=(0, 0, 255),  # Blue for suggested cameras
            position=camera_pos,
            wxyz=wxyz
        )

        # Add line connecting suggested camera to its target gap
        gap_center = vp["gap_center"]
        _add_line(
            server,
            name=f"/nfn/viewpoint_lines/line_{i}",
            start=camera_pos,
            end=gap_center,
            color=(0, 0, 255)
        )


def _add_wireframe_box(
    server: viser.ViserServer,
    name: str,
    center: np.ndarray,
    size: np.ndarray,
    color: Tuple[int, int, int]
):
    """
    Add a wireframe box to visualize gap region.
    """
    # Calculate box corners
    half_size = size / 2
    corners = np.array([
        center + half_size * np.array([-1, -1, -1]),
        center + half_size * np.array([+1, -1, -1]),
        center + half_size * np.array([+1, +1, -1]),
        center + half_size * np.array([-1, +1, -1]),
        center + half_size * np.array([-1, -1, +1]),
        center + half_size * np.array([+1, -1, +1]),
        center + half_size * np.array([+1, +1, +1]),
        center + half_size * np.array([-1, +1, +1])
    ])

    # Define edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Add lines for each edge
    for i, (start_idx, end_idx) in enumerate(edges):
        _add_line(
            server,
            name=f"{name}/edge_{i}",
            start=corners[start_idx],
            end=corners[end_idx],
            color=color
        )


def _add_line(
    server: viser.ViserServer,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    color: Tuple[int, int, int],
    line_width: float = 2.0
):
    """
    Add a line segment to the scene.
    """
    # Create line as a thin cylinder
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return

    direction = direction / length
    midpoint = (start + end) / 2

    # Calculate rotation to align cylinder with line direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6:
        # Parallel or anti-parallel
        if np.dot(z_axis, direction) > 0:
            wxyz = np.array([1.0, 0, 0, 0])
        else:
            wxyz = np.array([0.0, 1, 0, 0])
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        # Axis-angle to quaternion
        half_angle = angle / 2
        wxyz = np.array([
            np.cos(half_angle),
            rotation_axis[0] * np.sin(half_angle),
            rotation_axis[1] * np.sin(half_angle),
            rotation_axis[2] * np.sin(half_angle)
        ])

    # Add cylinder
    server.scene.add_mesh_simple(
        name=name,
        vertices=_create_cylinder_vertices(length, line_width / 1000.0),
        faces=_create_cylinder_faces(),
        color=color,
        position=midpoint,
        wxyz=wxyz
    )


def _create_cylinder_vertices(height: float, radius: float, segments: int = 8) -> np.ndarray:
    """Create vertices for a cylinder mesh."""
    vertices = []

    # Bottom circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, -height / 2])

    # Top circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, height / 2])

    return np.array(vertices, dtype=np.float32)


def _create_cylinder_faces(segments: int = 8) -> np.ndarray:
    """Create face indices for a cylinder mesh."""
    faces = []

    # Side faces
    for i in range(segments):
        next_i = (i + 1) % segments
        # Two triangles per side face
        faces.append([i, next_i, segments + i])
        faces.append([next_i, segments + next_i, segments + i])

    return np.array(faces, dtype=np.uint32)


def _add_gui_controls(
    server: viser.ViserServer,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    point_cloud_handle,
    initial_threshold: float
):
    """
    Add GUI controls for interactive visualization.
    """
    # Confidence threshold slider
    conf_slider = server.gui.add_slider(
        "Confidence Threshold (%)",
        min=0,
        max=100,
        step=1,
        initial_value=initial_threshold
    )

    # Update callback
    @conf_slider.on_update
    def update_confidence(_):
        threshold = conf_slider.value / 100.0
        percentile = conf_slider.value

        valid_conf = confidence[np.isfinite(confidence)]
        if len(valid_conf) > 0:
            actual_threshold = np.percentile(valid_conf, percentile)
            mask = (confidence >= actual_threshold) & np.isfinite(points).all(axis=1)

            filtered_points = points[mask]
            filtered_colors = colors[mask]

            # Update point cloud
            point_cloud_handle.points = filtered_points
            point_cloud_handle.colors = filtered_colors

            print(f"[NFN Visualization] Updated to {len(filtered_points)}/{len(points)} points (threshold: {conf_slider.value}%)")

    # Add visibility toggles
    show_gaps = server.gui.add_checkbox("Show Gaps", initial_value=True)
    show_viewpoints = server.gui.add_checkbox("Show Suggested Viewpoints", initial_value=True)
    show_existing_cameras = server.gui.add_checkbox("Show Existing Cameras", initial_value=True)

    @show_gaps.on_update
    def toggle_gaps(_):
        server.scene.set_node_visibility("/nfn/gaps", show_gaps.value)

    @show_viewpoints.on_update
    def toggle_viewpoints(_):
        server.scene.set_node_visibility("/nfn/suggested_viewpoints", show_viewpoints.value)
        server.scene.set_node_visibility("/nfn/viewpoint_lines", show_viewpoints.value)

    @show_existing_cameras.on_update
    def toggle_existing_cameras(_):
        server.scene.set_node_visibility("/nfn/existing_cameras", show_existing_cameras.value)


def _display_statistics(
    server: viser.ViserServer,
    predictions: Dict,
    gap_results: Dict,
    viewpoint_results: Dict
):
    """
    Display NFN statistics in GUI.
    """
    stats_text = "### NFN Statistics\n\n"

    # Point cloud stats
    world_points = predictions["world_points"]
    total_points = np.prod(world_points.shape[:-1])
    stats_text += f"**Total Points**: {total_points:,}\n\n"

    # Gap stats
    if gap_results:
        gap_stats = gap_results.get("statistics", {})
        num_gaps = gap_results.get("gap_count", 0)
        stats_text += f"**Detected Gaps**: {num_gaps}\n"

        if "gap_volume_m3" in gap_stats:
            stats_text += f"**Total Gap Volume**: {gap_stats['gap_volume_m3']:.2f} m³\n"

        if "occupancy_ratio" in gap_stats:
            stats_text += f"**Occupancy Ratio**: {gap_stats['occupancy_ratio']:.1%}\n\n"

    # Viewpoint stats
    if viewpoint_results:
        vp_stats = viewpoint_results.get("statistics", {})
        num_viewpoints = viewpoint_results.get("num_viewpoints", 0)
        stats_text += f"**Suggested Viewpoints**: {num_viewpoints}\n"

        if "viewpoints_per_gap_avg" in vp_stats:
            stats_text += f"**Avg Viewpoints/Gap**: {vp_stats['viewpoints_per_gap_avg']:.1f}\n"

    server.gui.add_markdown(stats_text)


# Example usage
if __name__ == "__main__":
    print("NFN Viser Viewer - Example usage")
    print("This module should be imported and used with VGGT predictions")
    print("See demo_gradio.py for integration example")
