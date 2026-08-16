#!/usr/bin/env python3
# Copyright (C) 2024 Carnegie Mellon University
"""
Confidence-based 3D mapping utilities for Real-time Drone Next Flight Navigation (NFN).
Generates confidence-colored point clouds to guide drone flight planning by identifying
areas with poor map quality.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import trimesh
import torch
from typing import Dict, Optional, Tuple, Union
import warnings

from swiftmap.core.primitives import geometry

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def generate_confidence_point_cloud(
    world_points: Union[torch.Tensor, np.ndarray],
    confidence: Union[torch.Tensor, np.ndarray], 
    conf_threshold: float = 0.6,
    colormap: str = "red_to_green",
    max_points: Optional[int] = 50000,  # Reduced for better 3D performance
    point_size: float = 0.002,
    save_ply: bool = True,
    ply_path: Optional[str] = None,
) -> Tuple[trimesh.Scene, Dict[str, float], Optional[str]]:
    """
    Generate a confidence-colored 3D point cloud for map quality visualization.
    
    Args:
        world_points: 3D world coordinates, shape [S, H, W, 3] or [N, 3]
        confidence: Confidence scores, shape [S, H, W] or [N,]
        conf_threshold: Percentile threshold as decimal (0.0-1.0). E.g., 0.6 filters out bottom 60% of points
        colormap: Matplotlib colormap name ('coolwarm', 'viridis', 'plasma', etc.)
        max_points: Maximum number of points to visualize (for performance)
        point_size: Visual size of points in the 3D scene
        save_ply: Generate PLY file with absolute confidence values alongside GLB scene
        
    Returns:
        Tuple of:
        - trimesh.Scene: 3D scene for GLB display
        - Dict: Statistics including coverage metrics and confidence distribution  
        - Optional[str]: PLY filename if save_ply=True, None otherwise
    """
    
    # Convert tensors to numpy if needed
    if isinstance(world_points, torch.Tensor):
        world_points = world_points.detach().cpu().numpy()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy()
    
    # Reshape to flat arrays if needed
    if world_points.ndim == 4:  # [S, H, W, 3]
        S, H, W, _ = world_points.shape
        world_points = world_points.reshape(-1, 3)
        confidence = confidence.reshape(-1)
    elif world_points.ndim == 3:  # [H, W, 3]
        H, W, _ = world_points.shape  
        world_points = world_points.reshape(-1, 3)
        confidence = confidence.reshape(-1)
    
    # Filter out invalid points (NaN, inf)
    valid_mask = (
        np.isfinite(world_points).all(axis=1) & 
        np.isfinite(confidence) &
        (confidence > 0)
    )
    
    world_points = world_points[valid_mask]
    confidence = confidence[valid_mask]
    
    if len(world_points) == 0:
        print("Warning: No valid points found after filtering")
        empty_scene = trimesh.Scene()
        stats = {
            "total_points": 0,
            "high_conf_points": 0,
            "coverage_ratio": 0.0,
            "mean_confidence": 0.0,
            "confidence_std": 0.0
        }
        return empty_scene, stats
    
    # Apply confidence threshold using percentile-based filtering (same as main 3D scene)
    # conf_threshold is expected to be a decimal (0.0-1.0) representing percentile
    # e.g., 0.6 means keep top 40% of points (filter bottom 60%)
    # conf_threshold is a decimal (0-1); geometry.confidence_mask takes a percentile.
    high_conf_mask = geometry.confidence_mask(confidence, conf_threshold * 100)
    filtered_points = world_points[high_conf_mask]
    filtered_conf = confidence[high_conf_mask]
    
    print(f"Points after confidence filtering: {len(filtered_points)}/{len(world_points)} "
          f"(percentile: {conf_threshold*100:.1f}%)")
    
    # Subsample for performance if needed
    if max_points is not None and len(filtered_points) > max_points:
        indices = np.random.choice(len(filtered_points), max_points, replace=False)
        filtered_points = filtered_points[indices]
        filtered_conf = filtered_conf[indices]
        print(f"Subsampled to {max_points} points for visualization")
    
    if len(filtered_points) == 0:
        print(f"Warning: No points above percentile threshold {conf_threshold*100:.1f}% ")
        empty_scene = trimesh.Scene()
        stats = {
            "total_points": len(world_points),
            "high_conf_points": 0,
            "coverage_ratio": 0.0,
            "mean_confidence": float(np.mean(confidence)),
            "confidence_std": float(np.std(confidence))
        }
        return empty_scene, stats
    
    # Generate colors based on confidence
    colors = confidence_to_colors(filtered_conf, colormap=colormap)
    
    # Flip the point cloud coordinates to match GLB transformation
    # This fixes the "upside down" issue in the confidence viewer
    flipped_points = filtered_points.copy()
    flipped_points[:, 2] = -flipped_points[:, 2]  # Flip Z coordinate
    # Or alternatively flip Y coordinate if Z doesn't work:
    flipped_points[:, 1] = -flipped_points[:, 1]  # Flip Y coordinate
    
    # Create point cloud
    point_cloud = trimesh.points.PointCloud(
        vertices=flipped_points,
        colors=colors
    )
    
    # Create scene with consistent camera view to match main 3D model
    scene = trimesh.Scene()
    scene.add_geometry(point_cloud, node_name="confidence_points")
    
    # Camera setup is now handled by Gradio Model3D camera_position parameter
    # The coordinate flip above should resolve the orientation mismatch
    
    # Add confidence colorbar as a separate geometry (optional)
    # Map custom colormaps to matplotlib equivalents for colorbar display
    if colormap == "red_to_green":
        colorbar_colormap = "RdYlGn"
    elif colormap == "heatmap":
        colorbar_colormap = "coolwarm"  # Blue-Red colormap that matches our heatmap concept
    else:
        colorbar_colormap = colormap
    colorbar_mesh = create_confidence_colorbar(
        conf_min=float(np.min(filtered_conf)),
        conf_max=float(np.max(filtered_conf)),
        colormap=colorbar_colormap
    )
    if colorbar_mesh is not None:
        scene.add_geometry(colorbar_mesh, node_name="confidence_colorbar")
    
    # Calculate statistics
    stats = {
        "total_points": len(world_points),
        "high_conf_points": len(filtered_points),
        "coverage_ratio": len(filtered_points) / len(world_points),
        "mean_confidence": float(np.mean(confidence)),
        "confidence_std": float(np.std(confidence)),
        "filtered_mean_confidence": float(np.mean(filtered_conf)) if len(filtered_conf) > 0 else 0.0,
        "filtered_confidence_std": float(np.std(filtered_conf)) if len(filtered_conf) > 0 else 0.0
    }
    
    # Optionally save PLY file with absolute confidence values
    ply_filename = None
    if save_ply:
        ply_filename = save_confidence_point_cloud_ply(
            points=flipped_points,
            confidence_values=filtered_conf,
            colors=colors,
            stats=stats,
            ply_path=ply_path,
        )
        
    # Always return GLB scene for display, plus optional PLY filename
    return scene, stats, ply_filename


def confidence_to_colors(
    confidence: np.ndarray, 
    colormap: str = "RdYlGn"
) -> np.ndarray:
    """
    Convert confidence scores to RGB colors using red-to-green quality mapping.
    
    Args:
        confidence: Confidence scores array (should be post-filtered)
        colormap: Colormap name (default: "RdYlGn" for red-yellow-green)
        
    Returns:
        RGB colors array with shape [N, 4] (RGBA)
    """
    # Post-filter normalization: normalize the filtered confidence values to [0, 1]
    # This ensures full color range utilization for the remaining points
    conf_min, conf_max = np.min(confidence), np.max(confidence)
    if conf_max > conf_min:
        normalized_conf = (confidence - conf_min) / (conf_max - conf_min)
    else:
        normalized_conf = np.ones_like(confidence)
    
    # Use different colormap schemes for quality visualization
    if colormap == "red_to_green":
        # Custom red-to-green mapping (legacy)
        colors = np.zeros((len(normalized_conf), 4), dtype=np.uint8)
        for i, conf in enumerate(normalized_conf):
            # Red to Green transition: (1-conf, conf, 0, 1)
            red = int((1 - conf) * 255)
            green = int(conf * 255)
            blue = 0
            alpha = 255
            colors[i] = [red, green, blue, alpha]
    elif colormap == "heatmap":
        # Professional heatmap: Blue -> Cyan -> Yellow -> Red
        # This provides excellent contrast and is colorblind-friendly
        colors = np.zeros((len(normalized_conf), 4), dtype=np.uint8)
        for i, conf in enumerate(normalized_conf):
            # Invert confidence so low confidence = hot colors (red)
            # High confidence = cool colors (blue)
            heat = 1.0 - conf  # 0 = high confidence (blue), 1 = low confidence (red)
            
            if heat <= 0.25:  # High confidence: Blue to Cyan
                ratio = heat / 0.25
                red = 0
                green = int(ratio * 255)
                blue = 255
            elif heat <= 0.5:  # Medium-high confidence: Cyan to Green  
                ratio = (heat - 0.25) / 0.25
                red = 0
                green = 255
                blue = int((1 - ratio) * 255)
            elif heat <= 0.75:  # Medium-low confidence: Green to Yellow
                ratio = (heat - 0.5) / 0.25
                red = int(ratio * 255)
                green = 255
                blue = 0
            else:  # Low confidence: Yellow to Red
                ratio = (heat - 0.75) / 0.25
                red = 255
                green = int((1 - ratio) * 255)
                blue = 0
            
            alpha = 255
            colors[i] = [red, green, blue, alpha]
    else:
        # Use matplotlib colormap for other options
        cmap = cm.get_cmap(colormap)
        colors = cmap(normalized_conf)
        # Convert to 0-255 RGBA
        colors = (colors * 255).astype(np.uint8)
    
    return colors


def create_confidence_colorbar(
    conf_min: float,
    conf_max: float, 
    colormap: str = "coolwarm",
    height: float = 0.5,
    width: float = 0.05,
    position: np.ndarray = np.array([0, 0, 2])
) -> Optional[trimesh.Trimesh]:
    """
    Create a 3D colorbar mesh to show confidence scale.
    
    Args:
        conf_min: Minimum confidence value
        conf_max: Maximum confidence value
        colormap: Matplotlib colormap name
        height: Height of the colorbar
        width: Width of the colorbar
        position: 3D position of the colorbar
        
    Returns:
        Trimesh object representing the colorbar, or None if creation fails
    """
    try:
        # Create a rectangular mesh for the colorbar
        n_segments = 50
        vertices = []
        faces = []
        colors = []
        
        for i in range(n_segments):
            y_bottom = (i / n_segments) * height + position[1]
            y_top = ((i + 1) / n_segments) * height + position[1]
            
            # Create quad vertices
            quad_vertices = [
                [position[0], y_bottom, position[2]],           # bottom-left
                [position[0] + width, y_bottom, position[2]],   # bottom-right
                [position[0] + width, y_top, position[2]],      # top-right
                [position[0], y_top, position[2]]               # top-left
            ]
            
            vertex_start = len(vertices)
            vertices.extend(quad_vertices)
            
            # Create two triangles for the quad
            faces.extend([
                [vertex_start, vertex_start + 1, vertex_start + 2],
                [vertex_start, vertex_start + 2, vertex_start + 3]
            ])
            
            # Color based on position
            conf_value = conf_min + (i / n_segments) * (conf_max - conf_min)
            normalized_conf = (conf_value - conf_min) / (conf_max - conf_min) if conf_max > conf_min else 0.5
            
            cmap = cm.get_cmap(colormap)
            color = cmap(normalized_conf)
            color_rgba = (np.array(color) * 255).astype(np.uint8)
            
            # Apply same color to all vertices of the quad
            colors.extend([color_rgba] * 4)
        
        # Create mesh
        colorbar_mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            vertex_colors=np.array(colors)
        )
        
        return colorbar_mesh
        
    except Exception as e:
        print(f"Warning: Could not create colorbar mesh: {e}")
        return None


def analyze_confidence_distribution(
    confidence: Union[torch.Tensor, np.ndarray],
    conf_threshold: float = 0.6
) -> Dict[str, float]:
    """
    Analyze confidence score distribution for map quality assessment.
    
    Args:
        confidence: Confidence scores array
        conf_threshold: Percentile threshold as decimal (0.0-1.0) for quality assessment
        
    Returns:
        Dictionary with distribution statistics
    """
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy()
    
    # Flatten if multi-dimensional
    confidence = confidence.flatten()
    
    # Filter valid values
    valid_conf = confidence[np.isfinite(confidence) & (confidence > 0)]
    
    if len(valid_conf) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "high_conf_ratio": 0.0,
            "percentile_25": 0.0,
            "percentile_50": 0.0,
            "percentile_75": 0.0,
            "percentile_95": 0.0
        }
    
    # Calculate actual threshold using percentile-based approach
    if conf_threshold == 0.0:
        actual_threshold = 0.0
    else:
        percentile = conf_threshold * 100
        actual_threshold = np.percentile(valid_conf, percentile)
    
    return {
        "mean": float(np.mean(valid_conf)),
        "std": float(np.std(valid_conf)),
        "min": float(np.min(valid_conf)),
        "max": float(np.max(valid_conf)),
        "high_conf_ratio": float(np.sum(valid_conf >= actual_threshold) / len(valid_conf)),
        "percentile_25": float(np.percentile(valid_conf, 25)),
        "percentile_50": float(np.percentile(valid_conf, 50)),
        "percentile_75": float(np.percentile(valid_conf, 75)),
        "percentile_95": float(np.percentile(valid_conf, 95)),
        "actual_threshold": float(actual_threshold),
        "percentile_threshold": float(conf_threshold * 100)
    }


def save_confidence_point_cloud_ply(
    points: np.ndarray,
    confidence_values: np.ndarray,
    colors: np.ndarray,
    stats: Dict[str, float],
    ply_path: Optional[str] = None,
    output_dir: str = "output/",
    filename_prefix: str = "confidence_map"
) -> str:
    """
    Save confidence point cloud as PLY file with absolute confidence values.

    Args:
        points: 3D points array [N, 3]
        confidence_values: Absolute confidence values [N,]
        colors: RGBA color array [N, 4]
        stats: Statistics dictionary from confidence analysis
        ply_path: Exact output path. If given, the file is written there directly
            (no staging dir). Otherwise a timestamped name under ``output_dir``.
        output_dir: Fallback directory when ``ply_path`` is not given.
        filename_prefix: Prefix for the fallback filename.

    Returns:
        str: Path to saved PLY file
    """
    import os
    from datetime import datetime

    if ply_path:
        os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
        ply_filename = ply_path
    else:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ply_filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.ply")

    comments = [
        f"NFN Confidence Map - {datetime.now().isoformat()}",
        f"Total Points: {stats['total_points']}",
        f"High Confidence Points: {stats['high_conf_points']}",
        f"Coverage Ratio: {stats['coverage_ratio']:.3f}",
        f"Mean Confidence: {stats['mean_confidence']:.4f}",
        f"Filtered Mean Confidence: {stats['filtered_mean_confidence']:.4f}",
    ]
    geometry.write_ply(ply_filename, points, colors, confidence=confidence_values, comments=comments)

    print(f"[NFN] Saved confidence PLY file: {ply_filename}")
    print(f"[NFN] Points: {len(points)}, Mean confidence: {stats['filtered_mean_confidence']:.4f}")
    return ply_filename