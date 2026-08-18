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

from swiftmap.core.pipeline.utils import geometry
from swiftmap.core.database import cloud as arrays

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def generate_confidence_point_cloud(
    world_points: Union[torch.Tensor, np.ndarray],
    confidence: Union[torch.Tensor, np.ndarray], 
    conf_threshold: float = 0.6,
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
    high_conf_mask = arrays.confidence_mask(confidence, conf_threshold * 100)
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
    colors = confidence_to_colors(filtered_conf)
    
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


def confidence_to_colors(confidence: np.ndarray) -> np.ndarray:
    """Confidence -> RGBA, red (low) to green (high), normalised over the given values."""
    lo, hi = np.min(confidence), np.max(confidence)
    norm = (confidence - lo) / (hi - lo) if hi > lo else np.ones_like(confidence)
    colors = np.zeros((len(norm), 4), dtype=np.uint8)
    colors[:, 0] = ((1.0 - norm) * 255).astype(np.uint8)
    colors[:, 1] = (norm * 255).astype(np.uint8)
    colors[:, 3] = 255
    return colors


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