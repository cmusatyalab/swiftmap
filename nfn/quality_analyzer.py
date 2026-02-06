"""
Quality Assessment Module

Analyzes 3D point clouds to identify low-quality regions that require additional
drone flight coverage based on multiple quality metrics.

Key Features:
- Multi-metric quality scoring (confidence, density, normal consistency, view diversity)
- Voxel-based spatial analysis for efficient computation
- Identifies low-quality zones for targeted re-mapping
- Generates quality heatmap for visualization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from scipy import ndimage


class QualityAnalyzer:
    """
    Analyzes point cloud quality using multiple metrics to identify regions
    requiring additional coverage.
    """

    def __init__(
        self,
        voxel_size: float = 0.5,
        quality_threshold: float = 0.5,
        min_region_voxels: int = 10
    ):
        """
        Initialize quality analyzer.

        Args:
            voxel_size: Size of each voxel in meters (default: 0.5m)
            quality_threshold: Quality score threshold (0-1, lower = worse quality)
            min_region_voxels: Minimum voxels for a low-quality region
        """
        self.voxel_size = voxel_size
        self.quality_threshold = quality_threshold
        self.min_region_voxels = min_region_voxels

        # Results storage
        self.voxel_grid = None
        self.grid_origin = None
        self.grid_dims = None
        self.quality_map = None
        self.low_quality_regions = []

    def analyze_quality(
        self,
        world_points: np.ndarray,
        confidence: np.ndarray,
        normals: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze point cloud quality using multiple metrics.

        Args:
            world_points: 3D world coordinates, shape [S, H, W, 3] or [N, 3]
            confidence: Confidence scores, shape [S, H, W] or [N]
            normals: Optional surface normals, shape [S, H, W, 3] or [N, 3]
            extrinsics: Optional camera extrinsics for view diversity, shape [S, 3, 4]

        Returns:
            Dictionary containing:
            - low_quality_regions: List of low-quality region bounding boxes
            - region_centers: Centers of low-quality regions
            - quality_map: Per-voxel quality scores
            - statistics: Quality analysis statistics
        """
        start_time = time.time()

        # Prepare points
        points, conf_values, normal_values = self._prepare_data(
            world_points, confidence, normals
        )

        if len(points) == 0:
            return self._empty_result("No valid points after filtering")

        # Build voxel grid
        self._build_voxel_grid(points)

        # Calculate quality metrics for each voxel
        self._calculate_voxel_quality(
            points, conf_values, normal_values, extrinsics
        )

        # Identify low-quality regions
        self._detect_low_quality_regions()

        # Calculate statistics
        stats = self._calculate_statistics(points)
        stats["processing_time"] = time.time() - start_time

        # Prepare results
        region_centers = [r["world_center"] for r in self.low_quality_regions]
        region_bboxes = [r["world_bbox"] for r in self.low_quality_regions]

        results = {
            "low_quality_regions": region_bboxes,
            "region_centers": region_centers,
            "region_count": len(self.low_quality_regions),
            "quality_map": self.quality_map,
            "statistics": stats,
            "voxel_grid": self.voxel_grid,
            "grid_origin": self.grid_origin,
            "grid_dims": self.grid_dims,
            "voxel_size": self.voxel_size
        }

        print(f"[NFN Quality Analysis] Found {len(self.low_quality_regions)} low-quality regions in {stats['processing_time']:.2f}s")

        return results

    def _prepare_data(
        self,
        world_points: np.ndarray,
        confidence: np.ndarray,
        normals: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare and filter point cloud data.
        """
        # Reshape to flat arrays
        if world_points.ndim == 4:  # [S, H, W, 3]
            world_points = world_points.reshape(-1, 3)
            confidence = confidence.reshape(-1)
            if normals is not None:
                normals = normals.reshape(-1, 3)
        elif world_points.ndim == 3:  # [H, W, 3]
            world_points = world_points.reshape(-1, 3)
            confidence = confidence.reshape(-1)
            if normals is not None:
                normals = normals.reshape(-1, 3)

        # Filter invalid points
        valid_mask = np.isfinite(world_points).all(axis=1) & np.isfinite(confidence)

        points = world_points[valid_mask]
        conf_values = confidence[valid_mask]
        normal_values = normals[valid_mask] if normals is not None else None

        print(f"[NFN Quality Analysis] Using {len(points)}/{len(world_points)} valid points")

        return points, conf_values, normal_values

    def _build_voxel_grid(self, points: np.ndarray):
        """
        Build 3D voxel grid from point cloud.
        """
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Small padding
        padding = self.voxel_size
        min_coords -= padding
        max_coords += padding

        # Calculate grid dimensions
        grid_size = max_coords - min_coords
        grid_dims = np.ceil(grid_size / self.voxel_size).astype(int)

        # Limit grid size
        max_dim = 200
        if np.any(grid_dims > max_dim):
            scale = max_dim / np.max(grid_dims)
            self.voxel_size = self.voxel_size / scale
            grid_dims = np.ceil(grid_size / self.voxel_size).astype(int)
            print(f"[NFN Quality Analysis] Adjusted voxel size to {self.voxel_size:.3f}m")

        self.voxel_grid = np.zeros(grid_dims, dtype=np.int32)  # Store point counts
        self.quality_map = np.zeros(grid_dims, dtype=np.float32)  # Store quality scores
        self.grid_origin = min_coords
        self.grid_dims = grid_dims

        print(f"[NFN Quality Analysis] Voxel grid: {grid_dims}, voxel size: {self.voxel_size:.3f}m")

    def _calculate_voxel_quality(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
        normals: Optional[np.ndarray],
        extrinsics: Optional[np.ndarray]
    ):
        """
        Calculate quality score for each voxel using multiple metrics.
        """
        # Get voxel indices for each point
        voxel_indices = np.floor((points - self.grid_origin) / self.voxel_size).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, self.grid_dims - 1)

        # Initialize metric storage
        voxel_conf_sum = np.zeros(self.grid_dims, dtype=np.float32)
        voxel_conf_count = np.zeros(self.grid_dims, dtype=np.int32)
        voxel_point_count = np.zeros(self.grid_dims, dtype=np.int32)

        # Metric 1: Confidence score (weight: 0.4)
        for i, (vi, conf) in enumerate(zip(voxel_indices, confidence)):
            vx, vy, vz = vi
            voxel_conf_sum[vx, vy, vz] += conf
            voxel_conf_count[vx, vy, vz] += 1
            voxel_point_count[vx, vy, vz] += 1

        # Average confidence per voxel
        conf_mask = voxel_conf_count > 0
        voxel_conf_avg = np.zeros_like(voxel_conf_sum)
        voxel_conf_avg[conf_mask] = voxel_conf_sum[conf_mask] / voxel_conf_count[conf_mask]

        # Normalize confidence to [0, 1]
        if voxel_conf_avg.max() > 0:
            conf_score = voxel_conf_avg / voxel_conf_avg.max()
        else:
            conf_score = voxel_conf_avg

        # Metric 2: Point density (weight: 0.3)
        # Higher density = better quality
        max_density = voxel_point_count.max()
        if max_density > 0:
            density_score = voxel_point_count / max_density
        else:
            density_score = np.zeros_like(voxel_point_count, dtype=np.float32)

        # Metric 3: Surface normal consistency (weight: 0.2)
        if normals is not None:
            normal_score = self._calculate_normal_consistency(
                voxel_indices, normals
            )
        else:
            normal_score = np.ones(self.grid_dims, dtype=np.float32)

        # Metric 4: View diversity (weight: 0.1)
        # For now, use uniform score (can be enhanced with extrinsics)
        view_score = np.ones(self.grid_dims, dtype=np.float32)

        # Combined quality score (weighted sum)
        self.quality_map = (
            0.4 * conf_score +
            0.3 * density_score +
            0.2 * normal_score +
            0.1 * view_score
        )

        # Store point counts
        self.voxel_grid = voxel_point_count

        occupied_voxels = np.sum(self.voxel_grid > 0)
        avg_quality = np.mean(self.quality_map[self.voxel_grid > 0]) if occupied_voxels > 0 else 0

        print(f"[NFN Quality Analysis] Average quality score: {avg_quality:.3f}")

    def _calculate_normal_consistency(
        self,
        voxel_indices: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        """
        Calculate surface normal consistency for each voxel.
        Higher consistency = better geometry quality.
        """
        normal_consistency = np.zeros(self.grid_dims, dtype=np.float32)
        voxel_normal_lists = {}

        # Group normals by voxel
        for vi, normal in zip(voxel_indices, normals):
            vx, vy, vz = tuple(vi)
            if (vx, vy, vz) not in voxel_normal_lists:
                voxel_normal_lists[(vx, vy, vz)] = []
            voxel_normal_lists[(vx, vy, vz)].append(normal)

        # Calculate consistency for each voxel
        for (vx, vy, vz), normal_list in voxel_normal_lists.items():
            if len(normal_list) < 2:
                normal_consistency[vx, vy, vz] = 1.0  # Single normal, assume good
                continue

            normals_array = np.array(normal_list)
            # Normalize normals
            norms = np.linalg.norm(normals_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            normals_array = normals_array / norms

            # Calculate pairwise dot products (cosine similarity)
            mean_normal = np.mean(normals_array, axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)

            # Consistency = average alignment with mean normal
            dot_products = np.abs(np.dot(normals_array, mean_normal))
            consistency = np.mean(dot_products)

            normal_consistency[vx, vy, vz] = consistency

        return normal_consistency

    def _detect_low_quality_regions(self):
        """
        Identify connected regions with quality below threshold.
        """
        # Find voxels below quality threshold
        low_quality_mask = (self.quality_map < self.quality_threshold) & (self.voxel_grid > 0)

        # Connected component labeling
        labeled_array, num_features = ndimage.label(low_quality_mask)

        self.low_quality_regions = []

        for label_id in range(1, num_features + 1):
            region_mask = (labeled_array == label_id)
            voxel_count = np.sum(region_mask)

            # Filter small regions
            if voxel_count < self.min_region_voxels:
                continue

            # Get region voxels
            region_voxels = np.argwhere(region_mask)

            # Calculate bounding box
            voxel_min = np.min(region_voxels, axis=0)
            voxel_max = np.max(region_voxels, axis=0)
            voxel_center = (voxel_min + voxel_max) / 2.0

            # Convert to world coordinates
            world_min = self._voxel_to_world(voxel_min)
            world_max = self._voxel_to_world(voxel_max)
            world_center = self._voxel_to_world(voxel_center)

            # Calculate average quality in this region
            region_quality = np.mean(self.quality_map[region_mask])

            region_info = {
                "label_id": label_id,
                "voxel_count": voxel_count,
                "voxel_bbox": (voxel_min, voxel_max),
                "voxel_center": voxel_center,
                "world_bbox": (world_min, world_max),
                "world_center": world_center,
                "average_quality": region_quality,
                "volume": voxel_count * (self.voxel_size ** 3)
            }

            self.low_quality_regions.append(region_info)

        # Sort by quality (worst first)
        self.low_quality_regions.sort(key=lambda x: x["average_quality"])

        print(f"[NFN Quality Analysis] Detected {len(self.low_quality_regions)} low-quality regions")

    def _voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates to world coordinates."""
        return self.grid_origin + voxel_coords * self.voxel_size

    def _calculate_statistics(self, points: np.ndarray) -> Dict:
        """Calculate quality analysis statistics."""
        occupied_voxels = np.sum(self.voxel_grid > 0)

        if occupied_voxels > 0:
            occupied_mask = self.voxel_grid > 0
            avg_quality = np.mean(self.quality_map[occupied_mask])
            min_quality = np.min(self.quality_map[occupied_mask])
            max_quality = np.max(self.quality_map[occupied_mask])

            # Quality distribution
            low_quality_count = np.sum((self.quality_map < 0.3) & occupied_mask)
            medium_quality_count = np.sum((self.quality_map >= 0.3) & (self.quality_map < 0.7) & occupied_mask)
            high_quality_count = np.sum((self.quality_map >= 0.7) & occupied_mask)
        else:
            avg_quality = min_quality = max_quality = 0.0
            low_quality_count = medium_quality_count = high_quality_count = 0

        total_volume = sum(r["volume"] for r in self.low_quality_regions)

        stats = {
            "total_points": len(points),
            "voxel_size": self.voxel_size,
            "occupied_voxels": int(occupied_voxels),
            "average_quality": float(avg_quality),
            "min_quality": float(min_quality),
            "max_quality": float(max_quality),
            "low_quality_voxels": int(low_quality_count),
            "medium_quality_voxels": int(medium_quality_count),
            "high_quality_voxels": int(high_quality_count),
            "num_low_quality_regions": len(self.low_quality_regions),
            "total_low_quality_volume_m3": float(total_volume)
        }

        return stats

    def _empty_result(self, message: str) -> Dict:
        """Return empty result with error message."""
        print(f"[NFN Quality Analysis] {message}")
        return {
            "low_quality_regions": [],
            "region_centers": [],
            "region_count": 0,
            "quality_map": None,
            "statistics": {"error": message},
            "voxel_grid": None,
            "grid_origin": None,
            "grid_dims": None,
            "voxel_size": self.voxel_size
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing QualityAnalyzer...")

    # Create synthetic point cloud with varying quality
    np.random.seed(42)

    # High quality region (high confidence, high density)
    high_quality_points = np.random.randn(2000, 3) * 0.3 + np.array([0, 0, 0])
    high_quality_conf = np.random.uniform(0.7, 1.0, 2000)

    # Low quality region (low confidence, low density)
    low_quality_points = np.random.randn(500, 3) * 0.3 + np.array([3, 0, 0])
    low_quality_conf = np.random.uniform(0.1, 0.4, 500)

    # Combine
    points = np.vstack([high_quality_points, low_quality_points])
    confidence = np.concatenate([high_quality_conf, low_quality_conf])

    # Initialize analyzer
    analyzer = QualityAnalyzer(voxel_size=0.3, quality_threshold=0.5)

    # Analyze quality
    results = analyzer.analyze_quality(points, confidence)

    print(f"\nQuality Analysis Results:")
    print(f"  Found {results['region_count']} low-quality regions")
    print(f"  Statistics: {results['statistics']}")

    for i, region in enumerate(results['low_quality_regions'][:3]):  # Show first 3
        world_min, world_max = region
        print(f"  Region {i}: center={np.mean([world_min, world_max], axis=0)}, quality=low")

    print("\nQualityAnalyzer test completed successfully!")
