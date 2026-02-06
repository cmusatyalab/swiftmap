"""
Coverage Gap Detection Module

Analyzes 3D point clouds to identify unmapped regions (coverage gaps) that
require additional drone flight coverage.

Key Features:
- 3D voxel grid analysis for efficient spatial queries
- Frontier region detection (boundaries between mapped and unmapped areas)
- Noise filtering to keep only significant gaps
- Bounding box generation for visualization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class CoverageGapDetector:
    """
    Detects coverage gaps in 3D reconstructions using voxel-based spatial analysis.
    """

    def __init__(self, voxel_size: float = 0.5, min_gap_voxels: int = 8):
        """
        Initialize gap detector.

        Args:
            voxel_size: Size of each voxel in meters (default: 0.5m)
            min_gap_voxels: Minimum number of connected empty voxels to be considered a gap
        """
        self.voxel_size = voxel_size
        self.min_gap_voxels = min_gap_voxels

        # Results storage
        self.voxel_grid = None
        self.grid_origin = None
        self.grid_dims = None
        self.gap_regions = []

    def analyze_coverage_gaps(
        self,
        world_points: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        conf_threshold: float = 0.6
    ) -> Dict:
        """
        Analyze point cloud to detect coverage gaps.

        Args:
            world_points: 3D world coordinates, shape [S, H, W, 3] or [N, 3]
            confidence: Optional confidence scores for filtering
            conf_threshold: Confidence threshold for point filtering (percentile, 0.0-1.0)

        Returns:
            Dictionary containing:
            - gap_regions: List of gap bounding boxes
            - gap_centers: Centers of gap regions
            - statistics: Gap detection statistics
            - voxel_grid: The computed voxel occupancy grid
        """
        start_time = time.time()

        # Flatten and filter points
        points = self._prepare_points(world_points, confidence, conf_threshold)

        if len(points) == 0:
            return self._empty_result("No valid points after filtering")

        # Build voxel grid
        self._build_voxel_grid(points)

        # Detect gap regions
        self._detect_gaps()

        # Calculate statistics
        stats = self._calculate_statistics(points)
        stats["processing_time"] = time.time() - start_time

        # Prepare results
        gap_centers = [self._voxel_to_world(bbox["voxel_center"]) for bbox in self.gap_regions]
        gap_bboxes = [bbox["world_bbox"] for bbox in self.gap_regions]

        results = {
            "gap_regions": gap_bboxes,
            "gap_centers": gap_centers,
            "gap_count": len(self.gap_regions),
            "statistics": stats,
            "voxel_grid": self.voxel_grid,
            "grid_origin": self.grid_origin,
            "grid_dims": self.grid_dims,
            "voxel_size": self.voxel_size
        }

        print(f"[NFN Gap Detection] Found {len(self.gap_regions)} gap regions in {stats['processing_time']:.2f}s")

        return results

    def _prepare_points(
        self,
        world_points: np.ndarray,
        confidence: Optional[np.ndarray],
        conf_threshold: float
    ) -> np.ndarray:
        """
        Prepare and filter point cloud data.
        """
        # Reshape to flat array if needed
        if world_points.ndim == 4:  # [S, H, W, 3]
            world_points = world_points.reshape(-1, 3)
        elif world_points.ndim == 3:  # [H, W, 3]
            world_points = world_points.reshape(-1, 3)

        # Filter invalid points
        valid_mask = np.isfinite(world_points).all(axis=1)

        # Apply confidence filtering if provided
        if confidence is not None:
            if confidence.ndim > 1:
                confidence = confidence.flatten()

            # Apply percentile-based threshold
            if conf_threshold > 0.0:
                percentile = conf_threshold * 100
                actual_threshold = np.percentile(confidence[valid_mask], percentile)
                conf_mask = confidence >= actual_threshold
                valid_mask = valid_mask & conf_mask

        points = world_points[valid_mask]

        print(f"[NFN Gap Detection] Using {len(points)}/{len(world_points)} points after filtering")

        return points

    def _build_voxel_grid(self, points: np.ndarray):
        """
        Build 3D voxel occupancy grid from point cloud.
        """
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Add padding to detect gaps around the edges
        padding = 2 * self.voxel_size
        min_coords -= padding
        max_coords += padding

        # Calculate grid dimensions
        grid_size = max_coords - min_coords
        grid_dims = np.ceil(grid_size / self.voxel_size).astype(int)

        # Limit grid size to prevent memory issues (max 200^3 voxels = 8M voxels)
        max_dim = 200
        if np.any(grid_dims > max_dim):
            scale = max_dim / np.max(grid_dims)
            self.voxel_size = self.voxel_size / scale
            grid_dims = np.ceil(grid_size / self.voxel_size).astype(int)
            print(f"[NFN Gap Detection] Adjusted voxel size to {self.voxel_size:.3f}m to fit grid")

        # Initialize empty grid (0 = empty, 1 = occupied)
        voxel_grid = np.zeros(grid_dims, dtype=np.uint8)

        # Populate grid with occupied voxels
        voxel_indices = np.floor((points - min_coords) / self.voxel_size).astype(int)

        # Clip to grid bounds (safety check)
        voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)

        # Mark occupied voxels
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

        self.voxel_grid = voxel_grid
        self.grid_origin = min_coords
        self.grid_dims = grid_dims

        occupied_count = np.sum(voxel_grid)
        total_voxels = np.prod(grid_dims)

        print(f"[NFN Gap Detection] Voxel grid: {grid_dims} ({occupied_count}/{total_voxels} occupied, {occupied_count/total_voxels*100:.1f}%)")

    def _detect_gaps(self):
        """
        Detect gap regions in the voxel grid using connected component analysis.
        """
        # Find empty voxels
        empty_grid = (self.voxel_grid == 0).astype(np.uint8)

        # Erode occupied regions to find internal gaps (frontier detection)
        # We look for empty voxels that are adjacent to occupied voxels
        from scipy import ndimage

        # Dilate occupied voxels to find frontier regions
        occupied_dilated = ndimage.binary_dilation(self.voxel_grid, iterations=1)

        # Frontier = empty voxels adjacent to occupied voxels
        frontier_mask = empty_grid & occupied_dilated

        # Find connected components in frontier regions
        labeled_array, num_features = ndimage.label(frontier_mask)

        # Analyze each gap region
        self.gap_regions = []

        for label_id in range(1, num_features + 1):
            region_mask = (labeled_array == label_id)
            voxel_count = np.sum(region_mask)

            # Filter small gaps (noise)
            if voxel_count < self.min_gap_voxels:
                continue

            # Get voxel coordinates of this gap
            gap_voxels = np.argwhere(region_mask)

            # Calculate bounding box in voxel coordinates
            voxel_min = np.min(gap_voxels, axis=0)
            voxel_max = np.max(gap_voxels, axis=0)
            voxel_center = (voxel_min + voxel_max) / 2.0

            # Convert to world coordinates
            world_min = self._voxel_to_world(voxel_min)
            world_max = self._voxel_to_world(voxel_max)
            world_center = self._voxel_to_world(voxel_center)

            gap_info = {
                "label_id": label_id,
                "voxel_count": voxel_count,
                "voxel_bbox": (voxel_min, voxel_max),
                "voxel_center": voxel_center,
                "world_bbox": (world_min, world_max),
                "world_center": world_center,
                "volume": voxel_count * (self.voxel_size ** 3)
            }

            self.gap_regions.append(gap_info)

        # Sort gaps by volume (largest first)
        self.gap_regions.sort(key=lambda x: x["volume"], reverse=True)

        print(f"[NFN Gap Detection] Detected {len(self.gap_regions)} gap regions (filtered from {num_features} candidates)")

    def _voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel coordinates to world coordinates.
        """
        return self.grid_origin + voxel_coords * self.voxel_size

    def _calculate_statistics(self, points: np.ndarray) -> Dict:
        """
        Calculate gap detection statistics.
        """
        total_volume = np.prod(self.grid_dims) * (self.voxel_size ** 3)
        occupied_voxels = np.sum(self.voxel_grid)
        occupied_volume = occupied_voxels * (self.voxel_size ** 3)

        gap_volume = sum(gap["volume"] for gap in self.gap_regions)

        stats = {
            "total_points": len(points),
            "voxel_size": self.voxel_size,
            "grid_dimensions": self.grid_dims.tolist(),
            "total_voxels": int(np.prod(self.grid_dims)),
            "occupied_voxels": int(occupied_voxels),
            "occupancy_ratio": float(occupied_voxels / np.prod(self.grid_dims)),
            "total_volume_m3": float(total_volume),
            "occupied_volume_m3": float(occupied_volume),
            "gap_volume_m3": float(gap_volume),
            "num_gaps": len(self.gap_regions),
            "largest_gap_volume_m3": float(self.gap_regions[0]["volume"]) if self.gap_regions else 0.0
        }

        return stats

    def _empty_result(self, message: str) -> Dict:
        """
        Return empty result with error message.
        """
        print(f"[NFN Gap Detection] {message}")
        return {
            "gap_regions": [],
            "gap_centers": [],
            "gap_count": 0,
            "statistics": {"error": message},
            "voxel_grid": None,
            "grid_origin": None,
            "grid_dims": None,
            "voxel_size": self.voxel_size
        }

    def get_gap_visualization_data(self) -> List[Dict]:
        """
        Get gap regions formatted for visualization.

        Returns:
            List of dictionaries with visualization data for each gap
        """
        viz_data = []

        for i, gap in enumerate(self.gap_regions):
            world_min, world_max = gap["world_bbox"]
            world_center = gap["world_center"]

            viz_data.append({
                "gap_id": i,
                "center": world_center,
                "min_corner": world_min,
                "max_corner": world_max,
                "size": world_max - world_min,
                "volume": gap["volume"],
                "voxel_count": gap["voxel_count"]
            })

        return viz_data


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing CoverageGapDetector...")

    # Create a synthetic point cloud with a gap in the middle
    np.random.seed(42)

    # Left cluster
    left_points = np.random.randn(1000, 3) * 0.5 + np.array([-2, 0, 0])

    # Right cluster
    right_points = np.random.randn(1000, 3) * 0.5 + np.array([2, 0, 0])

    # Combined (with gap in the middle)
    points = np.vstack([left_points, right_points])

    # Initialize detector
    detector = CoverageGapDetector(voxel_size=0.3, min_gap_voxels=5)

    # Detect gaps
    results = detector.analyze_coverage_gaps(points)

    print(f"\nDetection Results:")
    print(f"  Found {results['gap_count']} gaps")
    print(f"  Statistics: {results['statistics']}")

    for i, gap in enumerate(results['gap_regions']):
        world_min, world_max = gap
        print(f"  Gap {i}: center={np.mean([world_min, world_max], axis=0)}, size={world_max - world_min}")

    print("\nCoverageGapDetector test completed successfully!")
