"""
Viewpoint Planning Module

Generates suggested camera viewpoints to cover detected gaps in 3D reconstructions.

Key Features:
- Viewpoint generation for coverage gaps
- Configurable drone flight constraints (altitude, distance, angles)
- Viewpoint prioritization by coverage gain
- Camera pose generation compatible with VGGT format
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class ViewpointPlanner:
    """
    Plans camera viewpoints to cover detected gaps in 3D reconstructions.
    """

    def __init__(
        self,
        min_altitude: float = 2.0,
        max_altitude: float = 50.0,
        min_distance: float = 3.0,
        max_distance: float = 20.0,
        camera_fov_deg: float = 60.0,
        viewpoints_per_gap: int = 4
    ):
        """
        Initialize viewpoint planner.

        Args:
            min_altitude: Minimum camera altitude above gap center (meters)
            max_altitude: Maximum camera altitude above gap center (meters)
            min_distance: Minimum distance from gap center (meters)
            max_distance: Maximum distance from gap center (meters)
            camera_fov_deg: Camera field of view in degrees
            viewpoints_per_gap: Number of viewpoints to generate per gap
        """
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.camera_fov_deg = camera_fov_deg
        self.viewpoints_per_gap = viewpoints_per_gap

        # Results storage
        self.suggested_viewpoints = []

    def plan_viewpoints_for_gaps(
        self,
        gap_results: Dict,
        existing_camera_positions: Optional[np.ndarray] = None,
        max_viewpoints: int = 20
    ) -> Dict:
        """
        Generate suggested viewpoints to cover detected gaps.

        Args:
            gap_results: Results from CoverageGapDetector.analyze_coverage_gaps()
            existing_camera_positions: Existing camera positions to avoid redundancy
            max_viewpoints: Maximum total number of viewpoints to generate

        Returns:
            Dictionary containing:
            - viewpoints: List of suggested camera poses
            - statistics: Planning statistics
        """
        start_time = time.time()

        gap_regions = gap_results.get("gap_regions", [])
        gap_centers = gap_results.get("gap_centers", [])

        if len(gap_regions) == 0:
            return self._empty_result("No gaps to cover")

        self.suggested_viewpoints = []

        # Generate viewpoints for each gap
        for i, (gap_bbox, gap_center) in enumerate(zip(gap_regions, gap_centers)):
            gap_min, gap_max = gap_bbox
            gap_size = np.linalg.norm(gap_max - gap_min)

            # Generate viewpoints around this gap
            gap_viewpoints = self._generate_viewpoints_for_gap(
                gap_center=gap_center,
                gap_size=gap_size,
                gap_id=i
            )

            # Score viewpoints based on coverage and feasibility
            scored_viewpoints = self._score_viewpoints(
                gap_viewpoints,
                gap_center,
                existing_camera_positions
            )

            # Keep top viewpoints for this gap
            top_viewpoints = sorted(scored_viewpoints, key=lambda x: x["score"], reverse=True)
            top_viewpoints = top_viewpoints[:self.viewpoints_per_gap]

            self.suggested_viewpoints.extend(top_viewpoints)

        # Limit total viewpoints
        self.suggested_viewpoints.sort(key=lambda x: x["score"], reverse=True)
        self.suggested_viewpoints = self.suggested_viewpoints[:max_viewpoints]

        # Calculate statistics
        stats = {
            "num_gaps": len(gap_regions),
            "total_viewpoints": len(self.suggested_viewpoints),
            "viewpoints_per_gap_avg": len(self.suggested_viewpoints) / len(gap_regions) if gap_regions else 0,
            "processing_time": time.time() - start_time
        }

        print(f"[NFN Viewpoint Planning] Generated {len(self.suggested_viewpoints)} viewpoints for {len(gap_regions)} gaps in {stats['processing_time']:.2f}s")

        results = {
            "viewpoints": self.suggested_viewpoints,
            "statistics": stats,
            "num_viewpoints": len(self.suggested_viewpoints)
        }

        return results

    def _generate_viewpoints_for_gap(
        self,
        gap_center: np.ndarray,
        gap_size: float,
        gap_id: int
    ) -> List[Dict]:
        """
        Generate candidate viewpoints around a gap region.
        """
        viewpoints = []

        # Determine viewing distance based on gap size and FOV
        # Camera distance to see entire gap: distance = gap_size / (2 * tan(FOV/2))
        fov_rad = np.deg2rad(self.camera_fov_deg)
        recommended_distance = gap_size / (2 * np.tan(fov_rad / 2))

        # Clamp to constraints
        viewing_distance = np.clip(recommended_distance, self.min_distance, self.max_distance)

        # Generate viewpoints in multiple positions around the gap
        # Strategy: Ring of cameras at different altitudes and azimuths

        # Generate at 2-3 altitude levels
        altitudes = np.linspace(self.min_altitude, self.max_altitude, 3)

        for altitude in altitudes:
            # Generate viewpoints in a circle around the gap at this altitude
            num_azimuth = self.viewpoints_per_gap

            for azimuth_idx in range(num_azimuth):
                azimuth_angle = (azimuth_idx / num_azimuth) * 2 * np.pi

                # Camera position in cylindrical coordinates
                camera_x = gap_center[0] + viewing_distance * np.cos(azimuth_angle)
                camera_y = gap_center[1] + viewing_distance * np.sin(azimuth_angle)
                camera_z = gap_center[2] + altitude

                camera_position = np.array([camera_x, camera_y, camera_z])

                # Generate camera orientation (looking at gap center)
                camera_rotation = self._look_at_rotation(camera_position, gap_center)

                # Create intrinsic matrix (simple pinhole model)
                # Assuming 512x512 image with FOV
                image_size = 512
                focal_length = (image_size / 2) / np.tan(fov_rad / 2)

                intrinsic = np.array([
                    [focal_length, 0, image_size / 2],
                    [0, focal_length, image_size / 2],
                    [0, 0, 1]
                ], dtype=np.float32)

                # Create extrinsic matrix (camera-to-world)
                # Extrinsic = [R | t] where R is rotation and t is translation
                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = camera_rotation
                extrinsic[:3, 3] = camera_position

                viewpoint = {
                    "gap_id": gap_id,
                    "camera_position": camera_position,
                    "camera_rotation": camera_rotation,
                    "extrinsic": extrinsic[:3, :],  # 3x4 format
                    "intrinsic": intrinsic,
                    "viewing_distance": viewing_distance,
                    "altitude": altitude,
                    "azimuth": azimuth_angle,
                    "gap_center": gap_center,
                    "score": 0.0  # To be filled by scoring function
                }

                viewpoints.append(viewpoint)

        return viewpoints

    def _look_at_rotation(
        self,
        camera_position: np.ndarray,
        target_position: np.ndarray,
        up_vector: np.ndarray = np.array([0, 0, 1])
    ) -> np.ndarray:
        """
        Generate camera rotation matrix to look at a target position.

        Args:
            camera_position: Camera position in world coordinates
            target_position: Target position to look at
            up_vector: World up vector (default: Z-up)

        Returns:
            3x3 rotation matrix (camera-to-world)
        """
        # Forward vector (camera looks along -Z in camera space)
        forward = target_position - camera_position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Right vector (cross product of up and forward)
        right = np.cross(up_vector, forward)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up vector to ensure orthogonality
        up = np.cross(forward, right)
        up = up / (np.linalg.norm(up) + 1e-8)

        # Camera-to-world rotation matrix
        # Columns are the camera axes in world coordinates
        rotation = np.column_stack([right, up, -forward])

        return rotation

    def _score_viewpoints(
        self,
        viewpoints: List[Dict],
        gap_center: np.ndarray,
        existing_camera_positions: Optional[np.ndarray]
    ) -> List[Dict]:
        """
        Score viewpoints based on coverage quality and feasibility.
        """
        for vp in viewpoints:
            score = 0.0

            # Factor 1: Prefer moderate viewing distances (weight: 0.3)
            distance = vp["viewing_distance"]
            optimal_distance = (self.min_distance + self.max_distance) / 2
            distance_score = 1.0 - abs(distance - optimal_distance) / (self.max_distance - self.min_distance)
            score += 0.3 * distance_score

            # Factor 2: Prefer moderate altitudes (weight: 0.3)
            altitude = vp["altitude"]
            optimal_altitude = (self.min_altitude + self.max_altitude) / 2
            altitude_score = 1.0 - abs(altitude - optimal_altitude) / (self.max_altitude - self.min_altitude)
            score += 0.3 * altitude_score

            # Factor 3: Penalize viewpoints too close to existing cameras (weight: 0.4)
            novelty_score = 1.0
            if existing_camera_positions is not None and len(existing_camera_positions) > 0:
                camera_pos = vp["camera_position"]
                min_distance_to_existing = np.min(np.linalg.norm(existing_camera_positions - camera_pos, axis=1))

                # Penalize if too close (< 1 meter)
                if min_distance_to_existing < 1.0:
                    novelty_score = min_distance_to_existing / 1.0
                else:
                    novelty_score = 1.0

            score += 0.4 * novelty_score

            vp["score"] = score

        return viewpoints

    def plan_viewpoints_for_quality(
        self,
        quality_results: Dict,
        existing_camera_positions: Optional[np.ndarray] = None,
        max_viewpoints: int = 20
    ) -> Dict:
        """
        Generate suggested viewpoints to improve low-quality regions.

        Args:
            quality_results: Results from QualityAnalyzer.analyze_quality()
            existing_camera_positions: Existing camera positions to avoid redundancy
            max_viewpoints: Maximum total number of viewpoints to generate

        Returns:
            Dictionary containing:
            - viewpoints: List of suggested camera poses
            - statistics: Planning statistics
        """
        start_time = time.time()

        low_quality_regions = quality_results.get("low_quality_regions", [])
        region_centers = quality_results.get("region_centers", [])

        if len(low_quality_regions) == 0:
            return self._empty_result("No low-quality regions to improve")

        self.suggested_viewpoints = []

        # Generate viewpoints for each low-quality region
        for i, (region_bbox, region_center) in enumerate(zip(low_quality_regions, region_centers)):
            region_min, region_max = region_bbox
            region_size = np.linalg.norm(region_max - region_min)

            # Generate viewpoints around this region (same as gap method)
            region_viewpoints = self._generate_viewpoints_for_gap(
                gap_center=region_center,
                gap_size=region_size,
                gap_id=i  # Reuse gap_id for region_id
            )

            # Score viewpoints
            scored_viewpoints = self._score_viewpoints(
                region_viewpoints,
                region_center,
                existing_camera_positions
            )

            # Keep top viewpoints for this region
            top_viewpoints = sorted(scored_viewpoints, key=lambda x: x["score"], reverse=True)
            top_viewpoints = top_viewpoints[:self.viewpoints_per_gap]

            # Update gap_center to region_center in metadata
            for vp in top_viewpoints:
                vp["region_center"] = region_center
                vp["region_id"] = i

            self.suggested_viewpoints.extend(top_viewpoints)

        # Limit total viewpoints
        self.suggested_viewpoints.sort(key=lambda x: x["score"], reverse=True)
        self.suggested_viewpoints = self.suggested_viewpoints[:max_viewpoints]

        # Calculate statistics
        stats = {
            "num_regions": len(low_quality_regions),
            "total_viewpoints": len(self.suggested_viewpoints),
            "viewpoints_per_region_avg": len(self.suggested_viewpoints) / len(low_quality_regions) if low_quality_regions else 0,
            "processing_time": time.time() - start_time
        }

        print(f"[NFN Viewpoint Planning] Generated {len(self.suggested_viewpoints)} viewpoints for {len(low_quality_regions)} low-quality regions in {stats['processing_time']:.2f}s")

        results = {
            "viewpoints": self.suggested_viewpoints,
            "statistics": stats,
            "num_viewpoints": len(self.suggested_viewpoints)
        }

        return results

    def _empty_result(self, message: str) -> Dict:
        """
        Return empty result with message.
        """
        print(f"[NFN Viewpoint Planning] {message}")
        return {
            "viewpoints": [],
            "statistics": {"error": message},
            "num_viewpoints": 0
        }

    def get_visualization_data(self) -> List[Dict]:
        """
        Get viewpoint data formatted for visualization.

        Returns:
            List of dictionaries with camera pose data
        """
        viz_data = []

        for i, vp in enumerate(self.suggested_viewpoints):
            viz_data.append({
                "viewpoint_id": i,
                "gap_id": vp["gap_id"],
                "camera_position": vp["camera_position"],
                "camera_rotation": vp["camera_rotation"],
                "extrinsic": vp["extrinsic"],
                "intrinsic": vp["intrinsic"],
                "gap_center": vp["gap_center"],
                "score": vp["score"],
                "altitude": vp["altitude"],
                "viewing_distance": vp["viewing_distance"]
            })

        return viz_data


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic gap data
    print("Testing ViewpointPlanner...")

    # Create synthetic gap results
    gap_results = {
        "gap_regions": [
            (np.array([-1, -1, -1]), np.array([1, 1, 1])),  # Gap 1
            (np.array([5, -1, -1]), np.array([7, 1, 1]))    # Gap 2
        ],
        "gap_centers": [
            np.array([0, 0, 0]),
            np.array([6, 0, 0])
        ],
        "gap_count": 2
    }

    # Existing camera positions
    existing_cameras = np.array([
        [-5, 0, 5],
        [10, 0, 5]
    ])

    # Initialize planner
    planner = ViewpointPlanner(
        min_altitude=3.0,
        max_altitude=10.0,
        viewpoints_per_gap=2
    )

    # Plan viewpoints
    results = planner.plan_viewpoints_for_gaps(gap_results, existing_cameras, max_viewpoints=10)

    print(f"\nPlanning Results:")
    print(f"  Generated {results['num_viewpoints']} viewpoints")
    print(f"  Statistics: {results['statistics']}")

    for i, vp in enumerate(results['viewpoints'][:5]):  # Show first 5
        print(f"  Viewpoint {i}: gap_id={vp['gap_id']}, pos={vp['camera_position']}, score={vp['score']:.3f}")

    print("\nViewpointPlanner test completed successfully!")
