# Next Flight Navigation (NFN) Module

Real-time coverage gap detection and flight planning for drone mapping systems.

## Overview

The NFN module analyzes VGGT-generated 3D reconstructions to identify coverage gaps and suggest optimal camera viewpoints for additional drone flights. This helps ensure complete and high-quality scene coverage.

## Features

- **Coverage Gap Detection** ([gap_detector.py](gap_detector.py)): Voxel-based spatial analysis to identify unmapped regions
- **Viewpoint Planning** ([viewpoint_planner.py](viewpoint_planner.py)): Generates suggested camera poses to fill gaps
- **3D Visualization** ([visualization/nfn_viser_viewer.py](visualization/nfn_viser_viewer.py)): Interactive Viser-based visualization of gaps and suggested viewpoints

## Usage

### Basic Example

```python
from vggt_mapping.nfn import CoverageGapDetector, ViewpointPlanner
import numpy as np

# Assume you have VGGT predictions with world_points and confidence
world_points = predictions["world_points"]  # (S, H, W, 3)
confidence = predictions["world_points_conf"]  # (S, H, W)

# Detect gaps
gap_detector = CoverageGapDetector(voxel_size=0.5, min_gap_voxels=8)
gap_results = gap_detector.analyze_coverage_gaps(
    world_points=world_points,
    confidence=confidence,
    conf_threshold=0.6  # 60th percentile threshold
)

print(f"Found {gap_results['gap_count']} gaps")

# Plan viewpoints to cover gaps
viewpoint_planner = ViewpointPlanner(
    min_altitude=3.0,
    max_altitude=15.0,
    viewpoints_per_gap=3
)

viewpoint_results = viewpoint_planner.plan_viewpoints_for_gaps(
    gap_results=gap_results,
    max_viewpoints=20
)

print(f"Generated {viewpoint_results['num_viewpoints']} suggested viewpoints")
```

### Integration with demo_gradio.py

The NFN module is integrated into `demo_gradio.py` with the following workflow:

1. Upload images/video and run VGGT reconstruction
2. Generate confidence map (optional)
3. Click "🔍 Analyze Gaps & Plan Viewpoints"
4. View results in Viser 3D viewer at `http://<your-server-ip>:7777`
   - If running locally: `http://localhost:7777`
   - If running on remote server: Replace `<your-server-ip>` with your server's IP address

The visualization shows:
- 🔴 **Red wireframe boxes**: Coverage gaps
- 🔵 **Blue camera frustums**: Suggested viewpoints
- 🟢 **Green camera frustums**: Existing camera poses
- **Point cloud**: With confidence-based filtering

## Parameters

### CoverageGapDetector

- `voxel_size` (float): Size of voxels in meters (default: 0.5m)
  - Larger = faster but less detail
  - Smaller = more detail but slower
- `min_gap_voxels` (int): Minimum connected voxels to be considered a gap (default: 8)

### ViewpointPlanner

- `min_altitude` (float): Minimum camera altitude above gap (default: 2.0m)
- `max_altitude` (float): Maximum camera altitude above gap (default: 50.0m)
- `min_distance` (float): Minimum distance from gap center (default: 3.0m)
- `max_distance` (float): Maximum distance from gap center (default: 20.0m)
- `viewpoints_per_gap` (int): Number of viewpoints to generate per gap (default: 4)

## Output Format

### Gap Detection Results

```python
{
    "gap_regions": [  # List of (min_corner, max_corner) tuples
        (np.array([x1, y1, z1]), np.array([x2, y2, z2])),
        ...
    ],
    "gap_centers": [  # List of gap centers
        np.array([cx, cy, cz]),
        ...
    ],
    "gap_count": int,
    "statistics": {
        "total_points": int,
        "voxel_size": float,
        "occupied_voxels": int,
        "gap_volume_m3": float,
        ...
    }
}
```

### Viewpoint Planning Results

```python
{
    "viewpoints": [  # List of suggested camera poses
        {
            "gap_id": int,
            "camera_position": np.array([x, y, z]),
            "camera_rotation": np.ndarray (3x3),
            "extrinsic": np.ndarray (3x4),
            "intrinsic": np.ndarray (3x3),
            "score": float,
            ...
        },
        ...
    ],
    "num_viewpoints": int,
    "statistics": {...}
}
```

## Performance

Typical processing times (on modern GPU):
- Gap Detection: 50-200ms (depending on point count and voxel size)
- Viewpoint Planning: 10-50ms (depending on number of gaps)
- Visualization: <100ms (interactive real-time)

## Testing

### Individual Module Tests

Run individual module tests:

```bash
# Test gap detector
python gap_detector.py

# Test viewpoint planner
python viewpoint_planner.py
```

### Installation Verification

Run comprehensive verification (recommended after installation):

```bash
# Verify all fixes and configuration
python verify_nfn_installation.py
```

This checks:
- ✅ Percentile calculation fix
- ✅ Viser viewer code fixes
- ✅ Port configuration (7777)
- ✅ Documentation completeness

### Bug Fix Verification

If you encounter errors, verify specific fixes:

```bash
# Test percentile calculation
python test_nfn_percentile_fix.py

# Test port availability
python test_nfn_port.py
```

See [BUG_FIXES.md](BUG_FIXES.md) for details on known issues and fixes.

## Dependencies

- numpy
- scipy (for morphological operations)
- trimesh (for 3D visualization)
- viser (for interactive 3D viewer)

All dependencies are included in the main VGGT requirements.

## Algorithm Details

### Gap Detection

1. Build 3D voxel occupancy grid from point cloud
2. Apply confidence threshold to filter low-quality points
3. Dilate occupied regions to find frontier voxels (adjacent to occupied)
4. Connected component labeling to identify distinct gaps
5. Filter small gaps (noise) based on minimum voxel count
6. Calculate bounding boxes and statistics

### Viewpoint Planning

1. For each gap, generate candidate viewpoints in a cylindrical pattern:
   - Multiple altitude levels
   - Circular azimuth positions around gap center
2. Calculate viewing distance based on gap size and camera FOV
3. Generate camera rotation matrices (looking at gap center)
4. Score viewpoints based on:
   - Distance optimality (moderate distances preferred)
   - Altitude optimality (moderate altitudes preferred)
   - Novelty (avoid positions too close to existing cameras)
5. Select top-K viewpoints per gap

## Future Enhancements

See [TODO.md](../../TODO.md) for planned features:
- Multi-metric quality assessment (density, normal consistency, view diversity)
- Path optimization with TSP solver
- Flight plan export to drone mission formats
