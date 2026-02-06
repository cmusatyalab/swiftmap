# NFN Module Changelog

All notable changes to the Next Flight Navigation (NFN) module will be documented in this file.

## [1.0.1] - 2025-10-23

### Fixed
- **Percentile calculation bug in Viser visualization**: Fixed "Percentiles must be in the range [0, 100]" error
  - Issue: `conf_threshold` (already in percentage 0-100) was incorrectly multiplied by 100
  - Location: `visualization/nfn_viser_viewer.py` line 69
  - Impact: Viser server failed to start when analyzing gaps
  - Solution: Use `conf_threshold` directly as percentile value (already in correct range)

- **add_label() color parameter error**: Fixed "SceneApi.add_label() got an unexpected keyword argument 'color'" error
  - Issue: `add_label()` doesn't support `color` parameter in current viser version
  - Location: `visualization/nfn_viser_viewer.py` line 193
  - Impact: Viser server failed to add gap labels
  - Solution: Removed unsupported `color` parameter from `add_label()` call

- **Camera frustum scale mismatch**: Fixed oversized camera frustums in visualization
  - Issue: Camera scale was 0.3 (6x larger than demo_viser.py's 0.05)
  - Location: `visualization/nfn_viser_viewer.py` lines 160, 217
  - Impact: Green existing cameras appeared disproportionately large compared to point cloud
  - Solution:
    - Existing cameras: scale = 0.05 (matches demo_viser.py)
    - Suggested viewpoints: scale = 0.1 (2x larger for easy distinction)

### Changed
- **Server configuration**: Changed default Viser port from 8080 to 7777
  - Configured to use `0.0.0.0` host for remote server access
  - Updated documentation to reflect remote access instructions

### Added
- **Test script**: `test_nfn_percentile_fix.py` to verify percentile calculations
- **Chinese documentation**: `USAGE_CN.md` with comprehensive usage guide
- **Port test utility**: `test_nfn_port.py` to check port availability

## [1.0.0] - 2025-10-23

### Added
- **Coverage Gap Detection** (`gap_detector.py`)
  - 3D voxel grid analysis to identify unmapped regions
  - Frontier region detection (boundaries between mapped/unmapped)
  - Configurable voxel size and minimum gap size
  - Processing time: ~50-200ms for typical scenes

- **Viewpoint Planning** (`viewpoint_planner.py`)
  - Generates suggested camera poses around detected gaps
  - Cylindrical viewpoint generation (multiple altitudes/azimuths)
  - Viewing distance optimization based on gap size and FOV
  - Viewpoint scoring (distance, altitude, novelty)
  - Processing time: ~10-50ms

- **Viser 3D Visualization** (`visualization/nfn_viser_viewer.py`)
  - Interactive visualization of gaps and viewpoints
  - Red wireframe boxes for coverage gaps
  - Blue camera frustums for suggested viewpoints
  - Green camera frustums for existing camera poses
  - Point cloud with confidence-based filtering
  - GUI controls for threshold adjustment and visibility toggles

- **Gradio Integration** (updates to `demo_gradio.py`)
  - "Coverage Gap Detection" section in NFN panel
  - Voxel size slider (0.1-2.0m)
  - Viser port configuration
  - "🔍 Analyze Gaps & Plan Viewpoints" button
  - Status messages with gap count and viewpoint statistics

### Documentation
- Complete README with usage examples
- Algorithm details and performance benchmarks
- Parameter descriptions and tuning guide

## API Changes

### v1.0.1
- No breaking API changes
- Internal bug fixes only

### v1.0.0
- Initial release
- Core API:
  - `CoverageGapDetector.analyze_coverage_gaps()`
  - `ViewpointPlanner.plan_viewpoints_for_gaps()`
  - `visualize_nfn_with_viser()`

## Migration Guide

### From v1.0.0 to v1.0.1
No code changes required. Simply update the files:
- `vggt_mapping/nfn/visualization/nfn_viser_viewer.py`
- `demo_gradio.py`

The percentile bug fix is backward compatible.

## Known Issues

### v1.0.1
- [ ] Path optimization not yet implemented (planned for v1.1.0)
- [ ] Flight plan JSON export not yet implemented (planned for v1.1.0)
- [ ] Multi-metric quality assessment not yet implemented (planned for v1.2.0)

### v1.0.0
- [x] Percentile calculation error in Viser visualization (fixed in v1.0.1)

## Future Roadmap

### v1.1.0 (Planned)
- Path optimization using TSP solver
- Flight plan export to JSON waypoint format
- Improved viewpoint ranking with obstacle awareness

### v1.2.0 (Planned)
- Multi-metric quality assessment module
  - Point density analysis
  - Surface normal consistency
  - Viewing angle diversity
- Quality-based viewpoint planning

### v2.0.0 (Future)
- Real-time streaming mode for live drone feeds
- Integration with drone flight controllers (DJI, PX4)
- Automatic mission planning and execution

## Contributors

- Initial implementation and bug fixes by Claude Code Assistant
- Testing and validation by VGGT development team

---

For detailed usage instructions, see [README.md](README.md) and [USAGE_CN.md](USAGE_CN.md).
