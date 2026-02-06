# Quality-Based NFN (Next Flight Navigation)

## Overview

Quality-Based NFN analyzes 3D point cloud quality using multiple metrics to identify regions requiring additional drone coverage. This approach is more robust than gap detection as it considers the actual reconstruction quality.

## Key Advantages over Gap Detection

### Gap Detection Issues:
- ❌ Detects "empty" spaces that may not need coverage
- ❌ Sensitive to voxel grid parameters
- ❌ Doesn't consider reconstruction quality
- ❌ May suggest unnecessary flights for irrelevant gaps

### Quality-Based NFN Benefits:
- ✅ **Multi-metric quality scoring**: Confidence + Density + Normal Consistency + View Diversity
- ✅ **Focuses on existing reconstruction**: Only improves what's already mapped
- ✅ **More meaningful targets**: Suggests flights to improve poor-quality areas
- ✅ **Better results**: Targeted improvements for visible regions

## Quality Metrics

Quality-Based NFN uses 4 weighted metrics:

### 1. Confidence Score (40% weight)
- From VGGT predictions
- Higher confidence = better quality
- Most important metric

### 2. Point Density (30% weight)
- Points per voxel
- Higher density = better coverage
- Indicates sampling completeness

### 3. Surface Normal Consistency (20% weight)
- Alignment of normals within voxel
- Higher consistency = cleaner geometry
- Detects reconstruction artifacts

### 4. View Diversity (10% weight)
- Number of viewing angles
- More views = better geometry
- Currently simplified (future enhancement)

## Usage

### In demo_gradio.py:

1. **Upload images/video** and run "Reconstruct"

2. **Set parameters** in the "Quality-Based NFN" section:
   - **Voxel Size**: 0.5m (default) - larger = faster, smaller = more detail
   - **Quality Threshold**: 50% (default) - lower = stricter (more regions flagged)
   - **Viser Port**: 7778 (default)

3. **Click** "⭐ Analyze Quality & Plan Viewpoints"

4. **View results** at `http://<your-server-ip>:7778`

### Visualization:

- 🟠 **Orange wireframe boxes**: Low-quality regions (need improvement)
- 🔵 **Blue camera frustums**: Suggested viewpoints
- 🟢 **Green camera frustums**: Existing camera poses
- **Point cloud**: With confidence filtering

## Parameters

### Voxel Size
```python
voxel_size = 0.5  # meters
```
- **0.1-0.3m**: High detail, slower, indoor/small objects
- **0.5m**: Balanced (default), general scenes
- **1.0-2.0m**: Low detail, faster, large scenes

### Quality Threshold
```python
quality_threshold = 0.5  # 0.0-1.0 (50% in UI)
```
- **Low (20-40%)**: Strict - flags more regions as low-quality
- **Medium (40-60%)**: Balanced (default)
- **High (60-80%)**: Lenient - only worst regions flagged

Lower threshold → More suggested viewpoints
Higher threshold → Fewer suggested viewpoints

## API Usage

```python
from vggt_mapping.nfn import QualityAnalyzer, ViewpointPlanner

# Load VGGT predictions
world_points = predictions["world_points"]  # (S, H, W, 3)
confidence = predictions["world_points_conf"]  # (S, H, W)

# Analyze quality
analyzer = QualityAnalyzer(
    voxel_size=0.5,
    quality_threshold=0.5,
    min_region_voxels=10
)

quality_results = analyzer.analyze_quality(
    world_points=world_points,
    confidence=confidence
)

print(f"Found {quality_results['region_count']} low-quality regions")
print(f"Average quality: {quality_results['statistics']['average_quality']:.3f}")

# Plan viewpoints
planner = ViewpointPlanner(
    min_altitude=3.0,
    max_altitude=15.0,
    viewpoints_per_gap=3
)

viewpoint_results = planner.plan_viewpoints_for_quality(
    quality_results=quality_results,
    max_viewpoints=20
)

print(f"Generated {viewpoint_results['num_viewpoints']} viewpoints")

# Get suggested camera positions
for vp in viewpoint_results['viewpoints']:
    pos = vp['camera_position']
    print(f"Fly to: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
```

## Output Interpretation

### Quality Score Range:
- **0.7-1.0**: High quality (green in heatmap)
- **0.4-0.7**: Medium quality (yellow in heatmap)
- **0.0-0.4**: Low quality (red in heatmap) - **NFN targets these**

### Status Messages:

```
✅ Quality Analysis Complete!

**Low-Quality Regions**: 3
**Suggested Viewpoints**: 9
**Average Quality Score**: 0.612
**Total Low-Quality Volume**: 5.43 m³

**Quality Metrics:**
- High quality voxels: 150
- Medium quality voxels: 320
- Low quality voxels: 45
```

**Interpretation:**
- 3 regions need improvement
- 9 viewpoints suggested (3 per region)
- Overall quality: 0.612 (medium-good)
- 45 voxels are low-quality → target for next flight

## Performance

Typical processing times:

| Scene Size | Points | Voxel Size | Analysis Time | Planning Time |
|------------|--------|------------|---------------|---------------|
| Small | 100K | 0.3m | 50ms | 10ms |
| Medium | 500K | 0.5m | 120ms | 30ms |
| Large | 1M+ | 1.0m | 250ms | 50ms |

## Comparison: Gap vs Quality

### Example Scene (Building):

**Gap Detection:**
```
Detected Gaps: 5
Suggested Viewpoints: 15
Issues:
- Gap 1: Empty sky (no need to map)
- Gap 2: Behind wall (inaccessible)
- Gap 3: Far background (not important)
- Gap 4: Valid (building back)
- Gap 5: Ground shadow (low priority)

Useful viewpoints: ~40% (6/15)
```

**Quality-Based NFN:**
```
Low-Quality Regions: 2
Suggested Viewpoints: 6
Regions:
- Region 1: Building facade (low confidence, sparse)
- Region 2: Roof corner (inconsistent normals)

Useful viewpoints: ~100% (6/6)
```

**Result:** Quality-based NFN provides more targeted, efficient flight planning.

## Future Enhancements

Planned improvements (v1.2.0+):

1. **Enhanced View Diversity**
   - Calculate actual viewing angles from extrinsics
   - Weight by angle diversity

2. **Temporal Quality**
   - Detect motion blur or inconsistencies
   - Flag time-varying regions

3. **Semantic Awareness**
   - Higher priority for important objects (buildings vs trees)
   - Lower priority for dynamic/transient objects

4. **Adaptive Thresholds**
   - Auto-adjust quality threshold based on scene statistics
   - Per-region dynamic thresholds

## Troubleshooting

### Q: No low-quality regions detected?
**A**: Try:
1. Lower quality threshold (50% → 30%)
2. Reduce voxel size (0.5m → 0.3m)
3. Check if reconstruction is actually high-quality

### Q: Too many low-quality regions?
**A**: Try:
1. Raise quality threshold (50% → 70%)
2. Increase voxel size (0.5m → 1.0m)
3. Adjust min_region_voxels parameter

### Q: Viewpoints in unreachable locations?
**A**: Current version doesn't check obstacles. Manually review suggested positions. Future versions will add:
- Collision detection
- Flight envelope constraints
- No-fly zones

## See Also

- [README.md](README.md) - Complete NFN documentation
- [USAGE_CN.md](USAGE_CN.md) - Chinese user guide
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

**Recommended**: Use Quality-Based NFN instead of Gap Detection for better results! ⭐
