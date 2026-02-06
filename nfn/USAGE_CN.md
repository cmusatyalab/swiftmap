# NFN (Next Flight Navigation) 使用说明

## 简介

NFN模块分析VGGT生成的3D重建结果，自动检测覆盖空缺并规划无人机下一次飞行的最佳拍摄位置。

## 快速开始

### 1. 在 demo_gradio.py 中使用

```bash
# 启动 Gradio 界面
python demo_gradio.py
```

#### 操作流程：

1. **上传图片或视频**
   - 点击 "Upload Video" 或 "Upload Images"
   - 上传你的无人机拍摄数据

2. **运行VGGT重建**
   - 点击 "Reconstruct" 按钮
   - 等待3D重建完成（通常<1秒）

3. **（可选）查看置信度地图**
   - 点击 "Generate Confidence Map"
   - 查看重建质量分布

4. **分析覆盖空缺**
   - 调整 "Voxel Size (m)" 滑块（默认0.5m）
     - 更大 = 速度更快，细节较少
     - 更小 = 细节更多，速度较慢
   - 检查 "Viser Port" 设置（默认7777）
   - 点击 "🔍 Analyze Gaps & Plan Viewpoints"

5. **查看3D可视化**
   - 在浏览器中打开: `http://<你的服务器IP>:7777`
   - 如果是本地运行: `http://localhost:7777`

### 2. 3D可视化说明

打开Viser界面后，你会看到：

- 🔴 **红色线框方块**: 覆盖空缺区域（需要补飞的地方）
- 🔵 **蓝色相机锥体（中等大小）**: 建议的拍摄位置（下次应该飞到哪里）
- 🟢 **绿色相机锥体（小）**: 已有的相机位置（已经拍过的地方）
- ⚪ **白色文字标签**: Gap编号（如"Gap 0", "Gap 1"）
- **点云**: 带置信度过滤的3D重建结果

**视觉尺寸层次**（从小到大）：
1. 绿色相机（最小，scale=0.05）- 与demo_viser.py保持一致
2. 蓝色建议相机（中等，scale=0.1）- 2倍绿色相机大小，便于区分
3. 红色gap线框（实际大小）- 根据真实空缺区域尺寸显示

#### 交互控制：

- **鼠标左键拖动**: 旋转视角
- **鼠标滚轮**: 缩放
- **鼠标右键拖动**: 平移
- **侧边栏滑块**: 调整置信度阈值
- **复选框**:
  - Show Gaps: 显示/隐藏空缺区域
  - Show Suggested Viewpoints: 显示/隐藏建议拍摄位置
  - Show Existing Cameras: 显示/隐藏已有相机

## 参数说明

### Voxel Size (体素大小)

控制空间分析的精度：

- **0.1-0.3m**: 高精度，适合小范围场景（室内、小物体）
- **0.5m** (默认): 平衡速度和精度，适合一般场景
- **1.0-2.0m**: 低精度，适合大范围场景（建筑、地形）

### Viser Port (可视化端口)

- 默认: **7777**
- 确保此端口未被占用
- 如果端口冲突，可以修改为其他值（如8080, 8888等）

### Confidence Threshold (置信度阈值)

- 使用主界面的 "Confidence Threshold (%)" 滑块
- 默认: 60%（过滤掉最低的60%的点）
- 更高 = 更少的点，但质量更好
- 更低 = 更多的点，但可能包含噪声

## 输出结果解读

点击 "Analyze Gaps & Plan Viewpoints" 后，会显示：

```
✅ Gap Analysis Complete!

**Detected Gaps**: 3
**Suggested Viewpoints**: 9
**Total Gap Volume**: 45.32 m³

🌐 Viser Visualization: http://0.0.0.0:7777
*Access from your browser at: `http://<your-server-ip>:7777`*
```

### 结果说明：

- **Detected Gaps**: 检测到的空缺区域数量
- **Suggested Viewpoints**: 建议的拍摄位置数量（通常每个空缺生成3个位置）
- **Total Gap Volume**: 空缺区域的总体积（立方米）

### 飞行建议：

- 如果 **Gaps = 0**: 场景覆盖完整，无需补飞 ✅
- 如果 **Gaps = 1-3**: 少量空缺，建议补飞关键位置 🟡
- 如果 **Gaps > 5**: 较多空缺，需要系统性补飞 🔴

## 实际应用示例

### 场景1: 建筑物测绘

```
输入: 建筑物周围拍摄的30张图片
Voxel Size: 0.5m
结果: 检测到2个空缺（建筑物背面）
建议: 9个补充拍摄位置（3个/空缺）
```

**下一步**: 按照蓝色相机位置规划飞行路径，重点拍摄建筑物背面

### 场景2: 地形扫描

```
输入: 地形航拍的50张图片
Voxel Size: 1.0m
结果: 检测到5个空缺（树木遮挡区域）
建议: 15个补充拍摄位置
```

**下一步**: 规划低空飞行，从不同角度拍摄被遮挡区域

### 场景3: 室内重建

```
输入: 房间内部的20张图片
Voxel Size: 0.3m
结果: 检测到3个空缺（角落和家具后面）
建议: 9个补充拍摄位置
```

**下一步**: 手持拍摄角落和死角区域

## 高级用法

### Python API调用

```python
from vggt_mapping.nfn import CoverageGapDetector, ViewpointPlanner

# 加载VGGT预测结果
predictions = np.load("predictions.npz")
world_points = predictions["world_points"]
confidence = predictions["world_points_conf"]

# 检测空缺
detector = CoverageGapDetector(voxel_size=0.5)
gaps = detector.analyze_coverage_gaps(
    world_points=world_points,
    confidence=confidence,
    conf_threshold=0.6
)

print(f"Found {gaps['gap_count']} gaps")

# 规划飞行位置
planner = ViewpointPlanner(
    min_altitude=3.0,
    max_altitude=15.0
)
viewpoints = planner.plan_viewpoints_for_gaps(gaps)

print(f"Generated {viewpoints['num_viewpoints']} viewpoints")

# 获取建议的相机位置
for vp in viewpoints['viewpoints']:
    pos = vp['camera_position']
    print(f"Fly to: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
```

### 导出飞行计划（开发中）

未来版本将支持：
- 导出为JSON格式的航点文件
- 兼容主流无人机飞控系统（DJI, PX4等）
- TSP优化的飞行路径规划

## 常见问题

### Q1: Viser服务器无法访问？

**A**: 检查以下几点：
1. 防火墙是否开放7777端口
2. 端口是否被其他程序占用（运行 `python test_nfn_port.py` 检查）
3. 服务器IP地址是否正确

### Q2: 未检测到空缺但明显有缺失？

**A**: 尝试：
1. 减小Voxel Size（如从0.5改为0.3）
2. 降低Confidence Threshold（如从60%改为40%）
3. 检查原始图片质量和重建结果

### Q3: 建议的拍摄位置不合理？

**A**: 当前版本的建议位置是基于几何覆盖生成的，可能需要人工调整。未来版本将加入：
- 飞行约束（障碍物、禁飞区）
- 路径优化
- 电池续航考虑

### Q4: 处理速度慢？

**A**: 优化建议：
1. 增大Voxel Size（牺牲精度换速度）
2. 提高Confidence Threshold（减少处理点数）
3. 使用GPU加速（自动）

### Q5: 出现 "Percentiles must be in the range [0, 100]" 错误？

**A**: 这个bug已在v1.0.1版本修复。如果仍然遇到：
1. 确保使用最新的 `nfn_viser_viewer.py`
2. 检查 `conf_threshold` 参数是百分比形式（0-100），不是小数形式（0.0-1.0）
3. 运行测试脚本验证: `python test_nfn_percentile_fix.py`

### Q6: 出现 "add_label() got an unexpected keyword argument 'color'" 错误？

**A**: 这个bug已在v1.0.1版本修复。如果仍然遇到：
1. 确保使用最新的 `nfn_viser_viewer.py`（已移除不支持的color参数）
2. 检查viser版本是否兼容（建议使用0.1.x版本）
3. 文本标签会以默认颜色显示（功能不受影响）

## 性能参考

| 场景规模 | 点云数量 | Voxel Size | 检测时间 | 规划时间 |
|---------|---------|-----------|---------|---------|
| 小型（室内） | ~100K | 0.3m | 50ms | 10ms |
| 中型（建筑） | ~500K | 0.5m | 120ms | 30ms |
| 大型（地形） | ~1M | 1.0m | 200ms | 50ms |

*测试环境: NVIDIA RTX 3090, Intel i9-12900K*

## 技术支持

如有问题，请查看：
- [README.md](README.md) - 完整技术文档
- [TODO.md](../../TODO.md) - 开发计划
- GitHub Issues - 问题反馈

---

**提示**: 第一次使用建议从小场景开始测试，熟悉工作流程后再处理大型项目。
