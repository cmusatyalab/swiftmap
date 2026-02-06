# CLAUDE.md - VGGT Mapping System

This file provides guidance to Claude Code (claude.ai/code) when working with the VGGT real-time drone mapping system.

## Some high level working rules

1. **Never Over-Engineering**: Think carefully and only action the specific task I have given you with the most concise and elegant solution that changes as little code as possible.
2. **Maintain Clean Working Directory**: Always evaluate whether creating a new file is essential before generation. Avoid cluttering the main working directory with temporary or test files. If must maintain a test file, place all functional test files in the system temporary directory to keep the project structure organized and maintainable.
3. **For new generated code files, NOT MENTION COPYRIGHT at the begining!**
4. Whenever you want to run commands, please first activate the `vggt` conda virtual python environment
5. **Keep Codes Clean and Compact**: NEVER generating too much useless codes to increase the line numbers. You should focuse on realizing the functionality with compact and clean solutions

## Rules for Working on VGGT Mapping

1. **Preserve Directory Organization**: All mapping results must be saved in organized `input_stream_YYYYMMDD_HHMMSS/` directories. Never create persistent `temp/` directories in the project structure.

2. **Maintain Performance**: The optical flow keyframe selection and VGGT inference pipeline must maintain real-time capabilities. Target processing times: keyframe selection <100ms, VGGT inference <1s.

3. **Handle Multiple Output Formats**: The system generates GLB, PLY, NPZ, and JSON files. Always ensure proper file organization and consistent naming conventions.

4. **Thread Safety**: Be mindful of multi-threading in TCP server operations, keyframe collection, and GUI callbacks. Use proper synchronization when needed.

5. **GPU Memory Management**: Monitor GPU memory usage especially during VGGT inference. The system should gracefully handle batch processing of keyframes without memory overflow.

## System Architecture Overview

### Core Components

#### 1. VGGT Mapper (`core/vggt_mapper.py`)
- **VGGTMapper**: Core VGGT inference engine with complete 3D reconstruction pipeline
- **Key Features**:
  - Model loading and initialization with device management
  - Batch keyframe processing with tensor caching
  - 3D content generation (GLB, PLY, confidence maps)
  - Coordinate system handling and pose estimation
  - Memory-efficient processing with automatic cleanup

**Performance Optimizations:**
- bfloat16 support on Ampere GPUs for faster inference
- GPU memory management with automatic cleanup
- Efficient tensor operations and batch processing

#### 2. TCP Server (`core/tcp_server.py`) 
- **MappingTCPServer**: Real-time image streaming server for drone clients
- **Protocol**: Binary image transmission compatible with vggt_localization test client
- **Key Features**:
  - Multi-threaded client handling
  - Automatic keyframe collection and storage
  - Real-time statistics and monitoring
  - Graceful connection management and error recovery

**Storage Strategy:**
- Uses system temporary directories for live keyframe collection
- Automatic cleanup when server stops
- Keyframes copied to organized directories during processing

#### 3. Keyframe Selection (`utils/frame_tracker.py`)
- **FrameTracker**: Lucas-Kanade optical flow implementation
- **VGGT-SLAM Integration**: Uses algorithms from VGGT-SLAM for consistent keyframe selection
- **Key Parameters**:
  - `min_disparity`: Minimum pixel displacement threshold (default: 40 pixels)
  - Feature detection using goodFeaturesToTrack
  - Optical flow tracking with calcOpticalFlowPyrLK

**Algorithm Details:**
```python
# Keyframe selection logic
def compute_disparity(self, image, min_disparity):
    # Detect features in reference keyframe
    kf_points = cv2.goodFeaturesToTrack(self.reference_frame, ...)
    
    # Track features using Lucas-Kanade optical flow
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        self.reference_frame, image, kf_points, None, ...)
    
    # Compute mean displacement
    displacement = np.linalg.norm(good_next - good_kf, axis=1)
    mean_disparity = np.mean(displacement)
    
    return mean_disparity > min_disparity
```

#### 4. Web Interface (`core/gradio_interface.py`)
- **MappingGradioInterface**: Comprehensive web UI with dual 3D viewers
- **Interactive Controls**: Real-time parameter adjustment and server management
- **Live Monitoring**: Statistics display, processing logs, and status updates

**Dual Viewer System:**
- **Main Reconstruction Viewer**: GLB scenes from VGGT inference
- **Confidence Map Viewer**: NFN (Next Flight Navigation) confidence visualization
- **Synchronized Controls**: Unified threshold controls affect both viewers

### Operation Modes

#### 1. GUI Mode (Default)
```bash
python launch_mapping.py
```
- Web-based interface with dual 3D viewers
- Interactive TCP server controls
- Real-time statistics and monitoring
- Parameter adjustment and export capabilities

#### 2. TCP Server Mode
```bash
python launch_mapping.py --tcp --tcp-port 43322
```
- Headless keyframe collection from drone clients
- Compatible with existing test client protocol
- Automatic keyframe storage and statistics
- Multi-client support with error recovery

#### 3. Console Mode
```bash
python launch_mapping.py --console --keyframes-dir keyframes/
```
- Batch processing of existing keyframe directories
- Automated VGGT inference and 3D content generation
- Complete pipeline execution with final statistics

### File Organization System

All mapping results are saved in organized directories with consistent structure:

```
input_stream_20250905_143456/
├── images/                    # Processed keyframes
│   ├── frame_000001.jpg      # Keyframes copied from collection
│   ├── frame_000002.jpg
│   └── frame_NNNNNN.jpg
├── scene.glb                 # Main 3D reconstruction (for 3D viewers)
├── pointcloud_60_all_*.ply   # Main point cloud with parameters
├── confidence_map.glb        # NFN confidence mapping GLB
├── confidence_map.ply        # NFN confidence mapping PLY  
├── camera_poses.json         # Camera poses with intrinsics/extrinsics
└── predictions.npz           # Raw VGGT predictions (numpy format)
```

**Directory Naming Convention:**
- Format: `input_stream_YYYYMMDD_HHMMSS`
- Created automatically during processing
- Contains complete mapping session results
- Compatible with demo_gradio.py directory structure

### Protocol Compatibility

#### TCP Communication
- **Port**: 43322 (default for mapping, vs 43332 for localization)
- **Protocol**: Binary image transmission
- **Response**: Keyframe selection status (compatible with test client)

```python
# Client sends image
size_data = struct.pack('!I', len(image_bytes))
socket.sendall(size_data + image_bytes)

# Server responds with selection status
status_code = 1.0 if is_keyframe else 0.0
response = struct.pack('3d', status_code, keyframe_count, total_frames)
socket.sendall(response)
```

#### Test Client Usage
```bash
# Compatible with both mapping and localization systems
python utils/test_client.py --host localhost --port 43322 --image-dir test_images/
```

### Performance Characteristics

#### Keyframe Selection
- **Feature Detection**: ~10-30ms per frame
- **Optical Flow Tracking**: ~20-50ms per frame  
- **Disparity Computation**: ~5-15ms per frame
- **Total Keyframe Processing**: ~40-100ms per frame

#### VGGT Inference Pipeline
- **Model Loading**: ~5-15s (one-time initialization)
- **Image Preprocessing**: ~20-100ms per batch
- **VGGT Forward Pass**: ~500ms-2s (depends on batch size and GPU)
- **3D Content Generation**: ~200ms-1s (GLB/PLY export)
- **Total Processing**: ~1-4s per keyframe batch

#### Memory Usage
- **VGGT Model**: ~2-8GB GPU memory (depends on model size)
- **Keyframe Batch**: ~100-500MB GPU memory
- **3D Content**: ~10-100MB disk space per session
- **Temporary Storage**: Automatic cleanup, no persistent overhead

### Integration Guidelines

#### Adding New Keyframe Selection Algorithms
1. Extend `FrameTracker` class in `utils/frame_tracker.py`
2. Implement `compute_disparity()` method with consistent interface
3. Maintain compatibility with existing min_disparity parameter
4. Ensure thread safety for TCP server integration

#### Modifying 3D Output Formats
1. Update `_generate_3d_content()` in `core/vggt_mapper.py`
2. Maintain directory structure consistency
3. Ensure proper file naming conventions
4. Test with both GUI and console modes

#### Extending Web Interface
1. Use Gradio components with proper event handling
2. Implement proper callbacks with error recovery
3. Update component references in `_components` dictionary
4. Test with different browser configurations

### Development Commands

#### Testing TCP Server
```bash
# Start mapping server
python launch_mapping.py --tcp --tcp-port 43322

# Test with client (separate terminal)
python utils/test_client.py --host localhost --port 43322 --image-dir test_images/
```

#### Debugging VGGT Inference
```bash
# Console mode with detailed output
python launch_mapping.py --console --keyframes-dir keyframes/ --verbose

# Direct testing (in Python)
from vggt_mapping.core.vggt_mapper import VGGTMapper
mapper = VGGTMapper()
results = mapper.process_keyframes_from_directory('keyframes/')
```

#### GUI Development
```bash
# Start with custom host/port
python launch_mapping.py --gui --host 0.0.0.0 --gui-port 7860

# No automatic browser opening
python launch_mapping.py --no-browser
```

### Error Handling Patterns

#### Graceful Degradation
- TCP connection lost → Continue with collected keyframes
- VGGT inference failure → Return detailed error messages
- GPU memory exhaustion → Automatic batch size reduction
- File system errors → Fallback to alternative storage locations

#### Client Communication
- **Success**: Keyframe selection status in binary format
- **Error**: Proper status codes for initialization and processing failures
- **Recovery**: Automatic reconnection support for dropped connections

### Configuration Options

#### Default Parameters
```python
default_params = {
    "mask_sky": True,           # Sky filtering enabled
    "mask_dynamic": False,      # Dynamic object filtering disabled  
    "conf_threshold": 60.0,     # 60% confidence threshold
    "mask_black_bg": False,     # Black background masking
    "mask_white_bg": False,     # White background masking
    "show_cam": True           # Show camera poses in 3D viewer
}
```

#### Optical Flow Settings
```python
optical_flow_params = dict(
    winSize=(15, 15),          # Window size for tracking
    maxLevel=2,                # Pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
```

#### TCP Server Settings
```python
tcp_settings = {
    "host": "localhost",        # Server host address
    "port": 43322,             # Default mapping port
    "timeout": 30.0,           # Client timeout in seconds
    "max_clients": 10          # Maximum concurrent clients
}
```

## File-by-File Guide

### `launch_mapping.py`
**Purpose**: Main entry point with mode selection and argument parsing
**Key Functions**:
- `run_gui_mode()`: Start Gradio web interface
- `run_tcp_mode()`: Start headless TCP server
- `run_console_mode()`: Batch processing mode
**Usage**: Primary interface for all mapping operations

### `core/vggt_mapper.py`  
**Purpose**: VGGT model inference and 3D content generation
**Key Classes**:
- `VGGTMapper`: Complete inference pipeline
**Key Methods**:
- `process_keyframes_from_directory()`: Main processing entry point
- `_generate_3d_content()`: GLB/PLY generation with directory structure
- `_generate_confidence_mapping()`: NFN confidence visualization
**Critical**: Handles GPU memory management and file organization

### `core/tcp_server.py`
**Purpose**: Real-time TCP server for drone image streaming  
**Key Classes**:
- `MappingTCPServer`: Multi-threaded image reception
**Key Methods**:
- `start_server()`: Server initialization and client handling
- `save_keyframe()`: Image storage with metadata
- `get_collected_keyframes()`: Batch keyframe retrieval
**Critical**: Thread safety and proper resource cleanup

### `core/gradio_interface.py`
**Purpose**: Web-based user interface with dual 3D viewers
**Key Classes**:
- `MappingGradioInterface`: Complete GUI implementation
**Key Methods**:
- `create_interface()`: Gradio component layout
- `_process_keyframes()`: Main reconstruction trigger
- `_generate_confidence_map()`: NFN map generation
**Critical**: Component synchronization and error handling

### `utils/frame_tracker.py`  
**Purpose**: Optical flow keyframe selection using VGGT-SLAM algorithms
**Key Classes**:
- `FrameTracker`: Lucas-Kanade optical flow implementation
**Key Methods**:
- `compute_disparity()`: Core keyframe selection logic
- `update_reference()`: Reference frame management
**Critical**: Performance optimization and algorithm accuracy

### `utils/test_client.py`
**Purpose**: Test client for TCP server communication
**Key Classes**:
- `MappingTestClient`: Compatible with localization test client
**Key Methods**:
- `send_image_directory()`: Batch image transmission
- `send_image()`: Individual image sending with status response
**Usage**: Testing and debugging TCP communication

### Generated Output Structure

Each mapping session creates a complete, self-contained directory:

#### `input_stream_YYYYMMDD_HHMMSS/`
- **Purpose**: Complete mapping session results
- **Structure**: Compatible with demo_gradio.py expectations
- **Files**: All reconstruction outputs in one location

#### `images/` Subdirectory
- **Purpose**: Processed keyframes with consistent naming
- **Format**: `frame_NNNNNN.jpg` with zero-padded indices
- **Source**: Copied from TCP server temporary storage

#### 3D Content Files
- **`scene.glb`**: Primary 3D reconstruction for web viewers
- **`pointcloud_*.ply`**: Detailed point cloud with parameter encoding
- **`confidence_map.glb/.ply`**: NFN quality visualization

#### Metadata Files  
- **`predictions.npz`**: Raw VGGT output for post-processing
- **`camera_poses.json`**: Complete camera calibration data

This architecture ensures that VGGT Mapping integrates seamlessly with the broader VGGT ecosystem while providing specialized drone mapping capabilities with real-time performance and comprehensive output generation.