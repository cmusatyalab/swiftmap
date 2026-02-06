# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Mapping Inference Engine

Performs 3D reconstruction using VGGT model on collected keyframes.
Adapted from demo_gradio.py workflow but optimized for batch processing
of keyframes collected from drone flights.

Key Features:
- VGGT model loading and inference
- Batch processing of keyframe collections
- Sky filtering and confidence thresholding
- Point cloud and GLB generation
- Confidence mapping for NFN visualization
"""

import os
import sys
import torch
import numpy as np
import cv2
import time
import gc
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import trimesh

# Add vggt to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_root = os.path.dirname(os.path.dirname(current_dir))
if vggt_root not in sys.path:
    sys.path.append(vggt_root)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb, export_point_cloud_to_ply

# Try to import confidence mapping utilities
try:
    from vggt_localization.utils.confidence_mapping import (
        generate_confidence_point_cloud,
        analyze_confidence_distribution
    )
    CONFIDENCE_MAPPING_AVAILABLE = True
except ImportError:
    CONFIDENCE_MAPPING_AVAILABLE = False
    print("Warning: Confidence mapping utilities not available")


class VGGTMapper:
    """
    VGGT mapping inference engine for processing collected keyframes into 3D reconstructions.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize VGGT mapper.
        
        Args:
            device: Compute device ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_initialized = False
        
        # Default processing parameters (matching TODO requirements)
        self.default_params = {
            "mask_sky": True,           # Sky filtering enabled by default
            "mask_dynamic": False,      # Dynamic object filtering disabled
            "conf_threshold": 60.0,     # 60% confidence threshold
            "mask_black_bg": False,
            "mask_white_bg": False,
            "show_cam": True
        }
        
        # Results storage
        self.latest_predictions = None
        self.latest_scene = None
        self.latest_confidence_scene = None
        self.latest_scene_results = {}
        self.latest_confidence_results = {}
        self.latest_keyframes_info = None
        
        print(f"VGGT Mapper initialized on device: {self.device}")
    
    def initialize_model(self) -> bool:
        """
        Initialize and load the VGGT model.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            print("Initializing VGGT model...")
            
            # Load VGGT model
            self.model = VGGT()
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
            self.model.eval()
            self.model = self.model.to(self.device)
            
            self.is_initialized = True
            print("VGGT model initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing VGGT model: {e}")
            self.is_initialized = False
            return False
    
    def process_keyframes(self, 
                         keyframe_paths: List[str],
                         processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a collection of keyframes through VGGT inference.
        
        Args:
            keyframe_paths: List of paths to keyframe image files
            processing_params: Optional processing parameters
            
        Returns:
            Dictionary containing processing results and metadata
        """
        if not self.is_initialized:
            if not self.initialize_model():
                return {"success": False, "error": "Model initialization failed"}
        
        if not keyframe_paths:
            return {"success": False, "error": "No keyframes provided"}
        
        # Use default parameters if not provided
        params = self.default_params.copy()
        if processing_params:
            params.update(processing_params)
        
        try:
            print(f"Processing {len(keyframe_paths)} keyframes with VGGT...")
            print(f"Parameters: {params}")
            
            processing_start = time.time()
            
            # Load and preprocess keyframe images
            preprocess_start = time.time()
            images = load_and_preprocess_images(keyframe_paths).to(self.device)
            preprocess_time = time.time() - preprocess_start
            
            print(f"Loaded {len(keyframe_paths)} keyframes: {images.shape}")
            
            # Run VGGT inference
            inference_start = time.time()
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = self.model(images)
            
            inference_time = time.time() - inference_start
            
            # Post-process predictions
            postprocess_start = time.time()
            processed_predictions = self._postprocess_predictions(predictions, images, keyframe_paths)
            postprocess_time = time.time() - postprocess_start
            
            # Apply filtering and generate 3D content
            generation_start = time.time()
            scene_results = self._generate_3d_content(processed_predictions, params)
            generation_time = time.time() - generation_start
            
            # Generate confidence mapping if available
            confidence_results = {}
            if CONFIDENCE_MAPPING_AVAILABLE and scene_results.get("target_directory"):
                try:
                    conf_start = time.time()
                    confidence_results = self._generate_confidence_mapping(
                        processed_predictions, 
                        params,
                        scene_results["target_directory"]  # Pass target_dir to save in input_stream_xxx
                    )
                    conf_time = time.time() - conf_start
                    confidence_results["generation_time"] = conf_time
                except Exception as e:
                    print(f"Warning: Confidence mapping generation failed: {e}")
            
            total_processing_time = time.time() - processing_start
            
            # Store results for later access
            self.latest_predictions = processed_predictions
            self.latest_scene = scene_results.get("scene")
            self.latest_confidence_scene = confidence_results.get("scene")
            self.latest_scene_results = scene_results  # Store complete scene results for target_directory access
            self.latest_confidence_results = confidence_results  # Store complete confidence results
            self.latest_keyframes_info = {
                "keyframe_paths": keyframe_paths,
                "processing_params": params,
                "processing_time": total_processing_time
            }
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Prepare final results
            results = {
                "success": True,
                "keyframe_count": len(keyframe_paths),
                "processing_params": params,
                "predictions": processed_predictions,
                "scene_results": scene_results,
                "confidence_results": confidence_results,
                "timing": {
                    "total_processing": total_processing_time,
                    "preprocessing": preprocess_time,
                    "inference": inference_time,
                    "postprocessing": postprocess_time,
                    "3d_generation": generation_time
                }
            }
            
            print(f"VGGT processing completed in {total_processing_time:.2f}s")
            print(f"  - Preprocessing: {preprocess_time:.2f}s")
            print(f"  - Inference: {inference_time:.2f}s") 
            print(f"  - Postprocessing: {postprocess_time:.2f}s")
            print(f"  - 3D Generation: {generation_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"Error during VGGT processing: {e}")
            return {"success": False, "error": str(e)}
    
    def _postprocess_predictions(self, predictions: Dict[str, torch.Tensor], 
                                images: torch.Tensor, keyframe_paths: List[str]) -> Dict[str, Any]:
        """
        Post-process VGGT model predictions.
        
        Args:
            predictions: Raw model predictions
            images: Input images tensor
            keyframe_paths: List of keyframe file paths
            
        Returns:
            Dictionary with post-processed predictions
        """
        try:
            # Convert pose encoding to extrinsic and intrinsic matrices
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], images.shape[-2:]
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # Convert tensors to numpy
            processed_predictions = {}
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    processed_predictions[key] = predictions[key].cpu().numpy().squeeze(0)
                else:
                    processed_predictions[key] = predictions[key]
            
            # Extract camera positions and rotations (world coordinates)
            camera_positions = []
            camera_rotations = []
            for ext, _ in zip(processed_predictions["extrinsic"], processed_predictions["intrinsic"]):
                R = ext[:3, :3]
                t = ext[:3, 3]
                # Camera position in world coordinates: world_pos = -R.T @ t
                world_pos = -R.T @ t
                # Camera rotation in world coordinates: world_rot = R.T
                world_rot = R.T
                camera_positions.append(world_pos)
                camera_rotations.append(world_rot)
            
            processed_predictions["camera_positions"] = np.array(camera_positions)
            processed_predictions["camera_rotations"] = np.array(camera_rotations)
            
            # Add metadata
            processed_predictions["metadata"] = {
                "keyframe_paths": keyframe_paths,
                "num_keyframes": len(keyframe_paths),
                "processing_timestamp": datetime.now(),
                "input_image_shape": images.shape
            }
            
            return processed_predictions
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
            raise
    
    def _ensure_cpu_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert all CUDA tensors to CPU/numpy arrays.
        
        Args:
            data: Dictionary that may contain CUDA tensors
            
        Returns:
            Dictionary with all tensors converted to CPU/numpy
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Convert CUDA tensor to CPU numpy array
                result[key] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = self._ensure_cpu_tensors(value)
            elif isinstance(value, list):
                # Process lists that might contain tensors
                result[key] = [
                    item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
            else:
                # Keep non-tensor values as-is
                result[key] = value
        return result
    
    def _generate_3d_content(self, predictions: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate 3D content (GLB files, point clouds) from predictions.
        Creates directory structure similar to demo_gradio.py: input_stream_{timestamp}/
        
        Args:
            predictions: Post-processed predictions
            params: Processing parameters
            
        Returns:
            Dictionary with 3D content generation results
        """
        try:
            print("Generating 3D content...")
            
            # Create target directory similar to demo_gradio.py
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target_dir = f"input_stream_{timestamp}"
            
            # Create directory structure: target_dir/images/
            target_dir_images = os.path.join(target_dir, "images")
            os.makedirs(target_dir_images, exist_ok=True)
            
            print(f"Created target directory: {target_dir}")
            
            # Copy keyframe images to the images/ subdirectory
            keyframe_paths = predictions["metadata"]["keyframe_paths"]
            image_paths_in_target = []
            
            for i, keyframe_path in enumerate(keyframe_paths):
                # Copy each keyframe to the images directory
                import shutil
                dst_filename = f"frame_{i:06d}.jpg"
                dst_path = os.path.join(target_dir_images, dst_filename)
                shutil.copy2(keyframe_path, dst_path)
                image_paths_in_target.append(dst_path)
            
            print(f"Copied {len(keyframe_paths)} keyframes to {target_dir_images}")
            
            # Use predictions_to_glb from visual_util (same as demo_gradio.py)
            scene = predictions_to_glb(
                predictions=predictions,
                conf_thres=params["conf_threshold"],
                filter_by_frames="all",
                mask_black_bg=params["mask_black_bg"],
                mask_white_bg=params["mask_white_bg"],
                show_cam=params["show_cam"],
                mask_sky=params["mask_sky"],
                mask_dynamic=params["mask_dynamic"],
                target_dir=target_dir
            )
            
            # Save GLB file manually (similar to demo_gradio.py)
            glb_path = os.path.join(target_dir, "scene.glb")
            scene.export(glb_path)
            print(f"GLB scene generated: {glb_path}")
            
            # Save predictions.npz (like demo_gradio.py)
            # Ensure all tensors are converted to CPU/numpy before saving
            predictions_for_save = self._ensure_cpu_tensors(predictions)
            prediction_save_path = os.path.join(target_dir, "predictions.npz")
            np.savez(prediction_save_path, **predictions_for_save)
            print(f"Predictions saved: {prediction_save_path}")
            
            # Generate point cloud PLY file (like demo_gradio.py)
            ply_filename = os.path.join(
                target_dir,
                f"pointcloud_{params['conf_threshold']}_all_maskb{params['mask_black_bg']}_maskw{params['mask_white_bg']}_sky{params['mask_sky']}_dyn{params['mask_dynamic']}.ply"
            )
            
            try:
                # Use the same function as demo_gradio.py
                vertices_3d, colors_rgb = export_point_cloud_to_ply(
                    predictions, 
                    ply_filename,  # Add missing required parameter
                    conf_thres=params["conf_threshold"],
                    filter_by_frames="all",
                    mask_black_bg=params["mask_black_bg"],
                    mask_white_bg=params["mask_white_bg"],
                    mask_sky=params["mask_sky"],
                    mask_dynamic=params["mask_dynamic"],
                    target_dir=target_dir,
                    prediction_mode="Pointmap Regression"
                )
                
                # Save PLY file
                save_point_cloud_ply(vertices_3d, colors_rgb, ply_filename)
                print(f"Point cloud saved: {ply_filename}")
                
            except Exception as e:
                print(f"Warning: PLY export failed: {e}")
                ply_filename = None
            
            results = {
                "glb_path": glb_path if glb_path and os.path.exists(glb_path) else None,
                "ply_path": ply_filename if ply_filename and os.path.exists(ply_filename) else None,
                "scene": scene,
                "target_directory": target_dir,
                "images_directory": target_dir_images,
                "copied_images": image_paths_in_target
            }
            
            return results
            
        except Exception as e:
            print(f"Error generating 3D content: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _generate_confidence_mapping(self, predictions: Dict[str, Any], 
                                   params: Dict[str, Any], 
                                   target_dir: str) -> Dict[str, Any]:
        """
        Generate confidence mapping for NFN visualization.
        
        Args:
            predictions: Post-processed predictions
            params: Processing parameters
            target_dir: Target directory for saving files (input_stream_xxx)
            
        Returns:
            Dictionary with confidence mapping results
        """
        if not CONFIDENCE_MAPPING_AVAILABLE:
            return {"error": "Confidence mapping utilities not available"}
        
        try:
            print("Generating confidence mapping...")
            
            # Check for required data
            if "world_points" not in predictions or "world_points_conf" not in predictions:
                return {"error": "World points or confidence data not available in predictions"}
            
            world_points = predictions["world_points"]
            world_points_conf = predictions["world_points_conf"]
            
            # Generate confidence point cloud with consistent parameters as demo_gradio.py
            conf_threshold_normalized = params["conf_threshold"] / 100.0
            scene, stats, ply_path = generate_confidence_point_cloud(
                world_points=world_points,
                confidence=world_points_conf,
                conf_threshold=conf_threshold_normalized,
                colormap="red_to_green",  # Use same colormap as demo_gradio.py default
                save_ply=True  # Enable PLY file generation
            )
            
            # Move PLY file to target directory (input_stream_xxx) if generated
            target_ply_path = None
            if ply_path and os.path.exists(ply_path):
                import shutil
                target_ply_path = os.path.join(target_dir, "confidence_map.ply")
                shutil.move(ply_path, target_ply_path)
                print(f"Confidence map PLY moved to: {target_ply_path}")
            
            # Save confidence mapping GLB in the same target directory
            conf_glb_path = os.path.join(target_dir, "confidence_map.glb")
            
            if scene:
                scene.export(conf_glb_path)
                print(f"Confidence map GLB saved: {conf_glb_path}")
            
            # Analyze confidence distribution
            distribution_stats = analyze_confidence_distribution(world_points_conf)
            
            results = {
                "scene": scene,
                "glb_path": conf_glb_path if os.path.exists(conf_glb_path) else None,
                "ply_path": target_ply_path if target_ply_path and os.path.exists(target_ply_path) else None,
                "statistics": stats,
                "distribution": distribution_stats,
                "threshold_used": conf_threshold_normalized
            }
            
            print(f"Confidence mapping generated: {stats.get('high_conf_points', 0)}/{stats.get('total_points', 0)} high-confidence points")
            
            return results
            
        except Exception as e:
            print(f"Error generating confidence mapping: {e}")
            return {"error": str(e)}
    
    def get_latest_results(self) -> Dict[str, Any]:
        """
        Get the latest processing results.
        
        Returns:
            Dictionary with latest results, or None if no processing done
        """
        if self.latest_predictions is None:
            return {"error": "No processing results available"}
        
        return {
            "predictions": self.latest_predictions,
            "scene": self.latest_scene,
            "confidence_scene": self.latest_confidence_scene,
            "scene_results": getattr(self, 'latest_scene_results', {}),  # Include complete scene results
            "confidence_results": getattr(self, 'latest_confidence_results', {}),  # Include complete confidence results
            "keyframes_info": self.latest_keyframes_info
        }
    
    def save_camera_poses_json(self, output_path: str) -> bool:
        """
        Save camera poses to JSON file.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            bool: True if successful
        """
        if self.latest_predictions is None:
            print("No predictions available to save")
            return False
        
        try:
            # Convert predictions to JSON format (adapted from demo_gradio.py)
            predictions = self.latest_predictions
            keyframe_paths = self.latest_keyframes_info["keyframe_paths"]
            
            extrinsic = predictions["extrinsic"]
            intrinsic = predictions["intrinsic"]
            
            poses_data = {
                "metadata": {
                    "description": "Camera poses from VGGT Mapping",
                    "timestamp": datetime.now().isoformat(),
                    "num_keyframes": len(keyframe_paths),
                    "processing_params": self.latest_keyframes_info["processing_params"]
                },
                "frames": []
            }
            
            for i, (ext, intr) in enumerate(zip(extrinsic, intrinsic)):
                image_name = os.path.basename(keyframe_paths[i]) if i < len(keyframe_paths) else f"Keyframe_{i}"
                
                # Extract rotation and translation
                R = ext[:3, :3]
                t = ext[:3, 3]
                world_pos = -R.T @ t
                
                frame_data = {
                    "image_name": image_name,
                    "camera_position_world": world_pos.tolist(),
                    "rotation_matrix": R.tolist(),
                    "translation_vector": t.tolist(),
                    "intrinsic_matrix": intr.tolist(),
                    "extrinsic_matrix": ext.tolist()
                }
                
                poses_data["frames"].append(frame_data)
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(poses_data, f, indent=2)
            
            print(f"Camera poses saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving camera poses: {e}")
            return False
    
    def clear_results(self):
        """Clear stored results and free memory."""
        self.latest_predictions = None
        self.latest_scene = None
        self.latest_confidence_scene = None
        self.latest_keyframes_info = None
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("Results cleared")


# Example usage and testing
if __name__ == "__main__":
    import glob
    
    # Test VGGT mapper with example images
    mapper = VGGTMapper()
    
    # Look for test keyframes
    test_keyframes = glob.glob("examples/*/images/*.jpg")[:5]  # Take first 5 as test
    
    if not test_keyframes:
        print("No test images found. Create some test keyframes first.")
        print("Test completed - VGGTMapper class is ready for use")
    else:
        print(f"Testing VGGT mapper with {len(test_keyframes)} test keyframes...")
        
        # Process keyframes
        results = mapper.process_keyframes(test_keyframes)
        
        if results["success"]:
            print("Processing successful!")
            print(f"GLB saved: {results['scene_results'].get('glb_path')}")
            
            # Save camera poses
            poses_path = "test_camera_poses.json"  # Save in current directory for testing
            mapper.save_camera_poses_json(poses_path)
            
            # Print statistics
            timing = results["timing"]
            print(f"Total processing time: {timing['total_processing']:.2f}s")
            
        else:
            print(f"Processing failed: {results.get('error')}")
    
    print("VGGTMapper test completed")