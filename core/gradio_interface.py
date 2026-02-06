# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gradio Interface for VGGT Mapping System

Web-based interface for the VGGT mapping system featuring:
- TCP server controls for keyframe collection
- Dual 3D viewers (reconstruction + confidence mapping)
- Real-time statistics and monitoring
- Processing controls and parameter adjustment

Layout matches TODO requirements:
- No upload interface (processes TCP-collected images)
- Large dual 3D viewers side by side
- All controls positioned below viewers
"""

import os
import sys
import gradio as gr
import numpy as np
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json

# Add vggt root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_root = os.path.dirname(os.path.dirname(current_dir))
if vggt_root not in sys.path:
    sys.path.append(vggt_root)

from vggt_mapping.core.keyframe_selector import KeyframeSelector
from vggt_mapping.core.vggt_mapper import VGGTMapper


class MappingGradioInterface:
    """
    Gradio web interface for VGGT mapping system.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 7860):
        """
        Initialize the Gradio interface.
        
        Args:
            host: Interface host address
            port: Interface port
        """
        self.host = host
        self.port = port
        
        # Core system components
        self.keyframe_selector = None
        self.vggt_mapper = VGGTMapper()
        
        # Interface state
        self.server_running = False
        self.processing_active = False
        self.last_update_time = None
        
        # Statistics tracking
        self.stats_history = []
        
        # Default parameters matching TODO requirements
        self.default_params = {
            "tcp_port": 43322,
            "min_disparity": 40.0,
            "conf_threshold": 60.0,
            "visualize_flow": False,
            "mask_sky": True,
            "mask_dynamic": False
        }
        
        print("VGGT Mapping Gradio Interface initialized")
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        # Set theme
        try:
            theme = gr.themes.Ocean()
        except:
            theme = gr.themes.Default()
        
        with gr.Blocks(theme=theme, title="VGGT Mapping System") as interface:
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🚁 VGGT Mapping System</h1>
                <p>Real-time drone mapping with keyframe selection and 3D reconstruction</p>
            </div>
            """)
            
            # Dual 3D viewers (large, side by side as per TODO)
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3 style='text-align: center;'>3D Reconstruction</h3>")
                    model_viewer = gr.Model3D(
                        label="3D Model",
                        height=500,
                        camera_position=(2, 2, 2)
                    )
                
                with gr.Column():
                    gr.HTML("<h3 style='text-align: center;'>Confidence Map (NFN)</h3>")
                    confidence_viewer = gr.Model3D(
                        label="Confidence Map", 
                        height=500,
                        camera_position=(2, 2, 2)
                    )
            
            # Control panels below viewers as per TODO requirements
            with gr.Tabs():
                # TCP Server Control Tab
                with gr.TabItem("🌐 TerraSLAM Mapping Engine Control"):
                    with gr.Row():
                        with gr.Column():
                            tcp_port_input = gr.Number(
                                value=self.default_params["tcp_port"],
                                label="TCP Port",
                                precision=0,
                                minimum=1024,
                                maximum=65535
                            )
                            
                            min_disparity_input = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=self.default_params["min_disparity"],
                                step=1,
                                label="Min Disparity Threshold (pixels)"
                            )
                            
                            visualize_flow_checkbox = gr.Checkbox(
                                value=self.default_params["visualize_flow"],
                                label="Show Optical Flow Visualization"
                            )
                        
                        with gr.Column():
                            # Server control buttons
                            start_server_btn = gr.Button("🚀 Start TerraSLAM Mapping Engine", variant="primary")
                            stop_server_btn = gr.Button("⏹️ Stop TerraSLAM Mapping Engine", variant="secondary")
                            clear_keyframes_btn = gr.Button("🗑️ Clear Keyframes", variant="stop")
                            
                            # Status displays
                            server_status = gr.Textbox(
                                label="Server Status",
                                value="Stopped",
                                interactive=False
                            )
                            
                            keyframe_count = gr.Number(
                                label="Collected Keyframes",
                                value=0,
                                interactive=False
                            )
                
                # Processing Control Tab  
                with gr.TabItem("⚙️ Processing Control"):
                    with gr.Row():
                        with gr.Column():
                            # Processing parameters
                            conf_threshold_slider = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=self.default_params["conf_threshold"],
                                step=1,
                                label="Confidence Threshold (%)"
                            )
                            
                            mask_sky_checkbox = gr.Checkbox(
                                value=self.default_params["mask_sky"],
                                label="Filter Sky"
                            )
                            
                            mask_dynamic_checkbox = gr.Checkbox(
                                value=self.default_params["mask_dynamic"],
                                label="Filter Dynamic Objects"
                            )
                        
                        with gr.Column():
                            # Processing buttons
                            process_keyframes_btn = gr.Button("🔄 Process Keyframes with VGGT", variant="primary")
                            generate_confidence_btn = gr.Button("📊 Generate Confidence Map", variant="secondary")
                            export_results_btn = gr.Button("💾 Export Results", variant="secondary")
                            
                            # Processing status
                            processing_status = gr.Textbox(
                                label="Processing Status",
                                value="Ready",
                                interactive=False
                            )
                
                # Statistics Tab
                with gr.TabItem("📈 Statistics & Info"):
                    with gr.Row():
                        with gr.Column():
                            stats_display = gr.JSON(
                                label="Live Statistics",
                                value={}
                            )
                        
                        with gr.Column():
                            processing_log = gr.Textbox(
                                label="Processing Log",
                                value="System initialized",
                                lines=15,
                                interactive=False
                            )
            
            # Event handlers
            
            # TCP Server Controls
            start_server_btn.click(
                fn=self._start_server,
                inputs=[tcp_port_input, min_disparity_input, visualize_flow_checkbox],
                outputs=[server_status, processing_log]
            )
            
            stop_server_btn.click(
                fn=self._stop_server,
                outputs=[server_status, keyframe_count, processing_log]
            )
            
            clear_keyframes_btn.click(
                fn=self._clear_keyframes,
                outputs=[keyframe_count, processing_log]
            )
            
            # Processing Controls
            process_keyframes_btn.click(
                fn=self._process_keyframes,
                inputs=[conf_threshold_slider, mask_sky_checkbox, mask_dynamic_checkbox],
                outputs=[model_viewer, processing_status, processing_log, keyframe_count]
            )
            
            generate_confidence_btn.click(
                fn=self._generate_confidence_map,
                inputs=[conf_threshold_slider],
                outputs=[confidence_viewer, processing_log]
            )
            
            export_results_btn.click(
                fn=self._export_results,
                outputs=[processing_log]
            )
            
            # Auto-refresh keyframe count and statistics
            keyframe_count_refresh = gr.Timer(value=2.0)
            keyframe_count_refresh.tick(
                fn=self._update_statistics,
                outputs=[keyframe_count, stats_display]
            )
            
            # Store component references for internal use
            self._components = {
                "model_viewer": model_viewer,
                "confidence_viewer": confidence_viewer,
                "server_status": server_status,
                "keyframe_count": keyframe_count,
                "processing_status": processing_status,
                "processing_log": processing_log,
                "stats_display": stats_display
            }
        
        return interface
    
    def _start_server(self, tcp_port: int, min_disparity: float, visualize_flow: bool) -> Tuple[str, str]:
        """
        Start the TCP server for keyframe collection.
        
        Args:
            tcp_port: TCP port number
            min_disparity: Minimum disparity threshold
            visualize_flow: Enable optical flow visualization
            
        Returns:
            Tuple of (status_message, log_message)
        """
        try:
            if self.server_running:
                return "Already running", "TCP server is already running"
            
            # Create keyframe selector with specified parameters
            self.keyframe_selector = KeyframeSelector(
                port=int(tcp_port),
                min_disparity=float(min_disparity),
                visualize_flow=visualize_flow
            )
            
            # Add callback for keyframe notifications
            self.keyframe_selector.add_keyframe_callback(self._on_keyframe_selected)
            
            # Start server in background thread
            if self.keyframe_selector.start():
                self.server_running = True
                status = "Running"
                log_msg = f"TCP server started on port {tcp_port} with min_disparity={min_disparity}"
            else:
                status = "Failed to start"
                log_msg = "Failed to start TCP server"
            
            return status, log_msg
            
        except Exception as e:
            return "Error", f"Error starting server: {str(e)}"
    
    def _stop_server(self) -> Tuple[str, int, str]:
        """
        Stop the TCP server.
        
        Returns:
            Tuple of (status_message, keyframe_count, log_message)
        """
        try:
            if not self.server_running or not self.keyframe_selector:
                return "Stopped", 0, "TCP server was not running"
            
            self.keyframe_selector.stop()
            keyframes = self.keyframe_selector.get_keyframe_count()
            self.server_running = False
            
            log_msg = f"TCP server stopped. Collected {keyframes} keyframes."
            
            return "Stopped", keyframes, log_msg
            
        except Exception as e:
            return "Error", 0, f"Error stopping server: {str(e)}"
    
    def _clear_keyframes(self) -> Tuple[int, str]:
        """
        Clear collected keyframes.
        
        Returns:
            Tuple of (keyframe_count, log_message)
        """
        try:
            if self.keyframe_selector:
                self.keyframe_selector.clear_keyframes()
                return 0, "Keyframes cleared successfully"
            else:
                return 0, "No keyframe selector active"
                
        except Exception as e:
            return 0, f"Error clearing keyframes: {str(e)}"
    
    def _process_keyframes(self, conf_threshold: float, mask_sky: bool, mask_dynamic: bool) -> Tuple[str, str, str, int]:
        """
        Process collected keyframes with VGGT.
        
        Args:
            conf_threshold: Confidence threshold percentage
            mask_sky: Enable sky filtering
            mask_dynamic: Enable dynamic object filtering
            
        Returns:
            Tuple of (model_file, processing_status, log_message, keyframe_count)
        """
        try:
            if not self.keyframe_selector:
                return None, "Error", "No keyframe selector active", 0
            
            # Get collected keyframe paths
            keyframe_paths = self.keyframe_selector.get_keyframe_paths()
            
            if not keyframe_paths:
                return None, "No keyframes", "No keyframes collected yet", 0
            
            self.processing_active = True
            
            # Prepare processing parameters
            processing_params = {
                "conf_threshold": float(conf_threshold),
                "mask_sky": mask_sky,
                "mask_dynamic": mask_dynamic,
                "mask_black_bg": False,
                "mask_white_bg": False,
                "show_cam": True
            }
            
            log_msg = f"Processing {len(keyframe_paths)} keyframes..."
            
            # Process keyframes with VGGT
            results = self.vggt_mapper.process_keyframes(keyframe_paths, processing_params)
            
            if results["success"]:
                glb_path = results["scene_results"].get("glb_path")
                processing_time = results["timing"]["total_processing"]
                
                status = f"Completed in {processing_time:.1f}s"
                log_msg = f"VGGT processing completed in {processing_time:.1f}s. Model saved to {glb_path}"
                
                self.processing_active = False
                return glb_path, status, log_msg, len(keyframe_paths)
            else:
                error_msg = results.get("error", "Unknown error")
                self.processing_active = False
                return None, "Failed", f"Processing failed: {error_msg}", len(keyframe_paths)
                
        except Exception as e:
            self.processing_active = False
            return None, "Error", f"Error processing keyframes: {str(e)}", 0
    
    def _generate_confidence_map(self, conf_threshold: float) -> Tuple[str, str]:
        """
        Generate confidence map for NFN visualization.
        
        Args:
            conf_threshold: Confidence threshold percentage
            
        Returns:
            Tuple of (confidence_model_file, log_message)
        """
        try:
            # Get latest results from VGGT mapper
            latest_results = self.vggt_mapper.get_latest_results()
            
            if "error" in latest_results:
                return None, "No VGGT processing results available. Process keyframes first."
            
            # Check if confidence mapping was already generated
            if latest_results.get("confidence_scene"):
                # Re-export with new threshold if needed
                processing_params = {"conf_threshold": float(conf_threshold)}
                predictions = latest_results["predictions"]
                
                # Get target directory from latest results
                target_dir = latest_results.get("scene_results", {}).get("target_directory")
                if not target_dir:
                    return None, "No target directory available. Please process keyframes first to create input_stream directory."
                confidence_results = self.vggt_mapper._generate_confidence_mapping(predictions, processing_params, target_dir)
                
                if "error" not in confidence_results:
                    conf_glb_path = confidence_results.get("glb_path")
                    stats = confidence_results.get("statistics", {})
                    
                    high_conf = stats.get("high_conf_points", 0)
                    total_points = stats.get("total_points", 0)
                    
                    log_msg = f"Confidence map generated: {high_conf}/{total_points} high-confidence points"
                    
                    return conf_glb_path, log_msg
                else:
                    return None, f"Confidence map generation failed: {confidence_results['error']}"
            else:
                return None, "No confidence data available. Process keyframes first."
                
        except Exception as e:
            return None, f"Error generating confidence map: {str(e)}"
    
    def _export_results(self) -> str:
        """
        Export processing results.
        
        Returns:
            Log message
        """
        try:
            if not self.vggt_mapper.latest_predictions:
                return "No results to export. Process keyframes first."
            
            # Get latest target directory or create new one
            latest_results = self.vggt_mapper.get_latest_results()
            target_dir = latest_results.get("scene_results", {}).get("target_directory")
            
            if not target_dir:
                # Create new input_stream directory if none exists
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_dir = f"input_stream_{timestamp}"
                os.makedirs(target_dir, exist_ok=True)
            
            # Export camera poses to the input_stream directory
            poses_path = os.path.join(target_dir, "camera_poses.json")
            if self.vggt_mapper.save_camera_poses_json(poses_path):
                return f"Camera poses exported to {poses_path}"
            else:
                return "Failed to export camera poses"
                
        except Exception as e:
            return f"Error exporting results: {str(e)}"
    
    def _update_statistics(self) -> Tuple[int, Dict]:
        """
        Update keyframe count and statistics display.
        
        Returns:
            Tuple of (keyframe_count, statistics_dict)
        """
        try:
            keyframe_count = 0
            stats = {}
            
            if self.keyframe_selector:
                keyframe_count = self.keyframe_selector.get_keyframe_count()
                stats = self.keyframe_selector.get_stats()
                
                # Add server status
                stats["server_running"] = self.server_running
                stats["processing_active"] = self.processing_active
                stats["last_update"] = datetime.now().strftime("%H:%M:%S")
            
            return keyframe_count, stats
            
        except Exception as e:
            return 0, {"error": str(e)}
    
    def _on_keyframe_selected(self, keyframe_info: Dict[str, Any]):
        """
        Callback for keyframe selection notifications.
        
        Args:
            keyframe_info: Information about selected keyframe
        """
        # This could be used for real-time updates or notifications
        # For now, just update the last update time
        self.last_update_time = datetime.now()
    
    def launch(self, **kwargs) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments for gr.Blocks.launch()
        """
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": self.host,
            "server_port": self.port,
            "share": False,
            "inbrowser": True
        }
        launch_params.update(kwargs)
        
        print(f"Launching VGGT Mapping Interface on http://{self.host}:{self.port}")
        print("Features:")
        print("  • TCP server for drone image collection (port 43322)")
        print("  • Optical flow keyframe selection")
        print("  • VGGT 3D reconstruction") 
        print("  • Dual viewer with confidence mapping")
        
        interface.launch(**launch_params)


# Example usage and testing
if __name__ == "__main__":
    # Create and launch interface
    mapping_interface = MappingGradioInterface()
    
    try:
        mapping_interface.launch()
    except KeyboardInterrupt:
        print("\nInterface interrupted")
    except Exception as e:
        print(f"Error running interface: {e}")