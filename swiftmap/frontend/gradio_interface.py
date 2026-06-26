# Copyright (C) 2024 Carnegie Mellon University

"""
Gradio Interface for SwiftMap Mapping System

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
from swiftmap.frontend import _gradio_compat  # noqa: F401  (patches gradio_client schema parsing)
import numpy as np
import time
import threading
from datetime import datetime
from typing import Dict, Tuple
import json

# Add vggt root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_root = os.path.dirname(os.path.dirname(current_dir))
if vggt_root not in sys.path:
    sys.path.append(vggt_root)

from swiftmap.core.session import MappingSession


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

        # The mapping pipeline orchestrator (owns transport, selector, mapper, planner).
        # Long-lived: created once so the VGGT model persists across capture start/stop.
        self.session = MappingSession(host=host)

        # Interface state
        self.server_running = False
        self.processing_active = False

        # NFN (Next Flight Navigation) Viser server state
        self.nfn_server = None
        self.nfn_port = 7867
        
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
        
        print("SwiftMap Mapping Gradio Interface initialized")
    
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
        
        with gr.Blocks(theme=theme, title="SwiftMap Mapping System") as interface:
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🚁 SwiftMap Mapping System</h1>
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
                with gr.TabItem("🌐 SwiftMap Mapping Engine Control"):
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

                            keep_all_checkbox = gr.Checkbox(
                                value=False,
                                label="Keep all frames (skip keyframe selection)"
                            )

                        with gr.Column():
                            # Server control buttons
                            start_server_btn = gr.Button("🚀 Start SwiftMap Mapping Engine", variant="primary")
                            stop_server_btn = gr.Button("⏹️ Stop SwiftMap Mapping Engine", variant="secondary")
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

                            max_keyframes_input = gr.Number(
                                value=70,
                                label="Max Keyframes (cap, 0 = no cap)",
                                precision=0,
                                minimum=0
                            )

                            # GPS alignment: upload a GPS trace CSV and align the
                            # reconstruction so NFN viewpoints get real lat/lon/alt.
                            gps_csv_input = gr.File(
                                label="GPS Traces CSV (latitude, longitude, altitude)",
                                file_types=[".csv"]
                            )
                            gps_synced_checkbox = gr.Checkbox(
                                value=False,
                                label="GPS synced 1:1 with keyframes (skip ICP, exact pairs)"
                            )
                            calibrate_gps_btn = gr.Button("🛰️ Calibrate GPS Alignment", variant="secondary")
                            gps_status = gr.Textbox(
                                label="GPS Alignment",
                                value="Not aligned",
                                interactive=False
                            )

                        with gr.Column():
                            # Processing buttons
                            process_keyframes_btn = gr.Button("🔄 Process Keyframes with VGGT", variant="primary")
                            generate_confidence_btn = gr.Button("📊 Generate Confidence Map", variant="secondary")
                            analyze_nfn_btn = gr.Button("🧭 Analyze with NFN (opens new page)", variant="secondary")
                            export_results_btn = gr.Button("💾 Export Results", variant="secondary")

                            # Processing status
                            processing_status = gr.Textbox(
                                label="Processing Status",
                                value="Ready",
                                interactive=False
                            )

                            # NFN result link / summary (opens the Viser viewer in a new page)
                            nfn_link = gr.Markdown(
                                "Run VGGT first, then click **Analyze with NFN** to open the "
                                "coverage-gap & viewpoint viewer in a new page."
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
                inputs=[tcp_port_input, min_disparity_input, visualize_flow_checkbox, keep_all_checkbox],
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
                inputs=[conf_threshold_slider, mask_sky_checkbox, mask_dynamic_checkbox, max_keyframes_input],
                outputs=[model_viewer, processing_status, processing_log, keyframe_count]
            )
            
            generate_confidence_btn.click(
                fn=self._generate_confidence_map,
                inputs=[conf_threshold_slider],
                outputs=[confidence_viewer, processing_log]
            )

            calibrate_gps_btn.click(
                fn=self._calibrate_gps,
                inputs=[gps_csv_input, gps_synced_checkbox],
                outputs=[gps_status, processing_log]
            )

            analyze_nfn_btn.click(
                fn=self._run_nfn,
                inputs=[conf_threshold_slider],
                outputs=[nfn_link, processing_log]
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
    
    def _start_server(self, tcp_port: int, min_disparity: float, visualize_flow: bool,
                      keep_all: bool = False) -> Tuple[str, str]:
        """
        Start the TCP server for keyframe collection.

        Args:
            tcp_port: TCP port number
            min_disparity: Minimum disparity threshold
            visualize_flow: Enable optical flow visualization
            keep_all: keep every frame (skip keyframe selection)

        Returns:
            Tuple of (status_message, log_message)
        """
        try:
            if self.server_running:
                return "Already running", "TCP server is already running"

            # Start collection on the (long-lived) session with the chosen settings.
            if self.session.start(port=int(tcp_port),
                                  min_disparity=float(min_disparity),
                                  visualize_flow=visualize_flow,
                                  keep_all=keep_all):
                self.server_running = True
                status = "Running"
                mode = "keep ALL frames" if keep_all else f"min_disparity={min_disparity}"
                log_msg = f"TCP server started on port {tcp_port} ({mode})"
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
            if not self.server_running or not self.session:
                return "Stopped", 0, "TCP server was not running"
            
            self.session.stop()
            keyframes = self.session.get_keyframe_count()
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
            if self.session:
                self.session.clear_keyframes()
                return 0, "Keyframes cleared successfully"
            else:
                return 0, "No keyframe selector active"
                
        except Exception as e:
            return 0, f"Error clearing keyframes: {str(e)}"
    
    def _process_keyframes(self, conf_threshold: float, mask_sky: bool, mask_dynamic: bool,
                           max_keyframes: int = 70) -> Tuple[str, str, str, int]:
        """
        Process collected keyframes with VGGT.

        Args:
            conf_threshold: Confidence threshold percentage
            mask_sky: Enable sky filtering
            mask_dynamic: Enable dynamic object filtering
            max_keyframes: cap on keyframes sent to VGGT (0 = no cap)

        Returns:
            Tuple of (model_file, processing_status, log_message, keyframe_count)
        """
        try:
            self.processing_active = True
            self.session.max_keyframes = int(max_keyframes)

            processing_params = {
                "conf_threshold": float(conf_threshold),
                "mask_sky": mask_sky,
                "mask_dynamic": mask_dynamic,
                "mask_black_bg": False,
                "mask_white_bg": False,
                "show_cam": True
            }

            # Reconstruct the collected keyframes via the session.
            results = self.session.reconstruct(processing_params)
            keyframe_count = results.get("keyframe_count", self.session.get_keyframe_count())

            if results.get("success"):
                glb_path = results["scene_results"].get("glb_path")
                processing_time = results["timing"]["total_processing"]
                status = f"Completed in {processing_time:.1f}s"
                log_msg = f"VGGT processing completed in {processing_time:.1f}s. Model saved to {glb_path}"
                self.processing_active = False
                return glb_path, status, log_msg, keyframe_count
            else:
                error_msg = results.get("error", "Unknown error")
                self.processing_active = False
                return None, "Failed", f"Processing failed: {error_msg}", keyframe_count

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
            confidence_results = self.session.generate_confidence_map(conf_threshold)

            if "error" in confidence_results:
                return None, confidence_results["error"]

            conf_glb_path = confidence_results.get("glb_path")
            stats = confidence_results.get("statistics", {})
            high_conf = stats.get("high_conf_points", 0)
            total_points = stats.get("total_points", 0)
            log_msg = f"Confidence map generated: {high_conf}/{total_points} high-confidence points"
            return conf_glb_path, log_msg

        except Exception as e:
            return None, f"Error generating confidence map: {str(e)}"

    def _calibrate_gps(self, gps_file, synced: bool = False) -> Tuple[str, str]:
        """Align the reconstruction to a GPS trace CSV (so NFN viewpoints get GPS).

        synced=True: GPS rows correspond 1:1 to keyframes -> direct Umeyama, no ICP.
        synced=False: unsynced trajectory -> ICP refinement.
        """
        try:
            if gps_file is None:
                return "Not aligned", "GPS: upload a GPS trace CSV first."
            # gr.File gives a filepath string (or an object with .name).
            gps_path = getattr(gps_file, "name", gps_file)

            cfg = self.session.calibrate_gps(gps_path, use_icp=not synced)
            if "error" in cfg:
                return "Not aligned", f"GPS: {cfg['error']}"

            mode = "exact pairs, no ICP" if synced else "trajectory ICP"
            status = (f"✅ Aligned ({mode}) — scale {cfg['scale']:.3f}, "
                      f"RMSE {cfg['rmse']:.2f} m ({cfg['num_points']} pts)")
            log = (f"GPS alignment [{mode}]: scale={cfg['scale']:.4f}, "
                   f"RMSE={cfg['rmse']:.3f} m (init {cfg['init_rmse']:.3f} m, "
                   f"{cfg['num_points']} pts).\n"
                   f"Origin ({cfg['lat0']:.6f}, {cfg['lon0']:.6f}, {cfg['alt0']:.1f}).\n"
                   f"NFN viewpoints will now include GPS coordinates.")
            return status, log
        except Exception as e:
            return "Not aligned", f"GPS calibration error: {e}"

    def _run_nfn(self, conf_threshold: float) -> Tuple[str, str]:
        """
        Run Next Flight Navigation (NFN) analysis on the latest VGGT reconstruction
        and (re)launch the Viser viewer in a separate page.

        Args:
            conf_threshold: Confidence threshold percentage (shared with reconstruction).

        Returns:
            Tuple of (markdown_link, log_message).
        """
        try:
            # Compute the next-flight plan via the session (NFN on the latest run).
            plan = self.session.plan(low_percentile=60.0, high_percentile=80.0)
            if "error" in plan:
                return f"⚠️ {plan['error']}", f"NFN: {plan['error']}"

            predictions = self.session.latest_predictions
            conf_threshold = 60.0 if conf_threshold is None else float(conf_threshold)

            # Lazy import so the rest of the app works even without viser installed
            from swiftmap.frontend.viewers.viser_view import visualize_nfn_with_viser

            # (Re)start the Viser viewer on a fixed port
            if self.nfn_server is not None:
                try:
                    self.nfn_server.stop()
                except Exception:
                    pass
                self.nfn_server = None

            self.nfn_server = visualize_nfn_with_viser(
                predictions,
                plan,
                port=self.nfn_port,
                conf_threshold=float(conf_threshold),
                background_mode=True,
            )

            url = f"http://localhost:{self.nfn_port}"
            stats = plan.get("statistics", {})
            n_vp = plan.get("num_viewpoints", 0)
            link_md = (
                f"### 🧭 NFN viewer ready\n"
                f"**[▶ Open in a new page]({url})**  "
                f"(remote: `http://<server-ip>:{self.nfn_port}`)\n\n"
                f"- 🔴 To-improve points (P60–P80 band): **{stats.get('num_enhance_points', 0)}**\n"
                f"- 🟡 Clusters: **{stats.get('num_clusters', 0)}**\n"
                f"- 🔵 Suggested viewpoints: **{n_vp}**\n"
                f"- 🟢 Existing cameras + confidence-colored point cloud"
            )
            # Detailed log: each suggested viewpoint's location + look direction
            th = plan.get("thresholds", {})
            log_lines = [
                f"NFN plan ({datetime.now().strftime('%H:%M:%S')})",
                f"Band P{th.get('low_percentile', 60):.0f}-P{th.get('high_percentile', 80):.0f} | "
                f"to-improve {stats.get('num_enhance_points', 0)} pts | "
                f"{stats.get('num_clusters', 0)} clusters | {n_vp} viewpoints",
                "Suggested viewpoints  [#] position (x, y, z) -> look-dir (dx, dy, dz):",
            ]
            gps_aligned = any("camera_position_gps" in vp for vp in plan.get("viewpoints", []))
            if gps_aligned:
                log_lines[-1] += "  [GPS: lat, lon, alt]"
            for i, vp in enumerate(plan.get("viewpoints", [])):
                p = np.asarray(vp["camera_position"])
                d = np.asarray(vp["camera_rotation"])[:, 2]  # +Z = camera forward / look direction
                line = (f"  #{i:02d}  pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
                        f"dir=({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})")
                gps = vp.get("camera_position_gps")
                if gps:
                    line += f"  gps=({gps[0]:.6f}, {gps[1]:.6f}, {gps[2]:.1f})"
                log_lines.append(line)
            log_lines.append(f"Viewer: {url}")
            log_msg = "\n".join(log_lines)
            return link_md, log_msg

        except Exception as e:
            return f"❌ NFN failed: {e}", f"NFN error: {e}"

    def _export_results(self) -> str:
        """
        Export processing results.
        
        Returns:
            Log message
        """
        try:
            if not self.session.latest_predictions:
                return "No results to export. Process keyframes first."

            msgs = []
            poses_path = self.session.export_camera_poses()
            msgs.append(f"Camera poses → {poses_path}" if poses_path
                        else "Failed to export camera poses")

            # NFN plan (+ transform.json if GPS-aligned), if NFN has been run.
            nfn_path = self.session.export_nfn_plan()
            if nfn_path:
                msgs.append(f"NFN viewpoints → {nfn_path}")
                if self.session.gps_transform is not None:
                    msgs.append("GPS transform → transform.json")
                    msgs.append("Targets KML (Google My Maps) → next_flight_viewpoints.kml")

            return "\n".join(msgs)

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
            
            if self.session:
                keyframe_count = self.session.get_keyframe_count()
                stats = self.session.get_stats()
                
                # Add server status
                stats["server_running"] = self.server_running
                stats["processing_active"] = self.processing_active
                stats["last_update"] = datetime.now().strftime("%H:%M:%S")
            
            return keyframe_count, stats
            
        except Exception as e:
            return 0, {"error": str(e)}
    
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
            "inbrowser": False
        }
        launch_params.update(kwargs)
        
        print(f"Launching SwiftMap Mapping Interface on http://{self.host}:{self.port}")
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