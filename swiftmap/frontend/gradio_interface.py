# Copyright (C) 2024 Carnegie Mellon University

"""
Gradio web GUI for SwiftMap.

Two 3D viewers on top (reconstruction + confidence map) and control tabs below:
keyframe collection, model selection (reconstruction + segmentation), processing
(3D mapping, confidence, segmentation, GPS, NFN), selectable exports, and stats.
Processes TCP-collected frames — there is no image-upload widget.
"""

import os
import sys
import gradio as gr
from swiftmap.frontend import _gradio_compat  # noqa: F401  (patches gradio_client schema parsing)
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add vggt root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_root = os.path.dirname(os.path.dirname(current_dir))
if vggt_root not in sys.path:
    sys.path.append(vggt_root)

from swiftmap.core import constants
from swiftmap.core.transport import protocol
from swiftmap.core.session import MappingSession

# GPS alignment method choices (radio labels).
GPS_WITH_ICP = "With ICP — unsynced GPS trajectory (only GPS needed)"
GPS_NO_ICP = "Without ICP — GPS synced 1:1 with keyframes"


class MappingGradioInterface:
    """Gradio web GUI for SwiftMap (wraps a MappingSession)."""

    def __init__(self, host: str = "0.0.0.0", port: int = constants.GUI_PORT):
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
        self.nfn_port = constants.NFN_VISER_PORT

        # Default control values shown in the UI.
        self.default_params = {
            "tcp_port": protocol.TCP_PORT,
            "min_disparity": constants.DEFAULT_MIN_DISPARITY,
            "conf_threshold": constants.DEFAULT_CONF_THRESHOLD,
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
        except Exception:
            theme = gr.themes.Default()
        
        with gr.Blocks(theme=theme, title="SwiftMap Mapping System") as interface:
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🚁 SwiftMap Mapping System</h1>
                <p>Real-time drone mapping with keyframe selection and 3D reconstruction</p>
            </div>
            """)
            
            # Dual 3D viewers (large, side by side)
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
            
            # Control panels below the viewers
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
                            
                            keep_all_checkbox = gr.Checkbox(
                                value=False,
                                label="Without disparity-scored selection"
                            )

                            min_disparity_input = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=self.default_params["min_disparity"],
                                step=1,
                                label="Min Disparity Threshold (pixels)"
                            )

                            max_keyframes_input = gr.Number(
                                value=constants.DEFAULT_MAX_KEYFRAMES,
                                label="Max Frames (0 = no cap)",
                                precision=0,
                                minimum=0
                            )

                            visualize_flow_checkbox = gr.Checkbox(
                                value=self.default_params["visualize_flow"],
                                label="Show keyframe preview"
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

                    # Live keyframe preview (updates while "Show keyframe preview" is
                    # enabled): the raw last frame without disparity scoring, or the
                    # optical-flow overlay of the last keyframe with it.
                    with gr.Row():
                        flow_preview = gr.Image(
                            label="Keyframe Preview (enable 'Show keyframe preview')",
                            height=320,
                            interactive=False
                        )

                # Processing Control Tab
                with gr.TabItem("⚙️ Processing Control"):
                    # Model gate — choose the models first. Reconstruction controls
                    # enable once a reconstruction model is picked; the segmentation
                    # prompt enables once a segmentation model is picked. Everything
                    # below starts disabled (grayed out).
                    recon_choices = [(m["label"], m["name"])
                                     for m in self.session.available_backbones()]
                    seg_choices = [(m["label"], m["name"])
                                   for m in self.session.available_segmenter_models()]
                    with gr.Row():
                        recon_selector = gr.Radio(
                            choices=recon_choices, value=None,
                            label="① Reconstruction model",
                            info="Runs 3D Mapping / confidence / NFN. Switching clears the run.")
                        seg_selector = gr.Radio(
                            choices=seg_choices, value=None,
                            label="① Segmentation model",
                            info="Enables the text-prompt segmentation below.")

                    with gr.Row():
                        with gr.Column():
                            conf_threshold_slider = gr.Slider(
                                minimum=0, maximum=100,
                                value=self.default_params["conf_threshold"], step=1,
                                label="Confidence Threshold (%)", interactive=False)
                            mask_sky_checkbox = gr.Checkbox(
                                value=self.default_params["mask_sky"],
                                label="Filter Sky", interactive=False)
                            mask_dynamic_checkbox = gr.Checkbox(
                                value=self.default_params["mask_dynamic"],
                                label="Filter Dynamic Objects", interactive=False)

                            # Semantic segmentation prompt — shares the confidence
                            # threshold above; the Segment button (right column) runs
                            # it and the left viewer shows queried points red.
                            seg_query_input = gr.Textbox(
                                label="Segment query", placeholder="e.g. person",
                                interactive=False)

                            # GPS alignment source: uploaded CSV, or the live trace
                            # auto-filled here while streaming GPS-tagged frames.
                            gps_csv_input = gr.File(
                                label="GPS Trace CSV (auto-filled while streaming GPS)",
                                file_types=[".csv"], interactive=False)
                            gps_method_radio = gr.Radio(
                                choices=[GPS_NO_ICP, GPS_WITH_ICP], value=GPS_NO_ICP,
                                label="GPS Alignment Method",
                                info=("Without ICP needs GPS synced 1:1 with keyframes "
                                      "(paired stream, or a matching CSV). With ICP works "
                                      "from any GPS trajectory."),
                                interactive=False)
                            gps_status = gr.Textbox(
                                label="GPS Alignment", value="Not aligned", interactive=False)

                        with gr.Column():
                            process_keyframes_btn = gr.Button("🔄 3D Mapping", variant="primary", interactive=False)
                            generate_confidence_btn = gr.Button("📊 Generate Confidence Map", variant="secondary", interactive=False)
                            segment_btn = gr.Button("🔴 Segment", variant="secondary", interactive=False)
                            calibrate_gps_btn = gr.Button("🛰️ Calibrate GPS Alignment", variant="secondary", interactive=False)
                            analyze_nfn_btn = gr.Button("🧭 Analyze with NFN (opens new page)", variant="secondary", interactive=False)

                            # Selectable exports. Each option stays grayed until its
                            # artifact exists (validity refreshed live by the timer).
                            gr.Markdown("**Export**")
                            export_poses_cb = gr.Checkbox(label="Camera poses (JSON)", value=False, interactive=False)
                            export_nfn_cb = gr.Checkbox(label="NFN viewpoints (JSON + KML)", value=False, interactive=False)
                            export_seg_cb = gr.Checkbox(label="Segmented objects (JSON + KML)", value=False, interactive=False)
                            export_results_btn = gr.Button("💾 Export selected", variant="secondary", interactive=False)

                            processing_status = gr.Textbox(
                                label="Processing Status",
                                value="Select a reconstruction model to begin", interactive=False)
                            nfn_link = gr.Markdown(
                                "Select a model and run **3D Mapping** first, then click "
                                "**Analyze with NFN** to open the coverage-gap & viewpoint "
                                "viewer in a new page.")

                    # Gated groups (order must match the handler return order).
                    recon_gated = [
                        conf_threshold_slider, mask_sky_checkbox, mask_dynamic_checkbox,
                        gps_csv_input, gps_method_radio,
                        process_keyframes_btn, generate_confidence_btn,
                        calibrate_gps_btn, analyze_nfn_btn,
                    ]
                    seg_gated = [seg_query_input, segment_btn]
                    export_controls = [export_poses_cb, export_nfn_cb, export_seg_cb,
                                       export_results_btn]
                
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
            
            # Apply the collection cap to the session live, so it bounds the keyframe
            # buffer during capture (not just at reconstruct time).
            max_keyframes_input.change(fn=self._set_max_keyframes, inputs=[max_keyframes_input])

            # The disparity threshold only applies with disparity scoring -> grey it
            # out when "without disparity-scored selection" is checked.
            keep_all_checkbox.change(
                fn=lambda without: gr.update(interactive=not without),
                inputs=[keep_all_checkbox], outputs=[min_disparity_input])

            # Model gate: reconstruction selection enables the reconstruction
            # controls; segmentation selection enables the segment prompt.
            recon_selector.change(
                fn=self._select_reconstruction,
                inputs=[recon_selector],
                outputs=[processing_status] + recon_gated
            )
            seg_selector.change(
                fn=self._select_segmentation,
                inputs=[seg_selector],
                outputs=seg_gated
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

            # Semantic segmentation: render the queried objects (red) in the left
            # viewer, replacing the plain reconstruction with the highlighted cloud.
            segment_btn.click(
                fn=self._segment,
                inputs=[seg_query_input, conf_threshold_slider],
                outputs=[model_viewer, processing_status, processing_log]
            )

            calibrate_gps_btn.click(
                fn=self._calibrate_gps,
                inputs=[gps_csv_input, gps_method_radio],
                outputs=[gps_status, processing_log]
            )

            analyze_nfn_btn.click(
                fn=self._run_nfn,
                inputs=[conf_threshold_slider],
                outputs=[nfn_link, processing_log]
            )

            export_results_btn.click(
                fn=self._export_results,
                inputs=[export_poses_cb, export_nfn_cb, export_seg_cb],
                outputs=[processing_status, processing_log]
            )

            # Auto-refresh: keyframe count, stats, optical-flow preview, the live GPS
            # trace file box, and the validity (enabled/disabled) of export options.
            keyframe_count_refresh = gr.Timer(value=2.0)
            keyframe_count_refresh.tick(
                fn=self._update_ui,
                inputs=[gps_csv_input],
                outputs=[keyframe_count, stats_display, flow_preview, gps_csv_input]
                        + export_controls
            )

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

    def _set_max_keyframes(self, max_keyframes):
        """Apply the collection cap to the session so it bounds the live buffer."""
        try:
            self.session.max_keyframes = max(0, int(max_keyframes))
        except (TypeError, ValueError):
            pass

    # Sizes of the two gated groups (must match recon_gated / seg_gated in
    # create_interface()).
    _NUM_RECON_GATED = 9
    _NUM_SEG_GATED = 2

    def _select_reconstruction(self, model_name: str):
        """Gate the reconstruction controls on the chosen reconstruction model.

        Returns ``(processing_status, *recon_gated-updates)``.
        """
        disabled = [gr.update(interactive=False)] * self._NUM_RECON_GATED
        enabled = [gr.update(interactive=True)] * self._NUM_RECON_GATED
        if not model_name:
            return ("Select a reconstruction model to begin", *disabled)
        result = self.session.set_backbone(model_name)
        if "error" in result:
            return (f"⚠️ {result['error']}", *disabled)
        label = next((m["label"] for m in self.session.available_backbones()
                      if m["name"] == model_name), model_name)
        return (f"Model: {label} — ready. Collect keyframes, then 3D Mapping.", *enabled)

    def _select_segmentation(self, model_name: str):
        """Gate the segmentation prompt on the chosen segmentation model.

        Returns updates for ``seg_gated`` (seg query box, Segment button).
        """
        if not model_name:
            return [gr.update(interactive=False)] * self._NUM_SEG_GATED
        result = self.session.set_segmenter(model_name)
        enabled = "error" not in result
        return [gr.update(interactive=enabled)] * self._NUM_SEG_GATED

    def _segment(self, query: str, conf_threshold: float) -> Tuple[Optional[str], str, str]:
        """Run SAM 3 segmentation and show the highlighted (red) cloud on the left.

        Uses the same confidence threshold as reconstruction so the segmented
        points match the displayed cloud. Returns
        (segmented_glb_for_model_viewer, processing_status, log).
        """
        try:
            res = self.session.segment(query, conf_threshold=float(conf_threshold))
            if res.get("success"):
                tag = " (GPS-tagged)" if res.get("gps_aligned") else " — calibrate GPS to geotag objects"
                status = f"Segmented '{res['query']}': {res['num_objects']} object(s)"
                log = (f"Segmentation: '{res['query']}' @ conf>={res['conf_threshold']:.0f}% -> "
                       f"{res['num_points']} points, {res['num_objects']} object(s){tag}. "
                       f"Run NFN/Export to write per-object GPS.")
                return res.get("glb_path"), status, log
            return None, "Segmentation failed", res.get("error", "Unknown error")
        except Exception as e:
            return None, "Error", f"Segmentation error: {e}"

    def _process_keyframes(self, conf_threshold: float, mask_sky: bool, mask_dynamic: bool,
                           max_keyframes: int = constants.DEFAULT_MAX_KEYFRAMES
                           ) -> Tuple[str, str, str, int]:
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
                backbone = results.get("backbone", "model")
                status = f"Completed in {processing_time:.1f}s"
                log_msg = (f"{backbone} processing completed in {processing_time:.1f}s. "
                           f"Model saved to {glb_path}")
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

    def _calibrate_gps(self, gps_file, method: str = GPS_NO_ICP) -> Tuple[str, str]:
        """Align the reconstruction to GPS (so NFN viewpoints get lat/lon/alt).

        method selects with/without ICP; the GPS source is the uploaded CSV, or the
        live streamed trace when none is uploaded. The session validates availability
        and (for the no-ICP path) that the GPS is synced 1:1 with the keyframes.
        """
        try:
            use_icp = method == GPS_WITH_ICP
            # gr.File gives a filepath string (or an object with .name); None if empty.
            gps_path = getattr(gps_file, "name", gps_file) if gps_file else None

            cfg = self.session.align_gps(use_icp=use_icp, gps_csv_path=gps_path)
            if "error" in cfg:
                return f"⚠️ Not aligned — {cfg['error']}", f"GPS: {cfg['error']}"

            mode = cfg.get("mode", "icp" if use_icp else "synced")
            init_rmse = cfg.get("init_rmse", float("nan"))
            status = (f"✅ Aligned ({mode}) — scale {cfg['scale']:.3f}, "
                      f"RMSE {cfg['rmse']:.2f} m ({cfg['num_points']} pts)")
            log = (f"GPS alignment [{mode}]: scale={cfg['scale']:.4f}, "
                   f"RMSE={cfg['rmse']:.3f} m (init {init_rmse:.3f} m, "
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
            plan = self.session.plan(low_percentile=constants.NFN_LOW_PERCENTILE,
                                     high_percentile=constants.NFN_HIGH_PERCENTILE)
            if "error" in plan:
                return f"⚠️ {plan['error']}", f"NFN: {plan['error']}"

            # No viewpoints (e.g. saturated/uniform confidence): report why, don't
            # open an empty viewer.
            if plan.get("num_viewpoints", 0) == 0:
                msg = plan.get("statistics", {}).get(
                    "message", "No low-confidence regions to target.")
                return f"### ⚠️ No next-flight plan\n{msg}", f"NFN: {msg}"

            predictions = self.session.latest_predictions
            conf_threshold = (constants.DEFAULT_CONF_THRESHOLD
                              if conf_threshold is None else float(conf_threshold))

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

    def _export_results(self, want_poses: bool, want_nfn: bool,
                        want_seg: bool) -> Tuple[str, str]:
        """Export the selected artifacts. Returns (status, detailed file log)."""
        try:
            if not any([want_poses, want_nfn, want_seg]):
                return "⚠️ Nothing selected", "Select at least one export option."

            msgs, items, out_dir = [], [], None
            targets = []
            if want_poses:
                targets.append(("camera poses", self.session.export_camera_poses, "process keyframes first"))
            if want_nfn:
                targets.append(("NFN viewpoints", self.session.export_nfn_plan, "run NFN first"))
            if want_seg:
                targets.append(("segmented objects", self.session.export_segmented_objects, "run Segment first"))

            for label, export_fn, hint in targets:
                path = export_fn()
                if path:
                    msgs.append(f"{label} → {path}")
                    items.append(label)
                    out_dir = os.path.dirname(path)
                else:
                    msgs.append(f"{label}: nothing to export ({hint})")

            if not items:
                return "⚠️ Nothing was exported", "\n".join(msgs)
            where = f" to {out_dir}" if out_dir else ""
            return f"✅ Exported {', '.join(items)}{where}", "\n".join(msgs)

        except Exception as e:
            return "❌ Export failed", f"Error exporting results: {str(e)}"
    
    def _update_ui(self, current_gps_file):
        """Timer tick: refresh keyframe count, stats, flow preview, GPS file box,
        and the validity (enabled state) of the export options.

        Returns gr.update() where nothing should change so we neither clear the
        optical-flow image when viz is off nor fight a user's uploaded CSV.
        """
        keyframe_count, stats = self._update_statistics()

        # Optical-flow preview (None while viz is off -> leave the image as-is).
        vis = self.session.latest_flow_vis() if self.session else None
        flow_out = vis if vis is not None else gr.update()

        # Auto-fill the GPS box with the live trace while streaming GPS, unless the
        # user uploaded their own CSV (anything not our stream_gps.csv).
        gps_out = gr.update()
        cur = getattr(current_gps_file, "name", current_gps_file)
        cur = str(cur) if cur else ""
        has_stream = bool(self.session and self.session.has_stream_gps())
        stream_csv = self.session.stream_gps_csv_path if has_stream else None
        user_uploaded = bool(cur) and not cur.endswith("stream_gps.csv")
        if stream_csv and not user_uploaded and cur != stream_csv:
            gps_out = stream_csv

        return (keyframe_count, stats, flow_out, gps_out, *self._export_option_updates())

    def _export_option_updates(self):
        """Enable each export option only once its artifact exists; else grey it out
        (and uncheck). Order: camera poses, NFN viewpoints, segmented objects, button."""
        s = self.session
        poses_ok = bool(s and s.latest_predictions is not None)
        nfn_ok = bool(s and s.latest_plan is not None)
        seg_ok = bool(s and s.latest_segmentation is not None)

        def opt(valid):
            # Enable (preserve the user's check) when valid; grey out + uncheck when not.
            return gr.update(interactive=True) if valid else gr.update(interactive=False, value=False)

        return (opt(poses_ok), opt(nfn_ok), opt(seg_ok),
                gr.update(interactive=(poses_ok or nfn_ok or seg_ok)))

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