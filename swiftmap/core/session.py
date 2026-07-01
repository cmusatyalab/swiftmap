# Copyright (C) 2024 Carnegie Mellon University

"""
Mapping Session

The orchestration layer for one mapping run. A ``MappingSession`` is the single
middle layer between the frontend and the core stages — it owns and coordinates
all of them:

    collection :  MappingTCPServer (transport) + KeyframeSelector (decision)
    mapping    :  VGGTMapper       (reconstruction + confidence)
    planning   :  NextFlightPlanner (NFN)

The pipeline for a run:

    drone --TCP--> tcp_server --frame--> _on_frame -> selector.is_keyframe   (collect)
    reconstruct() -> mapper.process_keyframes(<collected keyframes>)         (map)
    generate_confidence_map() / plan()                                        (evaluate + replan)

The frontend drives the session (start/stop, reconstruct, plan, export) and reads
state from it; it never touches sockets, the model, or the planner directly. The
session is long-lived — created once — so the (expensive) VGGT model persists
across capture start/stop cycles. ``start(port, …)`` controls only the transport.
"""

import os
import glob
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from swiftmap.core import constants, protocol
from swiftmap.core.keyframe_selector import KeyframeSelector
from swiftmap.core.tcp_server import MappingTCPServer
from swiftmap.core.vggt_mapper import VGGTMapper
from swiftmap.core.nfn import NextFlightPlanner


class MappingSession:
    """Owns and coordinates the full mapping pipeline for one run."""

    def __init__(self,
                 host: str = "0.0.0.0",
                 min_disparity: float = constants.DEFAULT_MIN_DISPARITY,
                 visualize_flow: bool = False):
        self.host = host

        # Core stages (all long-lived; the model loads lazily on first reconstruct).
        self.selector = KeyframeSelector(min_disparity=min_disparity,
                                         visualize_flow=visualize_flow)
        self.mapper = VGGTMapper()
        self.planner = NextFlightPlanner()

        # Optional local->GPS alignment (set via calibrate_gps()).
        self.gps_transform = None
        # Latest NFN plan (set by plan(); used by export).
        self.latest_plan = None

        # Second-round cap: after optical-flow selection, the keyframe set passed to
        # VGGT is capped to this many, keeping the highest-priority (highest-disparity)
        # ones. 0 disables the cap. Keeps the VGGT batch within memory.
        self.max_keyframes = constants.DEFAULT_MAX_KEYFRAMES

        # Transport is created on start() (its port is chosen then).
        self.tcp_server: Optional[MappingTCPServer] = None
        self.port: Optional[int] = None

        # Session-owned scratch dir, reused across start/stop so keyframes survive a
        # Stop (a fresh start() wipes it). The TCP server writes keyframe JPEGs here.
        self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_session_")
        # Ongoing GPS trace written live as GPS-tagged keyframes stream in, so the UI
        # can show "a GPS file is present" exactly like a user upload.
        self.stream_gps_csv_path = os.path.join(self.temp_dir, "stream_gps.csv")
        self._stream_gps_rows = 0
        self._gps_csv_lock = threading.Lock()

        # Collected keyframe file paths, accumulated as the server queue drains.
        self._keyframe_paths: List[str] = []
        self._paths_lock = threading.Lock()
        self._keyframe_count = 0

        # Per-keyframe GPS [lat, lon, alt] paired with each selected keyframe (in
        # selection order; None when a frame arrived without GPS). Lets us calibrate
        # directly from the matched stream pairs (no CSV, no ICP).
        self._keyframe_gps: List = []
        # GPS for the capped keyframes actually sent to VGGT (set by reconstruct()).
        self._reconstructed_gps: List = []

        self.is_running = False
        self.server_thread = None
        self.start_time = None

    # ============================================================ collection stage
    def start(self,
              port: int = protocol.TCP_PORT,
              min_disparity: Optional[float] = None,
              visualize_flow: Optional[bool] = None,
              keep_all: Optional[bool] = None) -> bool:
        """Start the TCP transport and begin collecting keyframes.

        If ``keep_all`` is True, keyframe selection is skipped and every received
        frame is kept (denser trajectory; mind VGGT's batch limit).
        """
        if self.is_running:
            print("Mapping session already running")
            return True

        if min_disparity is not None:
            self.selector.configure_disparity_threshold(min_disparity)
        if visualize_flow is not None:
            self.selector.visualize_flow = visualize_flow
        if keep_all is not None:
            self.selector.keep_all = keep_all

        self.port = port
        self.tcp_server = MappingTCPServer(host=self.host, port=port,
                                           keyframe_callback=self._on_frame,
                                           temp_dir=self.temp_dir)

        print("Starting SwiftMap Mapping Session...")
        print(f"Server: {self.host}:{port} | min disparity: {self.selector.min_disparity} px")

        if not self.tcp_server.initialize():
            print("Failed to initialize TCP server")
            return False

        # Fresh capture: clear selector, keyframe lists, scratch dir, and GPS trace.
        self.selector.reset()
        self._keyframe_count = 0
        with self._paths_lock:
            self._keyframe_paths.clear()
        self._keyframe_gps.clear()
        self._reconstructed_gps.clear()
        self._wipe_temp_dir()
        self._reset_stream_gps_csv()
        self.start_time = datetime.now()

        self.server_thread = threading.Thread(target=self.tcp_server.start_server,
                                              daemon=True)
        self.server_thread.start()

        self.is_running = True
        print("Mapping session started; ready to receive drone images")
        return True

    def stop(self):
        """Stop the TCP transport (the session and its model stay alive)."""
        if not self.is_running:
            return
        print("Stopping SwiftMap Mapping Session...")
        self.is_running = False
        if self.tcp_server:
            self.tcp_server.stop_server()
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        cv2.destroyAllWindows()  # close optical-flow visualization windows, if any
        print("Mapping session stopped")

    def _on_frame(self, image, metadata: Dict[str, Any]) -> bool:
        """TCP-server callback: run selection, count keyframes, keep paired GPS."""
        is_keyframe = self.selector.is_keyframe(image)
        if is_keyframe:
            self._keyframe_count += 1
            gps = metadata.get("gps")            # paired GPS (or None)
            self._keyframe_gps.append(gps)
            if gps is not None:
                self._append_stream_gps(gps)     # grow the live GPS trace CSV
        return is_keyframe

    # ---------------------------------------------------------- streamed GPS trace
    def _reset_stream_gps_csv(self):
        """Truncate the live GPS trace and write just the header (fresh capture)."""
        with self._gps_csv_lock:
            self._stream_gps_rows = 0
            with open(self.stream_gps_csv_path, "w") as f:
                f.write("latitude,longitude,altitude\n")

    def _append_stream_gps(self, gps):
        """Append one (lat, lon, alt) row to the live GPS trace CSV."""
        with self._gps_csv_lock:
            with open(self.stream_gps_csv_path, "a") as f:
                f.write(f"{gps[0]:.8f},{gps[1]:.8f},{gps[2]:.3f}\n")
            self._stream_gps_rows += 1

    def has_stream_gps(self) -> bool:
        """True once at least one GPS-tagged keyframe has streamed in."""
        return self._stream_gps_rows > 0

    def _wipe_temp_dir(self):
        """Remove keyframe JPEGs from the scratch dir (keeps the dir itself)."""
        for p in glob.glob(os.path.join(self.temp_dir, "keyframe_*")):
            try:
                os.remove(p)
            except OSError:
                pass

    def latest_flow_vis(self):
        """Latest optical-flow preview (RGB) for the UI, or None if viz is off."""
        return self.selector.latest_flow_vis

    def _drain(self):
        """Pull newly-collected keyframes from the server queue into our list
        (the server's getter drains its internal queue)."""
        if not self.tcp_server:
            return
        for kf in self.tcp_server.get_collected_keyframes():
            path = kf.get("path")
            if path and os.path.exists(path):
                with self._paths_lock:
                    self._keyframe_paths.append(path)

    def get_keyframe_paths(self) -> List[str]:
        """File paths of all keyframes collected so far."""
        self._drain()
        with self._paths_lock:
            return list(self._keyframe_paths)

    def get_keyframe_count(self) -> int:
        """Number of keyframes selected so far (updated live as frames arrive)."""
        return self._keyframe_count

    def clear_keyframes(self):
        """Discard all collected keyframes and reset selection state."""
        if self.tcp_server:
            self.tcp_server.clear_keyframes()
        self.selector.reset()
        with self._paths_lock:
            self._keyframe_paths.clear()
        self._keyframe_count = 0
        self._keyframe_gps.clear()
        self._reconstructed_gps.clear()
        self._wipe_temp_dir()
        self._reset_stream_gps_csv()

    def configure_disparity_threshold(self, min_disparity: float):
        self.selector.configure_disparity_threshold(min_disparity)

    # =============================================================== mapping stage
    def reconstruct(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run VGGT reconstruction on the collected keyframes (capped if needed)."""
        paths = self._capped_keyframe_paths()
        if not paths:
            return {"success": False, "error": "No keyframes collected yet",
                    "keyframe_count": 0}
        return self.mapper.process_keyframes(paths, params)

    def _capped_keyframe_paths(self) -> List[str]:
        """Keyframe paths after the second-round cap.

        If more than ``max_keyframes`` were collected, keep the highest-priority
        (highest-disparity) ones; falls back to even subsampling when priorities
        aren't usable (e.g. keep-all). Returns paths in capture order.
        """
        paths = self.get_keyframe_paths()
        gps = list(self._keyframe_gps)
        cap = self.max_keyframes
        if not cap or len(paths) <= cap:
            keep = list(range(len(paths)))
        else:
            values = list(self.selector.keyframe_values)
            if len(values) == len(paths) and len(set(values)) > 1:
                # Priority: keep the top-`cap` by disparity, then restore capture order.
                order = sorted(range(len(paths)), key=lambda i: values[i], reverse=True)
                keep = sorted(order[:cap])
            else:
                # No usable priority -> evenly spaced across the sequence.
                keep = sorted(set(np.linspace(0, len(paths) - 1, cap).round().astype(int).tolist()))
            print(f"Keyframe cap: {len(paths)} -> {len(keep)} (max_keyframes={cap})")

        # Carry the paired GPS for exactly the kept keyframes (same order as paths).
        self._reconstructed_gps = [gps[i] if i < len(gps) else None for i in keep]
        return [paths[i] for i in keep]

    def get_latest_results(self) -> Dict[str, Any]:
        """Latest reconstruction results (or {'error': ...} if none yet)."""
        return self.mapper.get_latest_results()

    @property
    def latest_predictions(self) -> Optional[Dict[str, Any]]:
        """The latest raw VGGT predictions, or None if nothing processed yet."""
        return self.mapper.latest_predictions

    def generate_confidence_map(self, conf_threshold: float) -> Dict[str, Any]:
        """(Re)generate the map-quality (confidence) point cloud from the latest run."""
        latest = self.mapper.get_latest_results()
        if "error" in latest:
            return {"error": "No VGGT processing results available. Process keyframes first."}
        if not latest.get("confidence_scene"):
            return {"error": "No confidence data available. Process keyframes first."}
        target_dir = latest.get("scene_results", {}).get("target_directory")
        if not target_dir:
            return {"error": "No target directory available. Process keyframes first."}
        params = {"conf_threshold": float(conf_threshold)}
        return self.mapper._generate_confidence_mapping(
            latest["predictions"], params, target_dir)

    # ============================================================== planning stage
    def plan(self, low_percentile: float = 60.0,
             high_percentile: float = 80.0) -> Dict[str, Any]:
        """Run Next Flight Navigation on the latest reconstruction.

        Returns the plan dict, or {'error': ...} if there's no usable reconstruction.
        """
        latest = self.mapper.get_latest_results()
        if "error" in latest:
            return {"error": "No VGGT results yet — process keyframes first."}
        predictions = latest.get("predictions", {})
        if "world_points" not in predictions or "world_points_conf" not in predictions:
            return {"error": "Predictions have no world points — cannot run NFN."}
        self.planner.low_percentile = low_percentile
        self.planner.high_percentile = high_percentile
        plan = self.planner.plan(predictions)

        if self.gps_transform is not None:
            for vp in plan.get("viewpoints", []):
                pos_gps = self.to_gps(vp["camera_position"]).tolist()  # [lat, lon, alt]
                tgt_gps = self.to_gps(vp["target"]).tolist()
                vp["camera_position_gps"] = pos_gps
                vp["target_gps"] = tgt_gps

        self.latest_plan = plan
        return plan

    # ========================================================= geo-alignment stage
    def calibrate_gps(self, gps_lla, use_icp: bool = True) -> Dict[str, Any]:
        """Align the reconstruction to GPS using the keyframe camera trajectory.

        Args:
            gps_lla: GPS trajectory as an (M, 3) array of [lat, lon, alt], or a path
                     to a CSV with latitude/longitude/altitude columns, in capture order.
            use_icp: if True, treat the GPS as an *unsynced* trajectory and refine the
                     fit with ICP (count may differ from the keyframes). If False, treat
                     the GPS as **exactly matched 1:1 with the keyframes** (row i = GPS of
                     keyframe i) and do a single direct Umeyama fit -- this requires the
                     GPS count to equal the keyframe count.

        Returns the transform config (scale, rotation, translation, origin, rmse),
        or {'error': ...} on failure.
        """
        from swiftmap.core.geo_transform import geo

        preds = self.mapper.latest_predictions
        if not preds or "camera_positions" not in preds:
            return {"error": "No reconstruction with camera poses yet — process keyframes first."}

        if isinstance(gps_lla, str):
            gps_lla = geo.load_gps_csv(gps_lla)

        slam_xyz = np.asarray(preds["camera_positions"]).reshape(-1, 3)
        gps_arr = np.asarray(gps_lla, dtype=float).reshape(-1, 3)

        # Synced (no-ICP) mode requires a 1:1 GPS-per-keyframe correspondence.
        if not use_icp and len(gps_arr) != len(slam_xyz):
            return {"error": (f"Synced mode needs one GPS row per keyframe, but got "
                              f"{len(slam_xyz)} keyframes and {len(gps_arr)} GPS rows. "
                              f"Use 'Keep all frames' + a per-frame GPS CSV, or uncheck "
                              f"'GPS synced 1:1' to use ICP.")}

        try:
            self.gps_transform, cfg = geo.GpsTransform.from_calibration(
                slam_xyz, gps_arr, use_icp=use_icp)
        except Exception as e:
            return {"error": f"GPS alignment failed: {e}"}
        cfg["mode"] = "icp" if use_icp else "synced"
        print(f"GPS aligned ({cfg['mode']}): scale={cfg['scale']:.3f}, "
              f"RMSE={cfg['rmse']:.3f} m ({cfg['num_points']} points)")
        return cfg

    def calibrate_gps_from_stream(self) -> Dict[str, Any]:
        """Align using the GPS that streamed *paired* with each keyframe.

        Uses the per-keyframe GPS collected during streaming (1:1 with the camera
        poses of the last reconstruction) and does a single direct Umeyama fit --
        no CSV upload, no ICP. Run after reconstruct().
        """
        from swiftmap.core.geo_transform import geo

        preds = self.mapper.latest_predictions
        if not preds or "camera_positions" not in preds:
            return {"error": "No reconstruction with camera poses yet — process keyframes first."}

        cams = np.asarray(preds["camera_positions"]).reshape(-1, 3)
        gps = self._reconstructed_gps
        if not gps or len(gps) != len(cams):
            return {"error": (f"No paired stream GPS to align ({len(cams)} poses vs "
                              f"{len(gps)} gps). Stream frames carrying GPS, then reconstruct.")}

        # Keep only keyframes that actually had GPS (1:1, same order).
        pairs = [(c, g) for c, g in zip(cams, gps) if g is not None]
        if len(pairs) < 3:
            return {"error": f"Only {len(pairs)} keyframes have GPS; need at least 3."}
        slam = np.array([c for c, _ in pairs], dtype=float)
        gps_arr = np.array([g for _, g in pairs], dtype=float)

        try:
            self.gps_transform, cfg = geo.GpsTransform.from_calibration(
                slam, gps_arr, use_icp=False)
        except Exception as e:
            return {"error": f"GPS alignment failed: {e}"}
        cfg["mode"] = "stream-synced"
        print(f"GPS aligned (stream-synced): scale={cfg['scale']:.3f}, "
              f"RMSE={cfg['rmse']:.3f} m ({cfg['num_points']} points)")
        return cfg

    def align_gps(self, use_icp: bool, gps_csv_path: Optional[str] = None) -> Dict[str, Any]:
        """Unified GPS alignment entry point used by the UI.

        Resolves the GPS source, then aligns the reconstruction's camera trajectory:

          * with ICP    -> treat GPS as an *unsynced* trajectory (counts may differ);
                           any GPS trace works.
          * without ICP -> require GPS *synced 1:1* with the keyframes; error otherwise.

        GPS source: an uploaded CSV if one is given, otherwise the GPS that streamed
        paired with each frame (the ongoing trace). Returns the transform cfg (with a
        'mode' tag) or {'error': ...}.
        """
        preds = self.mapper.latest_predictions
        if not preds or "camera_positions" not in preds:
            return {"error": "No reconstruction with camera poses yet — process keyframes first."}

        # A missing path, or one that is our own live trace, means "use stream pairs".
        use_stream = (not gps_csv_path) or (
            os.path.abspath(gps_csv_path) == os.path.abspath(self.stream_gps_csv_path))

        if use_stream and not self.has_stream_gps():
            return {"error": "No GPS available. Stream frames carrying GPS, or upload a "
                             "GPS trace CSV, then reconstruct."}

        if not use_icp:
            # Synced (exact-pairs) mode.
            if use_stream:
                return self.calibrate_gps_from_stream()            # in-memory paired GPS
            return self.calibrate_gps(gps_csv_path, use_icp=False)  # user CSV, must be 1:1

        # ICP mode: any GPS trajectory works (counts need not match).
        gps_src = self.stream_gps_csv_path if use_stream else gps_csv_path
        return self.calibrate_gps(gps_src, use_icp=True)

    def to_gps(self, points):
        """Convert local point(s) to GPS [lat, lon, alt]; None if not calibrated."""
        if self.gps_transform is None:
            return None
        return self.gps_transform.to_lla(points)

    def _target_dir(self) -> str:
        """The current run's output directory (created if there isn't one yet)."""
        latest = self.mapper.get_latest_results()
        target_dir = latest.get("scene_results", {}).get("target_directory")
        if not target_dir:
            target_dir = f"input_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def export_nfn_plan(self) -> Optional[str]:
        """Write the latest NFN plan (GPS-tagged when aligned) to the run dir.

        Writes ``next_flight_viewpoints.json`` (+ ``transform.json`` if GPS-aligned)
        and returns the viewpoints path, or None if there's no plan yet.
        """
        if not self.latest_plan:
            return None
        import json

        target_dir = self._target_dir()
        viewpoints = []
        for i, vp in enumerate(self.latest_plan.get("viewpoints", [])):
            item = {
                "id": i,                                   # matches Viser "v{i}" and the log "#{i}"
                "cluster_id": int(vp.get("cluster_id", -1)),  # which ground-plane cell (not the marker label)
                "position": np.asarray(vp["camera_position"], dtype=float).tolist(),
                "look_dir": np.asarray(vp["camera_rotation"], dtype=float)[:, 2].tolist(),
                "target": np.asarray(vp["target"], dtype=float).tolist(),
                "score": float(vp.get("score", 0.0)),
            }
            if "camera_position_gps" in vp:
                item["position_gps"] = vp["camera_position_gps"]  # drone waypoint [lat, lon, 0]
            if "target_gps" in vp:
                item["target_gps"] = vp["target_gps"]            # ground patch [lat, lon, 0]
            viewpoints.append(item)

        out = {
            "num_viewpoints": len(viewpoints),
            "thresholds": self.latest_plan.get("thresholds", {}),
            "gps_aligned": self.gps_transform is not None,
            "viewpoints": viewpoints,
        }
        path = os.path.join(target_dir, "next_flight_viewpoints.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        if self.gps_transform is not None:
            with open(os.path.join(target_dir, "transform.json"), "w") as f:
                json.dump(self.gps_transform.cfg, f, indent=2)
            # KML of the target (ground) GPS, ready to import into Google My Maps.
            from swiftmap.core.nfn import kml
            kml.write_kml(viewpoints, os.path.join(target_dir, "next_flight_viewpoints.kml"),
                          gps_key="target_gps", doc_name="SwiftMap NFN Targets")

        return path

    # ================================================================ export stage
    def export_camera_poses(self) -> Optional[str]:
        """Write estimated camera poses to the run's output dir; return the path."""
        if not self.mapper.latest_predictions:
            return None
        latest = self.mapper.get_latest_results()
        target_dir = latest.get("scene_results", {}).get("target_directory")
        if not target_dir:
            target_dir = f"input_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(target_dir, exist_ok=True)
        poses_path = os.path.join(target_dir, "camera_poses.json")
        if self.mapper.save_camera_poses_json(poses_path):
            return poses_path
        return None

    # ======================================================================= stats
    def get_stats(self) -> Dict[str, Any]:
        """Combined session statistics (selection + transport)."""
        stats = self.selector.get_stats()
        if self.tcp_server:
            stats["tcp_server_stats"] = self.tcp_server.get_stats()
        stats["keyframes_selected"] = self._keyframe_count
        if self.start_time:
            stats["session_duration"] = str(datetime.now() - self.start_time)
        return stats
