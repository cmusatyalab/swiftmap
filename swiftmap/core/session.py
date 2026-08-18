# Copyright (C) 2024 Carnegie Mellon University

"""The mapping session: the one gateway between the server and the core.

It owns the transport (TCP ingest + keyframe selection), the reconstruction backbone,
the GPS aligner, the NFN planner, and the map database. A run goes:

    drone --TCP--> _on_frame -> selector.is_keyframe          collect
    new_map() -> reconstruct() -> align_gps() -> plan()       map
    grow_site() / render_map() / segment_map()                store and serve

The session is long-lived, so the (expensive) model survives capture start/stop cycles;
``start(port, ...)`` controls only the transport.
"""

import os
import glob
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from swiftmap.core import constants
from swiftmap.core.database import Database
from swiftmap.core.pipeline.utils import render as render_utils
from swiftmap.core.transport import protocol
from swiftmap.core.transport.keyframe_selector import KeyframeSelector
from swiftmap.core.transport.tcp_server import MappingTCPServer
from swiftmap.core.pipeline.reconstructor import get_mapper, is_registered
from swiftmap.core.pipeline.segmentor import get_segmenter, lift
from swiftmap.core.pipeline.next_flight_planner import NextFlightPlanner, write_plan
from swiftmap.core.pipeline.gps_transformer import GpsTransformer


class MappingSession:
    """Owns and coordinates the full mapping pipeline for one run."""

    def __init__(self,
                 host: str = "0.0.0.0",
                 min_disparity: float = constants.DEFAULT_MIN_DISPARITY,
                 root: str = None,
                 site: str = "map"):
        self.host = host

        # The results root this session reads and writes.
        self.db = Database(root or os.getcwd(), site)

        # Long-lived stages; the model loads lazily on first reconstruct.
        self.selector = KeyframeSelector(min_disparity=min_disparity)
        # Chosen at runtime: there is no mapper until set_backbone().
        self.backbone: Optional[str] = None
        self.mapper = None
        self.planner = NextFlightPlanner()
        self.aligner = GpsTransformer()

        # Text-promptable segmentation (SAM 3), built lazily on first request.
        self.segmenter = None

        # Optional local->GPS alignment (set via calibrate_gps()).
        self.gps_transform = None
        # Latest NFN plan (set by plan(); used by export).
        self.latest_plan = None

        # A new frame evicts the lowest-disparity one only if it scores higher; 0 disables.
        self.max_keyframes = constants.DEFAULT_MAX_KEYFRAMES

        # Transport is created on start() (its port is chosen then).
        self.tcp_server: Optional[MappingTCPServer] = None
        self.port: Optional[int] = None

        # Scratch dir for keyframe JPEGs; survives a stop, wiped by the next start().
        self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_session_")
        # Live GPS trace, rewritten to stay 1:1 with the retained keyframes.
        self.stream_gps_csv_path = os.path.join(self.temp_dir, "stream_gps.csv")
        self._stream_gps_rows = 0
        self._gps_csv_lock = threading.Lock()

        # Retained keyframes {"path", "gps"} in capture order.
        self._buffer: List[Dict[str, Any]] = []
        self._total_selected = 0  # keyframes ever selected (for stats)
        self._paths_lock = threading.Lock()
        # GPS of the keyframes sent to VGGT (set by reconstruct(); 1:1 with poses).
        self._reconstructed_gps: List = []

        self.is_running = False
        self.server_thread = None
        self.start_time = None

    # ============================================================ collection stage
    def start(self,
              port: int = protocol.TCP_PORT,
              min_disparity: Optional[float] = None,
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

        # Fresh capture: clear selector, keyframe buffer, scratch dir, and GPS trace.
        self.selector.reset()
        with self._paths_lock:
            self._buffer.clear()
            self._total_selected = 0
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
        """TCP-server callback: keep this frame? The saved keyframe joins the buffer
        on the next drain."""
        return self.selector.is_keyframe(image)

    # ---------------------------------------------------------- streamed GPS trace
    def _reset_stream_gps_csv(self):
        """Truncate the live GPS trace and write just the header (fresh capture)."""
        with self._gps_csv_lock:
            self._stream_gps_rows = 0
            with open(self.stream_gps_csv_path, "w") as f:
                f.write("latitude,longitude,altitude\n")

    def _write_gps_csv(self, entries):
        """Rewrite the GPS trace to mirror the retained keyframes (capture order)."""
        with self._gps_csv_lock:
            rows = 0
            with open(self.stream_gps_csv_path, "w") as f:
                f.write("latitude,longitude,altitude\n")
                for e in entries:
                    g = e["gps"]
                    if g is not None:
                        f.write(f"{g[0]:.8f},{g[1]:.8f},{g[2]:.3f}\n")
                        rows += 1
            self._stream_gps_rows = rows

    def has_stream_gps(self) -> bool:
        """True once at least one retained keyframe carries GPS."""
        return self._stream_gps_rows > 0

    def _wipe_temp_dir(self):
        """Remove keyframe JPEGs from the scratch dir (keeps the dir itself)."""
        for p in glob.glob(os.path.join(self.temp_dir, "keyframe_*")):
            self._remove_file(p)

    @staticmethod
    def _remove_file(path):
        try:
            os.remove(path)
        except OSError:
            pass

    # ---------------------------------------------------------------- keyframe buffer
    def _drain(self):
        """Fold newly-saved keyframes into the buffer and keep the GPS CSV in sync."""
        if not self.tcp_server:
            return
        with self._paths_lock:
            snapshot = self._drain_locked()
        if snapshot is not None:
            self._write_gps_csv(snapshot)

    def _drain_locked(self):
        """Same as ``_drain`` but the caller holds ``_paths_lock``. Returns the new
        buffer snapshot if it changed, else None."""
        changed = False
        for kf in self.tcp_server.get_collected_keyframes():
            path = kf.get("path")
            if not path or not os.path.exists(path):
                continue
            self._buffer.append({"path": path, "gps": kf.get("metadata", {}).get("gps")})
            self._total_selected += 1
            changed = True
        changed |= self._trim_to_cap()
        return list(self._buffer) if changed else None

    def _trim_to_cap(self) -> bool:
        """Drop the oldest keyframes while the buffer is over the cap. Caller holds the lock."""
        cap = self.max_keyframes
        if not cap or len(self._buffer) <= cap:
            return False
        for e in self._buffer[:-cap]:
            self._remove_file(e["path"])
        del self._buffer[:-cap]
        return True

    def get_keyframe_paths(self) -> List[str]:
        """JPEG paths of the retained keyframes, in capture order."""
        self._drain()
        with self._paths_lock:
            return [e["path"] for e in self._buffer]

    def get_keyframe_count(self) -> int:
        """Number of keyframes currently retained (<= the cap)."""
        self._drain()
        with self._paths_lock:
            return len(self._buffer)

    def clear_keyframes(self):
        """Discard all collected keyframes and reset selection state."""
        if self.tcp_server:
            self.tcp_server.clear_keyframes()
        self.selector.reset()
        with self._paths_lock:
            self._buffer.clear()
            self._total_selected = 0
        self._reconstructed_gps.clear()
        self._wipe_temp_dir()
        self._reset_stream_gps_csv()

    def send_to_client(self, payload: bytes):
        """Deliver payload back to the connected client"""
        if self.tcp_server:
            self.tcp_server.queue_outbound(payload)

    def configure_disparity_threshold(self, min_disparity: float):
        self.selector.configure_disparity_threshold(min_disparity)

    # =============================================================== mapping stage
    def set_backbone(self, name: str) -> Dict[str, Any]:
        """Select the reconstruction backbone by key (e.g. 'vggt', 'vggt_omega').

        Instantiating a backbone is cheap (weights load lazily on first
        reconstruct); switching to a different backbone drops the previous one
        and its results. Re-selecting the current backbone is a no-op.
        """
        if not is_registered(name):
            return {"error": f"Unknown model '{name}'."}
        if self.backbone == name and self.mapper is not None:
            return {"backbone": name, "changed": False}
        self.mapper = get_mapper(name)
        self.backbone = name
        print(f"Reconstruction backbone set to: {name}")
        return {"backbone": name, "changed": True}

    def set_segmenter(self, name: str) -> Dict[str, Any]:
        """Select the segmentation backend by key (e.g. 'sam3'); built lazily."""
        from swiftmap.core.pipeline.segmentor import is_registered as seg_registered
        if not seg_registered(name):
            return {"error": f"Unknown segmentation model '{name}'."}
        if getattr(self.segmenter, "name", None) == name:
            return {"segmenter": name, "changed": False}
        self.segmenter = get_segmenter(name)
        print(f"Segmentation model set to: {name}")
        return {"segmenter": name, "changed": True}

    def reconstruct(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run reconstruction on the retained keyframes (already <= the cap)."""
        if self.mapper is None:
            return {"success": False, "error": "No model selected — choose a model first.",
                    "keyframe_count": 0}
        with self._paths_lock:
            if self.tcp_server:
                self._drain_locked()
            entries = list(self._buffer)
            paths = [e["path"] for e in entries]
            self._reconstructed_gps = [e["gps"] for e in entries]
            self._buffer.clear()
        if not paths:
            return {"success": False, "error": "No keyframes collected yet",
                    "keyframe_count": 0}
        print(f"Reconstructing {len(paths)} keyframes (cap={self.max_keyframes})")
        return self.mapper.process_keyframes(paths, params)

    def get_latest_results(self) -> Dict[str, Any]:
        """Latest reconstruction results (or {'error': ...} if none yet)."""
        if self.mapper is None:
            return {"error": "No model selected — choose a model first."}
        return self.mapper.get_latest_results()

    @property
    def latest_predictions(self) -> Optional[Dict[str, Any]]:
        """The latest raw predictions, or None if no model / nothing processed yet."""
        return self.mapper.latest_predictions if self.mapper is not None else None

    def segment_map(self, map_tag: str, query: str,
                    conf_threshold: float = None) -> Dict[str, Any]:
        """Segment a stored map (the viewer's request path)."""
        m = self.db.get(map_tag)
        if m is None:
            return {"error": f"Unknown map '{map_tag}'."}
        conf = constants.DEFAULT_CONF_THRESHOLD if conf_threshold is None else float(conf_threshold)
        if self.segmenter is None:
            self.segmenter = get_segmenter(constants.DEFAULT_SEGMENTER)
        return lift.segment_map(m, query, self.segmenter, conf)

    def render_map(self, map_tag: str, conf_level: float) -> Dict[str, Any]:
        """Render a stored map's GLBs at ``conf_level`` (the viewer's request path)."""
        m = self.db.get(map_tag)
        if m is None:
            return {"error": f"Unknown map '{map_tag}'."}
        return render_utils.render(m, conf_level)

    def new_map(self, created=None):
        """Create the map this run will write into."""
        return self.db.create_map(created)

    def grow_site(self, m, conf_thres: float = 50.0, voxel_size: float = 0.1, created=None):
        """Merge a stored map into the site."""
        return self.db.grow(m, conf_thres=conf_thres, voxel_size=voxel_size, created=created)

    def map_tags(self):
        """Selectable map tags (site first, then stored maps)."""
        return self.db.tags()

    @property
    def site(self):
        """The growing map this session merges into."""
        return self.db.site

    def maps(self):
        """Stored maps, newest first."""
        return self.db.maps()

    @property
    def maps_dir(self) -> str:
        """Where stored maps live."""
        return self.db.maps_dir

    # ============================================================== planning stage
    def plan(self, low_percentile: float = 60.0,
             high_percentile: float = 80.0) -> Dict[str, Any]:
        """Run Next Flight Navigation on the latest reconstruction.

        Returns the plan dict, or {'error': ...} if there's no usable reconstruction.
        """
        latest = self.get_latest_results()
        if "error" in latest:
            return {"error": "No reconstruction yet — process keyframes first."}
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
        preds = self.latest_predictions
        if not preds or "camera_positions" not in preds:
            return {"error": "No reconstruction with camera poses yet — process keyframes first."}

        if isinstance(gps_lla, str):
            gps_lla = self.aligner.load_gps_csv(gps_lla)

        slam_xyz = np.asarray(preds["camera_positions"]).reshape(-1, 3)
        gps_arr = np.asarray(gps_lla, dtype=float).reshape(-1, 3)

        # Synced (no-ICP) mode requires a 1:1 GPS-per-keyframe correspondence.
        if not use_icp and len(gps_arr) != len(slam_xyz):
            return {"error": (f"Synced mode needs one GPS row per keyframe, but got "
                              f"{len(slam_xyz)} keyframes and {len(gps_arr)} GPS rows. "
                              f"Use 'Keep all frames' + a per-frame GPS CSV, or uncheck "
                              f"'GPS synced 1:1' to use ICP.")}

        try:
            self.gps_transform, cfg = self.aligner.from_calibration(
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
        preds = self.latest_predictions
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
            self.gps_transform, cfg = self.aligner.from_calibration(
                slam, gps_arr, use_icp=False)
        except Exception as e:
            return {"error": f"GPS alignment failed: {e}"}
        cfg["mode"] = "stream-synced"
        print(f"GPS aligned (stream-synced): scale={cfg['scale']:.3f}, "
              f"RMSE={cfg['rmse']:.3f} m ({cfg['num_points']} points)")
        return cfg

    def align_gps(self, use_icp: bool, gps_csv_path: Optional[str] = None) -> Dict[str, Any]:
        """Unified GPS alignment entry point.

        Resolves the GPS source, then aligns the reconstruction's camera trajectory:

          * with ICP    -> treat GPS as an *unsynced* trajectory (counts may differ);
                           any GPS trace works.
          * without ICP -> require GPS *synced 1:1* with the keyframes; error otherwise.

        GPS source: an uploaded CSV if one is given, otherwise the GPS that streamed
        paired with each frame (the ongoing trace). Returns the transform cfg (with a
        'mode' tag) or {'error': ...}.
        """
        preds = self.latest_predictions
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
        latest = self.get_latest_results()
        target_dir = latest.get("scene_results", {}).get("target_directory")
        if not target_dir:
            target_dir = f"input_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def export_nfn_plan(self) -> Optional[str]:
        """Write the latest NFN plan (GPS-tagged when aligned) to the run dir."""
        if not self.latest_plan:
            return None
        return write_plan(self.latest_plan, self.gps_transform, self._target_dir())
    def export_camera_poses(self) -> Optional[str]:
        """Write estimated camera poses to the run's output dir; return the path."""
        if not self.latest_predictions:
            return None
        poses_path = os.path.join(self._target_dir(), "camera_poses.json")
        return poses_path if self.mapper.save_camera_poses_json(poses_path) else None
    # ======================================================================= stats
    def get_stats(self) -> Dict[str, Any]:
        """Combined session statistics (selection + transport)."""
        stats = self.selector.get_stats()
        if self.tcp_server:
            stats["tcp_server_stats"] = self.tcp_server.get_stats()
        stats["keyframes_selected"] = self._total_selected
        stats["keyframes_retained"] = len(self._buffer)
        if self.start_time:
            stats["session_duration"] = str(datetime.now() - self.start_time)
        return stats
