# Copyright (C) 2024 Carnegie Mellon University

"""Headless auto-mapping server.

A long-lived ``MappingSession`` collects frame+GPS pairs over TCP. A background
monitor watches the retained keyframe count; once it reaches the cap, the server
maps that batch into an *area* and writes everything to the output dir:

    reconstruct (VGGT / VGGT-Omega)
      -> GPS-align from the streamed pairs (if GPS was sent)
      -> NFN next-flight plan
      -> export: scene/ply/npz, model-input images, camera poses,
                 NFN json+kml+area kml, and an area.json (tag + GPS center/geohash)

Segmentation is **not** part of this loop — it is a separate, request-driven
service (``segment_area``): a client asks to segment a query on an area tag, and
the server reloads that area from disk and runs the segmenter (see ``areas.py``).

Everything is written under the process working directory (each run creates an
``<area_tag>/`` folder), so pointing the container's working dir at a mounted
volume puts every export on the host.

Config comes from ``ServerConfig`` (populated from env/CLI in ``launch_server``).
"""

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from swiftmap.core import constants, protocol
from swiftmap.core.session import MappingSession
from swiftmap.server import areas


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = protocol.TCP_PORT
    backbone: str = "vggt"
    segmenter: str = "sam3"
    site: str = "area"
    max_keyframes: int = constants.DEFAULT_MAX_KEYFRAMES
    conf_threshold: float = constants.DEFAULT_CONF_THRESHOLD
    mask_sky: bool = True
    mask_dynamic: bool = False
    keep_all: bool = False
    min_disparity: float = constants.DEFAULT_MIN_DISPARITY
    output_dir: str = "output"
    poll_interval: float = 1.0
    viewer_host: str = "0.0.0.0"
    viewer_port: int = constants.GUI_PORT


class AutoMappingServer:
    """Collects keyframes over TCP and auto-runs the pipeline at the cap."""

    def __init__(self, config: ServerConfig):
        self.cfg = config
        os.makedirs(config.output_dir, exist_ok=True)
        os.chdir(config.output_dir)
        self._root = os.getcwd()

        self.session = MappingSession(host=config.host,
                                      min_disparity=config.min_disparity)
        self.session.max_keyframes = config.max_keyframes
        self.session.set_backbone(config.backbone)
        self.session.set_segmenter(config.segmenter)

        self._lock = threading.RLock()
        self.latest_run: dict = {}
        self._processing = False
        self._run_idx = 0
        self._stop = threading.Event()

    def run(self):
        """Start collecting and block, running the pipeline whenever the cap fills."""
        print(f"[swiftmap-server] starting on {self.cfg.host}:{self.cfg.port}")
        print(f"[swiftmap-server] site={self.cfg.site} backbone={self.cfg.backbone} "
              f"segmenter={self.cfg.segmenter} cap={self.cfg.max_keyframes}")
        print(f"[swiftmap-server] areas -> {self._root}")

        if not self.session.start(port=self.cfg.port, keep_all=self.cfg.keep_all):
            raise RuntimeError("Failed to start the TCP collection server")

        self._start_viewer()

        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            print("[swiftmap-server] interrupted")
        finally:
            self.session.stop()

    def _monitor_loop(self):
        """Poll the retained keyframe count; run the pipeline when it hits the cap."""
        while not self._stop.is_set():
            count = self.session.get_keyframe_count()
            if not self._processing and count >= self.cfg.max_keyframes:
                self._run_pipeline()
            time.sleep(self.cfg.poll_interval)

    _MIN_KEYFRAMES = 1

    def _run_pipeline(self):
        """Map one batch into an area: reconstruct -> GPS-align -> NFN -> export."""
        with self._lock:
            self._run_pipeline_locked()

    def map_now(self) -> dict:
        """Force-map the current (partial) batch immediately, then start a fresh one.

        Backs the viewer's "Map now" button — maps whatever keyframes are retained
        so far without waiting for the cap. Serialized against the auto-loop.
        """
        with self._lock:
            if self._processing:
                return {"error": "A mapping run is already in progress."}
            n = self.session.get_keyframe_count()
            if n < self._MIN_KEYFRAMES:
                return {"error": f"Only {n} keyframe(s) collected — need at least "
                                 f"{self._MIN_KEYFRAMES} to map."}
            self._run_pipeline_locked()
        lr = self.latest_run
        if lr:
            return {"success": True, "area_tag": lr.get("area_tag"),
                    "num_keyframes": lr.get("num_keyframes"),
                    "num_viewpoints": lr.get("num_viewpoints")}
        return {"error": "Mapping produced no result (check the server logs)."}

    def _run_pipeline_locked(self):
        self._processing = True
        self._run_idx += 1
        run = self._run_idx
        t0 = time.time()
        n = self.session.get_keyframe_count()
        created = datetime.now()
        area_tag = areas.make_area_tag(self.cfg.site, created)
        print(f"\n[swiftmap-server] === run #{run}: cap reached ({n} keyframes) "
              f"-> area '{area_tag}' ===")
        try:
            params = {
                "output_name": area_tag,
                "conf_threshold": float(self.cfg.conf_threshold),
                "mask_sky": self.cfg.mask_sky,
                "mask_dynamic": self.cfg.mask_dynamic,
                "mask_black_bg": False,
                "mask_white_bg": False,
                "show_cam": True,
            }
            result = self.session.reconstruct(params)
            if not result.get("success"):
                print(f"[swiftmap-server] run #{run} reconstruction failed: "
                      f"{result.get('error')}")
                return
            target_dir = result.get("scene_results", {}).get("target_directory")

            if self.session.has_stream_gps():
                cfg = self.session.align_gps(use_icp=False)
                if "error" in cfg:
                    print(f"[swiftmap-server] GPS align skipped: {cfg['error']}")
                else:
                    print(f"[swiftmap-server] GPS aligned "
                          f"(RMSE {cfg.get('rmse', float('nan')):.2f} m)")
            else:
                print("[swiftmap-server] no streamed GPS — exports stay in local coords")

            plan = self.session.plan(low_percentile=constants.NFN_LOW_PERCENTILE,
                                     high_percentile=constants.NFN_HIGH_PERCENTILE)
            print(f"[swiftmap-server] NFN: "
                  f"{plan.get('error') or str(plan.get('num_viewpoints', 0)) + ' viewpoints'}")

            preds = self.session.latest_predictions
            self.session.export_camera_poses()
            self.session.export_nfn_plan()
            areas.export_model_input_images(preds, target_dir)
            areas.write_area_metadata(target_dir, area_tag, self.cfg.site, created, n,
                                      preds, self.session.gps_transform)
            print(f"[swiftmap-server] run #{run} exported area '{area_tag}' "
                  f"in {time.time() - t0:.1f}s")

            self.latest_run = {
                "run": run, "area_tag": area_tag,
                "target_dir": os.path.abspath(target_dir),
                "scene_glb": self._run_artifact(target_dir, "scene.glb"),
                "confidence_glb": self._run_artifact(target_dir, "confidence_map.glb"),
                "num_keyframes": n,
                "num_viewpoints": plan.get("num_viewpoints", 0) if isinstance(plan, dict) else 0,
                "gps_aligned": self.session.gps_transform is not None,
                "elapsed": time.time() - t0,
            }

        except Exception as e:
            import traceback
            print(f"[swiftmap-server] run #{run} pipeline error: {e}")
            traceback.print_exc()
        finally:
            self.session.clear_keyframes()
            self._processing = False

    @staticmethod
    def _run_artifact(target_dir, name):
        """Absolute path to a run artifact if it exists, else None."""
        p = os.path.abspath(os.path.join(target_dir, name))
        return p if os.path.exists(p) else None

    def _start_viewer(self):
        """Launch the passive Gradio results viewer (non-blocking); headless if it fails."""
        try:
            from swiftmap.server.viewer import launch_viewer
            launch_viewer(self, host=self.cfg.viewer_host, port=self.cfg.viewer_port)
            print(f"[swiftmap-server] results viewer -> "
                  f"http://{self.cfg.viewer_host}:{self.cfg.viewer_port}")
        except Exception as e:
            print(f"[swiftmap-server] viewer disabled ({e}); running headless")

    def viewer_state(self) -> dict:
        """Snapshot of the latest run for the viewer, plus live status."""
        state = dict(self.latest_run)
        state["processing"] = self._processing
        state["keyframes"] = self.session.get_keyframe_count()
        state["cap"] = self.cfg.max_keyframes
        return state

    def list_area_tags(self) -> List[str]:
        """Area tags available to segment (newest first)."""
        return [m["area_tag"] for m in areas.list_areas(self._root)]

    def segment_area(self, area_tag: str, query: str) -> dict:
        """Segment ``query`` on a stored ``area_tag`` (request-driven service).

        Decoupled from the mission loop: reloads the area from disk. Serialized
        against the auto-pipeline only because they share the GPU. The returned
        ``glb_path`` is made absolute so the viewer can serve it.
        """
        with self._lock:
            res = areas.segment_area(self._root, area_tag, query,
                                     self.session.segmenter, self.cfg.conf_threshold)
        glb = res.get("glb_path")
        if glb and not os.path.isabs(glb):
            res["glb_path"] = os.path.abspath(glb)
        return res
