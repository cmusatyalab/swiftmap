# Copyright (C) 2024 Carnegie Mellon University

"""Headless auto-mapping server (grow-and-merge).

A long-lived ``MappingSession`` collects frame+GPS pairs over TCP. A background
monitor watches the retained keyframe count; once it reaches the cap, the server
reconstructs that batch and **merges it into the growing** ``Site``:

    reconstruct (VGGT / VGGT-Omega) -> GPS-align -> NFN plan
      -> store the batch as a map under ``maps/``
      -> grow the site with it (GPS co-registration, origin pinned so coordinates
         never drift, near-duplicate points collapsed).

The database root holds ``maps/`` (every generated map, kept and individually
segmentable) and ``site/`` (the one growing map, the merge of them all). On startup the
server resumes the existing site. Segmentation is request-driven; the site keeps its own
artifacts as it grows.

Everything is written under the process working directory; pointing the container's
working dir at a mounted volume puts the site on the host.
"""

import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from swiftmap.core import constants
from swiftmap.core.transport import protocol
from swiftmap.core.database import Database
from swiftmap.core.pipeline import renderer
from swiftmap.core.pipeline.segmentor import lift
from swiftmap.core.session import MappingSession


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = protocol.TCP_PORT
    backbone: str = "vggt"
    segmenter: str = "sam3"
    site: str = "map"
    max_keyframes: int = constants.DEFAULT_MAX_KEYFRAMES
    conf_threshold: float = constants.DEFAULT_CONF_THRESHOLD
    merge_voxel: float = 0.1
    mask_sky: bool = True
    mask_dynamic: bool = False
    keep_all: bool = False
    min_disparity: float = constants.DEFAULT_MIN_DISPARITY
    output_dir: str = "output"
    poll_interval: float = 1.0
    viewer_host: str = "0.0.0.0"
    viewer_port: int = constants.GUI_PORT


class AutoMappingServer:
    """Collects keyframes over TCP; at the cap, stores the batch and grows the ``Site``."""

    _STAGING = "_staging"

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

        self.db = Database(self._root, config.site)
        if self.db.site.exists():
            print(f"[swiftmap-server] resuming growth of site '{self.db.site.name}' "
                  f"({len(self.db.maps())} map(s) stored)")

    def run(self):
        """Start collecting and block, growing the map whenever the cap fills."""
        print(f"[swiftmap-server] starting on {self.cfg.host}:{self.cfg.port}")
        print(f"[swiftmap-server] site={self.cfg.site} backbone={self.cfg.backbone} "
              f"segmenter={self.cfg.segmenter} cap={self.cfg.max_keyframes} "
              f"merge_voxel={self.cfg.merge_voxel} m")
        print(f"[swiftmap-server] maps -> {self.db.maps_dir}")
        print(f"[swiftmap-server] site -> {self.db.site.path}")

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
        """Poll the retained keyframe count; grow the map when it hits the cap."""
        while not self._stop.is_set():
            count = self.session.get_keyframe_count()
            if not self._processing and count >= self.cfg.max_keyframes:
                self._run_pipeline()
            time.sleep(self.cfg.poll_interval)

    _MIN_KEYFRAMES = 1

    def _run_pipeline(self):
        with self._lock:
            self._run_pipeline_locked()

    def map_now(self) -> dict:
        """Force-map the current (partial) batch immediately and merge it in.

        Backs the viewer's "Map now" button. Serialized against the auto-loop.
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
            return {"success": True, "map_tag": lr.get("map_tag"),
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
        staging = os.path.join(self._root, self._STAGING)
        shutil.rmtree(staging, ignore_errors=True)
        print(f"\n[swiftmap-server] === run #{run}: {n} keyframes -> batch ===")
        try:
            params = {
                "output_name": self._STAGING,
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
            target_dir = result.get("scene_results", {}).get("target_directory") or staging

            if not self.session.has_stream_gps():
                print("[swiftmap-server] grow mode requires GPS; no streamed GPS -> batch dropped")
                return
            cfg = self.session.align_gps(use_icp=False)
            if "error" in cfg:
                print(f"[swiftmap-server] GPS align failed ({cfg['error']}) -> batch dropped")
                return
            print(f"[swiftmap-server] GPS aligned (RMSE {cfg.get('rmse', float('nan')):.2f} m)")

            plan = self.session.plan(low_percentile=constants.NFN_LOW_PERCENTILE,
                                     high_percentile=constants.NFN_HIGH_PERCENTILE)
            print(f"[swiftmap-server] NFN: "
                  f"{plan.get('error') or str(plan.get('num_viewpoints', 0)) + ' viewpoints'}")
            self.session.export_camera_poses()
            self.session.export_nfn_plan()
            self._send_nfn_kml(target_dir)

            # Store this batch as a map under maps/, then grow the site with it.
            grew = self.db.site.exists()
            stored = self.db.store(target_dir, created)
            self.db.grow(stored, conf_thres=self.cfg.conf_threshold,
                         voxel_size=self.cfg.merge_voxel, created=created)
            renderer.write_previews(self.db.site)
            n_points = int(self.db.site.metadata.get("num_points", 0))
            n_cameras = len(self.db.site.frames)

            self.latest_run = {
                "run": run, "map_tag": self.db.site.tag, "target_dir": self.db.site.path,
                "batch_tag": stored.tag,
                "num_keyframes": n_cameras,
                "num_points": n_points,
                "num_viewpoints": plan.get("num_viewpoints", 0) if isinstance(plan, dict) else 0,
                "gps_aligned": True, "elapsed": time.time() - t0,
                "grew": grew,
            }
            print(f"[swiftmap-server] run #{run}: {'grew' if grew else 'created'} site "
                  f"from '{stored.tag}' -> {n_points:,} pts, {n_cameras} cameras "
                  f"in {time.time() - t0:.1f}s")

        except Exception as e:
            import traceback
            print(f"[swiftmap-server] run #{run} pipeline error: {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self.session.clear_keyframes()
            self._processing = False

    def _send_nfn_kml(self, target_dir):
        """Queue the freshly planned NFN area polygon KML back to the connected engine.

        Delivered on the engine's next per-frame reply (see the reply protocol), which
        relays it to the Gabriel client.
        """
        path = os.path.join(target_dir, "next_flight_area.kml")
        if not os.path.isfile(path):
            print("[swiftmap-server] no NFN area KML to send back")
            return
        with open(path, "rb") as f:
            self.session.send_to_client(f.read())
        print("[swiftmap-server] queued NFN area KML for the next engine reply")

    def clean_keyframes(self) -> dict:
        """Discard the frames collected so far (the un-mapped queue). Backs the viewer's
        "Clear frames" button. Refused while a mapping run is in progress."""
        with self._lock:
            if self._processing:
                return {"error": "A mapping run is in progress; try again shortly."}
            n = self.session.get_keyframe_count()
            self.session.clear_keyframes()
        print(f"[swiftmap-server] cleared {n} collected keyframe(s)")
        return {"success": True, "cleared": int(n)}

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
        """Snapshot of the growing map for the viewer, plus live status."""
        state = dict(self.latest_run)
        state["processing"] = self._processing
        state["keyframes"] = self.session.get_keyframe_count()
        state["cap"] = self.cfg.max_keyframes
        state["current_map"] = self.db.site.tag if self.db.site.exists() else None
        state["num_maps"] = len(self.db.maps())
        return state

    def list_map_tags(self) -> List[str]:
        """Selectable tags: the site first, then every stored map (newest first)."""
        return self.db.tags()

    def current_map_tag(self) -> Optional[str]:
        """The map being grown -- always the site."""
        return self.db.site.tag if self.db.site.exists() else None

    def latest_map_tag(self) -> Optional[str]:
        tags = self.list_map_tags()
        return tags[0] if tags else None

    def latest_token(self) -> Optional[str]:
        """Change token for the viewer: bumps whenever the site grows."""
        tag = self.latest_map_tag()
        return None if tag is None else f"{tag}#{self.latest_run.get('run', 0)}"

    def render_map(self, map_tag: str, conf_level: float) -> dict:
        """Reconstruction + confidence-at-``conf_level`` GLBs for a stored map."""
        with self._lock:
            m = self.db.get(map_tag)
            return renderer.render(m, conf_level) if m else {"error": f"Unknown map '{map_tag}'."}

    def segment_map(self, map_tag: str, query: str, conf_level: float = None) -> dict:
        """Segment ``query`` on a stored ``map_tag`` at ``conf_level`` (request-driven).

        A merged map returns its inherited segmentation (no per-frame images). The
        returned ``glb_path`` is made absolute so the viewer can serve it.
        """
        conf = self.cfg.conf_threshold if conf_level is None else float(conf_level)
        with self._lock:
            m = self.db.get(map_tag)
            res = lift.segment(m, query, self.session.segmenter, conf) if m \
                else {"error": f"Unknown map '{map_tag}'."}
        glb = res.get("glb_path")
        if glb and not os.path.isabs(glb):
            res["glb_path"] = os.path.abspath(glb)
        return res
