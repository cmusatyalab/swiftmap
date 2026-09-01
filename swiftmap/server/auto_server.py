# Copyright (C) 2024 Carnegie Mellon University

"""Minimal headless auto-mapping server.

Owns the run: the TCP transport that collects and batches keyframes, the socket
thread that serves it, and the worker thread that feeds each closed Map through a
``MappingSession``. A run goes:

    drone --TCP--> Transporter selects, batches, creates a Map   (socket thread)
    worker thread: next_map_id() -> session.process(map_id)     (worker thread)

Site growth and segmentation are not wired up yet.
"""

import io
import os
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from swiftmap import constants
from swiftmap.server.transport import protocol
from swiftmap.server.transport.transporter import Transporter
from swiftmap.core.session import MappingSession
from swiftmap.database import Database


class _LogTail(io.TextIOBase):
    """Mirrors stdout into a bounded buffer, so the viewer can show recent progress."""

    def __init__(self, stream, lines: Deque[str]):
        self.stream, self.lines, self._partial = stream, lines, ""

    def write(self, text: str) -> int:
        self.stream.write(text)
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                self.lines.append(line.rstrip())
        return len(text)

    def flush(self):
        self.stream.flush()


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = protocol.TCP_PORT
    backbone: str = constants.DEFAULT_RECONSTRUCTOR
    segmenter: str = constants.DEFAULT_SEGMENTER
    site: str = "map"
    batch_size: int = constants.DEFAULT_MAX_KEYFRAMES
    min_disparity: float = constants.DEFAULT_MIN_DISPARITY
    output_dir: str = "output"


class AutoMappingServer:
    """Runs the TCP transport and drives each collected Map through the pipeline."""

    def __init__(self, config: ServerConfig):
        self.cfg = config
        os.makedirs(config.output_dir, exist_ok=True)

        self.db = Database(os.path.abspath(config.output_dir), config.site)
        self.session = MappingSession(self.db, backbone=[config.backbone, config.segmenter])

        # transport
        self.transporter: Optional[Transporter] = None
        self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_session_")  # scratch dir for keyframe JPEGs

        # run state
        self.is_running = False
        self.processing = False        # the worker is mid-pipeline on a batch
        self.log: Deque[str] = deque(maxlen=200)
        sys.stdout = _LogTail(sys.stdout, self.log)
        self.transport_thread = None
        self.pipeline_thread = None
        self.start_time = None

    # =============================================================== lifecycle
    def start(self) -> bool:
        """Start the TCP transport and the worker that maps each closed batch."""
        if self.is_running:
            print("Server already running")
            return True

        self.transporter = Transporter(self.cfg.batch_size, self.db,
                                       min_disparity=self.cfg.min_disparity,
                                       host=self.cfg.host, port=self.cfg.port,
                                       temp_dir=self.temp_dir)

        print(f"[swiftmap-server] starting on {self.cfg.host}:{self.cfg.port} "
              f"(backbone={self.cfg.backbone}, batch_size={self.cfg.batch_size}, "
              f"min disparity={self.cfg.min_disparity} px)")

        self.start_time = datetime.now()
        self.transport_thread = threading.Thread(target=self.transporter.start, daemon=True)
        self.transport_thread.start()

        self.pipeline_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.pipeline_thread.start()

        self.is_running = True
        print("[swiftmap-server] ready to receive drone images")
        return True

    def run(self):
        """Start the server and block until interrupted."""
        if not self.start():
            raise RuntimeError("Failed to start the TCP collection server")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[swiftmap-server] interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop the transport and worker (the session and its models stay alive)."""
        if not self.is_running:
            return
        print("[swiftmap-server] stopping...")
        self.is_running = False
        if self.transporter:
            self.transporter.stop()
        if self.transport_thread:
            self.transport_thread.join(timeout=2.0)
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5.0)
        print("[swiftmap-server] stopped")

    def _worker_loop(self):
        """Block on the next full Map and run the pipeline over it, until stop() unblocks it."""
        while True:
            map_id = self.transporter.next_map_id()
            if map_id is None:
                break

            self.processing = True
            try:
                result = self.session.process(map_id)
                if "error" in result:
                    print(f"[swiftmap-server] {result['error']}")
                    continue

                self.db.get_map(map_id).write2disk()
                if self.db.grow_site(map_id):
                    self.db.get_site().write2disk()
            finally:
                self.processing = False

    # ============================================================ viewer API
    def list_map_ids(self) -> List[str]:
        """Stored map ids, newest first."""
        return [m.meta.name for m in self.db.get_maps()]

    def map_scenes(self, map_id: str) -> Dict[str, Optional[str]]:
        """The local scene and confidence GLBs of one map, as paths a viewer can load."""
        map_ = self.db.get_map(map_id)
        if map_ is None:
            return {}
        return {name: self._existing(map_.local_dir, filename) for name, filename in
                (("scene", "scene.glb"), ("confidence", "scene_confidence.glb"))}

    def site_scenes(self) -> Dict[str, Optional[str]]:
        """The site's GLBs, as paths a viewer can load."""
        site = self.db.get_site()
        return {name: self._existing(site.path, filename) for name, filename in
                (("scene", "site.glb"), ("confidence", "site_confidence.glb"),
                 ("merged", "merged_res.glb"))}

    def batch_status(self) -> Dict[str, Any]:
        """How full the open batch is, plus whether the worker is busy with one."""
        status = (self.transporter.batch_status() if self.transporter else
                  {"collected": 0, "capacity": self.cfg.batch_size, "latest": None})
        status["processing"] = self.processing
        status["log"] = self.recent_log()
        return status

    def recent_log(self, lines: int = 10) -> List[str]:
        """The last few lines the pipeline printed, newest last."""
        return list(self.log)[-lines:]

    def clear_batch(self) -> Dict[str, Any]:
        """Throw away the frames collected so far."""
        if self.transporter is None:
            return {"error": "The transport is not running."}
        return {"success": True, "cleared": self.transporter.clear_batch()}

    def render_now(self) -> Dict[str, Any]:
        """Map whatever is in the open batch, without waiting for it to fill."""
        if self.transporter is None:
            return {"error": "The transport is not running."}
        map_id = self.transporter.flush_batch()
        if map_id is None:
            return {"error": "No frames collected yet."}
        return {"success": True, "map_id": map_id}

    def del_map(self, map_id: str) -> Dict[str, Any]:
        """Delete a stored map, and drop what it contributed to the site."""
        if not self.db.del_map(map_id):
            return {"error": f"Unknown map '{map_id}'."}
        self.db.get_site().write2disk()     # the site no longer includes it
        return {"success": True, "map_id": map_id}

    @staticmethod
    def _existing(directory: str, filename: str) -> Optional[str]:
        """The path if it is on disk, else None -- a viewer shows nothing for None."""
        path = os.path.join(directory, filename)
        return path if os.path.isfile(path) else None

    # ============================================================ helper
    def send_to_client(self, payload: bytes):
        """Deliver payload back to the connected client"""
        if self.transporter:
            self.transporter.queue_outbound(payload)

    def get_stats(self) -> Dict[str, Any]:
        """Combined run statistics (selection + transport)."""
        stats = {}
        if self.transporter:
            stats = self.transporter.keyframe_selector.get_stats()
            stats["transporter_stats"] = self.transporter.get_stats()
            stats["keyframes_selected"] = self.transporter.total_keyframes_selected
        if self.start_time:
            stats["session_duration"] = str(datetime.now() - self.start_time)
        return stats
