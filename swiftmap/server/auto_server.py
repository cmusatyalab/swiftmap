# Copyright (C) 2024 Carnegie Mellon University

"""Minimal headless auto-mapping server.

Owns the run: the TCP transport that collects and batches keyframes, the socket
thread that serves it, and the worker thread that feeds each closed Map through a
``MappingSession``. A run goes:

    drone --TCP--> Transporter selects, batches, creates a Map   (socket thread)
    worker thread: next_map_id() -> session.process(map_id)     (worker thread)

Site growth and segmentation are not wired up yet.
"""

import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from swiftmap import constants
from swiftmap.server.transport import protocol
from swiftmap.server.transport.transporter import Transporter
from swiftmap.core.session import MappingSession
from swiftmap.database import Database


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

            result = self.session.process(map_id)
            if "error" in result:
                print(f"[swiftmap-server] {result['error']}")
                continue

            map_ = self.db.get_map(map_id)
            map_.write2disk()
            if self.db.grow_site(map_):
                self.db.get_site().write2disk()

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
