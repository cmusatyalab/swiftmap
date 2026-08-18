# Copyright (C) 2024 Carnegie Mellon University

"""The mapping session: the one gateway between the server and the core.

It owns the transport (TCP ingest + keyframe selection), the reconstruction backbone,
the GPS aligner, the NFN planner, and the map database. A run goes:

    drone --TCP--> MappingTCPServer selects, batches      collect (socket thread)
    worker thread: next_batch() -> reconstruct() -> align_gps() -> plan()   map
    segment_map()                                          serve

The TCP server owns selection and batching; the session just owns the socket thread
that runs it and a worker thread that blocks on ``next_batch()`` and maps each one.

The session is long-lived, so the (expensive) model survives capture start/stop cycles;
``start(port, ...)`` controls only the transport.
"""

import os
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from swiftmap.core import constants
from swiftmap.core.database import Database
from swiftmap.core.transport import protocol
from swiftmap.core.transport.keyframe_selector import KeyframeSelector
from swiftmap.core.transport.tcp_server import MappingTCPServer
from swiftmap.core.pipeline.reconstructor import get_reconstructor
from swiftmap.core.pipeline.segmentor import get_segmenter, lift
from swiftmap.core.pipeline.next_flight_planner import NextFlightPlanner
from swiftmap.core.pipeline.gps_transformer import GpsTransformer


class MappingSession:
    """Owns and coordinates the full mapping pipeline for one run."""

    def __init__(self,
                 host: str = "0.0.0.0",
                 min_disparity: float = constants.DEFAULT_MIN_DISPARITY,
                 root: str = None,
                 site: str = "map",
                 backbone: [str] = [constants.DEFAULT_RECONSTRUCTOR, constants.DEFAULT_SEGMENTER]):
        self.host = host

        # db
        self.db = Database(root or os.getcwd(), site)

        # pipeline stages
        self.backbone: Optional[str] = None
        self.reconstructor = get_reconstructor(backbone[0])
        self.segmenter = get_segmenter(backbone[1])
        self.planner = NextFlightPlanner()
        self.aligner = GpsTransformer()

        # transport
        self.batch_size = constants.DEFAULT_MAX_KEYFRAMES  # a batch closes at this size
        self.selector = KeyframeSelector(min_disparity=min_disparity)
        self.tcp_server: Optional[MappingTCPServer] = None
        self.port: Optional[int] = None
        self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_session_")  # scratch dir for keyframe JPEGs

        # run state
        self.is_running = False
        self.transport_thread = None
        self.pipeline_thread = None
        self.start_time = None

    # =============================================================== lifecycle
    def start(self,
              port: int = protocol.TCP_PORT,
              min_disparity: Optional[float] = None,
              keep_all: Optional[bool] = None) -> bool:
        """Start the TCP transport and begin collecting keyframes."""
        if self.is_running:
            print("Mapping session already running")
            return True

        if min_disparity is not None:
            self.selector.configure_disparity_threshold(min_disparity)
        if keep_all is not None:
            self.selector.keep_all = keep_all

        self.port = port
        self.tcp_server = MappingTCPServer(self.selector, self.batch_size,
                                           host=self.host, port=port,
                                           temp_dir=self.temp_dir)

        print("Starting SwiftMap Mapping Session...")
        print(f"Server: {self.host}:{port} | min disparity: {self.selector.min_disparity} px, batch size: {self.batch_size}, keep_all: {self.selector.keep_all}")

        self.start_time = datetime.now()
        self.transport_thread = threading.Thread(target=self.tcp_server.start_server, daemon=True)
        self.transport_thread.start()

        self.pipeline_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.pipeline_thread.start()

        self.is_running = True
        print("Mapping session started; ready to receive drone images")
        return True

    def stop(self):
        """Stop the TCP transport and worker (the session and its model stay alive)."""
        if not self.is_running:
            return
        print("Stopping SwiftMap Mapping Session...")
        self.is_running = False
        if self.tcp_server:
            self.tcp_server.stop_server()
        if self.transport_thread:
            self.transport_thread.join(timeout=2.0)
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5.0)
        print("Mapping session stopped")

    def _worker_loop(self):
        """Block on the next full batch and map it, forever (until stop() unblocks it)."""
        while True:
            batch = self.tcp_server.next_batch()
            if batch is None:
                break
            try:
                result = self.reconstruct(batch, {})
                if not result.get("success"):
                    print(f"Reconstruction failed: {result.get('error')}")
                    continue

                cfg = self.align_gps()
                if "error" in cfg:
                    print(f"GPS align failed: {cfg['error']}")

                plan = self.plan()
                if "error" in plan:
                    print(f"NFN failed: {plan['error']}")
            finally:
                self.tcp_server.release_batch(batch)

    # ============================================================ helper
    def send_to_client(self, payload: bytes):
        """Deliver payload back to the connected client"""
        if self.tcp_server:
            self.tcp_server.queue_outbound(payload)

    # =============================================================== pipeline stage

    # ---------------------------------------------------------------- reconstruction
    def reconstruct(self, batch: List[Dict[str, Any]],
                    params: Dict[str, Any]) -> Dict[str, Any]:
        """Run reconstruction on one closed batch (from ``tcp_server.next_batch()``)."""
        paths = [e["path"] for e in batch]
        if not paths:
            return {"success": False, "error": "No keyframes collected yet",
                    "keyframe_count": 0}
        print(f"Reconstructing {len(paths)} keyframes (batch_size={self.batch_size})")
        return self.reconstructor.run(paths, params)
    # ---------------------------------------------------------------- planning

    # ------------------------------------------------------------ GPS alignment

    # --------------------------------------------------------- segmentation


    # ======================================================================= stats
    def get_stats(self) -> Dict[str, Any]:
        """Combined session statistics (selection + transport)."""
        stats = self.selector.get_stats()
        if self.tcp_server:
            stats["tcp_server_stats"] = self.tcp_server.get_stats()
            stats["keyframes_selected"] = self.tcp_server.total_keyframes_selected
        if self.start_time:
            stats["session_duration"] = str(datetime.now() - self.start_time)
        return stats
