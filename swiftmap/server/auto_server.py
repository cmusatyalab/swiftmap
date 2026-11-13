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
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    batch_size: int = constants.DEFAULT_MAX_KEYFRAMES
    min_disparity: float = constants.DEFAULT_MIN_DISPARITY
    output_dir: str = "output"


class AutoMappingServer:
    """Runs the TCP transport and drives each collected Map through the pipeline."""

    def __init__(self, config: ServerConfig):
        self.cfg = config
        os.makedirs(config.output_dir, exist_ok=True)

        self.db = Database(os.path.abspath(config.output_dir))
        self.session = MappingSession(self.db, backbone=[config.backbone, config.segmenter])

        # transport
        self.transporter: Optional[Transporter] = None
        self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_session_", dir=self.db.root)

        # run state
        self.is_running = False
        self.processing = False        # the worker is mid-pipeline on a batch
        self.keyframe_filter = False
        self.nfn_mode = True
        self.nfn_rounds = constants.NFN_ROUNDS
        self.reachback_count = constants.NFN_REACHBACK_COUNT
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
                                       temp_dir=self.temp_dir,
                                       keyframe_filter=self.keyframe_filter)
        self.transporter.nfn_mode = self.nfn_mode

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
                rounds = self.nfn_rounds if self.nfn_mode else 0
                planned = self._map(map_id, plan=rounds > 0)
                for done in range(1, rounds + 1):
                    frames = self._reach_back(map_id) if planned else []
                    if not frames:
                        break
                    print(f"[swiftmap-server] NFN round {done}/{rounds}: mapping {map_id} "
                          f"again with {len(frames)} frame(s) sent back")
                    map_id = self.transporter.rebuild_map(map_id, frames)
                    planned = self._map(map_id, plan=done < rounds)
                if self.db.grow_site(map_id):
                    self.db.get_site().write2disk()
            finally:
                self.transporter.release()
                self.processing = False

    def _map(self, map_id: str, plan: bool) -> bool:
        """Map one closed Map (its latest round) and plan its next flight if ``plan``;
        whether it was mapped and, if ``plan``, planned. The site takes the map once its
        last round is done."""
        result = self.session.process(map_id, plan=False)
        if "error" in result:
            print(f"[swiftmap-server] {result['error']}")
            return False
        map_ = self.db.get_map(map_id)
        planned = not plan or "error" not in self.session.plan(map_)
        map_.write2disk()
        return planned

    def _reach_back(self, map_id: str) -> List[Tuple[str, Any]]:
        """Send ``map_id``'s plan, and wait for the frames already taken sent back for it."""
        kml = self.db.get_map(map_id).get_flight_plan().to_kml(reachback=self.reachback_count)
        if not kml:
            return []
        self.transporter.expect_reachback()
        self.send_to_client(kml.encode())
        if not self.reachback_count:
            return []
        frames = self.transporter.wait_reachback(constants.NFN_REACHBACK_TIMEOUT_S)
        print(f"[swiftmap-server] {len(frames)} frame(s) sent back for {map_id}")
        return frames

    # ============================================================ viewer API
    def list_map_ids(self) -> List[str]:
        """Stored map ids, newest first."""
        return [m.meta.name for m in self.db.get_maps()]

    def map_rounds(self, map_id: str) -> List[Tuple[str, int]]:
        """(label, round) of each round of one map: the batch, then each NFN round."""
        return [("Batch" if k == 0 else f"Round {k}", k)
                for k in range(len(self.db.get_rounds(map_id)))]

    def map_round(self, map_id: str, round_: Optional[int] = None):
        """One round of one map, its latest if ``round_`` is None or not one of them."""
        rounds = self.db.get_rounds(map_id)
        if not rounds:
            return None
        return rounds[round_] if round_ is not None and 0 <= round_ < len(rounds) else rounds[-1]

    def map_scenes(self, map_id: str, round_: Optional[int] = None) -> Dict[str, Optional[str]]:
        """The local scene and confidence GLBs of one round of one map (its latest by
        default), as paths a viewer can load."""
        map_ = self.map_round(map_id, round_)
        if map_ is None:
            return {}
        return {name: self._existing(map_.local_dir, filename) for name, filename in
                (("scene", "scene.glb"), ("confidence", "scene_confidence.glb"))}

    def site_scenes(self) -> Dict[str, Optional[str]]:
        """The site's GLBs, as paths a viewer can load."""
        site = self.db.get_site()
        scenes = {view: site.stage_scene(len(site.maps), view)
                  for view in ("scene", "confidence", "merged")}
        scenes["seg"] = self._existing(site.path, "seg_site.glb")
        return scenes

    def site_stages(self) -> List[Tuple[str, int]]:
        """(label, stage) for each stage of the site on disk, labelled by its map's time."""
        site = self.db.get_site()
        return [(datetime.strptime(m.meta.name[4:19], "%Y%m%d_%H%M%S").strftime("%b %d · %H:%M:%S"), k)
                for k, m in enumerate(site.maps, 1) if site.stage_scene(k, "scene")]

    def site_stage(self, upto: int, view: str) -> Optional[str]:
        """One site view's GLB after the site's first ``upto`` maps."""
        return self.db.get_site().stage_scene(upto, view)

    def segment_site(self, query: str) -> Dict[str, Any]:
        """Lift one text query into the site; targets accumulate across queries.

        ``processing`` stays untouched: it means the worker is mapping a batch nobody
        clicked for, and this is click-driven, so the viewer's own scrim covers it.
        """
        result = self.session.segment(query)
        if "error" in result:
            return result
        self.db.get_site().write2disk()
        return result

    def clear_segmentations(self) -> Dict[str, Any]:
        """Drop every segmented target the site has accumulated."""
        site = self.db.get_site()
        cleared = site.clear_segmentations()
        site.write2disk()
        return {"success": True, "cleared": cleared}

    def batch_status(self) -> Dict[str, Any]:
        """How full the open batch is, plus whether the worker is busy with one."""
        status = (self.transporter.batch_status() if self.transporter else
                  {"collected": 0, "capacity": self.cfg.batch_size, "latest": None})
        status["processing"] = self.processing
        status["keyframe_filter"] = self.keyframe_filter
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

    def set_keyframe_filter(self, enabled: bool) -> Dict[str, Any]:
        """Select keyframes from the frames that follow, or keep every one of them."""
        self.keyframe_filter = bool(enabled)
        if self.transporter:
            self.transporter.set_keyframe_filter(self.keyframe_filter)
        return {"success": True, "enabled": self.keyframe_filter}

    def set_nfn_mode(self, enabled: bool) -> Dict[str, Any]:
        """Plan each batch, fetch frames already taken for its weak targets, and map it again
        with them, ignoring new frames meanwhile; or just map each batch."""
        self.nfn_mode = bool(enabled)
        if self.transporter:
            self.transporter.nfn_mode = self.nfn_mode
            if not self.nfn_mode:
                self.transporter.release()
        return {"success": True, "enabled": self.nfn_mode}

    def set_nfn_rounds(self, rounds) -> Dict[str, Any]:
        """How many NFN rounds each batch gets, each on the map the last one made."""
        self.nfn_rounds = max(1, int(rounds or 1))
        return {"success": True, "rounds": self.nfn_rounds}

    def set_reachback_count(self, count) -> Dict[str, Any]:
        """How many frames already taken each plan asks for."""
        self.reachback_count = max(0, int(count or 0))
        return {"success": True, "count": self.reachback_count}

    def del_map(self, map_id: str) -> Dict[str, Any]:
        """Delete a stored map, and drop what it contributed to the site."""
        if not self.db.del_map(map_id):
            return {"error": f"Unknown map '{map_id}'."}
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
