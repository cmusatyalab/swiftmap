# Copyright (C) 2024 Carnegie Mellon University

"""
Mapping Session

Orchestrates a live mapping run. The session owns its two collaborators -- the
TCP transport (:class:`MappingTCPServer`) and the pure
:class:`KeyframeSelector` -- and wires received frames through selection:

    drone --TCP--> MappingTCPServer --frame--> MappingSession._on_frame
                                                    -> KeyframeSelector.is_keyframe

The frontend drives the session (start / stop / clear) and reads collected
keyframes from it; it never touches sockets or the selector directly. This keeps
networking out of both the keyframe selector (pure domain logic) and the UI.
"""

import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List

import cv2

from swiftmap.core.keyframe_selector import KeyframeSelector
from swiftmap.core.tcp_server import MappingTCPServer


class MappingSession:
    """Owns the TCP transport + keyframe selector and wires them together."""

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 43322,
                 min_disparity: float = 40.0,
                 visualize_flow: bool = False):
        self.host = host
        self.port = port

        self.selector = KeyframeSelector(min_disparity=min_disparity,
                                         visualize_flow=visualize_flow)
        self.tcp_server = MappingTCPServer(host=host, port=port,
                                           keyframe_callback=self._on_frame)

        # Collected keyframe file paths, accumulated as the server's queue drains
        # (the server's getter is destructive, so we keep our own running list).
        self._keyframe_paths: List[str] = []
        self._paths_lock = threading.Lock()
        self._keyframe_count = 0

        self.is_running = False
        self.server_thread = None
        self.start_time = None

        self.keyframe_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    # ------------------------------------------------------------------ wiring
    def _on_frame(self, image, metadata: Dict[str, Any]) -> bool:
        """TCP-server callback: run selection, notify listeners on a keyframe."""
        is_keyframe = self.selector.is_keyframe(image)
        if is_keyframe:
            self._keyframe_count += 1
            self._notify({
                "image": image,
                "metadata": metadata,
                "keyframe_count": self._keyframe_count,
                "frame_tracker_stats": self.selector.frame_tracker.get_stats(),
            })
        return is_keyframe

    def add_keyframe_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback invoked whenever a keyframe is selected."""
        self.keyframe_callbacks.append(callback)

    def _notify(self, info: Dict[str, Any]):
        for callback in self.keyframe_callbacks:
            try:
                callback(info)
            except Exception as e:
                print(f"Error in keyframe callback: {e}")

    # --------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Initialize the TCP server and start receiving frames."""
        if self.is_running:
            print("Mapping session already running")
            return True

        print("Starting SwiftMap Mapping Session...")
        print(f"Server: {self.host}:{self.port}")
        print(f"Min disparity threshold: {self.selector.min_disparity} pixels")

        if not self.tcp_server.initialize():
            print("Failed to initialize TCP server")
            return False

        self.selector.reset()
        self._keyframe_count = 0
        with self._paths_lock:
            self._keyframe_paths.clear()
        self.start_time = datetime.now()

        self.server_thread = threading.Thread(target=self.tcp_server.start_server,
                                              daemon=True)
        self.server_thread.start()

        self.is_running = True
        print("Mapping session started; ready to receive drone images")
        return True

    def stop(self):
        """Stop the TCP server and the session."""
        if not self.is_running:
            return
        print("Stopping SwiftMap Mapping Session...")
        self.is_running = False
        self.tcp_server.stop_server()
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        cv2.destroyAllWindows()  # close optical-flow visualization windows, if any
        print("Mapping session stopped")

    # ----------------------------------------------------------- keyframe access
    def _drain(self):
        """Pull newly-collected keyframes from the server queue into our list.

        ``tcp_server.get_collected_keyframes()`` drains its internal queue, so we
        accumulate the paths here to keep them across repeated reads."""
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
        self.tcp_server.clear_keyframes()
        self.selector.reset()
        with self._paths_lock:
            self._keyframe_paths.clear()
        self._keyframe_count = 0

    def configure_disparity_threshold(self, min_disparity: float):
        self.selector.configure_disparity_threshold(min_disparity)

    # -------------------------------------------------------------------- stats
    def get_stats(self) -> Dict[str, Any]:
        """Combined session statistics (selection + transport)."""
        stats = self.selector.get_stats()
        stats["tcp_server_stats"] = self.tcp_server.get_stats()
        stats["keyframes_selected"] = self._keyframe_count
        if self.start_time:
            stats["session_duration"] = str(datetime.now() - self.start_time)
        return stats
