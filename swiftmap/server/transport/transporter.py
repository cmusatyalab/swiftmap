# Copyright (C) 2024 Carnegie Mellon University

"""
TCP Server for SwiftMap Mapping System

Receives frame+GPS pairs from a drone client over TCP. Each frame is checked against
``keyframe_selector`` (with the keyframe filter off, every frame passes); keyframes are
saved to disk and appended to the open batch. Once the batch reaches ``batch_size`` it
is queued, ready for handing to a reconstruction worker. 

The wire format lives in ``swiftmap.server.transport.protocol``.
"""

import os
import shutil
import socket
import csv
import struct
import threading
import time
import queue
from collections import deque
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Deque, List, Tuple

from swiftmap import constants
from swiftmap.server.transport import protocol
from swiftmap.server.transport.keyframe_selector import KeyframeSelector
from swiftmap.database.database import Database
from swiftmap.database.map import Map
from swiftmap.database.types import GPS


class Transporter:
    """Receives drone frames over TCP, selects keyframes, and batches them into a Map for reconstruction."""

    def __init__(self, batch_size: int, db,
                 min_disparity: float = constants.DEFAULT_MIN_DISPARITY,
                 host: str = "0.0.0.0", port: int = protocol.TCP_PORT,
                 temp_dir: Optional[str] = None, keyframe_filter: bool = True):
        
        self.host = host
        self.port = port
        self.keyframe_selector = KeyframeSelector(min_disparity=min_disparity)
        self.keyframe_filter = keyframe_filter      # off: keep every frame, unselected
        self.batch_size = batch_size
        self.db: Database  = db

        # Server state
        self.server_socket = None
        self.is_running = False
        self.client_threads = []

        # Keyframe counters
        self.total_frames_received = 0
        self.total_keyframes_selected = 0
        self._outbound_lock = threading.Lock()
        self._outbound: Deque[bytes] = deque()

        # Statistics
        self.stats = {
            "server_start_time": None,
            "total_connections": 0,
            "active_connections": 0,
            "total_frames": 0,
            "selected_keyframes": 0,
            "last_frame_time": None
        }

        # The open batch, and closed batches waiting for a worker.
        self._batch: List[Tuple[str, GPS]] = []
        self._batch_lock = threading.Lock()
        self._batches: "queue.Queue" = queue.Queue()

        # The frames sent back for the last NFN plan, and whether all are in. In NFN mode,
        # new frames are ignored from when a batch closes until its rounds are done.
        self._reachback: List[Tuple[str, GPS]] = []
        self._reachback_done = threading.Event()
        self.nfn_mode = False
        self.holding = False

        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.temp_gps_path = os.path.join(self.temp_dir, "gps.csv")
        

    # ===================================================================== lifecycle
    def start(self):
        """Start the TCP server and begin accepting client connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)

            self.is_running = True
            self.stats["server_start_time"] = datetime.now()

            print(f"SwiftMap Mapping TCP Server started on {self.host}:{self.port}")
            print("Waiting for drone clients...")
            print("Compatible with the bundled test client (test/test_client.py)")

            while self.is_running:
                try:
                    client_socket, client_address = self.server_socket.accept()

                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    self.client_threads.append(client_thread)

                except socket.timeout:
                    continue  # no client yet; re-check is_running
                except OSError as e:
                    if self.is_running:
                        print(f"Error accepting client connection: {e}")
                    break

        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.stop()


    def stop(self):
        """Stop the TCP server, unblock a worker waiting on next_map_id(), clean up."""
        print("Stopping SwiftMap Mapping TCP Server...")
        self.is_running = False

        if self.server_socket:
            self.server_socket.close()

        for thread in self.client_threads:
            thread.join(timeout=1.0)

        self._batches.put(None)  # unblock next_map_id()

        print("SwiftMap Mapping TCP Server stopped")
        self.print_final_stats()

    # ============================================================ connection handling
    def handle_client(self, client_socket, client_address):
        """Handle one client connection: receive frames, select, reply."""
        print(f"New mapping client connected: {client_address}")

        self.stats["total_connections"] += 1
        self.stats["active_connections"] += 1

        try:
            while self.is_running:
                result = self.receive_image_from_client(client_socket)
                if result is None:
                    break

                image, metadata = result
                if metadata["kind"] != protocol.KIND_FRAME:
                    if metadata["kind"] != protocol.KIND_POLL:
                        self._add_reachback(image, metadata)
                    self.send_status_response(client_socket, "success")
                    continue

                start_time = time.time()
                is_keyframe = self.process_received_image(image, metadata)
                processing_time = time.time() - start_time

                if is_keyframe:
                    self.send_status_response(client_socket, "keyframe_selected",
                                            f"Processing time: {processing_time*1000:.1f}ms")
                else:
                    self.send_status_response(client_socket, "frame_skipped",
                                            f"Processing time: {processing_time*1000:.1f}ms")

        except Exception as e:
            print(f"Client {client_address} error: {e}")
        finally:
            client_socket.close()
            self.stats["active_connections"] -= 1
            print(f"Mapping client {client_address} disconnected")

    def receive_image_from_client(self, client_socket) -> Optional[tuple]:
        """Receive image data from client socket."""
        try:
            # Receive the image-size header, then the JPEG body.
            size_data = protocol.recv_exact(client_socket, protocol.SIZE_NBYTES)
            if size_data is None:
                print("Client disconnected")
                return None
            image_size = struct.unpack(protocol.SIZE_FORMAT, size_data)[0]

            received_data = protocol.recv_exact(client_socket, image_size)
            if received_data is None:
                print("Connection broken while receiving image")
                return None

            # Decode image from bytes; a reachback-done marker has none.
            image = None
            if image_size:
                image = cv2.imdecode(np.frombuffer(received_data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print("Failed to decode image from bytes")
                    return None

            # Receive the paired per-frame GPS (NaN triple -> None), then the frame's kind.
            gps_bytes = protocol.recv_exact(client_socket, protocol.GPS_NBYTES)
            kind_bytes = protocol.recv_exact(client_socket, protocol.KIND_NBYTES)
            if gps_bytes is None or kind_bytes is None:
                print("Connection broken while receiving GPS")
                return None
            gps = protocol.unpack_gps(gps_bytes)
            kind = struct.unpack(protocol.KIND_FORMAT, kind_bytes)[0]
            if image is None and kind not in (protocol.KIND_REACHBACK_DONE, protocol.KIND_POLL):
                print("Received a frame without an image")
                return None

            metadata = {
                "timestamp": datetime.now(),
                "image_size": image_size,
                "image_shape": None if image is None else image.shape,
                "frame_id": self.total_frames_received + 1,
                "gps": gps,
                "kind": kind,
            }

            print(f"Successfully received image: {metadata['image_shape']} kind={kind}"
                  + (f" gps=({gps[0]:.6f},{gps[1]:.6f},{gps[2]:.1f})" if gps else " (no gps)"))
            return image, metadata

        except Exception as e:
            print(f"Error receiving image: {e}")
            return None

    def send_status_response(self, client_socket, status_code: str, message: str = "") -> bool:
        """Send status response back to client."""
        try:
            # 3 float64 status header plus any one-shot payload (e.g. the NFN area KML).
            kf, total = self.total_keyframes_selected, self.total_frames_received
            payload = self._take_outbound()
            if status_code == "keyframe_selected":
                response_data = protocol.pack_reply(protocol.STATUS_KEYFRAME, kf, total, payload)
            elif status_code == "frame_skipped":
                response_data = protocol.pack_reply(protocol.STATUS_SKIPPED, kf, total, payload)
            elif status_code == "error":
                response_data = protocol.pack_reply(protocol.STATUS_ERROR, -1.0, -1.0, payload)
            else:  # success
                response_data = protocol.pack_reply(protocol.STATUS_SUCCESS, kf, total, payload)

            client_socket.sendall(response_data)

            status_msg = f"{status_code}"
            if message:
                status_msg += f": {message}"
            print(f"Sent status: {status_msg}")

            return True

        except Exception as e:
            print(f"Error sending status response: {e}")
            return False

    def queue_outbound(self, payload: bytes):
        """Queue a payload to piggyback on a reply; one goes out per reply, in order."""
        if payload:
            with self._outbound_lock:
                self._outbound.append(payload)

    def _take_outbound(self) -> bytes:
        """Pop the next outbound payload (empty if none)."""
        with self._outbound_lock:
            return self._outbound.popleft() if self._outbound else b""

    def _hold(self):
        """In NFN mode, ignore new frames from the batch that just closed until its rounds
        are done, and tell the client to stop sending them."""
        self.holding = self.nfn_mode
        if self.holding:
            self.queue_outbound(protocol.NFN_HOLD)

    def release(self):
        """Take new frames again after a hold, and tell the client."""
        if self.holding:
            self.holding = False
            self.queue_outbound(protocol.NFN_RELEASE)

    # ===================================================== keyframe selection & batching
    def set_keyframe_filter(self, enabled: bool):
        """Select keyframes from the frames that follow, or keep every one of them."""
        if enabled and not self.keyframe_filter:
            # the tracker still holds the last keyframe from before the filter went off;
            # restart it, so the next frame becomes its fresh reference
            self.keyframe_selector.frame_tracker.reset()
        self.keyframe_filter = enabled
        print(f"[transport] keyframe filter {'on' if enabled else 'off'}")

    def process_received_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Select or drop a frame; a selected frame is saved and joins the open batch."""
        try:
            self.total_frames_received += 1
            self.stats["total_frames"] = self.total_frames_received
            self.stats["last_frame_time"] = metadata["timestamp"]

            if self.holding:
                print(f"Frame skipped: {self.total_frames_received} (NFN rounds running)")
                return False

            if self.keyframe_filter and not self.keyframe_selector.is_keyframe(image):
                print(f"Frame skipped: {self.total_frames_received} (not a keyframe)")
                return False

            self.total_keyframes_selected += 1
            self.stats["selected_keyframes"] = self.total_keyframes_selected
            img_path = self._save_keyframe(image, metadata)
            gps = self._save_gps(metadata)
            self._add_to_batch(img_path, gps)

            print(f"Keyframe selected: {self.total_keyframes_selected}/{self.total_frames_received}")
            return True

        except Exception as e:
            print(f"Error processing image: {e}")
            return False

    def _save_keyframe(self, image: np.ndarray, metadata: Dict[str, Any],
                       prefix: str = "keyframe") -> str:
        """Save a keyframe to the scratch dir; returns its path, or "" on failure."""
        try:
            timestamp_str = metadata["timestamp"].strftime("%Y%m%d_%H%M%S_%f")
            frame_id = metadata["frame_id"]
            image_name = f"{prefix}_{frame_id:06d}_{timestamp_str}.jpg"
            image_path = os.path.join(self.temp_dir, image_name)

            cv2.imwrite(image_path, image)
            print(f"Keyframe saved: {image_path}")
            return image_path

        except Exception as e:
            print(f"Error saving keyframe: {e}")
            return ""
        
    def _save_gps(self, metadata) -> Optional[GPS]:
        """Append one keyframe's GPS to the scratch dir's gps.csv, as the frame is saved."""
        gps = metadata.get("gps")
        with open(self.temp_gps_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "", ""] if gps is None else list(gps))
        return GPS(*gps) if gps is not None else None

    def _add_to_batch(self, img_path: str, gps: GPS):
        """Append to the open batch; once full, close it into a Map and queue its id."""
        with self._batch_lock:
            pack = (img_path, gps)
            self._batch.append(pack)
            if len(self._batch) < self.batch_size:
                return
            batch= self._batch
            self._batch = []
        self._hold()

        map_id = self._close_batch(batch).meta.name
        self._batches.put(map_id)

    def _close_batch(self, batch: List[Tuple[str, GPS]], map_: Optional[Map] = None) -> Map:
        map_ = map_ or self.db.create_map()
        images, all_gps = [], []
        
        for idx, (img_path, gps) in enumerate(batch):
            name = f"frame_{idx:06d}.jpg"
            try:
                os.replace(img_path, os.path.join(map_.images_dir, name))
            except OSError as err:
                print(f"Error moving keyframe {img_path} -> {name}: {err}")
                continue
            images.append(name)
            all_gps.append(gps)
        map_.update_input(images, all_gps)
        return map_

    # ===================================================================== reachback
    def _add_reachback(self, image: Optional[np.ndarray], metadata: Dict[str, Any]):
        """Keep a frame sent back for the last plan; the done marker says all are in."""
        if image is not None:
            path = self._save_keyframe(image, metadata, prefix="reachback")
            gps = metadata["gps"]
            with self._batch_lock:
                self._reachback.append((path, GPS(*gps) if gps is not None else None))
        if metadata["kind"] == protocol.KIND_REACHBACK_DONE:
            self._reachback_done.set()

    def expect_reachback(self):
        """Forget any frames sent back so far, before a plan asks for more."""
        with self._batch_lock:
            self._reachback = []
        self._reachback_done.clear()

    def wait_reachback(self, timeout: float) -> List[Tuple[str, GPS]]:
        """The frames sent back for the last plan, once all are in or ``timeout`` passed."""
        if not self._reachback_done.wait(timeout):
            print(f"[transport] reachback not done after {timeout:.0f}s; using what came")
        with self._batch_lock:
            frames, self._reachback = self._reachback, []
        return frames

    def rebuild_map(self, map_id: str, frames: List[Tuple[str, GPS]]) -> str:
        """The next round of ``map_id``: its latest round's images and the ``frames`` sent
        back, as its new latest round; its id."""
        batch = []
        old = self.db.get_map(map_id)
        for path, gps in zip(old.get_input_keyframe_paths(), old.get_input_gps()):
            copy = os.path.join(self.temp_dir, f"rebuild_{os.path.basename(path)}")
            shutil.copy2(path, copy)
            batch.append((copy, gps))
        return self._close_batch(batch + frames, self.db.create_round(map_id)).meta.name


    def next_map_id(self) -> Optional[str]:
        """Block until a Map is ready; its id. None once ``stop`` unblocks it."""
        return self._batches.get()

    # ==================================================================== open batch
    def batch_status(self) -> Dict[str, Any]:
        """How full the open batch is, and the newest keyframe in it."""
        with self._batch_lock:
            return {"collected": len(self._batch), "capacity": self.batch_size,
                    "latest": self._batch[-1][0] if self._batch else None}

    def clear_batch(self) -> int:
        """Throw away the frames collected so far; returns how many were dropped."""
        with self._batch_lock:
            batch, self._batch = self._batch, []
        for img_path, _ in batch:
            try:
                os.remove(img_path)
            except OSError:
                pass
        try:
            os.remove(self.temp_gps_path)     # the next batch starts its own csv
        except OSError:
            pass
        print(f"[transport] cleared {len(batch)} collected frame(s)")
        return len(batch)

    def flush_batch(self) -> Optional[str]:
        """Close the open batch early and queue it, however few frames it holds."""
        with self._batch_lock:
            if not self._batch:
                return None
            batch, self._batch = self._batch, []
        self._hold()

        map_id = self._close_batch(batch).meta.name
        self._batches.put(map_id)
        print(f"[transport] flushed {len(batch)} frame(s) as {map_id}")
        return map_id

    # ========================================================================= stats
    def get_keyframe_count(self) -> int:
        """Total keyframes selected so far."""
        return self.total_keyframes_selected

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.copy()

    def print_final_stats(self):
        """Print final server statistics."""
        print("\n" + "="*50)
        print("SWIFTMAP MAPPING TCP SERVER STATISTICS")
        print("="*50)
        print(f"Total connections: {self.stats['total_connections']}")
        print(f"Total frames received: {self.total_frames_received}")
        print(f"Keyframes selected: {self.total_keyframes_selected}")
        if self.total_frames_received > 0:
            selection_rate = (self.total_keyframes_selected / self.total_frames_received) * 100
            print(f"Keyframe selection rate: {selection_rate:.1f}%")

        if self.stats["server_start_time"]:
            runtime = datetime.now() - self.stats["server_start_time"]
            print(f"Server runtime: {runtime}")

        print("="*50)