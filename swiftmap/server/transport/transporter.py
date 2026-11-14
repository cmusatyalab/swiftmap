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


def log(message: str):
    """One timestamped line of the transport's log."""
    print(f"{datetime.now():%H:%M:%S} [transport] {message}")


def describe(metadata: Dict[str, Any]) -> str:
    """An image's size, encoded bytes, and GPS, for the log."""
    h, w = metadata["image_shape"][:2]
    gps = metadata["gps"]
    where = f"gps {gps[0]:.6f},{gps[1]:.6f},{gps[2]:.1f}" if gps else "no gps"
    return f"{w}x{h}, {metadata['image_size'] / 1e3:.0f} KB, {where}"


def image_extension(data: bytes) -> Optional[str]:
    """The file extension of the encoded image ``data``: .webp, .png or .jpg, or None
    if it is none of them."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    return None


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
        self._reachback_cutoff = False
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

            log(f"listening on {self.host}:{self.port}")

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
                        log(f"error accepting a client: {e}")
                    break

        except Exception as e:
            log(f"server error: {e}")
        finally:
            self.stop()


    def stop(self):
        """Stop the TCP server, unblock a worker waiting on next_map_id(), clean up. Once."""
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        log("stopping")
        self.is_running = False

        if self.server_socket:
            self.server_socket.close()

        for thread in self.client_threads:
            thread.join(timeout=1.0)

        self._batches.put(None)  # unblock next_map_id()

        log("stopped")
        self.print_final_stats()

    # ============================================================ connection handling
    def handle_client(self, client_socket, client_address):
        """Handle one client connection: receive frames, select, reply."""
        log(f"client {client_address[0]}:{client_address[1]} connected")

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

                is_keyframe = self.process_received_image(image, metadata)
                self.send_status_response(client_socket,
                                          "keyframe_selected" if is_keyframe else "frame_skipped")

        except Exception as e:
            log(f"client {client_address[0]}:{client_address[1]} error: {e}")
        finally:
            client_socket.close()
            self.stats["active_connections"] -= 1
            log(f"client {client_address[0]}:{client_address[1]} disconnected")

    def receive_image_from_client(self, client_socket) -> Optional[tuple]:
        """Receive image data from client socket."""
        try:
            # Receive the image-size header, then the JPEG body.
            size_data = protocol.recv_exact(client_socket, protocol.SIZE_NBYTES)
            if size_data is None:
                return None
            image_size = struct.unpack(protocol.SIZE_FORMAT, size_data)[0]

            received_data = protocol.recv_exact(client_socket, image_size)
            if received_data is None:
                log("connection broken while receiving an image")
                return None

            # Decode image from bytes; a reachback-done marker has none.
            image = None
            if image_size:
                image = cv2.imdecode(np.frombuffer(received_data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    log(f"failed to decode a {image_size}-byte image")
                    return None

            # Receive the paired per-frame GPS (NaN triple -> None), then the frame's kind.
            gps_bytes = protocol.recv_exact(client_socket, protocol.GPS_NBYTES)
            kind_bytes = protocol.recv_exact(client_socket, protocol.KIND_NBYTES)
            if gps_bytes is None or kind_bytes is None:
                log("connection broken while receiving GPS")
                return None
            gps = protocol.unpack_gps(gps_bytes)
            kind = struct.unpack(protocol.KIND_FORMAT, kind_bytes)[0]
            if image is None and kind not in (protocol.KIND_REACHBACK_DONE, protocol.KIND_POLL):
                log(f"dropped a kind-{kind} message without an image")
                return None

            metadata = {
                "timestamp": datetime.now(),
                "image_size": image_size,
                "encoded": received_data if image_size else None,
                "image_shape": None if image is None else image.shape,
                "frame_id": self.total_frames_received + 1,
                "gps": gps,
                "kind": kind,
            }
            return image, metadata

        except Exception as e:
            log(f"error receiving a message: {e}")
            return None

    def send_status_response(self, client_socket, status_code: str) -> bool:
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
            return True

        except Exception as e:
            log(f"error sending a reply: {e}")
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
        log(f"keyframe filter {'on' if enabled else 'off'}")

    def process_received_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Select or drop a frame; a selected frame is saved and joins the open batch."""
        try:
            self.total_frames_received += 1
            self.stats["total_frames"] = self.total_frames_received
            self.stats["last_frame_time"] = metadata["timestamp"]

            if self.holding:
                log(f"frame {self.total_frames_received} skipped: NFN round running")
                return False

            if self.keyframe_filter and not self.keyframe_selector.is_keyframe(image):
                log(f"frame {self.total_frames_received} skipped: not a keyframe")
                return False

            self.total_keyframes_selected += 1
            self.stats["selected_keyframes"] = self.total_keyframes_selected
            img_path = self._save_keyframe(image, metadata)
            gps = self._save_gps(metadata)
            log(f"keyframe {self.total_keyframes_selected} (batch {len(self._batch) + 1}/"
                f"{self.batch_size}): {describe(metadata)}")
            self._add_to_batch(img_path, gps)
            return True

        except Exception as e:
            log(f"error processing frame {self.total_frames_received}: {e}")
            return False

    def _save_keyframe(self, image: np.ndarray, metadata: Dict[str, Any],
                       prefix: str = "keyframe") -> str:
        """Save a keyframe to the scratch dir; returns its path, or "" on failure. The
        image file it came in as is kept byte for byte, so nothing is lost to encoding
        it again; an image that came some other way is saved as a JPEG."""
        try:
            timestamp_str = metadata["timestamp"].strftime("%Y%m%d_%H%M%S_%f")
            frame_id = metadata["frame_id"]
            encoded = metadata.get("encoded")
            ext = image_extension(encoded) if encoded else None
            image_name = f"{prefix}_{frame_id:06d}_{timestamp_str}{ext or '.jpg'}"
            image_path = os.path.join(self.temp_dir, image_name)

            if ext:
                with open(image_path, "wb") as f:
                    f.write(encoded)
            else:
                cv2.imwrite(image_path, image)
            return image_path

        except Exception as e:
            log(f"error saving a {prefix}: {e}")
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
        log(f"batch full: {len(batch)} keyframes as {map_id}"
            + ("; holding new frames for its NFN round" if self.holding else ""))

    def _close_batch(self, batch: List[Tuple[str, GPS]], map_: Optional[Map] = None) -> Map:
        map_ = map_ or self.db.create_map()
        images, all_gps = [], []
        
        for idx, (img_path, gps) in enumerate(batch):
            name = f"frame_{idx:06d}{os.path.splitext(img_path)[1] or '.jpg'}"
            try:
                os.replace(img_path, os.path.join(map_.images_dir, name))
            except OSError as err:
                log(f"error moving keyframe {img_path} -> {name}: {err}")
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
                n = len(self._reachback)
            log(f"reachback frame {n}: {describe(metadata)}")
        if metadata["kind"] == protocol.KIND_REACHBACK_DONE:
            self._reachback_done.set()
            log(f"reachback done: {len(self._reachback)} frame(s) in")

    def expect_reachback(self):
        """Forget any frames sent back so far, before a plan asks for more."""
        with self._batch_lock:
            self._reachback = []
        if not self._reachback_cutoff:
            self._reachback_done.clear()

    def finish_reachback(self):
        """Stop waiting for frames sent back: the current wait, and every later one,
        returns at once with whatever came."""
        self._reachback_cutoff = True
        self._reachback_done.set()
        log("reachback cut off; mapping with what came")

    def wait_reachback(self, timeout: float) -> List[Tuple[str, GPS]]:
        """The frames sent back for the last plan, once all are in or ``timeout`` passed."""
        if not self._reachback_done.wait(timeout):
            log(f"reachback not done after {timeout:.0f}s; using what came")
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
                    "latest": self._batch[-1][0] if self._batch else None,
                    "pending": self._batches.qsize()}

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
        log(f"cleared {len(batch)} collected frame(s)")
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
        log(f"flushed {len(batch)} frame(s) as {map_id}" + ("; holding new frames for its NFN round" if self.holding else ""))
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