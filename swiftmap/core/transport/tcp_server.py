# Copyright (C) 2024 Carnegie Mellon University

"""
TCP Server for SwiftMap Mapping System

Receives frame+GPS pairs from a drone client over TCP. Each frame is checked against
``keyframe_selector``; keyframes are saved to disk and appended to the open batch. Once
the batch reaches ``batch_size`` it is queued, ready for ``next_batch()`` to hand to a
reconstruction worker. The wire format lives in ``swiftmap.core.transport.protocol``.
"""

import os
import socket
import struct
import threading
import time
import queue
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

from swiftmap.core.transport import protocol


class MappingTCPServer:
    """Receives drone frames over TCP, selects keyframes, and batches them for reconstruction."""

    def __init__(self, keyframe_selector, batch_size: int,
                 host: str = "0.0.0.0", port: int = protocol.TCP_PORT,
                 temp_dir: Optional[str] = None):
        """
        Args:
            keyframe_selector: decides ``is_keyframe(image)`` for each received frame.
            batch_size: keyframes per batch; a batch is queued the moment it fills.
            host, port: where to listen.
            temp_dir: directory for keyframe JPEGs. If given (session-owned), it is
                      not deleted on stop. If None, the server creates and owns one.
        """
        self.host = host
        self.port = port
        self.keyframe_selector = keyframe_selector
        self.batch_size = batch_size

        # Server state
        self.server_socket = None
        self.is_running = False
        self.client_threads = []

        # Keyframe counters
        self.total_frames_received = 0
        self.total_keyframes_selected = 0
        self._outbound_lock = threading.Lock()
        self._pending_payload = b""

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
        self._batch: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._batches: "queue.Queue" = queue.Queue()

        # A session-owned temp_dir survives a stop; a self-created one is cleaned up.
        if temp_dir:
            self.temp_dir = temp_dir
            self._owns_temp_dir = False
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_")
            self._owns_temp_dir = True

    # ===================================================================== lifecycle
    def start_server(self):
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
            self.stop_server()


    def stop_server(self):
        """Stop the TCP server, unblock a worker waiting on next_batch(), clean up."""
        print("Stopping SwiftMap Mapping TCP Server...")
        self.is_running = False

        if self.server_socket:
            self.server_socket.close()

        for thread in self.client_threads:
            thread.join(timeout=1.0)

        self._batches.put(None)  # unblock next_batch()

        # Only remove a temp dir we created; a session-owned one outlives us.
        if self._owns_temp_dir and hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean temp directory: {e}")

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
        """
        Receive image data from client socket.
        Compatible with the bundled test client protocol.

        Returns:
            tuple: (image_array, image_metadata) or None if failed
        """
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

            # Decode image from bytes.
            nparr = np.frombuffer(received_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                print("Failed to decode image from bytes")
                return None

            # Receive the paired per-frame GPS (NaN triple -> None).
            gps_bytes = protocol.recv_exact(client_socket, protocol.GPS_NBYTES)
            if gps_bytes is None:
                print("Connection broken while receiving GPS")
                return None
            gps = protocol.unpack_gps(gps_bytes)

            metadata = {
                "timestamp": datetime.now(),
                "image_size": image_size,
                "image_shape": image.shape,
                "frame_id": self.total_frames_received + 1,
                "gps": gps,
            }

            print(f"Successfully received image: {image.shape}"
                  + (f" gps=({gps[0]:.6f},{gps[1]:.6f},{gps[2]:.1f})" if gps else " (no gps)"))
            return image, metadata

        except Exception as e:
            print(f"Error receiving image: {e}")
            return None

    def send_status_response(self, client_socket, status_code: str, message: str = "") -> bool:
        """
        Send status response back to client.

        Args:
            status_code: 'success', 'keyframe_selected', 'frame_skipped', or 'error'
        """
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
        """Stash a payload to piggyback on the next per-frame reply (delivered once)."""
        with self._outbound_lock:
            self._pending_payload = payload or b""

    def _take_outbound(self) -> bytes:
        """Pop the pending outbound payload (empty if none), clearing it."""
        with self._outbound_lock:
            payload, self._pending_payload = self._pending_payload, b""
        return payload

    # ===================================================== keyframe selection & batching
    def process_received_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Select or drop a frame; a selected frame is saved and joins the open batch."""
        try:
            self.total_frames_received += 1
            self.stats["total_frames"] = self.total_frames_received
            self.stats["last_frame_time"] = metadata["timestamp"]

            if not self.keyframe_selector.is_keyframe(image):
                print(f"Frame skipped: {self.total_frames_received} (not a keyframe)")
                return False

            self.total_keyframes_selected += 1
            self.stats["selected_keyframes"] = self.total_keyframes_selected
            path = self.save_keyframe(image, metadata)
            if path:
                self._add_to_batch(path, metadata.get("gps"))

            print(f"Keyframe selected: {self.total_keyframes_selected}/{self.total_frames_received}")
            return True

        except Exception as e:
            print(f"Error processing image: {e}")
            return False

    def save_keyframe(self, image: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Save a keyframe to the scratch dir; returns its path, or "" on failure."""
        try:
            timestamp_str = metadata["timestamp"].strftime("%Y%m%d_%H%M%S_%f")
            frame_id = metadata["frame_id"]
            filename = f"keyframe_{frame_id:06d}_{timestamp_str}.jpg"
            filepath = os.path.join(self.temp_dir, filename)

            cv2.imwrite(filepath, image)
            print(f"Keyframe saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"Error saving keyframe: {e}")
            return ""

    def _add_to_batch(self, path: str, gps):
        """Append to the open batch; queue it for a worker once it reaches batch_size."""
        with self._batch_lock:
            self._batch.append({"path": path, "gps": gps})
            if len(self._batch) < self.batch_size:
                return
            batch, self._batch = self._batch, []
        self._batches.put(batch)

    def next_batch(self):
        """Block until a batch is ready. Returns None once ``stop_server`` unblocks it."""
        return self._batches.get()

    def release_batch(self, batch: List[Dict[str, Any]]):
        """Delete a finished batch's JPEGs."""
        for e in batch:
            try:
                os.remove(e["path"])
            except OSError:
                pass

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


# Example usage and testing
if __name__ == "__main__":
    from swiftmap.core.transport.keyframe_selector import KeyframeSelector

    server = MappingTCPServer(keyframe_selector=KeyframeSelector(), batch_size=5)

    try:
        if server.initialize():
            print("Test server starting. Use test/test_client.py to test.")
            server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down test server...")
        server.stop_server()
