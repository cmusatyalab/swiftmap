# Copyright (C) 2024 Carnegie Mellon University

"""
TCP Server for SwiftMap Mapping System

Receives images (+ paired GPS) from drone clients via TCP, runs keyframe selection,
and replies with mapping status. The wire format lives in ``swiftmap.core.protocol``.
"""

import socket
import struct
import threading
import time
import queue
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import os

from swiftmap.core import protocol


class MappingTCPServer:
    """
    TCP server for receiving drone images and managing keyframe collection.
    
    The server receives images via TCP, performs keyframe selection,
    and stores selected keyframes for later VGGT mapping processing.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = protocol.TCP_PORT,
                 keyframe_callback: Optional[Callable] = None,
                 temp_dir: Optional[str] = None):
        """
        Initialize the mapping TCP server.

        Args:
            host: Server host address
            port: Server port (default from protocol.TCP_PORT)
            keyframe_callback: Callback function for keyframe selection
            temp_dir: Directory for keyframe JPEGs. If provided (session-owned), the
                      server uses it and does NOT delete it on stop -- so collected
                      keyframes survive a Stop and remain available for reconstruction.
                      If None, the server creates and owns its own temp dir.
        """
        self.host = host
        self.port = port
        self.keyframe_callback = keyframe_callback
        
        # Server state
        self.server_socket = None
        self.is_running = False
        self.client_threads = []
        
        # Keyframe collection
        self.collected_keyframes = queue.Queue()
        self.total_frames_received = 0
        self.total_keyframes_selected = 0
        
        # Statistics
        self.stats = {
            "server_start_time": None,
            "total_connections": 0,
            "active_connections": 0,
            "total_frames": 0,
            "selected_keyframes": 0,
            "last_frame_time": None
        }
        
        # Keyframe storage. A session-owned temp_dir is reused across start/stop so
        # keyframes survive a Stop; a self-created one is cleaned up on stop.
        if temp_dir:
            self.temp_dir = temp_dir
            self._owns_temp_dir = False
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_")
            self._owns_temp_dir = True
        
    def initialize(self) -> bool:
        """
        Initialize the TCP server.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            print(f"Initializing SwiftMap Mapping TCP Server on {self.host}:{self.port}")
            
            # Temp directory already created by tempfile.mkdtemp() in constructor
            
            print("SwiftMap Mapping TCP Server initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing TCP server: {e}")
            return False
    
    def receive_image_from_client(self, client_socket) -> Optional[tuple]:
        """
        Receive image data from client socket.
        Compatible with the bundled test client protocol.
        
        Args:
            client_socket: Connected client socket
            
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
            client_socket: Connected client socket
            status_code: Status code ('success', 'keyframe_selected', 'frame_skipped', 'error')
            message: Optional status message
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Reply as 3 float64 (see protocol): status, keyframe_count, total_frames.
            kf, total = self.total_keyframes_selected, self.total_frames_received
            if status_code == "keyframe_selected":
                response_data = protocol.pack_reply(protocol.STATUS_KEYFRAME, kf, total)
            elif status_code == "frame_skipped":
                response_data = protocol.pack_reply(protocol.STATUS_SKIPPED, kf, total)
            elif status_code == "error":
                response_data = protocol.pack_reply(protocol.STATUS_ERROR, -1.0, -1.0)
            else:  # success
                response_data = protocol.pack_reply(protocol.STATUS_SUCCESS, kf, total)

            client_socket.sendall(response_data)
            
            status_msg = f"{status_code}"
            if message:
                status_msg += f": {message}"
            print(f"Sent status: {status_msg}")
            
            return True
            
        except Exception as e:
            print(f"Error sending status response: {e}")
            return False
    
    def save_keyframe(self, image: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Save keyframe to temporary storage.
        
        Args:
            image: Image array to save
            metadata: Image metadata
            
        Returns:
            str: Path to saved keyframe file
        """
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
    
    def process_received_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Process received image and determine if it should be a keyframe.
        
        Args:
            image: Received image array
            metadata: Image metadata
            
        Returns:
            bool: True if image was selected as keyframe
        """
        try:
            self.total_frames_received += 1
            self.stats["total_frames"] = self.total_frames_received
            self.stats["last_frame_time"] = metadata["timestamp"]
            
            # Use keyframe callback if provided
            is_keyframe = False
            if self.keyframe_callback:
                is_keyframe = self.keyframe_callback(image, metadata)
            else:
                # Default: select every 5th frame as keyframe for testing
                is_keyframe = (self.total_frames_received % 5 == 1)
            
            if is_keyframe:
                self.total_keyframes_selected += 1
                self.stats["selected_keyframes"] = self.total_keyframes_selected
                
                # Save keyframe
                keyframe_path = self.save_keyframe(image, metadata)
                
                # Add to collection queue
                keyframe_data = {
                    "image": image,
                    "metadata": metadata,
                    "path": keyframe_path,
                    "keyframe_index": self.total_keyframes_selected
                }
                self.collected_keyframes.put(keyframe_data)
                
                print(f"Keyframe selected: {self.total_keyframes_selected}/{self.total_frames_received}")
                return True
            else:
                print(f"Frame skipped: {self.total_frames_received} (not a keyframe)")
                return False
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return False
    
    def handle_client(self, client_socket, client_address):
        """
        Handle individual client connection.
        
        Args:
            client_socket: Connected client socket
            client_address: Client address tuple
        """
        print(f"New mapping client connected: {client_address}")
        
        self.stats["total_connections"] += 1
        self.stats["active_connections"] += 1
        
        try:
            while self.is_running:
                # Receive image from client
                result = self.receive_image_from_client(client_socket)
                if result is None:
                    break
                
                image, metadata = result
                
                # Process image and check if keyframe
                start_time = time.time()
                is_keyframe = self.process_received_image(image, metadata)
                processing_time = time.time() - start_time
                
                # Send appropriate status response
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
    
    def start_server(self):
        """
        Start the TCP server and begin accepting client connections.
        """
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._bind_with_retry()
            self.server_socket.listen(5)
            # Time out accept() so the loop periodically checks is_running -- this lets
            # stop() release the port promptly instead of blocking in accept(), which
            # is what caused "Address already in use" on a quick restart.
            self.server_socket.settimeout(1.0)

            self.is_running = True
            self.stats["server_start_time"] = datetime.now()

            print(f"SwiftMap Mapping TCP Server started on {self.host}:{self.port}")
            print("Waiting for drone clients...")
            print("Compatible with the bundled test client (test/test_client.py)")

            while self.is_running:
                try:
                    client_socket, client_address = self.server_socket.accept()

                    # Handle client in separate thread
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

    def _bind_with_retry(self, attempts: int = 12, delay: float = 0.5):
        """Bind the listen socket, retrying briefly on EADDRINUSE.

        Even with SO_REUSEADDR, a just-stopped listener can hold the port for a
        moment; retrying makes a quick Stop -> Start reliable instead of crashing.
        """
        import errno
        last = None
        for _ in range(attempts):
            try:
                self.server_socket.bind((self.host, self.port))
                return
            except OSError as e:
                last = e
                if e.errno != errno.EADDRINUSE:
                    raise
                print(f"Port {self.port} busy (releasing); retrying bind in {delay}s...")
                time.sleep(delay)
        raise last
    
    def stop_server(self):
        """Stop the TCP server and clean up resources."""
        print("Stopping SwiftMap Mapping TCP Server...")
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        # Wait for client threads to finish
        for thread in self.client_threads:
            thread.join(timeout=1.0)

        # Only clean up a temp dir we created ourselves. A session-owned dir is left
        # intact so collected keyframes survive a Stop (the session manages its life).
        if self._owns_temp_dir and hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean temp directory: {e}")

        print("SwiftMap Mapping TCP Server stopped")
        self.print_final_stats()
    
    def get_collected_keyframes(self) -> list:
        """
        Get all collected keyframes.
        
        Returns:
            list: List of keyframe data dictionaries
        """
        keyframes = []
        while not self.collected_keyframes.empty():
            try:
                keyframes.append(self.collected_keyframes.get_nowait())
            except queue.Empty:
                break
        return keyframes
    
    def get_keyframe_count(self) -> int:
        """Get current number of collected keyframes."""
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
    
    def clear_keyframes(self):
        """Clear all collected keyframes and temp files."""
        try:
            # Clear queue
            while not self.collected_keyframes.empty():
                self.collected_keyframes.get_nowait()
            
            # Clear temp directory
            import glob
            temp_files = glob.glob(os.path.join(self.temp_dir, "*"))
            for file_path in temp_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing temp file {file_path}: {e}")
            
            # Reset counters
            self.total_frames_received = 0
            self.total_keyframes_selected = 0
            self.stats["total_frames"] = 0
            self.stats["selected_keyframes"] = 0
            
            print("Keyframes cleared successfully")
            
        except Exception as e:
            print(f"Error clearing keyframes: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Simple test server
    def test_keyframe_callback(image, metadata):
        # Simple test: select every 3rd frame as keyframe
        frame_id = metadata["frame_id"]
        return (frame_id % 3 == 1)
    
    server = MappingTCPServer(keyframe_callback=test_keyframe_callback)
    
    try:
        if server.initialize():
            print("Test server starting. Use test/test_client.py to test.")
            server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down test server...")
        server.stop_server()