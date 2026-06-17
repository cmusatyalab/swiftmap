# Copyright (C) 2024 Carnegie Mellon University

"""
TCP Server for SwiftMap Mapping System

Receives images from drone clients via TCP connection and manages
keyframe selection for 3D mapping. Compatible with the bundled SwiftMap
test clients but returns mapping status instead of GPS coordinates.

Port: 43322 (as specified in TODO requirements)
Protocol: Binary image data reception compatible with existing test clients
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


class MappingTCPServer:
    """
    TCP server for receiving drone images and managing keyframe collection.
    
    The server receives images via TCP, performs keyframe selection,
    and stores selected keyframes for later VGGT mapping processing.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 43322, 
                 keyframe_callback: Optional[Callable] = None):
        """
        Initialize the mapping TCP server.
        
        Args:
            host: Server host address
            port: Server port (default 43322 as specified in TODO)
            keyframe_callback: Callback function for keyframe selection
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
        
        # Use system temporary directory for keyframes (will be cleaned up when server stops)
        import tempfile
        self.temp_dir = tempfile.mkdtemp(prefix="swiftmap_")
        
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
            # Receive image size (4 bytes, big-endian unsigned int)
            size_data = client_socket.recv(4)
            if not size_data:
                print("Client disconnected")
                return None
            
            image_size = struct.unpack('!I', size_data)[0]
            print(f"Receiving image of size: {image_size} bytes")
            
            # Receive image data
            received_data = b''
            while len(received_data) < image_size:
                chunk = client_socket.recv(min(4096, image_size - len(received_data)))
                if not chunk:
                    print("Connection broken while receiving image")
                    return None
                received_data += chunk
            
            if len(received_data) != image_size:
                print(f"Incomplete image received: {len(received_data)}/{image_size} bytes")
                return None
            
            # Decode image from bytes
            nparr = np.frombuffer(received_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Failed to decode image from bytes")
                return None
            
            # Create metadata
            timestamp = datetime.now()
            metadata = {
                "timestamp": timestamp,
                "image_size": image_size,
                "image_shape": image.shape,
                "frame_id": self.total_frames_received + 1
            }
            
            print(f"Successfully received image: {image.shape}")
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
            # Send status as double values for compatibility with localization client
            if status_code == "keyframe_selected":
                # Positive values indicate keyframe selected
                response_data = struct.pack('3d', 1.0, float(self.total_keyframes_selected), float(self.total_frames_received))
            elif status_code == "frame_skipped":  
                # Zero values indicate frame skipped
                response_data = struct.pack('3d', 0.0, float(self.total_keyframes_selected), float(self.total_frames_received))
            elif status_code == "error":
                # Negative values indicate error
                response_data = struct.pack('3d', -1.0, -1.0, -1.0)
            else:  # success
                response_data = struct.pack('3d', 2.0, float(self.total_keyframes_selected), float(self.total_frames_received))
            
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
            
            # Save image
            cv2.imwrite(filepath, image)
            
            # Save metadata
            metadata_path = filepath.replace('.jpg', '_metadata.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"Frame ID: {frame_id}\n")
                f.write(f"Timestamp: {metadata['timestamp']}\n")
                f.write(f"Image Size: {metadata['image_size']} bytes\n")
                f.write(f"Image Shape: {metadata['image_shape']}\n")
                f.write(f"Keyframe Index: {self.total_keyframes_selected}\n")
            
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
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.is_running = True
            self.stats["server_start_time"] = datetime.now()
            
            print(f"SwiftMap Mapping TCP Server started on {self.host}:{self.port}")
            print("Waiting for drone clients...")
            print("Compatible with the bundled test client (swiftmap/utils/test_client.py)")
            
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
                    
                except Exception as e:
                    if self.is_running:
                        print(f"Error accepting client connection: {e}")
                    break
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the TCP server and clean up resources."""
        print("Stopping SwiftMap Mapping TCP Server...")
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        # Wait for client threads to finish
        for thread in self.client_threads:
            thread.join(timeout=1.0)
        
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
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
            print("Test server starting. Use swiftmap/utils/test_client.py to test.")
            server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down test server...")
        server.stop_server()