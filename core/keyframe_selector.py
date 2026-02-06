# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Keyframe Selection Engine

Integrates TCP server with optical flow-based frame tracking for intelligent
keyframe selection. This engine receives drone images and uses motion analysis
to determine which frames are suitable for 3D mapping.

Key Features:
- Real-time optical flow analysis
- Configurable disparity thresholds  
- Automatic keyframe collection and storage
- Statistics tracking and monitoring
"""

import cv2
import numpy as np
import threading
import time
import queue
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from vggt_mapping.utils.frame_tracker import FrameTracker
from vggt_mapping.core.tcp_server import MappingTCPServer


class KeyframeSelector:
    """
    Keyframe selection engine that combines TCP image reception with
    optical flow-based motion analysis for intelligent frame selection.
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 43322,
                 min_disparity: float = 40.0,
                 visualize_flow: bool = False):
        """
        Initialize the keyframe selection engine.
        
        Args:
            host: TCP server host address
            port: TCP server port
            min_disparity: Minimum motion disparity for keyframe selection (default 40 as per TODO)
            visualize_flow: Enable optical flow visualization for debugging
        """
        self.host = host
        self.port = port
        self.min_disparity = min_disparity
        self.visualize_flow = visualize_flow
        
        # Initialize frame tracker for optical flow analysis
        self.frame_tracker = FrameTracker()
        
        # Initialize TCP server with keyframe callback
        self.tcp_server = MappingTCPServer(
            host=host,
            port=port,
            keyframe_callback=self._process_frame_for_keyframe_selection
        )
        
        # Keyframe collection and management
        self.selected_keyframes = []
        self.keyframe_metadata = []
        self.keyframe_lock = threading.Lock()
        
        # Statistics and monitoring
        self.stats = {
            "total_frames_processed": 0,
            "keyframes_selected": 0,
            "selection_start_time": None,
            "last_keyframe_time": None,
            "average_disparity": 0.0,
            "disparity_history": [],
            "processing_times": []
        }
        
        # State management
        self.is_running = False
        self.server_thread = None
        
        # Callbacks for external integration
        self.keyframe_callbacks = []
        
    def add_keyframe_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add callback function to be called when a new keyframe is selected.
        
        Args:
            callback: Function that takes keyframe data dictionary as input
        """
        self.keyframe_callbacks.append(callback)
    
    def _process_frame_for_keyframe_selection(self, image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Process received frame to determine if it should be selected as keyframe.
        This function is called by the TCP server for each received image.
        
        Args:
            image: Received image array (H, W, C) in BGR format
            metadata: Image metadata from TCP server
            
        Returns:
            bool: True if frame should be selected as keyframe
        """
        try:
            processing_start = time.time()
            
            # Use frame tracker to compute optical flow disparity
            is_keyframe = self.frame_tracker.compute_disparity(
                image, self.min_disparity, self.visualize_flow
            )
            
            processing_time = time.time() - processing_start
            
            # Update statistics
            self._update_stats(processing_time, is_keyframe)
            
            if is_keyframe:
                # Store keyframe data
                self._store_keyframe(image, metadata)
                
                # Notify external callbacks
                self._notify_keyframe_callbacks(image, metadata)
                
                print(f"Keyframe selected: Frame {metadata['frame_id']} "
                      f"(Processing: {processing_time*1000:.1f}ms)")
            else:
                print(f"Frame skipped: Frame {metadata['frame_id']} "
                      f"(Processing: {processing_time*1000:.1f}ms)")
            
            return is_keyframe
            
        except Exception as e:
            print(f"Error in keyframe selection: {e}")
            return False
    
    def _store_keyframe(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Store selected keyframe data.
        
        Args:
            image: Keyframe image array
            metadata: Keyframe metadata
        """
        try:
            with self.keyframe_lock:
                keyframe_data = {
                    "image": image.copy(),
                    "metadata": metadata.copy(),
                    "keyframe_index": len(self.selected_keyframes),
                    "selection_time": datetime.now(),
                    "frame_tracker_stats": self.frame_tracker.get_stats().copy()
                }
                
                self.selected_keyframes.append(keyframe_data)
                self.keyframe_metadata.append(metadata.copy())
                
                self.stats["last_keyframe_time"] = datetime.now()
                
        except Exception as e:
            print(f"Error storing keyframe: {e}")
    
    def _notify_keyframe_callbacks(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Notify registered callbacks about new keyframe selection.
        
        Args:
            image: Selected keyframe image
            metadata: Keyframe metadata
        """
        try:
            keyframe_info = {
                "image": image,
                "metadata": metadata,
                "keyframe_count": len(self.selected_keyframes),
                "frame_tracker_stats": self.frame_tracker.get_stats()
            }
            
            for callback in self.keyframe_callbacks:
                try:
                    callback(keyframe_info)
                except Exception as e:
                    print(f"Error in keyframe callback: {e}")
                    
        except Exception as e:
            print(f"Error notifying keyframe callbacks: {e}")
    
    def _update_stats(self, processing_time: float, is_keyframe: bool):
        """
        Update processing statistics.
        
        Args:
            processing_time: Time taken to process current frame
            is_keyframe: Whether current frame was selected as keyframe
        """
        try:
            self.stats["total_frames_processed"] += 1
            self.stats["processing_times"].append(processing_time)
            
            if is_keyframe:
                self.stats["keyframes_selected"] += 1
            
            # Keep processing times history manageable
            if len(self.stats["processing_times"]) > 1000:
                self.stats["processing_times"] = self.stats["processing_times"][-500:]
                
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def start(self) -> bool:
        """
        Start the keyframe selection engine.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_running:
                print("Keyframe selector already running")
                return True
            
            print("Starting VGGT Mapping Keyframe Selection Engine...")
            print(f"Server: {self.host}:{self.port}")
            print(f"Min disparity threshold: {self.min_disparity} pixels")
            print(f"Optical flow visualization: {'Enabled' if self.visualize_flow else 'Disabled'}")
            
            # Initialize TCP server
            if not self.tcp_server.initialize():
                print("Failed to initialize TCP server")
                return False
            
            # Reset statistics
            self.stats["selection_start_time"] = datetime.now()
            self.stats["total_frames_processed"] = 0
            self.stats["keyframes_selected"] = 0
            
            # Reset frame tracker
            self.frame_tracker.reset()
            
            # Start TCP server in separate thread
            self.server_thread = threading.Thread(
                target=self.tcp_server.start_server,
                daemon=True
            )
            self.server_thread.start()
            
            self.is_running = True
            print("Keyframe Selection Engine started successfully")
            print("Ready to receive drone images for keyframe selection")
            
            return True
            
        except Exception as e:
            print(f"Error starting keyframe selector: {e}")
            return False
    
    def stop(self):
        """Stop the keyframe selection engine."""
        try:
            if not self.is_running:
                print("Keyframe selector not running")
                return
            
            print("Stopping Keyframe Selection Engine...")
            
            self.is_running = False
            
            # Stop TCP server
            self.tcp_server.stop_server()
            
            # Wait for server thread
            if self.server_thread:
                self.server_thread.join(timeout=2.0)
            
            # Close visualization windows
            cv2.destroyAllWindows()
            
            print("Keyframe Selection Engine stopped")
            self.print_session_stats()
            
        except Exception as e:
            print(f"Error stopping keyframe selector: {e}")
    
    def get_selected_keyframes(self) -> List[Dict[str, Any]]:
        """
        Get all selected keyframes.
        
        Returns:
            List of keyframe data dictionaries
        """
        with self.keyframe_lock:
            return self.selected_keyframes.copy()
    
    def get_keyframe_count(self) -> int:
        """Get current number of selected keyframes."""
        with self.keyframe_lock:
            return len(self.selected_keyframes)
    
    def get_keyframe_paths(self) -> List[str]:
        """
        Get file paths of all selected keyframes.
        
        Returns:
            List of keyframe file paths
        """
        try:
            # Get keyframes from TCP server (which saves them to files)
            server_keyframes = self.tcp_server.get_collected_keyframes()
            paths = [kf["path"] for kf in server_keyframes if "path" in kf and os.path.exists(kf["path"])]
            return paths
        except Exception as e:
            print(f"Error getting keyframe paths: {e}")
            return []
    
    def clear_keyframes(self):
        """Clear all selected keyframes and reset state."""
        try:
            with self.keyframe_lock:
                self.selected_keyframes.clear()
                self.keyframe_metadata.clear()
            
            # Clear TCP server keyframes
            self.tcp_server.clear_keyframes()
            
            # Reset frame tracker
            self.frame_tracker.reset()
            
            # Reset statistics
            self.stats["keyframes_selected"] = 0
            self.stats["total_frames_processed"] = 0
            self.stats["last_keyframe_time"] = None
            self.stats["processing_times"] = []
            
            print("Keyframes cleared successfully")
            
        except Exception as e:
            print(f"Error clearing keyframes: {e}")
    
    def configure_disparity_threshold(self, min_disparity: float):
        """
        Update the minimum disparity threshold for keyframe selection.
        
        Args:
            min_disparity: New minimum disparity threshold in pixels
        """
        self.min_disparity = min_disparity
        print(f"Updated disparity threshold to: {min_disparity} pixels")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.stats.copy()
        stats["tcp_server_stats"] = self.tcp_server.get_stats()
        stats["frame_tracker_stats"] = self.frame_tracker.get_stats()
        
        # Calculate derived statistics
        if self.stats["processing_times"]:
            stats["average_processing_time"] = np.mean(self.stats["processing_times"])
            stats["max_processing_time"] = np.max(self.stats["processing_times"])
            stats["min_processing_time"] = np.min(self.stats["processing_times"])
        
        if self.stats["total_frames_processed"] > 0:
            stats["keyframe_selection_rate"] = (self.stats["keyframes_selected"] / 
                                               self.stats["total_frames_processed"]) * 100
        
        return stats
    
    def print_session_stats(self):
        """Print comprehensive session statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("KEYFRAME SELECTION ENGINE SESSION STATISTICS")
        print("="*60)
        
        print(f"Total frames processed: {stats['total_frames_processed']}")
        print(f"Keyframes selected: {stats['keyframes_selected']}")
        
        if stats.get("keyframe_selection_rate"):
            print(f"Selection rate: {stats['keyframe_selection_rate']:.1f}%")
        
        if stats.get("average_processing_time"):
            print(f"Average processing time: {stats['average_processing_time']*1000:.1f}ms")
            print(f"Processing time range: {stats['min_processing_time']*1000:.1f}ms - {stats['max_processing_time']*1000:.1f}ms")
        
        print(f"Disparity threshold: {self.min_disparity} pixels")
        print(f"Visualization: {'Enabled' if self.visualize_flow else 'Disabled'}")
        
        if stats["selection_start_time"]:
            runtime = datetime.now() - stats["selection_start_time"]
            print(f"Session duration: {runtime}")
        
        print("="*60)
    
    def run_interactive_session(self):
        """
        Run an interactive keyframe selection session.
        Useful for testing and monitoring.
        """
        try:
            if not self.start():
                return
            
            print("\nInteractive Keyframe Selection Session")
            print("Commands:")
            print("  's' - Show current statistics")
            print("  'c' - Clear collected keyframes")
            print("  't' - Toggle optical flow visualization")
            print("  'd <value>' - Set new disparity threshold")
            print("  'q' - Quit session")
            print("\nWaiting for drone images...")
            
            while self.is_running:
                try:
                    command = input().strip().lower()
                    
                    if command == 'q':
                        break
                    elif command == 's':
                        self.print_session_stats()
                    elif command == 'c':
                        self.clear_keyframes()
                        print("Keyframes cleared")
                    elif command == 't':
                        self.visualize_flow = not self.visualize_flow
                        print(f"Optical flow visualization: {'Enabled' if self.visualize_flow else 'Disabled'}")
                    elif command.startswith('d '):
                        try:
                            new_threshold = float(command.split()[1])
                            self.configure_disparity_threshold(new_threshold)
                        except (IndexError, ValueError):
                            print("Usage: d <threshold_value>")
                    else:
                        print("Unknown command. Use 'q' to quit.")
                        
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
            
        finally:
            self.stop()


# Example usage and testing
if __name__ == "__main__":
    # Test keyframe selector
    def keyframe_notification(keyframe_info):
        """Example callback for keyframe notifications."""
        print(f"[CALLBACK] New keyframe selected! Count: {keyframe_info['keyframe_count']}")
    
    # Create and configure keyframe selector
    selector = KeyframeSelector(
        min_disparity=40.0,
        visualize_flow=True  # Enable visualization for testing
    )
    
    # Add notification callback
    selector.add_keyframe_callback(keyframe_notification)
    
    print("VGGT Mapping Keyframe Selector Test")
    print("Use vggt_localization/utils/test_client.py to send test images")
    print("Press Ctrl+C to stop")
    
    try:
        # Run interactive session
        selector.run_interactive_session()
    except KeyboardInterrupt:
        print("\nTest session interrupted")
        selector.stop()