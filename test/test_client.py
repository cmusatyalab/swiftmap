#!/usr/bin/env python3
# Copyright (C) 2024 Carnegie Mellon University

"""
Test Client for SwiftMap Mapping TCP Server

Sends test images to the VGGT mapping TCP server for keyframe selection testing.
Test client for streaming images to the SwiftMap TCP server.

Features:
- Sends images from directory to TCP server
- Receives keyframe selection status responses
- Displays real-time statistics
- Binary streaming protocol matching the SwiftMap TCP server

Usage:
    python test_client.py --image-dir test_images/
    python test_client.py --host localhost --port 43322
    python test_client.py --image-dir ../test_input --delay 1.0
"""

import os
import sys
import argparse
import socket
import time
import glob
from datetime import datetime
from typing import Optional, Tuple
import cv2

# Make the swiftmap package importable when run as a standalone script, so the
# wire protocol has a single source of truth (swiftmap.server.transport.protocol).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swiftmap.server.transport import protocol


class MappingTestClient:
    """
    Test client for VGGT mapping TCP server.
    
    Sends images to the mapping server and receives keyframe selection responses.
    """
    
    def __init__(self, host: str = "localhost", port: int = protocol.TCP_PORT):
        """
        Initialize test client.
        
        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        
        # Statistics
        self.stats = {
            "images_sent": 0,
            "keyframes_selected": 0,
            "frames_skipped": 0,
            "errors": 0,
            "start_time": None,
            "total_bytes_sent": 0
        }
        
    def connect(self) -> bool:
        """
        Connect to the mapping TCP server.
        
        Returns:
            bool: True if connection successful
        """
        try:
            print(f"📡 Connecting to SwiftMap Mapping server at {self.host}:{self.port}...")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout
            self.socket.connect((self.host, self.port))
            
            self.is_connected = True
            self.stats["start_time"] = datetime.now()
            
            print(f"✅ Connected successfully to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
        self.is_connected = False
        print("🔌 Disconnected from server")
    
    def send_image(self, image_path: str, gps=None) -> Optional[Tuple[bool, str]]:
        """
        Send image (+ paired GPS) to server and receive response.

        Args:
            image_path: Path to image file
            gps: optional (lat, lon, alt) paired with this frame; None -> sent as NaN.

        Returns:
            Tuple of (is_keyframe, status_message) or None if failed
        """
        if not self.is_connected:
            return None

        try:
            # Read and encode image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to read image: {image_path}")
                self.stats["errors"] += 1
                return None

            # Encode image as JPEG
            _, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = encoded_image.tobytes()

            # Send image size header, the JPEG body, then the paired GPS.
            self.socket.sendall(protocol.pack_size(len(image_bytes)))
            self.socket.sendall(image_bytes)
            self.socket.sendall(protocol.pack_gps(gps))

            self.stats["images_sent"] += 1
            self.stats["total_bytes_sent"] += (
                len(image_bytes) + protocol.SIZE_NBYTES + protocol.GPS_NBYTES)

            # Receive the status reply.
            reply = protocol.recv_reply(self.socket)
            if reply is None:
                print("❌ Invalid response from server")
                self.stats["errors"] += 1
                return None
            status_code, keyframe_count, total_frames, _payload = reply

            if status_code == protocol.STATUS_KEYFRAME:
                self.stats["keyframes_selected"] += 1
                return True, f"KEYFRAME selected ({int(keyframe_count)}/{int(total_frames)})"
            elif status_code == protocol.STATUS_SKIPPED:
                self.stats["frames_skipped"] += 1
                return False, f"Frame skipped ({int(keyframe_count)}/{int(total_frames)})"
            elif status_code == protocol.STATUS_ERROR:
                self.stats["errors"] += 1
                return False, "Server error"
            else:
                return False, f"Status: {status_code} ({int(keyframe_count)}/{int(total_frames)})"
                
        except socket.timeout:
            print("⏰ Socket timeout")
            self.stats["errors"] += 1
            return None
        except Exception as e:
            print(f"❌ Error sending image: {e}")
            self.stats["errors"] += 1
            return None
    
    def send_image_directory(self, image_dir: str, delay: float = 0.5,
                           max_images: Optional[int] = None, gps_csv: Optional[str] = None) -> bool:
        """
        Send all images from directory to server.
        
        Args:
            image_dir: Directory containing images
            delay: Delay between images in seconds
            max_images: Maximum number of images to send (None for all)
            
        Returns:
            bool: True if successful
        """
        # Find image files
        image_patterns = [
            os.path.join(image_dir, "*.jpg"),
            os.path.join(image_dir, "*.jpeg"),
            os.path.join(image_dir, "*.png"),
            os.path.join(image_dir, "*.bmp")
        ]
        
        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob.glob(pattern))
        
        image_paths = sorted(image_paths)
        
        if not image_paths:
            print(f"❌ No images found in {image_dir}")
            return False
        
        if max_images:
            image_paths = image_paths[:max_images]

        # Optional paired GPS, one row (lat,lon,alt) per image in sorted order.
        gps_rows = None
        if gps_csv:
            import csv
            gps_rows = []
            with open(gps_csv, newline="") as f:
                for r in csv.DictReader(f):
                    cols = {c.lower(): c for c in r}
                    gps_rows.append((float(r[cols["latitude"]]), float(r[cols["longitude"]]),
                                     float(r[cols.get("altitude", cols.get("height"))])))
            print(f"🛰️  Loaded {len(gps_rows)} GPS rows from {gps_csv}")
            if len(gps_rows) != len(image_paths):
                print(f"   ⚠️ gps rows ({len(gps_rows)}) != images ({len(image_paths)}); pairing by index")

        print(f"📁 Found {len(image_paths)} images in {image_dir}")
        print(f"⏱️  Sending with {delay}s delay between images")
        print(f"🎯 Target server: {self.host}:{self.port}")
        print("")

        try:
            for i, image_path in enumerate(image_paths):
                print(f"📷 [{i+1}/{len(image_paths)}] Sending: {os.path.basename(image_path)}")

                gps = gps_rows[i] if (gps_rows and i < len(gps_rows)) else None
                start_time = time.time()
                result = self.send_image(image_path, gps=gps)
                send_time = time.time() - start_time
                
                if result:
                    is_keyframe, status_msg = result
                    
                    # Color coding for terminal output
                    if is_keyframe:
                        print(f"   ✅ {status_msg} (Response: {send_time*1000:.0f}ms)")
                    else:
                        print(f"   ⏭️  {status_msg} (Response: {send_time*1000:.0f}ms)")
                else:
                    print(f"   ❌ Failed to send image")
                
                # Print statistics every 10 images
                if (i + 1) % 10 == 0:
                    self.print_statistics()
                    print("")
                
                # Delay before next image
                if delay > 0 and i < len(image_paths) - 1:
                    time.sleep(delay)
            
            print("\n🏁 All images sent successfully")
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Sending interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Error during batch sending: {e}")
            return False
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.stats
        
        print("\n📊 CLIENT STATISTICS")
        print("-" * 40)
        print(f"Images sent: {stats['images_sent']}")
        print(f"Keyframes selected: {stats['keyframes_selected']}")
        print(f"Frames skipped: {stats['frames_skipped']}")
        print(f"Errors: {stats['errors']}")
        
        if stats['images_sent'] > 0:
            selection_rate = (stats['keyframes_selected'] / stats['images_sent']) * 100
            print(f"Selection rate: {selection_rate:.1f}%")
        
        if stats['total_bytes_sent'] > 0:
            mb_sent = stats['total_bytes_sent'] / (1024 * 1024)
            print(f"Data sent: {mb_sent:.2f} MB")
        
        if stats['start_time']:
            duration = datetime.now() - stats['start_time']
            print(f"Duration: {duration}")
            
            if duration.total_seconds() > 0:
                fps = stats['images_sent'] / duration.total_seconds()
                print(f"Average FPS: {fps:.2f}")
        
        print("-" * 40)
    
    def run_interactive_test(self):
        """Run interactive test session."""
        print("🎮 INTERACTIVE TEST MODE")
        print("Commands:")
        print("  'send <image_path>' - Send single image")
        print("  'batch <directory>' - Send directory batch")
        print("  'stats' - Show statistics")
        print("  'quit' - Exit")
        print("")
        
        while True:
            try:
                command = input("mapping_client> ").strip()
                
                if command == "quit" or command == "q":
                    break
                elif command == "stats":
                    self.print_statistics()
                elif command.startswith("send "):
                    image_path = command[5:].strip()
                    if os.path.exists(image_path):
                        result = self.send_image(image_path)
                        if result:
                            is_keyframe, status_msg = result
                            print(f"Result: {status_msg}")
                        else:
                            print("Failed to send image")
                    else:
                        print(f"Image not found: {image_path}")
                elif command.startswith("batch "):
                    directory = command[6:].strip()
                    if os.path.exists(directory):
                        self.send_image_directory(directory)
                    else:
                        print(f"Directory not found: {directory}")
                else:
                    print("Unknown command. Use 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    def test_connection(self) -> bool:
        """Test connection to server."""
        print("🔍 Testing connection...")
        
        if not self.connect():
            return False
        
        # Try to send a simple test
        test_image = "test_connection.jpg"
        
        # Create a simple test image
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(test_image, test_img)
        
        try:
            result = self.send_image(test_image)
            os.remove(test_image)  # Clean up
            
            if result:
                print("✅ Connection test successful")
                return True
            else:
                print("❌ Connection test failed")
                return False
                
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            try:
                os.remove(test_image)
            except OSError:
                pass
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test client for SwiftMap Mapping TCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image-dir test_images/                    # Send all images from directory
  %(prog)s --host 192.168.1.100 --port 43322         # Custom server address
  %(prog)s --interactive                               # Interactive mode
  %(prog)s --test-connection                           # Test connection only
        """
    )
    
    # Server settings
    parser.add_argument("--host", type=str, default="localhost",
                       help="Server host address (default: localhost)")
    parser.add_argument("--port", type=int, default=43322,
                       help="Server port (default: 43322)")
    
    # Image settings
    parser.add_argument("--image-dir", type=str,
                       help="Directory containing test images")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between images in seconds (default: 0.1)")
    parser.add_argument("--max-images", type=int,
                       help="Maximum number of images to send")
    parser.add_argument("--gps-csv", type=str,
                       help="CSV (latitude,longitude,altitude) paired 1:1 with the sorted images")

    # Mode settings
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--test-connection", action="store_true",
                       help="Test connection and exit")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚁 SWIFTMAP MAPPING TEST CLIENT")
    print("="*60)
    print(f"Target server: {args.host}:{args.port}")
    print("Compatible with VGGT mapping keyframe selection server")
    print("")
    
    # Create client
    client = MappingTestClient(host=args.host, port=args.port)
    
    try:
        # Connect to server
        if not client.connect():
            return 1
        
        # Test connection mode
        if args.test_connection:
            success = client.test_connection()
            return 0 if success else 1
        
        # Interactive mode
        if args.interactive:
            client.run_interactive_test()
            return 0
        
        # Batch mode
        if args.image_dir:
            if not os.path.exists(args.image_dir):
                print(f"❌ Image directory not found: {args.image_dir}")
                return 1
            
            success = client.send_image_directory(
                args.image_dir,
                delay=args.delay,
                max_images=args.max_images,
                gps_csv=args.gps_csv
            )
            
            # Print final statistics
            client.print_statistics()
            
            return 0 if success else 1
        
        # Default: interactive mode if no image directory specified
        print("💡 No image directory specified. Starting interactive mode...")
        client.run_interactive_test()
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Client interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Client error: {e}")
        return 1
    finally:
        client.disconnect()


if __name__ == "__main__":
    sys.exit(main())