#!/usr/bin/env python3
# Copyright (C) 2024 Carnegie Mellon University
"""
SwiftMap Launch Script

Main entry point for the SwiftMap drone mapping system with multiple operation modes:
- Gradio GUI mode (default): Web interface with dual 3D viewers
- TCP server mode: Headless keyframe collection server
- Console mode: Process existing keyframes without GUI

Usage:
    python launch_mapping.py                    # GUI mode (default)
    python launch_mapping.py --server           # TCP server mode
    python launch_mapping.py --console          # Console mode
    python launch_mapping.py --help             # Show help
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add the repo root to path so the `swiftmap` and `vggt` packages resolve
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import components directly
from swiftmap.core.gradio_interface import MappingGradioInterface
from swiftmap.core.keyframe_selector import KeyframeSelector
from swiftmap.core.vggt_mapper import VGGTMapper


def print_banner():
    """Print system banner."""
    print("="*70)
    print("🚁 SwiftMap Mapping System")
    print("="*70)
    print("Real-time drone mapping with keyframe selection and 3D reconstruction")
    print("")
    print("Key Features:")
    print("  • TCP image reception on port 43322")
    print("  • Optical flow-based keyframe selection") 
    print("  • VGGT 3D reconstruction inference")
    print("  • Dual visualization (3D model + confidence mapping)")
    print("  • Compatible with the bundled test client (swiftmap/utils/test_client.py)")
    print("="*70)
    print("")


def gui_mode(args):
    """
    Run VGGT mapping system in GUI mode with Gradio interface.
    
    Args:
        args: Command line arguments
    """
    print("🖥️  Starting GUI mode with Gradio interface...")
    print(f"Interface will be available at: http://{args.host}:{args.gui_port}")
    print("Features:")
    print("  • Web-based control interface")
    print("  • Side-by-side 3D viewers (reconstruction + confidence)")
    print("  • Real-time keyframe collection monitoring")
    print("  • Interactive parameter adjustment")
    print("")
    
    try:
        # Create and launch Gradio interface
        interface = MappingGradioInterface(host=args.host, port=args.gui_port)
        
        interface.launch(
            share=args.share,
            inbrowser=not args.no_browser
        )
        
    except KeyboardInterrupt:
        print("\n🛑 GUI mode interrupted by user")
    except Exception as e:
        print(f"❌ Error in GUI mode: {e}")
        return False
    
    return True


def server_mode(args):
    """
    Run VGGT mapping system in server mode (TCP keyframe collection only).
    
    Args:
        args: Command line arguments
    """
    print("🌐 Starting TCP server mode...")
    print(f"TCP server will run on: {args.host}:{args.tcp_port}")
    print(f"Min disparity threshold: {args.min_disparity} pixels")
    print("Features:")
    print("  • Headless keyframe collection")
    print("  • Compatible with existing test clients")
    print("  • Automatic keyframe storage")
    print("  • Real-time statistics")
    print("")
    
    try:
        # Create keyframe selector
        selector = KeyframeSelector(
            host=args.host,
            port=args.tcp_port,
            min_disparity=args.min_disparity,
            visualize_flow=args.visualize_flow
        )
        
        # Add notification callback
        def keyframe_notification(keyframe_info):
            timestamp = datetime.now().strftime("%H:%M:%S")
            count = keyframe_info['keyframe_count']
            frame_id = keyframe_info['metadata']['frame_id']
            print(f"[{timestamp}] Keyframe {count} selected from frame {frame_id}")
        
        selector.add_keyframe_callback(keyframe_notification)
        
        # Start server
        if selector.start():
            print("✅ TCP server started successfully")
            print("📡 Waiting for drone clients...")
            print("📊 Keyframe statistics will be displayed in real-time")
            print("🛑 Press Ctrl+C to stop server")
            print("")
            
            # Keep server running
            while True:
                time.sleep(1)
                
        else:
            print("❌ Failed to start TCP server")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 TCP server mode interrupted by user")
        if 'selector' in locals():
            selector.stop()
    except Exception as e:
        print(f"❌ Error in server mode: {e}")
        return False
    
    return True


def console_mode(args):
    """
    Run VGGT mapping system in console mode (process existing keyframes).
    
    Args:
        args: Command line arguments
    """
    print("💻 Starting console mode...")
    print(f"Processing keyframes from: {args.keyframes_dir}")
    print("Features:")
    print("  • Batch processing of existing keyframes")
    print("  • 3D reconstruction generation")
    print("  • Confidence mapping")
    print("  • Results export")
    print("")
    
    try:
        # Find keyframe images
        import glob
        
        keyframe_patterns = [
            os.path.join(args.keyframes_dir, "*.jpg"),
            os.path.join(args.keyframes_dir, "*.jpeg"),
            os.path.join(args.keyframes_dir, "*.png")
        ]
        
        keyframe_paths = []
        for pattern in keyframe_patterns:
            keyframe_paths.extend(glob.glob(pattern))
        
        keyframe_paths = sorted(keyframe_paths)
        
        if not keyframe_paths:
            print(f"❌ No keyframes found in {args.keyframes_dir}")
            print("💡 Supported formats: JPG, JPEG, PNG")
            return False
        
        print(f"📁 Found {len(keyframe_paths)} keyframes")
        
        # Initialize VGGT mapper
        mapper = VGGTMapper()
        
        if not mapper.initialize_model():
            print("❌ Failed to initialize VGGT model")
            return False
        
        # Prepare processing parameters
        processing_params = {
            "conf_threshold": args.conf_threshold,
            "mask_sky": args.mask_sky,
            "mask_dynamic": not args.no_mask_dynamic,  # Invert for consistency
            "mask_black_bg": False,
            "mask_white_bg": False,
            "show_cam": True
        }
        
        print(f"⚙️  Processing parameters: {processing_params}")
        print("🔄 Starting VGGT processing...")
        
        # Process keyframes
        results = mapper.process_keyframes(keyframe_paths, processing_params)
        
        if results["success"]:
            print("✅ VGGT processing completed successfully")
            
            # Print results
            timing = results["timing"]
            scene_results = results["scene_results"]
            
            print(f"📊 Processing statistics:")
            print(f"  • Total time: {timing['total_processing']:.2f}s")
            print(f"  • Keyframes processed: {results['keyframe_count']}")
            print(f"  • 3D model: {scene_results.get('glb_path', 'Not generated')}")
            print(f"  • Point cloud: {scene_results.get('ply_path', 'Not generated')}")
            
            # Export camera poses to the same target directory
            target_dir = scene_results.get('target_directory', 'input_stream_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
            poses_path = os.path.join(target_dir, "camera_poses.json")
            
            if mapper.save_camera_poses_json(poses_path):
                print(f"  • Camera poses: {poses_path}")
            
            # Generate confidence mapping if available
            if results.get("confidence_results") and not results["confidence_results"].get("error"):
                conf_results = results["confidence_results"]
                print(f"  • Confidence map: {conf_results.get('glb_path', 'Not generated')}")
                
                stats = conf_results.get("statistics", {})
                if stats:
                    high_conf = stats.get("high_conf_points", 0)
                    total_points = stats.get("total_points", 0)
                    print(f"  • High-confidence points: {high_conf}/{total_points}")
            
            print("\n✅ Console mode completed successfully")
            
        else:
            print(f"❌ VGGT processing failed: {results.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in console mode: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SwiftMap - Real-time drone mapping with keyframe selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # GUI mode (default)
  %(prog)s --server --tcp-port 43322         # TCP server mode  
  %(prog)s --console --keyframes-dir images/ # Console mode
  %(prog)s --gui --host 0.0.0.0 --gui-port 8080  # Custom GUI settings
        """
    )
    
    # Operation mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", default=True,
                           help="Run in GUI mode with Gradio interface (default)")
    mode_group.add_argument("--server", action="store_true",
                           help="Run in TCP server mode (headless keyframe collection)")
    mode_group.add_argument("--console", action="store_true", 
                           help="Run in console mode (process existing keyframes)")
    
    # Network settings
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host address (default: 0.0.0.0)")
    parser.add_argument("--tcp-port", type=int, default=43322,
                       help="TCP server port for drone clients (default: 43322)")
    parser.add_argument("--gui-port", type=int, default=7866,
                       help="Gradio GUI port (default: 7866)")
    
    # Keyframe selection parameters
    parser.add_argument("--min-disparity", type=float, default=40.0,
                       help="Minimum optical flow disparity for keyframe selection (default: 40.0)")
    parser.add_argument("--visualize-flow", action="store_true",
                       help="Enable optical flow visualization for debugging")
    
    # Processing parameters
    parser.add_argument("--conf-threshold", type=float, default=60.0,
                       help="Confidence threshold percentage (default: 60.0)")
    parser.add_argument("--mask-sky", action="store_true", default=True,
                       help="Enable sky filtering (default: True)")
    parser.add_argument("--no-mask-dynamic", action="store_true",
                       help="Disable dynamic object filtering (default: False)")
    
    # GUI specific options
    parser.add_argument("--share", action="store_true",
                       help="Create shareable Gradio link")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't automatically open browser")
    
    # Console mode options
    parser.add_argument("--keyframes-dir", type=str, default="keyframes",
                       help="Directory containing keyframes for console mode")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine operation mode
    if args.server:
        success = server_mode(args)
    elif args.console:
        success = console_mode(args)
    else:  # GUI mode (default)
        success = gui_mode(args)
    
    if success:
        print("🏁 SwiftMap Mapping System finished successfully")
        return 0
    else:
        print("❌ SwiftMap Mapping System finished with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())