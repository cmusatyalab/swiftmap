#!/usr/bin/env python3
# Copyright (C) 2024 Carnegie Mellon University
"""
SwiftMap Launch Script

Main entry point for the SwiftMap drone mapping system. Starts the Gradio web
interface, which embeds the TCP keyframe-collection server and the VGGT mapping
pipeline.

Usage:
    python launch_mapping.py                         # default (http://localhost:7866)
    python launch_mapping.py --host 0.0.0.0 --gui-port 8080
    python launch_mapping.py --help
"""

import os
import sys
import argparse

# Add the repo root to path so the `swiftmap` and `vggt` packages resolve
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from swiftmap.frontend.gradio_interface import MappingGradioInterface


def print_banner():
    """Print system banner."""
    print("=" * 70)
    print("🚁 SwiftMap Mapping System")
    print("=" * 70)
    print("Real-time drone mapping with keyframe selection and 3D reconstruction")
    print("")
    print("Key Features:")
    print("  • TCP image reception on port 43322")
    print("  • Optical flow-based keyframe selection")
    print("  • VGGT 3D reconstruction inference")
    print("  • Dual visualization (3D model + confidence mapping)")
    print("  • Compatible with the bundled test client (test/test_client.py)")
    print("=" * 70)
    print("")


def gui_mode(args):
    """Launch the SwiftMap Gradio interface."""
    print("🖥️  Starting GUI mode with Gradio interface...")
    print(f"Interface will be available at: http://{args.host}:{args.gui_port}")
    print("Features:")
    print("  • Web-based control interface")
    print("  • Side-by-side 3D viewers (reconstruction + confidence)")
    print("  • Real-time keyframe collection monitoring")
    print("  • Interactive parameter adjustment")
    print("")

    try:
        interface = MappingGradioInterface(host=args.host, port=args.gui_port)
        interface.launch()
    except KeyboardInterrupt:
        print("\n🛑 GUI mode interrupted by user")
    except Exception as e:
        print(f"❌ Error in GUI mode: {e}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SwiftMap - Real-time drone mapping with keyframe selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                       # default GUI on http://localhost:7866
  %(prog)s --host 0.0.0.0 --gui-port 8080        # custom host/port
        """,
    )

    # Network settings
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address (default: 0.0.0.0)")
    parser.add_argument("--gui-port", type=int, default=7866,
                        help="Gradio GUI port (default: 7866)")

    args = parser.parse_args()

    print_banner()
    success = gui_mode(args)

    if success:
        print("🏁 SwiftMap Mapping System finished successfully")
        return 0
    else:
        print("❌ SwiftMap Mapping System finished with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
