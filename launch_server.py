#!/usr/bin/env python3
# Copyright (C) 2024 Carnegie Mellon University
"""
SwiftMap headless mapping server (no GUI).

Collects frame+GPS pairs over TCP and, each time the keyframe cap fills, runs the
whole pipeline (reconstruction, GPS alignment, NFN, segmentation) and exports the
results to the output directory. This is the container entrypoint that SteelEagle's
SwiftMap cognitive engine connects to.

Config is read from env vars (below), overridable by CLI flags:

    SWIFTMAP_HOST            bind address              (default 0.0.0.0)
    SWIFTMAP_PORT            TCP port                  (default 43322)
    SWIFTMAP_BACKBONE        vggt | vggt_omega         (default vggt)
    SWIFTMAP_SEGMENTER       segmentation model        (default sam3)
    SWIFTMAP_SITE            area-tag prefix           (default area)
    SWIFTMAP_MAX_KEYFRAMES   cap that triggers a run   (default 70)
    SWIFTMAP_CONF_THRESHOLD  confidence percentile     (default 60)
    SWIFTMAP_MASK_SKY        true|false                (default true)
    SWIFTMAP_KEEP_ALL        keep every frame          (default false)
    SWIFTMAP_OUTPUT_DIR      export dir (mount this)   (default output)
    SWIFTMAP_VIEWER_PORT     viewer web port           (default 7866)

Usage:
    python launch_server.py
    python launch_server.py --backbone vggt_omega --site drone-alpha
"""

import argparse
import os
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from swiftmap.core import constants, protocol
from swiftmap.server import AutoMappingServer, ServerConfig


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env(name: str, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main() -> int:
    p = argparse.ArgumentParser(
        description="SwiftMap headless auto-mapping server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default=_env("SWIFTMAP_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("SWIFTMAP_PORT", protocol.TCP_PORT)))
    p.add_argument("--backbone", default=_env("SWIFTMAP_BACKBONE", "vggt"))
    p.add_argument("--segmenter", default=_env("SWIFTMAP_SEGMENTER", "sam3"),
                   help="segmentation model used by the on-demand segment service")
    p.add_argument("--site", default=_env("SWIFTMAP_SITE", "area"),
                   help="area-tag prefix (drone / site name)")
    p.add_argument("--max-keyframes", type=int,
                   default=int(_env("SWIFTMAP_MAX_KEYFRAMES", constants.DEFAULT_MAX_KEYFRAMES)))
    p.add_argument("--conf-threshold", type=float,
                   default=float(_env("SWIFTMAP_CONF_THRESHOLD", constants.DEFAULT_CONF_THRESHOLD)))
    p.add_argument("--mask-sky", type=lambda s: str(s).lower() in ("1", "true", "yes", "on"),
                   default=_env_bool("SWIFTMAP_MASK_SKY", True))
    p.add_argument("--keep-all", action="store_true", default=_env_bool("SWIFTMAP_KEEP_ALL", False))
    p.add_argument("--output-dir", default=_env("SWIFTMAP_OUTPUT_DIR", "output"))
    p.add_argument("--viewer-host", default=_env("SWIFTMAP_VIEWER_HOST", "0.0.0.0"))
    p.add_argument("--viewer-port", type=int,
                   default=int(_env("SWIFTMAP_VIEWER_PORT", constants.GUI_PORT)))
    args = p.parse_args()

    cfg = ServerConfig(
        host=args.host, port=args.port,
        backbone=args.backbone, segmenter=args.segmenter, site=args.site,
        max_keyframes=args.max_keyframes,
        conf_threshold=args.conf_threshold,
        mask_sky=args.mask_sky, keep_all=args.keep_all,
        output_dir=args.output_dir,
        viewer_host=args.viewer_host, viewer_port=args.viewer_port,
    )
    AutoMappingServer(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
