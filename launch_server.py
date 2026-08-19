#!/usr/bin/env python3
# Copyright (C) 2024 Carnegie Mellon University
"""
SwiftMap minimal headless server: starts a MappingSession and blocks.

Collects frame+GPS pairs over TCP and reconstructs each batch as it fills. GPS
alignment, NFN, site growth, and segmentation are not wired up yet.

Config is read from env vars (below), overridable by CLI flags:

    SWIFTMAP_HOST            bind address              (default 0.0.0.0)
    SWIFTMAP_PORT            TCP port                  (default 43322)
    SWIFTMAP_BACKBONE        vggt | vggt_omega         (default vggt)
    SWIFTMAP_SEGMENTER       segmentation model        (default sam3)
    SWIFTMAP_SITE            site-tag prefix           (default map)
    SWIFTMAP_BATCH_SIZE      keyframes per batch       (default 70)
    SWIFTMAP_MIN_DISPARITY   keyframe-selection px     (default 40)
    SWIFTMAP_KEEP_ALL        keep every frame          (default false)
    SWIFTMAP_OUTPUT_DIR      export dir (mount this)   (default output)

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

from swiftmap.core import constants
from swiftmap.core.transport import protocol
from swiftmap.server import AutoMappingServer, ServerConfig


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env(name: str, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main() -> int:
    p = argparse.ArgumentParser(
        description="SwiftMap minimal headless server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default=_env("SWIFTMAP_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("SWIFTMAP_PORT", protocol.TCP_PORT)))
    p.add_argument("--backbone", default=_env("SWIFTMAP_BACKBONE", constants.DEFAULT_RECONSTRUCTOR))
    p.add_argument("--segmenter", default=_env("SWIFTMAP_SEGMENTER", constants.DEFAULT_SEGMENTER))
    p.add_argument("--site", default=_env("SWIFTMAP_SITE", "map"))
    p.add_argument("--batch-size", type=int,
                   default=int(_env("SWIFTMAP_BATCH_SIZE", constants.DEFAULT_MAX_KEYFRAMES)))
    p.add_argument("--min-disparity", type=float,
                   default=float(_env("SWIFTMAP_MIN_DISPARITY", constants.DEFAULT_MIN_DISPARITY)))
    p.add_argument("--keep-all", action="store_true", default=_env_bool("SWIFTMAP_KEEP_ALL", False))
    p.add_argument("--output-dir", default=_env("SWIFTMAP_OUTPUT_DIR", "output"))
    args = p.parse_args()

    cfg = ServerConfig(
        host=args.host, port=args.port,
        backbone=args.backbone, segmenter=args.segmenter, site=args.site,
        batch_size=args.batch_size, min_disparity=args.min_disparity,
        keep_all=args.keep_all, output_dir=args.output_dir,
    )
    AutoMappingServer(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
