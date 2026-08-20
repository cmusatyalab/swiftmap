# Copyright (C) 2024 Carnegie Mellon University

"""Minimal headless auto-mapping server.

Starts a ``MappingSession`` and keeps it running until stopped -- a thin process
wrapper so the session can be driven over TCP (e.g. by ``test/test_client.py``).
Everything past collection + reconstruction (GPS alignment, NFN, site growth,
segmentation) is not wired up yet.
"""

import os
import time
from dataclasses import dataclass

from swiftmap.core import constants
from swiftmap.core.transport import protocol
from swiftmap.core.session import MappingSession


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = protocol.TCP_PORT
    backbone: str = constants.DEFAULT_RECONSTRUCTOR
    segmenter: str = constants.DEFAULT_SEGMENTER
    site: str = "map"
    batch_size: int = constants.DEFAULT_MAX_KEYFRAMES
    min_disparity: float = constants.DEFAULT_MIN_DISPARITY
    output_dir: str = "output"


class AutoMappingServer:
    """Starts a MappingSession's TCP collection + reconstruction worker; blocks until stopped."""

    def __init__(self, config: ServerConfig):
        self.cfg = config
        os.makedirs(config.output_dir, exist_ok=True)
        root = os.path.abspath(config.output_dir)

        self.session = MappingSession(
            host=config.host, min_disparity=config.min_disparity,
            root=root, site=config.site,
            backbone=[config.backbone, config.segmenter],
        )
        self.session.batch_size = config.batch_size

    def run(self):
        """Start the session and block until interrupted."""
        print(f"[swiftmap-server] starting on {self.cfg.host}:{self.cfg.port} "
              f"(backbone={self.cfg.backbone}, batch_size={self.cfg.batch_size})")
        if not self.session.start(port=self.cfg.port):
            raise RuntimeError("Failed to start the TCP collection server")

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[swiftmap-server] interrupted")
        finally:
            self.session.stop()
