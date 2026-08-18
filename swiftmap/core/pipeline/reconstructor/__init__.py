# Copyright (C) 2024 Carnegie Mellon University
"""Pluggable reconstruction backbones.

Public API:
    BaseReconstructor            base class every backbone implements
    register_reconstructor       decorator to register a backbone
    get_reconstructor(name)      build a backbone by key ("vggt", "vggt_omega")
    available_reconstructors()   list backbones (for the UI model picker)

Importing this package also imports the backend adapters, which registers them.
``VGGTReconstructor`` is re-exported for backward compatibility.
"""
from swiftmap.core.pipeline.reconstructor.base import BaseReconstructor
from swiftmap.core.pipeline.reconstructor.registry import (
    register_reconstructor, get_reconstructor, available_reconstructors)

# Importing the backends registers them in the registry.
from swiftmap.core.pipeline.reconstructor.backends import VGGTReconstructor, VGGTOmegaReconstructor

__all__ = [
    "BaseReconstructor",
    "register_reconstructor",
    "get_reconstructor",
    "available_reconstructors",
    "VGGTReconstructor",
    "VGGTOmegaReconstructor",
]
