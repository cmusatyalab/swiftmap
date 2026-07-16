# Copyright (C) 2024 Carnegie Mellon University
"""Pluggable reconstruction backbones.

Public API:
    BaseMapper            base class every backbone implements
    register_mapper       decorator to register a backbone
    get_mapper(name)      build a backbone by key ("vggt", "vggt_omega")
    available_mappers()   list backbones (for the UI model picker)

Importing this package also imports the backend adapters, which registers them.
``VGGTMapper`` is re-exported for backward compatibility.
"""
from swiftmap.core.mapper.base import BaseMapper
from swiftmap.core.mapper.registry import (
    register_mapper, get_mapper, available_mappers, is_registered)

# Importing the backends registers them in the registry.
from swiftmap.core.mapper.backends import VGGTMapper, VGGTOmegaMapper

__all__ = [
    "BaseMapper",
    "register_mapper",
    "get_mapper",
    "available_mappers",
    "is_registered",
    "VGGTMapper",
    "VGGTOmegaMapper",
]
