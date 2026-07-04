# Copyright (C) 2024 Carnegie Mellon University

"""Backbone registry + factory.

Each reconstruction backbone registers itself under a stable string key (e.g.
``"vggt"``) together with a human-facing label and short description used to
populate the model picker in the UI. ``get_mapper(name)`` builds the selected
backbone. This is the single place SwiftMap chooses a model, so adding a new
backbone is: implement ``BaseMapper``, decorate it with ``@register_mapper``.
"""

from typing import Callable, Dict, List, Type

# name -> {"cls": BaseMapper subclass, "label": str, "description": str}
_REGISTRY: Dict[str, dict] = {}


def register_mapper(name: str, label: str, description: str = "") -> Callable:
    """Class decorator registering a backbone under ``name``."""
    def _wrap(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"Backbone '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = {"cls": cls, "label": label, "description": description}
        return cls
    return _wrap


def get_mapper(name: str, **kwargs):
    """Instantiate the backbone registered under ``name``.

    Backends are imported lazily so registering one does not import its heavy
    model package until it is actually selected.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]["cls"](**kwargs)


def available_mappers() -> List[dict]:
    """List registered backbones as ``[{"name","label","description"}, ...]``.

    Used by the UI to build the model selector.
    """
    return [
        {"name": name, "label": meta["label"], "description": meta["description"]}
        for name, meta in _REGISTRY.items()
    ]


def is_registered(name: str) -> bool:
    return name in _REGISTRY
