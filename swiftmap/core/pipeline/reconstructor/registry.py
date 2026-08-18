# Copyright (C) 2024 Carnegie Mellon University

"""Backbone registry + factory.

Each backbone registers under a stable key (``"vggt"``, ``"vggt_omega"``) with a label
and description; ``get_reconstructor(name)`` builds it. Adding a backbone means importing its
module -- no call site changes."""

from typing import Callable, Dict, List, Type

# name -> {"cls": BaseReconstructor subclass, "label": str, "description": str}
_REGISTRY: Dict[str, dict] = {}


def register_reconstructor(name: str, label: str, description: str = "") -> Callable:
    """Class decorator registering a backbone under ``name``."""
    def _wrap(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"Backbone '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = {"cls": cls, "label": label, "description": description}
        return cls
    return _wrap


def get_reconstructor(name: str, **kwargs):
    """Instantiate the backbone registered under ``name``.

    Backends are imported lazily so registering one does not import its heavy
    model package until it is actually selected.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]["cls"](**kwargs)


def available_reconstructors() -> List[dict]:
    """List registered backbones as ``[{"name","label","description"}, ...]``.

    Used by the UI to build the model selector.
    """
    return [
        {"name": name, "label": meta["label"], "description": meta["description"]}
        for name, meta in _REGISTRY.items()
    ]
