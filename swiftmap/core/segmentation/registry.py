# Copyright (C) 2024 Carnegie Mellon University

"""Segmenter registry + factory (mirrors the reconstruction-backbone registry)."""

from typing import Callable, Dict, List, Type

_REGISTRY: Dict[str, dict] = {}


def register_segmenter(name: str, label: str, description: str = "") -> Callable:
    def _wrap(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"Segmenter '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = {"cls": cls, "label": label, "description": description}
        return cls
    return _wrap


def get_segmenter(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown segmenter '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]["cls"](**kwargs)


def available_segmenters() -> List[dict]:
    return [{"name": n, "label": m["label"], "description": m["description"]}
            for n, m in _REGISTRY.items()]


def is_registered(name: str) -> bool:
    return name in _REGISTRY
