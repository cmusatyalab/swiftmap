# Copyright (C) 2024 Carnegie Mellon University

"""Array helpers for a map's point cloud, shared by the store and the pipeline."""

import numpy as np

_CONF_EPSILON = 1e-6


def flatten_colors(images: np.ndarray) -> np.ndarray:
    """(S,3,H,W) or (S,H,W,3) float[0,1] images -> (N,3) uint8 RGB."""
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    return (images.reshape(-1, 3) * 255).astype(np.uint8)


def confidence_mask(conf, percentile) -> np.ndarray:
    """Boolean keep-mask over a flat confidence array: ``conf >= P``-th percentile
    and strictly positive. ``percentile`` in [0, 100]; 0/None keeps all positive."""
    conf = np.asarray(conf, dtype=float).reshape(-1)
    if not percentile:
        return conf > _CONF_EPSILON
    thr = np.percentile(conf, float(percentile))
    return (conf >= thr) & (conf > _CONF_EPSILON)
