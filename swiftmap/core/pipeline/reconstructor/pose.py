# Copyright (C) 2024 Carnegie Mellon University

"""Extrinsics -> world-pose math"""

import numpy as np


def camera_poses_from_extrinsics(extrinsic: np.ndarray):
    """(positions, rotations) in world coords from (S,3,4) extrinsics: ``-R^T t``, ``R^T``."""
    positions, rotations = [], []
    for ext in extrinsic:
        R, t = ext[:3, :3], ext[:3, 3]
        positions.append(-R.T @ t)
        rotations.append(R.T)
    return np.array(positions), np.array(rotations)
