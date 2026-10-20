# Copyright (C) 2024 Carnegie Mellon University

"""Cross-view instance association: lifted instance masks (fragments) -> 3D instances.

Fragments are linked by voxel containment, |A & B| / min(|A|, |B|): first within a batch
on a fine grid, then across batches on a coarse grid that absorbs their misalignment.
Merges run strongest first and never join groups that share a keyframe.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import sparse

from swiftmap import constants
from swiftmap.database.site import Site
from swiftmap.database.types import SegmentedInstance


@dataclass
class Fragment:
    """One instance mask of one keyframe, lifted into the site's ENU frame."""
    points: np.ndarray
    conf: np.ndarray
    score: float
    batch: int
    frame: int


def associate(fragments: List[Fragment], site: Site) -> List[SegmentedInstance]:
    """Group fragments into instances seen in enough keyframes, most corroborated first."""
    groups = [[i] for i in range(len(fragments))]
    groups = _merge(fragments, groups, site, constants.SEG_LINK_VOXEL_M,
                    constants.SEG_LINK_CONTAINMENT, same_batch=True)
    groups = _merge(fragments, groups, site, constants.SEG_CROSS_VOXEL_M,
                    constants.SEG_CROSS_CONTAINMENT, same_batch=False)

    instances = []
    for group in groups:
        frames = {(fragments[i].batch, fragments[i].frame) for i in group}
        if len(frames) < constants.SEG_MIN_SUPPORT:
            continue
        points = np.concatenate([fragments[i].points for i in group])
        conf = np.concatenate([fragments[i].conf for i in group])
        best = site._best_per_cell(site._cell_keys(points, site.voxel_size), conf)
        score = float(np.mean([fragments[i].score for i in group]))
        instances.append(SegmentedInstance(points[best], len(frames), score))
    return sorted(instances, key=lambda inst: (-inst.support, -inst.score))


def _merge(fragments: List[Fragment], groups: List[List[int]], site: Site, voxel: float,
           threshold: float, same_batch: bool) -> List[List[int]]:
    """Merge groups whose voxels overlap by ``threshold``, within or across batches."""
    if len(groups) < 2:
        return groups

    keys = [np.unique(site._cell_keys(np.concatenate([fragments[i].points for i in g]), voxel))
            for g in groups]
    cells, cols = np.unique(np.concatenate(keys), return_inverse=True)
    rows = np.repeat(np.arange(len(groups)), [len(k) for k in keys])
    incidence = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                                  shape=(len(groups), len(cells)))
    shared = sparse.triu(incidence @ incidence.T, k=1).tocoo()
    size = np.array([len(k) for k in keys])
    containment = shared.data / np.minimum(size[shared.row], size[shared.col])

    batch = np.array([fragments[g[0]].batch for g in groups])
    link = (containment >= threshold) & ((batch[shared.row] == batch[shared.col]) == same_batch)
    order = np.argsort(-containment[link], kind="stable")
    a, b = shared.row[link][order], shared.col[link][order]

    parent = list(range(len(groups)))
    frames = [{(fragments[i].batch, fragments[i].frame) for i in g} for g in groups]

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(a, b):
        ri, rj = root(i), root(j)
        if ri == rj or frames[ri] & frames[rj]:
            continue
        parent[rj] = ri
        frames[ri] |= frames[rj]

    merged = {}
    for g, members in enumerate(groups):
        merged.setdefault(root(g), []).extend(members)
    return list(merged.values())
