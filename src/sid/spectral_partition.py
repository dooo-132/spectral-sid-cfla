#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral clustering of embedding DIMENSIONS into K semantic subspaces.

Not the usual usage: objects are dimensions (d of them), not samples. The
affinity matrix from `affinity.py` is fed to sklearn's SpectralClustering with
`affinity="precomputed"`, and the resulting cluster labels define the partition.

Extracted verbatim (numerically) from
`the frozen artifact root/rqvae/generate_spectral_sid_v3.py::spectral_group_dimensions`.

INPUT   X        : (N, d) item embeddings
        num_groups K
OUTPUT  groups   : list[K] of np.ndarray of dimension indices (a partition)
        dim_labels : (d,) int, group index of every dimension
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering

from .affinity import build_dimension_affinity


def spectral_group_dimensions(X: np.ndarray, num_groups: int = 3,
                              metric: str = 'nonneg_pearson',
                              random_state: int = 2025):
    """Partition the `d` embedding dimensions into `num_groups` subspaces.

    Returns
    -------
    groups : list of `num_groups` int arrays, disjoint, union == {0..d-1}
    dim_labels : (d,) int array
    """
    if num_groups < 2:
        raise ValueError(f"num_groups must be >= 2, got {num_groups}")
    num_items, emb_dim = X.shape
    if num_groups > emb_dim:
        raise ValueError(
            f"cannot form {num_groups} non-empty groups from {emb_dim} dimensions")

    S = build_dimension_affinity(X, metric=metric)

    # NOTE: random_state + n_init are part of the canonical configuration; the
    # bit-exact reproduction of the audited SID array depends on both.
    clustering = SpectralClustering(
        n_clusters=num_groups,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=random_state,
        n_init=20,
    )
    dim_labels = clustering.fit_predict(S).astype(np.int64)

    groups = [np.where(dim_labels == g)[0] for g in range(num_groups)]

    # A partition is a hard requirement for the SID construction: every
    # dimension must land in exactly one non-empty group.
    sizes = [len(g) for g in groups]
    if any(s == 0 for s in sizes):
        raise RuntimeError(
            f"spectral clustering produced an empty group (sizes={sizes}); "
            f"reduce K or check the affinity matrix")
    if sum(sizes) != emb_dim:
        raise RuntimeError(
            f"groups do not partition the {emb_dim} dimensions (sizes={sizes})")

    return groups, dim_labels


def partition_sizes(groups) -> list:
    return [int(len(g)) for g in groups]
