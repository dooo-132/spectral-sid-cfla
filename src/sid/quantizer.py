#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subspace-wise quantization: one SID token per semantic subspace.

For each dimension group, take the corresponding sub-embedding, standardise it,
optionally PCA-whiten it, and run k-means with `codebook_size` centroids. Each
item's token for that position is its cluster index.

Extracted verbatim (numerically) from
`the frozen artifact root/rqvae/generate_spectral_sid_v3.py::{reduce_if_needed,
generate_base_sid_by_group_kmeans}`.

INPUT   X            : (N, d) item embeddings
        groups       : list[K] of dimension-index arrays
        codebook_size: centroids per subspace (canonical 256)
        pca_dim      : PCA-whitening target; None/0 disables it (canonical 64)
OUTPUT  base_codes   : (N, K) int64, each entry in [0, codebook_size)
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def reduce_if_needed(X_sub: np.ndarray, pca_dim=None, random_state: int = 2025):
    """PCA-whiten a subspace when it is wider than `pca_dim`."""
    if pca_dim is None or pca_dim <= 0:
        return X_sub
    if X_sub.shape[1] <= pca_dim:
        return X_sub
    pca = PCA(n_components=pca_dim, whiten=True, random_state=random_state)
    return pca.fit_transform(X_sub)


def generate_base_sid_by_group_kmeans(X: np.ndarray, groups, codebook_size: int = 256,
                                      pca_dim: int = 64, random_state: int = 2025) -> np.ndarray:
    """Quantize each subspace independently -> (N, K) base semantic codes."""
    num_items = X.shape[0]
    num_groups = len(groups)
    codes = np.zeros((num_items, num_groups), dtype=np.int64)

    for g, dims in enumerate(groups):
        X_sub = X[:, dims]
        X_sub = StandardScaler().fit_transform(X_sub)
        X_sub = reduce_if_needed(X_sub, pca_dim=pca_dim, random_state=random_state)

        # NOTE: random_state and n_init belong to the canonical configuration.
        kmeans = KMeans(
            n_clusters=codebook_size,
            random_state=random_state,
            n_init=30,
            max_iter=500,
            verbose=0,
        )
        codes[:, g] = kmeans.fit_predict(X_sub)

    # Every code must fit the codebook; otherwise the token layout is ambiguous.
    if codes.min() < 0 or codes.max() >= codebook_size:
        raise RuntimeError(
            f"codes outside [0, {codebook_size}): "
            f"min={codes.min()} max={codes.max()}")
    return codes
