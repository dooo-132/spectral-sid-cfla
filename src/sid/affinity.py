#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimension-correlation affinity graph.

Each of the `d` embedding dimensions is a node. The edge weight between
dimensions `p` and `q` is derived from the Pearson correlation of their value
vectors across all `N` items.

Extracted verbatim (numerically) from the audited generator
`the frozen artifact root/rqvae/generate_spectral_sid_v3.py::build_dimension_affinity`.
Bit-exact reproduction of the canonical Beauty SID array is verified by
`tests/test_sid_generation.py` and documented in
`docs/SID_REPRODUCTION_REPORT.md`.

INPUT   X : (N, d) float32 item embeddings
OUTPUT  S : (d, d) float32 non-negative symmetric affinity, diag == 1

CANONICAL AFFINITY: `nonneg_pearson` -> S = (rho + 1) / 2
  (the audited artifact name `nonneg_pearson_v3` = this metric evaluated by
   generator version v3; `_v3` is NOT part of the metric name)
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

# canonical metric name -> audited CLI `--metric` value
CANONICAL_AFFINITY = 'nonneg_pearson'
AFFINITY_TO_METRIC = {
    'nonneg_pearson': 'pearson',
    'abs_pearson': 'abs_pearson',
    'cosine': 'cosine',
    # accept the audited artifact spelling as an alias
    'nonneg_pearson_v3': 'pearson',
    'abspearson_v3': 'abs_pearson',
}
METRIC_TO_CANONICAL = {
    'pearson': 'nonneg_pearson',
    'abs_pearson': 'abs_pearson',
    'cosine': 'cosine',
}


def build_dimension_affinity(X: np.ndarray, metric: str = 'nonneg_pearson',
                             eps: float = 1e-8) -> np.ndarray:
    """Build the (d, d) dimension affinity matrix.

    Parameters
    ----------
    X : (N, d) item embeddings.
    metric : one of `nonneg_pearson` (canonical), `abs_pearson`, `cosine`.
        The audited aliases `nonneg_pearson_v3` / `abspearson_v3` are accepted.
    eps : numerical guard for the cosine normalisation.

    Returns
    -------
    S : (d, d) float32, symmetric, non-negative, unit diagonal.
    """
    if metric not in AFFINITY_TO_METRIC:
        raise ValueError(
            f"unknown affinity {metric!r}; expected one of "
            f"{sorted(AFFINITY_TO_METRIC)}")
    m = AFFINITY_TO_METRIC[metric]

    # NOTE: standardise before correlating (upstream does this; it makes the
    # correlation invariant to per-dimension scale, which is what we want since
    # embedding dimensions have very different variances).
    X_std = StandardScaler().fit_transform(X)

    if m == 'pearson':
        S = np.corrcoef(X_std.T)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = (S + 1.0) / 2.0                      # -> [0, 1], non-negative
    elif m == 'abs_pearson':
        S = np.corrcoef(X_std.T)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = np.abs(S)
    else:  # cosine
        V = X_std.T
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)
        S = V @ V.T
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = (S + 1.0) / 2.0

    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)
