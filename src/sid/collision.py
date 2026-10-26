#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collision detection and deterministic disambiguation.

Items may share the same BASE semantic SID. A final token is appended that holds
the item's index inside its collision group, which makes the FINAL SID injective.

IMPORTANT INTERPRETATION
------------------------
`final_unique_sid_count == N` is a CORRECTNESS ASSERTION of the construction,
not an empirical result or a performance metric. The final collision rate is
identically zero BY DESIGN. The informative quantity is the BASE-prefix
collision rate, which is what the analysis reports.

Extracted verbatim (numerically) from
`the frozen artifact root/rqvae/generate_spectral_sid_v3.py::{compute_collision_rate,
add_collision_token}`.

INPUT   base_codes : (N, K) int64
OUTPUT  final_codes: (N, K+1) int64, last column = local disambiguation index
"""
from __future__ import annotations

import numpy as np


def compute_collision_rate(codes: np.ndarray):
    """Return (rate, n_unique, n_total) over full-length rows."""
    tuples = [tuple(row.tolist()) for row in codes]
    unique_num = len(set(tuples))
    total = len(tuples)
    return 1.0 - unique_num / total, unique_num, total


def collision_stats(base_codes: np.ndarray) -> dict:
    """Describe the BASE-prefix collisions (the meaningful statistic)."""
    keys = {}
    for i, row in enumerate(base_codes):
        keys.setdefault(tuple(int(x) for x in row), []).append(i)
    sizes = [len(v) for v in keys.values()]
    colliding_items = sum(s for s in sizes if s > 1)
    n = int(base_codes.shape[0])
    return {
        'num_items': n,
        'num_base_sid': len(keys),
        'num_collision_groups': int(sum(1 for s in sizes if s > 1)),
        'num_items_in_collision_groups': int(colliding_items),
        'max_collision_group': int(max(sizes)) if sizes else 0,
        'prefix_collision_rate': float(1.0 - len(keys) / n),
        'prefix_uniqueness': float(len(keys) / n),
    }


def add_collision_token(base_codes: np.ndarray, codebook_size: int = 256) -> np.ndarray:
    """Append the deterministic disambiguation token.

    The local index must fit in the codebook, otherwise a single extra token is
    not sufficient and the construction is invalid.
    """
    num_items, num_tokens = base_codes.shape

    groups = {}
    for i in range(num_items):
        groups.setdefault(tuple(int(x) for x in base_codes[i]), []).append(i)

    max_group = max(len(v) for v in groups.values()) if groups else 0
    if max_group > codebook_size:
        raise ValueError(
            f"largest collision group is {max_group} > codebook_size "
            f"{codebook_size}: one disambiguation token is not enough; "
            f"increase the codebook or use hierarchical disambiguation")

    new_codes = np.zeros((num_items, num_tokens + 1), dtype=np.int64)
    new_codes[:, :num_tokens] = base_codes
    for item_indices in groups.values():
        for local_id, item_idx in enumerate(item_indices):
            new_codes[item_idx, -1] = local_id

    # correctness assertion, not a performance claim
    _, n_unique, n_total = compute_collision_rate(new_codes)
    assert n_unique == n_total, (
        f"final SID is not injective: {n_unique} unique for {n_total} items")
    return new_codes
