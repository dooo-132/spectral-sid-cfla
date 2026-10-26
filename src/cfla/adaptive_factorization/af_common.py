#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared primitives for the adaptive-factorization study.

`fused_rank` reproduces exactly what `fg_common.union_fuse` computes for a fixed
weight vector, but from a compact per-row representation (top-K item ids with their
merged probabilities) rather than the full beam tensors.  This makes it possible to
evaluate a *per-user* weight vector on hundreds of thousands of rows.

Validation of the equivalence is in `tests/test_af_fusion_rank.py`: on real dumps it
must agree element-wise with `fg_common.union_fuse` at w = (0.5, 0.5) and at
non-uniform weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / 'analysis' / 'mechanism' / 'factorization_generality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.analysis.per_user import per_user_ndcg_vector  # noqa: E402

TOPK_EVAL = 20
BOOT_SEED = 20260912          # canonical statistics seed
NBIG = 10 ** 9   # sentinel key for invalid candidates when grouping by item id


def merge_probabilities(items, probs, k=None):
    """Merge beam columns that decode to the same item; return top-k (ids, probs).

    `items` (n, B) with -1 padding, `probs` (n, B) summing to 1 per row.
    Returns (top_ids (n, k), top_probs (n, k)) sorted by descending probability,
    ids = -1 with prob 0 in empty slots.  Vectorised via the audited grouped-
    probability machinery, so ties and duplicate beams are handled identically to
    `fg_common.union_fuse`.
    """
    from fg_common import _grouped_probabilities
    items = np.asarray(items); probs = np.asarray(probs)
    n, B = items.shape
    k = B if k is None else int(k)
    ks, score, real = _grouped_probabilities(items, probs)
    s = np.where(real, score, -np.inf)
    o = np.argsort(-s, axis=1, kind='stable')[:, :k]
    ids = np.take_along_axis(ks, o, axis=1)
    ps = np.take_along_axis(s, o, axis=1)
    good = np.isfinite(ps)
    return np.where(good, ids, -1).astype(np.int64), np.where(good, ps, 0.0)


class FusedCandidates:
    """Compact per-row candidate sets for the two factorizations.

    `ids_m`/`p_m` are (n, K) top-K item ids (merged over beams) and their
    probabilities under factorization m.  All that `fused_rank` needs.
    """

    def __init__(self, ids1, p1, ids2, p2, target):
        for arr in (ids1, p1, ids2, p2):
            if np.asarray(arr).ndim != 2:
                raise ValueError('candidate arrays must be 2-D')
        if not (ids1.shape == p1.shape == ids2.shape == p2.shape):
            raise ValueError('the two factorizations must have the same candidate width')
        n = ids1.shape[0]
        if not (len(target) == n):
            raise ValueError('target length mismatch')
        self.ids1 = np.asarray(ids1, dtype=np.int64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.ids2 = np.asarray(ids2, dtype=np.int64)
        self.p2 = np.asarray(p2, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.int64)
        self.n = n
        self.K = self.ids1.shape[1]

    def subset(self, keep):
        return FusedCandidates(self.ids1[keep], self.p1[keep], self.ids2[keep],
                               self.p2[keep], self.target[keep])

    def __len__(self):
        return self.n


def fused_rank(w1, cand):
    """1-based fused rank of the target; -1 when the target is in neither set.

    Implements, exactly and without regenerating beams,
    ``S(i) = w1*P1(i) + (1-w1)*P2(i)`` on the union of the two candidate sets.
    `w1` is a scalar or a per-row vector.  Ties in `S` are broken by ascending item
    id, which is the stable convention used throughout this repository.

    Equivalence with `fg_common.union_fuse` at constant weights is asserted by
    `tests/test_af_fusion_rank.py`.
    """
    ids1, p1 = cand.ids1, cand.p1
    ids2, p2 = cand.ids2, cand.p2
    n, K = ids1.shape
    w1 = np.asarray(w1, dtype=np.float64)
    if w1.ndim == 0:
        w1 = np.full(n, float(w1))
    w2 = 1.0 - w1

    rows = np.repeat(np.arange(n, dtype=np.int64), 2 * K)
    ids = np.concatenate([ids1, ids2], axis=1)
    contrib = np.concatenate([p1 * w1[:, None], p2 * w2[:, None]], axis=1)

    # A candidate participates only if the factorization it comes from actually
    # carries weight AND it has non-zero mass there.  Without the second condition a
    # weight of exactly 0 (pure single-factorization routing) would still let the
    # other arm's item ids enter the union with score 0, which changes the target's
    # rank relative to `fg_common.single_rank`.  This is what
    # `tests/test_af_fusion_rank.py` at w in {0.0, 1.0} guards.
    valid = (ids >= 0) & (contrib > 0.0)
    key = np.where(valid, ids, NBIG)
    order = np.argsort(key, axis=1, kind='stable')
    ks = np.take_along_axis(key, order, axis=1)
    vs = np.take_along_axis(contrib, order, axis=1)

    first = np.empty(ks.shape, dtype=bool)
    first[:, 0] = True
    first[:, 1:] = ks[:, 1:] != ks[:, :-1]
    grp = np.cumsum(first, axis=1) - 1

    acc = np.zeros_like(vs)
    np.add.at(acc, (rows, grp.reshape(-1)), vs.reshape(-1))
    grouped = np.take_along_axis(acc, grp, axis=1)
    real = first & (ks < NBIG)
    score = np.where(real, grouped, -np.inf)

    tgt = cand.target
    tgt_match = (ks == tgt[:, None]) & real
    tgt_score = np.where(tgt_match, score, -np.inf).max(axis=1)
    present = np.isfinite(tgt_score)
    rank = np.full(n, -1, dtype=np.int64)
    if present.any():
        rank[present] = 1 + (score[present] > tgt_score[present][:, None]).sum(axis=1)
    return rank


def per_user_ndcg(rank, k=TOPK_EVAL):
    return per_user_ndcg_vector(np.asarray(rank), k)


def summarize_rank(rank, topk=(5, 10, 20)):
    rank = np.asarray(rank)
    out = {}
    for nn in topk:
        hit = (rank >= 1) & (rank <= nn)
        out[f'Recall@{nn}'] = float(hit.mean())
        nd = np.zeros(len(rank))
        nd[hit] = 1.0 / np.log2(rank[hit] + 1)
        out[f'NDCG@{nn}'] = float(nd.mean())
    out['n'] = int(len(rank))
    out['target_present'] = float((rank >= 1).mean())
    return out


# ---------------------------------------------------------------------------
# paired user-level bootstrap (canonical settings)
# ---------------------------------------------------------------------------
def paired_bootstrap(a, b, n_boot=5000, seed=20260912, ci=95):
    """Paired user-level bootstrap of the mean difference a - b.

    `a` and `b` are per-user metric vectors on the SAME users, so the resample is
    shared.  Returns delta, CI bounds and a two-sided bootstrap p-value.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError('paired bootstrap needs equal-length vectors')
    d = a - b
    n = len(d)
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2.0))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2.0))
    # two-sided bootstrap p: how often the resampled mean crosses zero
    if delta >= 0:
        p = 2.0 * float((boots <= 0).mean())
    else:
        p = 2.0 * float((boots >= 0).mean())
    p = min(1.0, max(p, 1.0 / n_boot))
    return {'delta': delta, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
            'n_users': int(n), 'n_boot': int(n_boot), 'seed': int(seed)}


def mcnemar_exact(a_hit, b_hit):
    """Exact binomial McNemar test on paired hit indicators."""
    from math import comb
    a_hit = np.asarray(a_hit, dtype=bool); b_hit = np.asarray(b_hit, dtype=bool)
    n01 = int((~a_hit & b_hit).sum())
    n10 = int((a_hit & ~b_hit).sum())
    m = n01 + n10
    if m == 0:
        return {'n10': n10, 'n01': n01, 'p_value': 1.0}
    k = min(n10, n01)
    p = sum(comb(m, i) for i in range(0, k + 1)) / (2.0 ** m) * 2.0
    return {'n10': n10, 'n01': n01, 'p_value': float(min(1.0, p))}
