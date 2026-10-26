#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-user, per-K analysis view over canonical beam dumps.

Everything downstream (complementarity, subsets, oracle, partition families,
generation-side metrics) is built from ONE function here so that no two analyses
can disagree about normalization, merging or ranking.

Definitions (fixed, not tunable):

  For each model K and user u:
    * row-wise softmax over that model's VALID beams -> beam probability
    * item probability P_K(i|u) = sum of probabilities of beams decoding to i
    * rank_K(target) = 1 + #{items with strictly greater P_K}
      rank is -1 when the target is absent from that model's candidates
    * TopN_K(u) = the N highest-probability items

  Uniform fusion over a subset S of granularities:
    P_S(i|u) = (1/|S|) * sum_{K in S} P_K(i|u), computed on the union of the
    models' candidates.

Canonical conventions: affinity=nonneg_pearson, alignment_mode=canonical,
0-based dumped item ids canonicalised to 1-based.

Read-only with respect to every artifact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.utils import repo_paths as rp
from src.fusion.common import _row_softmax, canonicalize_item_ids

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
K_LIST = [2, 3, 4, 5]
AFFINITY = 'nonneg_pearson'
ALIGNMENT_MODE = 'canonical'


def dump_path(dataset, k, split='test', variant='nonneg_pearson_v3'):
    """Beam dump for a partition family.

    `variant` is the artifact suffix used in the beam-dump directory name:
      'nonneg_pearson_v3' (canonical spectral), 'evensplit_v3', 'randomsplit_v3'.
    """
    return (rp.ARTIFACT_ROOT / f'model/beam_dumps_{dataset}_{variant}' /
            f'{split}_k{k}_sid{k+1}.npz')


def sid_array_path(dataset, k, variant='nonneg_pearson_v3'):
    """Full SID array for a partition family (filename spelling differs)."""
    d = rp.ARTIFACT_ROOT / 'data' / dataset
    if variant == 'nonneg_pearson_v3':
        return d / f'{dataset}_t5_spectral_{variant}_k{k}_sid{k+1}.npy'
    return d / f'{dataset}_t5_{variant}_k{k}_sid{k+1}.npy'


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load_model(dataset, k, split='test', variant='nonneg_pearson_v3'):
    """Return (item_prob_dict_per_user, target, item_ids, prob_matrix)."""
    p = dump_path(dataset, k, split, variant)
    z = np.load(p, allow_pickle=True)
    items = canonicalize_item_ids(z['candidate_item_ids'])
    tgt = canonicalize_item_ids(z['target_item_ids'])
    scores = np.asarray(z['sequence_scores'], dtype=np.float64)
    raw = np.asarray(z['candidate_item_ids'])
    probs = _row_softmax(scores, raw >= 0)
    if not np.isfinite(probs).all():
        raise AssertionError(f'non-finite probabilities in {p}')
    # probabilities must sum to 1 per user over valid beams
    rs = probs.sum(axis=1)
    if not np.allclose(rs, 1.0, atol=1e-6):
        raise AssertionError(f'beam probabilities do not sum to 1 in {p}')
    return {'probs': probs, 'items': items, 'target': tgt,
            'z': z, 'path': p, 'sha256': sha256_file(p)}


def per_user_items(model):
    """[(item -> probability)] per user, merging beams that decode to one item."""
    items, probs = model['items'], model['probs']
    n, B = items.shape
    out = []
    for i in range(n):
        d = {}
        for j in range(B):
            it = int(items[i, j])
            if it < 0:
                continue
            d[it] = d.get(it, 0.0) + float(probs[i, j])
        out.append(d)
    return out


def rank_of(target_item, dist):
    """1-based rank, or -1 when absent. Ties counted as strictly-greater only."""
    if target_item not in dist:
        return -1
    t = dist[target_item]
    return 1 + sum(1 for v in dist.values() if v > t)


def topn(dist, n):
    return [it for it, _ in sorted(dist.items(), key=lambda kv: -kv[1])[:n]]


def ndcg_from_rank(rank, k):
    if rank < 1 or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def build_view(dataset, split='test', variant='nonneg_pearson_v3',
               ks=tuple(K_LIST)):
    """Load all granularities and materialise the per-user view."""
    models = {}
    for k in ks:
        m = load_model(dataset, k, split, variant)
        m['per_user'] = per_user_items(m)
        m['ranks'] = np.array([rank_of(int(m['target'][i]), m['per_user'][i])
                               for i in range(len(m['target']))], dtype=np.int64)
        models[k] = m
    # all K must share the same users and targets
    base = models[ks[0]]['target']
    for k in ks[1:]:
        if not np.array_equal(models[k]['target'], base):
            raise AssertionError(
                f'{dataset}: K={ks[0]} and K={k} use different test targets')
    return {'dataset': dataset, 'split': split, 'variant': variant,
            'ks': list(ks), 'target': base, 'models': models,
            'n_users': int(len(base))}


def fuse_subset(view, subset, topn_cap=None):
    """Uniform item-level fusion over a subset of granularities.

    Returns (per_user_dist, ranks).
    """
    models = view['models']
    n = view['n_users']
    dists, ranks = [], np.full(n, -1, dtype=np.int64)
    for i in range(n):
        acc = {}
        for k in subset:
            for it, p in models[k]['per_user'][i].items():
                acc[it] = acc.get(it, 0.0) + p / len(subset)
        if topn_cap is not None:
            keep = dict(sorted(acc.items(), key=lambda kv: -kv[1])[:topn_cap])
            acc = keep
        dists.append(acc)
        ranks[i] = rank_of(int(view['target'][i]), acc)
    return dists, ranks


def oracle_ranks(view, subset=None, topn_cap=None):
    """min_K rank_K(target); -1 if the target is in no model's candidates."""
    subset = subset or view['ks']
    n = view['n_users']
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        best = -1
        for k in subset:
            r = int(view['models'][k]['ranks'][i])
            if r < 1:
                continue
            if topn_cap is not None and r > topn_cap:
                continue
            best = r if best < 1 else min(best, r)
        out[i] = best
    return out


def summarize_ranks(ranks, topk=(5, 10, 20)) -> dict:
    """Recall@N (target present within N) and NDCG@N from a rank vector."""
    out = {}
    for n in topk:
        hit = (ranks >= 1) & (ranks <= n)
        out[f'Recall@{n}'] = float(hit.mean())
        nd = np.zeros(len(ranks))
        nd[hit] = 1.0 / np.log2(ranks[hit] + 1)
        out[f'NDCG@{n}'] = float(nd.mean())
    out['target_present'] = float((ranks >= 1).mean())
    return out


def paired_bootstrap(a, b, n_boot=5000, seed=20260912):
    """Paired user-level bootstrap on two per-user metric vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    if np.allclose(d, 0.0):
        return {'delta': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
                'p_value': float('nan'), 'note': 'identical'}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {'delta': float(d.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'p_value': float(min(1.0, max(p, 1.0 / n_boot))),
            'pct_change': float(100 * d.mean() / b.mean()) if b.mean() else float('nan')}


def per_user_ndcg_vector(ranks, n):
    hit = (ranks >= 1) & (ranks <= n)
    out = np.zeros(len(ranks))
    out[hit] = 1.0 / np.log2(ranks[hit] + 1)
    return out
