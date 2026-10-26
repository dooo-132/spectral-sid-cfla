#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared machinery for the final candidate-likelihood ablation suite.

Everything here re-uses the FROZEN definitions rather than restating them.  The scoring
core is copied verbatim in behaviour from
`analysis/mechanism/candidate_conditioned/cc_rerank_methods.py`, which is the module that
produced the published numbers; the ablation only adds extra score variants around it.

Frozen definitions confirmed from disk (section 4 of the round prompt says not to trust the
prompt, so these were read out of cc_rerank_methods.py):

    zNLL_F(u,i) = (TotalNLL_F(u,i) - mu_F) / sigma_F
        mu/sigma fitted over ALL candidates of the split (never targets only)
    Uniform   S = 0.5*P_F1 + 0.5*P_F2          P_F = retrieval score, 0 outside F's beam
    MinNLL    S = -min(z1, z2)
    LSE       S = logsumexp(-z1, -z2)
    Gate      S = w1*P1 + (1-w1)*P2,  w1 = 1/(1+exp((z1-z2)/T))
    Hybrid    S = lam*Z(Uniform) + (1-lam)*Z(LSE),  Z = per-user z-score

All ablations are target-free at scoring time.  The target enters ONLY in
`rank_of_target`, which is used to score a finished ranking.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'outputs' / 'final_ablation'

# ---- frozen artifact locations (read-only) --------------------------------
CC = REPO / 'outputs' / 'candidate_conditioned'
CANON_RUN = CC / 'runs' / 'multi_candidate_nll_seed2025_20260914T194657'
CANON_SCALER = CC / 'validation' / 'candidate_nll_scaler.json'
SELECTED_HYPER = CC / 'validation' / 'selected_hyperparameters.json'
DG = REPO / 'outputs' / 'disagreement_gate'
SPORTS = REPO / 'outputs' / 'sports_confirmation'
SC = REPO / 'outputs' / 'cross_tokenizer_shared'
STEP1 = SC / 'step1_cross_tokenizer'
STEP2 = SC / 'step2_shared_model'
AF_FEAT = REPO / 'outputs' / 'adaptive_factorization' / 'selector' / 'features'
SIDS = REPO / 'outputs' / 'sids'

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
            'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
SPORTS_DS = 'Sports_and_Outdoors'
SID_LEN = 4                      # both primary factorizations are L=4
F1, F2 = 'spectral_k3', 'rqvae3'

# item counts per dataset (rows of each SID array), from TOKENIZER_INVENTORY.md
N_ITEMS = {'Beauty': 12101, 'Cell_Phones_and_Accessories': 10429,
           'Health_and_Personal_Care': 18534, 'Toys_and_Games': 11924,
           'Sports_and_Outdoors': 18357}

# statistics (section 55)
BOOT_N = 5000
BOOT_SEED = 20260912
CI = 95.0


# ---------------------------------------------------------------------------
# run scaffolding (sections 60-62)
# ---------------------------------------------------------------------------
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def make_run(ablation, dataset, cfg):
    """Create outputs/final_ablation/runs/<ablation>_<dataset>_<config_hash>/.

    Section 60: if the path already exists this is a HARD FAIL, never an overwrite.
    """
    h = config_hash(cfg)
    d = OUT / 'runs' / f'{ablation}_{dataset}_{h}'
    if d.exists():
        raise SystemExit(f'HARD FAIL: run directory already exists: {d}')
    d.mkdir(parents=True)
    (d / 'config.json').write_text(json.dumps(
        {'ablation': ablation, 'dataset': dataset, 'config_hash': h, 'config': cfg},
        indent=2, default=str))
    (d / 'PRE_RUN_ARTIFACT_MANIFEST.sha256').write_text(
        f'# config-only run directory created {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n'
        + '\n'.join(f'{sha256_file(p)}  {p}' for p in sorted(d.glob('config.json')))
        + '\n')
    return d


def finish_run(d, status='completed', detail=''):
    """Write status.json then POST_RUN manifest (status excluded, matching prior rounds)."""
    files = [p for p in sorted(Path(d).rglob('*'))
             if p.is_file() and p.name not in ('POST_RUN_ARTIFACT_MANIFEST.sha256',
                                               'status.json')]
    (Path(d) / 'POST_RUN_ARTIFACT_MANIFEST.sha256').write_text(
        '\n'.join(f'{sha256_file(p)}  {p}' for p in files) + '\n')
    (Path(d) / 'status.json').write_text(json.dumps(
        {'run_id': Path(d).name, 'status': status, 'detail': detail,
         'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2) + '\n')


# ---------------------------------------------------------------------------
# loading — frozen parquet layouts
# ---------------------------------------------------------------------------
def load_cells_dev(ds, split):
    """(union, nll) DataFrames for a development dataset from the canonical frozen run."""
    import pandas as pd
    u = pd.read_parquet(CANON_RUN / split / f'{ds}_candidate_union.parquet')
    n = pd.read_parquet(CANON_RUN / split / f'{ds}_candidate_nll.parquet')
    _assert_aligned(u, n, f'{ds}/{split}')
    return u, n


def load_cells_sports(split):
    """(union, nll) DataFrames for Sports (files are not dataset-prefixed)."""
    import pandas as pd
    u = pd.read_parquet(SPORTS / split / 'candidate_union.parquet')
    n = pd.read_parquet(SPORTS / split / 'candidate_nll.parquet')
    _assert_aligned(u, n, f'Sports/{split}')
    return u, n


def _assert_aligned(u, n, tag):
    if not (len(u) == len(n)
            and (u.user.values == n.user.values).all()
            and (u.candidate_item.values == n.candidate_item.values).all()):
        raise AssertionError(f'{tag}: union and NLL parquet are not row-aligned')


def grouped_arrays(u, n):
    """Per-user contiguous views so no vectorised op crosses a user boundary.

    Identical to cc_rerank_methods.grouped_arrays, plus beam ranks for the
    candidate-budget ablation.
    """
    order = np.lexsort((u.candidate_item.values, u.user.values))
    users, starts = np.unique(u.user.values[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    g = {
        'users': users, 'starts': starts, 'ends': ends, 'order': order,
        'item': u.candidate_item.values[order],
        'p1': u.retrieval_score_F1.values[order].astype(np.float64),
        'p2': u.retrieval_score_F2.values[order].astype(np.float64),
        'in1': u.source_in_F1_beam.values[order],
        'in2': u.source_in_F2_beam.values[order],
        'rank1': u.rank_F1_if_present.values[order].astype(np.int64),
        'rank2': u.rank_F2_if_present.values[order].astype(np.int64),
        'N1': n.TotalNLL_F1.values[order].astype(np.float64),
        'N2': n.TotalNLL_F2.values[order].astype(np.float64),
    }
    return g


def grouped_arrays_asis(u, n):
    """Same fields as grouped_arrays but WITHOUT reordering rows.

    The Sports confirmation round groups with `np.unique(u.user.values)` plus
    `np.searchsorted` on the raw row order (sc_gate_eval.ranks_and_scores), so its
    `tgt_rel` offsets index the parquet's original within-user order.  Sorting by
    candidate_item first -- which the development side does, because it carries a global
    target id instead -- silently mis-indexes every Sports offset.  Reproducing the frozen
    published NDCG requires this exact layout.
    """
    users = np.unique(u.user.values)
    starts = np.searchsorted(u.user.values, users)
    ends = np.append(starts[1:], len(u))
    return {
        'users': users, 'starts': starts, 'ends': ends,
        'item': u.candidate_item.values,
        'p1': u.retrieval_score_F1.values.astype(np.float64),
        'p2': u.retrieval_score_F2.values.astype(np.float64),
        'in1': u.source_in_F1_beam.values,
        'in2': u.source_in_F2_beam.values,
        'rank1': u.rank_F1_if_present.values.astype(np.int64),
        'rank2': u.rank_F2_if_present.values.astype(np.int64),
        'N1': n.TotalNLL_F1.values.astype(np.float64),
        'N2': n.TotalNLL_F2.values.astype(np.float64),
    }


def load_targets_dev(ds, split):
    """Per-user target as a SID-array row index (0-based), in sorted-unique-user order.

    cc_common's convention note: cand_target holds 0-based SID row indices, so the
    candidate_item column is directly comparable to it.  Verified in round 3 after a bug
    where sid[i-1] scored the wrong item.
    """
    z = np.load(AF_FEAT / f'{ds}_{split}.npz', allow_pickle=True)
    return np.asarray(z['cand_target'], dtype=np.int64)


def load_targets_sports_test():
    """Sports test target row indices, from the frozen confirmation round's dump."""
    z = np.load(SPORTS / 'test' / 'per_user_ranks.npz', allow_pickle=True)
    return np.asarray(z['tgt_rel'], dtype=np.int64)


# ---------------------------------------------------------------------------
# calibration — frozen definition
# ---------------------------------------------------------------------------
def fit_scaler(g, keys=('N1', 'N2')):
    """mu/sigma per factorization over ALL candidates (never targets only)."""
    out = {}
    for k in keys:
        v = g[k]
        v = v[np.isfinite(v)]
        out[k] = (float(v.mean()), float(v.std(ddof=0)))
    return out


def scaler_from_json(path, datasets_key=None):
    """Read a frozen scaler json into the {N1:(mu,sd), N2:(mu,sd)} shape."""
    d = json.loads(Path(path).read_text())
    if datasets_key is not None:
        d = d[datasets_key]
    return {'N1': (float(d['mu_F1']), float(d['sigma_F1'])),
            'N2': (float(d['mu_F2']), float(d['sigma_F2']))}


def zscores(g, sc):
    sd1 = sc['N1'][1] if abs(sc['N1'][1]) > 1e-12 else 1.0
    sd2 = sc['N2'][1] if abs(sc['N2'][1]) > 1e-12 else 1.0
    return (g['N1'] - sc['N1'][0]) / sd1, (g['N2'] - sc['N2'][0]) / sd2


# ---------------------------------------------------------------------------
# scores — the frozen core plus the ablation variants
# ---------------------------------------------------------------------------
def _lse2(a, b):
    """Numerically stable logsumexp of two arrays (same form as the frozen code)."""
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def uniform(g):
    return 0.5 * g['p1'] + 0.5 * g['p2']


def zscore_by_user(x, g):
    """Per-user z-score; sd <= 1e-12 maps to 0.0, exactly as the frozen implementation."""
    out = np.empty_like(x)
    for s, e in zip(g['starts'], g['ends']):
        seg = x[s:e]
        mu = seg.mean()
        sd = seg.std(ddof=0)
        out[s:e] = (seg - mu) / sd if sd > 1e-12 else 0.0
    return out


def score(g, method, sc=None, T=1.0, lam=0.5):
    """Per-candidate score, aligned with the grouped arrays.

    sc is required for every method that consumes calibrated NLL.
    """
    if method == 'Uniform':
        return uniform(g)
    if method == 'RawLSE':
        return _lse2(-g['N1'], -g['N2'])
    if method == 'MeanLSE':
        return _lse2(-g['N1'] / SID_LEN, -g['N2'] / SID_LEN)
    z1, z2 = zscores(g, sc)
    if method == 'ZLSE':
        return _lse2(-z1, -z2)
    if method == 'CenteredLSE':
        return _lse2(-(g['N1'] - sc['N1'][0]), -(g['N2'] - sc['N2'][0]))
    if method == 'MinNLL':
        return -np.minimum(z1, z2)
    if method == 'AvgNLL':
        return 0.5 * (-z1) + 0.5 * (-z2)
    if method == 'WorstNLL':
        return -np.maximum(z1, z2)
    if method == 'LSE_T':
        return T * _lse2(-z1 / T, -z2 / T)
    if method == 'Hybrid':
        return lam * zscore_by_user(uniform(g), g) + (1 - lam) * zscore_by_user(
            _lse2(-z1, -z2), g)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def rank_of_target(g, scores, target_as_row):
    """1-based rank of the target inside its user's candidate list; -1 if absent."""
    n = len(g['users'])
    rank = np.full(n, -1, dtype=np.int64)
    for u in range(n):
        s, e = g['starts'][u], g['ends'][u]
        t = target_as_row[u]
        if t < 0:
            continue
        hit = np.where(g['item'][s:e] == t)[0]
        if len(hit) == 0:
            continue
        seg = scores[s:e]
        st = seg[hit[0]]
        rank[u] = 1 + int((seg > st).sum())
    return rank


def rank_of_target_rel(g, scores, rel):
    """Rank of the target when it is given as an offset WITHIN the user's segment.

    The Sports confirmation round stores `tgt_rel` this way (range -1..58, i.e. a local
    candidate offset), not as a global SID row index: sc_gate_eval.py reads it as
    `u.rank_F1_if_present.values[starts[i] + tgt_rel[i]]`.  Using it as a global id
    silently yields ~0 coverage, which is how this was caught.
    """
    n = len(g['users'])
    rank = np.full(n, -1, dtype=np.int64)
    for u in range(n):
        r = int(rel[u])
        if r < 0:
            continue
        s, e = int(g['starts'][u]), int(g['ends'][u])
        idx = s + r
        if idx >= e:
            continue
        seg = scores[s:e]
        st = seg[r]
        rank[u] = 1 + int((seg > st).sum())
    return rank


def ndcg_from_rank(rank, k=20):
    hit = (rank >= 1) & (rank <= k)
    out = np.zeros(len(rank))
    out[hit] = 1.0 / np.log2(rank[hit] + 1)
    return out


def recall_from_rank(rank, k=20):
    return ((rank >= 1) & (rank <= k)).astype(np.float64)


def summarize(rank, ks=(5, 10, 20)):
    out = {}
    for k in ks:
        out[f'NDCG@{k}'] = float(ndcg_from_rank(rank, k).mean())
        out[f'Recall@{k}'] = float(recall_from_rank(rank, k).mean())
    out['n_users'] = int(len(rank))
    out['coverage'] = float((rank >= 1).mean())
    return out


def assert_targets_align(g, tgt, n_items=None):
    """Guards against the 0-based/1-based trap silently shifting every target.

    Targets are SID-array ROW indices, so row 0 is legitimate; the only sentinel is -1
    ("target not in the union").  An earlier version of this guard wrongly treated 0 as
    out of range, which is exactly the kind of off-by-one this is meant to catch.
    """
    if len(tgt) != len(g['users']):
        raise AssertionError(f'target length {len(tgt)} != n_users {len(g["users"])}')
    if (tgt < -1).any():
        raise AssertionError(f'target sentinel other than -1: {tgt[tgt < -1][:5]}')
    if n_items is not None:
        bad = tgt[tgt >= n_items]
        if len(bad):
            raise AssertionError(f'target row index >= n_items ({n_items}): {bad[:5]}')


# ---------------------------------------------------------------------------
# statistics (section 55)
# ---------------------------------------------------------------------------
def paired_bootstrap(a, b, n=BOOT_N, seed=BOOT_SEED, chunk=250):
    """Paired user-level bootstrap on the per-user metric difference a - b.

    Resamples are drawn in chunks: the naive (n, m) index matrix is 1.5 GB at
    m = 38,609, which is why the first version was killed for memory.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    m = len(d)
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        k = min(chunk, n - i)
        boots[i:i + k] = d[rng.integers(0, m, size=(k, m))].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    p = 2 * float((boots <= 0).mean()) if delta >= 0 else 2 * float((boots >= 0).mean())
    return {'delta': delta, 'ci_low': float(lo), 'ci_high': float(hi),
            'p_value': min(1.0, max(p, 1.0 / n)), 'n_users': m}


def write_csv(path, rows, fieldnames=None):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    if fieldnames is None:
        # union of keys in order of first appearance: rows for different datasets carry
        # different optional keys (e.g. the Sports lambda-selection note)
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')
