#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen evaluation, coverage analysis and paired bootstrap.

Consumes frozen artifacts only:
  * `validation/validation_nll_scalers.json`        (validation-fitted zNLL scalers)
  * `validation/validation_baseline_selection.json` (selected lambda)
  * `runs/gate_<pool>/gate.pt`                      (frozen gate checkpoints)
and never re-fits anything.  The test split is read only from `stage=test` onward.

Methods per pool (prompt sections 47-48):
    BestSingle_{K2,K3,K4,K5,RQ}, Uniform, LSE, Hybrid, Gate, GateLikelihood,
    MeanNegZNLL, Oracle
plus the frozen old DynamicFusion-S4 replay, evaluated on BOTH candidate unions so
the two read-outs can be compared honestly.

Metrics: Recall@5/10/20, NDCG@5/10/20.  Statistics: paired user-level bootstrap,
5000 resamples, 95% CI, two-sided p.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import DATASETS, FLABEL, M5, OUT, S4, SHORT, write_json  # noqa: E402
from mf_core import (TOPK, group_view, load_split, per_user_lse_rows,  # noqa: E402
                     rank_and_metrics, user_zscore)
from mf_gate import build_features  # noqa: E402
from mf_stage import (LAMBDA_GRID, gate_predict, load_gate, load_scalers,  # noqa: E402
                      load_targets, pool_factorizations, score_block)

METRIC_KEYS = ['Recall@5', 'Recall@10', 'Recall@20',
               'NDCG@5', 'NDCG@10', 'NDCG@20']


class Cell:
    """Frozen arrays and scores for one (dataset, pool, split) cell."""

    def __init__(self, ds, pool, split, scalers_all, lam, device='cuda'):
        self.ds, self.pool, self.split = ds, pool, split
        self.device = device
        self.f_list = pool_factorizations(pool)
        df, upath = load_split(OUT, ds, pool, split)
        df.attrs['scalers'] = scalers_all[ds]
        self.df, self.union_path = df, upath
        users, starts, ends, order = group_view(df.user.values, df.candidate_item.values)
        self.users, self.starts, self.ends, self.order = users, starts, ends, order
        self.items = df.candidate_item.values[order]
        Z, P, present, R = score_block(df, self.f_list)
        self.Z, self.P, self.present, self.R = Z[order], P[order], present[order], R[order]
        self.target = load_targets(ds, split)[users - 1]
        self.tpos = np.full(len(users), -1, dtype=np.int64)
        for u, (s, e) in enumerate(zip(starts, ends)):
            w = np.where(self.items[s:e] == self.target[u])[0]
            if len(w):
                self.tpos[u] = s + int(w[0])
        self.lam = lam
        self.gate = (gate_model(pool, device)
                     if (_gate_exists(pool) and gate_ready(pool)) else None)
        if self.gate is not None:
            ck = self.gate[1]
            X = build_features(df, self.f_list, Z, P.astype(float), present, R)[order]
            self.Xn = ((X - ck['feat_mu']) / ck['feat_sd']).astype(np.float32)
        else:
            self.Xn = None

    # -- score formulas ----------------------------------------------------
    def scores(self):
        uni = self.P.mean(axis=1)
        lse = per_user_lse_rows(-self.Z)
        out = {'Uniform': uni, 'LSE': lse,
               'Hybrid': (self.lam * user_zscore(uni, self.starts, self.ends)
                          + (1 - self.lam) * user_zscore(lse, self.starts, self.ends)),
               'MeanNegZNLL': (-self.Z).mean(axis=1)}
        for j, f in enumerate(self.f_list):
            out[f'BestSingle_{FLABEL[f]}'] = self.P[:, j]
        if self.gate is not None:
            model = self.gate[0]
            W = gate_predict(model, self.Xn, device=self.device)
            out['Gate'] = (W * self.P).sum(axis=1)
            out['GateLikelihood'] = (W * (-self.Z)).sum(axis=1)
            out['_weights'] = W
        sc = np.zeros(len(self.items))
        ok = self.tpos >= 0
        sc[self.tpos[ok]] = 1.0
        out['Oracle'] = sc
        return out

    def packs(self):
        scores = self.scores()
        packs = {}
        for name, s in scores.items():
            if name.startswith('_'):
                continue
            packs[name] = rank_and_metrics(s, self.items, self.target,
                                           self.starts, self.ends)
        for f in self.f_list:
            key = f'BestSingle_{FLABEL[f]}'
            packs.setdefault(key, packs.get(key))
        return packs, scores


_GATE_CACHE = {}


def _gate_exists(pool):
    return (OUT / 'validation' / 'gate_training.json').exists()


def gate_ready(pool):
    """True when the frozen gate checkpoint for this pool exists on disk.

    Guard so that the S4-vs-DynamicFusion table can be produced before Gate-M5 has
    finished training, without silently scoring a nonexistent gate.
    """
    p = OUT / 'runs' / f'gate_{pool}' / 'gate.pt'
    if not p.exists():
        return False
    try:
        import json
        info = json.loads((OUT / 'validation' / 'gate_training.json').read_text())
        return pool in info
    except Exception:                                     # pragma: no cover
        return False


def gate_model(pool, device='cuda'):
    if pool not in _GATE_CACHE:
        _GATE_CACHE[pool] = None
    if _GATE_CACHE[pool] is None:
        _GATE_CACHE[pool] = load_gate(pool, device=device)
    return _GATE_CACHE[pool]


def clear_gate_cache():
    _GATE_CACHE.clear()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_bootstrap(a, b, n=5000, seed=20260912):
    """Paired user-level bootstrap of mean(a) - mean(b).

    delta, 95% percentile CI, and the two-sided bootstrap p-value
    p = (1 + #{|delta*| >= |delta_obs|}) / (n + 1).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise AssertionError(f'paired bootstrap needs equal lengths: {len(a)} vs {len(b)}')
    d = a - b
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    m = len(d)
    boots = np.array([d[rng.integers(0, m, m)].mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided bootstrap p-value under H0: delta = 0.  The resampled deltas are
    # centred at the observed value, so the null distribution is the centred cloud;
    # the achieved significance level is the share of |delta* - obs| that reaches
    # |obs|.  (Using |delta*| directly would measure P(|delta*| >= |obs|), which is
    # ~1 - p and has no decision meaning.)
    centred = boots - obs
    p = (1.0 + float((np.abs(centred) >= abs(obs)).sum())) / (n + 1.0)
    return {'delta': obs, 'ci_low': float(lo), 'ci_high': float(hi), 'p': p,
            'n_resamples': n, 'n_users': m, 'seed': seed}


BOOTSTRAP_PAIRS = [
    ('S4_Gate_vs_OldDynamicFusion_on_S4union', 'spectral4', 'Gate'),
    ('S4_LSE_vs_OldDynamicFusion_on_S4union', 'spectral4', 'LSE'),
    ('S4_Uniform_vs_OldDynamicFusion_on_S4union', 'spectral4', 'Uniform'),
    ('M5_Gate_vs_S4_Gate', None, 'Gate'),
    ('M5_Gate_vs_M5_LSE', None, 'M5GateVsLSE'),
    ('M5_LSE_vs_S4_LSE', None, 'M5LSEVsS4LSE'),
    ('S4_Gate_vs_S4_LSE', 'spectral4', 'GateVsLSE'),
]


# ---------------------------------------------------------------------------
# the frozen old DynamicFusion replayed on an arbitrary candidate union
# ---------------------------------------------------------------------------
def dynamic_scores_on_union(ds, split, cell, gamma=None, eta=None):
    """Old DynamicFusion's score for every candidate of THIS round's union.

    The four per-factorization probabilities and the four confidences come from the
    historical dumps; a union candidate that a factorization's beam does not contain
    gets probability 0 under that factorization, exactly as in the original script
    (where beam entries and union slots coincide).  Weights are per-user.
    """
    from mf_dynamic_fusion import dynamic_arrays
    da = dynamic_arrays(ds, split, gamma=gamma, eta=eta)
    raw_lut = {}
    n, U = da['items'].shape
    for i in range(n):
        raw_lut[i] = {int(it): j for j, it in enumerate(da['items'][i]) if it >= 0}
    w = da['w']
    fused = da['score']
    # this round's union stores canonical 1-based ids; the dumps are 0-based rows
    target_rows = cell.users - 1
    out = np.zeros(len(cell.items), dtype=np.float64)
    pos = 0
    for u in range(len(cell.users)):
        s, e = cell.starts[u], cell.ends[u]
        row = int(target_rows[u]) if u < len(target_rows) else u
        lut = raw_lut.get(row, {})
        seg_items = cell.items[s:e] - 1
        for k, it in enumerate(seg_items):
            j = lut.get(int(it))
            out[pos + k] = fused[row, j] if j is not None else 0.0
        pos += e - s
    return {'score': out, 'weights': w[target_rows] if len(target_rows) == n else None,
            'confidence': da['conf'][target_rows] if len(target_rows) == n else None,
            'gamma': da['gamma'], 'eta': da['eta'], 'q_raw': da['q_raw'],
            'q_norm': da['q_norm']}
