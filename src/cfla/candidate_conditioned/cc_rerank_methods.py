#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE E + F — candidate-level calibration and the non-learned rerankers.

Calibration (E).  F1 and F2 are different SID systems, so their NLLs are not on one
scale.  The scaler is fitted on **all VALIDATION candidates** — not only validation
targets, which would distort the distribution exactly the way section 25 warns about —
and is frozen before any test metric is touched:

    zNLL_F(u,i) = (TotalNLL_F(u,i) - mu_F) / sigma_F

Scoring (F).  Four non-learned methods, all target-free, all deterministic:

    MinNLL  S = -min(z1, z2)
    LSE     S = logsumexp(-z1, -z2)                     (numerically stable)
    Gate    S = w1*P1(i|u) + w2*P2(i|u),  w = softmax(-z/T)
    Hybrid  S = lam*Z(S_uniform) + (1-lam)*Z(S_LSE)     (Z = user-level z-score)

    P_F(i|u) = 0 for a candidate outside F's beam, while zNLL_F is still computed
    (section 33).  That case is common here and is counted and reported.

Temperature T and weight lambda are chosen on VALID only, over the pre-declared grids.

Usage
  python3 cc_rerank_methods.py --rundir <run> --outdir <outputs/candidate_conditioned>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_common import DATASETS, F1, F2, SHORT, write_json  # noqa: E402

T_GRID = [0.25, 0.5, 1.0, 2.0]
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_cells(rundir, ds, split):
    import pandas as pd
    d = Path(rundir) / split
    u = pd.read_parquet(d / f'{ds}_candidate_union.parquet')
    n = pd.read_parquet(d / f'{ds}_candidate_nll.parquet')
    if not (len(u) == len(n) and (u.user.values == n.user.values).all()
            and (u.candidate_item.values == n.candidate_item.values).all()):
        raise AssertionError(f'{ds}/{split}: union and NLL parquet are not row-aligned')
    return u, n


def grouped_arrays(u, n):
    """Per-user views, so vectorised maths never crosses a user boundary."""
    order = np.lexsort((u.candidate_item.values, u.user.values))
    g = {}
    users, starts = np.unique(u.user.values[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    g['users'] = users; g['starts'] = starts; g['ends'] = ends
    g['item'] = u.candidate_item.values[order]
    g['p1'] = u.retrieval_score_F1.values[order].astype(np.float64)
    g['p2'] = u.retrieval_score_F2.values[order].astype(np.float64)
    g['in1'] = u.source_in_F1_beam.values[order]
    g['in2'] = u.source_in_F2_beam.values[order]
    g['z1'] = n.TotalNLL_F1.values[order].astype(np.float64)
    g['z2'] = n.TotalNLL_F2.values[order].astype(np.float64)
    g['order'] = order
    return g


def fit_scaler(g):
    """mu/sigma per factorization over ALL candidates in this split."""
    out = {}
    for key in ('z1', 'z2'):
        v = g[key]
        v = v[np.isfinite(v)]
        out[key] = (float(v.mean()), float(v.std(ddof=0)))
    return out


def apply_scaler(g, sc):
    z1 = (g['z1'] - sc['z1'][0]) / sc['z1'][1]
    z2 = (g['z2'] - sc['z2'][0]) / sc['z2'][1]
    return z1, z2


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def score_cell(g, z1, z2, method, T=None, lam=None):
    """Return a per-candidate score array aligned with the grouped arrays."""
    if method == 'MinNLL':
        return -np.minimum(z1, z2)
    if method == 'LSE':
        m = np.maximum(-z1, -z2)
        return m + np.log(np.exp(-z1 - m) + np.exp(-z2 - m))
    if method == 'Gate':
        w1 = 1.0 / (1.0 + np.exp((z1 - z2) / T))
        return w1 * g['p1'] + (1.0 - w1) * g['p2']
    if method == 'Uniform':
        return 0.5 * g['p1'] + 0.5 * g['p2']
    if method == 'Hybrid':
        lse = score_cell(g, z1, z2, 'LSE')
        uni = 0.5 * g['p1'] + 0.5 * g['p2']
        return lam * _zscore_by_user(uni, g) + (1 - lam) * _zscore_by_user(lse, g)
    raise ValueError(method)


def _zscore_by_user(x, g):
    out = np.empty_like(x)
    for s, e in zip(g['starts'], g['ends']):
        seg = x[s:e]
        mu = seg.mean(); sd = seg.std(ddof=0)
        out[s:e] = (seg - mu) / sd if sd > 1e-12 else 0.0
    return out


def rank_of_target(g, scores, target_as_row):
    """1-based rank of each user's target inside that user's candidate list, -1 if absent.

    `target_as_row` is a per-user SID-array row index (see cc_common's convention note).
    """
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


def ndcg_from_rank(rank, k=20):
    hit = (rank >= 1) & (rank <= k)
    out = np.zeros(len(rank))
    out[hit] = 1.0 / np.log2(rank[hit] + 1)
    return out


def _targets(rundir, ds, split, featdir):
    """Evaluation-only helper: the user's target as a SID-array row index.

    This is the ONLY place the target enters, and it is used exclusively to score a
    method's ranking.  It is never passed to a scorer.
    """
    z = np.load(Path(featdir) / f'{ds}_{split}.npz', allow_pickle=True)
    return np.asarray(z['cand_target'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--outdir', default='outputs/candidate_conditioned')
    ap.add_argument('--featdir',
                    default='outputs/adaptive_factorization/selector/features')
    ap.add_argument('--datasets', default=','.join(DATASETS),
                    help='restrict to the datasets whose parquet files already exist')
    a = ap.parse_args(argv)
    datasets = a.datasets.split(',')

    rundir, out = Path(a.rundir), Path(a.outdir)
    (out / 'validation').mkdir(parents=True, exist_ok=True)
    scaler_out = {}

    # ---------------- PHASE E: fit on VALID only ----------------
    for ds in datasets:
        u, n = load_cells(rundir, ds, 'valid')
        g = grouped_arrays(u, n)
        sc = fit_scaler(g)
        scaler_out[ds] = {'mu_F1': sc['z1'][0], 'sigma_F1': sc['z1'][1],
                          'mu_F2': sc['z2'][0], 'sigma_F2': sc['z2'][1],
                          'n_candidates': int(len(g['item'])),
                          'fitted_on': 'ALL validation candidates (not targets only)'}
        print(f'  {SHORT[ds]:12s} mu_F1={sc["z1"][0]:.4f} sd_F1={sc["z1"][1]:.4f}  '
              f'mu_F2={sc["z2"][0]:.4f} sd_F2={sc["z2"][1]:.4f}  '
              f'({len(g["item"])} candidates)')
    write_json(out / 'validation' / 'candidate_nll_scaler.json', scaler_out)

    # ---------------- PHASE F: select T and lambda on VALID ----------------
    selection_rows = []
    for ds in datasets:
        u, n = load_cells(rundir, ds, 'valid')
        g = grouped_arrays(u, n)
        z1, z2 = apply_scaler(g, {'z1': (scaler_out[ds]['mu_F1'], scaler_out[ds]['sigma_F1']),
                                  'z2': (scaler_out[ds]['mu_F2'], scaler_out[ds]['sigma_F2'])})
        tgt = _targets(rundir, ds, 'valid', a.featdir)
        for T in T_GRID:
            r = rank_of_target(g, score_cell(g, z1, z2, 'Gate', T=T), tgt)
            selection_rows.append({'dataset': ds, 'grid': 'T', 'value': T,
                                   'valid_NDCG@20': float(ndcg_from_rank(r, 20).mean()),
                                   'valid_Recall@20': float(((r >= 1) & (r <= 20)).mean())})
        for lam in LAMBDA_GRID:
            r = rank_of_target(g, score_cell(g, z1, z2, 'Hybrid', lam=lam), tgt)
            selection_rows.append({'dataset': ds, 'grid': 'lambda', 'value': lam,
                                   'valid_NDCG@20': float(ndcg_from_rank(r, 20).mean()),
                                   'valid_Recall@20': float(((r >= 1) & (r <= 20)).mean())})
        for m in ('Uniform', 'MinNLL', 'LSE'):
            r = rank_of_target(g, score_cell(g, z1, z2, m), tgt)
            selection_rows.append({'dataset': ds, 'grid': 'method', 'value': m,
                                   'valid_NDCG@20': float(ndcg_from_rank(r, 20).mean()),
                                   'valid_Recall@20': float(((r >= 1) & (r <= 20)).mean())})
        del u, n, g, z1, z2

    import csv
    with (out / 'validation' / 'validation_method_results.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['dataset', 'grid', 'value', 'valid_NDCG@20',
                                           'valid_Recall@20'])
        w.writeheader()
        for r in selection_rows:
            w.writerow(r)

    print('\n=== VALID selection ===')
    chosen = {}
    for ds in datasets:
        rows = [r for r in selection_rows if r['dataset'] == ds]
        bestT = max((r for r in rows if r['grid'] == 'T'), key=lambda r: r['valid_NDCG@20'])
        bestL = max((r for r in rows if r['grid'] == 'lambda'), key=lambda r: r['valid_NDCG@20'])
        vals = {r['value']: r['valid_NDCG@20'] for r in rows if r['grid'] == 'method'}
        chosen[ds] = {'T': bestT['value'], 'lambda': bestL['value'],
                      'valid_ndcg20': {'Uniform': vals.get('Uniform'),
                                       'MinNLL': vals.get('MinNLL'),
                                       'LSE': vals.get('LSE'),
                                       'Gate_best_T': bestT['valid_NDCG@20'],
                                       'Hybrid_best_lambda': bestL['valid_NDCG@20']}}
        print(f'  {SHORT[ds]:12s} T*={bestT["value"]:<5} lam*={bestL["value"]:<5} '
              f'uniform={vals.get("Uniform"):.6f} minNLL={vals.get("MinNLL"):.6f} '
              f'LSE={vals.get("LSE"):.6f} gate={bestT["valid_NDCG@20"]:.6f} '
              f'hybrid={bestL["valid_NDCG@20"]:.6f}')
    write_json(out / 'validation' / 'selected_hyperparameters.json', chosen)
    print(f'\nwrote scaler, validation results and selection to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
