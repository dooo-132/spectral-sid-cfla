#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core scoring, calibration and metric primitives for the multi-factorization
candidate-level fusion round.

Frozen definitions used everywhere downstream:

  P_F(u, i)   per-factorization retrieval probability: the item's beam probability
              after merging beams that decode to the same item (sum), and 0 when the
              item is not in F's beam.
  zNLL_F(u, i) = (TotalNLL_F(u, i) - mu_F) / sigma_F, with (mu, sigma) fitted on ALL
              VALIDATION candidates of that (dataset, factorization).  NEVER on
              targets only, and NEVER on test.
  Score_S4_uniform = mean_{F in S4} P_F
  Score_S4_LSE     = logsumexp(-zNLL_K2, -zNLL_K3, -zNLL_K4, -zNLL_K5)
  Score_M5_uniform = mean_{F in M5} P_F
  Score_M5_LSE     = logsumexp over the five -zNLL
  Score_hybrid     = lam * Z(Uniform) + (1 - lam) * Z(LSE), Z = per-user z-score
  Score_gate       = sum_F w_F(u,i) * P_F(u,i)
  Score_gate_lik   = sum_F w_F(u,i) * (-zNLL_F(u,i))

Higher score is better.  Longer SID length means more scored positions, so raw
TotalNLL is NOT comparable across K; every cross-F likelihood quantity therefore
uses zNLL (prompt sections 10 and 25).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import FLABEL, M5, S4  # noqa: E402

TOPK = (5, 10, 20)


# ---------------------------------------------------------------------------
# grouped (per-user) views
# ---------------------------------------------------------------------------
def group_view(user_col, item_col):
    """Return (users, starts, ends, order) for a row block sorted by user.

    `order` maps grouped position -> original row index, so any per-row array can be
    aligned with `arr[order]`.
    """
    user_col = np.asarray(user_col, dtype=np.int64)
    item_col = np.asarray(item_col, dtype=np.int64)
    order = np.lexsort((item_col, user_col))
    users, starts = np.unique(user_col[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    return users, starts, ends, order


def user_zscore(x, starts, ends):
    """Per-user z-score of a score vector, grouped by [starts, ends)."""
    out = np.empty_like(x, dtype=np.float64)
    for s, e in zip(starts, ends):
        seg = x[s:e]
        sd = seg.std()
        out[s:e] = 0.0 if sd < 1e-12 else (seg - seg.mean()) / sd
    return out


def per_user_logsumexp(parts, starts, ends):
    """logsumexp over a list of per-row arrays, computed inside each user."""
    stack = np.stack(parts, axis=0)                     # (F, N)
    out = np.empty(stack.shape[1], dtype=np.float64)
    for s, e in zip(starts, ends):
        blk = stack[:, s:e]
        m = blk.max(axis=0)
        out[s:e] = m + np.log(np.exp(blk - m).sum(axis=0))
    return out


# ---------------------------------------------------------------------------
# score formulas
# ---------------------------------------------------------------------------
def probs_matrix(df, f_list):
    """(N, F) per-factorization retrieval probability in the parquet's row order."""
    return np.stack([df[f'prob_{FLABEL[f]}'].values.astype(np.float64)
                     for f in f_list], axis=1)


def znll_matrix(df, f_list):
    """(N, F) validation-calibrated zNLL, and the (F,) scaler actually used."""
    cols, mus, sigmas = [], [], []
    for f in f_list:
        v = df[f'TotalNLL_{FLABEL[f]}'].values.astype(np.float64)
        mu, sd = df.attrs['scalers'][f]
        cols.append((v - mu) / sd)
        mus.append(mu); sigmas.append(sd)
    return np.stack(cols, axis=1), (np.array(mus), np.array(sigmas))


def score_uniform(P):
    return P.mean(axis=1)


def per_user_lse_rows(X):
    """Row-wise logsumexp over the last axis (no user grouping needed)."""
    m = X.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(X - m).sum(axis=1, keepdims=True))).ravel()


def score_lse_from_znll(Z):
    """logsumexp(-zNLL_F) over F, row-wise."""
    return per_user_lse_rows(-Z)


def score_hybrid(uni, lse, starts, ends, lam):
    return lam * user_zscore(uni, starts, ends) + (1.0 - lam) * user_zscore(lse, starts, ends)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rank_and_metrics(scores, items=None, target=None, starts=None, ends=None,
                     topk=TOPK, target_rank=None):
    """Rank each user's candidates by score and compute Recall/NDCG at k.

    Two modes:
      * `items` given  -> the target is located by matching the item id inside the
        user's union (the real, deployable ranking).
      * `target_rank` given -> a precomputed 1-based rank per user is used directly
        (used for training-time dev scoring, where the rank comes from the same
        items array anyway but recomputing it is wasteful).

    `target_rank` == 0 means "target not in the union" and contributes 0.

    Returns dict with metrics, per-user NDCG/Recall arrays, per-user hit flags and
    the 1-based target rank (0 = outside the union).
    """
    n_users = len(starts)
    ndcg = {k: np.zeros(n_users) for k in topk}
    recall = {k: np.zeros(n_users) for k in topk}
    out_rank = np.zeros(n_users, dtype=np.int64)
    in_union = np.zeros(n_users, dtype=bool)
    for u, (s, e) in enumerate(zip(starts, ends)):
        if target_rank is not None:
            r = int(target_rank[u])
        elif items is None:
            raise ValueError('rank_and_metrics needs either `items` or `target_rank`')
        else:
            sc = scores[s:e]
            it = items[s:e]
            order = np.argsort(-sc, kind='stable')
            ranked = it[order]
            pos = np.where(ranked == int(target[u]))[0]
            r = int(pos[0]) + 1 if len(pos) else 0
        if r <= 0:
            continue
        out_rank[u] = r
        in_union[u] = True
        for k in topk:
            if r <= k:
                recall[k][u] = 1.0
                ndcg[k][u] = 1.0 / np.log2(r + 1)
    metrics = {}
    for k in topk:
        metrics[f'Recall@{k}'] = float(recall[k].mean())
        metrics[f'NDCG@{k}'] = float(ndcg[k].mean())
    return {'metrics': metrics, 'ndcg': ndcg, 'recall': recall,
            'rank': out_rank, 'in_union': in_union,
            'ndcg20': ndcg[20], 'recall20': recall[20], 'hit20': recall[20] > 0}


def target_rank_from_scores(scores, target_pos_abs, starts, ends):
    """1-based rank of each user's target inside its own block; 0 when absent.

    `target_pos_abs` holds the ABSOLUTE row position of the target inside the
    concatenated block (-1 when the target is outside the union), which is the
    layout the gate-training blocks use.
    """
    out = np.zeros(len(starts), dtype=np.int64)
    for u, (s, e) in enumerate(zip(starts, ends)):
        tp = int(target_pos_abs[u])
        if tp < 0:
            continue
        sc = scores[s:e]
        local = tp - s
        if local < 0 or local >= len(sc):
            raise AssertionError(f'target row {tp} outside user block [{s},{e})')
        # candidates strictly better, plus earlier ties broken by row order
        better = int((sc > sc[local]).sum())
        equal_before = int((sc[:local] == sc[local]).sum())
        out[u] = better + equal_before + 1
    return out


def load_split(outdir, ds, pool, split):
    """Load a union (and, if present, its NLL file); attach the frozen scalers."""
    import pandas as pd
    d = Path(outdir) / ('validation' if split == 'valid' else 'test')
    up = d / f'{ds}_{pool}_candidate_union.parquet'
    npth = d / f'{ds}_{pool}_candidate_nll.parquet'
    df = pd.read_parquet(up)
    if npth.exists():
        nll = pd.read_parquet(npth)
        if not (len(df) == len(nll)
                and (df.user.values == nll.user.values).all()
                and (df.candidate_item.values == nll.candidate_item.values).all()):
            raise AssertionError(f'{ds}/{split}/{pool}: union and NLL not row-aligned')
        # only the likelihood columns come from the NLL file; the union parquet is
        # the authority for prob_/rank_/present_ (the NLL file carries copies for the
        # multi5 borrow path, and letting them overwrite would be a silent aliasing)
        for c in nll.columns:
            if c.startswith('TotalNLL_') or c.startswith('sid_len_'):
                df[c] = nll[c].values
    return df, up


def apply_scalers(df, scalers):
    """Attach the (frozen, validation-fitted) scalers to the dataframe's attrs."""
    df.attrs['scalers'] = {f: (float(mu), float(sd)) for f, (mu, sd) in scalers.items()}
    return df


def fit_scalers(df, f_list):
    """Fit mu/sigma per factorization on ALL rows of this (validation) frame."""
    out = {}
    for f in f_list:
        v = df[f'TotalNLL_{FLABEL[f]}'].values.astype(np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            raise AssertionError(f'no finite NLL for {f}')
        out[f] = (float(v.mean()), float(v.std(ddof=0)))
    return out


def z_of(df, f_list):
    """(N, F) zNLL using the scalers in df.attrs."""
    cols = []
    for f in f_list:
        mu, sd = df.attrs['scalers'][f]
        cols.append((df[f'TotalNLL_{FLABEL[f]}'].values.astype(np.float64) - mu) / sd)
    return np.stack(cols, axis=1)


def rank_from_probs(df, f_list):
    """(N, F) per-factorization rank already stored by the union builder (>=1 or 0)."""
    return np.stack([np.where(df[f'present_{FLABEL[f]}'].values,
                              df[f'rank_{FLABEL[f]}'].values, 0)
                     for f in f_list], axis=1)
