#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE G + H — the learned multi-factorization candidate gate.

Design frozen before training (prompt sections 32-37):

  per-(user, candidate, factorization) features -> ONE shared MLP -> one logit per
  factorization -> softmax ACROSS factorizations, so sum_F w_F(u,i) = 1.

  primary   Score_gate(u,i)       = sum_F w_F(u,i) * P_F(u,i)
  secondary Score_gate_likelihood = sum_F w_F(u,i) * (-zNLL_F(u,i))

  loss      listwise: L = -log softmax_i Score(u,i) [target]
  optimiser AdamW, lr 1e-3, weight decay 1e-4, max 100 epochs, patience 10
  selection RouterDev NDCG@20 (pooled over the four development datasets)

The base recommenders and every SID tokenizer stay frozen: this file only consumes
frozen arrays (beam probabilities, candidate NLL, ranks) and trains the gate.

Feature discipline: every feature below is computed from generation output and
candidate likelihood only.  The target never enters `build_features`; it enters
only as the listwise label, and only for RouterTrain / RouterDev users.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import FLABEL  # noqa: E402

EMB_DIM = 6              # <= 8 as required by prompt section 34
PER_F_FEATURES = [
    'zNLL_F', 'retrieval_probability_F', 'rank_F', 'present_in_beam_F',
    'log1p_retrieval_probability_F', 'factorisation_sid_length_centred',
    'factorisation_embedding(learned)',
]
CAND_AGG_FEATURES = [
    'min_zNLL', 'mean_zNLL', 'std_zNLL', 'max_zNLL',
    'rank_disagreement', 'n_factorisations_containing_candidate',
    'retrieval_score_mean', 'retrieval_score_std', 'retrieval_score_max',
    'mean_rank', 'min_rank', 'present_count',
]
D_IN = len(PER_F_FEATURES) - 1 + EMB_DIM + len(CAND_AGG_FEATURES)


def factorization_embeddings(n_f):
    """Deterministic initialisation for the learned factorization embedding."""
    e = np.zeros((n_f, EMB_DIM), dtype=np.float32)
    for j in range(n_f):
        e[j, j % EMB_DIM] = 1.0
        e[j, (j + 3) % EMB_DIM] = 0.5
    return e


def build_features(df, f_list, Z, P, present, R):
    """(N, F, D_in) float32 feature tensor.  Target-free by construction."""
    n = len(df)
    F = len(f_list)
    L = np.array([int(df[f'sid_len_{FLABEL[f]}'].iloc[0]) for f in f_list], np.float64)
    Zp = np.where(present, Z, np.nan)
    Rp = np.where(present, R.astype(np.float64), np.nan)
    with np.errstate(all='ignore'), np.testing.suppress_warnings() as _sw:
        _sw.filter(RuntimeWarning)
        agg = {
            'min_zNLL': np.nanmin(Zp, axis=1),
            'mean_zNLL': np.nanmean(Zp, axis=1),
            'std_zNLL': np.nanstd(Zp, axis=1),
            'max_zNLL': np.nanmax(Zp, axis=1),
            'rank_disagreement': np.nanstd(Rp, axis=1),
            'n_factorisations_containing_candidate': present.sum(axis=1).astype(float),
            'retrieval_score_mean': P.mean(axis=1),
            'retrieval_score_std': P.std(axis=1),
            'retrieval_score_max': P.max(axis=1),
            'mean_rank': np.nanmean(Rp, axis=1),
            'min_rank': np.nanmin(Rp, axis=1),
            'present_count': present.sum(axis=1).astype(float),
        }
    for k in list(agg):
        agg[k] = np.nan_to_num(agg[k], nan=0.0, posinf=0.0, neginf=0.0)

    emb = factorization_embeddings(F)
    X = np.zeros((n, F, D_IN), dtype=np.float32)
    for j in range(F):
        n = Z.shape[0]
        cols = [Z[:, j], P[:, j], R[:, j].astype(float), present[:, j].astype(float),
                np.log1p(np.clip(P[:, j], 0, None)), np.full(n, L[j] - L.mean())]
        cols.extend(np.full(n, float(v)) for v in emb[j])
        cols.extend([agg[k] for k in CAND_AGG_FEATURES])
        X[:, j, :] = np.stack(cols, axis=1)
    return X


def feature_names():
    return (PER_F_FEATURES + CAND_AGG_FEATURES,
            f'{D_IN} = {len(PER_F_FEATURES) - 1} scalar per-factorization features '
            f'+ {EMB_DIM}-d learned factorization embedding '
            f'+ {len(CAND_AGG_FEATURES)} candidate-level aggregates')


def torch_module():
    import torch
    import torch.nn as nn

    class Gate(nn.Module):
        """Shared MLP over per-factorization features -> one logit per factorization.

        Linear(d, 64) -> GELU -> Dropout(0.1) -> Linear(64, 32) -> GELU ->
        Dropout(0.1) -> Linear(32, 1), applied identically to every factorization,
        then a softmax across factorizations.
        """

        def __init__(self, d_in, n_f, dropout=0.1):
            super().__init__()
            self.n_f = n_f
            self.emb = nn.Parameter(torch.tensor(factorization_embeddings(n_f)))
            self.net = nn.Sequential(
                nn.Linear(d_in + EMB_DIM, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(32, 1))

        def forward(self, x):
            B, F, _ = x.shape
            e = self.emb.unsqueeze(0).expand(B, F, self.emb.shape[1])
            logits = self.net(torch.cat([x, e], dim=2)).squeeze(-1)
            return torch.softmax(logits, dim=1)

    return Gate


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
