#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1 — Only-tokenizer NLL alignment.

Question: for users that ONLY one tokenizer retrieves, is that tokenizer also the
one under which the ground-truth target SID is cheaper to generate?

For each compared pair (A, B) and each dataset:
  * group means / medians of teacher-forced TotalNLL by Only-tokenizer group
  * per-user argmin accuracy: P(argmin_F TotalNLL_F == the unique hitting F)
  * permutation null: >= 1000 random permutations of the tokenizer labels within
    each user, holding the NLL vector fixed
  * random baseline 50%, and the "always pick the globally cheaper tokenizer"
    baseline
  * the same statistics on the validation-normalised z-score, so a difference in
    raw calibration between the two independently trained models cannot produce
    the result on its own

Outputs -> step1_cross_tokenizer/cross_tokenizer_nll_alignment.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, STEP1, TOKENIZERS  # noqa: E402

N_PERM = 1000
PERM_SEED = 20260912


def permutation_acc(argmin: np.ndarray, m: int, observed: float):
    """Null distribution of P(argmin == random label) over N_PERM permutations."""
    rng = np.random.default_rng(PERM_SEED)
    n = len(argmin)
    if n == 0:
        return float('nan'), float('nan')
    accs = np.empty(N_PERM)
    for i in range(N_PERM):
        lab = rng.integers(0, m, size=n)
        accs[i] = (lab == argmin).mean()
    return float(accs.mean()), float((accs >= observed).mean())


def align_pair(nll: pd.DataFrame, groups: pd.DataFrame, ds: str, ta: str, tb: str,
               tier: str, rows: list, use_z: bool = False):
    d = nll[(nll['dataset'] == ds) & (nll['tokenizer'].isin([ta, tb]))]
    col = 'z_total_nll' if use_z else 'total_nll'
    piv = d.pivot(index='user_index', columns='tokenizer', values=col)
    if ta not in piv.columns or tb not in piv.columns:
        return
    g = groups[groups['dataset'] == ds].set_index('user_index')
    piv = piv.join(g[['group']], how='inner')
    names = ['Only-Spectral', 'Only-RQVAE']
    label_for = {ta: names[0], tb: names[1]}
    mat = piv[[ta, tb]].to_numpy(dtype=float)
    argmin = mat.argmin(axis=1)
    grp = piv['group'].to_numpy()

    for gi, (tok, other) in enumerate(((ta, tb), (tb, ta))):
        sel = grp == label_for[tok]
        if sel.sum() == 0:
            continue
        own = mat[sel, gi]
        oth = mat[sel, 1 - gi]
        acc = float((argmin[sel] == gi).mean())
        null_mean, p = permutation_acc(argmin[sel], 2, acc)
        # within-user rank of THIS tokenizer's target NLL (1 = cheaper of the two)
        within_rank = np.where(own <= oth, 1.0, 2.0)
        rows.append({
            'dataset': ds, 'tier': tier, 'pair': f'{ta}|{tb}',
            'normalisation': 'z_valid' if use_z else 'raw',
            'group': label_for[tok], 'only_tokenizer': tok,
            'other_tokenizer': other,
            'n_users': int(sel.sum()),
            'mean_nll_own': float(own.mean()), 'mean_nll_other': float(oth.mean()),
            'median_nll_own': float(np.median(own)),
            'median_nll_other': float(np.median(oth)),
            'own_minus_other_mean': float(own.mean() - oth.mean()),
            'argmin_match_accuracy': acc,
            'frac_own_rank_1': float((within_rank == 1).mean()),
            'mean_within_user_rank_own': float(within_rank.mean()),
            'permutation_null_mean': null_mean,
            'permutation_p_value': p,
            'random_baseline': 0.5,
            'global_cheaper_tokenizer': TOKENIZERS[
                ta if mat[:, 0].mean() <= mat[:, 1].mean() else tb]['label'],
            'global_baseline_accuracy': float(
                (argmin[sel] == int(mat[:, 0].mean() > mat[:, 1].mean())).mean()),
        })

    # pooled over both Only groups
    sel_any = (grp == names[0]) | (grp == names[1])
    own_idx = np.where(grp == names[0], 0, 1)
    if sel_any.sum():
        acc = float((argmin[sel_any] == own_idx[sel_any]).mean())
        null_mean, p = permutation_acc(argmin[sel_any], 2, acc)
        rows.append({
            'dataset': ds, 'tier': tier, 'pair': f'{ta}|{tb}',
            'normalisation': 'z_valid' if use_z else 'raw',
            'group': 'BOTH_Only_groups', 'only_tokenizer': 'pooled',
            'other_tokenizer': 'pooled', 'n_users': int(sel_any.sum()),
            'mean_nll_own': float(mat[sel_any, own_idx[sel_any]].mean()),
            'mean_nll_other': float(mat[sel_any, 1 - own_idx[sel_any]].mean()),
            'median_nll_own': float(np.median(mat[sel_any, own_idx[sel_any]])),
            'median_nll_other': float(np.median(mat[sel_any, 1 - own_idx[sel_any]])),
            'own_minus_other_mean': float(
                mat[sel_any, own_idx[sel_any]].mean() -
                mat[sel_any, 1 - own_idx[sel_any]].mean()),
            'argmin_match_accuracy': acc,
            'frac_own_rank_1': float((within_rank == 1).mean()),
            'mean_within_user_rank_own': float(within_rank.mean()),
            'permutation_null_mean': null_mean,
            'permutation_p_value': p,
            'random_baseline': 0.5,
            'global_cheaper_tokenizer': TOKENIZERS[
                ta if mat[:, 0].mean() <= mat[:, 1].mean() else tb]['label'],
            'global_baseline_accuracy': float(
                (argmin[sel_any] == int(mat[:, 0].mean() > mat[:, 1].mean())).mean()),
        })

    # ---- §17 within-user rank over ALL test users, not only the Only groups ----
    r1 = mat[:, 0] <= mat[:, 1]          # tokenizer ta is the cheaper one for this user
    rows.append({
        'dataset': ds, 'tier': tier, 'pair': f'{ta}|{tb}',
        'normalisation': 'z_valid' if use_z else 'raw',
        'group': 'ALL_USERS', 'only_tokenizer': 'n/a', 'other_tokenizer': 'n/a',
        'n_users': int(len(r1)),
        'mean_nll_own': float(mat[:, 0].mean()),
        'mean_nll_other': float(mat[:, 1].mean()),
        'median_nll_own': float(np.median(mat[:, 0])),
        'median_nll_other': float(np.median(mat[:, 1])),
        'own_minus_other_mean': float(mat[:, 0].mean() - mat[:, 1].mean()),
        'argmin_match_accuracy': float(r1.mean()),
        'frac_own_rank_1': float(r1.mean()),
        'mean_within_user_rank_own': float(np.where(r1, 1.0, 2.0).mean()),
        'permutation_null_mean': 0.5,
        'permutation_p_value': None,
        'random_baseline': 0.5,
        'global_cheaper_tokenizer': TOKENIZERS[
            ta if mat[:, 0].mean() <= mat[:, 1].mean() else tb]['label'],
        'global_baseline_accuracy': float(
            (r1 == bool(mat[:, 0].mean() <= mat[:, 1].mean())).mean()),
        'note': 'within-user rank of the target NLL for every test user; '
                'frac_own_rank_1 = P(this tokenizer is the cheaper of the two)',
    })


def main():
    nll = pd.read_parquet(STEP1 / 'cross_tokenizer_target_nll.parquet')
    groups = pd.read_parquet(STEP1 / 'cross_tokenizer_only_groups.parquet')
    rows = []

    # The Only-tokenizer groups are defined for the primary pair, which is the
    # comparison the protocol asks for.  Secondary tokenizers are covered by the
    # retrieval tables, not by a re-defined "only" partition.
    have = {ds: set(nll[nll.dataset == ds]['tokenizer']) for ds in DATASETS}
    for ds in DATASETS:
        if {'spectral_k3', 'rqvae3'} - have[ds]:
            print(f'  {ds}: primary tokenizers missing from the NLL table, skipped')
            continue
        align_pair(nll, groups, ds, 'spectral_k3', 'rqvae3', 'PRIMARY', rows,
                   use_z=False)
        align_pair(nll, groups, ds, 'spectral_k3', 'rqvae3', 'PRIMARY', rows,
                   use_z=True)

    out = pd.DataFrame(rows)
    out.to_csv(STEP1 / 'cross_tokenizer_nll_alignment.csv', index=False)
    print(f'wrote cross_tokenizer_nll_alignment.csv ({len(out)} rows)')
    if len(out):
        p = out[(out.tier == 'PRIMARY') & (out.normalisation == 'raw')]
        print(p[['dataset', 'group', 'n_users', 'mean_nll_own', 'mean_nll_other',
                 'argmin_match_accuracy', 'permutation_null_mean',
                 'permutation_p_value', 'global_baseline_accuracy']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
