#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART A — build the leakage-safe candidate-gate feature dataset.

Source: the frozen canonical VALID candidate unions and their cross-factorization NLL
(`outputs/candidate_conditioned/runs/<run>/{valid,test}/<dataset>_candidate_{union,nll}.parquet`),
plus the frozen NLL scaler and the previous round's user split.

WHAT IS NOT USED
----------------
* No target item id, target SID, target rank or hit flag enters the feature matrix.  The
  target is used for exactly two things, both outside the feature build: to mark which
  users have the target inside the union (so the listwise loss has a positive), and in
  evaluation-only diagnostics.
* No test statistic is computed here at all; this script can be run on `valid` only, and
  the test features are built later from the same code with the split switched.
* The Only-F1 / Only-F2 / Both group labels are **never** features or supervision.  They
  depend on the target and appear only in the evaluation section.

Usage
  python3 dg_features.py --split valid --outdir outputs/disagreement_gate
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'candidate_conditioned'),
           str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_common import DATASETS, F1, F2, SHORT  # noqa: E402

CC_RUN = ('outputs/candidate_conditioned/runs/'
          'multi_candidate_nll_seed2025_20260914T194657')
SCALER = 'outputs/candidate_conditioned/validation/candidate_nll_scaler.json'
SPLIT_CSV = 'outputs/adaptive_factorization/selector/router_user_split.csv'
ABSENT_RANK = 31          # prompt section 11: absent from a beam -> rank 31, P = 0

# ---------------------------------------------------------------------------
# the complete, closed feature schema
# ---------------------------------------------------------------------------
RETRIEVAL = ['present_F1', 'present_F2', 'rank_F1', 'rank_F2',
             'normalized_rank_F1', 'normalized_rank_F2', 'abs_rank_difference',
             'P_F1', 'P_F2', 'P_diff', 'abs_P_diff', 'P_max', 'P_min', 'UniformScore']
LIKELIHOOD = ['zNLL_F1', 'zNLL_F2', 'Delta_zNLL', 'abs_Delta_zNLL',
              'min_zNLL', 'max_zNLL', 'LSE_score', 'MinNLL_score']
DISAGREEMENT = ['only_F1', 'only_F2', 'both_present', 'rank_sign_disagreement',
                'one_factorization_top5', 'one_factorization_top10',
                'one_factorization_top20', 'both_top5', 'both_top10', 'both_top20']
USER_LEVEL = ['u_candidate_jaccard', 'u_intersection_size', 'u_union_size',
              'u_top1_agreement', 'u_top5_jaccard', 'u_top10_jaccard',
              'u_top20_jaccard', 'u_rank_correlation', 'u_entropy_F1', 'u_entropy_F2']
DATASET_ONEHOT = [f'ds_{SHORT[d].replace(" ", "_")}' for d in DATASETS]
FEATURES = RETRIEVAL + LIKELIHOOD + DISAGREEMENT + USER_LEVEL + DATASET_ONEHOT

# features that need NLL scoring to exist (reported in FEATURE_AUDIT.md)
NEEDS_NLL = set(LIKELIHOOD) | {'LSE_score'}


def load_split_roles():
    """user -> role, and the (dataset, user) -> role map.  CSV stores users as strings."""
    roles = {}
    with open(SPLIT_CSV) as fh:
        for r in csv.DictReader(fh):
            roles[(r['dataset'], int(r['user']))] = r['split']
    return roles


def _entropy(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum()) if len(q) else 0.0


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def per_user_features(items, p1, p2, r1, r2, z1, z2):
    """All per-candidate features for one user's candidate union."""
    n = len(items)
    present1 = items == items  # placeholder replaced below
    # presence is encoded as rank >= 1 in the frozen union file
    present1 = r1 >= 1
    present2 = r2 >= 1
    rank1 = np.where(present1, r1, ABSENT_RANK).astype(np.float64)
    rank2 = np.where(present2, r2, ABSENT_RANK).astype(np.float64)
    nunion = float(n)
    norm1 = (rank1 - 1.0) / max(nunion - 1.0, 1.0)
    norm2 = (rank2 - 1.0) / max(nunion - 1.0, 1.0)

    uscore = 0.5 * p1 + 0.5 * p2
    lse = _lse(z1, z2)

    # ---- user-level disagreement context (constants within the user) ----
    s1 = set(items[present1].tolist()); s2 = set(items[present2].tolist())
    t1 = set(items[np.argsort(-p1, kind='stable')[:1]].tolist())
    t2 = set(items[np.argsort(-p2, kind='stable')[:1]].tolist())

    def topk(p, k):
        idx = np.argsort(-p, kind='stable')[:k]
        return set(items[idx][p[idx] > 0].tolist())

    u = {
        'u_candidate_jaccard': _jaccard(s1, s2),
        'u_intersection_size': float(len(s1 & s2)),
        'u_union_size': nunion,
        'u_top1_agreement': float(len(t1 & t2) > 0),
        'u_top5_jaccard': _jaccard(topk(p1, 5), topk(p2, 5)),
        'u_top10_jaccard': _jaccard(topk(p1, 10), topk(p2, 10)),
        'u_top20_jaccard': _jaccard(topk(p1, 20), topk(p2, 20)),
        'u_rank_correlation': _rank_corr(p1, p2),
        'u_entropy_F1': _entropy(p1),
        'u_entropy_F2': _entropy(p2),
    }

    both = present1 & present2
    one5 = (present1 & (rank1 <= 5)) | (present2 & (rank2 <= 5))
    one10 = (present1 & (rank1 <= 10)) | (present2 & (rank2 <= 10))
    one20 = (present1 & (rank1 <= 20)) | (present2 & (rank2 <= 20))
    b5 = (rank1 <= 5) & (rank2 <= 5)
    b10 = (rank1 <= 10) & (rank2 <= 10)
    b20 = (rank1 <= 20) & (rank2 <= 20)

    cols = {
        'present_F1': present1.astype(np.float64),
        'present_F2': present2.astype(np.float64),
        'rank_F1': rank1, 'rank_F2': rank2,
        'normalized_rank_F1': norm1, 'normalized_rank_F2': norm2,
        'abs_rank_difference': np.abs(rank1 - rank2),
        'P_F1': p1, 'P_F2': p2,
        'P_diff': p1 - p2, 'abs_P_diff': np.abs(p1 - p2),
        'P_max': np.maximum(p1, p2), 'P_min': np.minimum(p1, p2),
        'UniformScore': uscore,
        'zNLL_F1': z1, 'zNLL_F2': z2,
        'Delta_zNLL': z1 - z2, 'abs_Delta_zNLL': np.abs(z1 - z2),
        'min_zNLL': np.minimum(z1, z2), 'max_zNLL': np.maximum(z1, z2),
        'LSE_score': lse, 'MinNLL_score': -np.minimum(z1, z2),
        'only_F1': (present1 & ~present2).astype(np.float64),
        'only_F2': (present2 & ~present1).astype(np.float64),
        'both_present': both.astype(np.float64),
        'rank_sign_disagreement': ((rank1 > ABSENT_RANK - 1) != (rank2 > ABSENT_RANK - 1)
                                   ).astype(np.float64),
        'one_factorization_top5': one5.astype(np.float64),
        'one_factorization_top10': one10.astype(np.float64),
        'one_factorization_top20': one20.astype(np.float64),
        'both_top5': b5.astype(np.float64),
        'both_top10': b10.astype(np.float64),
        'both_top20': b20.astype(np.float64),
    }
    for k, v in u.items():
        cols[k] = np.full(n, v, dtype=np.float64)
    return cols, uscore, lse


def _lse(z1, z2):
    m = np.maximum(-z1, -z2)
    return m + np.log(np.exp(-z1 - m) + np.exp(-z2 - m))


def _rank_corr(p1, p2):
    """Spearman over items present in BOTH factorizations, 0 if fewer than 3."""
    m = (p1 > 0) & (p2 > 0)
    if m.sum() < 3:
        return 0.0
    a = _rankdata(p1[m]); b = _rankdata(p2[m])
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _rankdata(x):
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return r


def zscore_by_user(x, starts, ends):
    out = np.empty_like(x)
    for s, e in zip(starts, ends):
        seg = x[s:e]
        mu = seg.mean(); sd = seg.std(ddof=0)
        out[s:e] = (seg - mu) / sd if sd > 1e-8 else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--outdir', default='outputs/disagreement_gate')
    ap.add_argument('--cc-run', default=CC_RUN)
    a = ap.parse_args(argv)

    import pandas as pd
    out = Path(a.outdir)
    featdir = out / 'features'
    featdir.mkdir(parents=True, exist_ok=True)
    roles = load_split_roles()
    scaler = json.loads(Path(SCALER).read_text())
    cols_all, meta = {}, []
    t0 = time.time()

    for di, ds in enumerate(DATASETS):
        d = Path(a.cc_run) / a.split
        u = pd.read_parquet(d / f'{ds}_candidate_union.parquet')
        nn = pd.read_parquet(d / f'{ds}_candidate_nll.parquet')
        if not (len(u) == len(nn) and (u.user.values == nn.user.values).all()
                and (u.candidate_item.values == nn.candidate_item.values).all()):
            raise AssertionError(f'{ds}/{a.split}: union and NLL not row-aligned')

        sc = scaler[ds]
        z1_all = (nn.TotalNLL_F1.values - sc['mu_F1']) / sc['sigma_F1']
        z2_all = (nn.TotalNLL_F2.values - sc['mu_F2']) / sc['sigma_F2']

        users = np.unique(u.user.values)
        starts = np.searchsorted(u.user.values, users)
        ends = np.append(starts[1:], len(u))
        parts = {k: [] for k in FEATURES}
        for k in ('role', 'user_idx', 'is_union_member'):
            parts[k] = []
        U_all, L_all = [], []
        for ui, (s, e) in enumerate(zip(starts, ends)):
            items = u.candidate_item.values[s:e].astype(np.int64)
            p1 = u.retrieval_score_F1.values[s:e].astype(np.float64)
            p2 = u.retrieval_score_F2.values[s:e].astype(np.float64)
            r1 = u.rank_F1_if_present.values[s:e].astype(np.int64)
            r2 = u.rank_F2_if_present.values[s:e].astype(np.int64)
            c, usc, lse = per_user_features(items, p1, p2, r1, r2,
                                            z1_all[s:e], z2_all[s:e])
            for k in FEATURES:
                parts[k].append(c[k] if k in c else np.zeros(e - s))
            for k in ('role', 'user_idx', 'is_union_member'):
                parts[k].append(np.full(e - s, ui))
            U_all.append(usc); L_all.append(lse)
        user = np.concatenate(parts['user_idx']).astype(np.int64)
        role = np.array([roles.get((ds, int(users[i])), 'UNASSIGNED') for i in user])
        # dataset one-hot
        for j, _d2 in enumerate(DATASETS):
            # one constant per user, length = that user's candidate count
            parts[DATASET_ONEHOT[j]] = [np.full(int(ends[i2] - starts[i2]), float(j == di))
                                        for i2 in range(len(users))]
        mat = np.column_stack([np.concatenate(parts[k]) for k in FEATURES]).astype(np.float64)
        if not np.isfinite(mat).all():
            bad = [FEATURES[j] for j in np.where(~np.isfinite(mat).any(axis=0))[0]]
            raise AssertionError(f'{ds}: non-finite features {bad}')

        user_ids = users[user]
        np.savez_compressed(
            featdir / f'{ds}_{a.split}_gate_features.npz',
            X=mat, features=np.array(FEATURES), user=user_ids, role=role,
            dataset=np.array([ds] * len(user)),
            U=np.concatenate(U_all), L=np.concatenate(L_all),
            cand_item=u.candidate_item.values.astype(np.int64),
            # frozen per-factorization candidate ranks, carried for EVALUATION-ONLY
            # subgroup definition (top-20 hit semantics); never a gate feature.
            cand_rank1=u.rank_F1_if_present.values.astype(np.int64),
            cand_rank2=u.rank_F2_if_present.values.astype(np.int64),
            starts=starts.astype(np.int64), ends=ends.astype(np.int64),
            users=users.astype(np.int64),
            # evaluation-only: the union row of the user's target (0-based SID row).
            # This is an .npz, not parquet, and the target column is never a feature.
            z_target=np.asarray(
                np.load(Path('outputs/adaptive_factorization/selector/features') /
                        f'{ds}_{a.split}.npz', allow_pickle=True)['cand_target'],
                dtype=np.int64))
        meta.append({'dataset': ds, 'rows': int(mat.shape[0]), 'users': int(len(users)),
                     'features': len(FEATURES),
                     'role_counts': {r: int((role == r).sum()) for r in
                                     ('RouterTrain', 'RouterDev', 'RouterHoldout',
                                      'UNASSIGNED')}})
        print(f'  {SHORT[ds]:12s} rows={mat.shape[0]:>9d} users={len(users):>6d} '
              f'features={mat.shape[1]}', flush=True)
        del u, nn, mat

    json.dump({'split': a.split, 'features': FEATURES, 'needs_nll': sorted(NEEDS_NLL),
               'cells': meta, 'seconds': round(time.time() - t0, 1),
               'target_used_for_features': False,
               'group_labels_used_for_features': False},
              open(featdir / f'FEATURE_BUILD_{a.split}.json', 'w'), indent=2)
    print(f'wrote {len(FEATURES)} features x {sum(m["rows"] for m in meta)} rows '
          f'to {featdir} ({time.time()-t0:.1f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
