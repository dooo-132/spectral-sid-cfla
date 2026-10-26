#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE C — Instance heterogeneity.

Which instances does each granularity succeed on, and do the internal features of
those instances differ? Descriptive only.

HARD CONSTRAINT: only recommendation-pipeline-internal quantities are used. No
ASIN, category, title, price, description or any external text is read or joined.
This matters because the canonical row order in Beauty and Toys is not the
identity on item ids (see docs/ITEM_ID_CONVENTION.md), so external joins would
silently mis-pair rows.

Features per test instance (all internal):
  user side  : history_length, unique_item_count, history_item_popularity_mean,
               history_item_popularity_std, history_diversity (mean pairwise
               cosine DISTANCE of the history embeddings)
  target side: target_popularity (train interaction count), target_novelty
               (1 - max cosine similarity to any history item),
               target_to_history_max_cos, target_to_history_mean_cos

Success set S(u) = {K : target in Top20_K}. Best-K = the K with the smallest
target rank (ties reported separately).

No selector or gating network is trained.

Outputs -> outputs/mechanism/phaseC_heterogeneity/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.per_user import (                                       # noqa: E402
    DATASETS, K_LIST, build_view, canonicalize_item_ids, dump_path,
)
from src.utils import repo_paths as rp                                    # noqa: E402

OUT = REPO / 'outputs' / 'mechanism' / 'phaseC_heterogeneity'
TOPN = 20


def load_embeddings(ds):
    """L2-normalised embeddings, ROW-indexed.

    Canonical item id i corresponds to embedding row i-1, so callers must index
    with `item_id - 1` (see docs/ITEM_ID_CONVENTION.md).
    """
    df = pd.read_parquet(rp.item_embedding(ds))
    X = np.stack(df['embedding'].values).astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn


def rows_of(ids, n_rows):
    """Map 1-based canonical item ids to embedding row indices; drop out-of-range."""
    a = np.asarray(ids, dtype=np.int64) - 1
    return a[(a >= 0) & (a < n_rows)]


def history_and_target(ds, split='test'):
    """Per-user history (1-based ids) and target, in pipeline row order."""
    df = pd.read_parquet(rp.ARTIFACT_ROOT / 'data' / ds / f'{split}.parquet')
    hist = [list(map(int, h)) for h in df['history']]
    tgt = [int(t) for t in df['target']]
    return hist, np.array(tgt, dtype=np.int64)


def train_popularity(ds):
    df = pd.read_parquet(rp.ARTIFACT_ROOT / 'data' / ds / 'train.parquet')
    n_items = np.load(rp.ARTIFACT_ROOT / 'data' / ds /
                      f'{ds}_t5_spectral_nonneg_pearson_v3_k2_sid3.npy').shape[0] \
        if (rp.ARTIFACT_ROOT / 'data' / ds /
            f'{ds}_t5_spectral_nonneg_pearson_v3_k2_sid3.npy').exists() else None
    if n_items is None:
        n_items = int(max(max(h) for h in df['history']))
    cnt = np.zeros(n_items + 2, dtype=np.int64)
    for h in df['history']:
        for it in h:
            if 1 <= it < len(cnt):
                cnt[it] += 1
    for t in df['target']:
        if 1 <= t < len(cnt):
            cnt[t] += 1
    return cnt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = {'phase': 'C_instance_heterogeneity', 'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'external_metadata_used': False, 'topN': TOPN,
            'note': 'internal quantities only; no ASIN/category/text', 'inputs': {}}
    feat_rows, bestk_rows, summary_rows = [], [], []

    for ds in DATASETS:
        print(f'=== {ds} ===', flush=True)
        Xn = load_embeddings(ds)
        hist, tgt_raw = history_and_target(ds)
        pop = train_popularity(ds)
        view = build_view(ds)
        ks = view['ks']
        n = view['n_users']
        prov['inputs'][ds] = {str(k): {'path': str(view['models'][k]['path'].relative_to(rp.ARTIFACT_ROOT)),
                                       'sha256': view['models'][k]['sha256']} for k in ks}

        # canonical targets from the dumps (identical across K, already verified)
        tgt = view['target']
        # `build_view` already canonicalises the dumped (0-based) targets to the
        # 1-based dataset convention, so the canonical target must equal the
        # parquet target directly. (Canonicalising a second time would shift it.)
        assert np.array_equal(tgt, tgt_raw), (
            f'{ds}: parquet target order does not match the beam-dump order')
        assert tgt.min() >= 1, f'{ds}: canonical targets are not 1-based'

        ranks = {k: view['models'][k]['ranks'] for k in ks}
        for i in range(n):
            h = hist[i]
            # user-side
            hl = len(h)
            uq = len(set(h))
            pops = np.array([pop[x] if 0 <= x < len(pop) else 0 for x in h], dtype=np.float64)
            hr = rows_of(h, Xn.shape[0])
            if hr.size >= 2:
                H = Xn[hr]
                C = H @ H.T
                iu = np.triu_indices(hr.size, k=1)
                div = float(1.0 - C[iu].mean())
            else:
                div = 0.0
            # target-side
            t = int(tgt[i])
            tp = pop[t] if 0 <= t < len(pop) else 0
            if hr.size > 0 and 1 <= t <= Xn.shape[0]:
                sims = Xn[hr] @ Xn[t - 1]
                tmax = float(sims.max()); tmean = float(sims.mean())
            else:
                tmax = tmean = float('nan')
            # success set
            hits = {k: bool((ranks[k][i] >= 1) and (ranks[k][i] <= TOPN)) for k in ks}
            S = [k for k in ks if hits[k]]
            allk = all(hits.values()); nonek = not any(hits.values())
            ones = sum(hits.values())
            cat = ('No K' if nonek else 'All K' if allk else
                   'Multiple K' if ones > 1 else f'Only K{S[0]}')
            # best K by smallest rank among those that hit
            cand = [(int(ranks[k][i]), k) for k in ks if ranks[k][i] >= 1]
            if cand:
                mr = min(c for c, _ in cand)
                winners = [k for c, k in cand if c == mr]
                best_k = min(winners)
                tie = len(winners) > 1
            else:
                best_k = -1; tie = False
            feat_rows.append({
                'dataset': ds, 'user_index': i, 'target': t,
                'history_length': hl, 'unique_item_count': uq,
                'history_item_popularity_mean': float(pops.mean()) if pops.size else float('nan'),
                'history_item_popularity_std': float(pops.std()) if pops.size else float('nan'),
                'history_diversity': div,
                'target_popularity': int(tp),
                'target_novelty': (1.0 - tmax) if np.isfinite(tmax) else float('nan'),
                'target_to_history_max_cos': tmax,
                'target_to_history_mean_cos': tmean,
                'n_hit_K': ones, 'success_set': ','.join(map(str, S)),
                'success_category': cat, 'all_K': allk, 'no_K': nonek,
                'best_K': best_k, 'best_K_tie': tie,
                **{f'hit_K{k}': int(hits[k]) for k in ks},
                **{f'rank_K{k}': int(ranks[k][i]) for k in ks},
            })
        print(f'  {ds}: features for {n} instances', flush=True)

        df = pd.DataFrame([r for r in feat_rows if r['dataset'] == ds])
        feats = ['history_length', 'unique_item_count', 'history_item_popularity_mean',
                 'history_item_popularity_std', 'history_diversity',
                 'target_popularity', 'target_novelty',
                 'target_to_history_max_cos', 'target_to_history_mean_cos']
        for cat in ['Only K2', 'Only K3', 'Only K4', 'Only K5',
                    'Multiple K', 'All K', 'No K']:
            sub = df[df.success_category == cat]
            row = {'dataset': ds, 'group': cat, 'n_instances': len(sub),
                   'pct_instances': 100.0 * len(sub) / n}
            for f in feats:
                v = sub[f].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                row[f'{f}_mean'] = float(v.mean()) if v.size else float('nan')
                row[f'{f}_median'] = float(np.median(v)) if v.size else float('nan')
                row[f'{f}_IQR'] = (float(np.percentile(v, 75) - np.percentile(v, 25))
                                   if v.size else float('nan'))
            summary_rows.append(row)
        # best-K summary + tie rate
        tie_rate = float(df['best_K_tie'].mean())
        for bk in list(ks) + [-1]:
            sub = df[df.best_K == bk]
            row = {'dataset': ds, 'best_K': bk, 'n_instances': len(sub),
                   'pct_instances': 100.0 * len(sub) / n,
                   'best_K_tie_rate_overall': tie_rate}
            for f in feats:
                v = sub[f].to_numpy(dtype=float); v = v[np.isfinite(v)]
                row[f'{f}_median'] = float(np.median(v)) if v.size else float('nan')
            bestk_rows.append(row)
        summary_rows.append({'dataset': ds, 'group': '__tie_rate__',
                             'n_instances': int(df['best_K_tie'].sum()),
                             'pct_instances': 100.0 * tie_rate})
        print(f'  best-K tie rate = {tie_rate:.4f}', flush=True)

    pd.DataFrame(feat_rows).to_parquet(
        OUT / 'instance_granularity_features.parquet', index=False)
    pd.DataFrame(feat_rows).to_csv(
        OUT / 'instance_granularity_features.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / 'success_group_feature_summary.csv', index=False)
    pd.DataFrame(bestk_rows).to_csv(OUT / 'best_k_feature_summary.csv', index=False)
    print('  wrote instance_granularity_features.{csv,parquet}, '
          'success_group_feature_summary.csv, best_k_feature_summary.csv')

    import subprocess
    prov['git_sha'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                     capture_output=True, text=True).stdout.strip()
    (OUT / '_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
