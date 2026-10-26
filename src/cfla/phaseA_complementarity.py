#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE A — Error complementarity.

H1: different granularities are not simply better/worse versions of each other;
they have systematically different hit sets, so

    BestSingle  <  Uniform fusion  <  Oracle multi-granularity

Outputs (outputs/mechanism/phaseA_complementarity/):
    <dataset>/user_hit_matrix.parquet
    <dataset>/hit_pattern_distribution.csv
    <dataset>/pairwise_hit_complementarity.csv
    <dataset>/candidate_overlap.csv
    <dataset>/candidate_overlap_user_level.parquet
    <dataset>/oracle_gap.csv
    <dataset>/fusion_subset_results.csv
    phaseA_bootstrap.csv
    _PROVENANCE.json

Read-only with respect to all artifacts. No training. No test-set selection.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.per_user import (                                     # noqa: E402
    DATASETS, K_LIST, build_view, dump_path, fuse_subset, oracle_ranks,
    paired_bootstrap, per_user_ndcg_vector, rank_of, sha256_file,
    summarize_ranks, topn,
)

OUT = REPO / 'outputs' / 'mechanism' / 'phaseA_complementarity'
TOPN_LEVELS = (5, 10, 20, 30)


def hit_pattern(dist, target, n):
    return tuple(int(1 if target in set(topn(dist, n)) else 0) for _ in [0])


def pattern_bits(per_user, target, n, ks):
    t = int(target)
    return tuple(1 if rank_of(t, per_user[i]) in range(1, n + 1) else 0
                 for i, k in enumerate(ks))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = {'phase': 'A_error_complementarity', 'alignment_mode': 'canonical',
            'affinity': 'nonneg_pearson', 'split': 'test',
            'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'inputs': {}, 'datasets': {}}
    boot_rows, pat_rows, pair_rows, ov_rows, orc_rows, sub_rows = [], [], [], [], [], []

    for ds in DATASETS:
        print(f'=== {ds} ===')
        t0 = time.time()
        view = build_view(ds)
        ks = view['ks']
        n = view['n_users']
        tgt = view['target']
        models = view['models']
        per_user = {k: models[k]['per_user'] for k in ks}
        ranks = {k: models[k]['ranks'] for k in ks}
        prov['inputs'][ds] = {str(k): {'path': str(models[k]['path'].relative_to(REPO.parent)),
                                       'sha256': models[k]['sha256'],
                                       'n_users': int(n)} for k in ks}
        d_out = OUT / ds
        d_out.mkdir(parents=True, exist_ok=True)

        # ---------- 4. user-level hit matrix ----------
        cols = {'user_index': np.arange(n), 'target': tgt}
        for k in ks:
            for nn in TOPN_LEVELS:
                cols[f'hit_k{k}_{nn}'] = ((ranks[k] >= 1) & (ranks[k] <= nn)).astype(np.int8)
            cols[f'rank_k{k}'] = ranks[k]
        pd.DataFrame(cols).to_parquet(d_out / 'user_hit_matrix.parquet', index=False)
        print(f'  hit matrix written ({n} users x {len(cols)} cols)')

        # ---------- 5. hit-pattern distribution @20 ----------
        patterns = {}
        for i in range(n):
            bits = tuple(int((ranks[k][i] >= 1) and (ranks[k][i] <= 20)) for k in ks)
            patterns[bits] = patterns.get(bits, 0) + 1
        for bits in itertools.product([0, 1], repeat=len(ks)):
            c = patterns.get(bits, 0)
            ones = sum(bits)
            cat = ('No K' if ones == 0 else
                   'All K' if ones == len(ks) else
                   'Multiple K' if ones > 1 else
                   f'Only K{ks[bits.index(1)]}')
            pat_rows.append({'dataset': ds, 'pattern': ''.join(map(str, bits)),
                             'which': ','.join(str(k) for k, b in zip(ks, bits) if b),
                             'n_k': ones, 'category': cat,
                             'count': c, 'percentage': 100.0 * c / n})
        # category roll-up
        for cat in ['Only K2', 'Only K3', 'Only K4', 'Only K5',
                    'Multiple K', 'All K', 'No K']:
            c = sum(r['count'] for r in pat_rows
                    if r['dataset'] == ds and r['category'] == cat)
            pat_rows.append({'dataset': ds, 'pattern': '(category)',
                             'which': cat, 'n_k': None, 'category': cat,
                             'count': c, 'percentage': 100.0 * c / n})
        print(f"  hit patterns: all-K={patterns.get((1,)*len(ks),0)/n:.4f} "
              f"none={patterns.get((0,)*len(ks),0)/n:.4f}")

        # ---------- 6. pairwise complementarity ----------
        for a, b in itertools.combinations(ks, 2):
            for nn in TOPN_LEVELS:
                ha = (ranks[a] >= 1) & (ranks[a] <= nn)
                hb = (ranks[b] >= 1) & (ranks[b] <= nn)
                both = int((ha & hb).sum())
                only_a = int((ha & ~hb).sum())
                only_b = int((~ha & hb).sum())
                neither = int((~ha & ~hb).sum())
                union = both + only_a + only_b
                pa, pb = ha.mean(), hb.mean()
                # phi coefficient
                num = both * neither - only_a * only_b
                den = np.sqrt((both + only_a) * (only_b + neither) *
                              (both + only_b) * (only_a + neither))
                phi = float(num / den) if den > 0 else float('nan')
                pair_rows.append({
                    'dataset': ds, 'K_a': a, 'K_b': b, 'topN': nn,
                    'P_agree': (both + neither) / n, 'P_disagree': (only_a + only_b) / n,
                    'P_a1_b0': only_a / n, 'P_a0_b1': only_b / n,
                    'n_both': both, 'n_only_a': only_a, 'n_only_b': only_b,
                    'n_neither': neither,
                    'hit_a': pa, 'hit_b': pb,
                    'jaccard_hitset': both / union if union else float('nan'),
                    'phi': phi,
                })
        print('  pairwise complementarity done')

        # ---------- 7. candidate-set overlap ----------
        for a, b in itertools.combinations(ks, 2):
            for nn in TOPN_LEVELS:
                js, inter, ua, ub = [], [], [], []
                for i in range(n):
                    sa = set(topn(per_user[a][i], nn))
                    sb = set(topn(per_user[b][i], nn))
                    I = len(sa & sb)
                    U = len(sa | sb)
                    js.append(I / U if U else 0.0)
                    inter.append(I)
                    ua.append(len(sa - sb))
                    ub.append(len(sb - sa))
                ov_rows.append({
                    'dataset': ds, 'K_a': a, 'K_b': b, 'topN': nn,
                    'jaccard_mean': float(np.mean(js)),
                    'jaccard_median': float(np.median(js)),
                    'jaccard_p10': float(np.percentile(js, 10)),
                    'jaccard_p90': float(np.percentile(js, 90)),
                    'intersection_mean': float(np.mean(inter)),
                    'unique_from_a_mean': float(np.mean(ua)),
                    'unique_from_b_mean': float(np.mean(ub)),
                })
        # union size over all K, per user
        urows = {'user_index': np.arange(n)}
        for nn in TOPN_LEVELS:
            sizes = []
            for i in range(n):
                u = set()
                for k in ks:
                    u |= set(topn(per_user[k][i], nn))
                sizes.append(len(u))
            urows[f'union_size_top{nn}'] = np.array(sizes)
            urows[f'best_single_union_top{nn}'] = np.array(
                [max(len(set(topn(per_user[k][i], nn))) for k in ks) for i in range(n)])
        pd.DataFrame(urows).to_parquet(
            d_out / 'candidate_overlap_user_level.parquet', index=False)
        for nn in TOPN_LEVELS:
            s = urows[f'union_size_top{nn}']
            ov_rows.append({'dataset': ds, 'K_a': 'ALL', 'K_b': 'ALL', 'topN': nn,
                            'jaccard_mean': float('nan'), 'jaccard_median': float('nan'),
                            'jaccard_p10': float('nan'), 'jaccard_p90': float('nan'),
                            'intersection_mean': float('nan'),
                            'unique_from_a_mean': float('nan'),
                            'unique_from_b_mean': float('nan'),
                            'union_size_mean': float(s.mean()),
                            'union_size_median': float(np.median(s)),
                            'nominal_size': nn * len(ks)})
        print('  candidate overlap done')

        # ---------- 8. oracle upper bound ----------
        single = {k: summarize_ranks(ranks[k]) for k in ks}
        best_single_k = max(ks, key=lambda k: single[k]['NDCG@20'])
        fus_d, fus_r = fuse_subset(view, tuple(ks))
        oracle_r_full = oracle_ranks(view)
        oracle_r_top = {nn: oracle_ranks(view, topn_cap=nn) for nn in TOPN_LEVELS}
        fus_sum = summarize_ranks(fus_r)
        orc_sum = summarize_ranks(oracle_r_full)
        for nn in (5, 10, 20):
            bs = single[best_single_k][f'NDCG@{nn}']
            uf = fus_sum[f'NDCG@{nn}']
            orc = orc_sum[f'NDCG@{nn}']
            denom = orc - bs
            orc_rows.append({
                'dataset': ds, 'topN': nn, 'best_single_K': best_single_k,
                'BestSingle_NDCG': bs, 'Uniform_NDCG': uf, 'Oracle_NDCG': orc,
                'fusion_gain': uf - bs, 'remaining_oracle_gap': orc - uf,
                'exploited_fraction': (uf - bs) / denom if denom > 0 else float('nan'),
                'oracle_recall': orc_sum[f'Recall@{nn}'],
                'uniform_recall': fus_sum[f'Recall@{nn}'],
                'best_single_recall': single[best_single_k][f'Recall@{nn}'],
            })
        print(f"  best single K={best_single_k}  uniform N@20={fus_sum['NDCG@20']:.6f}  "
              f"oracle N@20={orc_sum['NDCG@20']:.6f}")

        # ---------- oracle vs uniform bootstrap ----------
        for nn in (10, 20):
            a = per_user_ndcg_vector(oracle_r_full, nn)
            b = per_user_ndcg_vector(fus_r, nn)
            r = paired_bootstrap(a, b)
            boot_rows.append({'dataset': ds, 'comparison': 'Oracle_vs_Uniform',
                              'topN': nn, 'a_value': float(a.mean()),
                              'b_value': float(b.mean()), **r,
                              'n_boot': 5000, 'n_users': n,
                              'significant_95': bool(r['ci_lo'] > 0 or r['ci_hi'] < 0)})
            c = per_user_ndcg_vector(ranks[best_single_k], nn)
            r2 = paired_bootstrap(b, c)
            boot_rows.append({'dataset': ds, 'comparison': 'Uniform_vs_BestSingle',
                              'topN': nn, 'a_value': float(b.mean()),
                              'b_value': float(c.mean()), **r2,
                              'n_boot': 5000, 'n_users': n,
                              'significant_95': bool(r2['ci_lo'] > 0 or r2['ci_hi'] < 0)})

        # ---------- 9. fusion subsets ----------
        subsets = [(k,) for k in ks]
        subsets += list(itertools.combinations(ks, 2))
        subsets += list(itertools.combinations(ks, 3))
        subsets += [tuple(ks)]
        for S in subsets:
            if len(S) == 1:
                r = ranks[S[0]]
            else:
                _, r = fuse_subset(view, S)
            s = summarize_ranks(r)
            sub_rows.append({'dataset': ds, 'subset': '+'.join(map(str, S)),
                             'n_models': len(S),
                             'is_full': tuple(S) == tuple(ks), **s})
        print(f'  fusion subsets done ({len(subsets)})')

        prov['datasets'][ds] = {'n_users': int(n), 'elapsed_s': round(time.time() - t0, 1)}

    # ---- write CSVs ----
    def w(name, rows, order=None):
        if not rows:
            return
        df = pd.DataFrame(rows)
        if order:
            df = df[order]
        df.to_csv(OUT / name, index=False)
        print(f'  wrote {name} ({len(df)} rows)')

    w('phaseA_bootstrap.csv', boot_rows)
    w('_hit_pattern_distribution.csv', pat_rows)
    w('_pairwise_hit_complementarity.csv', pair_rows)
    w('_candidate_overlap.csv', ov_rows)
    w('_oracle_gap.csv', orc_rows)
    w('_fusion_subset_results.csv', sub_rows)

    # per-dataset copies in the required per-dataset layout
    for ds in DATASETS:
        d_out = OUT / ds
        d_out.mkdir(parents=True, exist_ok=True)
        for src, dst in [('_hit_pattern_distribution.csv', 'hit_pattern_distribution.csv'),
                         ('_pairwise_hit_complementarity.csv', 'pairwise_hit_complementarity.csv'),
                         ('_candidate_overlap.csv', 'candidate_overlap.csv'),
                         ('_oracle_gap.csv', 'oracle_gap.csv'),
                         ('_fusion_subset_results.csv', 'fusion_subset_results.csv')]:
            df = pd.read_csv(OUT / src)
            df[df['dataset'] == ds].drop(columns=['dataset']).to_csv(d_out / dst, index=False)

    # ---- provenance ----
    import subprocess
    try:
        prov['git_sha'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                         capture_output=True, text=True).stdout.strip()
    except Exception:
        prov['git_sha'] = None
    prov['outputs'] = sorted(p.name for p in OUT.rglob('*') if p.is_file())
    (OUT / '_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print(f'\nwrote {OUT}/_PROVENANCE.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
