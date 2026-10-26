#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE D -- controlled THREE-MODEL diversity study.

  G-triplets: same partition family, three different K        C(4,3) per family
  P-triplets: same K, the three partition families
  S-triplets: same family and K, three different training seeds (where present)

Same rules as the two-model study: uniform item-level probability averaging, no
tuned weights, identical beam budget.  Every G-triplet is reported (never only
the test-best one); the mean and range across G-triplets are also emitted so the
G-vs-P comparison cannot be driven by a lucky subset.

Output: three_model_diversity_results.csv
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import (  # noqa: E402
    DATASETS, FAMILY_LABELS, K_LIST, OUT, SEED_CANONICAL, load_cell, model_name,
    pairwise_diversity, paired_delta_ci, seed_beam_path, summarize, union_fuse,
)
from fg_pairs import available_seeds  # noqa: E402


def set_row(ds, dtype, cells, label):
    names = '+'.join(model_name(c['family'], c['k'], c['seed']) for c in cells)
    singles = {}
    for c in cells:
        r = union_fuse([c])
        singles[model_name(c['family'], c['k'], c['seed'])] = (r, summarize(r['rank']))
    rAB = union_fuse(cells)
    sF = summarize(rAB['rank'])

    best_name = max(singles, key=lambda n: singles[n][1]['NDCG@20'])
    best_rank = singles[best_name][0]['rank']
    best_s = singles[best_name][1]

    stack = np.stack([v[0]['rank'] for v in singles.values()], axis=0)
    pos = np.where(stack >= 1, stack, 10 ** 9)
    orank = pos.min(axis=0)
    ok = orank < 10 ** 9
    orank_full = np.full(orank.shape, -1, dtype=np.int64)
    orank_full[ok] = orank[ok]
    sO = summarize(orank_full)

    fusion_gain = sF['NDCG@20'] - best_s['NDCG@20']
    oracle_gain = sO['NDCG@20'] - best_s['NDCG@20']

    divs = []
    ranks = {n: v[0]['rank'] for n, v in singles.items()}
    for a, b in itertools.combinations(sorted(ranks), 2):
        divs.append(pairwise_diversity(ranks[a], ranks[b], topn=20))

    def _ndcg20(rank):
        return np.where((rank >= 1) & (rank <= 20),
                        1.0 / np.log2(np.maximum(rank, 1) + 1), 0.0)

    ci = paired_delta_ci(_ndcg20(rAB['rank']), _ndcg20(best_rank))

    row = {
        'dataset': ds, 'diversity_type': dtype, 'model_set': names,
        'label': label, 'n_models': len(cells),
        'families': '+'.join(sorted({c['family'] for c in cells})),
        'Ks': '+'.join(str(x) for x in sorted({c['k'] for c in cells})),
        'seeds': '+'.join(str(x) for x in sorted({c['seed'] for c in cells})),
        'same_family': int(len({c['family'] for c in cells}) == 1),
        'same_K': int(len({c['k'] for c in cells}) == 1),
        'n_users': int(cells[0]['n_users']),
        'best_single_model': best_name,
        'best_single_ndcg5': best_s['NDCG@5'],
        'best_single_ndcg10': best_s['NDCG@10'],
        'best_single_ndcg20': best_s['NDCG@20'],
        'best_single_recall20': best_s['Recall@20'],
        'fusion_ndcg5': sF['NDCG@5'], 'fusion_ndcg10': sF['NDCG@10'],
        'fusion_ndcg20': sF['NDCG@20'], 'fusion_recall20': sF['Recall@20'],
        'fusion_gain': fusion_gain,
        'fusion_gain_pct': 100.0 * fusion_gain / best_s['NDCG@20'] if best_s['NDCG@20'] else float('nan'),
        'oracle_ndcg20': sO['NDCG@20'], 'oracle_recall20': sO['Recall@20'],
        'oracle_gain': oracle_gain,
        'oracle_exploited_fraction': (fusion_gain / oracle_gain) if oracle_gain > 0 else float('nan'),
        'fusion_vs_best_ci_lo': ci['ci_lo'], 'fusion_vs_best_ci_hi': ci['ci_hi'],
        'fusion_vs_best_significant_95': int(ci['significant_95']),
        'mean_hit_disagreement': float(np.mean([d['hit_disagreement'] for d in divs])),
        'mean_hit_jaccard': float(np.mean([d['hit_jaccard'] for d in divs])),
        'mean_phi': float(np.mean([d['phi'] for d in divs])),
    }
    return row


def main(argv):
    only = None
    for a in argv:
        if a.startswith('--datasets='):
            only = a.split('=', 1)[1].split(',')
    datasets = only or DATASETS

    rows = []
    for ds in datasets:
        t0 = time.time()
        print(f'=== {ds} ===', flush=True)
        canon = {(f, k): load_cell(ds, f, k, 'test', SEED_CANONICAL)
                 for f in FAMILY_LABELS for k in K_LIST}

        # G-triplets
        for f in FAMILY_LABELS:
            for kk in itertools.combinations(K_LIST, 3):
                rows.append(set_row(ds, 'G', [canon[(f, k)] for k in kk],
                                    f'{f}:K' + '+'.join(map(str, kk))))
        print(f'  G triplets done ({len(FAMILY_LABELS) * 4})', flush=True)

        # P-triplets
        for k in K_LIST:
            rows.append(set_row(ds, 'P', [canon[(f, k)] for f in FAMILY_LABELS],
                                f'K{k}:all-families'))
        print(f'  P triplets done ({len(K_LIST)})', flush=True)

        # S-triplets
        n_s = 0
        for k in K_LIST:
            seeds = available_seeds(ds, 'spectral', k)
            if len(seeds) < 3:
                continue
            cells = {}
            for s in seeds:
                try:
                    cells[s] = load_cell(ds, 'spectral', k, 'test', s)
                except FileNotFoundError:
                    pass
            for ss in itertools.combinations(sorted(cells), 3):
                rows.append(set_row(ds, 'S', [cells[s] for s in ss],
                                    'seed' + '+'.join(map(str, ss))))
                n_s += 1
        print(f'  S triplets done ({n_s})  [{time.time()-t0:.0f}s]', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'three_model_diversity_results.csv', index=False)
    print(f'\nwrote three_model_diversity_results.csv ({len(df)} rows)')
    print(df.groupby('diversity_type')[
        ['fusion_gain', 'oracle_gain', 'oracle_exploited_fraction',
         'mean_hit_disagreement']].agg(['mean', 'min', 'max']).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
