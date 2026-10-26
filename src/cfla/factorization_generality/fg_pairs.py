#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE C -- controlled TWO-MODEL diversity study.

Three axes, each with the same fusion rule (uniform item-level probability
averaging, no weights tuned) and the same beam budget (30 per model):

  G-diversity  same partition family, different K
  P-diversity  same K, different partition family
  S-diversity  same family, same K, different training seed

For every pair the script reports, for both single models and their uniform
fusion: NDCG@5/10/20 and Recall@5/10/20; the oracle over the pair; the fusion
gain over the stronger single; the oracle gap; the fraction of that gap the
fusion exploits; and the error-diversity statistics (hit disagreement, phi,
hit-set and candidate-set Jaccard, union size, and Spearman/Kendall ranking
correlation on the two models' COMMON candidate items).

Output: pairwise_diversity_results.csv
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
    DATASETS, FAMILIES, FAMILY_LABELS, K_LIST, OUT, SEED_CANONICAL,
    candidate_overlap, common_rank_correlation, load_cell, model_name,
    oracle_rank, paired_delta_ci, pairwise_diversity, seed_beam_path,
    summarize, union_fuse, SEED_ROOT,
)


def available_seeds(dataset, family, k):
    """Seeds with an existing TEST beam for this (dataset, family, K)."""
    seeds = [SEED_CANONICAL]
    if family != 'spectral' or not SEED_ROOT.exists():
        return seeds
    for ck in sorted((SEED_ROOT / 'ckpt').glob(f'tiger_{dataset}_k{k}_sid{k+1}_seed*.pth')):
        s = int(ck.stem.rsplit('_seed', 1)[1])
        if seed_beam_path(dataset, k, s, 'test').exists():
            seeds.append(s)
    return sorted(seeds)


def pair_row(ds, dtype, cA, cB, do_corr=True):
    rA = union_fuse([cA])
    rB = union_fuse([cB])
    rAB = union_fuse([cA, cB])
    sA, sB, sF = summarize(rA['rank']), summarize(rB['rank']), summarize(rAB['rank'])

    orank = oracle_rank([cA, cB])
    sO = summarize(orank)

    if sA['NDCG@20'] >= sB['NDCG@20']:
        best_name, best_s, best_rank = model_name(cA['family'], cA['k'], cA['seed']), sA, rA['rank']
    else:
        best_name, best_s, best_rank = model_name(cB['family'], cB['k'], cB['seed']), sB, rB['rank']

    fusion_gain = sF['NDCG@20'] - best_s['NDCG@20']
    oracle_gain = sO['NDCG@20'] - best_s['NDCG@20']
    exploited = fusion_gain / oracle_gain if oracle_gain > 0 else float('nan')

    div = pairwise_diversity(rA['rank'], rB['rank'], topn=20)
    ov = candidate_overlap(rA['top_items'], rA['top_items'] >= 0,
                           rB['top_items'], rB['top_items'] >= 0,
                           np.minimum(rA['n_valid_top'], 20),
                           np.minimum(rB['n_valid_top'], 20))

    def _ndcg20(rank):
        return np.where((rank >= 1) & (rank <= 20),
                        1.0 / np.log2(np.maximum(rank, 1) + 1), 0.0)

    ci = paired_delta_ci(_ndcg20(rAB['rank']), _ndcg20(best_rank))

    row = {
        'dataset': ds, 'diversity_type': dtype,
        'model_A': model_name(cA['family'], cA['k'], cA['seed']),
        'model_B': model_name(cB['family'], cB['k'], cB['seed']),
        'family_A': cA['family'], 'family_B': cB['family'],
        'K_A': cA['k'], 'K_B': cB['k'],
        'seed_A': cA['seed'], 'seed_B': cB['seed'],
        'same_family': int(cA['family'] == cB['family']),
        'same_K': int(cA['k'] == cB['k']),
        'same_seed': int(cA['seed'] == cB['seed']),
        'n_users': int(cA['n_users']),
        'single_A_ndcg5': sA['NDCG@5'], 'single_A_ndcg10': sA['NDCG@10'],
        'single_A_ndcg20': sA['NDCG@20'], 'single_A_recall20': sA['Recall@20'],
        'single_B_ndcg5': sB['NDCG@5'], 'single_B_ndcg10': sB['NDCG@10'],
        'single_B_ndcg20': sB['NDCG@20'], 'single_B_recall20': sB['Recall@20'],
        'best_single_model': best_name,
        'best_single_ndcg5': best_s['NDCG@5'],
        'best_single_ndcg10': best_s['NDCG@10'],
        'best_single_ndcg20': best_s['NDCG@20'],
        'best_single_recall20': best_s['Recall@20'],
        'fusion_ndcg5': sF['NDCG@5'], 'fusion_ndcg10': sF['NDCG@10'],
        'fusion_ndcg20': sF['NDCG@20'], 'fusion_recall5': sF['Recall@5'],
        'fusion_recall10': sF['Recall@10'], 'fusion_recall20': sF['Recall@20'],
        'fusion_gain': fusion_gain,
        'fusion_gain_pct': 100.0 * fusion_gain / best_s['NDCG@20'] if best_s['NDCG@20'] else float('nan'),
        'oracle_ndcg5': sO['NDCG@5'], 'oracle_ndcg10': sO['NDCG@10'],
        'oracle_ndcg20': sO['NDCG@20'], 'oracle_recall20': sO['Recall@20'],
        'oracle_gain': oracle_gain,
        'oracle_exploited_fraction': exploited,
        'fusion_vs_best_delta_se': ci['se'],
        'fusion_vs_best_ci_lo': ci['ci_lo'], 'fusion_vs_best_ci_hi': ci['ci_hi'],
        'fusion_vs_best_z': ci['z'],
        'fusion_vs_best_significant_95': int(ci['significant_95']),
    }
    row.update({k: v for k, v in div.items() if k != 'n_users'})
    row.update(ov)
    if do_corr:
        row.update(common_rank_correlation(cA, cB))
    return row


def main(argv):
    only = None
    do_corr = True
    for a in argv:
        if a.startswith('--datasets='):
            only = a.split('=', 1)[1].split(',')
        if a == '--no-corr':
            do_corr = False
    datasets = only or DATASETS

    rows = []
    for ds in datasets:
        t0 = time.time()
        print(f'=== {ds} ===', flush=True)
        canon = {(f, k): load_cell(ds, f, k, 'test', SEED_CANONICAL)
                 for f in FAMILY_LABELS for k in K_LIST}

        # ---------- G: same family, different K ----------
        for f in FAMILY_LABELS:
            for k1, k2 in itertools.combinations(K_LIST, 2):
                rows.append(pair_row(ds, 'G', canon[(f, k1)], canon[(f, k2)], do_corr))
        print(f'  G pairs done ({len(FAMILY_LABELS) * 6})', flush=True)

        # ---------- P: same K, different family ----------
        for k in K_LIST:
            for f1, f2 in itertools.combinations(FAMILY_LABELS, 2):
                rows.append(pair_row(ds, 'P', canon[(f1, k)], canon[(f2, k)], do_corr))
        print(f'  P pairs done ({len(K_LIST) * 3})', flush=True)

        # ---------- S: same family, same K, different seed ----------
        n_s = 0
        for k in K_LIST:
            seeds = available_seeds(ds, 'spectral', k)
            if len(seeds) < 2:
                continue
            cells = {}
            for s in seeds:
                try:
                    cells[s] = load_cell(ds, 'spectral', k, 'test', s)
                except FileNotFoundError:
                    pass
            for s1, s2 in itertools.combinations(sorted(cells), 2):
                rows.append(pair_row(ds, 'S', cells[s1], cells[s2], do_corr))
                n_s += 1
        print(f'  S pairs done ({n_s})  [{time.time()-t0:.0f}s]', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'pairwise_diversity_results.csv', index=False)
    print(f'\nwrote pairwise_diversity_results.csv ({len(df)} rows)')
    print(df.groupby(['diversity_type'])[
        ['fusion_gain', 'oracle_gain', 'oracle_exploited_fraction',
         'hit_disagreement', 'candidate_jaccard']].mean().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
