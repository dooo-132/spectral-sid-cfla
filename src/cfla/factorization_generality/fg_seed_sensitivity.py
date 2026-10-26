#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed-diversity sensitivity analysis.

The `experiments/seed_ensemble` artifacts were produced by a separate, only
partially-logged run.  Their single-model NDCG@20 spans a very wide range, so a
raw "seed diversity" average mixes genuine seed variance with checkpoints that
never converged.  Reporting only the raw number would let failed training runs
masquerade as a diversity result; silently dropping them would be test-set
selection.  This script therefore reports BOTH:

  * `all`: every available seed pair;
  * `quality_matched`: pairs whose BOTH checkpoints reach at least
    `--min-ratio` (default 0.5) of the canonical (dataset, family=spectral, K)
    single-model NDCG@20.

The threshold is a fixed, pre-declared convergence filter applied to
validation-free single-model quality, not a search over configurations.

Output: seed_diversity_sensitivity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import OUT  # noqa: E402

MIN_RATIO = 0.5


def main():
    pw = pd.read_csv(OUT / 'pairwise_diversity_results.csv')
    art = pd.read_csv(OUT / 'ARTIFACT_MATRIX.csv')
    import json
    art = art[art['family'] == 'spectral'].copy()
    art['canon_ndcg20'] = art['metrics'].apply(lambda s: json.loads(s)['NDCG@20'])
    canon = {(r['dataset'], r['K']): r['canon_ndcg20'] for _, r in art.iterrows()}

    s = pw[pw['diversity_type'] == 'S'].copy()
    s = s[s['K_A'] == s['K_B']]
    s['canon_ndcg20'] = [canon[(d, k)] for d, k in zip(s['dataset'], s['K_A'])]
    s['ratio_A'] = s['single_A_ndcg20'] / s['canon_ndcg20']
    s['ratio_B'] = s['single_B_ndcg20'] / s['canon_ndcg20']
    s['min_ratio'] = s[['ratio_A', 'ratio_B']].min(axis=1)
    s['quality_matched'] = s['min_ratio'] >= MIN_RATIO

    cols = ['dataset', 'K_A', 'model_A', 'model_B', 'single_A_ndcg20',
            'single_B_ndcg20', 'canon_ndcg20', 'ratio_A', 'ratio_B',
            'quality_matched', 'fusion_gain', 'oracle_gain',
            'oracle_exploited_fraction', 'hit_disagreement',
            'candidate_jaccard', 'spearman_common']
    s[cols].to_csv(OUT / 'seed_diversity_sensitivity.csv', index=False)
    print(f'wrote seed_diversity_sensitivity.csv ({len(s)} rows, '
          f'{int(s["quality_matched"].sum())} quality-matched at ratio>={MIN_RATIO})')

    def summ(x):
        return dict(n=len(x), fusion_gain=float(x['fusion_gain'].mean()),
                    oracle_gain=float(x['oracle_gain'].mean()),
                    hit_disagreement=float(x['hit_disagreement'].mean()),
                    candidate_jaccard=float(x['candidate_jaccard'].mean()))

    print('\nS-pairs, ALL:', summ(s))
    print('S-pairs, QUALITY-MATCHED:', summ(s[s['quality_matched']]))
    for d in sorted(s['dataset'].unique()):
        for k in sorted(s['K_A'].unique()):
            g = s[(s['dataset'] == d) & (s['K_A'] == k)]
            if len(g) == 0:
                continue
            q = g[g['quality_matched']]
            print(f'  {d} K{k}: all n={len(g)} {summ(g)} | '
                  f'quality-matched n={len(q)} {summ(q) if len(q) else "-"}')

    print('\ncheckpoint quality relative to canonical (min over the pair):')
    print(s[['dataset', 'K_A', 'model_A', 'model_B', 'ratio_A', 'ratio_B',
             'quality_matched']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
