#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — the core Independent-vs-Shared comparison table.

Same factorization pair (Spectral-K3 vs RQ-VAE-3layer), same users, same
evaluator.  The only difference is whether the two factorizations are served by
two independently trained models or by ONE shared checkpoint.

Outputs -> step2_shared_model/independent_vs_shared_factorization.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, PRIMARY, STEP1, STEP2  # noqa: E402

METRICS = [
    ('hit_disagreement', 'Hit@20 disagreement', 'lower'),
    ('candidate_jaccard', 'Top-20 candidate Jaccard', 'lower'),
    ('only_fraction', 'Only-one-factorization fraction', 'lower'),
    ('oracle_gain', 'Oracle gain over best single (NDCG@20)', 'higher'),
    ('fusion_gain', 'Uniform fusion gain over best single (NDCG@20)', 'higher'),
    ('nll_alignment_accuracy', 'Own-factorization NLL alignment accuracy', 'higher'),
]


def main():
    pd_div = pd.read_csv(STEP1 / 'cross_tokenizer_pair_diversity.csv')
    pd_div = pd_div[pd_div['tier'] == 'PRIMARY'].set_index('dataset')
    ig = pd.read_parquet(STEP1 / 'cross_tokenizer_only_groups.parquet')
    ial = pd.read_csv(STEP1 / 'cross_tokenizer_nll_alignment.csv')
    ial = ial[(ial['tier'] == 'PRIMARY') & (ial['normalisation'] == 'raw') &
              (ial['group'] == 'BOTH_Only_groups')].set_index('dataset')

    sh = pd.read_csv(STEP2 / 'shared_factorization_complementarity.csv').set_index('dataset')
    sal_path = STEP2 / 'shared_nll_alignment.csv'
    sal = (pd.read_csv(sal_path).set_index('dataset')
           if sal_path.exists() else None)
    sal = (sal[sal['group'] == 'BOTH_Only_groups'] if sal is not None else None)

    rows = []
    for ds in DATASETS:
        if ds not in pd_div.index or ds not in sh.index:
            print(f'  {ds}: incomplete inputs, skipped')
            continue
        g = ig[ig['dataset'] == ds]
        indep = {
            'hit_disagreement': float(pd_div.loc[ds, 'hit_disagreement']),
            'candidate_jaccard': float(pd_div.loc[ds, 'candidate_jaccard']),
            'only_fraction': float(((g['group'] == 'Only-Spectral') |
                                    (g['group'] == 'Only-RQVAE')).mean()),
            'oracle_gain': float(pd_div.loc[ds, 'oracle_gain']),
            'fusion_gain': float(pd_div.loc[ds, 'fusion_gain']),
            'nll_alignment_accuracy': (float(ial.loc[ds, 'argmin_match_accuracy'])
                                       if ds in ial.index else float('nan')),
            'nll_alignment_n_users': (int(ial.loc[ds, 'n_users'])
                                      if ds in ial.index else 0),
        }
        shared = {
            'hit_disagreement': float(sh.loc[ds, 'hit_disagreement']),
            'candidate_jaccard': float(sh.loc[ds, 'candidate_jaccard']),
            'only_fraction': float(sh.loc[ds, 'only_f1_fraction'] + sh.loc[ds, 'only_f2_fraction']),
            'oracle_gain': float(sh.loc[ds, 'shared_oracle_gain']),
            'fusion_gain': float(sh.loc[ds, 'shared_fusion_gain']),
            'nll_alignment_accuracy': (float(sal.loc[ds, 'argmin_match_accuracy'])
                                       if sal is not None and ds in sal.index
                                       else float('nan')),
            'nll_alignment_n_users': (int(sal.loc[ds, 'n_users'])
                                      if sal is not None and ds in sal.index else 0),
        }
        for key, label, direction in METRICS:
            iv, sv = indep[key], shared[key]
            note = ''
            if key == 'only_fraction':
                # P(exactly one factorization retrieves) IS the Hit@20 disagreement
                # rate; the two rows are the same quantity by definition, not a
                # copy error. Recorded so the equality is not read as a mistake.
                note = ('identical to Hit@20 disagreement by definition: '
                        'Only-F1 + Only-F2 = P(exactly one hits)')
            rows.append({
                'dataset': ds, 'metric': key, 'metric_label': label,
                'preferred_direction': direction, 'note': note,
                'independent_factors': iv, 'shared_checkpoint': sv,
                'shared_minus_independent': sv - iv,
                'shared_over_independent': (sv / iv) if iv not in (0, None) and np.isfinite(iv) and iv != 0 else float('nan'),
                'shared_retains_direction': (
                    (sv > 0) if direction == 'higher' else (sv < iv)),
                'independent_n_users': indep['nll_alignment_n_users'],
                'shared_n_users': shared['nll_alignment_n_users'],
            })
        rows.append({
            'dataset': ds, 'metric': '_best_single_ndcg20',
            'metric_label': 'best single factorization NDCG@20',
            'preferred_direction': 'higher',
            'independent_factors': float(pd_div.loc[ds, 'best_single_ndcg20']),
            'shared_checkpoint': float(sh.loc[ds, 'best_shared_single_ndcg20']),
            'shared_minus_independent': float(sh.loc[ds, 'best_shared_single_ndcg20'] -
                                              pd_div.loc[ds, 'best_single_ndcg20']),
            'shared_over_independent': float(sh.loc[ds, 'best_shared_single_ndcg20'] /
                                             pd_div.loc[ds, 'best_single_ndcg20']),
            'shared_retains_direction': True,
            'independent_n_users': int(len(ig[ig['dataset'] == ds])),
            'shared_n_users': int(len(ig[ig['dataset'] == ds])),
        })

    out = pd.DataFrame(rows)
    out.to_csv(STEP2 / 'independent_vs_shared_factorization.csv', index=False)
    print(f'wrote independent_vs_shared_factorization.csv ({len(out)} rows)')
    piv = out.pivot_table(index=['dataset', 'metric_label'],
                          values=['independent_factors', 'shared_checkpoint'])
    print(piv.round(5).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
