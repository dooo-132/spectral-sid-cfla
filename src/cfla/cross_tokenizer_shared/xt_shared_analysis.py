#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — shared-checkpoint evaluation.

All retrieval results here come from ONE shared checkpoint evaluated once per
factorization (different control token, different SID decoding).

Outputs -> step2_shared_model/
  shared_single_factorization_metrics.csv   Shared/F_x vs Independent/F_x
  shared_factorization_complementarity.csv  Only-F groups, disagreement, oracle, fusion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (DATASETS, PRIMARY, STEP2, TOKENIZERS, load_cell,
                       oracle_rank, paired_bootstrap_chunked, summarize,
                       union_fuse)  # noqa: E402
from xt_pair_analysis import pair_stats, rrf_fuse  # noqa: E402

RRF_K = 60
N_BOOT = 5000
BOOT_SEED = 20260912


def ndcg_vec(rank, n=20):
    return np.where((rank >= 1) & (rank <= n),
                    1.0 / np.log2(np.maximum(rank, 1) + 1), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--factorizations', default=':'.join(PRIMARY))
    ap.add_argument('--datasets', default=','.join(DATASETS))
    a = ap.parse_args(argv)
    facs = a.factorizations.split(':')

    metric_rows, comp_rows, group_frames = [], [], []
    for ds in a.datasets.split(','):
        shared, indep = {}, {}
        ok = True
        for f in facs:
            p = STEP2 / 'beams' / f'{ds}_{f}_test.npz'
            if not p.exists():
                print(f'  {ds} {f}: shared test beam missing')
                ok = False
                break
            shared[f] = load_cell(f, ds, 'test', beam_override=p)
            indep[f] = load_cell(f, ds, 'test')
        if not ok:
            continue
        print(f'=== {ds} ===', flush=True)

        for f in facs:
            si = union_fuse([indep[f]])['rank']
            ss = union_fuse([shared[f]])['rank']
            mi, ms = summarize(si), summarize(ss)
            metric_rows.append({
                'dataset': ds, 'factorization': f, 'label': TOKENIZERS[f]['label'],
                'independent_ndcg5': mi['NDCG@5'], 'independent_ndcg10': mi['NDCG@10'],
                'independent_ndcg20': mi['NDCG@20'], 'independent_recall20': mi['Recall@20'],
                'shared_ndcg5': ms['NDCG@5'], 'shared_ndcg10': ms['NDCG@10'],
                'shared_ndcg20': ms['NDCG@20'], 'shared_recall20': ms['Recall@20'],
                'shared_over_independent_ndcg20': (ms['NDCG@20'] / mi['NDCG@20']) if mi['NDCG@20'] else float('nan'),
                'severe_sharing_collapse': bool(mi['NDCG@20'] > 0 and ms['NDCG@20'] < 0.7 * mi['NDCG@20']),
            })

        # ---- restricted to the primary pair for the group structure ----
        if len(facs) != 2:
            continue
        ca, cb = shared[facs[0]], shared[facs[1]]
        ra = union_fuse([ca])['rank']
        rb = union_fuse([cb])['rank']
        rf = union_fuse([ca, cb])['rank']
        rrrf = rrf_fuse([ca, cb])
        ro = oracle_rank([ca, cb])
        sa, sb, sf, srrf, so = (summarize(ra), summarize(rb), summarize(rf),
                                summarize(rrrf), summarize(ro))
        best_f, best_s, best_rank = ((facs[0], sa, ra) if sa['NDCG@20'] >= sb['NDCG@20']
                                     else (facs[1], sb, rb))
        div, resA, resB, ha, hb = pair_stats(ca, cb)
        boot_f = paired_bootstrap_chunked(ndcg_vec(rf), ndcg_vec(best_rank),
                                          n_boot=N_BOOT, seed=BOOT_SEED)
        boot_o = paired_bootstrap_chunked(ndcg_vec(ro), ndcg_vec(rf),
                                          n_boot=N_BOOT, seed=BOOT_SEED)
        group = np.where(ha & hb, 'Both', np.where(ha, f'Only-{facs[0]}',
                        np.where(hb, f'Only-{facs[1]}', 'Neither')))
        group_frames.append(pd.DataFrame({
            'dataset': ds, 'user_index': np.arange(len(ha)), 'target': ca['target'],
            'hit_' + facs[0]: ha.astype(np.int8), 'hit_' + facs[1]: hb.astype(np.int8),
            'rank_' + facs[0]: resA['rank'], 'rank_' + facs[1]: resB['rank'],
            'group': group}))
        comp_rows.append({
            'dataset': ds, 'factorizations': '+'.join(facs),
            'best_shared_single': best_f,
            'best_shared_single_ndcg20': best_s['NDCG@20'],
            'shared_fusion_ndcg20': sf['NDCG@20'],
            'shared_fusion_gain': sf['NDCG@20'] - best_s['NDCG@20'],
            'shared_rrf_ndcg20': srrf['NDCG@20'],
            'shared_rrf_gain': srrf['NDCG@20'] - best_s['NDCG@20'],
            'shared_oracle_ndcg20': so['NDCG@20'],
            'shared_oracle_gain': so['NDCG@20'] - best_s['NDCG@20'],
            'shared_oracle_minus_fusion': so['NDCG@20'] - sf['NDCG@20'],
            'oracle_exploited_fraction': ((sf['NDCG@20'] - best_s['NDCG@20']) /
                                          (so['NDCG@20'] - best_s['NDCG@20'])
                                          if so['NDCG@20'] > best_s['NDCG@20'] else float('nan')),
            'fusion_vs_best_delta': boot_f['delta'], 'fusion_vs_best_ci_lo': boot_f['ci_lo'],
            'fusion_vs_best_ci_hi': boot_f['ci_hi'], 'fusion_vs_best_p': boot_f['p_value'],
            'fusion_vs_best_significant_95': bool(boot_f['ci_lo'] > 0),
            'oracle_vs_fusion_delta': boot_o['delta'], 'oracle_vs_fusion_ci_lo': boot_o['ci_lo'],
            'oracle_vs_fusion_ci_hi': boot_o['ci_hi'], 'oracle_vs_fusion_p': boot_o['p_value'],
            'oracle_vs_fusion_significant_95': bool(boot_o['ci_lo'] > 0),
            'only_f1_fraction': float((ha & ~hb).mean()),
            'only_f2_fraction': float((~ha & hb).mean()),
            'both_fraction': float((ha & hb).mean()),
            'neither_fraction': float((~ha & ~hb).mean()),
            'hit_disagreement': div['hit_disagreement'],
            'directional_asymmetry': div['directional_asymmetry'],
            'phi': div['phi'], 'hit_jaccard': div['hit_jaccard'],
            'candidate_jaccard': div['candidate_jaccard'],
            'candidate_union_size': div['candidate_union_size'],
        })
        print(f'  best_shared={best_f} fusion_gain={sf["NDCG@20"]-best_s["NDCG@20"]:+.5f} '
              f'oracle_gain={so["NDCG@20"]-best_s["NDCG@20"]:+.5f} '
              f'only1={(ha&~hb).mean():.4f} only2={(~ha&hb).mean():.4f}', flush=True)

    pd.DataFrame(metric_rows).to_csv(
        STEP2 / 'shared_single_factorization_metrics.csv', index=False)
    pd.DataFrame(comp_rows).to_csv(
        STEP2 / 'shared_factorization_complementarity.csv', index=False)
    if group_frames:
        pd.concat(group_frames, ignore_index=True).to_parquet(
            STEP2 / 'shared_only_groups.parquet', index=False)
    print('wrote shared_single_factorization_metrics.csv, '
          'shared_factorization_complementarity.csv, shared_only_groups.parquet')
    return 0


if __name__ == '__main__':
    sys.exit(main())
