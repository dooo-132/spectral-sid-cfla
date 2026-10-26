#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1 — cross-tokenizer retrieval diversity.

Primary comparison (pre-registered): Spectral-K3 vs RQ-VAE-3layer.
Secondary (structural conditions met, but a different checkpoint-selection
metric was used historically): OPQ-3, RQ1-OPQ2.

Everything is computed by ONE evaluator from the beam dumps:
  * single-model Recall/NDCG @5/10/20
  * uniform item-probability fusion (primary)
  * fixed rank-based fusion diagnostic (RRF, constant k=60 pre-registered)
  * oracle over the pair
  * pairwise disagreement / overlap / rank correlation
  * Only-tokenizer groups at Hit@20

Outputs -> step1_cross_tokenizer/
  cross_tokenizer_retrieval.csv
  cross_tokenizer_pair_diversity.csv
  cross_tokenizer_oracle.csv
  cross_tokenizer_only_groups.parquet
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (DATASETS, PRIMARY, SECONDARY, STEP1, TOKENIZERS, load_cell,
                       oracle_rank, paired_bootstrap_chunked, summarize,
                       union_fuse)  # noqa: E402

RRF_K = 60          # pre-registered constant, never tuned
N_BOOT = 5000
BOOT_SEED = 20260912


def ndcg_vec(rank, n=20):
    return np.where((rank >= 1) & (rank <= n),
                    1.0 / np.log2(np.maximum(rank, 1) + 1), 0.0)


def rrf_fuse(cells):
    """Fixed rank-based fusion: score(i) = sum_m 1/(RRF_K + rank_m(i)).

    Ranking is over each model's own candidate list; items absent from a model's
    candidate list contribute 0 from that model.  Equal weight, constant fixed
    in advance, no tuning.
    """
    n = cells[0]['items'].shape[0]
    acc = {}
    for i in range(n):
        s = {}
        for c in cells:
            items, probs = c['items'][i], c['probs'][i]
            order = np.argsort(-probs, kind='stable')
            seen = set()
            r = 0
            for j in order:
                it = int(items[j])
                if it < 0 or it in seen:
                    continue
                seen.add(it)
                r += 1
                s[it] = s.get(it, 0.0) + 1.0 / (RRF_K + r)
        acc[i] = s
    ranks = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        t = int(cells[0]['target'][i])
        d = acc[i]
        if t in d:
            tv = d[t]
            ranks[i] = 1 + sum(1 for v in d.values() if v > tv)
    return ranks


def top_set(cell, n=20):
    res = union_fuse([cell])
    t = res['top_items'][:, :n].copy()
    valid = t >= 0
    return t, valid, res


def pair_stats(ca, cb):
    """Disagreement / overlap / rank-correlation between two single models."""
    ra, va, resA = top_set(ca)
    rb, vb, resB = top_set(cb)
    hr_a = resA['rank']
    hr_b = resB['rank']
    ha = (hr_a >= 1) & (hr_a <= 20)
    hb = (hr_b >= 1) & (hr_b <= 20)
    n = len(ha)
    both = int((ha & hb).sum())
    only_a = int((ha & ~hb).sum())
    only_b = int((~ha & hb).sum())
    neither = int((~ha & ~hb).sum())
    union = both + only_a + only_b
    num = both * neither - only_a * only_b
    den = np.sqrt((both + only_a) * (only_b + neither) *
                  (both + only_b) * (only_a + neither))
    ia = ra[:, None, :]
    ib = rb[:, :, None]
    inter = ((ia == ib) & va[:, None, :] & vb[:, :, None]).any(axis=2).sum(axis=1)
    na = va.sum(axis=1).astype(np.int64)
    nb = vb.sum(axis=1).astype(np.int64)
    un = na + nb - inter
    jac = np.where(un > 0, inter / np.maximum(un, 1), np.nan)
    out = {
        'n_users': n,
        'hit_spectral': float(ha.mean()), 'hit_rqvae': float(hb.mean()),
        'hit_disagreement': float((only_a + only_b) / n),
        'p_a1_b0': float(only_a / n), 'p_a0_b1': float(only_b / n),
        'directional_asymmetry': float(abs(only_a - only_b) / n),
        'phi': float(num / den) if den > 0 else float('nan'),
        'hit_jaccard': float(both / union) if union else float('nan'),
        'candidate_jaccard': float(np.nanmean(jac)),
        'candidate_intersection': float(inter.mean()),
        'candidate_union_size': float(un.mean()),
        'topn_a': float(na.mean()), 'topn_b': float(nb.mean()),
    }
    return out, resA, resB, ha, hb


def main():
    ret_rows, div_rows, orc_rows = [], [], []
    only_frames = []

    pairs = [('PRIMARY', PRIMARY[0], PRIMARY[1])]
    for a, b in itertools.combinations(SECONDARY, 2):
        pairs.append(('SECONDARY', a, b))
    for s in SECONDARY:
        pairs.append(('SECONDARY_vs_primary', PRIMARY[0], s))
        pairs.append(('SECONDARY_vs_primary', PRIMARY[1], s))

    for ds in DATASETS:
        print(f'=== {ds} ===', flush=True)
        cells = {}
        for tok in TOKENIZERS:
            try:
                cells[tok] = load_cell(tok, ds, 'test')
            except FileNotFoundError:
                print(f'  {tok}: test beam missing, skipped')
        for tier, ta, tb in pairs:
            if ta not in cells or tb not in cells:
                continue
            ca, cb = cells[ta], cells[tb]
            ra = union_fuse([ca])['rank']
            rb = union_fuse([cb])['rank']
            rf = union_fuse([ca, cb])['rank']
            rrrf = rrf_fuse([ca, cb])
            ro = oracle_rank([ca, cb])

            sa, sb, sf, srrf, so = (summarize(ra), summarize(rb), summarize(rf),
                                    summarize(rrrf), summarize(ro))
            for tok, s in ((ta, sa), (tb, sb)):
                ret_rows.append({
                    'dataset': ds, 'tier': tier, 'tokenizer': tok,
                    'label': TOKENIZERS[tok]['label'],
                    'family': TOKENIZERS[tok]['family'],
                    **{f'single_{k}': v for k, v in s.items()}})
            ret_rows.append({
                'dataset': ds, 'tier': tier, 'tokenizer': 'uniform_fusion',
                'label': f'{TOKENIZERS[ta]["label"]}+{TOKENIZERS[tb]["label"]}',
                'family': 'fusion',
                **{f'single_{k}': v for k, v in sf.items()}})
            ret_rows.append({
                'dataset': ds, 'tier': tier, 'tokenizer': 'rrf_fusion_diagnostic',
                'label': f'RRF(k={RRF_K})',
                'family': 'fusion',
                **{f'single_{k}': v for k, v in srrf.items()}})
            ret_rows.append({
                'dataset': ds, 'tier': tier, 'tokenizer': 'oracle',
                'label': 'oracle', 'family': 'oracle',
                **{f'single_{k}': v for k, v in so.items()}})

            div, resA, resB, ha, hb = pair_stats(ca, cb)
            best_tok, best_s = ((ta, sa) if sa['NDCG@20'] >= sb['NDCG@20'] else (tb, sb))
            fusion_gain = sf['NDCG@20'] - best_s['NDCG@20']
            oracle_gain = so['NDCG@20'] - best_s['NDCG@20']
            rrf_gain = srrf['NDCG@20'] - best_s['NDCG@20']

            boot_f = paired_bootstrap_chunked(ndcg_vec(rf), ndcg_vec(ra if best_tok == ta else rb),
                                              n_boot=N_BOOT, seed=BOOT_SEED)
            boot_o = paired_bootstrap_chunked(ndcg_vec(ro), ndcg_vec(rf),
                                              n_boot=N_BOOT, seed=BOOT_SEED)

            div_rows.append({
                'dataset': ds, 'tier': tier,
                'model_A': ta, 'model_B': tb,
                'label_A': TOKENIZERS[ta]['label'], 'label_B': TOKENIZERS[tb]['label'],
                'best_single_tokenizer': best_tok,
                'best_single_ndcg20': best_s['NDCG@20'],
                'fusion_ndcg20': sf['NDCG@20'],
                'fusion_gain': fusion_gain,
                'fusion_gain_pct': 100 * fusion_gain / best_s['NDCG@20'] if best_s['NDCG@20'] else float('nan'),
                'rrf_ndcg20': srrf['NDCG@20'], 'rrf_gain': rrf_gain,
                'oracle_ndcg20': so['NDCG@20'],
                'oracle_gain': oracle_gain,
                'oracle_minus_fusion': so['NDCG@20'] - sf['NDCG@20'],
                'oracle_exploited_fraction': (fusion_gain / oracle_gain) if oracle_gain > 0 else float('nan'),
                'fusion_vs_best_delta': boot_f['delta'],
                'fusion_vs_best_ci_lo': boot_f['ci_lo'],
                'fusion_vs_best_ci_hi': boot_f['ci_hi'],
                'fusion_vs_best_p': boot_f['p_value'],
                'fusion_vs_best_significant_95': bool(boot_f['ci_lo'] > 0),
                'oracle_vs_fusion_delta': boot_o['delta'],
                'oracle_vs_fusion_ci_lo': boot_o['ci_lo'],
                'oracle_vs_fusion_ci_hi': boot_o['ci_hi'],
                'oracle_vs_fusion_p': boot_o['p_value'],
                'oracle_vs_fusion_significant_95': bool(boot_o['ci_lo'] > 0),
                **div,
            })
            orc_rows.append({
                'dataset': ds, 'tier': tier, 'model_A': ta, 'model_B': tb,
                'best_single_tokenizer': best_tok,
                'best_single_recall20': best_s['Recall@20'],
                'best_single_ndcg20': best_s['NDCG@20'],
                'fusion_recall20': sf['Recall@20'], 'fusion_ndcg20': sf['NDCG@20'],
                'oracle_recall20': so['Recall@20'], 'oracle_ndcg20': so['NDCG@20'],
                'best_single_recall5': best_s['Recall@5'],
                'best_single_recall10': best_s['Recall@10'],
                'fusion_recall5': sf['Recall@5'], 'fusion_recall10': sf['Recall@10'],
                'oracle_recall5': so['Recall@5'], 'oracle_recall10': so['Recall@10'],
                'fusion_gain': fusion_gain, 'oracle_gain': oracle_gain,
                'oracle_minus_fusion': so['NDCG@20'] - sf['NDCG@20'],
                'oracle_exploited_fraction': (fusion_gain / oracle_gain) if oracle_gain > 0 else float('nan'),
            })

            if tier == 'PRIMARY':
                group = np.where(ha & hb, 'Both', np.where(ha, 'Only-Spectral',
                                np.where(hb, 'Only-RQVAE', 'Neither')))
                only_frames.append(pd.DataFrame({
                    'dataset': ds, 'user_index': np.arange(len(ha)),
                    'target': ca['target'],
                    'hit_spectral': ha.astype(np.int8),
                    'hit_rqvae': hb.astype(np.int8),
                    'rank_spectral': resA['rank'], 'rank_rqvae': resB['rank'],
                    'group': group,
                }))

    ret = pd.DataFrame(ret_rows)
    div = pd.DataFrame(div_rows)
    orc = pd.DataFrame(orc_rows)
    ret.to_csv(STEP1 / 'cross_tokenizer_retrieval.csv', index=False)
    div.to_csv(STEP1 / 'cross_tokenizer_pair_diversity.csv', index=False)
    orc.to_csv(STEP1 / 'cross_tokenizer_oracle.csv', index=False)
    if only_frames:
        pd.concat(only_frames, ignore_index=True).to_parquet(
            STEP1 / 'cross_tokenizer_only_groups.parquet', index=False)

    print(f'\nwrote cross_tokenizer_retrieval.csv ({len(ret)} rows)')
    print(f'wrote cross_tokenizer_pair_diversity.csv ({len(div)} rows)')
    print(f'wrote cross_tokenizer_oracle.csv ({len(orc)} rows)')
    print(f'wrote cross_tokenizer_only_groups.parquet')
    prim = div[div.tier == 'PRIMARY']
    if len(prim):
        print('\nPRIMARY (Spectral-K3 vs RQ-VAE-3layer):')
        print(prim[['dataset', 'best_single_tokenizer', 'best_single_ndcg20',
                    'fusion_ndcg20', 'fusion_gain', 'oracle_ndcg20', 'oracle_gain',
                    'oracle_exploited_fraction', 'hit_disagreement',
                    'candidate_jaccard', 'phi']].round(5).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
