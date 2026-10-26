#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-ID factorization re-audit.

Core object of comparison is the FINAL ITEM IDENTITY (the complete SID of length
K+1), not the length-varying semantic prefix.

PHASE B  fair full-ID generation metrics from the existing beam dumps:
         exact full-SID top-1 accuracy and beam coverage @1/5/10/20/30,
         decoded item coverage, full-SID target rank (MRR), item target rank.
PHASE C  teacher-forced log p(c_t | history, c_<t) for the GROUND-TRUTH target
         SID, using the existing checkpoint. Evaluation only, no training.
PHASE D  NLL decomposition into semantic vs collision positions, and collision
         burden stratified by base-prefix collision-group size.
PHASE E  structural information: marginal and conditional token entropy of the
         SID mapping itself, plus model cross-entropy per position.
PHASE F  link to complementarity: Only-K membership vs full-ID NLL, and whether
         the "easier" side of a disagreement is the one that hits.

Outputs -> outputs/mechanism/full_id_reaudit/
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.per_user import (                                       # noqa: E402
    DATASETS, K_LIST, AFFINITY, build_view, canonicalize_item_ids, sha256_file,
)
from src.analysis.teacher_forced import (                                 # noqa: E402
    load_model_from_ckpt, teacher_forced_logprobs, tokenise_sid,
)
from src.tiger.dataset import GenRecDataset                                # noqa: E402
from src.tiger.dataloader import GenRecDataLoader                          # noqa: E402
from src.utils import repo_paths as rp                                    # noqa: E402

OUT = REPO / 'outputs' / 'mechanism' / 'full_id_reaudit'
OUT.mkdir(parents=True, exist_ok=True)
CB = 256
COVER_N = (1, 5, 10, 20, 30)
GROUP_BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
BUCKET_LABELS = ['g=1', 'g=2', 'g=3-5', 'g=6-10', 'g=11-20', 'g>20']


def sid_array_path(ds, k):
    return (rp.ARTIFACT_ROOT / 'data' / ds /
            f'{ds}_t5_spectral_{AFFINITY}_v3_k{k}_sid{k+1}.npy')


def ckpt_path(ds, k):
    return (rp.ARTIFACT_ROOT / 'model' / 'ckpt' /
            f'tiger_{ds}_nonneg_pearson_v3_k{k}_sid{k+1}.pth')


def main():
    prov = {'phase': 'full_id_factorization_reaudit',
            'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'alignment_mode': 'canonical', 'affinity': AFFINITY,
            'codebook_size': CB, 'training_performed': 0,
            'scored_tokens': 'all K+1 real item-ID tokens; BOS not scored',
            'bos_scored': False, 'eos_scored': False, 'pad_scored': False,
            'inputs': {}}

    top1_rows, cov_rows, itemcov_rows, sidrank_rows, itemrank_rows = [], [], [], [], []
    tf_rows, nll_rows, fact_rows, strat_rows, tokdist_rows = [], [], [], [], []
    posent_rows, condent_rows, structce_rows, errcomp_rows, diffhit_rows = [], [], [], [], []
    user_rows = []

    for ds in DATASETS:
        print(f'=== {ds} ===', flush=True)
        view = build_view(ds)
        ks = view['ks']
        n = view['n_users']
        tgt = view['target']
        prov['inputs'][ds] = {}

        for k in ks:
            m = view['models'][k]
            items = m['items']
            z = m['z']
            cand_tok = np.asarray(z['candidate_tokens'])
            raw_items = np.asarray(z['candidate_item_ids'])
            L = int(cand_tok.shape[2])
            beam = items.shape[1]
            per_user = m['per_user']
            ranks = m['ranks']
            sid = np.load(sid_array_path(ds, k), allow_pickle=True)
            prov['inputs'][ds][str(k)] = {
                'beam': {'path': str(m['path'].relative_to(rp.ARTIFACT_ROOT)), 'sha256': m['sha256']},
                'sid': {'path': str(sid_array_path(ds, k).relative_to(rp.ARTIFACT_ROOT)),
                        'sha256': sha256_file(sid_array_path(ds, k))},
                'ckpt': {'path': str(ckpt_path(ds, k).relative_to(rp.ARTIFACT_ROOT)),
                         'sha256': sha256_file(ckpt_path(ds, k))},
            }
            gt_tok = tokenise_sid(sid[tgt - 1], CB)          # (n, L)
            assert gt_tok.shape == (n, L)
            # sanity: tokens decode back to the target item
            back = gt_tok - np.arange(L)[None, :] * CB - 1
            assert np.array_equal(back, sid[tgt - 1]), 'token/SID round-trip failed'

            # ---------- PHASE B ----------
            exact = (cand_tok == gt_tok[:, None, :]).all(axis=2)      # (n, beam)
            top1 = exact[:, 0]
            top1_rows.append({'dataset': ds, 'K': k, 'L': L, 'N': n,
                              'full_sid_top1_accuracy': float(top1.mean())})
            for nn in COVER_N:
                cov_rows.append({'dataset': ds, 'K': k, 'topN': nn,
                                 'full_sid_beam_coverage': float(exact[:, :nn].any(axis=1).mean()),
                                 'definition': 'exact token-sequence equality of all K+1 tokens'})
            for nn in COVER_N:
                itemcov_rows.append({'dataset': ds, 'K': k, 'topN': nn,
                                     'item_hit': float(((ranks >= 1) & (ranks <= nn)).mean()),
                                     'definition': 'decoded item after beam merge'})
            # full-SID rank
            sr = np.where(exact.any(axis=1), exact.argmax(axis=1) + 1, beam + 1)
            cond = sr[sr <= beam]
            sidrank_rows.append({
                'dataset': ds, 'K': k, 'absent_rate': float((sr > beam).mean()),
                'mean_rank_uncond': float(sr.mean()), 'median_rank_uncond': float(np.median(sr)),
                'p75_rank_uncond': float(np.percentile(sr, 75)),
                'p90_rank_uncond': float(np.percentile(sr, 90)),
                'MRR_uncond': float((1.0 / sr).mean()),
                'mean_rank_cond': float(cond.mean()) if cond.size else float('nan'),
                'MRR_cond': float((1.0 / cond).mean()) if cond.size else float('nan'),
                'n_cond': int(cond.size)})
            ir = np.where(ranks >= 1, ranks, beam + 1)
            icond = ir[ir <= beam]
            itemrank_rows.append({
                'dataset': ds, 'K': k,
                'mean_rank_uncond': float(ir.mean()), 'median_rank_uncond': float(np.median(ir)),
                'p75_rank_uncond': float(np.percentile(ir, 75)),
                'p90_rank_uncond': float(np.percentile(ir, 90)),
                'MRR_uncond': float((1.0 / ir).mean()),
                'mean_rank_cond': float(icond.mean()) if icond.size else float('nan'),
                'MRR_cond': float((1.0 / icond).mean()) if icond.size else float('nan')})

            # ---------- PHASE C: teacher forcing ----------
            print(f'  K={k}: teacher forcing ...', flush=True)
            model, vocab = load_model_from_ckpt(str(ckpt_path(ds, k)))
            dset = GenRecDataset(
                dataset_path=str(rp.ARTIFACT_ROOT / 'data' / ds / 'test.parquet'),
                code_path=str(sid_array_path(ds, k)),
                mode='evaluation', max_len=20, codebook_size=CB)
            assert len(dset) == n, (len(dset), n)
            ld = GenRecDataLoader(dset, batch_size=64, shuffle=False)
            logp = np.zeros((n, L), dtype=np.float64)
            p = 0
            for b in ld:
                hb = b['history'].numpy()
                tb = b['target'].numpy()
                am = b['attention_mask'].numpy()
                # `GenRecDataset` yields the target already in TOKEN space
                # (token = code + pos*codebook + 1). Decode it back to codes and
                # assert it equals the SID array row for this item, so the two
                # sources can never silently diverge.
                codes = tb - np.arange(tb.shape[1])[None, :] * CB - 1
                assert np.array_equal(codes, sid[tgt[p:p + hb.shape[0]] - 1]), \
                    'dataset target codes disagree with the SID array'
                lp = teacher_forced_logprobs(model, hb, tb, device='cuda',
                                             attention_mask=am)
                logp[p:p + hb.shape[0]] = lp
                p += hb.shape[0]
            del model
            import torch
            torch.cuda.empty_cache()
            nll = -logp
            total_nll = nll.sum(axis=1)
            assert np.allclose(-logp.sum(axis=1), total_nll), 'NLL sum inconsistency'

            tf_rows.append(pd.DataFrame({
                'dataset': ds,
                'user_index': np.repeat(np.arange(n), L),
                'K': k,
                'position': np.tile(np.arange(L), n),
                'token_type': np.tile(
                    np.array(['semantic'] * (L - 1) + ['collision']), n),
                'target_token': gt_tok.reshape(-1),
                'log_prob': logp.reshape(-1),
                'nll': nll.reshape(-1),
            }))
            nll_rows.append({
                'dataset': ds, 'K': k, 'L': L, 'N': n,
                'mean_total_nll': float(total_nll.mean()),
                'median_total_nll': float(np.median(total_nll)),
                'std_total_nll': float(total_nll.std()),
                'p25_total_nll': float(np.percentile(total_nll, 25)),
                'p75_total_nll': float(np.percentile(total_nll, 75)),
                'p90_total_nll': float(np.percentile(total_nll, 90)),
                'mean_token_nll': float(nll.mean()),
                'median_token_nll': float(np.median(nll)),
                'mean_logP_full_sid': float(logp.sum(axis=1).mean()),
            })
            # ---------- PHASE D ----------
            sem_nll = nll[:, :L - 1].sum(axis=1)
            col_nll = nll[:, L - 1]
            fact_rows.append({
                'dataset': ds, 'K': k, 'L': L,
                'semantic_nll': float(sem_nll.mean()),
                'collision_nll': float(col_nll.mean()),
                'total_nll': float(total_nll.mean()),
                'semantic_fraction': float(sem_nll.mean() / total_nll.mean()),
                'collision_fraction': float(col_nll.mean() / total_nll.mean()),
            })
            # collision-group stratification
            prefixes = sid[:, :k]
            keys = {}
            for i, r in enumerate(prefixes):
                keys.setdefault(tuple(int(x) for x in r), 0)
                keys[tuple(int(x) for x in r)] += 1
            g = np.array([keys[tuple(int(x) for x in prefixes[i])] for i in range(len(prefixes))])
            gt_g = g[tgt - 1]
            for (lo, hi), lab in zip(GROUP_BUCKETS, BUCKET_LABELS):
                sel = (gt_g >= lo) & (gt_g <= hi)
                if sel.sum() == 0:
                    continue
                col_hit = (cand_tok[:, :, L - 1] == gt_tok[:, L - 1][:, None]).any(axis=1)
                strat_rows.append({
                    'dataset': ds, 'K': k, 'group_bucket': lab,
                    'n_users': int(sel.sum()),
                    'mean_prefix_group_size': float(gt_g[sel].mean()),
                    'collision_token_accuracy': float(col_hit[sel].mean()),
                    'collision_token_nll': float(col_nll[sel].mean()),
                    'full_sid_nll': float(total_nll[sel].mean()),
                    'full_sid_top1_accuracy': float(top1[sel].mean()),
                    'beam_coverage@20': float(exact[sel][:, :20].any(axis=1).mean()),
                    'item_hit@20': float(((ranks[sel] >= 1) & (ranks[sel] <= 20)).mean()),
                })
            # collision token distribution
            gt_col = gt_tok[:, L - 1]
            vals, counts = np.unique(gt_col, return_counts=True)
            pv = counts / counts.sum()
            H = float(-(pv * np.log(pv)).sum())
            tokdist_rows.append({
                'dataset': ds, 'K': k, 'n_distinct_collision_tokens': int(len(vals)),
                'entropy_H_collision': H,
                'majority_token': int(vals[np.argmax(counts)]),
                'majority_token_frequency': float(counts.max() / counts.sum()),
                'prefix_unique_n': int((gt_g == 1).sum()),
                'prefix_unique_mean_collision_nll': float(col_nll[gt_g == 1].mean()) if (gt_g == 1).any() else float('nan'),
                'prefix_unique_majority_freq': float(
                    pd.Series(gt_col[gt_g == 1]).value_counts(normalize=True).iloc[0]) if (gt_g == 1).any() else float('nan'),
            })
            # ---------- PHASE E: structural entropy ----------
            for t in range(L):
                col = sid[:, t]
                vals, counts = np.unique(col, return_counts=True)
                pv = counts / counts.sum()
                posent_rows.append({'dataset': ds, 'K': k, 'position': t,
                                    'token_type': 'collision' if t == L - 1 else 'semantic',
                                    'n_unique_tokens': int(len(vals)),
                                    'marginal_entropy_H': float(-(pv * np.log(pv)).sum())})
            # conditional entropy H(C_t | C_<t) from the SID mapping
            groups: dict = {}
            for i, r in enumerate(sid):
                key = tuple(int(x) for x in r[:L - 1]) if L > 1 else ()
                groups.setdefault(key, []).append(i)
            # exact conditional entropy via prefix grouping
            cond_H = []
            for t in range(L):
                total = 0.0
                for key, idxs in _prefix_groups(sid, t):
                    col = sid[idxs, t]
                    vals, counts = np.unique(col, return_counts=True)
                    pv = counts / counts.sum()
                    total += (len(idxs) / len(sid)) * (-(pv * np.log(pv)).sum())
                cond_H.append(total)
            for t in range(L):
                condent_rows.append({'dataset': ds, 'K': k, 'position': t,
                                     'token_type': 'collision' if t == L - 1 else 'semantic',
                                     'conditional_entropy_H_given_prev': float(cond_H[t])})
            # model CE per position vs structural conditional entropy
            for t in range(L):
                ce = float(nll[:, t].mean())
                hstruct = float(cond_H[t])
                structce_rows.append({
                    'dataset': ds, 'K': k, 'position': t,
                    'token_type': 'collision' if t == L - 1 else 'semantic',
                    'structural_conditional_entropy': hstruct,
                    'model_cross_entropy': ce,
                    'CE_over_H': (ce / hstruct) if hstruct > 1e-9 else float('nan'),
                })

            # ---------- PHASE F ----------
            user_rows.append(pd.DataFrame({
                'dataset': ds, 'user_index': np.arange(n), 'K': k,
                'rank': ranks, 'total_nll': total_nll,
                'semantic_nll': sem_nll, 'collision_nll': col_nll,
                'prefix_group_size': gt_g,
                'hit20': ((ranks >= 1) & (ranks <= 20)).astype(np.int8),
                'full_sid_top1': top1.astype(np.int8),
                'beam20': exact[:, :20].any(axis=1).astype(np.int8),
            }))

        print(f'  {ds} done', flush=True)

    # ---- write ----
    def w(name, rows, parquet=False):
        df = pd.DataFrame(rows)
        if parquet:
            df.to_parquet(OUT / name, index=False)
        else:
            df.to_csv(OUT / name, index=False)
        print(f'  wrote {name} ({len(df)} rows)')
        return df

    w('full_sid_top1_accuracy.csv', top1_rows)
    w('full_sid_beam_coverage.csv', cov_rows)
    w('full_item_coverage.csv', itemcov_rows)
    w('full_sid_target_rank.csv', sidrank_rows)
    w('item_target_rank.csv', itemrank_rows)
    pd.concat(tf_rows, ignore_index=True).to_parquet(
        OUT / 'teacher_forced_token_logprob.parquet', index=False)
    print(f'  wrote teacher_forced_token_logprob.parquet')
    w('teacher_forced_full_sid_nll.csv', nll_rows)
    w('nll_factorization.csv', fact_rows)
    w('collision_group_stratified_metrics.csv', strat_rows)
    w('collision_token_distribution.csv', tokdist_rows)
    w('position_token_entropy.csv', posent_rows)
    w('sid_conditional_entropy.csv', condent_rows)
    w('structural_vs_model_position_cost.csv', structce_rows)
    udf = pd.concat(user_rows, ignore_index=True)
    udf.to_parquet(OUT / 'user_level_full_id_metrics.parquet', index=False)
    print(f'  wrote user_level_full_id_metrics.parquet ({len(udf)} rows)')
    (OUT / '_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print('provenance written')
    return 0


def _prefix_groups(sid, t):
    """Group item row indices by their first t codes (t=0 -> one group)."""
    if t == 0:
        return [((), np.arange(len(sid)))]
    groups = {}
    for i, r in enumerate(sid):
        groups.setdefault(tuple(int(x) for x in r[:t]), []).append(i)
    return list(groups.items())


if __name__ == '__main__':
    sys.exit(main())
