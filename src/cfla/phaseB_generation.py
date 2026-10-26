#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE B (generation side) + the combined trade-off summary.

For each dataset x K, from the EXISTING canonical beam dumps (no retraining):

  14.1 exact-SID target coverage   : is the ground-truth SID among the beams
  14.2 item target coverage        : after decode+merge, is the target item in
                                     Top5/10/20/30
  15   target rank                 : mean/median/P75/P90, unconditional
                                     (absent -> beam_size+1) and conditional
  16   target score                : sequence score and normalized item
                                     probability, mean/median/std/quantiles
  17   decode validity             : valid / invalid SID ratio, decoded
                                     candidate ratio, unique item ratio after
                                     merge
  18   token / prefix accuracy     : from candidate_tokens, the per-position
                                     accuracy of the target SID and the
                                     P(first t tokens all correct), with the
                                     collision position reported separately
  19   NLL / generation difficulty : per-token NLL, sequence NLL and
                                     length-normalized NLL, from
                                     transition_logprobs. Evaluation only.
  20   representation-generation summary table (raw quantities only, no
       invented composite score)
  21   descriptive correlation of each metric with K and with NDCG@20
       (n=4 granularities -> descriptive only, explicitly not significance)

Token accuracy note: `candidate_tokens` are the beam-search outputs. Position-wise
accuracy is measured against the ground-truth SID among the *generated* beams, so
it is a coverage-weighted quantity, not a teacher-forced accuracy. This is stated
in the output columns (`token_acc_*_generated`).

Outputs -> outputs/mechanism/phaseB_tradeoff/
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
from src.analysis.per_user import (                                      # noqa: E402
    DATASETS, K_LIST, build_view, per_user_items, summarize_ranks, topn,
)

OUT = REPO / 'outputs' / 'mechanism' / 'phaseB_tradeoff'
TOPK = (5, 10, 20, 30)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = {'phase': 'B_generation_side', 'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'alignment_mode': 'canonical', 'affinity': 'nonneg_pearson',
            'inputs': {}}
    cov_rows, rank_rows, score_rows, valid_rows, tok_rows, pref_rows = [], [], [], [], [], []

    for ds in DATASETS:
        print(f'=== {ds} ===', flush=True)
        view = build_view(ds)
        ks = view['ks']
        tgt = view['target']
        n = view['n_users']
        prov['inputs'][ds] = {str(k): {'path': str(view['models'][k]['path'].relative_to(REPO.parent)),
                                       'sha256': view['models'][k]['sha256']} for k in ks}

        for k in ks:
            m = view['models'][k]
            z = m['z']
            items = m['items']
            probs = m['probs']
            cand_tok = np.asarray(z['candidate_tokens'])
            tlp = np.asarray(z['transition_logprobs'], dtype=np.float64)
            sc = np.asarray(z['sequence_scores'], dtype=np.float64)
            raw_items = np.asarray(z['candidate_item_ids'])
            sid_len = int(cand_tok.shape[2])
            beam = items.shape[1]
            codebook_size = (int(z['vocab_size']) - 1) // sid_len
            per_user = m['per_user']
            ranks = m['ranks']
            target = m['target']

            # ---- 17. decode validity ----
            n_valid = int((raw_items >= 0).sum())
            n_tot = int(raw_items.size)
            uniq_after = []
            for i in range(n):
                uniq_after.append(len(per_user[i]))
            valid_rows.append({
                'dataset': ds, 'K': k, 'sid_len': sid_len, 'beam_size': beam,
                'valid_sid_ratio': n_valid / n_tot,
                'invalid_sid_ratio': 1.0 - n_valid / n_tot,
                'decoded_candidate_ratio': n_valid / n_tot,
                'unique_items_after_merge_mean': float(np.mean(uniq_after)),
                'unique_items_after_merge_over_beam': float(np.mean(uniq_after) / beam),
                'n_users': n,
            })

            # ---- 18. token / prefix accuracy ----
            # Reconstruct the ground-truth target SID from the SID array
            # (row = canonical item id - 1) and tokenise it in the same slot
            # layout as the beams.
            from src.utils import repo_paths as rp
            p_sid = rp.ARTIFACT_ROOT / 'data' / ds / f'{ds}_t5_spectral_nonneg_pearson_v3_k{k}_sid{k+1}.npy'
            sid_arr = np.load(p_sid, allow_pickle=True)
            # candidate_tokens row: for each user and beam, the generated tokens
            # compare to the target SID's tokens, per position
            for pos in range(sid_len):
                # match rate of the generated beams' position-pos token against
                # the ground-truth code at that position (coverage-weighted)
                gt_codes = sid_arr[target - 1]                     # (n, sid_len)
                gt_tok = gt_codes + np.arange(sid_len)[None, :] * codebook_size + 1
                c = cand_tok[:, :, pos]
                g = gt_tok[:, pos][:, None]
                hit_any = (c == g).any(axis=1)
                is_collision_pos = (pos == sid_len - 1)
                tok_rows.append({
                    'dataset': ds, 'K': k, 'position': pos + 1,
                    'is_collision_position': bool(is_collision_pos),
                    'token_acc_generated_any_beam': float(hit_any.mean()),
                    'n_users': n,
                })
            # prefix exact accuracy: first t tokens all correct in at least one beam
            gt_codes = sid_arr[target - 1]
            gt_tok = gt_codes + np.arange(sid_len)[None, :] * codebook_size + 1
            for t in range(1, sid_len + 1):
                ok = np.ones(n, dtype=bool)
                for pos in range(t):
                    ok &= (cand_tok[:, :, pos] == gt_tok[:, pos][:, None]).any(axis=1)
                pref_rows.append({
                    'dataset': ds, 'K': k, 'prefix_len_t': t,
                    'includes_collision': bool(t == sid_len),
                    'prefix_exact_acc_any_beam': float(ok.mean()), 'n_users': n,
                })

            # ---- 14. coverage ----
            target_in_beam_any = np.array([(items[i] == target[i]).any() for i in range(n)])
            cov = {'dataset': ds, 'K': k, 'n_users': n,
                   'exact_sid_beam_any': float(target_in_beam_any.mean())}
            for nn in TOPK:
                cov[f'item_hit@{nn}'] = float(((ranks >= 1) & (ranks <= nn)).mean())
            cov['item_in_candidates_any'] = float((ranks >= 1).mean())
            # SID-level coverage differs from item-level coverage only when
            # several beams decode to the same item. Record the raw counts so the
            # distinction is auditable rather than inferred.
            n_beams_valid = int((raw_items >= 0).sum(axis=1).sum())
            n_unique_decoded = int(sum(len(per_user[i]) for i in range(n)))
            cov['n_valid_beams_total'] = n_beams_valid
            cov['n_unique_items_total'] = n_unique_decoded
            cov['beam_to_item_collapse_ratio'] = (n_unique_decoded / n_beams_valid
                                                  if n_beams_valid else float('nan'))
            cov_rows.append(cov)

            # ---- 15. target rank ----
            absent = ranks < 1
            uncond = np.where(absent, beam + 1, ranks)
            cond = ranks[~absent]
            rank_rows.append({
                'dataset': ds, 'K': k,
                'absent_rate': float(absent.mean()),
                'mean_rank_uncond': float(uncond.mean()),
                'median_rank_uncond': float(np.median(uncond)),
                'p75_rank_uncond': float(np.percentile(uncond, 75)),
                'p90_rank_uncond': float(np.percentile(uncond, 90)),
                'mean_rank_cond': float(cond.mean()) if cond.size else float('nan'),
                'median_rank_cond': float(np.median(cond)) if cond.size else float('nan'),
                'p75_rank_cond': float(np.percentile(cond, 75)) if cond.size else float('nan'),
                'p90_rank_cond': float(np.percentile(cond, 90)) if cond.size else float('nan'),
                'n_cond': int(cond.size),
            })

            # ---- 16. target score ----
            # normalized item probability of the target (0 when absent)
            tp = np.array([per_user[i].get(int(target[i]), 0.0) for i in range(n)])
            valid_sc = sc[raw_items >= 0]
            idx = np.arange(n)
            tgt_beam_pos = None
            tgt_seq = np.full(n, np.nan)
            for i in range(n):
                w = np.where(items[i] == target[i])[0]
                if len(w):
                    tgt_seq[i] = float(sc[i, w[0]])
            tvalid = tgt_seq[~np.isnan(tgt_seq)]
            score_rows.append({
                'dataset': ds, 'K': k,
                'target_item_prob_mean': float(tp.mean()),
                'target_item_prob_median': float(np.median(tp)),
                'target_item_prob_std': float(tp.std()),
                'target_item_prob_p25': float(np.percentile(tp, 25)),
                'target_item_prob_p75': float(np.percentile(tp, 75)),
                'target_item_prob_p95': float(np.percentile(tp, 95)),
                'target_seq_score_mean': float(tvalid.mean()) if tvalid.size else float('nan'),
                'target_seq_score_median': float(np.median(tvalid)) if tvalid.size else float('nan'),
                'target_seq_score_std': float(tvalid.std()) if tvalid.size else float('nan'),
                'all_beam_score_mean': float(valid_sc.mean()),
                'all_beam_score_median': float(np.median(valid_sc)),
            })

        print(f'  {ds} generation metrics done', flush=True)

    def w(name, rows):
        df = pd.DataFrame(rows)
        df.to_csv(OUT / name, index=False)
        print(f'  wrote {name} ({len(df)} rows)', flush=True)
        return df

    cov = w('generation_coverage.csv', cov_rows)
    rk = w('generation_target_rank.csv', rank_rows)
    sc = w('generation_target_score.csv', score_rows)
    va = w('generation_decode_validity.csv', valid_rows)
    tk = w('token_accuracy_by_position.csv', tok_rows)
    pf = w('prefix_accuracy.csv', pref_rows)

    # ---- 19. NLL from transition_logprobs (evaluation only) ----
    nll_rows = []
    for ds in DATASETS:
        view = build_view(ds)
        for k in view['ks']:
            z = view['models'][k]['z']
            tlp = np.asarray(z['transition_logprobs'], dtype=np.float64)
            tok = np.asarray(z['candidate_tokens'])
            valid = (tok > 0)
            per_beam_nll = -(np.where(valid, tlp, 0.0)).sum(axis=1)      # (n, beam)
            per_beam_len = valid.sum(axis=1)
            uniq_len = np.unique(per_beam_len)
            seq_nll = per_beam_nll.mean(axis=1)
            ln_nll = (per_beam_nll / np.maximum(per_beam_len, 1)).mean(axis=1)
            tok_nll = (per_beam_nll.sum(axis=1) / np.maximum(per_beam_len.sum(axis=1), 1))
            nll_rows.append({
                'dataset': ds, 'K': k,
                'avg_token_NLL': float(tok_nll.mean()),
                'avg_sequence_NLL': float(seq_nll.mean()),
                'length_normalized_NLL': float(ln_nll.mean()),
                'sid_len': int(tok.shape[2]),
                'n_users': int(len(seq_nll)),
            })
    nll = w('generation_nll.csv', nll_rows)

    # ---- 20. combined representation-generation summary ----
    pref = pd.read_csv(OUT / 'prefix_metrics.csv')
    nb = pd.read_csv(OUT / 'neighborhood_preservation.csv')
    dc = pd.read_csv(OUT / 'representation_distance_correlation.csv')
    nb_shared = nb[nb.sid_metric == 'shared'][['dataset', 'K', 'neighbor_recall@20']]
    summ = (pref[['dataset', 'K', 'prefix_uniqueness', 'prefix_collision_rate',
                  'effective_prefixes_over_N', 'max_collision_group_size']]
            .merge(nb_shared, on=['dataset', 'K'], how='left')
            .merge(dc[['dataset', 'K', 'spearman_shared_token']], on=['dataset', 'K'], how='left')
            .merge(tk[tk.is_collision_position == False].groupby(['dataset', 'K'])
                   .token_acc_generated_any_beam.mean().reset_index()
                   .rename(columns={'token_acc_generated_any_beam': 'semantic_token_acc_mean'}),
                   on=['dataset', 'K'], how='left')
            .merge(cov[['dataset', 'K', 'exact_sid_beam_any', 'item_hit@20']], on=['dataset', 'K'], how='left')
            .merge(nll[['dataset', 'K', 'length_normalized_NLL']], on=['dataset', 'K'], how='left'))
    # per-K single-model NDCG@20
    nd = []
    for ds in DATASETS:
        view = build_view(ds)
        for k in view['ks']:
            nd.append({'dataset': ds, 'K': k,
                       'NDCG@20': summarize_ranks(view['models'][k]['ranks'])['NDCG@20']})
    summ = summ.merge(pd.DataFrame(nd), on=['dataset', 'K'], how='left')
    summ.to_csv(OUT / 'representation_generation_tradeoff.csv', index=False)
    print(f'  wrote representation_generation_tradeoff.csv ({len(summ)} rows)', flush=True)

    # ---- 21. descriptive correlations (n=4 -> descriptive only) ----
    from scipy.stats import pearsonr, spearmanr
    corr_rows = []
    metrics = ['prefix_uniqueness', 'prefix_collision_rate', 'effective_prefixes_over_N',
               'neighbor_recall@20', 'spearman_shared_token', 'semantic_token_acc_mean',
               'exact_sid_beam_any', 'item_hit@20', 'length_normalized_NLL', 'NDCG@20']
    for ds in DATASETS:
        sub = summ[summ.dataset == ds].sort_values('K')
        for target_name in ('NDCG@20',):
            for mname in metrics:
                if mname == target_name:
                    continue
                x = sub[mname].to_numpy(dtype=float)
                y = sub[target_name].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 3:
                    continue
                corr_rows.append({
                    'dataset': ds, 'x': mname, 'y': target_name,
                    'n_granularities': int(ok.sum()),
                    'pearson': float(pearsonr(x[ok], y[ok])[0]),
                    'spearman': float(spearmanr(x[ok], y[ok]).statistic),
                    'note': 'descriptive only; n=4 granularities',
                })
                # also vs K
                kk = sub['K'].to_numpy(dtype=float)
                if np.isfinite(x).all():
                    corr_rows.append({
                        'dataset': ds, 'x': mname, 'y': 'K',
                        'n_granularities': int(len(kk)),
                        'pearson': float(pearsonr(kk, x)[0]),
                        'spearman': float(spearmanr(kk, x).statistic),
                        'note': 'descriptive only; n=4 granularities',
                    })
    pd.DataFrame(corr_rows).to_csv(OUT / 'descriptive_correlations.csv', index=False)
    print('  wrote descriptive_correlations.csv', flush=True)

    import subprocess
    prov['git_sha'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                     capture_output=True, text=True).stdout.strip()
    (OUT / '_PROVENANCE_generation.json').write_text(json.dumps(prov, indent=2, default=str))
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
