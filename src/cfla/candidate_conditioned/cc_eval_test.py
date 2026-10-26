#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE G + H + I + J — freeze, then evaluate the candidate-conditioned rerankers on test.

Order of operations is the point of this file:

  1. `CANDIDATE_RERANK_FREEZE_MANIFEST.json` is written FIRST: git SHA, shared
     checkpoint hashes, candidate-generation config, beam size, union rule, the NLL
     scaler, the selected T and lambda, the method definitions and the validation
     NDCG.  After it exists nothing about the method may change.
  2. Only then is the canonical TEST split read.

Produces the test deliverables required by the prompt:
  test/test_results.csv, target_rank_movement.csv, uniform_failure_rescue.csv,
  subgroup_results.csv, bootstrap.csv

Usage
  python3 cc_eval_test.py --rundir <run> --outdir outputs/candidate_conditioned
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_common import (DATASETS, F1, F2, SHORT, frozen_checkpoint,  # noqa: E402
                       sha256_file, write_json)
from cc_rerank_methods import (LAMBDA_GRID, T_GRID, _targets, apply_scaler,  # noqa: E402
                               grouped_arrays, load_cells, ndcg_from_rank,
                               rank_of_target, score_cell)

BOOT_N = 5000
BOOT_SEED = 20260912
METRIC_KS = (5, 10, 20)
METHODS = ('BestF1', 'BestF2', 'Uniform', 'Static', 'MinNLL', 'LSE', 'Gate',
           'Hybrid', 'Oracle')


def git_sha():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()


def summarize(rank, ks=METRIC_KS):
    out = {}
    for k in ks:
        hit = (rank >= 1) & (rank <= k)
        out[f'Recall@{k}'] = float(hit.mean())
        nd = np.zeros(len(rank)); nd[hit] = 1.0 / np.log2(rank[hit] + 1)
        out[f'NDCG@{k}'] = float(nd.mean())
    out['n'] = int(len(rank))
    return out


def paired_bootstrap(a, b, n_boot=BOOT_N, seed=BOOT_SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d); delta = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(boots, 2.5)); hi = float(np.percentile(boots, 97.5))
    p = 2.0 * float((boots <= 0).mean()) if delta >= 0 else 2.0 * float((boots >= 0).mean())
    return {'delta': delta, 'ci_low': lo, 'ci_high': hi,
            'p_value': min(1.0, max(p, 1.0 / n_boot)), 'n_users': n}


def freeze_manifest(rundir, out, scaler, selected, extra):
    ck = {ds: {'path': str(frozen_checkpoint(ds)),
               'sha256': sha256_file(frozen_checkpoint(ds))} for ds in DATASETS}
    man = {
        'study': 'candidate-conditioned factorization reranking',
        'git_sha': git_sha(),
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'shared_checkpoints': ck,
        'candidate_generation': {
            'source': 'frozen shared model beam dumps, beam=30 per factorization',
            'union_rule': 'C_union(u) = C_F1(u) UNION C_F2(u), generation output only',
            'beam_size': 30,
            'candidate_width_per_factorization': 30,
            'target_added_to_union': False,
            'pairs_per_dataset_split': extra['pairs'],
        },
        'nll_scaler': scaler,
        'nll_scaler_fitted_on': 'ALL validation candidates, per factorization',
        'selected_T': {ds: selected[ds]['T'] for ds in DATASETS},
        'selected_lambda': {ds: selected[ds]['lambda'] for ds in DATASETS},
        'grids': {'T': T_GRID, 'lambda': LAMBDA_GRID},
        'method_definitions': {
            'MinNLL': 'S = -min(zNLL_F1, zNLL_F2)',
            'LSE': 'S = logsumexp(-zNLL_F1, -zNLL_F2)',
            'Gate': 'S = w1*P_F1 + w2*P_F2 with w = softmax(-zNLL/T)',
            'Hybrid': 'S = lambda*Z(S_uniform) + (1-lambda)*Z(S_LSE), Z = user-level z',
            'Uniform': 'S = 0.5*P_F1 + 0.5*P_F2 (audited implementation)',
            'Static': 'S = w*P_F1 + (1-w)*P_F2 with the previous round frozen w = 0.5',
        },
        'validation_ndcg20': {ds: selected[ds]['valid_ndcg20'] for ds in DATASETS},
        'test_metrics_computed_at_freeze_time': False,
        **extra,
    }
    p = Path(out) / 'CANDIDATE_RERANK_FREEZE_MANIFEST.json'
    p.write_text(json.dumps(man, indent=2, default=str))
    return p, man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--outdir', default='outputs/candidate_conditioned')
    ap.add_argument('--featdir',
                    default='outputs/adaptive_factorization/selector/features')
    ap.add_argument('--static-w', type=float, default=0.5)
    a = ap.parse_args(argv)
    rundir, out = Path(a.rundir), Path(a.outdir)
    (out / 'test').mkdir(parents=True, exist_ok=True)

    # Refuse to do anything unless the candidate-NLL run is complete for every
    # dataset and split.  This matters because the freeze manifest is written before
    # test is read: running on a partial run would freeze a manifest that does not
    # describe the study.
    missing = []
    for split in ('valid', 'test'):
        for ds in DATASETS:
            for kind in ('candidate_union', 'candidate_nll'):
                f = rundir / split / f'{ds}_{kind}.parquet'
                if not f.exists():
                    missing.append(str(f))
    if missing:
        print('INCOMPLETE RUN — refusing to freeze or evaluate. Missing:')
        for m in missing[:10]:
            print(f'  {m}')
        return 2

    scaler = json.loads((out / 'validation' / 'candidate_nll_scaler.json').read_text())
    selected = json.loads((out / 'validation' / 'selected_hyperparameters.json').read_text())
    pairs = {}
    for split in ('valid', 'test'):
        for ds in DATASETS:
            u = load_cells(rundir, ds, split)[0]
            pairs.setdefault(split, {})[ds] = int(len(u))
            del u

    # ---------------- 1. FREEZE (test not yet read) ----------------
    man_p, man = freeze_manifest(rundir, out, scaler, selected,
                                 {'pairs': pairs, 'phase': 'FROZEN_BEFORE_TEST_READ'})
    print(f'[freeze] wrote {man_p}', flush=True)

    # ---------------- 2. TEST ----------------
    ranks = {m: {} for m in METHODS}
    per_ds_detail = {}
    for ds in DATASETS:
        u, n = load_cells(rundir, ds, 'test')
        g = grouped_arrays(u, n)
        z1, z2 = apply_scaler(g, {'z1': (scaler[ds]['mu_F1'], scaler[ds]['sigma_F1']),
                                  'z2': (scaler[ds]['mu_F2'], scaler[ds]['sigma_F2'])})
        tgt = _targets(rundir, ds, 'test', a.featdir)
        T = selected[ds]['T']; lam = selected[ds]['lambda']

        sc = {
            'Uniform': score_cell(g, z1, z2, 'Uniform'),
            'MinNLL': score_cell(g, z1, z2, 'MinNLL'),
            'LSE': score_cell(g, z1, z2, 'LSE'),
            'Gate': score_cell(g, z1, z2, 'Gate', T=T),
            'Hybrid': score_cell(g, z1, z2, 'Hybrid', lam=lam),
        }
        # single-factorization baselines and the in-union oracle
        sc['BestF1'] = g['p1'].copy()
        sc['BestF2'] = g['p2'].copy()
        sc['Static'] = a.static_w * g['p1'] + (1 - a.static_w) * g['p2']
        # Oracle = perfect reranking WITHIN the given union: the target goes first if the
        # union contains it at all.  Its NDCG@20 therefore equals union coverage, which is
        # what keeps reranking failure separable from candidate-generation failure.
        ranks['Oracle'][ds] = _oracle_rank(g, tgt)
        for m in METHODS:
            if m == 'Oracle':
                continue
            ranks[m][ds] = rank_of_target(g, sc[m], tgt)
        # the in-union ceiling: target present in the union at all
        present = np.array([tgt[u] in set(g['item'][g['starts'][u]:g['ends'][u]].tolist())
                            for u in range(len(g['users']))])
        per_ds_detail[ds] = {'g': g, 'tgt': tgt, 'scores': sc, 'z1': z1, 'z2': z2,
                             'present': present, 'T': T, 'lam': lam}
        print(f'  {SHORT[ds]:12s} test users {len(g["users"])}  '
              f'union {len(g["item"])}  present {present.mean():.4f}', flush=True)

    # ---------------- test_results.csv ----------------
    with (out / 'test' / 'test_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'method', 'n'] + [f'Recall@{k}' for k in METRIC_KS]
                   + [f'NDCG@{k}' for k in METRIC_KS])
        for ds in DATASETS:
            for m in METHODS:
                s = summarize(ranks[m][ds])
                w.writerow([ds, m, s['n']] + [f'{s[f"Recall@{k}"]:.6f}' for k in METRIC_KS]
                           + [f'{s[f"NDCG@{k}"]:.6f}' for k in METRIC_KS])
        for m in METHODS:
            r = np.concatenate([ranks[m][ds] for ds in DATASETS])
            s = summarize(r)
            w.writerow(['ALL_POOLED', m, s['n']]
                       + [f'{s[f"Recall@{k}"]:.6f}' for k in METRIC_KS]
                       + [f'{s[f"NDCG@{k}"]:.6f}' for k in METRIC_KS])

    # ---------------- bootstrap.csv ----------------
    with (out / 'test' / 'bootstrap.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['comparison', 'dataset', 'metric', 'delta', 'ci_low', 'ci_high',
                    'p_value', 'n_users'])
        for metric_k in METRIC_KS:
            for ds in DATASETS + ['ALL_POOLED']:
                def vec(m):
                    r = (ranks[m][ds] if ds != 'ALL_POOLED'
                         else np.concatenate([ranks[m][d] for d in DATASETS]))
                    return ndcg_from_rank(r, metric_k)
                for other in ('Uniform', 'Static', 'BestF1'):
                    st = paired_bootstrap(vec('LSE'), vec(other))
                    w.writerow([f'LSE_vs_{other}', ds, f'NDCG@{metric_k}',
                                f'{st["delta"]:.8f}', f'{st["ci_low"]:.8f}',
                                f'{st["ci_high"]:.8f}', f'{st["p_value"]:.6f}', st['n_users']])

    # ---------------- target_rank_movement.csv ----------------
    with (out / 'test' / 'target_rank_movement.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'n_target_in_union', 'method', 'improved_vs_uniform',
                    'unchanged_vs_uniform', 'worsened_vs_uniform', 'mean_rank_delta',
                    'median_rank_delta', 'mean_rank', 'median_rank'])
        for ds in DATASETS:
            d = per_ds_detail[ds]
            base = ranks['Uniform'][ds]
            m_in = d['present']
            for m in ('MinNLL', 'LSE', 'Gate', 'Hybrid'):
                r = ranks[m][ds]
                a_ = np.where(base[m_in] < 0, 10 ** 6, base[m_in])
                b_ = np.where(r[m_in] < 0, 10 ** 6, r[m_in])
                delta = a_ - b_                       # positive = improved
                w.writerow([ds, int(m_in.sum()), m,
                            int((delta > 0).sum()), int((delta == 0).sum()),
                            int((delta < 0).sum()),
                            f'{delta.mean():.4f}', f'{np.median(delta):.1f}',
                            f'{b_[b_ < 10**6].mean():.4f}' if (b_ < 10**6).any() else '',
                            f'{np.median(b_[b_ < 10**6]):.1f}' if (b_ < 10**6).any() else ''])

    # ---------------- uniform_failure_rescue.csv (core diagnostic) ----------------
    with (out / 'test' / 'uniform_failure_rescue.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'n_uniform_failures', 'method', 'rescued_top5',
                    'rescued_top10', 'rescued_top20', 'rescue_rate_top20'])
        for ds in DATASETS:
            d = per_ds_detail[ds]
            base = ranks['Uniform'][ds]
            fail = d['present'] & ((base < 0) | (base > 20))
            nf = int(fail.sum())
            for m in ('MinNLL', 'LSE', 'Gate', 'Hybrid'):
                r = ranks[m][ds][fail]
                w.writerow([ds, nf, m,
                            int(((r >= 1) & (r <= 5)).sum()),
                            int(((r >= 1) & (r <= 10)).sum()),
                            int(((r >= 1) & (r <= 20)).sum()),
                            f'{((r >= 1) & (r <= 20)).mean():.6f}' if nf else ''])

    # ---------------- subgroup_results.csv (Only-F1 / Only-F2) ----------------
    with (out / 'test' / 'subgroup_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'group', 'n', 'method', 'NDCG@20', 'Recall@20'])
        for ds in DATASETS:
            d = per_ds_detail[ds]
            r1 = rank_of_target(d['g'], d['scores']['BestF1'], d['tgt'])
            r2 = rank_of_target(d['g'], d['scores']['BestF2'], d['tgt'])
            h1 = (r1 >= 1) & (r1 <= 20); h2 = (r2 >= 1) & (r2 <= 20)
            groups = {'Only_F1': h1 & ~h2, 'Only_F2': h2 & ~h1, 'Both': h1 & h2,
                      'Neither': ~h1 & ~h2}
            for gname, gm in groups.items():
                if gm.sum() == 0:
                    continue
                for m in ('Uniform', 'MinNLL', 'LSE', 'Gate', 'Hybrid', 'Oracle'):
                    s = summarize(ranks[m][ds][gm], (20,))
                    w.writerow([ds, gname, int(gm.sum()), m,
                                f'{s["NDCG@20"]:.6f}', f'{s["Recall@20"]:.6f}'])

    # ---------------- PHASE J: mechanism diagnostics ----------------
    write_json(out / 'test' / 'mechanism_diagnostics.json', _mechanism(per_ds_detail, out))

    man['test_evaluated'] = True
    man['test_evaluated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    man_p.write_text(json.dumps(man, indent=2, default=str))

    print('\n=== TEST NDCG@20 (pooled) ===')
    for m in METHODS:
        r = np.concatenate([ranks[m][ds] for ds in DATASETS])
        print(f'  {m:9s} {summarize(r)["NDCG@20"]:.6f}')
    print('\n=== per dataset: uniform vs LSE vs oracle ===')
    for ds in DATASETS:
        u_ = summarize(ranks['Uniform'][ds])['NDCG@20']
        l_ = summarize(ranks['LSE'][ds])['NDCG@20']
        o_ = summarize(ranks['Oracle'][ds])['NDCG@20']
        print(f'  {SHORT[ds]:12s} uniform={u_:.6f}  LSE={l_:.6f}  oracle={o_:.6f}')
    print('done', flush=True)
    return 0


def _oracle_rank(g, tgt):
    """Best achievable rank within the candidate union given perfect candidate choice.

    Uses the target only to RANK methods, never to score candidates: the oracle rank of a
    user whose target is in the union is 1, otherwise -1 (the candidate generator missed
    it).  This separates reranking failure from candidate-generation failure (section 40).
    """
    n = len(g['users'])
    r = np.full(n, -1, dtype=np.int64)
    for u in range(n):
        s, e = g['starts'][u], g['ends'][u]
        if tgt[u] >= 0 and (g['item'][s:e] == tgt[u]).any():
            r[u] = 1
    return r


def _mechanism(detail, out):
    """Phase J: does the candidate NLL actually explain the win?  Evaluation only."""
    res = {}
    for ds, d in detail.items():
        g = d['g']; tgt = d['tgt']
        # target vs non-target LSE score, pairwise AUC
        lse = np.empty_like(d['z1'])
        m = np.maximum(-d['z1'], -d['z2'])
        lse = m + np.log(np.exp(-d['z1'] - m) + np.exp(-d['z2'] - m))
        num = den = 0
        tgt_delta = []
        for u in range(len(g['users'])):
            s, e = g['starts'][u], g['ends'][u]
            seg = lse[s:e]; it = g['item'][s:e]
            hit = np.where(it == tgt[u])[0]
            if len(hit) == 0:
                continue
            ts = seg[hit[0]]
            others = np.delete(seg, hit[0])
            if len(others) == 0:
                continue
            num += float((ts > others).sum()); den += len(others)
            tgt_delta.append(float(d['z1'][s + hit[0]] - d['z2'][s + hit[0]]))
        res[ds] = {'pairwise_auc_target_vs_nontarget_LSE':
                       (num / den) if den else None,
                   'n_pairs': int(den),
                   'mean_target_delta_nll': float(np.mean(tgt_delta)) if tgt_delta else None,
                   'positive_delta_share': float(np.mean(np.array(tgt_delta) > 0))
                       if tgt_delta else None}
    return res


if __name__ == '__main__':
    sys.exit(main())
