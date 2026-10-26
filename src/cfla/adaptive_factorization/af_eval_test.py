#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B — TEST evaluation for the adaptive factorization selector.

ORDER OF OPERATIONS IS THE POINT OF THIS FILE:

  1. `write_freeze_manifest()` runs FIRST and hashes everything that defines the
     selector (features, scaler, architecture, checkpoint, loss, hyper-parameters,
     selected static weight, RouterDev result).  After it is written the design is
     frozen.
  2. Only then is the canonical TEST split read.

If step 1 cannot be completed the script aborts before touching test, so a missing
manifest can never be papered over by a test result.

Evaluated inference modes
  A  best single factorization
  B  uniform fusion                     w = (0.5, 0.5)
  C  static validation-selected weight  w = (w*, 1-w*)
  D  Router H hard
  E  Router H soft
  F  Router P hard
  G  Router P soft
     Oracle (explanatory only, uses the ground-truth target, NOT deployable)

Usage
  python3 af_eval_test.py --selectordir <selector dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from af_common import (FusedCandidates, fused_rank, per_user_ndcg,  # noqa: E402
                       paired_bootstrap, mcnemar_exact, summarize_rank)
from af_router_train import (DATASET_ORDER, LinearRouter, MLPRouter,  # noqa: E402
                             count_params, read_split_csv)
from xt_common import sha256_file  # noqa: E402

METRIC_KS = (5, 10, 20)
BOOT_N = 5000
BOOT_SEED = 20260912


def git_sha():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                           cwd=str(HERE.parents[2]))
        return r.stdout.strip() or None
    except Exception:
        return None


def load_test_cells(sel):
    d = Path(sel) / 'features'
    out = {}
    for ds in DATASET_ORDER:
        p = d / f'{ds}_test.npz'
        if not p.exists():
            raise FileNotFoundError(f'test cell missing: {p}')
        out[ds] = dict(np.load(p, allow_pickle=True))
    return out


def cand_of(z):
    return FusedCandidates(np.asarray(z['cand_ids1']), np.asarray(z['cand_p1']),
                           np.asarray(z['cand_ids2']), np.asarray(z['cand_p2']),
                           np.asarray(z['cand_target']))


def load_scaler(sel, key):
    z = np.load(Path(sel) / 'scalers' / f'{key}_scaler.npz')
    return z['mean'], z['std']


def build_router(rec, sel):
    ck = Path(sel) / 'checkpoints' / f'{rec["tag"]}.pth'
    d = torch.load(ck, map_location='cpu')
    model = LinearRouter(d['d_in']) if d['kind'] == 'linear' else MLPRouter(d['d_in'])
    model.load_state_dict(d['state_dict'])
    model.eval()
    return model, d, ck


def write_freeze_manifest(sel, dev_summary, split_csv, static_w, chosen_tag, extra):
    sel = Path(sel)
    feats = sorted(str(p.relative_to(sel)) for p in (sel / 'features').glob('*.npz'))
    scalers = sorted(str(p.relative_to(sel)) for p in (sel / 'scalers').glob('*.npz'))
    ckpts = sorted(str(p.relative_to(sel)) for p in (sel / 'checkpoints').glob('*.pth'))
    chosen_ck = sel / 'checkpoints' / f'{chosen_tag}.pth'
    man = {
        'study': 'target-free adaptive factorization selector',
        'git_sha': git_sha(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'host': platform.node(), 'platform': platform.platform(),
        'features': {
            'router_h_dim': dev_summary['router_h_dim'],
            'router_p_dim': dev_summary['router_p_dim'],
            'router_h_names': dev_summary['feature_names_h'],
            'router_p_names': dev_summary['feature_names_p'],
            'target_free': True,
            'user_id_feature': False,
            'target_nll_feature': False,
            'test_labels_read_before_freeze': False,
        },
        'feature_files': feats + scalers,
        'scaler_files': scalers,
        'scaler_hash': dev_summary['scaler_hash'],
        'scaler_fitted_on': 'RouterTrain only',
        'router_checkpoints': ckpts,
        'chosen_router': chosen_tag,
        # Two hashes are recorded because two different, both-deterministic digests
        # exist for a checkpoint and mixing them up looks like a provenance break:
        #   * file hash        = SHA-256 of the .pth bytes on disk
        #   * state_dict hash  = SHA-256 of the re-serialised state_dict
        # torch.save() container bytes and a re-serialisation of the loaded state are
        # not the same byte string, though each is stable across runs.
        'chosen_router_sha256': sha256_file(chosen_ck),
        'chosen_router_state_dict_sha256': next(
            (r['checkpoint_sha256'] for r in dev_summary['results']
             if r['tag'] == chosen_tag), None),
        'hash_method': {
            'chosen_router_sha256': 'SHA-256 of the checkpoint .pth file bytes',
            'chosen_router_state_dict_sha256':
                'SHA-256 of torch.save() of the loaded state_dict (as recorded per '
                'router in router_dev_results.csv)',
        },
        'architecture': {'family': chosen_tag,
                         'layers': 1 if 'linear' in chosen_tag else 2,
                         'hidden': 64 if 'mlp' in chosen_tag else None,
                         'activation': 'GELU' if 'mlp' in chosen_tag else None,
                         'dropout': 0.1 if 'mlp' in chosen_tag else None,
                         'output': '2 logits -> softmax'},
        'loss': 'soft-label cross entropy, L = -sum_m q_m log w_m, '
                'sample weight 0.25 when U1 = U2 = 0 else 1.0',
        'label': {'temperature': 0.1, 'utility': '1/log2(r+1) for r <= 20 else 0',
                  'supervision_event': 'chronological canonical VALIDATION instance',
                  'q_when_both_zero': [0.5, 0.5]},
        'hyperparameters': dev_summary['training'],
        'static_weight': static_w,
        'user_split': {'file': str(Path(split_csv).relative_to(sel)),
                       'seed': dev_summary['split_seed'],
                       'fractions': dev_summary['split_fractions'],
                       'sha256': sha256_file(split_csv)},
        'router_dev_results': dev_summary['results'],
        'router_dev_static_grid': dev_summary['static_grid'],
        'best_static': dev_summary['best_static'],
        'uniform_dev_fusion_ndcg20': dev_summary['uniform_dev_fusion_ndcg20'],
        'router_train_n': dev_summary['router_train_n'],
        'router_dev_n': dev_summary['router_dev_n'],
        'router_holdout_n': dev_summary['router_holdout_n'],
        'selection_rule': 'best RouterDev soft-fusion NDCG@20 among '
                          '{Router H, Router P} x {linear, MLP}',
        'test_metrics_computed_at_freeze_time': False,
        **extra,
    }
    p = sel / 'SELECTOR_FREEZE_MANIFEST.json'
    p.write_text(json.dumps(man, indent=2))
    return p, man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selectordir', required=True)
    ap.add_argument('--n_boot', type=int, default=BOOT_N)
    a = ap.parse_args(argv)
    sel = Path(a.selectordir)

    dev_summary = json.loads((sel / 'router_dev_summary.json').read_text())
    dev_results = dev_summary['results']
    # RouterDev-based selection: the single target-free adaptive selector for the
    # headline comparison is the best RouterDev soft-fusion router.
    chosen = max(dev_results, key=lambda r: r['dev_fusion_ndcg20'])
    chosen_tag = chosen['tag']
    static_w = float(dev_summary['best_static']['w_F1'])

    # ---------------- STEP 1: FREEZE (test not yet read) ----------------
    man_p, man = write_freeze_manifest(
        sel, dev_summary, sel / 'router_user_split.csv', static_w, chosen_tag,
        {'phase': 'FREEZE_WRITTEN_BEFORE_TEST_READ'})
    print(f'[freeze] wrote {man_p}', flush=True)
    print(f'[freeze] chosen router = {chosen_tag} '
          f'(RouterDev NDCG@20 {chosen["dev_fusion_ndcg20"]:.6f})', flush=True)
    print(f'[freeze] static w_F1 = {static_w}', flush=True)

    # ---------------- STEP 2: TEST ----------------
    test = load_test_cells(sel)
    split_assign = read_split_csv(sel / 'router_user_split.csv')

    routers = {}
    for r in dev_results:
        model, d, ck = build_router(r, sel)
        routers[r['tag']] = (model, d, r)

    def zscore(key, X):
        mu, sd = load_scaler(sel, key)
        return (X - mu) / sd

    pred = {}   # tag -> {ds: w1 vector}
    for tag, (model, d, r) in routers.items():
        key = 'router_h' if d['feat_key'] == 'H' else 'router_p'
        pred[tag] = {}
        for ds in DATASET_ORDER:
            X = zscore(key, np.asarray(test[ds][key]))
            with torch.no_grad():
                w = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
            pred[tag][ds] = w[:, 0]

    # ---- per-dataset ranks for every mode ----
    ranks = {k: {} for k in ('A_F1', 'A_F2', 'B_uniform', 'C_static',
                             'D_H_hard', 'E_H_soft', 'F_P_hard', 'G_P_soft', 'Oracle')}
    hard_choice = {}
    for ds in DATASET_ORDER:
        c = cand_of(test[ds])
        n = len(c)
        ranks['A_F1'][ds] = fused_rank(1.0, c)
        ranks['A_F2'][ds] = fused_rank(0.0, c)
        ranks['B_uniform'][ds] = fused_rank(0.5, c)
        ranks['C_static'][ds] = fused_rank(static_w, c)
        # Router H / Router P each use their own best-on-dev member of the family
        h_tags = [t for t in pred if t.startswith('router_H_')]
        p_tags = [t for t in pred if t.startswith('router_P_')]
        h_best = max(h_tags, key=lambda t: next(r['dev_fusion_ndcg20'] for r in dev_results
                                                if r['tag'] == t))
        p_best = max(p_tags, key=lambda t: next(r['dev_fusion_ndcg20'] for r in dev_results
                                                if r['tag'] == t))
        wH = pred[h_best][ds]
        wP = pred[p_best][ds]
        # hard routing: argmax, ties -> F1
        hc = np.where(wH >= 0.5, 1.0, 0.0)
        pc = np.where(wP >= 0.5, 1.0, 0.0)
        ranks['D_H_hard'][ds] = fused_rank(hc, c)
        ranks['E_H_soft'][ds] = fused_rank(wH, c)
        ranks['F_P_hard'][ds] = fused_rank(pc, c)
        ranks['G_P_soft'][ds] = fused_rank(wP, c)
        r1 = fused_rank(1.0, c); r2 = fused_rank(0.0, c)
        oracle = np.where(r1 < 0, r2, np.where(r2 < 0, r1, np.minimum(r1, r2)))
        oracle = np.where((r1 < 0) & (r2 < 0), -1, oracle)
        ranks['Oracle'][ds] = oracle
        hard_choice[ds] = {'H': hc, 'P': pc, 'rank1': r1, 'rank2': r2}
        pred['_h_best'] = h_best; pred['_p_best'] = p_best
    print(f'[test] Router H family best on dev: {pred["_h_best"]}', flush=True)
    print(f'[test] Router P family best on dev: {pred["_p_best"]}', flush=True)

    # ---- test_results.csv ----
    with (sel / 'test_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'mode', 'n'] + [f'Recall@{k}' for k in METRIC_KS]
                   + [f'NDCG@{k}' for k in METRIC_KS])
        for ds in DATASET_ORDER:
            for mode in ranks:
                s = summarize_rank(ranks[mode][ds], METRIC_KS)
                w.writerow([ds, mode, s['n']] + [f'{s[f"Recall@{k}"]:.6f}' for k in METRIC_KS]
                           + [f'{s[f"NDCG@{k}"]:.6f}' for k in METRIC_KS])
        # pooled
        for mode in ranks:
            r = np.concatenate([ranks[mode][ds] for ds in DATASET_ORDER])
            s = summarize_rank(r, METRIC_KS)
            w.writerow(['ALL_POOLED', mode, s['n']]
                       + [f'{s[f"Recall@{k}"]:.6f}' for k in METRIC_KS]
                       + [f'{s[f"NDCG@{k}"]:.6f}' for k in METRIC_KS])

    # ---- bootstrap: adaptive vs uniform, adaptive vs static ----
    with (sel / 'bootstrap.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['comparison', 'dataset', 'metric', 'delta', 'ci_low', 'ci_high',
                    'p_value', 'n_users', 'n_boot'])
        for metric_k in METRIC_KS:
            for ds in DATASET_ORDER + ['ALL_POOLED']:
                def vec(mode):
                    r = (ranks[mode][ds] if ds != 'ALL_POOLED'
                         else np.concatenate([ranks[mode][d] for d in DATASET_ORDER]))
                    return per_user_ndcg(r, metric_k)
                adv = vec('G_P_soft')
                for other, label in (('B_uniform', 'G_P_soft_vs_B_uniform'),
                                     ('C_static', 'G_P_soft_vs_C_static')):
                    st = paired_bootstrap(adv, vec(other), n_boot=a.n_boot, seed=BOOT_SEED)
                    w.writerow([label, ds, f'NDCG@{metric_k}', f'{st["delta"]:.8f}',
                                f'{st["ci_low"]:.8f}', f'{st["ci_high"]:.8f}',
                                f'{st["p_value"]:.6f}', st['n_users'], st['n_boot']])

    # ---- preference diagnostics ----
    with (sel / 'preference_diagnostics.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'router', 'group', 'n', 'accuracy', 'balanced_accuracy',
                    'auroc', 'mean_w_F1', 'hard_matches_better_rank'])
        for ds in DATASET_ORDER:
            z = test[ds]
            r1 = hard_choice[ds]['rank1']; r2 = hard_choice[ds]['rank2']
            better = np.where((r1 >= 1) & (r2 >= 1), np.where(r1 < r2, 1,
                             np.where(r2 < r1, 2, 0)),
                              np.where(r1 >= 1, 1, np.where(r2 >= 1, 2, 0)))
            for tag in (pred['_h_best'], pred['_p_best']):
                wq = pred[tag][ds]
                hard = np.where(wq > 0.5, 1, 2)
                for gname, gmask in (('all', np.ones(len(r1), dtype=bool)),
                                     ('Only_F1', (r1 >= 1) & (r2 < 0)),
                                     ('Only_F2', (r2 >= 1) & (r1 < 0)),
                                     ('Both', (r1 >= 1) & (r2 >= 1)),
                                     ('Neither', (r1 < 0) & (r2 < 0))):
                    sub = gmask & (better > 0) & (wq != 0.5)
                    if sub.sum() == 0:
                        w.writerow([ds, tag, gname, 0, '', '', '', '', ''])
                        continue
                    acc = float((hard[sub] == better[sub]).mean())
                    pos = better[sub] == 1
                    neg = better[sub] == 2
                    tpr = float((hard[sub][pos] == 1).mean()) if pos.any() else 0.0
                    tnr = float((hard[sub][neg] == 2).mean()) if neg.any() else 0.0
                    bal = (tpr + tnr) / 2.0
                    # AUROC from the continuous w_F1 score against "F1 is better"
                    lab = (better[sub] == 1).astype(int)
                    if 0 < lab.sum() < len(lab):
                        order = np.argsort(wq[sub])
                        ranks_ = np.empty(len(lab)); ranks_[order] = np.arange(1, len(lab) + 1)
                        n1 = int(lab.sum()); n0 = len(lab) - n1
                        auroc = float((ranks_[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
                    else:
                        auroc = float('nan')
                    w.writerow([ds, tag, gname, int(sub.sum()), f'{acc:.6f}',
                                f'{bal:.6f}', f'{auroc:.6f}', f'{float(wq[sub].mean()):.6f}',
                                f'{acc:.6f}'])

    # ---- NLL diagnostics (needs the separately dumped frozen-checkpoint NLL) ----
    nll_dir = Path(sel) / 'features' / 'nll'
    with (sel / 'nll_diagnostics.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'n', 'router', 'router_argmin_matches_nll_argmin',
                    'nll_argmin_matches_better_rank', 'mean_nll_F1', 'mean_nll_F2'])
        for ds in DATASET_ORDER:
            p1 = nll_dir / f'{ds}_spectral_k3_test.npz'
            p2 = nll_dir / f'{ds}_rqvae3_test.npz'
            if not (p1.exists() and p2.exists()):
                w.writerow([ds, 0, '', 'NLL_UNAVAILABLE', '', '', ''])
                continue
            n1 = np.load(p1)['total_nll']; n2 = np.load(p2)['total_nll']
            nll_choice = np.where(n1 < n2, 1, 2)
            r1 = hard_choice[ds]['rank1']; r2 = hard_choice[ds]['rank2']
            better = np.where((r1 >= 1) & (r2 >= 1),
                              np.where(r1 < r2, 1, np.where(r2 < r1, 2, 0)),
                              np.where(r1 >= 1, 1, np.where(r2 >= 1, 2, 0)))
            for tag in (pred['_h_best'], pred['_p_best']):
                wq = pred[tag][ds]
                hard = np.where(wq >= 0.5, 1, 2)
                agree_nll = float((hard == nll_choice).mean())
                m = better > 0
                agree_better = float((nll_choice[m] == better[m]).mean()) if m.any() else float('nan')
                w.writerow([ds, len(n1), tag, f'{agree_nll:.6f}', f'{agree_better:.6f}',
                            f'{float(n1.mean()):.6f}', f'{float(n2.mean()):.6f}'])

    # ---- subgroup results ----
    with (sel / 'subgroup_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'group', 'n', 'mode', 'NDCG@20', 'Recall@20',
                    'mean_pred_F1_weight', 'mean_pred_F2_weight',
                    'hard_selection_accuracy'])
        for ds in DATASET_ORDER:
            r1 = hard_choice[ds]['rank1']; r2 = hard_choice[ds]['rank2']
            wq = pred[pred['_p_best']][ds]; wh = pred[pred['_h_best']][ds]
            groups = {
                'Only_F1': (r1 >= 1) & (r2 < 0),
                'Only_F2': (r2 >= 1) & (r1 < 0),
                'Both': (r1 >= 1) & (r2 >= 1),
                'Neither': (r1 < 0) & (r2 < 0),
            }
            better = np.where((r1 >= 1) & (r2 >= 1),
                              np.where(r1 < r2, 1, np.where(r2 < r1, 2, 0)),
                              np.where(r1 >= 1, 1, np.where(r2 >= 1, 2, 0)))
            hard = np.where(wq >= 0.5, 1, 2)
            for gname, gm in groups.items():
                if gm.sum() == 0:
                    continue
                for mode in ('B_uniform', 'C_static', 'D_H_hard', 'E_H_soft',
                             'F_P_hard', 'G_P_soft', 'Oracle'):
                    r = ranks[mode][ds][gm]
                    s = summarize_rank(r, (20,))
                    m2 = gm & (better > 0)
                    # only count a decision the router actually made: a soft weight of
                    # exactly 0.5 is a tie, not a choice, so it is excluded rather than
                    # credited to the F1 tie-break
                    decided = m2 & (wq != 0.5)
                    acc = (float((hard[decided] == better[decided]).mean())
                           if decided.any() else float('nan'))
                    w.writerow([ds, gname, int(gm.sum()), mode,
                                f'{s["NDCG@20"]:.6f}', f'{s["Recall@20"]:.6f}',
                                f'{float(wq[gm].mean()):.6f}', f'{float(1-wq[gm].mean()):.6f}',
                                f'{acc:.6f}' if acc == acc else ''])

    # ---- seen/unseen-user robustness ----
    with (sel / 'robustness_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'validation_role', 'n', 'mode', 'NDCG@20', 'Recall@20'])
        for ds in DATASET_ORDER:
            roles = split_assign[ds]
            for role in ('RouterTrain', 'RouterDev', 'RouterHoldout'):
                gm = roles == role
                if gm.sum() == 0:
                    continue
                for mode in ('B_uniform', 'C_static', 'F_P_hard', 'G_P_soft', 'Oracle'):
                    s = summarize_rank(ranks[mode][ds][gm], (20,))
                    w.writerow([ds, role, int(gm.sum()), mode, f'{s["NDCG@20"]:.6f}',
                                f'{s["Recall@20"]:.6f}'])

    # ---- finalise manifest with the post-freeze facts ----
    man['test_evaluated'] = True
    man['test_evaluated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    man['test_result_files'] = ['test_results.csv', 'bootstrap.csv',
                                'preference_diagnostics.csv', 'nll_diagnostics.csv',
                                'subgroup_results.csv', 'robustness_results.csv']
    man['router_family_best_on_dev'] = {'H': pred['_h_best'], 'P': pred['_p_best']}
    man_p.write_text(json.dumps(man, indent=2))

    # console summary
    print('\n=== TEST NDCG@20 (pooled) ===', flush=True)
    for mode in ranks:
        r = np.concatenate([ranks[mode][d] for d in DATASET_ORDER])
        print(f'  {mode:10s} {summarize_rank(r, (20,))["NDCG@20"]:.6f}', flush=True)
    print('\n=== per dataset NDCG@20: uniform vs Router P soft vs oracle ===', flush=True)
    for ds in DATASET_ORDER:
        u = summarize_rank(ranks['B_uniform'][ds], (20,))['NDCG@20']
        g = summarize_rank(ranks['G_P_soft'][ds], (20,))['NDCG@20']
        o = summarize_rank(ranks['Oracle'][ds], (20,))['NDCG@20']
        b = summarize_rank(ranks['A_F1'][ds], (20,))['NDCG@20']
        b2 = summarize_rank(ranks['A_F2'][ds], (20,))['NDCG@20']
        print(f'  {ds:32s} F1={b:.6f} F2={b2:.6f} uniform={u:.6f} '
              f'routerP_soft={g:.6f} oracle={o:.6f}', flush=True)
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
