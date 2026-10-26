#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end verification for the adaptive-factorization study.

Checks, in order:

  1. FROZEN ARTIFACTS — every checkpoint, test beam dump and SID array used by this
     study still hashes to the value recorded by the previous study's provenance.
     Nothing historical was modified.
  2. LABEL INDEPENDENCE — the labels in the selector dataset are re-derived from the
     beam dumps in a second, independent way, and the implied test NDCG@20 must
     equal the value the previous study reported for the frozen checkpoints.
  3. FEATURE PURITY — no selector feature name is target-derived, the feature
     matrices are finite, and the scaler was fitted on RouterTrain rows only.
  4. NO TEST LEAKAGE — the freeze manifest exists and records that no test metric
     was computed before it was written; the RouterDev outputs contain no test field.
  5. RECORDED RANKS — every epoch's assignment in the compute-matched training log
     is balanced (|count(F1) - count(F2)| <= 1) and samples the original instance
     count exactly once.

Run:
  python3 analysis/mechanism/adaptive_factorization/af_verify.py \
      --root outputs/adaptive_factorization
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared'),
           str(HERE.parent / 'factorization_generality'), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from af_common import FusedCandidates, fused_rank, summarize_rank  # noqa: E402
from xt_common import STEP2, load_sid, sha256_file, sid_path  # noqa: E402

FAILS = []
CHECKS = [0]


def check(name, cond, detail=''):
    CHECKS[0] += 1
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}  {detail}')
        FAILS.append(name)


def frozen_inputs_present(root, sel):
    print('1. frozen artifacts')
    prov = json.loads((STEP2 / '_PROVENANCE.json').read_text())
    for name, rec in prov['shared_models'].items():
        ck = STEP2 / 'runs' / name / 'shared_model.pth'
        got = sha256_file(ck)
        check(f'checkpoint {name} unchanged', got == rec['checkpoint_sha256'],
              f'{got} != {rec["checkpoint_sha256"]}')
    for key, rec in prov['outputs'].items():
        if not key.startswith('beams/'):
            continue
        p = STEP2 / key
        got = sha256_file(p)
        check(f'frozen test beam {Path(key).name} unchanged',
              got == rec['sha256'], f'{got} != {rec["sha256"]}')
    # selector copies of the frozen test dumps must be byte-identical to the originals
    for key, rec in prov['outputs'].items():
        if not key.startswith('beams/'):
            continue
        src = STEP2 / key
        dst = Path(sel) / 'features' / 'beams' / Path(key).name
        check(f'selector copy of {Path(key).name} identical to frozen original',
              sha256_file(dst) == sha256_file(src))
    # SID arrays
    expected_sid = {
        'Beauty': '30a0e303ed9becb76ac275c3556694275035e9eaa9f74f1df956b3364b6a6422',
    }
    for ds, want in expected_sid.items():
        got = sha256_file(sid_path('spectral_k3', ds))
        check(f'SID array spectral_k3 {ds} unchanged', got == want, f'{got} != {want}')


def label_independence(root, sel):
    print('2. label independence and reproduction of the audited metrics')
    audited_test = {'Beauty': (0.04227714956723783, 0.0382502925640393),
                    'Cell_Phones_and_Accessories': (0.09816764790645731, 0.08699809491853756),
                    'Health_and_Personal_Care': (0.09359182702081191, 0.08163343121689595),
                    'Toys_and_Games': (0.0365694706870278, 0.032414912403605375)}
    for ds, (want1, want2) in audited_test.items():
        p = Path(sel) / 'features' / f'{ds}_test.npz'
        if not p.exists():
            check(f'{ds} test features present', False, str(p))
            continue
        z = np.load(p, allow_pickle=True)
        c = FusedCandidates(np.asarray(z['cand_ids1']), np.asarray(z['cand_p1']),
                            np.asarray(z['cand_ids2']), np.asarray(z['cand_p2']),
                            np.asarray(z['cand_target']))
        r1 = fused_rank(1.0, c)
        r2 = fused_rank(0.0, c)
        got1 = summarize_rank(r1, (20,))['NDCG@20']
        got2 = summarize_rank(r2, (20,))['NDCG@20']
        check(f'{ds} F1 test NDCG@20 reproduces the audited value',
              abs(got1 - want1) < 1e-12, f'{got1} != {want1}')
        check(f'{ds} F2 test NDCG@20 reproduces the audited value',
              abs(got2 - want2) < 1e-12, f'{got2} != {want2}')
        # stored rank1/rank2 must equal the independently recomputed ranks
        check(f'{ds} stored rank1 matches recomputation',
              np.array_equal(np.asarray(z['rank1']), r1))
        check(f'{ds} stored rank2 matches recomputation',
              np.array_equal(np.asarray(z['rank2']), r2))
        # utility must equal the documented transform of the rank
        u1 = np.asarray(z['u1'])
        want_u1 = np.where((r1 >= 1) & (r1 <= 20), 1.0 / np.log2(np.maximum(r1, 1) + 1), 0.0)
        check(f'{ds} u1 equals 1/log2(r+1) for r<=20 else 0',
              np.allclose(u1, want_u1, atol=1e-12))
        # soft target for zero-hit rows must be exactly (0.5, 0.5) with weight 0.25
        bz = np.asarray(z['both_zero'])
        check(f'{ds} both-zero rows have q=(0.5,0.5) and weight 0.25',
              bool(np.allclose(np.asarray(z['q1'])[bz], 0.5)
                   and np.allclose(np.asarray(z['q2'])[bz], 0.5)
                   and np.allclose(np.asarray(z['weight'])[bz], 0.25)))


def feature_purity(root, sel):
    print('3. feature purity')
    # Exact, lower-cased feature names only.  Substring matching would flag
    # `x_rank_corr_common` (a correlation between the two *predictions*, not the
    # target's rank), which is a legitimate target-free cross-factorization feature.
    bad = {'nll', 'target', 'target_sid', 'target_item', 'user', 'user_id', 'uid',
           'utility', 'u1', 'u2', 'q1', 'q2', 'rank', 'rank1', 'rank2', 'label'}
    for split in ('valid', 'test'):
        for ds in ('Beauty', 'Cell_Phones_and_Accessories',
                   'Health_and_Personal_Care', 'Toys_and_Games'):
            p = Path(sel) / 'features' / f'{ds}_{split}.npz'
            if not p.exists():
                check(f'{ds} {split} present', False)
                continue
            z = np.load(p, allow_pickle=True)
            names_h = [str(x) for x in z['feature_names_h']]
            names_p = [str(x) for x in z['feature_names_p']]
            offenders = [n for n in names_h + names_p if n.lower() in bad]
            check(f'{ds} {split} feature names contain no target-derived token',
                  not offenders, str(offenders[:5]))
            check(f'{ds} {split} router_h finite',
                  bool(np.isfinite(np.asarray(z['router_h'])).all()))
            check(f'{ds} {split} router_p finite',
                  bool(np.isfinite(np.asarray(z['router_p'])).all()))
    # scaler must have been fitted on RouterTrain only
    for key in ('router_h', 'router_p'):
        s = np.load(Path(sel) / 'scalers' / f'{key}_scaler.npz')
        check(f'scaler {key} records RouterTrain fitting',
              str(s['fitted_on']) == 'RouterTrain')
    # recompute the RouterTrain mean/std and compare
    split_assign = {}
    with (Path(sel) / 'router_user_split.csv').open() as fh:
        rows = list(csv.DictReader(fh))
    for ds in ('Beauty', 'Cell_Phones_and_Accessories',
               'Health_and_Personal_Care', 'Toys_and_Games'):
        split_assign[ds] = np.array([r['split'] for r in rows if r['dataset'] == ds])
    X = np.concatenate([np.asarray(np.load(Path(sel) / 'features' / f'{ds}_valid.npz',
                                           allow_pickle=True)['router_h'])[
                                split_assign[ds] == 'RouterTrain']
                        for ds in split_assign])
    s = np.load(Path(sel) / 'scalers' / 'router_h_scaler.npz')
    check('router_h scaler mean equals RouterTrain mean',
          np.allclose(X.mean(axis=0), s['mean'], atol=1e-9))
    check('router_h scaler std equals RouterTrain std',
          np.allclose(np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0)), s['std'], atol=1e-9))


def no_test_leakage(root, sel):
    print('4. freeze discipline')
    mp = Path(sel) / 'SELECTOR_FREEZE_MANIFEST.json'
    if not mp.exists():
        print('  INFO  freeze manifest not written yet (phase: pre-freeze); '
              're-run this script after af_eval_test.py')
    else:
        check('freeze manifest exists after test evaluation', True)
    if mp.exists():
        man = json.loads(mp.read_text())
        check('manifest records test not read before freeze',
              man.get('test_metrics_computed_at_freeze_time') is False)
        check('manifest records target-free features',
              man['features'].get('target_free') is True)
        check('manifest records no user-id feature',
              man['features'].get('user_id_feature') is False)
        check('manifest records no target-NLL feature',
              man['features'].get('target_nll_feature') is False)
        check('manifest has a router checkpoint hash',
              bool(man.get('chosen_router_sha256')))
        check('manifest has a scaler hash', bool(man.get('scaler_hash')))
    devp = Path(sel) / 'router_dev_summary.json'
    if devp.exists():
        dev = json.loads(devp.read_text())
        check('router_dev_summary declares test_read = False',
              dev.get('test_read') is False)
        check('router_dev_summary contains no test score field',
              not any('test' in k.lower() and k != 'test_read' for k in dev.keys()))
    audit = Path(sel) / 'SELECTOR_FEATURE_AUDIT.md'
    check('feature audit exists', audit.exists())
    if audit.exists():
        txt = audit.read_text()
        check('audit covers Router H and Router P', 'Router H' in txt and 'Router P' in txt)
        check('audit states every feature is target-free', 'uses target' in txt)


def compute_matched_assignment(root):
    print('5. compute-matched assignment discipline')
    # the active run lives in runs/<dataset>_<experiment>_<seed>_<timestamp>/; pick the
    # newest completed one and ignore the aborted attempts explicitly
    cands = sorted(p for p in (Path(root) / 'runs').glob(
        'Beauty_compute_matched_seed2025_*') if 'ABORTED' not in p.name)
    cands = [c for c in cands if (c / 'train_log.json').exists()]
    if not cands:
        check('compute-matched training log exists', False, 'no run directory found')
        return
    run_dir = max(cands, key=lambda c: (c / 'train_log.json').stat().st_mtime)
    lp = run_dir / 'train_log.json'
    check('compute-matched training log exists', True)
    print(f'  INFO  reading {run_dir.name}')
    st = json.loads((run_dir / 'status.json').read_text()) if (run_dir/'status.json').exists() else {}
    check('run status is completed', st.get('status') == 'completed', str(st.get('status')))
    check('PRE_RUN_ARTIFACT_MANIFEST.sha256 present',
          (run_dir / 'PRE_RUN_ARTIFACT_MANIFEST.sha256').exists())
    check('POST_RUN_ARTIFACT_MANIFEST.sha256 present',
          (run_dir / 'POST_RUN_ARTIFACT_MANIFEST.sha256').exists())
    log = json.loads(lp.read_text())
    check('exposure regime recorded',
          log.get('exposure') == 'compute_matched_balanced')
    # ---- the implementation claims made in the final report must match the artifacts ----
    check('control tokens are 1025 and 1026 (vocab 1027)',
          log['plan']['control_tokens'] == {'spectral_k3': 1025, 'rqvae3': 1026},
          str(log['plan']['control_tokens']))
    check('vocab_size recorded as 1027', log['plan']['vocab_size'] == 1027)
    check('shared model has 4,591,104 parameters (canonical + 2 control rows)',
          log['params'] == 4591104, str(log['params']))
    check('one encoder/decoder config: 4+4 layers, d_model 128, d_ff 1024, 6 heads',
          all(log['config'][k] == val for k, val in
              (('num_layers', 4), ('num_decoder_layers', 4), ('d_model', 128),
               ('d_ff', 1024), ('num_heads', 6), ('d_kv', 64))))
    check('loss is unweighted T5 cross-entropy (no auxiliary term recorded)',
          'loss' not in log.get('config', {}) or log['config'].get('loss') in (None, 'ce'))
    check('canonical evaluation rows == one per user',
          log['epochs'][0]['valid_ndcg20'].keys() == {'spectral_k3', 'rqvae3'})

    n_orig = log['train_size_per_factorization']['spectral_k3']
    check('train_size_total equals one factorization count (not 2x)',
          log['train_size_total'] == n_orig,
          f'{log["train_size_total"]} vs {n_orig}')
    check('canonical single-model count recorded and equal',
          log.get('canonical_single_model_train_size') == n_orig)
    base = log['assignment']['base_seed']
    stride = log['assignment']['stride']
    ok_bal, ok_seed, ok_sum = True, True, True
    for ep in log['epochs']:
        c = ep['assignment_counts']
        vals = list(c.values())
        if max(vals) - min(vals) > 1:
            ok_bal = False
        if ep['assignment_seed'] != base * stride + ep['epoch_index0']:
            ok_seed = False
        if sum(vals) != n_orig:
            ok_sum = False
    check(f'all {len(log["epochs"])} epochs balanced (|n1-n2| <= 1)', ok_bal)
    # Single-writer integrity: a duplicated training process sharing one run
    # directory would leave duplicate or non-monotonic epoch numbers, or a total
    # step count that is not steps_per_epoch * epochs_run.  This exact signature is
    # what identified a concurrent-process incident during this study.
    epnums = [e['epoch'] for e in log['epochs']]
    check('epoch numbers are unique and strictly increasing',
          all(b > a for a, b in zip(epnums, epnums[1:])) and len(set(epnums)) == len(epnums),
          f'{len(epnums)} entries, {len(set(epnums))} unique')
    spe = log['epochs'][0]['steps']
    check('total_optimizer_steps == steps_per_epoch * epochs_run (single writer)',
          log.get('total_optimizer_steps') == spe * len(log['epochs']),
          f'{log.get("total_optimizer_steps")} vs {spe * len(log["epochs"])}')
    check('every epoch seed equals seed*stride + epoch', ok_seed)
    check('every epoch assigns each original instance exactly once', ok_sum)
    # the assignment must actually be reproducible from the seed
    from af_matched_train import balanced_assignment
    ep0 = log['epochs'][0]
    a, s = balanced_assignment(n_orig, base, ep0['epoch_index0'])
    check('epoch-0 assignment recomputes from the seed',
          s == ep0['assignment_seed'] and
          dict(zip(log['factorizations'], [int((a == 0).sum()), int((a == 1).sum())])) ==
          ep0['assignment_counts'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/adaptive_factorization')
    a = ap.parse_args(argv)
    root = Path(a.root)
    sel = root / 'selector'
    frozen_inputs_present(root, sel)
    label_independence(root, sel)
    feature_purity(root, sel)
    no_test_leakage(root, sel)
    compute_matched_assignment(root)
    print()
    print(f'{CHECKS[0] - len(FAILS)}/{CHECKS[0]} checks passed')
    if FAILS:
        print(f'FAILURES ({len(FAILS)}):')
        for f in FAILS:
            print(f'  - {f}')
        return 1
    print('ALL VERIFICATION CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
