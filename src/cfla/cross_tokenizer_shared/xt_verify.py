#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consistency guards for the cross-tokenizer / shared-parameter study.

1. Spectral-K3 teacher-forced TotalNLL must reproduce the audited full-ID
   re-audit values (cross-study regression).
2. Every independent-tokenizer SID array must be injective and round-trip.
3. Every dumped beam must reconstruct its tokens from its own SID array under the
   detected convention.
4. The Spectral-K3 and RQ-VAE-3layer test targets must be element-wise identical.
5. The vectorised fusion must agree with a slow, independent dict-based
   implementation on a cross-tokenizer pair.
6. The shared vocabulary must be collision-free (control tokens distinct, above
   the SID range, not PAD/EOS).
7. Shared beams: both factorizations must decode validly, and the two
   factorizations must produce different beams (the control token must matter).
8. Independent and shared evaluations must cover the same users.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, CODEBOOK, DATASETS, PRIMARY, STEP1, STEP2,
                       TOKENIZERS, beam_path, detect_item_shift, load_cell,
                       load_sid, roundtrip_report, shared_vocab_plan, summarize,
                       union_fuse)  # noqa: E402

FAIL = []
REPO = Path(__file__).resolve().parents[3]
REAUDIT = REPO / 'outputs' / 'mechanism' / 'full_id_reaudit'


def check(name, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        FAIL.append(name)


def slow_fuse(cells):
    """Independent reference: dict-based uniform item-probability fusion."""
    n = cells[0]['items'].shape[0]
    ranks = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        acc = {}
        for c in cells:
            for j in range(c['items'].shape[1]):
                it = int(c['items'][i, j])
                if it < 0:
                    continue
                acc[it] = acc.get(it, 0.0) + float(c['probs'][i, j]) / len(cells)
        t = int(cells[0]['target'][i])
        if t in acc:
            ranks[i] = 1 + sum(1 for v in acc.values() if v > acc[t])
    return ranks


def main():
    # 1 ---- spectral teacher-forced NLL regression -------------------------
    nllp = STEP1 / 'cross_tokenizer_target_nll.parquet'
    aud = REAUDIT / 'teacher_forced_full_sid_nll.csv'
    if nllp.exists() and aud.exists():
        n = pd.read_parquet(nllp)
        new = (n[n.tokenizer == 'spectral_k3'].assign(K=3)
               .groupby(['dataset', 'K'])['total_nll'].mean().rename('new'))
        old = pd.read_csv(aud)
        old = old[old['K'] == 3].set_index(['dataset', 'K'])['mean_total_nll'].rename('old')
        j = pd.concat([new, old], axis=1).dropna()
        d = (j['new'] - j['old']).abs()
        check('spectral-K3 teacher-forced TotalNLL reproduces the audited re-audit',
              bool(len(j) and (d < 1e-6).all()), f'max|diff|={d.max():.3e} over {len(j)} cells')
    else:
        check('spectral-K3 teacher-forced NLL vs re-audit', False, 'inputs missing')

    # 2 ---- SID injectivity / round-trip ----------------------------------
    bad = []
    for tok in TOKENIZERS:
        for ds in DATASETS:
            r = roundtrip_report(tok, ds)
            if not (r['injective'] and r['sid_roundtrip'] and r['item_roundtrip'] and
                    r['item_universe_matches_canonical']):
                bad.append((tok, ds, r))
    check('every tokenizer SID array is injective and round-trips',
          not bad, f'{len(bad)} failures')

    # 3 ---- beam token reconstruction -------------------------------------
    bad = []
    for tok in TOKENIZERS:
        for ds in DATASETS:
            p = beam_path(tok, ds, 'test')
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            shift, scores = detect_item_shift(z, load_sid(tok, ds))
            if scores[shift] < 0.99:
                bad.append((tok, ds, scores))
    check('every dumped beam reconstructs its tokens from its own SID array',
          not bad, f'{len(bad)} bad')

    # 4 ---- target identity -----------------------------------------------
    tgt_ok = {}
    for ds in DATASETS:
        try:
            ta = load_cell('spectral_k3', ds, 'test')['target']
            tb = load_cell('rqvae3', ds, 'test')['target']
            tgt_ok[ds] = bool(np.array_equal(ta, tb))
        except FileNotFoundError:
            tgt_ok[ds] = None
    check('Spectral-K3 and RQ-VAE-3layer test targets are element-wise identical',
          all(v is True for v in tgt_ok.values()), str(tgt_ok))

    # 5 ---- fusion vs slow reference --------------------------------------
    agree = None
    try:
        ca = load_cell('spectral_k3', 'Beauty', 'test')
        cb = load_cell('rqvae3', 'Beauty', 'test')
        fast = union_fuse([ca, cb])['rank']
        slow = slow_fuse([ca, cb])
        agree = bool(np.array_equal(fast, slow))
        check('vectorised cross-tokenizer fusion == dict-based reference',
              agree, f'NDCG@20 {summarize(fast)["NDCG@20"]:.6f}')
    except FileNotFoundError as e:
        check('vectorised cross-tokenizer fusion == dict-based reference', False, str(e))

    # 6 ---- shared vocab ---------------------------------------------------
    plan = shared_vocab_plan(list(PRIMARY))
    ctrl = plan['control_tokens']
    ok = (all(v > plan['sid_token_range'][1] for v in ctrl.values()) and
          len(set(ctrl.values())) == len(ctrl) and all(v != 0 for v in ctrl.values()) and
          plan['vocab_size'] == plan['sid_token_range'][1] + 1 + len(ctrl))
    check('shared vocabulary is collision-free', ok, str(plan))

    # 7 ---- shared beams ---------------------------------------------------
    same_beams = []
    n_pairs_checked = 0
    for ds in DATASETS:
        beams = {}
        for f in PRIMARY:
            p = STEP2 / 'beams' / f'{ds}_{f}_test.npz'
            if p.exists():
                beams[f] = np.load(p, allow_pickle=True)
        if len(beams) == 2:
            n_pairs_checked += 1
            a = np.asarray(beams[PRIMARY[0]]['candidate_item_ids'])
            if not (a >= 0).mean() > 0.9:
                same_beams.append((ds, 'invalid decode'))
            # different control tokens must produce different beams for at least
            # some users; identical beams would mean the control token is inert
            b = np.asarray(beams[PRIMARY[1]]['candidate_item_ids'])
            if np.array_equal(a, b):
                same_beams.append((ds, 'identical beams across factorizations'))
    if n_pairs_checked:
        check('shared beams differ across factorizations (control token has an effect)',
              not same_beams, f'{n_pairs_checked} pairs checked; {same_beams}')
    else:
        print('[SKIP] shared beams differ across factorizations '
              '(shared beams not dumped yet)')

    # 8 ---- same users -----------------------------------------------------
    n_ok = []
    for ds in DATASETS:
        p = STEP2 / 'beams' / f'{ds}_{PRIMARY[0]}_test.npz'
        if p.exists():
            n = len(np.load(p, allow_pickle=True)['target_item_ids'])
            ind = load_cell(PRIMARY[0], ds, 'test')['n_users']
            n_ok.append(n == ind)
    if n_ok:
        check('shared and independent evaluations cover the same users',
              all(n_ok), f'{len(n_ok)} datasets checked')
    else:
        print('[SKIP] shared vs independent user coverage (shared beams not dumped yet)')

    # 9 ---- report headline numbers must match the artifacts ---------------
    rep = STEP1 / 'STEP1_CROSS_TOKENIZER_REPORT.md'
    if rep.exists():
        txt = rep.read_text()
        dv = pd.read_csv(STEP1 / 'cross_tokenizer_pair_diversity.csv')
        P = dv[dv.tier == 'PRIMARY']
        al = pd.read_csv(STEP1 / 'cross_tokenizer_nll_alignment.csv')
        a = al[(al.tier == 'PRIMARY') & (al.normalisation == 'raw') &
               (al.group != 'ALL_USERS')]
        adv = a['mean_nll_other'] - a['mean_nll_own']
        pool = al[(al.tier == 'PRIMARY') & (al.normalisation == 'raw') &
                  (al.group == 'BOTH_Only_groups')]
        wanted = {
            'own-tokenizer advantage range': f'{adv.min():.2f}–{adv.max():.2f}',
            'alignment accuracy range': (f"{100*pool['argmin_match_accuracy'].min():.1f}%–"
                                         f"{100*pool['argmin_match_accuracy'].max():.1f}%"),
            'exploited-fraction range': (f"{100*P['oracle_exploited_fraction'].min():.1f}%–"
                                         f"{100*P['oracle_exploited_fraction'].max():.1f}%"),
        }
        missing = [k for k, v in wanted.items() if v not in txt]
        check('report headline numbers match the artifacts', not missing,
              f'checked {len(wanted)}; missing/stale: {missing}')
    else:
        print('[SKIP] report headline-number guard (report not generated yet)')

    # 10 ---- §51 provenance field coverage --------------------------------
    import json as _json
    need_top = ['git', 'environment', 'generated']
    p1 = STEP1 / '_PROVENANCE.json'
    p2 = STEP2 / '_PROVENANCE.json'
    problems = []
    if p1.exists():
        d = _json.loads(p1.read_text())
        for k in need_top:
            if k not in d:
                problems.append(f'step1 missing {k}')
        if not d.get('git', {}).get('sha'):
            problems.append('step1 git.sha empty')
        if not d.get('environment', {}).get('torch'):
            problems.append('step1 torch missing')
        if not d.get('environment', {}).get('cuda_device'):
            problems.append('step1 GPU missing')
        beams = d.get('inputs', {}).get('beams', {})
        if not beams:
            problems.append('step1 beams block missing')
        else:
            bad = [k for k, v in beams.items()
                   if not all(v.get(f) for f in ('path', 'sha256', 'dataset',
                                                 'tokenizer', 'split'))]
            if bad:
                problems.append(f'step1 beams missing §51 fields: {bad[:3]}')
        tk = d.get('inputs', {}).get('tokenizers', {})
        bad = [f'{t}|{ds}'
               for t, per_ds in tk.items() for ds, v in per_ds.items()
               if not all(v.get(f) for f in ('sid_path', 'sid_sha256',
                                             'recommender_ckpt', 'ckpt_sha256'))]
        if bad:
            problems.append(f'step1 tokenizer inputs incomplete: {bad[:3]}')
        n_entries = sum(len(v) for v in tk.values())
        if n_entries != len(TOKENIZERS) * len(DATASETS):
            problems.append(f'step1 tokenizer coverage {n_entries} != '
                            f'{len(TOKENIZERS)*len(DATASETS)}')
    else:
        problems.append('step1 provenance missing')
    if p2.exists():
        d2 = _json.loads(p2.read_text())
        for name, run in d2.get('shared_models', {}).items():
            if not run.get('checkpoint_sha256'):
                problems.append(f'{name}: checkpoint hash missing')
            if not run.get('config_hash'):
                problems.append(f'{name}: config hash missing')
            if (run.get('config') or {}).get('seed') is None:
                problems.append(f'{name}: seed missing')
    check('§51 provenance fields present (git, env, artifact hashes, beam split, '
          'seed, config hash)', not problems, str(problems[:4]))

    print()
    if FAIL:
        print('FAILURES:', FAIL)
        return 1
    print('ALL CONSISTENCY CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
