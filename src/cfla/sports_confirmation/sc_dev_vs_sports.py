#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP H — development vs pristine comparison (harness section 37/38).

Sports is reported as its own row and is NEVER pooled into the development numbers.
Every development figure is read from the frozen development artefacts; every Sports
figure from this round's frozen test artefacts.

Usage:  python3 sc_dev_vs_sports.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# SC_OUT redirects the round-4 tree for smoke-testing, matching sc_pipeline.py,
# sc_gate_eval.py and sc_provenance.py.
OUT = Path(os.environ.get('SC_OUT') or (REPO / 'outputs' / 'sports_confirmation'))
DEV_GATE = REPO / 'outputs' / 'disagreement_gate'
DEV_CC = REPO / 'outputs' / 'candidate_conditioned'
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
DIDS = list(SHORT)
SPORTS = 'Sports_and_Outdoors'
FIELDS = ['dataset', 'role', 'uniform_to_lse', 'uniform_to_gate', 'lse_to_gate',
          'target_vs_nontarget_auc', 'union_coverage', 'only_f1_fraction',
          'only_f2_fraction', 'both_fraction', 'n_union_users', 'n_all_users']


def rd(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def ndcg(rows, method, ds=None):
    for r in rows:
        if r.get('method') == method and (ds is None or r.get('dataset') == ds):
            return float(r['NDCG@20'])
    return float('nan')


def main():
    rows = []

    # ---------------- development datasets ----------------
    gtr = rd(DEV_GATE / 'test' / 'test_results.csv')
    gsub = rd(DEV_GATE / 'test' / 'subgroup_results.csv')
    ccmech = json.loads((DEV_CC / 'test' / 'mechanism_diagnostics.json').read_text())
    ccmove = rd(DEV_CC / 'test' / 'target_rank_movement.csv')
    au = {r['dataset']: r for r in ccmove}
    grp = defaultdict(dict); tot = defaultdict(int)
    for r in gsub:
        if r['method'] == 'Uniform':
            grp[r['dataset']][r['group']] = int(r['n']); tot[r['dataset']] += int(r['n'])

    for ds in DIDS:
        u = ndcg(gtr, 'Uniform', ds); l = ndcg(gtr, 'LSE', ds); g = ndcg(gtr, 'Gate', ds)
        n_in = int(au[ds]['n_target_in_union']) if ds in au else 0
        n_all = tot[ds] if tot[ds] else 1
        rows.append({
            'dataset': SHORT[ds], 'role': 'development',
            'uniform_to_lse': l - u, 'uniform_to_gate': g - u, 'lse_to_gate': g - l,
            'target_vs_nontarget_auc':
                ccmech[ds]['pairwise_auc_target_vs_nontarget_LSE'],
            'union_coverage': n_in / n_all,
            'only_f1_fraction': grp[ds].get('Only_F1', 0) / n_all,
            'only_f2_fraction': grp[ds].get('Only_F2', 0) / n_all,
            'both_fraction': grp[ds].get('Both', 0) / n_all,
            'n_union_users': n_in, 'n_all_users': n_all})

    # ---------------- Sports (pristine) ----------------
    str_ = rd(OUT / 'test' / 'test_results.csv')
    ssub = rd(OUT / 'test' / 'subgroup_results.csv')
    smech = {r['metric']: r for r in rd(OUT / 'test' / 'mechanism_diagnostics.csv')}
    u = ndcg(str_, 'Uniform'); l = ndcg(str_, 'LSE'); g = ndcg(str_, 'Gate')
    # Total users = the sum of the per-group counts for ONE method, which is how the
    # development side computes it (see `tot` above).  It is NOT ssub[0]['n']: the
    # producer emits RUNGS = (Only_F1, Only_F2, Both, Neither) and writes each group's own
    # count, so ssub[0]['n'] is the Only_F1 count alone and would inflate every fraction
    # below by roughly the number of groups.
    _uni = [int(r['n']) for r in ssub if r['method'] == 'Uniform']
    _reported = smech.get('n_users', {}).get('value')
    n_all = sum(_uni) if _uni else (int(float(_reported)) if _reported else 1)
    if _reported and _uni and int(float(_reported)) != n_all:
        print(f'[warn] subgroup total {n_all} != reported n_users {int(float(_reported))}')
    only1 = next((int(r['n']) for r in ssub if r['group'] == 'Only_F1' and
                  r['method'] == 'Uniform'), 0)
    only2 = next((int(r['n']) for r in ssub if r['group'] == 'Only_F2' and
                  r['method'] == 'Uniform'), 0)
    both = next((int(r['n']) for r in ssub if r['group'] == 'Both' and
                 r['method'] == 'Uniform'), 0)
    # measured union coverage (never inferred): a Neither user can still have the target
    # present in the union but outside both top-20s, so subgroup counts do not determine it
    cover = float(smech.get('union_coverage', {}).get('value', float('nan')))
    n_in = int(smech.get('union_coverage', {}).get('n_pairs', 0))
    rows.append({
        'dataset': 'Sports (pristine)', 'role': 'confirmatory',
        'uniform_to_lse': l - u, 'uniform_to_gate': g - u, 'lse_to_gate': g - l,
        'target_vs_nontarget_auc':
            float(smech.get('target_vs_nontarget_LSE_AUC', {}).get('value', float('nan'))),
        'union_coverage': cover,
        'only_f1_fraction': only1 / n_all, 'only_f2_fraction': only2 / n_all,
        'both_fraction': both / n_all,
        'n_union_users': n_in, 'n_all_users': n_all})

    with (OUT / 'development_vs_sports.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {OUT/"development_vs_sports.csv"}')
    print(f'{"dataset":18s} {"role":12s} {"U->LSE":>9s} {"U->Gate":>9s} {"LSE->Gate":>9s} '
          f'{"AUC":>7s} {"cover":>7s} {"OnlyF1":>7s} {"OnlyF2":>7s} {"Both":>7s}')
    for r in rows:
        print(f'{r["dataset"]:18s} {r["role"]:12s} {r["uniform_to_lse"]:+9.6f} '
              f'{r["uniform_to_gate"]:+9.6f} {r["lse_to_gate"]:+9.6f} '
              f'{r["target_vs_nontarget_auc"]:7.4f} {r["union_coverage"]:7.4f} '
              f'{r["only_f1_fraction"]:7.4f} {r["only_f2_fraction"]:7.4f} '
              f'{r["both_fraction"]:7.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
