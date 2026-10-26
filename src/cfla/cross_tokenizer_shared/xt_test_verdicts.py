#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the STEP 2 verdict rules.

The verdicts are assigned by `decide_verdicts` in `xt_step2_report.py`. Those
rules decide the study's headline conclusion, so they are tested here on
synthetic cases covering every branch, including the §44 sharing-collapse
branch and the necessity inversion.

Run:  python3 xt_test_verdicts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_step2_report import decide_verdicts  # noqa: E402
from xt_step1_report import decide_step1_verdicts  # noqa: E402

N = 4  # datasets


def make(fusion_pos, fusion_sig, oracle_pos, oracle_sig, only_pop, collapse=0):
    comp = pd.DataFrame({
        'shared_fusion_gain': [0.001] * fusion_pos + [-0.001] * (N - fusion_pos),
        'fusion_vs_best_significant_95': [True] * fusion_sig + [False] * (N - fusion_sig),
        'shared_oracle_gain': [0.01] * oracle_pos + [-0.01] * (N - oracle_pos),
        'oracle_vs_fusion_significant_95': [True] * oracle_sig + [False] * (N - oracle_sig),
        'only_f1_fraction': [0.02] * only_pop + [0.0] * (N - only_pop),
        'only_f2_fraction': [0.02] * only_pop + [0.0] * (N - only_pop),
    })
    met = pd.DataFrame({'severe_sharing_collapse': [True] * collapse + [False] * (N - collapse)})
    return decide_verdicts(met, comp)


CASES = [
    ('strong shared effect: fusion 4/4 (3 sig), oracle 4/4 (4 sig), Only-F 4/4',
     (4, 3, 4, 4, 4, 0), ('SUPPORTED', 'NOT SUPPORTED')),
    ('oracle present but fusion positive in only 2/4',
     (2, 1, 4, 4, 4, 0), ('PARTIALLY SUPPORTED', 'NOT ESTABLISHED')),
    ('oracle only, one Only-F group empty',
     (1, 0, 4, 4, 3, 0), ('PARTIALLY SUPPORTED', 'NOT ESTABLISHED')),
    ('no complementarity at all',
     (0, 0, 0, 0, 4, 0), ('NOT SUPPORTED', 'SUPPORTED')),
    ('severe sharing collapse in one factorization',
     (4, 4, 4, 4, 4, 1), ('CONFOUNDED', 'NOT ESTABLISHED')),
]


def make_step1(fusion_pos, fusion_sig, oracle_pos, only_pop, collapse_fair=False,
               acc=None, p_ok=True, both_dirs=True, agree=True):
    fair = [{'criterion': 'x', 'status': ('FAIL' if collapse_fair else 'PASS')}] * 4
    div = pd.DataFrame({
        'fusion_gain': [0.001] * fusion_pos + [-0.001] * (N - fusion_pos),
        'fusion_vs_best_significant_95': [True] * fusion_sig + [False] * (N - fusion_sig),
        'oracle_gain': [0.01] * oracle_pos + [-0.01] * (N - oracle_pos),
        'p_a1_b0': [0.02] * only_pop + [0.0] * (N - only_pop),
        'p_a0_b1': [0.02] * only_pop + [0.0] * (N - only_pop),
    })
    acc = acc if acc is not None else [0.97] * N
    rows, rows_z = [], []
    for i in range(N):
        ds = f'd{i}'
        for grp, own, oth in (('Only-Spectral', 4.5, 6.5), ('Only-RQVAE', 4.6, 6.4)):
            rows.append({'dataset': ds, 'group': grp, 'mean_nll_own': own,
                         'mean_nll_other': oth, 'argmin_match_accuracy': acc[i],
                         'permutation_p_value': 0.0 if p_ok else 0.9})
        rows.append({'dataset': ds, 'group': 'BOTH_Only_groups',
                     'mean_nll_own': 4.55, 'mean_nll_other': 6.45,
                     'argmin_match_accuracy': acc[i],
                     'permutation_p_value': 0.0 if p_ok else 0.9})
        rows_z.append(dict(rows[-1], argmin_match_accuracy=acc[i] + (0.0 if agree else 0.2)))
    return decide_step1_verdicts(fair, div, pd.DataFrame(rows), pd.DataFrame(rows_z))


STEP1_CASES = [
    ('strong cross-tokenizer effect',
     dict(fusion_pos=4, fusion_sig=3, oracle_pos=4, only_pop=4),
     ('SUPPORTED', 'SUPPORTED')),
    ('fairness gate fails -> both verdicts NOT ESTABLISHED',
     dict(fusion_pos=4, fusion_sig=3, oracle_pos=4, only_pop=4, collapse_fair=True),
     ('NOT ESTABLISHED', 'NOT ESTABLISHED')),
    ('no effect at all',
     dict(fusion_pos=0, fusion_sig=0, oracle_pos=0, only_pop=4, acc=[0.51] * N),
     ('NOT SUPPORTED', 'NOT SUPPORTED')),
    ('complementarity weak, preference strong',
     dict(fusion_pos=1, fusion_sig=0, oracle_pos=4, only_pop=4),
     ('PARTIALLY SUPPORTED', 'SUPPORTED')),
]


def main():
    ok = True
    for name, kw, expect in STEP1_CASES:
        comp, pref, _ = make_step1(**kw)
        good = (comp, pref) == expect
        ok &= good
        print(f'[{"PASS" if good else "FAIL"}] STEP1 {name}\n'
              f'        got={(comp, pref)}  expected={expect}')
    print()
    for name, args, expect in CASES:
        shared, nec, _ = make(*args)
        good = (shared, nec) == expect
        ok &= good
        print(f'[{"PASS" if good else "FAIL"}] {name}\n'
              f'        got={(shared, nec)}  expected={expect}')
    print()
    print('ALL VERDICT-RULE TESTS PASSED' if ok else 'VERDICT-RULE TEST FAILURE')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
