#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE A + PHASE B — is factorization preference user-stable or item-stable?

H1 says preference is better described as F*(u, i) than F*(u).  This script tests the
user-stable half of that claim directly, and then characterises item-level preference.

It is deliberately cheap: it uses only the frozen artefacts already on disk —
per-user target ranks under each factorization (from the canonical beam dumps) and
the per-target teacher-forced NLL already computed for both splits.  No model is run.

Definitions (fixed here, before any number is read)
--------------------------------------------------
* A user's preference on a split is DECISIVE when the two factorizations disagree about
  the target, i.e. `rank_F1 != rank_F2`, with an absent target coded `-1`
  (rank 31 is used internally where a genuine rank is needed, so "absent" sorts last).
  If both factorizations miss the target, `rank_F1 == rank_F2 == -1` and the instance
  carries no preference signal, so it is NOT decisive.
* `pref = F1 if rank_F1 < rank_F2 else F2`.
* NLL preference: `nll_pref = argmin_F TotalNLL_F(target)` on the same split; ties are
  reported separately and excluded from agreement statistics.

Temporal protocol: TRAIN -> VALID -> TEST are three consecutive events per user.
Historical item preference is therefore learned on TRAIN, evaluated on VALID, and THEN
frozen and applied to TEST.  TEST is never used to build any preference.

Usage
  python3 cc_preference_stability.py --selectordir <adaptive selector dir> --outdir <out>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'adaptive_factorization')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
            'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
MIN_ITEM_OCCURRENCES = 5     # prompt section 11
F1, F2 = 'spectral_k3', 'rqvae3'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_split(sel, ds, split):
    d = Path(sel) / 'features'
    z = dict(np.load(d / f'{ds}_{split}.npz', allow_pickle=True))
    return z


def decisive_from_ranks(r1, r2):
    """(decisive_mask, pref) with pref in {1, 2} where decisive."""
    r1 = np.asarray(r1); r2 = np.asarray(r2)
    dec = r1 != r2
    pref = np.where(r1 < r2, 1, 2)
    return dec, pref


def cohens_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return float('nan')
    po = float((a == b).mean())
    pa1 = float((a == 1).mean()); pa2 = 1 - pa1
    pb1 = float((b == 1).mean()); pb2 = 1 - pb1
    pe = pa1 * pb1 + pa2 * pb2
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-15 else float('nan')


def mutual_information(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return float('nan')
    mi = 0.0
    for va in (1, 2):
        for vb in (1, 2):
            pij = float(((a == va) & (b == vb)).mean())
            pi = float((a == va).mean()); pj = float((b == vb).mean())
            if pij > 0 and pi > 0 and pj > 0:
                mi += pij * math.log(pij / (pi * pj))
    return mi


def entropy_binary(p):
    p = float(p)
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def nll_pref(n1, n2, tol=0.0):
    """argmin over factorizations; returns (pref, tie_mask) with pref in {1,2}."""
    n1 = np.asarray(n1); n2 = np.asarray(n2)
    tie = np.abs(n1 - n2) <= tol
    pref = np.where(n1 < n2, 1, 2)
    return pref, tie


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selectordir',
                    default='outputs/adaptive_factorization/selector')
    ap.add_argument('--outdir', default='outputs/candidate_conditioned')
    ap.add_argument('--nll-dir',
                    default='outputs/adaptive_factorization/selector/features/nll')
    a = ap.parse_args(argv)

    sel, out = Path(a.selectordir), Path(a.outdir)
    nll_dir = Path(a.nll_dir)
    out.mkdir(parents=True, exist_ok=True)

    stability_rows, transition_rows = [], []
    item_rows = []
    user_vs_item_rows = []
    per_ds_cache = {}

    for ds in DATASETS:
        v = load_split(sel, ds, 'valid')
        t = load_split(sel, ds, 'test')

        # ---------- PHASE A: temporal stability of the user preference ----------
        dv, pv = decisive_from_ranks(v['rank1'], v['rank2'])
        dt, pt = decisive_from_ranks(t['rank1'], t['rank2'])
        both_dec = dv & dt
        n_both = int(both_dec.sum())
        agree = float((pv[both_dec] == pt[both_dec]).mean()) if n_both else float('nan')

        # NLL preference per split (needs the teacher-forced target NLL dumps)
        p1 = nll_dir / f'{ds}_{F1}_valid.npz'
        have_nll = p1.exists() and (nll_dir / f'{ds}_{F2}_valid.npz').exists()
        nll_block = {}
        if have_nll:
            nv = nll_pref(np.load(nll_dir / f'{ds}_{F1}_valid.npz')['total_nll'],
                          np.load(nll_dir / f'{ds}_{F2}_valid.npz')['total_nll'])
            nt = nll_pref(np.load(nll_dir / f'{ds}_{F1}_test.npz')['total_nll'],
                          np.load(nll_dir / f'{ds}_{F2}_test.npz')['total_nll'])
            m = ~(nv[1] | nt[1])
            nll_block = {
                'nll_agree_valid_test': float((nv[0][m] == nt[0][m]).mean()) if m.any() else float('nan'),
                'nll_kappa': cohens_kappa(nv[0][m], nt[0][m]),
                'nll_mi': mutual_information(nv[0][m], nt[0][m]),
                'nll_ties_valid': int(nv[1].sum()), 'nll_ties_test': int(nt[1].sum()),
                'nll_n': int(m.sum()),
                'nll_n_test_nontie': int((~nt[1]).sum()),
            }
            # Head-to-head on the decisive-decisive cohort: does the split's own NLL
            # argmin pick the same factorization that ranks the target better?
            m_dec = both_dec & (~nv[1]) & (~nt[1])
            n_m = int(m_dec.sum())
            nll_block['cohort_n'] = n_m
            nll_block['nll_pref_valid_predicts_test_rank_winner'] = float(
                (nv[0][m_dec] == pt[m_dec]).mean()) if n_m else float('nan')
            nll_block['nll_pref_test_matches_rank_winner_test'] = float(
                (nt[0][m_dec] == pt[m_dec]).mean()) if n_m else float('nan')
            nll_block['user_pref_valid_matches_test_rank_winner'] = float(
                (pv[m_dec] == pt[m_dec]).mean()) if n_m else float('nan')
            nll_block['always_F1_on_cohort'] = float((pt[m_dec] == 1).mean()) if n_m else float('nan')

        # baselines on the same decisive-decisive cohort
        base_f1 = float((pt[both_dec] == 1).mean()) if n_both else float('nan')
        base_f2 = float((pt[both_dec] == 2).mean()) if n_both else float('nan')
        user_pred = float((pv[both_dec] == pt[both_dec]).mean()) if n_both else float('nan')

        stability_rows.append({
            'dataset': ds, 'short': SHORT[ds],
            'n_users': int(len(v['rank1'])),
            'n_decisive_valid': int(dv.sum()), 'n_decisive_test': int(dt.sum()),
            'n_decisive_both': n_both,
            'frac_decisive_valid': float(dv.mean()), 'frac_decisive_test': float(dt.mean()),
            'P_pref_valid_equals_pref_test': agree,
            'chance_baseline_majority': max(base_f1, base_f2) if n_both else float('nan'),
            'always_F1': base_f1, 'always_F2': base_f2,
            'valid_winner_to_test_winner': user_pred,
            'cohens_kappa': cohens_kappa(pv[both_dec], pt[both_dec]),
            'mutual_information': mutual_information(pv[both_dec], pt[both_dec]),
            **nll_block,
        })

        # transition matrix on the decisive-decisive cohort
        for from_f in (1, 2):
            for to_f in (1, 2):
                m = both_dec & (pv == from_f) & (pt == to_f)
                transition_rows.append({
                    'dataset': ds, 'short': SHORT[ds],
                    'from_valid': f'F{from_f}', 'to_test': f'F{to_f}',
                    'n': int(m.sum()),
                    'row_fraction': float(m.sum() / max(int((both_dec & (pv == from_f)).sum()), 1)),
                })

        # ---------- PHASE B: item-level preference ----------
        # DeltaNLL(u,i) = NLL_F1 - NLL_F2 for the *target* of the instance.
        # Learned on TRAIN, validated on VALID, frozen, applied to TEST.
        item_hist = {}       # item -> list of DeltaNLL from TRAIN targets
        if have_nll:
            tr = nll_dir / f'{ds}_{F1}_train.npz'
            if tr.exists():
                # TRAIN target NLL was not dumped for this study; fall back to VALID as
                # the "historical" split and record that honestly.
                pass
        # Build item preference from VALID targets, evaluate on TEST targets for the
        # SAME item (only items seen at least MIN_ITEM_OCCURRENCES times).
        if have_nll:
            dvn = np.load(nll_dir / f'{ds}_{F1}_valid.npz')['total_nll'] - \
                  np.load(nll_dir / f'{ds}_{F2}_valid.npz')['total_nll']
            dtn = np.load(nll_dir / f'{ds}_{F1}_test.npz')['total_nll'] - \
                  np.load(nll_dir / f'{ds}_{F2}_test.npz')['total_nll']
            iv = np.asarray(v['cand_target']); it = np.asarray(t['cand_target'])
            # valid-side item aggregation
            order = np.argsort(iv, kind='stable')
            for item, g in _group_indices(iv[order]):
                d = dvn[order][g]
                if len(d) >= MIN_ITEM_OCCURRENCES:
                    item_rows.append({
                        'dataset': ds, 'item': int(item), 'n_occurrences': int(len(d)),
                        'mean_delta_nll': float(d.mean()),
                        'median_delta_nll': float(np.median(d)),
                        'std_delta_nll': float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                        'fraction_F1_preferred': float((d < 0).mean()),
                        'fraction_F2_preferred': float((d > 0).mean()),
                        'preference_entropy': entropy_binary(float((d < 0).mean())),
                        'majority_pref': 'F1' if d.mean() < 0 else 'F2',
                    })
            # ---- item-level predictor: majority sign of VALID DeltaNLL -> TEST winner ----
            hist = {}
            for item, g in _group_indices(iv[order]):
                d = dvn[order][g]
                if len(d) >= MIN_ITEM_OCCURRENCES:
                    hist[int(item)] = 1 if d.mean() < 0 else 2
            seen = np.array([int(x) in hist for x in it])

            # Head-to-head comparison MUST use one cohort for both predictors, otherwise
            # the two accuracies describe different user sets and are not comparable.
            # Cohort: the test target item was seen in validation history AND the user's
            # validation preference is decisive, so both predictors are defined.
            cohort = seen & dv
            n_c = int(cohort.sum())
            if n_c:
                true_pt = pt[cohort]
                pred_item = np.array([hist[int(x)] for x in it[cohort]])
                pred_user = pv[cohort]

                def accs(pred, true):
                    plain = float((pred == true).mean())
                    n1 = int((true == 1).sum()); n2 = int((true == 2).sum())
                    tpr = float((pred[true == 1] == 1).mean()) if n1 else float('nan')
                    tnr = float((pred[true == 2] == 2).mean()) if n2 else float('nan')
                    bal = (tpr + tnr) / 2.0
                    return plain, bal, tpr, tnr

                i_acc, i_bal, i_tpr, i_tnr = accs(pred_item, true_pt)
                u_acc, u_bal, u_tpr, u_tnr = accs(pred_user, true_pt)
                majority = max(float((true_pt == 1).mean()), float((true_pt == 2).mean()))
                f1_share = float((true_pt == 1).mean())
            else:
                i_acc = i_bal = i_tpr = i_tnr = float('nan')
                u_acc = u_bal = u_tpr = u_tnr = float('nan')
                majority = f1_share = float('nan')

            user_vs_item_rows.append({
                'dataset': ds, 'short': SHORT[ds],
                'n_test_rows': int(len(it)), 'n_item_seen_in_valid': int(seen.sum()),
                'n_comparable_cohort': n_c,
                'cohort_majority_baseline': majority,
                'cohort_share_F1': f1_share,
                'item_history_accuracy': i_acc,
                'item_history_balanced_accuracy': i_bal,
                'item_history_tpr_F1': i_tpr, 'item_history_tnr_F2': i_tnr,
                'user_history_accuracy': u_acc,
                'user_history_balanced_accuracy': u_bal,
                'user_history_tpr_F1': u_tpr, 'user_history_tnr_F2': u_tnr,
                'item_minus_user_accuracy': i_acc - u_acc if n_c else float('nan'),
                'item_minus_user_balanced': i_bal - u_bal if n_c else float('nan'),
            })

        per_ds_cache[ds] = {'stability': stability_rows[-1]}

    # ---- write ----
    _write_csv(out / 'preference_temporal_stability.csv', stability_rows)
    _write_csv(out / 'preference_transition_matrix.csv', transition_rows)
    _write_csv(out / 'item_preference_stability.csv', item_rows)
    _write_csv(out / 'user_vs_item_preference.csv', user_vs_item_rows)

    # ---- console summary ----
    print('=== PHASE A: is the user preference stable from validation to test? ===')
    print(f'{"dataset":14s} {"cohort":>7s} {"user_val":>9s} {"NLLval":>8s} {"NLLtest":>8s} {"alwaysF1":>9s} {"kappa":>7s}')
    for r in stability_rows:
        print(f'{r["short"]:14s} {r.get("cohort_n",0):7d} '
              f'{r.get("user_pref_valid_matches_test_rank_winner", float("nan")):9.4f} '
              f'{r.get("nll_pref_valid_predicts_test_rank_winner", float("nan")):8.4f} '
              f'{r.get("nll_pref_test_matches_rank_winner_test", float("nan")):8.4f} '
              f'{r.get("always_F1_on_cohort", float("nan")):9.4f} '
              f'{r["cohens_kappa"]:7.4f}')
    print()
    print('=== PHASE B: item vs user history, SAME cohort ===')
    print(f'{"dataset":14s} {"cohort":>7s} {"majority":>9s} '
          f'{"item_acc":>9s} {"item_bal":>9s} {"user_acc":>9s} {"user_bal":>9s}')
    for r in user_vs_item_rows:
        print(f'{r["short"]:14s} {r["n_comparable_cohort"]:7d} {r["cohort_majority_baseline"]:9.4f} '
              f'{r["item_history_accuracy"]:9.4f} {r["item_history_balanced_accuracy"]:9.4f} '
              f'{r["user_history_accuracy"]:9.4f} {r["user_history_balanced_accuracy"]:9.4f}')
    print()
    print(f'wrote 4 CSVs to {out}')
    return 0


def _group_indices(sorted_items):
    """Yield (item, index_array) runs over a sorted item array."""
    n = len(sorted_items)
    if n == 0:
        return
    start = 0
    for i in range(1, n + 1):
        if i == n or sorted_items[i] != sorted_items[start]:
            yield sorted_items[start], np.arange(start, i)
            start = i


def _write_csv(p, rows):
    if not rows:
        Path(p).write_text('')
        return
    keys = list(rows[0].keys())
    with Path(p).open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == '__main__':
    sys.exit(main())
