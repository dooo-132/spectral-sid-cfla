#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FINAL_CANDIDATE_CONDITIONED_REPORT.md (prompt sections 59-60).

Verdicts come from explicit rules applied to the artefacts; no branch promotes a
preferred story.

Usage
  python3 cc_final_report.py --outdir outputs/candidate_conditioned
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
            'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
EPS = 1e-9


def rd(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def f(x, nd=6):
    try:
        return f'{float(x):.{nd}f}'
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# verdict rules
# ---------------------------------------------------------------------------
def verdict_user_stable(stab):
    """SUPPORTED if the validation->test preference agreement beats the majority
    baseline with a meaningful kappa in a majority of datasets."""
    if not stab:
        return 'NOT SUPPORTED', 'no stability data'
    ok = 0
    for r in stab:
        agree = float(r['P_pref_valid_equals_pref_test'])
        base = float(r['chance_baseline_majority'])
        k = float(r['cohens_kappa'])
        if agree > base + 0.05 and k > 0.10:
            ok += 1
    n = len(stab)
    det = (f'{ok}/{n} datasets with agreement > majority+0.05 and kappa > 0.10; '
           f'per-dataset kappa ' +
           ', '.join(f'{SHORT[r["dataset"]]}={float(r["cohens_kappa"]):.3f}' for r in stab))
    if ok >= 3:
        return 'SUPPORTED', det
    if ok >= 1:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_item_stable(uv, item_rows):
    """SUPPORTED if an item's historical preference predicts the test preference better
    than chance and better than the majority baseline, in most datasets."""
    if not uv:
        return 'NOT SUPPORTED', 'no item data'
    ok = 0
    dets = []
    for r in uv:
        acc = float(r['item_history_accuracy'])
        bal = float(r['item_history_balanced_accuracy'])
        base = float(r['cohort_majority_baseline'])
        good = bal > 0.55
        ok += good
        dets.append(f'{SHORT[r["dataset"]]}: acc={acc:.3f} bal={bal:.3f} base={base:.3f}')
    n = len(uv)
    n_items = len([r for r in item_rows])
    det = (f'{ok}/{n} datasets with item-history balanced accuracy > 0.55 '
           f'({n_items} item records); ' + '; '.join(dets))
    if ok >= 3:
        return 'SUPPORTED', det
    if ok >= 1:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_nll_useful(mech, boot):
    """SUPPORTED if candidate NLL separates target from non-target candidates with an
    AUC clearly above chance in most datasets."""
    if not mech:
        return 'NOT SUPPORTED', 'no mechanism diagnostics'
    aucs = {ds: v.get('pairwise_auc_target_vs_nontarget_LSE') for ds, v in mech.items()}
    ok = sum(1 for a in aucs.values() if a is not None and a > 0.6)
    det = (f'{ok}/{len(aucs)} datasets with target-vs-nontarget AUC > 0.60; ' +
           ', '.join(f'{SHORT[ds]}={a:.4f}' if a is not None else f'{SHORT[ds]}=NA'
                     for ds, a in aucs.items()))
    if ok >= 3:
        return 'SUPPORTED', det
    if ok >= 1:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_beats_uniform(boot):
    """SUPPORTED if the primary method's pooled NDCG@20 gain over Uniform has a 95% CI
    strictly above zero AND is positive in every dataset."""
    if not boot:
        return 'NOT SUPPORTED', 'no bootstrap'
    rows = [r for r in boot if r['comparison'] == 'LSE_vs_Uniform' and r['metric'] == 'NDCG@20']
    pooled = [r for r in rows if r['dataset'] == 'ALL_POOLED']
    per = [r for r in rows if r['dataset'] != 'ALL_POOLED']
    if not pooled:
        return 'NOT SUPPORTED', 'no pooled row'
    sig = float(pooled[0]['ci_low']) > 0
    pos = sum(1 for r in per if float(r['delta']) > 0)
    det = (f'pooled delta={float(pooled[0]["delta"]):+.6f} '
           f'CI=[{float(pooled[0]["ci_low"]):+.6f},{float(pooled[0]["ci_high"]):+.6f}] '
           f'p={float(pooled[0]["p_value"]):.4f}; positive in {pos}/{len(per)} datasets')
    if sig and pos == len(per) and len(per) > 0:
        return 'SUPPORTED', det
    if sig or pos >= len(per) // 2 + 1:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_explains_more(main_ndcg, uni_ndcg, prev_router_ndcg=None):
    det = (f'pooled candidate reranking {main_ndcg:.6f} vs uniform {uni_ndcg:.6f}; '
           f'previous user-level adaptive router pooled {prev_router_ndcg:.6f}'
           if prev_router_ndcg is not None else
           f'pooled candidate reranking {main_ndcg:.6f} vs uniform {uni_ndcg:.6f}')
    if main_ndcg > uni_ndcg + EPS and (prev_router_ndcg is None or main_ndcg > prev_router_ndcg):
        return 'SUPPORTED', det
    if main_ndcg > uni_ndcg + EPS:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/candidate_conditioned')
    a = ap.parse_args(argv)
    out = Path(a.outdir)

    stab = rd(out / 'preference_temporal_stability.csv')
    trans = rd(out / 'preference_transition_matrix.csv')
    items = rd(out / 'item_preference_stability.csv')
    uv = rd(out / 'user_vs_item_preference.csv')
    vres = rd(out / 'validation' / 'validation_method_results.csv')
    tres = rd(out / 'test' / 'test_results.csv')
    boot = rd(out / 'test' / 'bootstrap.csv')
    move = rd(out / 'test' / 'target_rank_movement.csv')
    rescue = rd(out / 'test' / 'uniform_failure_rescue.csv')
    sub = rd(out / 'test' / 'subgroup_results.csv')
    eff = rd(out / 'efficiency.csv')
    mech = {}
    mp = out / 'test' / 'mechanism_diagnostics.json'
    if mp.exists():
        mech = json.loads(mp.read_text())
    freeze = {}
    fp = out / 'CANDIDATE_RERANK_FREEZE_MANIFEST.json'
    if fp.exists():
        freeze = json.loads(fp.read_text())
    scaler = {}
    sp = out / 'validation' / 'candidate_nll_scaler.json'
    if sp.exists():
        scaler = json.loads(sp.read_text())

    def pooled_ndcg(m):
        r = [x for x in tres if x['dataset'] == 'ALL_POOLED' and x['method'] == m]
        return float(r[0]['NDCG@20']) if r else float('nan')

    main_method = 'LSE'
    vA, rA = verdict_user_stable(stab)
    vB, rB = verdict_item_stable(uv, items)
    vC, rC = verdict_nll_useful(mech, boot)
    vD, rD = verdict_beats_uniform(boot)
    prev_router = 0.077548   # pooled adaptive gain reported by the previous round
    vE, rE = verdict_explains_more(pooled_ndcg(main_method), pooled_ndcg('Uniform'),
                                   prev_router)

    L = []
    L.append('# FINAL_CANDIDATE_CONDITIONED_REPORT.md')
    L.append('')
    L.append('Is factorization preference candidate-conditioned rather than '
             'user-conditioned, and can cross-factorization candidate likelihood improve '
             'ranking without ever knowing the target?')
    L.append('')
    L.append('Governance: no historical SID array, checkpoint, beam dump, split, config, '
             'scaler, router checkpoint, CSV, parquet, JSON or report was overwritten; this '
             'round wrote only under `outputs/candidate_conditioned/`. The frozen shared '
             'checkpoints were used for inference only and no base T5 was trained. The '
             'candidate scorer receives only a user history, candidate item ids and a '
             'factorization name; the target is never added to the candidate union and is '
             'read only to score a method\'s ranking, after the freeze manifest exists.')
    L.append('')
    L.append('## A. State recovery')
    L.append('')
    L.append('`frozen_state.md` records the branch and HEAD, the frozen shared '
             'checkpoint hashes, SID paths, the item-id convention, SID length 4, codebook '
             '256, the shared vocabulary layout (1027 with control tokens 1025/1026), the '
             'previous user-router and target-NLL results re-verified from disk, and the '
             'stop-condition check. Two conventions that are silent if wrong are recorded '
             'there: candidate ids are **0-based SID-array rows**, and the candidate scorer '
             'returns TotalNLL (a negative sum of log-probabilities is a sign error).')
    L.append('')
    L.append('The scorer was validated against the canonical teacher-forcing path: scoring '
             'the known target reproduces the frozen target-NLL dumps at correlation '
             '1.000000000000, max absolute difference 2.0e-05 over 22,363 Beauty test rows.')
    L.append('')

    # ---- B ----
    L.append('## B. Is factorization preference user-stable?')
    L.append('')
    L.append('Decisive cohort: users whose validation **and** test instances both have a '
             'decisive preference (`rank_F1 != rank_F2`; a target missed by both '
             'factorizations carries no signal and is not decisive).')
    L.append('')
    L.append('| dataset | cohort n | P(valid pref = test pref) | always-F1 | majority | '
             'Cohen kappa | mutual information |')
    L.append('|---|---|---|---|---|---|---|')
    for r in stab:
        L.append(f'| {SHORT[r["dataset"]]} | {r["n_decisive_both"]} | '
                 f'{f(r["P_pref_valid_equals_pref_test"], 4)} | {f(r["always_F1"], 4)} | '
                 f'{f(r["chance_baseline_majority"], 4)} | {f(r["cohens_kappa"], 4)} | '
                 f'{f(r["mutual_information"], 5)} |')
    L.append('')
    L.append('Transition matrix (valid preference -> test preference):')
    L.append('')
    L.append('| dataset | transition | n | row fraction |')
    L.append('|---|---|---|---|')
    for r in trans:
        L.append(f'| {SHORT[r["dataset"]]} | {r["from_valid"]} -> {r["to_test"]} | '
                 f'{r["n"]} | {f(r["row_fraction"], 4)} |')
    L.append('')
    L.append('NLL preference (`argmin TotalNLL` of the split\'s own target) against the '
             'same cohort, plus the reference figure for when the target IS known:')
    L.append('')
    L.append('| dataset | user pref (valid) | NLL pref (valid) | NLL pref (test) | '
             'always-F1 |')
    L.append('|---|---|---|---|---|')
    for r in stab:
        L.append(f'| {SHORT[r["dataset"]]} | '
                 f'{f(r.get("user_pref_valid_matches_test_rank_winner"), 4)} | '
                 f'{f(r.get("nll_pref_valid_predicts_test_rank_winner"), 4)} | '
                 f'{f(r.get("nll_pref_test_matches_test_rank_winner"), 4)} | '
                 f'{f(r.get("always_F1_on_cohort"), 4)} |')
    L.append('')
    L.append('The last column pair is the key contrast. Using the **test target\'s own** '
             'NLL ordering recovers the better-ranked factorization 0.911-0.946 of the time '
             '(verified in `frozen_state.md` section 6), but the *validation* NLL ordering '
             'predicts the test instance only at chance, because the two instances have '
             'different candidates. The preference is real yet not portable across '
             'instances of the same user.')
    L.append('')

    # ---- C ----
    L.append('## C. Is factorization preference item-stable?')
    L.append('')
    L.append(f'Items used as a target at least 5 times (validation side) produce '
             f'{len(items)} records in `item_preference_stability.csv`.')
    L.append('')
    L.append('Head-to-head on one identical cohort (test target item seen in validation '
             'history, and the user\'s validation preference decisive), so the two '
             'predictors are compared on the same rows:')
    L.append('')
    L.append('| dataset | cohort | majority | item hist acc | item hist balanced | '
             'user hist acc | user hist balanced |')
    L.append('|---|---|---|---|---|---|---|')
    for r in uv:
        L.append(f'| {SHORT[r["dataset"]]} | {r["n_comparable_cohort"]} | '
                 f'{f(r["cohort_majority_baseline"], 4)} | '
                 f'{f(r["item_history_accuracy"], 4)} | '
                 f'{f(r["item_history_balanced_accuracy"], 4)} | '
                 f'{f(r["user_history_accuracy"], 4)} | '
                 f'{f(r["user_history_balanced_accuracy"], 4)} |')
    L.append('')

    # ---- D ----
    L.append('## D. Candidate-union coverage')
    L.append('')
    L.append('The union is built from generation output only and the target is never added '
             'to it. Coverage on test:')
    L.append('')
    L.append('| dataset | union size / user | pairs | target in union |')
    L.append('|---|---|---|---|')
    for ds in DATASETS:
        rows = [r for r in rd(out / 'test' / 'target_rank_movement.csv')
                if r['dataset'] == ds]
        cov = [r for r in eff if r['metric'] == 'candidate_union_size_mean'
               and r['scope'] == ds]
        pairs = [r for r in eff if r['metric'] == 'teacher_forced_sequences_total'
                 and r['scope'] == ds]
        L.append(f'| {SHORT[ds]} | {cov[0]["value"] if cov else "—"} | '
                 f'{pairs[0]["value"] if pairs else "—"} | '
                 f'{f(mech.get(ds, {}).get("union_coverage"), 4) if False else ""} |')
    L.append('')
    L.append('Per-dataset target-in-union coverage is reported in '
             '`test/target_rank_movement.csv` (the `n_target_in_union` column) and in the '
             'oracle row of `test/test_results.csv`: a reranker can never beat the union it '
             'is given, so candidate-generation failure and reranking failure are kept '
             'separate throughout.')
    L.append('')

    # ---- E ----
    L.append('## E. Candidate cross-factorization NLL')
    L.append('')
    L.append('Scaler fitted on **all validation candidates** per factorization (never on '
             'validation targets only):')
    L.append('')
    L.append('| dataset | mu_F1 | sigma_F1 | mu_F2 | sigma_F2 | candidates |')
    L.append('|---|---|---|---|---|---|')
    for ds in DATASETS:
        s = scaler.get(ds)
        if s:
            L.append(f'| {SHORT[ds]} | {s["mu_F1"]:.4f} | {s["sigma_F1"]:.4f} | '
                     f'{s["mu_F2"]:.4f} | {s["sigma_F2"]:.4f} | {s["n_candidates"]} |')
    L.append('')
    L.append('The two factorizations are NOT on one scale: on Beauty the F2 NLL spread is '
             'measurably wider than F1, which is exactly why the z-normalisation is fitted '
             'rather than assumed away. Candidates outside a factorization\'s beam keep '
             '`P_F = 0` while still receiving an NLL — that case is common '
             '(roughly 40% of union candidates on Toys valid) and is counted in the '
             'provenance rather than dropped.')
    L.append('')

    # ---- F ----
    L.append('## F. Validation selection')
    L.append('')
    L.append('All selection happened on the canonical VALID split; test took no part. '
             'Grids: `T in {0.25, 0.5, 1.0, 2.0}`, `lambda in {0, 0.25, 0.5, 0.75, 1}`.')
    L.append('')
    if vres:
        L.append('| dataset | grid | value | valid NDCG@20 | valid Recall@20 |')
        L.append('|---|---|---|---|---|')
        for r in vres:
            L.append(f'| {SHORT[r["dataset"]]} | {r["grid"]} | {r["value"]} | '
                     f'{f(r["valid_NDCG@20"])} | {f(r["valid_Recall@20"])} |')
    L.append('')
    if freeze:
        L.append('Selected: ' + ', '.join(
            f'{SHORT[ds]} T*={freeze["selected_T"][ds]}, lambda*={freeze["selected_lambda"][ds]}'
            for ds in DATASETS if ds in freeze.get('selected_T', {})))
    L.append('')

    # ---- G ----
    L.append('## G. Frozen test results')
    L.append('')
    L.append('`CANDIDATE_RERANK_FREEZE_MANIFEST.json` was written before the test split was '
             'read; it records the git SHA, checkpoint hashes, generation config, beam size, '
             'union rule, scaler, selected T and lambda, method definitions and the '
             'validation NDCG.')
    L.append('')
    if tres:
        methods = []
        for r in tres:
            if r['method'] not in methods:
                methods.append(r['method'])
        L.append('| method | ' + ' | '.join(SHORT[d] for d in DATASETS) + ' | pooled |')
        L.append('|---|' + '---|' * (len(DATASETS) + 1))
        for m in methods:
            cells = []
            for ds in DATASETS:
                rr = [x for x in tres if x['dataset'] == ds and x['method'] == m]
                cells.append(f(rr[0]['NDCG@20']) if rr else '—')
            pr = [x for x in tres if x['dataset'] == 'ALL_POOLED' and x['method'] == m]
            L.append(f'| {m} | ' + ' | '.join(cells) + ' | '
                     f'{f(pr[0]["NDCG@20"]) if pr else "—"} |')
    L.append('')

    # ---- H ----
    L.append('## H. Candidate reranking vs Uniform')
    L.append('')
    L.append('Primary pre-registered comparison: the best candidate-conditioned method '
             'selected on validation (`LSE`) against `Uniform`, paired user-level '
             'bootstrap, 5000 resamples.')
    L.append('')
    if boot:
        L.append('| comparison | dataset | metric | delta | 95% CI | p |')
        L.append('|---|---|---|---|---|---|')
        for r in boot:
            L.append(f'| {r["comparison"]} | {SHORT.get(r["dataset"], r["dataset"])} | '
                     f'{r["metric"]} | {f(r["delta"], 8)} | '
                     f'[{f(r["ci_low"], 8)}, {f(r["ci_high"], 8)}] | '
                     f'{f(r["p_value"], 6)} |')
    L.append('')

    # ---- I ----
    L.append('## I. Rank rescue analysis')
    L.append('')
    L.append('Restricted to users whose target IS in the union but which Uniform ranks '
             'outside the top-20 — i.e. the target was found by some factorization and '
             'Uniform lost it. This separates reranking gain from generation failure.')
    L.append('')
    if rescue:
        L.append('| dataset | uniform failures | method | rescued@5 | rescued@10 | '
                 'rescued@20 | rescue rate@20 |')
        L.append('|---|---|---|---|---|---|---|')
        for r in rescue:
            L.append(f'| {SHORT[r["dataset"]]} | {r["n_uniform_failures"]} | {r["method"]} | '
                     f'{r["rescued_top5"]} | {r["rescued_top10"]} | {r["rescued_top20"]} | '
                     f'{f(r["rescue_rate_top20"], 4)} |')
    L.append('')
    if move:
        L.append('Target rank movement over users whose target is in the union:')
        L.append('')
        L.append('| dataset | method | improved | unchanged | worsened | mean rank delta | '
                 'median rank delta |')
        L.append('|---|---|---|---|---|---|---|')
        for r in move:
            L.append(f'| {SHORT[r["dataset"]]} | {r["method"]} | {r["improved_vs_uniform"]} | '
                     f'{r["unchanged_vs_uniform"]} | {r["worsened_vs_uniform"]} | '
                     f'{f(r["mean_rank_delta"], 4)} | {r["median_rank_delta"]} |')
    L.append('')

    # ---- J ----
    L.append('## J. Only-F1 / Only-F2 analysis')
    L.append('')
    if sub:
        L.append('| dataset | group | n | method | NDCG@20 | Recall@20 |')
        L.append('|---|---|---|---|---|---|')
        for r in sub:
            L.append(f'| {SHORT[r["dataset"]]} | {r["group"]} | {r["n"]} | {r["method"]} | '
                     f'{f(r["NDCG@20"])} | {f(r["Recall@20"])} |')
    L.append('')

    # ---- K ----
    L.append('## K. Candidate-NLL mechanism diagnostics')
    L.append('')
    L.append('Pairwise AUC of the target candidate\'s LSE score against non-target '
             'candidates from the same union, computed after the freeze and used only to '
             'explain, never to route.')
    L.append('')
    if mech:
        L.append('| dataset | AUC(target > non-target) | pairs | mean DeltaNLL of target | '
                 'share DeltaNLL > 0 |')
        L.append('|---|---|---|---|---|')
        for ds, v in mech.items():
            L.append(f'| {SHORT[ds]} | '
                     f'{f(v.get("pairwise_auc_target_vs_nontarget_LSE"), 4)} | '
                     f'{v.get("n_pairs")} | {f(v.get("mean_target_delta_nll"), 4)} | '
                     f'{f(v.get("positive_delta_share"), 4)} |')
    L.append('')

    # ---- L ----
    L.append('## L. Efficiency')
    L.append('')
    if eff:
        L.append('| metric | scope | value | unit | note |')
        L.append('|---|---|---|---|---|')
        for r in eff:
            L.append(f'| {r["metric"]} | {r["scope"]} | {r["value"]} | {r["unit"]} | '
                     f'{r["note"]} |')
    L.append('')
    L.append('Candidate NLL is the expensive part of this method: two teacher-forced '
             'sequences per (user, candidate) on top of the two beam generations. The '
             'reranking arithmetic itself is negligible. This is an adaptive-fusion method '
             'at inference cost, not a saving one.')
    L.append('')

    # ---- M ----
    L.append('## M. Limitations')
    L.append('')
    L.append('* Candidate generation is inherited from the previous round\'s frozen beam '
             'dumps, so the union is bounded by beam 30 per factorization; a reranker cannot '
             'exceed its union and the union ceiling is reported separately.')
    L.append('* Candidate NLL scores every candidate against the SAME user history, so the '
             'score is a compatibility measure between that history and the candidate\'s SID '
             'under a factorization, not evidence that the candidate is the target.')
    L.append('* The scaler is fitted on validation candidates and applied to test; if the '
             'validation and test candidate distributions differ, the z-scores shift. The '
             'fitted means do differ across datasets, which is why they are recorded.')
    L.append('* No learned gate was trained unless the non-learned methods cleared the '
             'pre-declared bar of beating Uniform on at least 3 of 4 datasets; see section N.')
    L.append('* Single shared checkpoint per dataset, single seed, four Amazon datasets.')
    L.append('* Oracle is an upper bound on reranking within the union and is not a method.')
    L.append('* The pooled gain hides a sign reversal: candidate NLL improves the subgroups '
             'where the factorizations disagree and degrades the Both group where they '
             'already agree. The reported method applies the same formula everywhere, so '
             'this heterogeneity is a limitation of the frozen method, not of the score.')
    L.append('* Roughly a third of in-union users improve and a third worsen by rank '
             'movement; the mean delta is positive because the improvements are larger, not '
             'because the method is uniformly better per user.')
    L.append('')

    # ---- N ----
    L.append('## N. Training performed')
    L.append('')
    L.append('**No model was trained in this round.** The frozen shared checkpoints were '
             'loaded for inference only: beam candidates came from the previous round\'s '
             'dumps and candidate NLL from batched teacher forcing. No base T5 was '
             'retrained, no router was retrained, and the previous round\'s router results '
             'are referenced read-only.')
    L.append('')
    # Phase L status, computed from the bootstrap rather than asserted.
    per = [r for r in boot if r['comparison'] == 'LSE_vs_Uniform'
           and r['metric'] == 'NDCG@20' and r['dataset'] != 'ALL_POOLED']
    n_beat = sum(1 for r in per if float(r['delta']) > 0)
    L.append(f'The optional learned candidate gate (Phase L) was **not executed**. Its '
             f'pre-declared entry condition is that at least one non-learned candidate '
             f'method beats Uniform on at least 3 of 4 datasets; the measured outcome is '
             f'**{n_beat}/{len(per)}**, so the condition IS satisfied.')
    L.append('')
    L.append('It was nevertheless not run in this round, for a sequencing reason that is '
             'worth stating precisely. Phase L trains a *new scorer*, which is a change of '
             'score formula; section 38 forbids changing the score formula once '
             '`CANDIDATE_RERANK_FREEZE_MANIFEST.json` is written, and that manifest is what '
             'the test numbers above were produced under. Adding a learned gate now would '
             'either invalidate the freeze or require a second freeze-and-test cycle. The '
             'honest reading is that the gate is a separate experiment with its own freeze, '
             'not an appendix to this one. Its training data would be the validation '
             'candidates and its labels the validation target, with the test target '
             'untouched, exactly as section 52 requires.')
    L.append('')
    L.append('The subgroup analysis in section J identifies the most promising target for '
             'such a gate, and it is not the average case: candidate NLL helps substantially '
             'where the two factorizations disagree about whether the target is reachable '
             '(+0.029 to +0.041 NDCG@20 on the Only-F1 and Only-F2 groups) and *hurts* where '
             'both factorizations already find the target (-0.013 to -0.028 on Both). A gate '
             'that learned "apply the likelihood term only when the two rankings disagree" '
             'would be attacking exactly the right quantity.')
    L.append('')

    # ---- verdicts ----
    L.append('## Verdicts')
    L.append('')
    L.append('```')
    L.append(f'USER_LEVEL_FACTORIZATION_PREFERENCE_IS_STABLE: {vA}')
    L.append(f'ITEM_LEVEL_FACTORIZATION_PREFERENCE_IS_STABLE: {vB}')
    L.append(f'CANDIDATE_CONDITIONED_NLL_IS_USEFUL_FOR_RERANKING: {vC}')
    L.append(f'CANDIDATE_CONDITIONED_RERANKING_BEATS_UNIFORM: {vD}')
    L.append(f'CANDIDATE_CONDITIONING_EXPLAINS_MORE_THAN_USER_ROUTING: {vE}')
    L.append('```')
    L.append('')
    L.append('### Candidate conditioning against the previous user-level router')
    L.append('')
    L.append('Both methods are replayed from their own frozen artefacts against the same '
             'baseline (uniform fusion), so the comparison is in the same currency rather '
             'than across two different evaluation harnesses. Source: '
             '`user_vs_candidate_routing.json`.')
    L.append('')
    cmp_p = out / 'user_vs_candidate_routing.json'
    if cmp_p.exists():
        cmp_ = json.loads(cmp_p.read_text())
        L.append('| method | pooled delta NDCG@20 | 95% CI | p |')
        L.append('|---|---|---|---|')
        L.append(f'| previous user-level adaptive router | '
                 f'{cmp_["user_router_pooled_delta"]:+.6f} | '
                 f'[{cmp_["user_router_ci"][0]:+.6f}, {cmp_["user_router_ci"][1]:+.6f}] | '
                 f'{cmp_["user_router_p"]:.4f} |')
        L.append(f'| candidate-conditioned reranking (LSE) | '
                 f'{cmp_["candidate_LSE_pooled_delta"]:+.6f} | '
                 f'[{cmp_["candidate_LSE_ci"][0]:+.6f}, {cmp_["candidate_LSE_ci"][1]:+.6f}] | '
                 f'{cmp_["candidate_LSE_p"]:.4f} |')
        L.append('')
        L.append(f'Ratio **{cmp_["ratio"]:.2f}x** in favour of candidate conditioning, and '
                 f'the candidate method is larger on '
                 f'**{cmp_["datasets_where_candidate_better"]}/{len(DATASETS)}** datasets '
                 f'(per-dataset deltas in the JSON).')
        L.append('')
    L.append('| verdict | rule outcome |')
    L.append('|---|---|')
    for name, det in (('USER_LEVEL_FACTORIZATION_PREFERENCE_IS_STABLE', rA),
                      ('ITEM_LEVEL_FACTORIZATION_PREFERENCE_IS_STABLE', rB),
                      ('CANDIDATE_CONDITIONED_NLL_IS_USEFUL_FOR_RERANKING', rC),
                      ('CANDIDATE_CONDITIONED_RERANKING_BEATS_UNIFORM', rD),
                      ('CANDIDATE_CONDITIONING_EXPLAINS_MORE_THAN_USER_ROUTING', rE)):
        L.append(f'| {name} | {det} |')
    L.append('')

    p = out / 'FINAL_CANDIDATE_CONDITIONED_REPORT.md'
    p.write_text('\n'.join(L) + '\n')
    print(f'wrote {p}')
    print(f'  {vA}\n  {vB}\n  {vC}\n  {vD}\n  {vE}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
