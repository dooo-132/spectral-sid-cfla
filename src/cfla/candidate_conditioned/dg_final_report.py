#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FINAL_DISAGREEMENT_GATE_REPORT.md (prompt sections 66-67).

Verdicts come from rules applied to the artefacts; no branch promotes a preferred story.

Usage
  python3 dg_final_report.py --outdir outputs/disagreement_gate
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
RUNGS = ('Only_F1', 'Only_F2', 'Both', 'Neither')


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


def gv(row, *keys, default=None):
    for k in keys:
        if k in row and row[k] not in ('', None):
            return row[k]
    return default


# ---------------------------------------------------------------------------
# verdict rules (pre-declared)
# ---------------------------------------------------------------------------
def verdict_nontrivial(dev_rows):
    """The learned gate must be more than a constant: its RouterDev NDCG@20 must differ
    from every constant-gate value AND its gate values must vary across candidates."""
    cons = [r for r in dev_rows if r['kind'] == 'constant']
    learned = [r for r in dev_rows if r['kind'] != 'constant']
    if not learned or not cons:
        return 'NOT SUPPORTED', 'missing dev rows'
    best_c = max(float(r['dev_ndcg20']) for r in cons)
    best_l = max(learned, key=lambda r: float(r['dev_ndcg20']))
    spread = None
    gp = Path('outputs/disagreement_gate/test/gate_behaviour.csv')
    if gp.exists():
        vals = [float(r['mean_g']) for r in rd(gp) if r['group'].startswith('source_')
                or r['group'].startswith('rankdis_')]
        if vals:
            spread = max(vals) - min(vals)
    det = (f'best learned dev NDCG@20={float(best_l["dev_ndcg20"]):.6f} vs best constant '
           f'{best_c:.6f}; per-view mean-g spread={spread:.4f}' if spread is not None else
           f'best learned {float(best_l["dev_ndcg20"]):.6f} vs best constant {best_c:.6f}')
    nonconst = spread is not None and spread > 0.02
    better = float(best_l['dev_ndcg20']) > best_c + 1e-9
    if nonconst and better:
        return 'SUPPORTED', det
    if nonconst or better:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def _boot(boot, comp, ds='ALL_POOLED', metric='NDCG@20'):
    for r in boot:
        if r['comparison'] == comp and r['dataset'] == ds and r['metric'] == metric:
            return r
    return None


def verdict_beats_uniform(boot):
    if not boot:
        return 'NOT SUPPORTED', 'no bootstrap'
    p = _boot(boot, 'Gate_vs_Uniform')
    per = [r for r in boot if r['comparison'] == 'Gate_vs_Uniform'
           and r['metric'] == 'NDCG@20' and r['dataset'] != 'ALL_POOLED']
    if not p:
        return 'NOT SUPPORTED', 'no pooled row'
    sig = float(p['ci_low']) > 0
    pos = sum(1 for r in per if float(r['delta']) > 0)
    det = (f'pooled delta={float(p["delta"]):+.6f} '
           f'CI=[{float(p["ci_low"]):+.6f},{float(p["ci_high"]):+.6f}] '
           f'p={float(p["p_value"]):.4f}; positive in {pos}/{len(per)} datasets')
    if sig and pos == len(per):
        return 'SUPPORTED', det
    if sig or pos >= len(per) // 2 + 1:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_beats_lse(boot):
    if not boot:
        return 'NOT SUPPORTED', 'no bootstrap'
    p = _boot(boot, 'Gate_vs_LSE')
    per = [r for r in boot if r['comparison'] == 'Gate_vs_LSE'
           and r['metric'] == 'NDCG@20' and r['dataset'] != 'ALL_POOLED']
    if not p:
        return 'NOT SUPPORTED', 'no pooled row'
    sig = float(p['ci_low']) > 0
    pos = sum(1 for r in per if float(r['delta']) > 0)
    sig_n = sum(1 for r in per if float(r['ci_low']) > 0)
    det = (f'pooled delta={float(p["delta"]):+.6f} '
           f'CI=[{float(p["ci_low"]):+.6f},{float(p["ci_high"]):+.6f}] '
           f'p={float(p["p_value"]):.4f}; positive in {pos}/{len(per)} datasets, '
           f'CI above zero in {sig_n}/{len(per)}')
    if sig and pos == len(per):
        return 'SUPPORTED', det
    if sig:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_preserves(retention):
    """SUPPORTED only if BOTH halves of the success pattern hold:
       Only groups keep most of the LSE gain, AND Both recovers most of the Uniform level."""
    if not retention:
        return 'NOT SUPPORTED', 'no retention rows'
    only_ret, both_rec = [], []
    for r in retention:
        if r['group'] in ('Only_F1', 'Only_F2') and r.get('OnlyGainRetention'):
            only_ret.append(float(r['OnlyGainRetention']))
        if r['group'] == 'Both' and r.get('BothRecovery'):
            both_rec.append(float(r['BothRecovery']))
    m_only = float(np.mean(only_ret)) if only_ret else float('nan')
    m_both = float(np.mean(both_rec)) if both_rec else float('nan')
    only_ok = only_ret and min(only_ret) >= 0.5
    both_ok = both_rec and m_both >= 0.5
    det = (f'OnlyGainRetention mean={m_only:.3f} (min '
           f'{min(only_ret):.3f}, n={len(only_ret)}); '
           f'BothRecovery mean={m_both:.3f} (n={len(both_rec)})')
    if only_ok and both_ok:
        return 'SUPPORTED', det
    if only_ok or both_ok:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_cheap(eff_dir, distillation_done):
    if not distillation_done:
        return 'NOT RUN', ('optional Phase L not executed; its entry condition '
                           '(learned gate beating frozen LSE on RouterHoldout) is '
                           'evaluated in section L')
    return 'NOT RUN', 'see efficiency/ outputs'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/disagreement_gate')
    a = ap.parse_args(argv)
    out = Path(a.outdir)

    dev = rd(out / 'router_dev_results.csv')
    hold = rd(out / 'router_holdout_results.csv')
    tres = rd(out / 'test' / 'test_results.csv')
    boot = rd(out / 'test' / 'bootstrap.csv')
    sub = rd(out / 'test' / 'subgroup_results.csv')
    ret = rd(out / 'test' / 'retention.csv')
    beh = rd(out / 'test' / 'gate_behaviour.csv')
    src = rd(out / 'test' / 'candidate_source_behaviour.csv')
    rescue = rd(out / 'test' / 'rank_rescue.csv')
    damage = rd(out / 'test' / 'damage_analysis.csv')
    eff = rd(out / 'efficiency.csv')
    man = json.loads((out / 'GATE_FREEZE_MANIFEST.json').read_text()) \
        if (out / 'GATE_FREEZE_MANIFEST.json').exists() else {}
    cfg = json.loads((out / 'training_config.json').read_text()) \
        if (out / 'training_config.json').exists() else {}

    vA, rA = verdict_nontrivial(dev)
    vB, rB = verdict_beats_uniform(boot)
    vC, rC = verdict_beats_lse(boot)
    vD, rD = verdict_preserves(ret)
    distill_done = (out / 'efficiency' / 'DISTILLATION.json').exists()
    vE, rE = verdict_cheap(out / 'efficiency', distill_done)

    def pooled(m):
        for r in tres:
            if r['dataset'] == 'ALL_POOLED' and r['method'] == m:
                return float(r['NDCG@20'])
        return float('nan')

    def held(role, m):
        for r in hold:
            if r['role'] == role:
                return float(r.get(f'{m}_pooled', 'nan'))
        return float('nan')

    L = []
    L.append('# FINAL_DISAGREEMENT_GATE_REPORT.md')
    L.append('')
    L.append('A leakage-safe, listwise-trained candidate gate that decides per candidate '
             'how far to move its score from the strong Uniform baseline toward the '
             'cross-factorization likelihood score, using only inference-visible evidence.')
    L.append('')
    L.append('**The canonical test section of this report is a POST-HOC CANONICAL TEST '
             'BENCHMARK.** The gate idea was motivated by the previous round\'s canonical '
             'test subgroup analysis, so the canonical test set is **not** a pristine, '
             'discovery-independent confirmatory holdout and is not described as one here. '
             'Section M records what a genuinely pristine dataset would require.')
    L.append('')
    L.append('Governance: no historical SID array, checkpoint, beam dump, parquet, CSV, '
             'JSON, scaler, manifest, report or router output was overwritten; this round '
             'wrote only under `outputs/disagreement_gate/`. The base shared T5 was never '
             'loaded for training or modified, no SID tokenizer was retrained, and '
             '`paper/` was not touched. The cross-factorization candidate NLL was **reused, '
             'not recomputed** (`CANDIDATE_NLL_REUSE_AUDIT.md`).')
    L.append('')

    # ---- A ----
    L.append('## A. State recovery')
    L.append('')
    L.append('`frozen_state.md` records the branch and HEAD, the four frozen shared '
             'checkpoint hashes, the candidate union and NLL paths, the validation scaler, '
             'the frozen Uniform/LSE/MinNLL definitions and per-user normalisation, the '
             'previous test results re-derived from disk, the 0-based SID-row item-id '
             'convention, and the stop-condition check. The previous round\'s headline '
             'findings were re-verified rather than trusted: LSE beats Uniform on 4/4 '
             'datasets, and the NLL correction helps the Only-F1/Only-F2 groups '
             '(+0.029..+0.041 NDCG@20) while hurting Both (-0.013..-0.028).')
    L.append('')

    # ---- B ----
    L.append('## B. Training protocol')
    L.append('')
    L.append('Data: canonical VALID candidate unions only, with their frozen retrieval '
             'scores and frozen candidate NLL. The test split contributed nothing to '
             'training or selection.')
    L.append('')
    L.append('Score definition (frozen before training, prompt section 19):')
    L.append('')
    L.append('```')
    L.append('S(u,i) = (1 - g(u,i)) * Z_U(u,i) + g(u,i) * Z_L(u,i)')
    L.append('g = 0 reproduces Uniform exactly;  g = 1 reproduces the frozen LSE exactly')
    L.append('```')
    L.append('')
    L.append('`Z_U` and `Z_L` are the **same** per-user candidate-union z-scores the '
             'previous round used, so the gate cannot redefine the scoring space; it only '
             'chooses a mixing weight.')
    L.append('')
    L.append('Objective: a user-level **listwise** loss over users whose validation target '
             'lies inside their frozen candidate union,')
    L.append('')
    L.append('```')
    L.append('L = -log softmax_i S(u,i) [i*]  +  lambda_g * mean_i g(u,i)')
    L.append('```')
    L.append('')
    L.append('Users whose target is outside the union carry weight 0 and are recorded as '
             '`TARGET_OUTSIDE_UNION`; the target was never added to any union. The '
             'Only-F1/Only-F2/Both group label was never an input and never supervision.')
    L.append('')
    if cfg:
        L.append('| setting | value |')
        L.append('|---|---|')
        for k in ('seed', 'lr', 'weight_decay', 'batch_users', 'max_epochs', 'patience'):
            if k in cfg:
                L.append(f'| {k} | {cfg[k]} |')
        L.append(f'| lambda_g grid | {cfg.get("lambda_g")} |')
        L.append(f'| selection | {cfg.get("selection")} |')
        L.append(f'| loss | {cfg.get("loss")} |')
    L.append('')
    L.append('One **global** gate is trained on the four datasets\' RouterTrain candidates '
             'pooled, after per-dataset feature standardisation (mean/std fitted on that '
             "dataset's RouterTrain rows) and with a pre-declared 4-dimensional dataset "
             'one-hot. Optimiser AdamW, lr 1e-3, weight decay 1e-4, batch 64 users, max 100 '
             'epochs, patience 10, early stopping on pooled RouterDev NDCG@20.')
    L.append('')
    ep = out / 'eligible_users.json'
    if ep.exists():
        e = json.loads(ep.read_text())['eligible_users']
        L.append('Eligible users (validation target inside the frozen candidate union): '
                 f'RouterTrain {e["RouterTrain"]}, RouterDev {e["RouterDev"]}, '
                 f'RouterHoldout {e["RouterHoldout"]}. Users whose target is outside the '
                 'union are recorded as `TARGET_OUTSIDE_UNION` and carry zero loss weight.')
        L.append('')

    # ---- C ----
    L.append('## C. Feature audit')
    L.append('')
    L.append('`FEATURE_AUDIT.md` lists all 46 features with dimension, source, '
             '`uses target? NO`, `available at inference? YES`, and `requires NLL scoring?` '
             'for each. The 387-dim history hidden states were deliberately not used in the '
             'primary gate (prompt section 15).')
    L.append('')

    # ---- D ----
    L.append('## D. RouterTrain / RouterDev / RouterHoldout')
    L.append('')
    L.append('| role | Uniform | LSE | Gate | source |')
    L.append('|---|---|---|---|---|')
    for role in ('RouterTrain', 'RouterDev', 'RouterHoldout'):
        L.append(f'| {role} | {held(role, "Uniform"):.6f} | {held(role, "LSE"):.6f} | '
                 f'**{held(role, "Gate"):.6f}** | `router_holdout_results.csv` |')
    L.append('')
    L.append('RouterHoldout was touched exactly once, after architecture, `lambda_g` and '
             'early stopping were fixed; it took no part in selection.')
    L.append('')
    L.append('RouterDev model selection (`router_dev_results.csv`), pooled NDCG@20:')
    L.append('')
    L.append('| model | kind | lambda_g | params | RouterDev NDCG@20 |')
    L.append('|---|---|---|---|---|')
    for r in sorted(dev, key=lambda r: -float(r['dev_ndcg20']))[:12]:
        L.append(f'| {r["model"]} | {r["kind"]} | {r["lambda_g"]} | {r["params"]} | '
                 f'{f(r["dev_ndcg20"])} |')
    L.append('')
    if man.get('gate'):
        g = man['gate']
        L.append(f'Winner: **{g["name"]}** ({g["architecture"]}), {g["params"]} parameters, '
                 f'`lambda_g` = {g["lambda_g"]}, best epoch {g["best_epoch"]}, training seed '
                 f'{g["training_seed"]}.')
    L.append('')

    # ---- E ----
    L.append('## E. Frozen gate')
    L.append('')
    if man:
        L.append('```')
        L.append(f'git SHA                  : {man.get("git_sha")}')
        L.append(f'gate                     : {man["gate"]["name"]}')
        L.append(f'architecture             : {man["gate"]["architecture"]}')
        L.append(f'parameters               : {man["gate"]["params"]}')
        L.append(f'checkpoint sha256        : {man["gate"]["checkpoint_sha256"][:32]}…')
        L.append(f'training seed            : {man["gate"]["training_seed"]}')
        L.append(f'lambda_g                 : {man["gate"]["lambda_g"]}')
        L.append(f'features                 : {man["features"]["dimension"]} '
                 f'(uses target: {man["features"]["uses_target"]})')
        L.append(f'manifest written         : {man["created_local"]}')
        L.append(f'test read at freeze time : {man["test_metrics_computed_at_freeze_time"]}')
        L.append('```')
        L.append('')
        L.append('The frozen score definitions and the per-dataset feature scaler are '
                 'recorded in the same manifest, so `g = 0` and `g = 1` remain checkable '
                 'against the previous round\'s Uniform and LSE.')
    L.append('')

    # ---- F ----
    L.append('## F. Post-hoc canonical test benchmark')
    L.append('')
    L.append('This is a **post-hoc replay on a benchmark whose subgroup structure motivated '
             'the method**, not a pristine confirmation. Method selection used RouterTrain '
             'and RouterDev only; the test split was read after the freeze manifest existed.')
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
            L.append(f'| {m} | ' + ' | '.join(cells) + f' | {f(pooled(m))} |')
    L.append('')
    L.append('`Oracle` here is the in-union ceiling: NDCG@20 for a perfect reranker given '
             'the same candidate union, i.e. the fraction of users whose target the '
             'generator reached at all. It bounds what any reranker can do and is not a '
             'method. The gap between it and every real method is candidate-generation '
             'failure, not reranking failure.')
    L.append('')

    # ---- G ----
    L.append('## G. Uniform vs LSE vs LearnedGate')
    L.append('')
    L.append('Paired user-level bootstrap, 5000 resamples, seed 20260912:')
    L.append('')
    if boot:
        L.append('| comparison | dataset | metric | delta | 95% CI | p |')
        L.append('|---|---|---|---|---|---|')
        for r in boot:
            if r['metric'] != 'NDCG@20':
                continue
            L.append(f'| {r["comparison"]} | {SHORT.get(r["dataset"], r["dataset"])} | '
                     f'{r["metric"]} | {f(r["delta"], 8)} | '
                     f'[{f(r["ci_low"], 8)}, {f(r["ci_high"], 8)}] | {f(r["p_value"], 6)} |')
    L.append('')

    # ---- H ----
    L.append('## H. Only-F1 / Only-F2')
    L.append('')
    L.append('Per-subgroup NDCG@20 on test, with the two retention quantities:')
    L.append('')
    L.append('```')
    L.append('OnlyGainRetention = (Gate - Uniform) / (LSE - Uniform)   where LSE > Uniform')
    L.append('BothRecovery     = (Gate - LSE)     / (Uniform - LSE)   where Uniform > LSE')
    L.append('```')
    L.append('')
    if ret:
        L.append('| dataset | group | n | Uniform | LSE | Gate | OnlyGainRetention | '
                 'BothRecovery |')
        L.append('|---|---|---|---|---|---|---|---|')
        for r in ret:
            L.append(f'| {SHORT[r["dataset"]]} | {r["group"]} | {r["n"]} | '
                     f'{f(r["Uniform"])} | {f(r["LSE"])} | {f(r["Gate"])} | '
                     f'{r.get("OnlyGainRetention", "") or "—"} | '
                     f'{r.get("BothRecovery", "") or "—"} |')
    L.append('')

    # ---- I ----
    L.append('## I. Both protection')
    L.append('')
    br = [float(r['BothRecovery']) for r in ret if r['group'] == 'Both'
          and r.get('BothRecovery')]
    if br:
        L.append(f'BothRecovery per dataset: ' +
                 ', '.join(f'{SHORT[r["dataset"]]}={float(r["BothRecovery"]):+.4f}'
                           for r in ret if r['group'] == 'Both' and r.get('BothRecovery')))
        L.append('')
        L.append(f'Mean {np.mean(br):+.4f}. A value of 1.0 would mean the gate fully '
                 f'restored Uniform\'s ranking on the Both group; 0.0 would mean it stayed '
                 f'at LSE. Values near 0 mean the gate did **not** protect the agreement '
                 f'cases, even where its overall NDCG improved.')
    L.append('')

    # ---- J ----
    L.append('## J. Rescue vs damage')
    L.append('')
    L.append('Reranking has two failure modes: failing to rescue what Uniform lost, and '
             'destroying what Uniform had. Both are reported.')
    L.append('')
    if rescue:
        L.append('| dataset | method | uniform failures | rescued@5 | rescued@10 | '
                 'rescued@20 |')
        L.append('|---|---|---|---|---|---|')
        for r in rescue:
            L.append(f'| {SHORT[r["dataset"]]} | {r["method"]} | {r["n_uniform_failures"]} | '
                     f'{r["rescued@5"]} | {r["rescued@10"]} | {r["rescued@20"]} |')
        L.append('')
    if damage:
        L.append('| dataset | method | uniform successes | pushed out of top-20 | '
                 'damage rate |')
        L.append('|---|---|---|---|---|')
        for r in damage:
            L.append(f'| {SHORT[r["dataset"]]} | {r["method"]} | '
                     f'{r["n_uniform_successes"]} | {r["pushed_out_top20"]} | '
                     f'{f(r["damage_rate"], 4)} |')
    L.append('')

    # ---- K ----
    L.append('## K. Gate behaviour')
    L.append('')
    L.append('The design hoped for larger `g` on disagreement and smaller on agreement. '
             'What was measured is reported as-is.')
    L.append('')
    if src:
        L.append('Candidate-level views (target-free, so these are pure properties of the '
                 'gate):')
        L.append('')
        L.append('| dataset | view | candidates | mean g |')
        L.append('|---|---|---|---|')
        for r in src:
            L.append(f'| {SHORT[r["dataset"]]} | {r["view"]} | {r["n_candidates"]} | '
                     f'{r["mean_g"]} |')
        L.append('')
    if beh:
        L.append('By target subgroup (evaluation only):')
        L.append('')
        L.append('| dataset | group | users | mean g | median g |')
        L.append('|---|---|---|---|---|')
        for r in beh:
            if r['group'] in RUNGS:
                L.append(f'| {SHORT[r["dataset"]]} | {r["group"]} | {r["n_users"]} | '
                         f'{r["mean_g"]} | {r["median_g"]} |')
    L.append('')

    # ---- L ----
    L.append('## L. Efficiency')
    L.append('')
    L.append('The learned gate still requires candidate NLL for every candidate, so it '
             '**does not save compute**; it changes only how the frozen scores are mixed. '
             'The optional cheap-gate distillation (Phase L of the prompt) was entered only '
             'if the learned gate beat frozen LSE on RouterHoldout **and** the pooled test '
             'direction was positive.')
    L.append('')
    gh, lh = held('RouterHoldout', 'Gate'), held('RouterHoldout', 'LSE')
    L.append(f'RouterHoldout: Gate {gh:.6f} vs LSE {lh:.6f} → '
             f'{"condition met" if gh > lh else "condition NOT met"}. Pooled test direction: '
             f'{"positive" if pooled("Gate") > pooled("LSE") else "not positive"}.')
    L.append('')
    if eff:
        L.append('| metric | scope | value | unit | note |')
        L.append('|---|---|---|---|---|')
        for r in eff:
            L.append(f'| {r["metric"]} | {r.get("scope","")} | {r["value"]} | '
                     f'{r.get("unit","")} | {r.get("note","")} |')
    L.append('')

    # ---- M ----
    L.append('## M. Confirmatory-dataset availability')
    L.append('')
    L.append('`CONFIRMATORY_DATASET_PLAN.md`. The only dataset never used in any prior '
             'report is **`Sports_and_Outdoors`** (0 occurrences across every study '
             'output). It has canonical splits (35,598 users), item embeddings (18,357 '
             'items) and an RQ-VAE-3layer SID at exactly the right shape (18357, 4), but its '
             'spectral SID is **K=2, L=3** (`t5_spectral_v2.npy`), not the K=3/L=4 arm this '
             'study uses, and no shared factorization model exists for it.')
    L.append('')
    L.append('```')
    L.append('NO PRISTINE DATASET-LEVEL CONFIRMATORY SET AVAILABLE AT ZERO TRAINING COST')
    L.append('```')
    L.append('')
    L.append('Reaching one needs a generated K=3 SID plus a shared model trained on that '
             'dataset, about 3.5-4 GPU-hours, and it would then confirm the *mechanism* '
             'rather than the specific frozen gate. No large-scale training was started.')
    L.append('')

    # ---- N ----
    L.append('## N. Limitations')
    L.append('')
    L.append('* **The canonical test replay is post-hoc.** It is named as such in section F '
             'and must not be quoted as pristine confirmation.')
    L.append('* **The gate did not learn the intended disagreement logic.** Section K shows '
             'higher `g` where both factorizations already agree, and a U-shaped profile '
             'against |rank difference| rather than the hoped-for monotone increase. The '
             'gain is therefore coming from the likelihood term generally, not from the '
             'targeted protection the design assumed.')
    L.append('* **Agreement protection largely failed.** BothRecovery stays near zero '
             '(section I), so the mechanism the gate was built for is not what produced the '
             'improvement.')
    L.append('* **The gain over LSE is small.** Pooled +0.000418 NDCG@20, significant; '
             'per dataset the CI includes zero on three of four.')
    L.append('* **Requires full candidate NLL at inference**, so it is an adaptive-fusion '
             'method, not an efficiency method.')
    L.append('* Single seed, single shared checkpoint per dataset, four Amazon datasets, '
             'one global gate.')
    L.append('* `Oracle` is a ceiling, not a method.')
    L.append('')

    # ---- O ----
    L.append('## O. Training performed')
    L.append('')
    L.append('| run | what trained | params | selection | result |')
    L.append('|---|---|---|---|---|')
    for r in dev:
        if r['kind'] != 'constant':
            L.append(f'| `{r["model"]}` | candidate gate | {r["params"]} | '
                     f'RouterDev pooled NDCG@20 | {f(r["dev_ndcg20"])} |')
    L.append('')
    L.append('Eight gate configurations were trained (2 architectures x 4 `lambda_g` '
             'values) on pooled RouterTrain candidates; the winner is frozen. Constant-gate '
             'baselines were evaluated, not trained. **The base shared T5 was not trained, '
             'modified or even loaded**; its beam outputs and the previously computed '
             'candidate NLL were consumed as frozen arrays. No SID tokenizer was retrained.')
    L.append('')

    # ---- verdicts ----
    L.append('## Verdicts')
    L.append('')
    L.append('```')
    L.append(f'DISAGREEMENT_AWARE_GATE_LEARNS_NONTRIVIAL_CANDIDATE_WEIGHTS: {vA}')
    L.append(f'LEARNED_GATE_BEATS_UNIFORM: {vB}')
    L.append(f'LEARNED_GATE_BEATS_FROZEN_LSE: {vC}')
    L.append(f'GATE_PRESERVES_DISAGREEMENT_GAIN_WHILE_PROTECTING_AGREEMENT_CASES: {vD}')
    L.append(f'CHEAP_SELECTIVE_NLL_SCORING_IS_VIABLE: {vE}')
    L.append('```')
    L.append('')
    L.append('| verdict | rule outcome |')
    L.append('|---|---|')
    for n_, d_ in (('DISAGREEMENT_AWARE_GATE_LEARNS_NONTRIVIAL_CANDIDATE_WEIGHTS', rA),
                   ('LEARNED_GATE_BEATS_UNIFORM', rB),
                   ('LEARNED_GATE_BEATS_FROZEN_LSE', rC),
                   ('GATE_PRESERVES_DISAGREEMENT_GAIN_WHILE_PROTECTING_AGREEMENT_CASES', rD),
                   ('CHEAP_SELECTIVE_NLL_SCORING_IS_VIABLE', rE)):
        L.append(f'| {n_} | {d_} |')
    L.append('')

    p = out / 'FINAL_DISAGREEMENT_GATE_REPORT.md'
    p.write_text('\n'.join(L) + '\n')
    print(f'wrote {p}')
    print(f'  {vA}\n  {vB}\n  {vC}\n  {vD}\n  {vE}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
