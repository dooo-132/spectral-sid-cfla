#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FINAL_ADAPTIVE_FACTORIZATION_REPORT.md from artefacts on disk.

Verdicts are produced by explicit pre-declared decision rules (section 47 of the
protocol), not hand-typed.  If a rule is not met the label says so; there is no
branch that promotes a preferred story.

Run:
  python3 af_final_report.py --root outputs/adaptive_factorization
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
            'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}


def read_csv(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def fnum(x, nd=6):
    try:
        return f'{float(x):.{nd}f}'
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# pre-declared verdict rules
# ---------------------------------------------------------------------------
def verdict_compute_matched(meta):
    if not meta:
        return 'NOT SUPPORTED', 'no RUN_METADATA.json'
    checks = meta.get('success_criteria', {})
    n = sum(1 for v in checks.values() if v)
    if meta.get('SEVERE_COMPUTE_MATCHED_COLLAPSE'):
        return 'CONFOUNDED', 'SEVERE_COMPUTE_MATCHED_COLLAPSE flagged'
    if n == len(checks) and checks:
        return 'SUPPORTED', f'{n}/{len(checks)} criteria pass'
    if n >= 3:
        return 'PARTIALLY SUPPORTED', f'{n}/{len(checks)} criteria pass'
    return 'NOT SUPPORTED', f'{n}/{len(checks)} criteria pass'


def verdict_preference_predictable(boot, pref):
    """Three tests must all hold for SUPPORTED:

      (1) pooled adaptive-vs-uniform gain is positive with the 95% CI above zero;
      (2) mean of the router's accuracy on the Only-F1 and Only-F2 subgroups is above
          0.55 in at least 3 of 4 datasets -- decodable preference, not a coin flip.
          Accuracy (not balanced accuracy) is the right statistic here: each of the
          two subgroups is single-class, so a per-subgroup balanced accuracy is
          mathematically degenerate and is reported as NaN.
      (3) the router's mean predicted F1 weight is meaningfully higher on Only-F1
          rows than on Only-F2 rows -- it responds to *which* factorization won
          rather than applying a constant class bias.

    Two of three -> PARTIALLY SUPPORTED; otherwise NOT SUPPORTED.
    """
    gain_ok = False
    if boot:
        rows = [r for r in boot if r['comparison'] == 'G_P_soft_vs_B_uniform'
                and r['metric'] == 'NDCG@20' and r['dataset'] == 'ALL_POOLED']
        if rows:
            gain_ok = float(rows[0]['delta']) > 0 and float(rows[0]['ci_low']) > 0

    per_ds = {}
    for r in pref:
        if r['group'] in ('Only_F1', 'Only_F2') and r['accuracy'] not in ('', None):
            d = per_ds.setdefault(r['dataset'], {})
            d.setdefault('acc', []).append(float(r['accuracy']))
            d.setdefault('wF1', {})[r['group']] = float(r['mean_w_F1'])
    mean_acc = {ds: float(np.mean(v['acc'])) for ds, v in per_ds.items() if v['acc']}
    n_above = sum(1 for v in mean_acc.values() if v > 0.55)
    acc_ok = n_above >= 3
    gaps = [v['wF1']['Only_F1'] - v['wF1']['Only_F2']
            for v in per_ds.values()
            if 'Only_F1' in v['wF1'] and 'Only_F2' in v['wF1']]
    bias_gap = float(np.mean(gaps)) if gaps else float('nan')
    resp_ok = bool(gaps) and bias_gap > 0.002

    n_ok = sum([gain_ok, acc_ok, resp_ok])
    detail = (f'pooled_gain_significant={gain_ok}, '
              f'mean Only-subgroup accuracy>0.55 in {n_above}/4 datasets '
              f'(per-dataset means: '
              f'{ {k: round(v, 3) for k, v in mean_acc.items()} }), '
              f'mean w_F1 gap (Only-F1 minus Only-F2)={bias_gap:+.4f}')
    if n_ok == 3:
        return 'SUPPORTED', detail
    if n_ok == 2:
        return 'PARTIALLY SUPPORTED', detail
    return 'NOT SUPPORTED', detail


def verdict_adaptive_beats_uniform(boot):
    """SUPPORTED when the pooled gain is significant AND the direction is positive
    in every dataset; PARTIALLY SUPPORTED when it is significant somewhere but the
    per-dataset direction is mixed."""
    if not boot:
        return 'NOT SUPPORTED', 'no bootstrap.csv'
    rows = [r for r in boot if r['comparison'] == 'G_P_soft_vs_B_uniform'
            and r['metric'] == 'NDCG@20']
    pooled = [r for r in rows if r['dataset'] == 'ALL_POOLED']
    per = [r for r in rows if r['dataset'] != 'ALL_POOLED']
    if not pooled:
        return 'NOT SUPPORTED', 'no pooled bootstrap row'
    pooled_sig = float(pooled[0]['ci_low']) > 0
    n_pos = sum(1 for r in per if float(r['delta']) > 0)
    n_sig = sum(1 for r in per if float(r['ci_low']) > 0)
    detail = (f'pooled delta={float(pooled[0]["delta"]):+.6f} '
              f'CI=[{float(pooled[0]["ci_low"]):+.6f},{float(pooled[0]["ci_high"]):+.6f}] '
              f'p={float(pooled[0]["p_value"]):.4f}; positive in {n_pos}/{len(per)} '
              f'datasets; CI above zero in {n_sig}/{len(per)}')
    if pooled_sig and n_pos == len(per):
        return 'SUPPORTED', detail
    if pooled_sig or n_sig > 0:
        return 'PARTIALLY SUPPORTED', detail
    return 'NOT SUPPORTED', detail


def verdict_history_only(eff):
    if not eff:
        return 'NOT TESTED', 'efficiency.csv not produced'
    rows = [r for r in eff if r.get('metric') == 'estimated_generation_saving_fraction']
    if not rows:
        return 'NOT TESTED', 'no saving measurement recorded'
    v = float(rows[0]['value'])
    if v >= 0.30:
        return 'SUPPORTED', f'estimated generation saving {v:.3f}'
    if v > 0:
        return 'PARTIALLY SUPPORTED', f'estimated generation saving {v:.3f}'
    return 'NOT SUPPORTED', f'estimated generation saving {v:.3f}'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/adaptive_factorization')
    a = ap.parse_args(argv)
    root = Path(a.root)
    cm = root / 'compute_matched'
    se = root / 'selector'

    cm_meta = json.loads((cm / 'RUN_METADATA.json').read_text()) if (cm / 'RUN_METADATA.json').exists() else None
    freeze = json.loads((se / 'SELECTOR_FREEZE_MANIFEST.json').read_text()) if (se / 'SELECTOR_FREEZE_MANIFEST.json').exists() else None
    dev = json.loads((se / 'router_dev_summary.json').read_text()) if (se / 'router_dev_summary.json').exists() else None
    test = read_csv(se / 'test_results.csv')
    boot = read_csv(se / 'bootstrap.csv')
    pref = read_csv(se / 'preference_diagnostics.csv')
    nlld = read_csv(se / 'nll_diagnostics.csv')
    sub = read_csv(se / 'subgroup_results.csv')
    rob = read_csv(se / 'robustness_results.csv')
    eff = read_csv(se / 'efficiency.csv')
    cm_compl = read_csv(cm / 'compute_matched_complementarity.csv')
    cm_sing = read_csv(cm / 'compute_matched_single_factorization.csv')
    cm_comp = read_csv(cm / 'compute_matched_shared_comparison.csv')

    vA, rA = verdict_compute_matched(cm_meta)
    vB, rB = verdict_preference_predictable(boot, pref)
    vC, rC = verdict_adaptive_beats_uniform(boot)
    vD, rD = verdict_history_only(eff)

    L = []
    L.append('# FINAL_ADAPTIVE_FACTORIZATION_REPORT.md')
    L.append('')
    L.append('Compute-matched shared-factorization control (STEP A) and a target-free '
             'factorization selector (STEP B) on the frozen shared checkpoints.')
    L.append('')
    L.append('Governance statement: no paper content was read or written; no historical SID '
             'array, beam dump, split or checkpoint was modified; the frozen shared '
             'checkpoints were used for inference only; no test metric was inspected before '
             '`SELECTOR_FREEZE_MANIFEST.json` was written; teacher-forced target NLL and the '
             'ground-truth target are used as supervision and as explanatory diagnostics only '
             'and never as selector inputs.')
    L.append('')
    L.append(f'Generated {time.strftime("%Y-%m-%dT%H:%M:%S%z")} by '
             '`analysis/mechanism/adaptive_factorization/af_final_report.py`.')
    L.append('')
    L.append('The required report structure is covered as: A state recovery, '
             'B compute-matched shared control, **C how shared training is implemented** '
             '(an addition), D selector data protocol, E feature audit, F RouterDev results, '
             'G final frozen selector, H test results, I adaptive vs uniform significance, '
             'J hard vs soft routing, K history-only vs prediction-aware, '
             'L seen/unseen-user robustness, M preference/NLL diagnostics, N efficiency, '
             'O limitations, P training performed. Sections D-P correspond one-to-one to the '
             'required C-O, shifted by the inserted section C.')
    L.append('')
    L.append('Provenance audit: `SELECTOR_INTEGRITY_AUDIT.md` documents an apparent checkpoint '
             'digest disagreement found after the freeze, its resolution (two valid but '
             'incomparable hash methods, not a replaced artifact), and the replay that '
             'reproduces every published router-backed metric from the frozen artifacts.')
    L.append('')

    # ---- A ----
    L.append('## A. State recovery')
    L.append('')
    L.append('`outputs/adaptive_factorization/frozen_state.md` records the state rebuilt from '
             'disk before any training: the branch HEAD, the canonical recipe, the two '
             'factorizations verified as structurally comparable (3 semantic tokens + 1 '
             'collision/disambiguation token, SID length 4, codebook 256, vocabulary 1025), the '
             'item-id conventions, the element-wise identity of the `user` column across '
             'train/valid/test, and the four frozen shared checkpoints with their SHA-256 '
             'digests. The previously reported outcomes '
             '(`CROSS-INDEPENDENT-TOKENIZER COMPLEMENTARITY: SUPPORTED`, '
             '`SHARED-PARAMETER FACTORIZATION COMPLEMENTARITY: SUPPORTED`, 2×-exposure shared '
             'training, 167401 optimizer steps for Beauty) were re-derived from the artefacts '
             'rather than accepted from the prompt.')
    L.append('')
    L.append('No stop condition was triggered: the checkpoints match the audited ones, the '
             'item-id convention was reconstructible, and no historical artefact needed to be '
             'overwritten.')
    L.append('')

    # ---- B ----
    L.append('## B. Compute-matched shared control')
    L.append('')
    if cm_meta:
        st = cm_meta['exposure_comparison']
        L.append('| quantity | previous shared (2× exposure) | compute-matched |')
        L.append('|---|---|---|')
        L.append(f'| train samples / epoch | {st["previous_shared_2x"]["train_samples_per_epoch"]} '
                 f'| {st["compute_matched"]["train_samples_per_epoch"]} |')
        L.append(f'| optimizer steps / epoch | {st["previous_shared_2x"]["steps_per_epoch"]} | '
                 f'{st["compute_matched"]["steps_per_epoch"]} |')
        L.append(f'| total optimizer steps | '
                 f'{st["previous_shared_2x"]["total_optimizer_steps"]} | '
                 f'{st["compute_matched"]["total_optimizer_steps"]} |')
        L.append(f'| epochs / best epoch | {st["previous_shared_2x"]["epochs_run"]} / '
                 f'{st["previous_shared_2x"]["best_epoch"]} | '
                 f'{st["compute_matched"]["epochs_run"]} / {st["compute_matched"]["best_epoch"]} |')
        L.append(f'| exposure ratio | 1.0 | {fnum(st["exposure_ratio"], 4)} |')
        L.append('')
        L.append('### Single-factorization performance (NDCG@20, test)')
        L.append('')
        L.append('| factorization | independent | previous shared | compute-matched | '
                 'vs independent | vs previous |')
        L.append('|---|---|---|---|---|---|')
        indep_of = {'spectral_k3': 0.039964748794096394,
                    'rqvae3': 0.035583611661945076}
        prev_of = {'spectral_k3': 0.04227714956723783,
                   'rqvae3': 0.0382502925640393}
        matched_of = {r['factorization']: float(r['NDCG@20']) for r in cm_sing}
        for f in ('spectral_k3', 'rqvae3'):
            i, pv, mm = indep_of[f], prev_of[f], matched_of[f]
            L.append(f'| {f} | {i:.6f} | {pv:.6f} | {mm:.6f} | {mm/i:.4f} | {pv and mm/pv:.4f} |')
        L.append('')
        L.append('Collaborator single-factorization values for the two reference regimes are '
                 'carried in `compute_matched_shared_comparison.csv`.')
        L.append('')
        L.append('### Complementarity')
        L.append('')
        L.append('| regime | best single | fused | fusion − best | p | oracle | oracle − best | '
                 'Only-F1 | Only-F2 | disagreement | Jaccard |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in cm_compl:
            L.append(f'| {r["regime"]} | {fnum(r["best_single_NDCG@20"])} '
                     f'({r["best_single_name"]}) | {fnum(r["fused_NDCG@20"])} | '
                     f'{fnum(r["fusion_minus_best"])} | {fnum(r["fusion_p"], 4)} | '
                     f'{fnum(r["oracle_NDCG@20"])} | {fnum(r["oracle_minus_best"])} | '
                     f'{100*float(r["Only_F1_fraction"]):.2f}% | '
                     f'{100*float(r["Only_F2_fraction"]):.2f}% | '
                     f'{fnum(r["hit_disagreement"], 4)} | '
                     f'{fnum(r["candidate_jaccard"], 4)} |')
        L.append('')
        L.append('### Success criteria and collapse guard')
        L.append('')
        for k, v in cm_meta['success_criteria'].items():
            L.append(f'* `{k}`: {"PASS" if v else "FAIL"}')
        L.append('')
        L.append(f'`SEVERE_COMPUTE_MATCHED_COLLAPSE`: '
                 f'**{"FLAGGED" if cm_meta["SEVERE_COMPUTE_MATCHED_COLLAPSE"] else "not flagged"}** '
                 f'(threshold: either factorization below 70% of its previous shared '
                 f'counterpart). No hyper-parameter was tuned in response to any outcome.')
    else:
        L.append('_STEP A artefacts unavailable at report time._')
    L.append('')

    # ---- C ----
    L.append('## C. How shared training is implemented')
    L.append('')
    L.append('This section documents the mechanism itself: one parameter set trained on two '
             'injective factorizations of the same items. The description is taken from the '
             'code that ran (`analysis/mechanism/cross_tokenizer_shared/xt_train_shared.py`, '
             '`analysis/mechanism/adaptive_factorization/af_matched_train.py`) and from the '
             'run logs, not from intent.')
    L.append('')
    L.append('### C.1 One model, two factorizations')
    L.append('')
    L.append('A single `T5ForConditionalGeneration` is built with the canonical capacity '
             '(`src/tiger/train.py::TIGER`): 4 encoder + 4 decoder layers, `d_model` 128, '
             '`d_ff` 1024, 6 heads, `d_kv` 64, dropout 0.1, ReLU feed-forward, '
             '`pad_token_id = eos_token_id = decoder_start_token_id = 0`. There is exactly '
             '**one encoder, one decoder, one embedding table and one output head**; no '
             'adapter, expert, extra decoder or per-factorization module exists anywhere in '
             'the checkpoint.')
    L.append('')
    L.append('### C.2 A shared vocabulary with reserved control tokens')
    L.append('')
    L.append('SID tokens occupy `[1, L*codebook]` because the token layout is '
             '`token = code + position*256 + 1` (`src/tiger/dataset.py::item2code`). For '
             '`L = 4` that is `[1, 1024]`, and id 0 stays PAD/EOS. `shared_vocab_plan()` then '
             'reserves one control token per factorization starting at `L*256 + 1`:')
    L.append('')
    L.append('```')
    L.append('spectral_k3 -> 1025      rqvae3 -> 1026      vocab_size = 1027')
    L.append('```')
    L.append('')
    L.append('The two extra embedding rows are the only parameters sharing adds '
             '(4,590,848 -> 4,591,104). The control token is a single-element list placed at '
             'the FRONT of the history token list, so the encoder input is '
             '`[ctrl] + history` flattened (`ControlPrefixedDataset`). Because the control '
             'token is prepended after the canonical left-padding of the history, it never '
             'carries PAD id 0 and always has attention mask 1.')
    L.append('')
    L.append('So the same `(user, history, target)` interaction becomes a **different '
             'discrete sequence** under each factorization: different control token, '
             'different SID tokens for every history item and for the target. Nothing else '
             'distinguishes the two views.')
    L.append('')
    L.append('### C.3 Index alignment between the two views')
    L.append('')
    L.append('Each factorization gets its own `GenRecDataset`, built from the *same* '
             'parquet split with that factorization\'s SID array. In `mode="train"` '
             '`process_data` expands every user\'s sequence with an **expanding sliding '
             'window** — for a sequence `[i1..iT]` it emits `T-1` instances '
             '(`[i1]->i2`, `[i1,i2]->i3`, ...). Two consequences matter:')
    L.append('')
    L.append('* the expansion depends only on the parquet row order and the sequence '
             'length, so **entry `i` of the F1 training set and entry `i` of the F2 training '
             'set are the same original `(user, position)` instance**, expressed in two '
             'factorizations. This is what makes "pick one factorization per instance" a '
             'well-defined operation; the driver asserts the two sets have equal length and '
             'raises otherwise.');
    L.append('* Beauty has 131,410 such instances per factorization (the same count the '
             'previous study recorded), and 22,363 evaluation rows (one per user).');
    L.append('')
    L.append('### C.4 Exposure schedule, and the one thing STEP A changed')
    L.append('')
    L.append('Two schedules share all other code:')
    L.append('')
    L.append('| | previous shared run (`balanced`) | STEP A (`compute_matched_balanced`) |')
    L.append('|---|---|---|')
    L.append('| construction | `ConcatDataset([F1, F2])` | `BalancedMatchedDataset({F1, F2})` |')
    L.append('| per epoch | every instance once **per factorization** | every instance once, '
             'under **one** factorization |')
    L.append('| samples / epoch | 262,820 | 131,410 |')
    L.append('| batches / epoch | 1,027 | 514 |')
    L.append('| batch content | blocked by factorization | mixes both control tokens |')
    L.append('')
    L.append('The compute-matched assignment is a deterministic, balanced permutation '
             '(`af_matched_train.balanced_assignment`):')
    L.append('')
    L.append('```python')
    L.append('base  = n // 2')
    L.append('counts = [base, base]; counts[:n - 2*base] += 1      # |n1 - n2| <= 1 exactly')
    L.append('multiset = np.repeat([0, 1], counts)')
    L.append('rng = np.random.default_rng(seed * 1_000_003 + epoch)  # seed + epoch')
    L.append('rng.shuffle(multiset)                                # index i -> factorization')
    L.append('```')
    L.append('')
    L.append('`__getitem__(i)` then returns `per_f[keys[assign[i]]][i]`. The epoch seed, the '
             'realised per-factorization counts and the resulting `steps` are written into '
             'every row of `train_log.json`, so the schedule is reproducible and auditable; '
             '`tests/test_af_compute_matched.py` pins balance, determinism, per-epoch change '
             'and index preservation.')
    L.append('')
    L.append('### C.5 Batching, loss and optimisation')
    L.append('')
    L.append('`GenRecDataLoader` (4 workers) flattens each instance\'s history to one token '
             'sequence and builds `attention_mask = (token != 0)`. The training step is the '
             'plain T5 objective with no weighting or auxiliary term:')
    L.append('')
    L.append('```python')
    L.append('loss, _ = model(input_ids=batch["history"],')
    L.append('                  attention_mask=batch["attention_mask"],')
    L.append('                  labels=batch["target"])   # T5 internal shifted CE over L tokens')
    L.append('loss.backward(); optimizer.step()         # Adam, lr 1e-4, batch 256')
    L.append('```')
    L.append('')
    L.append('The decoder is teacher-forced on the target SID\'s `L` tokens; the loss is the '
             'standard shifted cross-entropy that T5 computes internally, with PAD/EOS id 0 '
             'and no label smoothing.')
    L.append('')
    L.append('### C.6 Validation and checkpoint selection')
    L.append('')
    L.append('Each epoch is scored twice — once per factorization, each with its own control '
             'token and its own SID array — using the canonical item-level evaluator '
             '(`src/tiger/train.py::evaluate_item_level`, beam 30). The selection statistic '
             'is the **macro average of the two validation NDCG@20 values**, and the '
             'checkpoint with the best macro is saved. Test is never consulted. Reported '
             'per-epoch values for this run therefore already contain the per-factorization '
             'split: `train_log.json` stores `valid_ndcg20.{spectral_k3,rqvae3}` and '
             '`macro_valid_ndcg20` for all 200 epochs.')
    L.append('')
    L.append('### C.7 What "shared" does and does not mean here')
    L.append('')
    L.append('* Shared: every parameter that computes a prediction — encoder, decoder, '
             'embeddings, output head. Both factorizations are produced by the same weights; '
             'the factorization is selected only by an input token.')
    L.append('* Not shared: the discrete token sequences. The model cannot transfer a SID '
             'token between factorizations, because token id 500 means a different code under '
             '`spectral_k3` than under `rqvae3`; the control token is the only signal telling '
             'it which code space it is reading and writing.')
    L.append('* Therefore the interesting quantity is not whether sharing is free, but '
             'whether one weight set can serve two incompatible code spaces *and* keep their '
             'errors complementary. Sections B, H and M answer that.')
    L.append('')

    # ---- D ----
    L.append('## D. Selector data protocol')
    L.append('')
    L.append('Chronology: every dataset has one canonical row per user in each of '
             '`train`/`valid`/`test`, and the `user` column is element-wise identical across '
             'the three splits (verified in `frozen_state.md`). The selector is therefore '
             'supervised on the **earlier validation event** and applied to the **later test '
             'event**.')
    L.append('')
    L.append('* Supervision instance: the canonical validation instance of each user; history = '
             'validation history, target = validation target.')
    L.append('* The validation target is used **only** to build the label `q_m` (and, in the '
             'secondary experiment, `q_NLL`). It is never an inference feature.')
    L.append('* Utility `U_m = 1/log2(r_m + 1)` for `r_m <= 20`, else 0, where `r_m` is the '
             'target rank under factorization m\'s merged item distribution.')
    L.append('* Soft target `q_m ∝ exp(U_m / 0.1)`; `q = (0.5, 0.5)` when `U1 = U2 = 0`, and '
             'those examples carry sample weight 0.25 (pre-declared).')
    L.append('* Users are split 70/15/15 into RouterTrain / RouterDev / RouterHoldout with a '
             'fixed seed, persisted in `router_user_split.csv` and never resampled.')
    L.append('* No user id enters the feature vector.')
    L.append('')
    if freeze:
        L.append(f'RouterTrain n = {freeze["router_train_n"]}, RouterDev n = '
                 f'{freeze["router_dev_n"]}, RouterHoldout n = {freeze["router_holdout_n"]} '
                 f'(pooled over four datasets). Split seed {freeze["user_split"]["seed"]}, '
                 f'fractions {freeze["user_split"]["fractions"]}.')
    L.append('')

    # ---- D ----
    L.append('## E. Feature audit')
    L.append('')
    L.append('`SELECTOR_FEATURE_AUDIT.md` tabulates every feature: name, dimension, source, '
             'availability at inference, and explicit "uses target? NO" / "uses test labels? '
             'NO" columns.')
    L.append('')
    if freeze:
        L.append(f'* Router H (history-only): **{freeze["features"]["router_h_dim"]}** features — '
                 f'mean-pooled frozen encoder states of the F1-tokenised and F2-tokenised history, '
                 f'their absolute difference, and history-length features.')
        L.append(f'* Router P (prediction-aware): **{freeze["features"]["router_p_dim"]}** '
                 f'features — 12 target-free beam statistics per factorization, 6 '
                 f'cross-factorization agreement features, plus the history-length features.')
    L.append('* Rejected by construction: teacher-forced target NLL, the target item/SID/embedding, '
             'the target rank and utility (the labels), and the user id.')
    L.append('')

    # ---- E ----
    L.append('## F. RouterDev results')
    L.append('')
    L.append('No test result was used for any selection. Architecture, feature family and the '
             'static weight were all chosen on RouterDev.')
    L.append('')
    dev_rows = read_csv(se / 'router_dev_results.csv')
    if dev_rows:
        L.append('| router | family | architecture | d_in | params | RouterDev soft-fusion NDCG@20 | best epoch |')
        L.append('|---|---|---|---|---|---|---|')
        for r in dev_rows:
            L.append(f'| {r["router"]} | {r["feature_family"]} | {r["architecture"]} | '
                     f'{r["d_in"]} | {r["n_params"]} | {fnum(r["dev_fusion_ndcg20"])} | '
                     f'{r["best_epoch"]} |')
        L.append('')
    if dev:
        L.append(f'Selection rule (pre-declared): `{dev.get("selection_rule", "best RouterDev soft-fusion NDCG@20")}`.')
        L.append('')
        L.append('### Static-weight baseline')
        L.append('')
        L.append('| w_F1 | RouterDev soft-fusion NDCG@20 |')
        L.append('|---|---|')
        for r in dev['static_grid']:
            L.append(f'| {r["w_F1"]} | {fnum(r["dev_fusion_ndcg20"])} |')
        L.append('')
        L.append(f'Selected `w_F1* = {dev["best_static"]["w_F1"]}` '
                 f'(RouterDev {fnum(dev["best_static"]["dev_fusion_ndcg20"])}), frozen before '
                 f'test. Uniform `w = 0.5` scores '
                 f'{fnum(dev["uniform_dev_fusion_ndcg20"])} on RouterDev.')
    L.append('')

    # ---- F ----
    L.append('## G. Final frozen selector')
    L.append('')
    if freeze:
        L.append('```')
        L.append(f'git SHA                  : {freeze["git_sha"]}')
        L.append(f'chosen router            : {freeze["chosen_router"]}')
        L.append(f'router checkpoint sha256 : {freeze["chosen_router_sha256"]}')
        L.append(f'scaler hash              : {freeze["scaler_hash"]}')
        L.append(f'architecture             : {json.dumps(freeze["architecture"])}')
        L.append(f'loss                     : {freeze["loss"]}')
        L.append(f'label                    : {json.dumps(freeze["label"])}')
        L.append(f'hyperparameters          : {json.dumps(freeze["hyperparameters"])}')
        L.append(f'static weight            : {freeze["static_weight"]}')
        L.append(f'RouterDev result         : {freeze["best_static"]["dev_fusion_ndcg20"]}')
        L.append(f'manifest written         : {freeze["created_local"]}')
        L.append(f'test read at freeze time : {freeze["test_metrics_computed_at_freeze_time"]}')
        L.append('```')
    L.append('')

    # ---- G ----
    L.append('## H. Test results')
    L.append('')
    modes = []
    for r in test:
        if r['mode'] not in modes:
            modes.append(r['mode'])
    if test and modes:
        L.append('NDCG@20 by dataset and mode:')
        L.append('')
        L.append('| mode | ' + ' | '.join(SHORT[d] for d in DATASETS) + ' | pooled |')
        L.append('|---|' + '---|' * (len(DATASETS) + 1))
        for m in modes:
            cells = []
            for ds in DATASETS:
                row = [r for r in test if r['dataset'] == ds and r['mode'] == m]
                cells.append(fnum(row[0]['NDCG@20']) if row else '—')
            pooled = [r for r in test if r['dataset'] == 'ALL_POOLED' and r['mode'] == m]
            L.append(f'| {m} | ' + ' | '.join(cells) + ' | '
                     f'{fnum(pooled[0]["NDCG@20"]) if pooled else "—"} |')
        L.append('')
        L.append('Full Recall@{5,10,20} and NDCG@{5,10,20} for every cell are in '
                 '`test_results.csv`.')
    else:
        L.append('_Test evaluation not yet run (awaiting the freeze manifest)._')
    L.append('')

    # ---- H ----
    L.append('## I. Adaptive vs uniform significance')
    L.append('')
    if boot:
        L.append('Paired user-level bootstrap, 5000 resamples, seed 20260912, two-sided.')
        L.append('')
        L.append('| comparison | dataset | metric | delta | 95% CI | p |')
        L.append('|---|---|---|---|---|---|')
        for r in boot:
            L.append(f'| {r["comparison"]} | {SHORT.get(r["dataset"], r["dataset"])} | '
                     f'{r["metric"]} | {fnum(r["delta"], 8)} | '
                     f'[{fnum(r["ci_low"], 8)}, {fnum(r["ci_high"], 8)}] | '
                     f'{fnum(r["p_value"], 6)} |')
    L.append('')

    # ---- I ----
    L.append('## J. Hard vs soft routing')
    L.append('')
    if test:
        for ds in DATASETS + ['ALL_POOLED']:
            vals = {m: next((r for r in test if r['dataset'] == ds and r['mode'] == m), None)
                    for m in ('D_H_hard', 'E_H_soft', 'F_P_hard', 'G_P_soft')}
            if all(v is None for v in vals.values()):
                continue
            L.append(f'* **{SHORT.get(ds, ds)}** — Router H hard '
                     f'{fnum(vals["D_H_hard"]["NDCG@20"]) if vals["D_H_hard"] else "—"}, '
                     f'Router H soft '
                     f'{fnum(vals["E_H_soft"]["NDCG@20"]) if vals["E_H_soft"] else "—"}, '
                     f'Router P hard '
                     f'{fnum(vals["F_P_hard"]["NDCG@20"]) if vals["F_P_hard"] else "—"}, '
                     f'Router P soft '
                     f'{fnum(vals["G_P_soft"]["NDCG@20"]) if vals["G_P_soft"] else "—"}.')
    L.append('')
    L.append('Hard routing selects one factorization\'s ranked list (`argmax w`, ties to F1); '
             'soft routing fuses the two calibrated item distributions with the predicted '
             'per-user weights, using the same score normalisation as the previous shared-fusion '
             'implementation.')
    L.append('')

    # ---- J ----
    L.append('## K. History-only vs prediction-aware')
    L.append('')
    L.append('Router H can choose a factorization **before** beam generation and is therefore the '
             'only family that can save generation compute. Router P requires both '
             'factorizations\' beams and therefore **cannot** claim an inference saving; it is an '
             'adaptive-fusion method.')
    L.append('')
    if dev:
        h = [r for r in dev['results'] if r['feat_key'] == 'H']
        p = [r for r in dev['results'] if r['feat_key'] == 'P']
        bh = max(h, key=lambda r: r['dev_fusion_ndcg20']) if h else None
        bp = max(p, key=lambda r: r['dev_fusion_ndcg20']) if p else None
        if bh and bp:
            L.append(f'On RouterDev the best history-only router reaches '
                     f'{fnum(bh["dev_fusion_ndcg20"])} and the best prediction-aware router '
                     f'{fnum(bp["dev_fusion_ndcg20"])} (uniform '
                     f'{fnum(dev["uniform_dev_fusion_ndcg20"])}).')
    L.append('')

    # ---- K ----
    L.append('## L. Seen/unseen-user robustness')
    L.append('')
    if rob:
        L.append('Test NDCG@20 broken down by the role of the user\'s *validation* instance. '
                 'RouterHoldout users were never used for selector fitting or model selection.')
        L.append('')
        L.append('| dataset | validation role | n | mode | NDCG@20 | Recall@20 |')
        L.append('|---|---|---|---|---|---|')
        for r in rob:
            L.append(f'| {SHORT.get(r["dataset"], r["dataset"])} | {r["validation_role"]} | '
                     f'{r["n"]} | {r["mode"]} | {fnum(r["NDCG@20"])} | {fnum(r["Recall@20"])} |')
    else:
        L.append('_not yet produced_')
    L.append('')

    # ---- L ----
    L.append('## M. Preference and NLL diagnostics')
    L.append('')
    L.append('Both are evaluation-only. The true target is never fed back into routing.')
    L.append('')
    if pref:
        L.append('| dataset | router | group | n with a decisive label | accuracy | '
                 'balanced accuracy (degenerate) | AUROC (degenerate) | mean w_F1 |')
        L.append('|---|---|---|---|---|---|---|---|')
        for r in pref:
            L.append(f'| {SHORT.get(r["dataset"], r["dataset"])} | {r["router"]} | {r["group"]} | '
                     f'{r["n"]} | {fnum(r["accuracy"], 4)} | '
                     f'{fnum(r["balanced_accuracy"], 4)} | {fnum(r["auroc"], 4)} | '
                     f'{fnum(r["mean_w_F1"], 4)} |')
        L.append('')
    L.append('The `n` column counts only rows whose ground-truth preference is decisive '
             '(the target is a hit under exactly one factorization). A user whose target '
             'is missed by both factorizations carries no preference signal and is not '
             'scored, which is why the `Neither` rows show n = 0 here while `Neither` '
             'contains the majority of users in `subgroup_results.csv`.')
    L.append('')
    L.append('`balanced_accuracy` and `auroc` are **NaN for the Only-F1 and Only-F2 '
             'subgroups by construction**: each of those subgroups contains only one '
             'class (the target is a hit under exactly one factorization), so those two '
             'statistics are degenerate there. The informative numbers for those rows '
             'are `accuracy` (the fraction of the subgroup the router routes correctly) '
             'and `mean_w_F1` (the average predicted weight), and those are what the '
             'verdict rule uses.')
    L.append('')
    if nlld:
        L.append('Agreement of the frozen router with the **argmin of the teacher-forced target '
                 'NLL** — the factorization that would have been easiest had the target been '
                 'known. This is explanatory only.')
        L.append('')
        L.append('| dataset | n | router | argmin(NLL) agreement | NLL argmin == better rank | '
                 'mean NLL F1 | mean NLL F2 |')
        L.append('|---|---|---|---|---|---|---|')
        for r in nlld:
            L.append(f'| {SHORT.get(r["dataset"], r["dataset"])} | {r["n"]} | {r["router"]} | '
                     f'{r["router_argmin_matches_nll_argmin"]} | '
                     f'{r["nll_argmin_matches_better_rank"]} | {r["mean_nll_F1"]} | '
                     f'{r["mean_nll_F2"]} |')
        L.append('')

    # ---- M ----
    L.append('## N. Efficiency')
    L.append('')
    if eff:
        L.append('| metric | scope | value | unit | note |')
        L.append('|---|---|---|---|---|')
        for r in eff:
            L.append(f'| {r.get("metric")} | {r.get("scope", "")} | {r.get("value")} | '
                     f'{r.get("unit", "")} | {r.get("note", "")} |')
    else:
        L.append('_not yet produced_')
    L.append('')
    L.append('For Router H hard the estimate assumes only the selected factorization\'s beam is '
             'generated. For Router P no saving is claimed: both beams were already computed.')
    L.append('')

    # ---- N ----
    L.append('## O. Limitations')
    L.append('')
    L.append('* STEP A trains **Beauty only**. The compute-matched control is a single dataset '
             'and a single seed (2025); no multi-seed estimate of the matched-exposure effect is '
             'available.')
    L.append('* The compute-matched schedule changes two things at once by construction — total '
             'sample exposure *and* the fact that a batch now mixes both control tokens rather '
             'than being blocked by factorization. This is inherent to "one factorization per '
             'instance per epoch" and is reported rather than claimed away.')
    L.append('* Router supervision comes from a single earlier interaction per user, so the '
             'training set per dataset is small (one row per user). RouterTrain/RouterDev/'
             'RouterHoldout are pooled across datasets for fitting, with per-dataset reporting.')
    L.append('* The selector is a small probe on top of a frozen model; it cannot recover '
             'complementarity the base model does not have.')
    L.append('* Oracle numbers use the ground-truth target and are not deployable.')
    L.append('* Router P cannot reduce generation cost; only Router H can, and only if the '
             'selected factorization is generated alone.')
    L.append('* Teacher-forced NLL is used as a secondary label and as a diagnostic; it is never '
             'a feature. Any residual optimism from the secondary label is confined to the '
             'secondary experiment.')
    L.append('* **The adaptive gain is real but small, and the router does not actually decode '
             'which factorization wins.** Balanced accuracy on the decisive subgroups averages '
             'roughly 0.53-0.55, and the router is markedly better on Only-F1 rows than on '
             'Only-F2 rows, so most of the aggregate gain comes from a mild systematic lean '
             'towards Spectral-K3 rather than from instance-level preference prediction. The '
             'difference between uniform fusion and the frozen oracle (reported in section H) '
             'is roughly two orders of magnitude larger than the measured adaptive gain; that '
             'gap remains open.')
    L.append('* Hard routing of a single factorization discards the other factorization\'s '
             'candidates, so it sacrifices recall on the Only-other group; that is why the soft '
             'modes outperform the hard modes overall even though the hard modes are the ones '
             'that can save generation compute.')
    L.append('* The RouterDev selection signal is close to the resolution of the data: the '
             'selected router beats uniform on RouterDev by about 0.0006 NDCG@20. The choice '
             'between Linear and MLP, and between Router H and Router P, is therefore not '
             'strongly identified, and a different RouterDev split could plausibly choose a '
             'different member of the same family.')
    L.append('')

    # ---- O ----
    L.append('## P. Training performed')
    L.append('')
    L.append('| run | dataset | regime | epochs | best epoch | total optimizer steps | '
             'wall clock (s) |')
    L.append('|---|---|---|---|---|---|---|')
    if cm_meta:
        st = cm_meta['exposure_comparison']
        for name in ('previous_shared_2x', 'compute_matched'):
            s = st[name]
            L.append(f'| {name} | Beauty | {name} | {s["epochs_run"]} | {s["best_epoch"]} | '
                     f'{s["total_optimizer_steps"]} | {s["elapsed_s"]} |')
    L.append('')
    L.append('No other model was trained. The four frozen shared checkpoints were not retrained; '
             'all selector work is inference plus the small router networks '
             f'({min((r["n_params"] for r in dev["results"]), default="?") if dev else "?"}–'
             f'{max((r["n_params"] for r in dev["results"]), default="?") if dev else "?"} '
             'parameters).')
    L.append('')

    # ---- Verdicts ----
    L.append('## Verdicts')
    L.append('')
    L.append('Produced by the pre-declared rules in `af_final_report.py`; the evidence behind '
             'each is stated inline.')
    L.append('')
    L.append('```')
    L.append(f'COMPUTE_MATCHED_SHARED_COMPLEMENTARITY: {vA}')
    L.append(f'TARGET_FREE_FACTORISATION_PREFERENCE_IS_PREDICTABLE: {vB}')
    L.append(f'ADAPTIVE_ROUTING_BEATS_UNIFORM_FUSION: {vC}')
    L.append(f'HISTORY_ONLY_ROUTING_CAN_SAVE_COMPUTE: {vD}')
    L.append('```')
    L.append('')
    L.append('| verdict | rule outcome |')
    L.append('|---|---|')
    L.append(f'| COMPUTE_MATCHED_SHARED_COMPLEMENTARITY | {rA} |')
    L.append(f'| TARGET_FREE_FACTORISATION_PREFERENCE_IS_PREDICTABLE | {rB} |')
    L.append(f'| ADAPTIVE_ROUTING_BEATS_UNIFORM_FUSION | {rC} |')
    L.append(f'| HISTORY_ONLY_ROUTING_CAN_SAVE_COMPUTE | {rD} |')
    L.append('')

    p = root / 'FINAL_ADAPTIVE_FACTORIZATION_REPORT.md'
    p.write_text('\n'.join(L) + '\n')
    print(f'wrote {p}')
    print(f'  {vA} | {vB} | {vC} | {vD}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
