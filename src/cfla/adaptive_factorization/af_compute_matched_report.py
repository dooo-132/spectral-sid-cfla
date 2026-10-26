#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate COMPUTE_MATCHED_REPORT.md for STEP A from the artefacts on disk.

Run:
  python3 af_compute_matched_report.py --outdir outputs/adaptive_factorization/compute_matched
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def read_csv(p):
    with Path(p).open() as fh:
        return list(csv.DictReader(fh))


def write_config_yaml(out, meta):
    """Emit `config.yaml` describing the exposure-matched run (spec section 45)."""
    # the run directory records which train_log.json it used, so the report never has
    # to guess a timestamped run_id
    ref = Path(out) / 'run_log_path.txt'
    log_p = Path(ref.read_text().strip()) if ref.exists() else None
    if log_p is None or not log_p.exists():
        raise SystemExit('run_log_path.txt missing; run af_compute_matched_eval.py first')
    log = json.loads(log_p.read_text())
    st = meta['exposure_comparison']
    lines = [
        '# Compute-matched shared-factorization control (STEP A)',
        '# Only the factorization exposure schedule differs from the previous shared run.',
        'dataset: Beauty',
        'factorizations: [spectral_k3, rqvae3]',
        f'exposure: {log["exposure"]}',
        'assignment:',
        f'  scheme: {log["assignment"]["scheme"]}',
        f'  rng: {log["assignment"]["rng"]}',
        f'  seed_formula: {log["assignment"]["seed_formula"]}',
        f'  base_seed: {log["assignment"]["base_seed"]}',
        f'  stride: {log["assignment"]["stride"]}',
        f'  balance_guarantee: "{log["assignment"]["balance_guarantee"]}"',
        f'  index_alignment: "{log["assignment"]["index_alignment"]}"',
        'training:',
        f'  num_epochs: {log["config"]["num_epochs"]}',
        f'  batch_size: {log["config"]["batch_size"]}',
        f'  infer_size: {log["config"]["infer_size"]}',
        f'  lr: {log["config"]["lr"]}',
        f'  early_stop: {log["config"]["early_stop"]}',
        f'  beam_size: {log["config"]["beam_size"]}',
        f'  max_len: {log["config"]["max_len"]}',
        f'  seed: {log["config"]["seed"]}',
        'canonical_capacity:',
        '  num_layers: 4',
        '  num_decoder_layers: 4',
        '  d_model: 128',
        '  d_ff: 1024',
        '  num_heads: 6',
        '  d_kv: 64',
        '  dropout_rate: 0.1',
        f'vocab_size: {log["vocab_size"]}',
        f'control_tokens: {json.dumps(log["plan"]["control_tokens"])}',
        'selection: >-',
        '  macro-average item-level validation NDCG@20 across factorizations',
        'test_used_for_selection: false',
        'exposure_accounting:',
        f'  train_samples_per_epoch: {st["compute_matched"]["train_samples_per_epoch"]}',
        f'  canonical_single_model_train_samples: '
        f'{log["canonical_single_model_train_size"]}',
        f'  optimizer_steps_per_epoch: {st["compute_matched"]["steps_per_epoch"]}',
        f'  total_optimizer_steps: {st["compute_matched"]["total_optimizer_steps"]}',
        f'  previous_shared_total_optimizer_steps: '
        f'{st["previous_shared_2x"]["total_optimizer_steps"]}',
        f'  exposure_ratio_vs_previous_shared: {st["exposure_ratio"]:.6f}',
        f'collapse_guard_threshold: 0.70',
        f'SEVERE_COMPUTE_MATCHED_COLLAPSE: '
        f'{str(meta["SEVERE_COMPUTE_MATCHED_COLLAPSE"]).lower()}',
        f'verdict: {meta["verdict"]}',
    ]
    (Path(out) / 'config.yaml').write_text('\n'.join(lines) + '\n')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args(argv)
    out = Path(a.outdir)

    meta = json.loads((out / 'RUN_METADATA.json').read_text())
    sing = read_csv(out / 'compute_matched_single_factorization.csv')
    compl = read_csv(out / 'compute_matched_complementarity.csv')
    comp = read_csv(out / 'compute_matched_shared_comparison.csv')
    nll_p = out / 'compute_matched_nll_alignment.csv'
    nll = read_csv(nll_p) if nll_p.exists() else None
    steps = meta['exposure_comparison']
    collapse = meta['collapse_guard']
    checks = meta['success_criteria']

    L = []
    L.append('# COMPUTE_MATCHED_REPORT.md')
    L.append('')
    L.append('STEP A of the adaptive-factorization study: does factorization '
             'complementarity survive when the shared model\'s optimizer-step exposure is '
             'matched to a single independent model?')
    L.append('')
    L.append('## 1. What was changed, and what was held fixed')
    L.append('')
    L.append('Exactly one thing changed relative to the previous shared run: the '
             '**factorization exposure schedule**. Architecture, optimizer, learning rate, '
             'batch size, patience, beam size, seed, vocabulary, control tokens and the '
             'validation-selection rule are identical.')
    L.append('')
    L.append('| item | previous shared (`balanced`) | this run (`compute_matched_balanced`) |')
    L.append('|---|---|---|')
    L.append(f'| factorizations per original instance per epoch | both (F1 **and** F2) | '
             f'**exactly one**, F1 or F2 |')
    L.append(f'| training samples per epoch | {steps["previous_shared_2x"]["train_samples_per_epoch"]} '
             f'(2 × 131410) | {steps["compute_matched"]["train_samples_per_epoch"]} '
             f'(= canonical single-model count) |')
    L.append(f'| optimizer steps per epoch | {steps["previous_shared_2x"]["steps_per_epoch"]} | '
             f'{steps["compute_matched"]["steps_per_epoch"]} |')
    L.append(f'| total optimizer steps executed | '
             f'{steps["previous_shared_2x"]["total_optimizer_steps"]} | '
             f'{steps["compute_matched"]["total_optimizer_steps"]} |')
    L.append(f'| epochs run | {steps["previous_shared_2x"]["epochs_run"]} | '
             f'{steps["compute_matched"]["epochs_run"]} |')
    L.append(f'| best epoch | {steps["previous_shared_2x"]["best_epoch"]} | '
             f'{steps["compute_matched"]["best_epoch"]} |')
    L.append(f'| wall clock (s) | {steps["previous_shared_2x"]["elapsed_s"]} | '
             f'{steps["compute_matched"]["elapsed_s"]} |')
    L.append(f'| assignment | deterministic concat (all F1 then all F2) | deterministic '
             f'balanced random permutation, seed = `seed*1000003 + epoch` |')
    L.append('')
    L.append(f'Optimizer-step exposure ratio (compute-matched / previous): '
             f'**{steps["exposure_ratio"]:.4f}**. The compute-matched run therefore performs '
             f'{"approximately" if abs(steps["exposure_ratio"] - 0.5) < 0.15 else "notably fewer"} '
             f'the optimization steps of the confounded shared run and '
             f'approximately the same number as one independent model.')
    L.append('')
    L.append('### Assignment balance')
    L.append('')
    L.append('Every epoch\'s assignment is a permutation of a multiset holding `ceil(n/2)` '
             'copies of F1 and `floor(n/2)` copies of F2, so `|count(F1) − count(F2)| <= 1` '
             'exactly (never uncontrolled Bernoulli). The per-epoch seed and the realised counts '
             'are recorded in every row of `runs/.../train_log.json`.')
    L.append('')
    L.append('## 2. Single-factorization performance')
    L.append('')
    L.append('Test NDCG@20 for the compute-matched checkpoint, against the two reference '
             'regimes.')
    L.append('')
    L.append('| factorization | independent | previous shared (2× exposure) | '
             'compute-matched | vs independent | vs previous shared | severe collapse |')
    L.append('|---|---|---|---|---|---|---|')
    # `AF1`/`AF2` are the generic arm labels; they map to spectral_k3 and rqvae3.
    indep_of = {'spectral_k3': 0.039964748794096394, 'rqvae3': 0.035583611661945076}
    prev_of = {'spectral_k3': 0.04227714956723783, 'rqvae3': 0.0382502925640393}
    matched_of = {r['factorization']: float(r['NDCG@20']) for r in sing}
    for f in ('spectral_k3', 'rqvae3'):
        i, pv, m = indep_of[f], prev_of[f], matched_of[f]
        L.append(f'| {f} | {i:.6f} | {pv:.6f} | {m:.6f} | {m/i:.4f} | {m/pv:.4f} | '
                 f'{"YES" if collapse[f]["severe_collapse"] else "no"} |')
    L.append('')
    L.append('| factorization | matched NDCG@5 | matched NDCG@10 | matched NDCG@20 | '
             'matched Recall@20 |')
    L.append('|---|---|---|---|---|')
    for r in comp:
        if r['regime'] != 'compute_matched':
            continue
        L.append(f'| {r["factorization"]} | {float(r["NDCG@5"]):.6f} | '
                 f'{float(r["NDCG@10"]):.6f} | {float(r["NDCG@20"]):.6f} | '
                 f'{float(r["Recall@20"]):.6f} |')
    L.append('')
    if meta['SEVERE_COMPUTE_MATCHED_COLLAPSE']:
        L.append('> **SEVERE_COMPUTE_MATCHED_COLLAPSE** — at least one factorization falls '
                 'below 70% of its previous shared counterpart. Flagged and reported; no '
                 'hyper-parameter was changed in response.')
    else:
        L.append('**No severe collapse.** Both factorizations stay at or above 70% of their '
                 'previous shared counterparts, and no hyper-parameter was changed.')
    L.append('')
    L.append('## 3. Complementarity under matched compute')
    L.append('')
    L.append('| regime | best single | fused | fusion − best | 95% CI | p | oracle | '
             'oracle − best | Only-F1 | Only-F2 | hit disagreement | candidate Jaccard |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in compl:
        ci = f'[{float(r["fusion_ci_low"]):+.5f}, {float(r["fusion_ci_high"]):+.5f}]'
        L.append(f'| {r["regime"]} | {float(r["best_single_NDCG@20"]):.6f} '
                 f'({r["best_single_name"]}) | {float(r["fused_NDCG@20"]):.6f} | '
                 f'**{float(r["fusion_minus_best"]):+.6f}** | {ci} | '
                 f'{float(r["fusion_p"]):.4f} | {float(r["oracle_NDCG@20"]):.6f} | '
                 f'{float(r["oracle_minus_best"]):+.6f} | '
                 f'{100*float(r["Only_F1_fraction"]):.2f}% | '
                 f'{100*float(r["Only_F2_fraction"]):.2f}% | '
                 f'{float(r["hit_disagreement"]):.4f} | '
                 f'{float(r["candidate_jaccard"]):.4f} |')
    L.append('')
    L.append('## 4. Success criteria')
    L.append('')
    L.append('The question is **not** whether compute-matched sharing gives the highest NDCG. '
             'It is whether factorization complementarity survives.')
    L.append('')
    L.append('| criterion | result |')
    L.append('|---|---|')
    label = {
        'bidirectional_only_groups_exist': 'bidirectional Only-F groups exist',
        'hit_disagreement_gt_0': 'hit disagreement > 0',
        'oracle_gt_best_shared_single': 'oracle > best shared single',
        'fusion_gt_best_shared_single': 'fusion > best shared single',
        'fusion_ci_above_zero': 'fusion gain 95% CI strictly above 0',
        'own_factorization_nll_alignment_far_above_50pct':
            'own-factorization NLL alignment far above 50%',
    }
    for k, v in checks.items():
        L.append(f'| {label.get(k, k)} | {"**PASS**" if v else "FAIL"} |')
    L.append('')
    L.append(f'Criteria satisfied: **{sum(1 for v in checks.values() if v)}/{len(checks)}**.')
    L.append('')
    if nll:
        L.append('### Teacher-forced target-NLL alignment')
        L.append('')
        L.append('Computed with the FROZEN full-exposure shared checkpoint on the test split '
                 '(the compute-matched checkpoint\'s own NLL dump is a separate artefact); this '
                 'column is explanatory only and is never an inference-time signal.')
        L.append('')
        L.append('| Only group | n | mean NLL own | mean NLL other | argmin == own |')
        L.append('|---|---|---|---|---|')
        for r in nll:
            L.append(f'| {r["group"]} | {r["n"]} | {float(r["mean_nll_own"]):.4f} | '
                     f'{float(r["mean_nll_other"]):.4f} | '
                     f'{100*float(r["argmin_equals_own"]):.2f}% |')
        L.append('')
    L.append('## 5. Verdict')
    L.append('')
    L.append('```')
    L.append(f'COMPUTE_MATCHED_SHARED_COMPLEMENTARITY: {meta["verdict"]}')
    if meta['SEVERE_COMPUTE_MATCHED_COLLAPSE']:
        L.append('SEVERE_COMPUTE_MATCHED_COLLAPSE: FLAGGED')
    L.append('```')
    L.append('')
    L.append('## 6. Files')
    L.append('')
    L.append('```')
    for f in sorted(p.name for p in out.iterdir() if p.is_file()):
        L.append(f)
    L.append('```')

    write_config_yaml(out, meta)
    p = out / 'COMPUTE_MATCHED_REPORT.md'
    p.write_text('\n'.join(L) + '\n')
    print(f'wrote {p}')
    print(f'wrote {out / "config.yaml"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
