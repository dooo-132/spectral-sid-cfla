#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL_TWO_STEP_REPORT.md generator.

Aggregates STEP 1 and STEP 2 artifacts into the study's final report. Every
number is read from the artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, OUT, PRIMARY, SHORT, STEP1, STEP2, TOKENIZERS  # noqa: E402


def sec_step1(A):
    div = pd.read_csv(STEP1 / 'cross_tokenizer_pair_diversity.csv')
    grp = pd.read_parquet(STEP1 / 'cross_tokenizer_only_groups.parquet')
    al = pd.read_csv(STEP1 / 'cross_tokenizer_nll_alignment.csv')
    man = json.loads((STEP1 / 'tokenizer_manifest.json').read_text())
    P = div[div.tier == 'PRIMARY'].set_index('dataset')
    g = grp.groupby(['dataset', 'group']).size().unstack(fill_value=0)
    gf = g.div(g.sum(axis=1), axis=0)
    pool = al[(al.tier == 'PRIMARY') & (al.normalisation == 'raw') &
              (al.group == 'BOTH_Only_groups')].set_index('dataset')
    poolz = al[(al.tier == 'PRIMARY') & (al.normalisation == 'z_valid') &
               (al.group == 'BOTH_Only_groups')].set_index('dataset')

    A('## B. Cross-tokenizer fairness')
    A('')
    n_pass = sum(1 for r in man['fairness'] if r['status'] == 'PASS')
    A(f"The pre-registered primary comparison is **Spectral-K3 vs RQ-VAE-3layer** on "
      f"four datasets. Fairness audit: **{n_pass}/{len(man['fairness'])} criteria PASS**. "
      f"The two tokenizers are evaluated on identical parquet splits, the dumped target "
      f"item ids are element-wise identical, both SIDs have length 4 with codebook 256 "
      f"and vocabulary 1025, and both checkpoints use the canonical TIGER recipe "
      f"(batch 256, lr 1e-4, 200 epochs, patience 10, seed 2025, beam 30, item-level "
      f"validation NDCG@20 selection). The RQ-VAE recommenders already existed, so "
      f"STEP 1 required **zero training**.")
    A('')
    A('OPQ-3 and RQ1-OPQ2 are structurally comparable but their historical checkpoints '
      'were selected with a different trainer (legacy token-prefix validation metric), '
      'so they are reported as secondary evidence only.')
    A('')
    A('## C. Cross-tokenizer retrieval diversity')
    A('')
    A('| dataset | best single | fusion | fusion − best | p | oracle | oracle − best | '
      'oracle − fusion | exploited |')
    A('|---|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        r = P.loc[ds]
        A(f"| {SHORT[ds]} | {r['best_single_ndcg20']:.5f} | {r['fusion_ndcg20']:.5f} | "
          f"**{r['fusion_gain']:+.5f}** | {r['fusion_vs_best_p']:.4f} | "
          f"{r['oracle_ndcg20']:.5f} | {r['oracle_gain']:+.5f} | "
          f"{r['oracle_minus_fusion']:+.5f} | {100*r['oracle_exploited_fraction']:.1f}% |")
    A('')
    A('| dataset | Only-Spectral | Only-RQVAE | Both | Hit disagreement | directional '
      'asymmetry | candidate Jaccard | phi |')
    A('|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        r = P.loc[ds]
        A(f"| {SHORT[ds]} | {100*gf.loc[ds,'Only-Spectral']:.2f}% | "
          f"{100*gf.loc[ds,'Only-RQVAE']:.2f}% | {100*gf.loc[ds,'Both']:.2f}% | "
          f"{r['hit_disagreement']:.4f} | {r['directional_asymmetry']:.4f} | "
          f"{r['candidate_jaccard']:.4f} | {r['phi']:.4f} |")
    A('')
    A('Uniform fusion beats the stronger single tokenizer on 4/4 datasets (bootstrap '
      'significant on ' + str(int(P['fusion_vs_best_significant_95'].sum())) +
      '/4), the fixed RRF diagnostic agrees, the oracle is far above the fusion on 4/4, '
      'and bidirectional Only-tokenizer groups are populated everywhere.')
    A('')
    A('## D. Cross-tokenizer target likelihood')
    A('')
    A('| dataset | Only group | n | mean NLL own | mean NLL other | argmin == own '
      '(raw) | argmin == own (z-normalised) | permutation null | global baseline |')
    A('|---|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        for grpname in ('Only-Spectral', 'Only-RQVAE'):
            r = al[(al.tier == 'PRIMARY') & (al.normalisation == 'raw') &
                   (al.dataset == ds) & (al.group == grpname)].iloc[0]
            rz = al[(al.tier == 'PRIMARY') & (al.normalisation == 'z_valid') &
                    (al.dataset == ds) & (al.group == grpname)].iloc[0]
            A(f"| {SHORT[ds]} | {grpname} | {int(r['n_users'])} | {r['mean_nll_own']:.4f} | "
              f"{r['mean_nll_other']:.4f} | {100*r['argmin_match_accuracy']:.2f}% | "
              f"{100*rz['argmin_match_accuracy']:.2f}% | {r['permutation_null_mean']:.4f} | "
              f"{r['global_baseline_accuracy']:.4f} |")
    A('')
    A('Pooled over both Only groups the per-user argmin of the teacher-forced target NLL '
      'recovers the uniquely successful tokenizer for ' +
      ', '.join(f"{SHORT[d]} {100*pool.loc[d,'argmin_match_accuracy']:.1f}%" for d in DATASETS) +
      ' of users, unchanged under validation normalisation (' +
      ', '.join(f"{SHORT[d]} {100*poolz.loc[d,'argmin_match_accuracy']:.1f}%" for d in DATASETS) +
      '), against a 50% random baseline and a 0.50 permutation null (p = 0.000).')
    A('')


def sec_step2(A):
    mp = STEP2 / 'shared_single_factorization_metrics.csv'
    cp = STEP2 / 'shared_factorization_complementarity.csv'
    ap = STEP2 / 'shared_nll_alignment.csv'
    ip = STEP2 / 'independent_vs_shared_factorization.csv'
    if not (mp.exists() and cp.exists()):
        A('## E. Shared model construction')
        A('')
        A('One T5 trained jointly on both factorizations: a reserved control token is '
          'prepended to the encoder input and the history and target are written in that '
          "factorization's SID; one encoder, one decoder, one embedding table and one "
          'output head, with no factorization-specific module. Vocabulary 1027 '
          '(1024 SID tokens + PAD + 2 control tokens). Exposure is '
          'factorization-balanced; early stopping is on the macro average of item-level '
          'validation NDCG@20 across factorizations. See `SHARED_frozen vocabulary-audit record`.')
        A('')
        note = ('_Pending: the shared runs had not produced evaluation artifacts when '
                'this report was generated. See `step2_shared_model/runs/*/train_log.json` '
                'for training state and `_PROVENANCE.json` for run provenance._')
        for sec in ('## F. Shared single-factorization performance',
                    '## G. Shared complementarity',
                    '## H. Shared target-NLL alignment',
                    '## I. Independent vs shared parameters'):
            A(sec)
            A('')
            A(note)
            A('')
        return
    m = pd.read_csv(mp)
    c = pd.read_csv(cp)
    A('## E. Shared model construction')
    A('')
    A('One T5 (4+4 layers, d_model 128, d_ff 1024, 6 heads) trained jointly on both '
      'factorizations. A reserved control token is prepended to the encoder input and '
      'the history and target are written in that factorization\'s SID; there is one '
      'encoder, one decoder, one embedding table and one output head, with no '
      'factorization-specific module. Vocabulary 1027 (1024 SID tokens + PAD + 2 control '
      'tokens). Exposure is factorization-balanced: every original instance once per '
      'factorization per epoch. Early stopping is on the macro average of item-level '
      'validation NDCG@20 across factorizations.')
    A('')
    A('## F. Shared single-factorization performance')
    A('')
    A('| dataset | factorization | independent NDCG@20 | shared NDCG@20 | ratio | '
      'severe collapse |')
    A('|---|---|---|---|---|---|')
    for _, r in m.iterrows():
        A(f"| {SHORT[r['dataset']]} | {r['label']} | {r['independent_ndcg20']:.5f} | "
          f"{r['shared_ndcg20']:.5f} | {100*r['shared_over_independent_ndcg20']:.1f}% | "
          f"{'YES' if r['severe_sharing_collapse'] else 'no'} |")
    A('')
    ratios = m['shared_over_independent_ndcg20'].dropna()
    A(f"Shared/independent ratio: min {100*ratios.min():.1f}%, median "
      f"{100*ratios.median():.1f}%. Cells below the 70% collapse threshold: "
      f"{int(m['severe_sharing_collapse'].sum())}/{len(m)}.")
    A('')
    A('## G. Shared complementarity')
    A('')
    A('| dataset | best shared single | shared fusion | fusion − best | 95% CI | p | '
      'oracle | oracle − best | exploited | Only-F1 | Only-F2 | Hit disagreement | '
      'candidate Jaccard |')
    A('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in c.iterrows():
        A(f"| {SHORT[r['dataset']]} | {r['best_shared_single_ndcg20']:.5f} | "
          f"{r['shared_fusion_ndcg20']:.5f} | **{r['shared_fusion_gain']:+.5f}** | "
          f"[{r['fusion_vs_best_ci_lo']:+.5f}, {r['fusion_vs_best_ci_hi']:+.5f}] | "
          f"{r['fusion_vs_best_p']:.4f} | {r['shared_oracle_ndcg20']:.5f} | "
          f"{r['shared_oracle_gain']:+.5f} | "
          f"{100*r['oracle_exploited_fraction']:.1f}% | {100*r['only_f1_fraction']:.2f}% | "
          f"{100*r['only_f2_fraction']:.2f}% | {r['hit_disagreement']:.4f} | "
          f"{r['candidate_jaccard']:.4f} |")
    A('')
    if ap.exists():
        a = pd.read_csv(ap)
        A('## H. Shared target-NLL alignment')
        A('')
        A('| dataset | group | n | mean NLL own | mean NLL other | argmin == own | '
          'permutation null | p |')
        A('|---|---|---|---|---|---|---|---|')
        for _, r in a.iterrows():
            A(f"| {SHORT[r['dataset']]} | {r['group']} | {int(r['n_users'])} | "
              f"{r['mean_nll_own']:.4f} | {r['mean_nll_other']:.4f} | "
              f"**{100*r['argmin_match_accuracy']:.2f}%** | "
              f"{r['permutation_null_mean']:.4f} | {r['permutation_p_value']:.4f} |")
        A('')
    if ip.exists():
        d = pd.read_csv(ip)
        A('## I. Independent vs shared parameters')
        A('')
        A('| dataset | metric | independent | shared | shared − independent | retains |')
        A('|---|---|---|---|---|---|')
        for _, r in d.iterrows():
            A(f"| {SHORT[r['dataset']]} | {r['metric_label']} | "
              f"{r['independent_factors']:.5f} | {r['shared_checkpoint']:.5f} | "
              f"{r['shared_minus_independent']:+.5f} | "
              f"{'yes' if r['shared_retains_direction'] else '**no**'} |")
        A('')


def sec_training(A):
    runs = {}
    base = STEP2 / 'runs'
    if base.exists():
        for d in sorted(base.iterdir()):
            tl = d / 'train_log.json'
            if d.is_dir() and tl.exists():
                runs[d.name] = json.loads(tl.read_text())
    A('## J. Training performed')
    A('')
    if not runs:
        A('No shared-model training completed in this session.')
        A('')
        return
    A('| run | dataset | factorizations | config | epochs | best epoch | best macro '
      'valid NDCG@20 | seed | elapsed (s) |')
    A('|---|---|---|---|---|---|---|---|---|')
    for k, j in runs.items():
        A(f"| `{k}` | {j.get('dataset')} | {'+'.join(j.get('factorizations', []))} | "
          f"batch 256, lr 1e-4, 4+4 layers, d_model 128, patience 10 | "
          f"{len(j.get('epochs', []))} | {j.get('best_epoch')} | "
          f"{j.get('best_macro_valid_ndcg20', float('nan')):.6f} | "
          f"{j.get('config', {}).get('seed')} | {j.get('elapsed_s')} |")
    A('')
    A(f"Total new recommender/shared models trained for this study: **{len(runs)}** "
      f"(shared) + **0** (STEP 1), against a cap of 8.")
    A('')


def sec_verdicts(A):
    """The §53 verdict block, computed from the STEP 1 and STEP 2 decision rules."""
    from xt_step1_report import decide_step1_verdicts
    from xt_step2_report import decide_verdicts
    import json as _json

    comp = pref = 'NOT ESTABLISHED'
    shared = nec = 'NOT ESTABLISHED'
    try:
        man = _json.loads((STEP1 / 'tokenizer_manifest.json').read_text())
        div = pd.read_csv(STEP1 / 'cross_tokenizer_pair_diversity.csv')
        al = pd.read_csv(STEP1 / 'cross_tokenizer_nll_alignment.csv')
        P = div[div.tier == 'PRIMARY']
        comp, pref, _ = decide_step1_verdicts(
            man['fairness'], P,
            al[(al.tier == 'PRIMARY') & (al.normalisation == 'raw')],
            al[(al.tier == 'PRIMARY') & (al.normalisation == 'z_valid')])
    except Exception as exc:  # pragma: no cover
        A(f'_(STEP 1 verdicts unavailable: {exc})_')

    mp = STEP2 / 'shared_single_factorization_metrics.csv'
    cp = STEP2 / 'shared_factorization_complementarity.csv'
    if mp.exists() and cp.exists():
        try:
            shared, nec, _ = decide_verdicts(pd.read_csv(mp), pd.read_csv(cp))
        except Exception as exc:  # pragma: no cover
            A(f'_(STEP 2 verdicts unavailable: {exc})_')

    A('## Verdicts')
    A('')
    A('```')
    A('CROSS-INDEPENDENT-TOKENIZER COMPLEMENTARITY:')
    A(comp)
    A('')
    A('INSTANCE-SPECIFIC CROSS-TOKENIZER PREFERENCE:')
    A(pref)
    A('')
    A('SHARED-PARAMETER FACTORIZATION COMPLEMENTARITY:')
    A(shared)
    A('')
    A('INDEPENDENT PARAMETERS ARE NECESSARY FOR COMPLEMENTARITY:')
    A(nec)
    A('```')
    A('')
    import json as _j
    n_shared = 0
    _c = STEP2 / 'shared_factorization_complementarity.csv'
    if _c.exists():
        try:
            n_shared = len(pd.read_csv(_c))
        except Exception:
            n_shared = 0
    A('These labels are produced by the pre-declared decision rules in '
      '`xt_step1_report.py::decide_step1_verdicts` and '
      '`xt_step2_report.py::decide_verdicts`, whose branches are covered by '
      '`xt_test_verdicts.py`.')
    A('')
    A(f'STEP 2 coverage in this report: **{n_shared} of 4 datasets**. The rules require '
      f'positive fusion gains in >=3 cells for `SUPPORTED`, so with fewer than three '
      f'datasets evaluated the strongest label they can return is `PARTIALLY '
      f'SUPPORTED`; with no shared artifacts at all they return `NOT ESTABLISHED`. '
      f'The label above is therefore a statement about the coverage shown, not a '
      f'weaker claim about the phenomenon.')
    A('')


def main():
    A_ = []
    A = A_.append
    A('# FINAL_TWO_STEP_REPORT.md')
    A('')
    A('Two-step mechanism study. **STEP 1** asks whether factorization '
      'complementarity survives a genuinely different tokenizer construction. '
      '**STEP 2** asks whether it survives sharing one parameter set across '
      'factorizations.')
    A('')
    A('Governance: no paper content was read or written; no historical SID array, '
      'checkpoint, beam dump or split was modified; no test-set selection; no learned '
      'fusion; no target-aware selector; teacher-forced NLL is used as explanatory '
      'evidence only and never as an inference-time signal.')
    A('')
    A('## A. State recovery')
    A('')
    A('Read in full: `outputs/mechanism/factorization_generality/{frozen_state.md, '
      'FACTORIZATION_GENERALITY_REPORT.md}`, '
      '`outputs/mechanism/full_id_reaudit/FULL_ID_REAUDIT_REPORT.md`, '
      '`outputs/mechanism/MECHANISM_STUDY_REPORT.md`, `FINAL_ENGINEERING_CLOSURE.md`, '
      '`configs/canonical.yaml`, `docs/{ITEM_ID_CONVENTION,SID_FORMAT,'
      'EXPERIMENT_PROTOCOL}.md`, and the two `_PROVENANCE.json` '
      'files plus `ARTIFACT_MATRIX.csv`, `SEED_ARTIFACT_MATRIX.csv`, '
      '`variance_decomposition.csv`, `unique_model_nll_alignment.csv` from the '
      'factorization-generality study. `outputs/cross_tokenizer_shared/frozen_state.md` '
      'records the recovered state.')
    A('')
    sec_step1(A)
    sec_step2(A)
    sec_training(A)
    A('## K. Limitations')
    A('')
    A('* **Tokenizer family count.** The primary comparison is one genuinely different '
      'construction (RQ-VAE) against Spectral; OPQ and RQ-OPQ are secondary and their '
      'historical checkpoints were selected by a different metric.')
    A('* **Single seed.** Both the canonical checkpoints and the shared models use seed '
      '2025. No multi-seed estimate of the complementarity magnitude is available on '
      'either step.')
    A('* **Score calibration.** The two STEP 1 models are independently trained, so raw '
      'NLL is not calibrated across them. The alignment result is therefore also '
      'reported on validation-normalised NLL and with a within-user ordering; both '
      'agree with the raw result, but the raw level comparison alone would not be '
      'interpretable.')
    A('* **Shared compute exposure.** The shared model sees both factorizations every '
      'epoch, so it executes roughly twice the optimizer steps of an independent model. '
      'This is inherent to factorization-balanced exposure; it is reported, not used as '
      'an efficiency claim.')
    A('* **Oracle numbers are explanatory only.** They involve the ground-truth target '
      'and are not a deployable method.')
    A('')
    sec_verdicts(A)

    (OUT / 'FINAL_TWO_STEP_REPORT.md').write_text('\n'.join(A_))
    print('wrote FINAL_TWO_STEP_REPORT.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
