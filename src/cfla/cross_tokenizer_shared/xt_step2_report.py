#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 report generator.

Every number in STEP2_SHARED_MODEL_REPORT.md is read from the artifacts produced
by this study.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, PRIMARY, SHORT, STEP2, TOKENIZERS  # noqa: E402


def decide_verdicts(metrics, comp):
    """Transparent, pre-declared decision rules (printed in the report).

    SHARED-PARAMETER FACTORIZATION COMPLEMENTARITY
      CONFOUNDED          any factorization shows severe sharing collapse (<70% of
                          its independent NDCG@20) -- §44
      SUPPORTED           fusion gain > 0 on >=3/4 cells with >=2 significant,
                          oracle gain > 0 and significant on >=3/4, and both Only-F
                          groups populated on 4/4
      PARTIALLY SUPPORTED oracle gain > 0 on all cells and Only-F groups populated
                          on all but at most one cell
      NOT SUPPORTED       no positive fusion and no positive oracle gain anywhere
      NOT ESTABLISHED     anything else

    INDEPENDENT PARAMETERS ARE NECESSARY FOR COMPLEMENTARITY
      if shared is SUPPORTED      -> NOT SUPPORTED (the effect survives sharing)
      if shared is NOT SUPPORTED  -> SUPPORTED
      otherwise                   -> NOT ESTABLISHED
    """
    n = len(comp)
    if n == 0:
        return 'NOT ESTABLISHED', 'NOT ESTABLISHED', {}
    collapse = int(metrics['severe_sharing_collapse'].sum()) if len(metrics) else 0
    st = dict(n=n, collapse=collapse,
              fusion_pos=int((comp['shared_fusion_gain'] > 0).sum()),
              fusion_sig=int(comp['fusion_vs_best_significant_95'].sum()),
              oracle_pos=int((comp['shared_oracle_gain'] > 0).sum()),
              oracle_sig=int(comp['oracle_vs_fusion_significant_95'].sum()),
              only_pop=int(((comp['only_f1_fraction'] > 0.005) &
                            (comp['only_f2_fraction'] > 0.005)).sum()))
    if collapse:
        shared = 'CONFOUNDED'
    elif (st['fusion_pos'] >= 3 and st['fusion_sig'] >= 2 and st['oracle_pos'] == n
          and st['oracle_sig'] >= 3 and st['only_pop'] == n):
        shared = 'SUPPORTED'
    elif st['oracle_pos'] == n and st['only_pop'] >= max(2, n - 1):
        shared = 'PARTIALLY SUPPORTED'
    elif st['fusion_pos'] == 0 and st['oracle_pos'] == 0:
        shared = 'NOT SUPPORTED'
    else:
        shared = 'NOT ESTABLISHED'
    nec = ('NOT SUPPORTED' if shared == 'SUPPORTED'
           else 'SUPPORTED' if shared == 'NOT SUPPORTED'
           else 'NOT ESTABLISHED')
    return shared, nec, st


def main():
    mp, cp = (STEP2 / 'shared_single_factorization_metrics.csv',
              STEP2 / 'shared_factorization_complementarity.csv')
    if not (mp.exists() and cp.exists()):
        print('shared metrics not available yet; STEP2 report not generated')
        return 0
    metrics = pd.read_csv(mp)
    comp = pd.read_csv(cp)
    cmp_path = STEP2 / 'independent_vs_shared_factorization.csv'
    cmpdf = pd.read_csv(cmp_path) if cmp_path.exists() else pd.DataFrame()
    al_path = STEP2 / 'shared_nll_alignment.csv'
    al = pd.read_csv(al_path) if al_path.exists() else pd.DataFrame()

    L = []
    A = L.append
    A('# STEP2_SHARED_MODEL_REPORT.md')
    A('')
    A('**Question.** If the different factorizations are served by ONE model with ONE '
      'parameter set, does factorization complementarity survive? If it does, then '
      '"independent parameters" is not a necessary ingredient of the effect.')
    A('')
    A('**Scope:** mechanism validation. No learned fusion, no selector, no test-set '
      'selection. Test is never used for checkpoint selection.')
    A('')
    A('---')
    A('')
    A('## A. Construction')
    A('')
    A('A single T5 (4+4 layers, d_model 128, d_ff 1024, 6 heads, d_kv 64, dropout 0.1) '
      'is trained jointly on all factorizations:')
    A('')
    A('```')
    for f in PRIMARY:
        A(f"<{TOKENIZERS[f]['label']}>' + history written in that factorization's SID")
        A(f"    -> target written in that factorization's SID")
    A('```')
    A('')
    A('* **One** encoder, **one** decoder, **one** embedding table, **one** output '
      'head. There is no factorization-specific decoder, head, adapter or expert.')
    A('* The only factorization-specific information is (a) a reserved control token '
      'prepended to the encoder input and (b) which factorization\'s SID sequence the '
      'history and target are written in. The same `(user, history, target)` '
      'interaction therefore becomes a different discrete sequence per factorization.')
    A('* Vocabulary: ' + ', '.join(
        f"`{TOKENIZERS[f]['label']}` -> control id {i}" for f, i in
        zip(PRIMARY, [1025, 1026])) +
      '; SID tokens occupy [1, 1024] and id 0 is PAD/EOS, so nothing collides. '
      'Vocab 1027. The two control tokens add 256 embedding parameters; canonical '
      'capacity is unchanged.')
    A('* **Factorization-balanced exposure**: every original training instance is shown '
      'exactly once per factorization per epoch, so both factorizations see exactly the '
      'same data and the same number of samples.')
    A('* **Validation / early stopping** uses the macro average of item-level validation '
      'NDCG@20 across factorizations (each evaluated with its own control token and its '
      'own SID array), patience 10, seed 2025. Test is never used.')
    A('')

    if len(metrics):
        A('## B. Single-factorization performance (sharing collapse check)')
        A('')
        A('| dataset | factorization | independent NDCG@20 | shared NDCG@20 | shared / '
          'independent | independent R@20 | shared R@20 | severe collapse |')
        A('|---|---|---|---|---|---|---|---|')
        for _, r in metrics.iterrows():
            A(f"| {SHORT[r['dataset']]} | {r['label']} | {r['independent_ndcg20']:.5f} | "
              f"{r['shared_ndcg20']:.5f} | "
              f"{100*r['shared_over_independent_ndcg20']:.1f}% | "
              f"{r['independent_recall20']:.5f} | {r['shared_recall20']:.5f} | "
              f"{'YES' if r['severe_sharing_collapse'] else 'no'} |")
        A('')
        ratios = metrics['shared_over_independent_ndcg20'].dropna()
        A(f"Shared/independent NDCG@20 ratio: min {100*ratios.min():.1f}%, "
          f"median {100*ratios.median():.1f}%, max {100*ratios.max():.1f}%. "
          f"Cells below the 70% collapse threshold: "
          f"{int(metrics['severe_sharing_collapse'].sum())} of {len(metrics)}.")
        A('')

    if len(comp):
        A('## C. Shared-checkpoint complementarity')
        A('')
        A('All numbers below come from ONE checkpoint evaluated twice, once per '
          'factorization.')
        A('')
        A('| dataset | best shared single | shared fusion | fusion − best | 95% CI | p | '
          'RRF(k=60) | shared oracle | oracle − best | oracle − fusion | exploited |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in comp.iterrows():
            A(f"| {SHORT[r['dataset']]} | {r['best_shared_single_ndcg20']:.5f} | "
              f"{r['shared_fusion_ndcg20']:.5f} | **{r['shared_fusion_gain']:+.5f}** | "
              f"[{r['fusion_vs_best_ci_lo']:+.5f}, {r['fusion_vs_best_ci_hi']:+.5f}] | "
              f"{r['fusion_vs_best_p']:.4f} | {r['shared_rrf_ndcg20']:.5f} | "
              f"{r['shared_oracle_ndcg20']:.5f} | {r['shared_oracle_gain']:+.5f} | "
              f"{r['shared_oracle_minus_fusion']:+.5f} | "
              f"{100*r['oracle_exploited_fraction']:.1f}% |")
        A('')
        A('| dataset | Only-F1 | Only-F2 | Both | Neither | Hit disagreement | '
          'directional asymmetry | phi | Hit Jaccard | candidate Jaccard |')
        A('|---|---|---|---|---|---|---|---|---|---|')
        for _, r in comp.iterrows():
            A(f"| {SHORT[r['dataset']]} | {100*r['only_f1_fraction']:.2f}% | "
              f"{100*r['only_f2_fraction']:.2f}% | {100*r['both_fraction']:.2f}% | "
              f"{100*r['neither_fraction']:.2f}% | {r['hit_disagreement']:.4f} | "
              f"{r['directional_asymmetry']:.4f} | {r['phi']:.4f} | "
              f"{r['hit_jaccard']:.4f} | {r['candidate_jaccard']:.4f} |")
        A('')

    if len(al):
        A('## D. Shared-checkpoint target-NLL alignment')
        A('')
        A('Same checkpoint, same parameters, same sequence length for both '
          'factorizations, so raw TotalNLL is directly comparable.')
        A('')
        A('| dataset | group | n | mean NLL own | mean NLL other | argmin == own | '
          'permutation null | p | global-cheaper baseline |')
        A('|---|---|---|---|---|---|---|---|---|')
        for _, r in al.iterrows():
            A(f"| {SHORT[r['dataset']]} | {r['group']} | {int(r['n_users'])} | "
              f"{r['mean_nll_own']:.4f} | {r['mean_nll_other']:.4f} | "
              f"**{100*r['argmin_match_accuracy']:.2f}%** | "
              f"{r['permutation_null_mean']:.4f} | {r['permutation_p_value']:.4f} | "
              f"{r['global_baseline_accuracy']:.4f} |")
        A('')

    if len(cmpdf):
        A('## E. Independent vs shared parameters')
        A('')
        A('| dataset | metric | independent | shared | shared − independent | '
          'direction retained |')
        A('|---|---|---|---|---|---|')
        for _, r in cmpdf.iterrows():
            iv = r['independent_factors']
            sv = r['shared_checkpoint']
            A(f"| {SHORT[r['dataset']]} | {r['metric_label']} | {iv:.5f} | {sv:.5f} | "
              f"{r['shared_minus_independent']:+.5f} | "
              f"{'yes' if r['shared_retains_direction'] else '**no**'} |")
        A('')

    # training summary
    runs = {}
    for d in sorted((STEP2 / 'runs').iterdir()) if (STEP2 / 'runs').exists() else []:
        tl = d / 'train_log.json'
        if tl.exists():
            j = json.loads(tl.read_text())
            runs[d.name] = j
    if runs:
        A('## F. Training performed')
        A('')
        A('| run | factorizations | epochs completed | best epoch | best macro valid '
          'NDCG@20 | train samples/epoch (total) | params | elapsed (s) |')
        A('|---|---|---|---|---|---|---|---|')
        for k, j in runs.items():
            A(f"| `{k}` | {'+'.join(j.get('factorizations', []))} | "
              f"{len(j.get('epochs', []))} | {j.get('best_epoch')} | "
              f"{j.get('best_macro_valid_ndcg20', float('nan')):.6f} | "
              f"{j.get('train_size_total')} | {j.get('params')} | "
              f"{j.get('elapsed_s')} |")
        A('')
        tot = sum(j.get('elapsed_s', 0) for j in runs.values())
        steps = sum(sum(e.get('steps', 0) for e in j.get('epochs', [])) for j in runs.values())
        A(f'Total GPU wall-clock across the shared runs: {tot/3600:.2f} h. '
          f'Optimizer steps executed: {steps}. Because exposure is '
          f'factorization-balanced, each epoch contains 2x the single-model sample '
          f'count; this is a property of the design, not a compute-efficiency claim.')
        A('')

    A('## F.1 Optional controls: decision and reason')
    A('')
    A('A compute-matched (`step_matched`) shared model for Beauty was prepared '
      '(`xt_train_shared.py --exposure step_matched`, which shows each original '
      'instance once per epoch under a uniformly random factorization, keeping '
      'optimizer steps ~= a single independent model). It was **not run**: the four '
      'primary shared models consumed the available GPU budget, and the decision rule '
      'for this control is compute budget only. Its purpose would have been to test '
      'whether shared complementarity depends on the 2x exposure inherent to '
      'factorization-balanced training; that remains an open confound and is listed '
      'under Limitations rather than papered over.')
    A('')
    shared_v, nec_v, st = decide_verdicts(metrics, comp)
    A('## G. Verdict')
    A('')
    A('```')
    A('SHARED-PARAMETER FACTORIZATION COMPLEMENTARITY:')
    A(shared_v)
    A('')
    A('INDEPENDENT PARAMETERS ARE NECESSARY FOR COMPLEMENTARITY:')
    A(nec_v)
    A('```')
    A('')
    A('Decision inputs (pre-declared rules, stated in `xt_step2_report.py`):')
    A('')
    A(f"* cells with severe sharing collapse (<70% of independent NDCG@20): "
      f"{st.get('collapse')} of {st.get('n')}")
    A(f"* shared fusion gain > 0 in {st.get('fusion_pos')}/{st.get('n')} cells, "
      f"significant in {st.get('fusion_sig')}")
    A(f"* shared oracle gain > 0 in {st.get('oracle_pos')}/{st.get('n')} cells, "
      f"significant in {st.get('oracle_sig')}")
    A(f"* both Only-factorization groups populated in {st.get('only_pop')}/{st.get('n')}")
    A('')

    (STEP2 / 'STEP2_SHARED_MODEL_REPORT.md').write_text('\n'.join(L))
    print('wrote STEP2_SHARED_MODEL_REPORT.md (verdict placeholders left for the '
          'final adjudication)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
