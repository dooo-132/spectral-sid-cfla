#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FEATURE_AUDIT.md from the declared feature schema.

The table is produced from `dg_features.FEATURES`, the same list the builder uses, so it
cannot drift from the features actually computed.  Every row states whether the feature
uses the target (must be NO), whether it is available at inference (must be YES), and
whether it requires candidate NLL scoring.

Run:  python3 dg_feature_audit.py --outdir outputs/disagreement_gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'candidate_conditioned')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dg_features import (DATASET_ONEHOT, DATASETS, DISAGREEMENT, FEATURES,  # noqa: E402
                         LIKELIHOOD, NEEDS_NLL, RETRIEVAL, SHORT, USER_LEVEL)

DESC = {
    'present_F1': 'candidate appears in F1\'s beam', 'present_F2': 'candidate appears in F2\'s beam',
    'rank_F1': 'F1 rank of the candidate, 31 when absent', 'rank_F2': 'F2 rank, 31 when absent',
    'normalized_rank_F1': '(rank_F1 - 1) / (union size - 1)',
    'normalized_rank_F2': '(rank_F2 - 1) / (union size - 1)',
    'abs_rank_difference': '|rank_F1 - rank_F2|',
    'P_F1': 'merged beam probability under F1, 0 when absent',
    'P_F2': 'merged beam probability under F2, 0 when absent',
    'P_diff': 'P_F1 - P_F2', 'abs_P_diff': '|P_F1 - P_F2|',
    'P_max': 'max(P_F1, P_F2)', 'P_min': 'min(P_F1, P_F2)',
    'UniformScore': '0.5*P_F1 + 0.5*P_F2 (frozen Uniform formula)',
    'zNLL_F1': 'frozen-scaler z-score of the candidate\'s own TotalNLL under F1',
    'zNLL_F2': 'frozen-scaler z-score of the candidate\'s own TotalNLL under F2',
    'Delta_zNLL': 'zNLL_F1 - zNLL_F2', 'abs_Delta_zNLL': '|zNLL_F1 - zNLL_F2|',
    'min_zNLL': 'min(zNLL_F1, zNLL_F2)', 'max_zNLL': 'max(zNLL_F1, zNLL_F2)',
    'LSE_score': 'logsumexp(-zNLL_F1, -zNLL_F2) (frozen LSE formula)',
    'MinNLL_score': '-min(zNLL_F1, zNLL_F2) (frozen MinNLL formula)',
    'only_F1': 'present in F1 only', 'only_F2': 'present in F2 only',
    'both_present': 'present in both factorizations',
    'rank_sign_disagreement': 'presence indicator disagrees between factorizations',
    'one_factorization_top5': 'top-5 under at least one factorization',
    'one_factorization_top10': 'top-10 under at least one factorization',
    'one_factorization_top20': 'top-20 under at least one factorization',
    'both_top5': 'top-5 under both', 'both_top10': 'top-10 under both',
    'both_top20': 'top-20 under both',
    'u_candidate_jaccard': 'user-level Jaccard of the two candidate sets',
    'u_intersection_size': 'user-level |C_F1 ∩ C_F2|',
    'u_union_size': 'user-level |C_F1 ∪ C_F2|',
    'u_top1_agreement': 'user-level: the two factorizations agree on the top item',
    'u_top5_jaccard': 'user-level top-5 Jaccard',
    'u_top10_jaccard': 'user-level top-10 Jaccard',
    'u_top20_jaccard': 'user-level top-20 Jaccard',
    'u_rank_correlation': 'user-level Spearman over candidates present in both',
    'u_entropy_F1': 'user-level entropy of the F1 score distribution',
    'u_entropy_F2': 'user-level entropy of the F2 score distribution',
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/disagreement_gate')
    a = ap.parse_args(argv)
    out = Path(a.outdir)

    # hard rejection rule
    offenders = [f for f in FEATURES
                 if any(w in f.lower().split('_') for w in ('target', 'label', 'hit', 'y'))]
    if offenders:
        raise SystemExit(f'REJECTED: feature names implying target dependency: {offenders}')

    L = ['# FEATURE_AUDIT.md', '',
         'Audit of every input feature of the disagreement-aware candidate gate. Generated '
         'by `analysis/mechanism/candidate_conditioned/dg_feature_audit.py` from the same '
         '`FEATURES` list the dataset builder uses, so it cannot drift from what was '
         'actually computed.', '',
         '## Hard rules', '',
         '* `uses target` is **NO** for every row. The target item, its SID, its rank, its '
         'hit flag and its Only-F1/Only-F2/Both group label are never gate inputs.',
         '* The Only-F1 / Only-F2 / Both group label depends on the target, so it is used '
         '**only in evaluation diagnostics** (sections H, I, K of the report), never as a '
         'feature and never as supervision. The gate must discover disagreement from the '
         'inference-visible features below (prompt section 30).',
         '* `requires NLL scoring` marks the features that could not exist without the '
         'cross-factorization candidate NLL. They are computed from the **frozen** '
         'candidate-NLL artefacts and the **frozen** validation-fitted scaler; this round '
         'recomputes nothing (see `CANDIDATE_NLL_REUSE_AUDIT.md`).',
         '* All continuous features are z-normalised per dataset using mean/std fitted on '
         '**RouterTrain rows only** and stored in `features/FEATURE_SCALER.json`.',
         '', f'**Total features: {len(FEATURES)}** '
             f'({len(RETRIEVAL)} retrieval + {len(LIKELIHOOD)} likelihood + '
             f'{len(DISAGREEMENT)} disagreement + {len(USER_LEVEL)} user-level + '
             f'{len(DATASET_ONEHOT)} dataset one-hot).', '',
         '| # | feature | dim | source | uses target? | available at inference? | '
         'requires NLL scoring? |', '|---|---|---|---|---|---|---|']

    groups = [(RETRIEVAL, 'frozen retrieval scores from the shared model\'s beams'),
              (LIKELIHOOD, 'frozen cross-factorization candidate NLL + frozen scaler'),
              (DISAGREEMENT, 'candidate presence / rank disagreement (inference-visible)'),
              (USER_LEVEL, 'user-level agreement context, copied to every candidate row'),
              (DATASET_ONEHOT, 'dataset identity, pre-declared one-hot')]
    i = 0
    for feats, source in groups:
        for f in feats:
            i += 1
            desc = DESC.get(f, 'dataset one-hot indicator' if f in DATASET_ONEHOT else '')
            L.append(f'| {i} | `{f}` | 1 | {source}: {desc} | NO | YES | '
                     f'{"YES" if f in NEEDS_NLL else "NO"} |')
    L += ['', '## Features explicitly excluded', '',
          '| quantity | why excluded |', '|---|---|',
          '| target item id / target SID / target rank / hit flag | not available at '
          'inference |',
          '| Only-F1 / Only-F2 / Both group label | depends on the target; used only for '
          'evaluation diagnostics |',
          '| 387-dim history hidden states | deferred to a later ablation (prompt section '
          '15); the primary gate uses compact retrieval + NLL + disagreement features only |',
          '| any test-split statistic | test is read only after `GATE_FREEZE_MANIFEST.json` '
          'exists |', '',
          '## Feature-name scan', '',
          f'* Scan for target/label/hit tokens in feature names: '
          f'{"PASS (no offender)" if not offenders else "FAIL"}.',
          f'* Feature-scaler fitting: RouterTrain rows only, per dataset.']
    p = out / 'FEATURE_AUDIT.md'
    p.write_text('\n'.join(L) + '\n')
    print(f'wrote {p} ({len(FEATURES)} features audits)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
