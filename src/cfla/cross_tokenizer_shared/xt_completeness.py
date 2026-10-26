#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Completeness audit for the required output structure (§50) and the required
final-report sections (§52).

The numeric guards in `xt_verify.py` check that values are *correct*. This checks
that the required artifacts and sections *exist* — a different failure mode, and
one that already bit this study once: the final report ran A–K but carried no
verdict block at all, which no numeric guard could detect.

Writes a machine-readable report and exits non-zero if anything required for the
current stage is missing.

Usage:  python3 xt_completeness.py [--allow-pending-step2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import OUT, STEP1, STEP2  # noqa: E402

# §50 required layout. `pending_ok` marks artifacts that legitimately do not
# exist until the shared models finish training.
REQUIRED = [
    ('frozen_state.md', OUT, False),
    ('step1_cross_tokenizer/TOKENIZER_INVENTORY.md', OUT, False),
    ('step1_cross_tokenizer/CROSS_TOKENIZER_FAIRNESS_AUDIT.md', OUT, False),
    ('step1_cross_tokenizer/cross_tokenizer_retrieval.csv', OUT, False),
    ('step1_cross_tokenizer/cross_tokenizer_pair_diversity.csv', OUT, False),
    ('step1_cross_tokenizer/cross_tokenizer_oracle.csv', OUT, False),
    ('step1_cross_tokenizer/cross_tokenizer_only_groups.parquet', OUT, False),
    ('step1_cross_tokenizer/cross_tokenizer_target_nll.parquet', OUT, False),
    ('step1_cross_tokenizer/cross_tokenizer_nll_alignment.csv', OUT, False),
    ('step1_cross_tokenizer/STEP1_CROSS_TOKENIZER_REPORT.md', OUT, False),
    ('step1_cross_tokenizer/_PROVENANCE.json', OUT, False),
    ('step2_shared_model/SHARED_frozen vocabulary-audit record', OUT, False),
    ('step2_shared_model/configs', OUT, False),
    ('step2_shared_model/runs', OUT, False),
    ('step2_shared_model/shared_single_factorization_metrics.csv', OUT, True),
    ('step2_shared_model/shared_factorization_complementarity.csv', OUT, True),
    ('step2_shared_model/shared_nll_alignment.csv', OUT, True),
    ('step2_shared_model/independent_vs_shared_factorization.csv', OUT, True),
    ('step2_shared_model/STEP2_SHARED_MODEL_REPORT.md', OUT, True),
    ('step2_shared_model/_PROVENANCE.json', OUT, False),
    ('figures', OUT, False),
    ('FINAL_TWO_STEP_REPORT.md', OUT, False),
    ('_PROVENANCE.json', OUT, False),
]

# §52 required sections of the final report
SECTIONS = ['## A. State recovery', '## B. Cross-tokenizer fairness',
            '## C. Cross-tokenizer retrieval diversity',
            '## D. Cross-tokenizer target likelihood',
            '## E. Shared model construction',
            '## F. Shared single-factorization performance',
            '## G. Shared complementarity', '## H. Shared target-NLL alignment',
            '## I. Independent vs shared parameters', '## J. Training performed',
            '## K. Limitations', '## Verdicts']

# §53 verdict labels
VERDICTS = ['CROSS-INDEPENDENT-TOKENIZER COMPLEMENTARITY:',
            'INSTANCE-SPECIFIC CROSS-TOKENIZER PREFERENCE:',
            'SHARED-PARAMETER FACTORIZATION COMPLEMENTARITY:',
            'INDEPENDENT PARAMETERS ARE NECESSARY FOR COMPLEMENTARITY:']

VALID_LABELS = {'SUPPORTED', 'PARTIALLY SUPPORTED', 'NOT SUPPORTED', 'CONFOUNDED',
                'NOT ESTABLISHED'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-pending-step2', action='store_true',
                    help='treat STEP 2 evaluation artifacts as not-yet-required')
    a = ap.parse_args(argv)

    missing, pending = [], []
    for rel, _, is_pending in REQUIRED:
        p = OUT / rel
        exists = p.exists()
        if not exists:
            (pending if (is_pending and a.allow_pending_step2) else missing).append(rel)

    # a directory is only "present" if it has content
    for d in ('step2_shared_model/configs', 'step2_shared_model/runs', 'figures'):
        p = OUT / d
        if p.exists() and not any(p.iterdir()) and d not in missing:
            missing.append(f'{d} (empty)')

    report = OUT / 'FINAL_TWO_STEP_REPORT.md'
    sec_missing, verdict_problems = [], []
    if report.exists():
        txt = report.read_text()
        sec_missing = [s for s in SECTIONS if s not in txt]
        for v in VERDICTS:
            if v not in txt:
                verdict_problems.append(f'missing label: {v}')
                continue
            tail = txt.split(v, 1)[1].lstrip('`\n ').split('\n', 1)[0].strip('` ')
            if tail not in VALID_LABELS:
                verdict_problems.append(f'{v} -> invalid label {tail!r}')
    else:
        sec_missing = SECTIONS[:]

    structural_ok = not missing and not sec_missing and not verdict_problems
    if structural_ok and not pending:
        status = 'COMPLETE'
    elif structural_ok and pending:
        status = 'PARTIAL_PENDING_STEP2'
    else:
        status = 'INCOMPLETE'
    ok = status == 'COMPLETE'
    out = {
        'missing_required_artifacts': missing,
        'pending_step2_artifacts': pending,
        'missing_final_report_sections': sec_missing,
        'verdict_label_problems': verdict_problems,
        'status': status,
    }
    (OUT / 'COMPLETENESS_AUDIT.json').write_text(json.dumps(out, indent=2))

    print(f"required artifacts missing : {missing or 'none'}")
    print(f"pending (STEP 2 training)   : {pending or 'none'}")
    print(f"final-report sections missing: {sec_missing or 'none'}")
    print(f"verdict-label problems      : {verdict_problems or 'none'}")
    print(f"status: {out['status']}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
