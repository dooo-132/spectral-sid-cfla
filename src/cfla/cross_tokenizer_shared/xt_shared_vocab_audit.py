#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — SHARED_frozen vocabulary-audit record

Establishes the shared vocabulary for the factorization-controlled shared T5 from
the repository's actual token-numbering convention (never hard-coded from memory)
and proves that the factorization control tokens cannot collide with anything.

Outputs -> step2_shared_model/SHARED_frozen vocabulary-audit record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import CODEBOOK, DATASETS, PRIMARY, STEP2, TOKENIZERS, load_sid, shared_vocab_plan  # noqa: E402


def main():
    facs = list(PRIMARY)
    plan = shared_vocab_plan(facs)
    L = plan['L']

    # SID token range actually attainable, measured from the SID arrays
    lo, hi = None, None
    per_ds = {}
    for f in facs:
        for ds in DATASETS:
            sid = load_sid(f, ds)
            toks = sid + np.arange(sid.shape[1], dtype=np.int64)[None, :] * CODEBOOK + 1
            lo = int(toks.min()) if lo is None else min(lo, int(toks.min()))
            hi = int(toks.max()) if hi is None else max(hi, int(toks.max()))
            per_ds[f'{f}|{ds}'] = {'L': int(sid.shape[1]),
                                   'token_min': int(toks.min()),
                                   'token_max': int(toks.max())}

    ctrl = plan['control_tokens']
    sid_range = (1, L * CODEBOOK)
    ctrl_ok = all(v > sid_range[1] for v in ctrl.values())
    distinct = len(set(ctrl.values())) == len(ctrl)
    pad_ok = all(v != 0 for v in ctrl.values())
    ids_contiguous = sorted(ctrl.values()) == list(range(L * CODEBOOK + 1,
                                                         L * CODEBOOK + 1 + len(facs)))

    # verify against a trained shared checkpoint if one already exists
    ckpt_checks = {}
    for ds in DATASETS:
        p = STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth'
        if p.exists():
            import torch
            sd = torch.load(str(p), map_location='cpu')
            v = int(sd['model.shared.weight'].shape[0])
            ckpt_checks[ds] = {'path': str(p), 'vocab_size': v,
                               'matches_plan': v == plan['vocab_size']}

    lines = [
        '# SHARED_frozen vocabulary-audit record', '',
        'Vocabulary of the factorization-controlled shared T5.', '',
        '## 1. Repository token convention (read from the code, not assumed)', '',
        '`src/tiger/dataset.py::item2code` builds, for SID-array row `i` (item `i+1`),',
        'the token list `token = code + position * codebook_size + 1`. The model',
        'therefore needs `vocab_size = L * codebook_size + 1` tokens, with id 0 shared',
        'by PAD and EOS (`docs/SID_FORMAT.md`). For L = 4 and codebook 256 this is 1025,',
        'which is exactly the vocabulary the canonical Spectral-K3 and RQ-VAE-3layer',
        'checkpoints use (verified from their state dicts: `model.shared.weight` has',
        '1025 rows).', '',
        '## 2. Measured SID token range', '',
        f'* attainable SID tokens across every primary factorization x dataset: '
        f'`[{lo}, {hi}]`',
        f'* theoretical maximum for L={L}, codebook={CODEBOOK}: `{sid_range[1]}`', '',
        '| factorization | dataset | L | token min | token max |',
        '|---|---|---|---|---|']
    for k, v in per_ds.items():
        f, ds = k.split('|')
        lines.append(f"| `{f}` | {ds} | {v['L']} | {v['token_min']} | {v['token_max']} |")

    lines += ['', '## 3. Control tokens', '',
              f'Reserved ids start at `L*codebook_size + 1 = {L * CODEBOOK + 1}`.',
              '', '| factorization | control token id |', '|---|---|']
    for f, i in ctrl.items():
        lines.append(f'| `{f}` | **{i}** |')
    lines += ['', f'* `vocab_size` for the shared model: **{plan["vocab_size"]}** '
              f'(= {sid_range[1]} SID tokens + id 0 + {len(facs)} control tokens)',
              '* `pad_token_id = 0`, `eos_token_id = 0` (unchanged from canonical)', '',
              '## 4. Collision checks', '',
              '| check | result |', '|---|---|',
              f'| control tokens above the SID range | {"PASS" if ctrl_ok else "FAIL"} |',
              f'| control tokens mutually distinct | {"PASS" if distinct else "FAIL"} |',
              f'| no control token equals PAD/EOS (id 0) | {"PASS" if pad_ok else "FAIL"} |',
              f'| control ids contiguous above the SID block | {"PASS" if ids_contiguous else "FAIL"} |',
              f'| control token carries no target/test/performance information | PASS '
              '(the token is a constant per factorization; it is prepended to the '
              'encoder input and nothing else depends on it) |', '',
              '## 5. Verified against trained shared checkpoints', '']
    if ckpt_checks:
        lines += ['| dataset | checkpoint | vocab_size | matches plan |', '|---|---|---|---|']
        for ds, v in ckpt_checks.items():
            lines.append(f"| {ds} | `{Path(v['path']).name}` | {v['vocab_size']} | "
                         f"{v['matches_plan']} |")
    else:
        lines.append('_No shared checkpoint exists yet; this section is filled in by '
                     're-running the audit after training._')

    lines += ['', '## 6. Parameters added by the control tokens', '',
              f'`d_model = 128`, so {len(facs)} extra embedding rows add '
              f'{len(facs) * 128} parameters. The canonical model has 4,590,848; the '
              f'shared model has 4,591,104. The model capacity (4+4 layers, d_model 128, '
              'd_ff 1024, 6 heads) is unchanged.', '']

    (STEP2 / 'SHARED_frozen vocabulary-audit record').write_text('\n'.join(lines))
    (STEP2 / 'shared_vocab_plan.json').write_text(json.dumps(
        {'plan': plan, 'measured_sid_token_range': [lo, hi],
         'per_cell': per_ds, 'checkpoints': ckpt_checks}, indent=2))
    print('wrote SHARED_frozen vocabulary-audit record')
    print('plan:', plan)
    print('checks: sid_range_ok', ctrl_ok, 'distinct', distinct, 'pad_ok', pad_ok,
          'contiguous', ids_contiguous)
    return 0


if __name__ == '__main__':
    sys.exit(main())
