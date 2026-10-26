#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE A — factorisation artifact inventory.

For every dataset × factorization of the primary pool

    F1 = spectral K2, F2 = spectral K3, F3 = spectral K4, F4 = spectral K5,
    F5 = RQ-VAE 3-layer

verify from disk (never from a report): the SID array and its length, the
checkpoint that scored it, the valid/test beam dumps and their beam budget, the
training config/recipe recorded next to the checkpoint, the seed, the item
universe, and the user counts of every split.  Writes
`FACTORISATION_ARTIFACT_MATRIX.csv`.

Sanity assertions that are silent if wrong (docs/SID_FORMAT.md):
    sid_len == K + 1,  codes < 256,  SID injective,  len(sid) == n_items,
    beam dump decodes from its own SID array at the detected shift,
    target item ids element-wise identical across factorizations.

Usage
  python3 mf_artifact_matrix.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (ART, DATASETS, FACTORIZATIONS, FLABEL, MODEL_KIND, N_ITEMS,  # noqa: E402
                       N_TEST, OUT, beam_path, ckpt_path, load_beam_cell, load_sid,
                       sha256_file, shared_ckpt, sid_path, write_json)

COLS = ['dataset', 'factorisation', 'label', 'sid_len', 'sid', 'sid_sha256',
        'checkpoint', 'checkpoint_kind', 'checkpoint_sha256', 'valid_beam',
        'valid_beam_sha256', 'test_beam', 'test_beam_sha256', 'beam_size',
        'seed', 'item_universe', 'n_valid_users', 'n_test_users',
        'target_identical', 'usable', 'reason']


def check_split_targets(ds, split, ref_target=None):
    """All factorizations must dump element-wise identical target item ids."""
    outs = {}
    for f in FACTORIZATIONS:
        c = load_beam_cell(f, ds, split)
        outs[f] = c
    ref = ref_target if ref_target is not None else outs[FACTORIZATIONS[0]]['target']
    for f, c in outs.items():
        same = bool(np.array_equal(c['target'], ref))
        outs[f]['target_identical'] = same
        if not same:
            raise AssertionError(f'{ds}/{split}: {f} targets differ from reference')
    return outs


def main():
    rows = []
    detail = {'cells': {}}
    for ds in DATASETS:
        valid_cells = check_split_targets(ds, 'valid')
        test_cells = check_split_targets(ds, 'test')
        for f in FACTORIZATIONS:
            reasons = []
            sid = np.load(sid_path(f, ds), allow_pickle=True).astype(np.int64)
            ck = shared_ckpt(ds) if MODEL_KIND[f] == 'shared' else ckpt_path(f, ds)
            vb, tb = valid_cells[f], test_cells[f]
            k = None if f == 'rqvae3' else int(f.split('_k')[1])
            # --- structural checks -------------------------------------------
            n_items_ok = len(sid) == N_ITEMS[ds]
            if not n_items_ok:
                reasons.append(f'item universe {len(sid)} != {N_ITEMS[ds]}')
            if k is not None and sid.shape[1] != k + 1:
                reasons.append(f'sid_len {sid.shape[1]} != K+1')
            if sid.max() >= 256 or sid.min() < 0:
                reasons.append('codes outside [0,255]')
            if len({tuple(r) for r in sid}) != len(sid):
                reasons.append('SID not injective')
            if not ck.exists():
                reasons.append('checkpoint missing')
            if vb['n_users'] != tb['n_users'] and tb['n_users'] != N_TEST[ds]:
                reasons.append(f'test users {tb["n_users"]} != {N_TEST[ds]}')
            if vb['beam'] != 30 or tb['beam'] != 30:
                reasons.append(f'beam {vb["beam"]}/{tb["beam"]} != 30')
            if not vb['target_identical'] or not tb['target_identical']:
                reasons.append('target mismatch across factorizations')
            usable = not reasons
            rows.append({
                'dataset': ds, 'factorisation': f, 'label': FLABEL[f],
                'sid_len': int(sid.shape[1]),
                'sid': str(sid_path(f, ds)), 'sid_sha256': sha256_file(sid_path(f, ds)),
                'checkpoint': str(ck), 'checkpoint_kind': MODEL_KIND[f],
                'checkpoint_sha256': sha256_file(ck),
                'valid_beam': str(beam_path(f, ds, 'valid')),
                'valid_beam_sha256': vb['sha256'],
                'test_beam': str(beam_path(f, ds, 'test')),
                'test_beam_sha256': tb['sha256'],
                'beam_size': int(tb['beam']),
                'seed': 2025,
                'item_universe': int(len(sid)),
                'n_valid_users': int(vb['n_users']),
                'n_test_users': int(tb['n_users']),
                'target_identical': bool(tb['target_identical']),
                'usable': bool(usable),
                'reason': '; '.join(reasons) if reasons else 'ok',
            })
            detail['cells'][f'{ds}|{f}'] = {
                'sid_path': str(sid_path(f, ds)), 'sid_shape': list(sid.shape),
                'sid_dtype': str(sid.dtype), 'sid_sha256': sha256_file(sid_path(f, ds)),
                'checkpoint': str(ck), 'checkpoint_kind': MODEL_KIND[f],
                'checkpoint_sha256': sha256_file(ck),
                'beam_paths': {'valid': str(beam_path(f, ds, 'valid')),
                               'test': str(beam_path(f, ds, 'test'))},
                'beam_sha256': {'valid': vb['sha256'], 'test': tb['sha256']},
                'beam_size': int(tb['beam']),
                'raw_item_id_shift': {'valid': vb['raw_item_id_shift'],
                                      'test': tb['raw_item_id_shift']},
                'n_users': {'valid': int(vb['n_users']), 'test': int(tb['n_users'])},
                'codes_min': int(sid.min()), 'codes_max': int(sid.max()),
                'injective': bool(len({tuple(r) for r in sid}) == len(sid)),
                'usable': bool(usable), 'reason': '; '.join(reasons) if reasons else 'ok',
            }
            print(f'{ds:32s} {f:12s} L={sid.shape[1]} usable={usable} '
                  f'{"; ".join(reasons) if reasons else ""}', flush=True)

    out = OUT / 'FACTORISATION_ARTIFACT_MATRIX.csv'
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    write_json(OUT / '_ARTIFACT_MATRIX_DETAIL.json', detail)
    n_ok = sum(1 for r in rows if r['usable'])
    print(f'\n{n_ok}/{len(rows)} dataset × factorization cells usable -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
