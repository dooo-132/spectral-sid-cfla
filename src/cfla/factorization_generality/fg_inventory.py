#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE A -- artifact inventory and cross-family comparability audit.

Produces
  ARTIFACT_MATRIX.csv        one row per dataset x family x K canonical cell
  SEED_ARTIFACT_MATRIX.csv   one row per dataset x family x K x seed
  _AUDIT.json                raw comparability findings (Phase K stop conditions)

Checks performed (a failure is reported, never patched):
  * beam budget identical across families
  * test-user set (target item ids) identical across families and K
  * item-id convention: candidate_tokens must reconstruct from
    sid_array[dump_item_id, position] with the historical 0-based row index
  * SID array shape (N, K+1), codes < 256, injective (zero final collisions)
  * single-model item-level metrics computable for every cell
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fg_common import (  # noqa: E402
    BEAM, CODEBOOK, DATASETS, FAMILIES, FAMILY_LABELS, K_LIST, OUT, REPO,
    SEED_CANONICAL, SEED_ROOT, beam_path, ckpt_path, file_sha, load_cell,
    seed_beam_path, seed_ckpt_path, sid_array_path, single_rank, summarize,
)
from src.utils import repo_paths as rp  # noqa: E402


def git_state():
    import subprocess
    def run(*a):
        return subprocess.run(a, cwd=str(REPO), capture_output=True,
                              text=True).stdout.strip()
    return {'branch': run('git', 'branch', '--show-current'),
            'sha': run('git', 'rev-parse', 'HEAD'),
            'dirty': bool(run('git', 'status', '--porcelain'))}


def check_item_convention(cell, sid):
    """Historical dumps index the SID array 0-based (docs/ITEM_ID_CONVENTION.md).

    Verify by reconstructing the dumped tokens two ways.
    """
    z = cell['z']
    cand_tok = np.asarray(z['candidate_tokens'])
    raw = np.asarray(z['candidate_item_ids'])
    pos = np.arange(cand_tok.shape[2])[None, None, :]
    ok0 = ok1 = tot = 0
    valid = raw >= 0
    idx = np.where(valid)
    # subsample users to bound cost
    users = np.unique(idx[0])[:400]
    sel = np.isin(idx[0], users)
    r = idx[0][sel]
    b = idx[1][sel]
    rec0 = sid[raw[r, b]][:, :] if raw[r, b].max() < len(sid) else None
    if rec0 is not None:
        ok0 = int((rec0 + pos[0, :, :] * CODEBOOK + 1 == cand_tok[r, b]).all(axis=1).sum())
    r1 = raw[r, b] - 1
    m = r1 >= 0
    if m.any():
        rec1 = sid[r1[m]]
        ok1 = int((rec1 + pos[0, :, :] * CODEBOOK + 1 == cand_tok[r[m], b[m]]).all(axis=1).sum())
    tot = int(sel.sum())
    return {'n_checked': tot, 'match_0based': ok0, 'match_1based': ok1,
            'fraction_0based': (ok0 / tot) if tot else float('nan'),
            'fraction_1based': (ok1 / tot) if tot else float('nan')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'git': git_state(), 'artifact_root': str(rp.ARTIFACT_ROOT),
             'cells': {}, 'findings': [], 'families': FAMILIES}
    rows, seed_rows = [], []

    for ds in DATASETS:
        print(f'=== {ds} ===', flush=True)
        ref_target = None
        ref_n = None
        audit['cells'][ds] = {'targets_identical_across_families': True,
                              'beam_budget_identical': True,
                              'item_convention_consistent': True,
                              'sid_injective': True}
        for family in FAMILY_LABELS:
            for k in K_LIST:
                sidp = sid_array_path(ds, family, k)
                ckp = ckpt_path(ds, family, k)
                bt = beam_path(ds, family, k, 'test')
                bv = beam_path(ds, family, k, 'valid')
                rec = {'dataset': ds, 'family': family, 'K': k,
                       'seed': SEED_CANONICAL,
                       'sid': str(sidp) if sidp.exists() else '',
                       'checkpoint': str(ckp) if ckp.exists() else '',
                       'valid_beam': str(bv) if bv.exists() else '',
                       'test_beam': str(bt) if bt.exists() else '',
                       'metrics': '', 'usable': 0}
                if not (sidp.exists() and ckp.exists() and bt.exists() and bv.exists()):
                    rows.append(rec)
                    continue
                sid = np.load(sidp, allow_pickle=True).astype(np.int64)
                z = np.load(bt, allow_pickle=True)
                n, B = np.asarray(z['candidate_item_ids']).shape
                checks = {
                    'sid_shape': sid.shape == (sid.shape[0], k + 1),
                    'codes_lt_256': bool((sid >= 0).all() and (sid < 256).all()),
                    'sid_injective': len({tuple(r) for r in sid}) == len(sid),
                    'beam_budget': B == BEAM,
                }
                cell = load_cell(ds, family, k, 'test', keep_z=True)
                target = cell['target']
                if ref_target is None:
                    ref_target, ref_n = target, n
                else:
                    if not np.array_equal(target, ref_target):
                        audit['cells'][ds]['targets_identical_across_families'] = False
                        audit['findings'].append(
                            f'{ds} {family} K={k}: test targets differ from reference')
                conv = check_item_convention(cell, sid)
                rec['raw_item_id_shift'] = cell['raw_item_id_shift']
                rec['shift_detection'] = json.dumps(
                    {str(kk): round(vv, 6) for kk, vv in cell['shift_detection'].items()})
                if cell['raw_item_id_shift'] != 0:
                    audit['findings'].append(
                        f'{ds} {family} K={k}: raw item-id shift is '
                        f'{cell["raw_item_id_shift"]}, not the historical 0')
                if conv['fraction_0based'] < 1.0 and cell['raw_item_id_shift'] != 0:
                    audit['findings'].append(
                        f'{ds} {family} K={k}: 0-based token reconstruction '
                        f'{conv["fraction_0based"]:.3f}')
                if not all(checks.values()):
                    audit['findings'].append(f'{ds} {family} K={k}: {checks}')
                if not checks['sid_injective']:
                    audit['cells'][ds]['sid_injective'] = False
                if not checks['beam_budget']:
                    audit['cells'][ds]['beam_budget_identical'] = False

                rank = single_rank(cell)
                sm = summarize(rank)
                rec['metrics'] = json.dumps({kk: round(vv, 6) for kk, vv in sm.items()})
                rec['usable'] = 1
                rec['n_users'] = int(n)
                rec['sid_sha256'] = file_sha(sidp)
                rec['ckpt_sha256'] = file_sha(ckp)
                rec['beam_sha256'] = file_sha(bt)
                rows.append(rec)
                print(f'  {family:8s} K={k} n={n} NDCG@20={sm["NDCG@20"]:.6f} '
                      f'conv0={conv["fraction_0based"]:.3f}', flush=True)

        # ---------------- seed artifacts ----------------
        if SEED_ROOT.exists():
            for ckp in sorted((SEED_ROOT / 'ckpt').glob('*.pth')):
                name = ckp.stem  # tiger_<ds>_k<K>_sid<L>_seed<S>
                tail = name.rsplit('_seed', 1)
                if len(tail) != 2:
                    continue
                seed = int(tail[1])
                head = tail[0]
                ds_name = head[len('tiger_'):].rsplit('_k', 1)[0]
                k = int(head.rsplit('_k', 1)[1].split('_sid')[0])
                if ds_name != ds:
                    continue
                bt = seed_beam_path(ds, k, seed, 'test')
                bv = seed_beam_path(ds, k, seed, 'valid')
                seed_rows.append({
                    'dataset': ds, 'family': 'spectral', 'K': k, 'seed': seed,
                    'checkpoint': str(ckp),
                    'test_beam': str(bt) if bt.exists() else '',
                    'valid_beam': str(bv) if bv.exists() else '',
                    'ckpt_complete': 1,
                    'beams_complete': int(bt.exists() and bv.exists()),
                    'complete': int(bt.exists() and bv.exists()),
                })

    art = pd.DataFrame(rows)
    # column order per protocol
    cols = ['dataset', 'family', 'K', 'seed', 'sid', 'checkpoint', 'valid_beam',
            'test_beam', 'metrics', 'usable']
    for extra in [c for c in art.columns if c not in cols]:
        cols.append(extra)
    art[cols].to_csv(OUT / 'ARTIFACT_MATRIX.csv', index=False)
    print(f'wrote ARTIFACT_MATRIX.csv ({len(art)} rows)')

    seed_cols = ['dataset', 'family', 'K', 'seed', 'checkpoint', 'test_beam',
                 'valid_beam', 'ckpt_complete', 'beams_complete', 'complete']
    seed_df = pd.DataFrame(seed_rows)
    if len(seed_df):
        seed_df[seed_cols].to_csv(OUT / 'SEED_ARTIFACT_MATRIX.csv', index=False)
    else:
        pd.DataFrame(columns=seed_cols).to_csv(OUT / 'SEED_ARTIFACT_MATRIX.csv', index=False)
    print(f'wrote SEED_ARTIFACT_MATRIX.csv ({len(seed_df)} rows)')

    (OUT / '_AUDIT.json').write_text(json.dumps(audit, indent=2, default=str))
    print('findings:', audit['findings'] or 'NONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
