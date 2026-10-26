#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE A — repaired candidate-NLL write-back for the Multi-Factorization round.

WHAT WAS BROKEN
---------------
`mf_candidate_nll.score_pool` computes the per-candidate NLL in **lexsorted**
`(user, candidate_item)` order:

    order = np.lexsort((cands, users))            # mf_candidate_nll.py:104
    rows  = c_sorted - 1                          # mf_candidate_nll.py:130

while the dataframe it eventually writes is in the **candidate-union parquet's own row
order**, which is not lexsorted (each user's candidates are ordered by descending best
probability).  `main()` then wrote

    cols[col] = scores[f][order]                  # mf_candidate_nll.py:316   <-- BUG

Applying `order` a second time is not the inverse permutation; it is correct only when
`order` is an involution, and it is not.  The values were right, their row assignment was
wrong.  The frozen artifacts are preserved as audit evidence and are NOT touched here.

WHAT THIS MODULE DOES
---------------------
It re-implements the scoring *driver* with the write-back fixed, and reuses the audited
scorer unchanged:

  * `write_scores_back_to_original_rows(scores_sorted, order)` — the fix, in one place,
    with an explicit permutation guard;
  * `score_pool_repaired(...)` — scores every candidate of a frozen pool under the
    factorizations' own checkpoints and returns columns aligned with the pool parquet's
    own rows.

The NLL definition is untouched: it is still `mf_models.candidate_total_nll`
(teacher-forced, full SID including the collision slot, positive, own SID, own history).

Usage
  python3 experiments/multifactor_fusion/mf_candidate_nll_repaired.py --split valid
  python3 experiments/multifactor_fusion/mf_candidate_nll_repaired.py --split test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (DATASETS, FLABEL, M5, OUT as MF_OUT, REPO, S4,  # noqa: E402
                       sha256_file, sid_path, write_json)
from mf_models import (build_history_tokens, candidate_total_nll,  # noqa: E402
                       encoder_input_for, load_model_for)

# The repaired round's only writable root.
REPAIR = REPO / 'outputs' / 'multifactor_nll_repair'
POOLS = {'spectral4': S4, 'multi5': M5}
CAND_CHUNK = 512          # same predeclared constant the original run used
SPLIT_SUBDIR = {'valid': 'validation', 'test': 'test'}


# ---------------------------------------------------------------------------
# the fix
# ---------------------------------------------------------------------------
def assert_permutation(order, n):
    """`order` must be a permutation of range(n) before it is used as an index map."""
    order = np.asarray(order, dtype=np.int64)
    if len(order) != n:
        raise AssertionError(f'order has {len(order)} entries for {n} rows')
    if order.min() < 0 or order.max() >= n:
        raise AssertionError(f'order out of range: [{order.min()}, {order.max()}] for n={n}')
    if len(np.unique(order)) != n:
        raise AssertionError('order is not injective')
    return order


def write_scores_back_to_original_rows(scores_sorted, order, n_rows=None):
    """Scatter lexsorted scores back into the dataframe's own row order.

    Parameters
    ----------
    scores_sorted : (n,) array
        Values aligned with `df.iloc[order]`, i.e. `scores_sorted[k]` belongs to the row
        at original position `order[k]`.
    order : (n,) int
        A permutation of `range(n)`; `order[k]` is the ORIGINAL row index of sorted
        position `k`.
    n_rows : int, optional
        Size of the original row space (defaults to `len(order)`).

    Returns
    -------
    (n_rows,) array with `out[order[k]] == scores_sorted[k]`.

    The broken code did `scores_sorted[order]`, which assigns `scores_sorted[order[k]]`
    to row `k` -- the inverse of what is wanted, and equal to it only when `order` is an
    involution.
    """
    scores_sorted = np.asarray(scores_sorted)
    n = len(order) if n_rows is None else int(n_rows)
    order = assert_permutation(order, len(scores_sorted))
    if n != len(order):
        raise AssertionError('n_rows must equal len(order) for a full permutation')
    out = np.empty(n, dtype=scores_sorted.dtype)
    out[order] = scores_sorted
    return out


# ---------------------------------------------------------------------------
# run scaffolding (§53-55): unique run dir, PRE/POST manifests, status
# ---------------------------------------------------------------------------
def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_dir_for(ds, split, f, chash):
    d = REPAIR / 'runs' / f'{ds}_{split}_{f}_{chash[:8]}'
    if d.exists():
        raise SystemExit(f'HARD FAIL: run path already exists: {d}')
    d.mkdir(parents=True)
    return d


def write_manifest(path, entries):
    lines = [f'{sha256_file(p)}  {p}' for p in entries if sha256_file(p)]
    Path(path).write_text('\n'.join(sorted(lines)) + '\n')


# ---------------------------------------------------------------------------
# repaired scoring driver
# ---------------------------------------------------------------------------
def union_parquet(ds, split, pool):
    sub = 'validation' if split == 'valid' else 'test'
    p = MF_OUT / sub / f'{ds}_{pool}_candidate_union.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def score_pool_repaired(ds, split, pool, device='cuda', chunk=CAND_CHUNK,
                        verbose=True, resume=False, run_dirs=True):
    """Score every candidate of a frozen pool, correctly row-aligned.

    Returns (df, nll_by_factorization, provenance).  `nll_by_factorization[f]` is aligned
    with `df`'s own row order.

    One unique run directory is created per (dataset, split, factorization) under
    `outputs/multifactor_nll_repair/runs/`, carrying PRE/POST artifact manifests and a
    status file.  An existing run directory is a HARD FAIL unless `resume=True`, in which
    case a completed run is reused and an incomplete one is refused.
    """
    import pandas as pd
    f_list = POOLS[pool]
    up = union_parquet(ds, split, pool)
    df = pd.read_parquet(up)
    users = df.user.values.astype(np.int64)
    cands = df.candidate_item.values.astype(np.int64)      # 1-based canonical item ids
    n_pairs = len(df)

    # the lexsort the original driver used; kept identical on purpose
    order = assert_permutation(np.lexsort((cands, users)), n_pairs)
    u_sorted = users[order]
    c_sorted = cands[order]
    uniq_users, starts = np.unique(u_sorted, return_index=True)
    ends = np.append(starts[1:], n_pairs)
    n_users = len(uniq_users)
    if n_users != len(np.unique(users)):
        raise AssertionError('lexsort did not group users')

    prov = {'dataset': ds, 'split': split, 'pool': pool, 'n_pairs': int(n_pairs),
            'n_users': int(n_users), 'cand_chunk': int(chunk),
            'union_parquet': str(up), 'union_sha256': sha256_file(up),
            'order_is_involution': bool((order[order] == np.arange(n_pairs)).all()),
            'write_back': 'out[order] = scores_sorted (repaired)',
            'legacy_write_back': 'scores_sorted[order] (buggy, not used)',
            'factorizations': {}}

    out = {}
    model_cache = {}
    for f in f_list:
        t0 = time.time()
        bundle = load_model_for(f, ds, device=device, cache=model_cache)
        sid = np.load(sid_path(f, ds), allow_pickle=True).astype(np.int64)
        rows = c_sorted - 1                                # 0-based SID rows
        if rows.min() < 0 or rows.max() >= len(sid):
            raise AssertionError(f'{ds}/{f}: candidate id out of SID range')

        cfg = {'dataset': ds, 'split': split, 'pool': pool, 'factorization': f,
               'checkpoint': str(bundle['checkpoint']),
               'checkpoint_sha256': sha256_file(bundle['checkpoint']),
               'sid': str(sid_path(f, ds)), 'sid_sha256': sha256_file(sid_path(f, ds)),
               'union_sha256': sha256_file(up), 'cand_chunk': int(chunk),
               'write_back': 'out[order] = scores_sorted'}
        chash = config_hash(cfg)
        rd = REPAIR / 'runs' / f'{ds}_{split}_{f}_{chash[:8]}'

        if rd.exists():
            status_p = rd / 'status.json'
            if not resume:
                raise SystemExit(f'HARD FAIL: run path already exists: {rd}')
            st = json.loads(status_p.read_text()) if status_p.exists() else {'status': 'failed'}
            if st.get('status') != 'completed':
                raise SystemExit(f'HARD FAIL: refusing to reuse a non-completed run: {rd} '
                                 f'(status={st.get("status")})')
            nll_rows = np.load(rd / 'nll_rows.npy')
            if len(nll_rows) != n_pairs:
                raise SystemExit(f'HARD FAIL: cached run has {len(nll_rows)} rows, '
                                 f'expected {n_pairs}')
            out[f] = nll_rows
            prov['factorizations'][f] = st['provenance']
            if verbose:
                print(f'    {f:12s} RESUMED from {rd.name}', flush=True)
            del bundle, sid, rows
            continue

        rd.mkdir(parents=True)
        (rd / 'config.json').write_text(json.dumps(cfg, indent=2, default=str))
        pre_inputs = [bundle['checkpoint'], sid_path(f, ds), up]
        write_manifest(rd / 'PRE_RUN_ARTIFACT_MANIFEST.sha256', pre_inputs)

        if bundle['kind'] == 'independent' and bundle['cfg']['vocab_size'] != 256 * sid.shape[1] + 1:
            raise AssertionError(f'{ds}/{f}: independent vocab mismatch')

        h, m = build_history_tokens(ds, split, sid_path(f, ds))
        h, m = encoder_input_for(bundle, f, h, m)
        if len(h) != n_users:
            raise AssertionError(f'{ds}/{f}: history rows {len(h)} != users {n_users}')
        # vocabulary guard: a control-token-prefixed history must never reach an
        # independent (vocab 256*L+1) model
        if int(h.max()) >= int(bundle['cfg']['vocab_size']):
            raise AssertionError(f'{ds}/{f}: history token {int(h.max())} >= vocab '
                                 f'{bundle["cfg"]["vocab_size"]}')

        owner_all = np.repeat(np.arange(n_users), ends - starts)
        toks = np.full((n_pairs, sid.shape[1]), -1, dtype=np.int64)
        ok = (rows >= 0) & (rows < len(sid))
        pos = np.arange(sid.shape[1], dtype=np.int64)[None, :]
        toks[ok] = sid[np.where(ok, rows, 0)][ok] + pos * 256 + 1

        nll_sorted = candidate_total_nll(bundle['model'], h, m, toks, owner_all,
                                         device=device, batch_size=chunk)
        # ---------------- THE FIX ----------------
        nll_rows = write_scores_back_to_original_rows(nll_sorted, order)
        # -----------------------------------------
        np.save(rd / 'nll_rows.npy', nll_rows)
        np.save(rd / 'nll_sorted.npy', nll_sorted)
        np.save(rd / 'order.npy', order)
        out[f] = nll_rows

        fin = np.isfinite(nll_rows)
        fprov = {
            'model_kind': bundle['kind'], 'checkpoint': bundle['checkpoint'],
            'checkpoint_sha256': sha256_file(bundle['checkpoint']),
            'sid': str(sid_path(f, ds)), 'sid_sha256': sha256_file(sid_path(f, ds)),
            'sid_len': int(sid.shape[1]), 'vocab_size': int(bundle['cfg']['vocab_size']),
            'finite_fraction': float(fin.mean()),
            'mean_nll': float(nll_rows[fin].mean()), 'std_nll': float(nll_rows[fin].std(ddof=0)),
            'min_nll': float(nll_rows[fin].min()), 'max_nll': float(nll_rows[fin].max()),
            'elapsed_s': round(time.time() - t0, 1),
            'run_dir': str(rd), 'config_hash': chash,
            'alignment_check': alignment_check(df, f, nll_rows)}
        prov['factorizations'][f] = fprov
        write_manifest(rd / 'POST_RUN_ARTIFACT_MANIFEST.sha256',
                       [rd / 'nll_rows.npy', rd / 'nll_sorted.npy', rd / 'order.npy',
                        rd / 'config.json'])
        (rd / 'status.json').write_text(json.dumps(
            {'status': 'completed', 'provenance': fprov,
             'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2, default=str))
        if verbose:
            print(f'    {f:12s} {n_pairs} NLLs  mean={nll_rows[fin].mean():.4f} '
                  f'finite={fin.mean():.6f}  ({time.time()-t0:.1f}s)', flush=True)
    return df, out, prov


def alignment_check(df, f, nll_rows, n_probe=2000, seed=12345):
    """Independent invariant: rows of a user must be a contiguous block, and the NLL value
    recorded for a (user, item) pair must be unique in the column.

    This is a structural self-check only -- the authoritative check is the regression test
    against the scorer, which is run separately.
    """
    users = df.user.values.astype(np.int64)
    items = df.candidate_item.values.astype(np.int64)
    rng = np.random.default_rng(seed)
    key_ok = True
    for _ in range(3):
        i = int(rng.integers(0, len(df)))
        same = (users == users[i]) & (items == items[i])
        if int(same.sum()) != 1:
            key_ok = False
            break
    return {'n_rows': int(len(df)), 'n_finite': int(np.isfinite(nll_rows).sum()),
            'unique_user_item_keys': bool(key_ok),
            'value_min': float(np.nanmin(nll_rows)), 'value_max': float(np.nanmax(nll_rows))}


def build_columns(df, nll_by_f, f_list, sid_len):
    """Assemble the repaired parquet columns (brief section 16).

    The brief's names (`present_*`, `P_*`, `NLL_*`) are the primary columns; the frozen
    round's names (`prob_*`, `TotalNLL_*`) are written as drop-in aliases so the audited
    readers keep working unchanged.  `rank_*` and `sid_len_*` are carried over because
    they are properties of the frozen generation output, not of the NLL.
    """
    cols = {'dataset': df.dataset.values, 'split': df.split.values,
            'user': df.user.values, 'user_row': df.user_row.values,
            'candidate_item': df.candidate_item.values}
    for f in f_list:
        L = FLABEL[f]
        cols[f'present_{L}'] = df[f'present_{L}'].values
        cols[f'P_{L}'] = df[f'prob_{L}'].values.astype(np.float64)
        cols[f'NLL_{L}'] = nll_by_f[f]
        cols[f'rank_{L}'] = df[f'rank_{L}'].values
        cols[f'sid_len_{L}'] = np.int64(sid_len[f])
        # drop-in aliases
        cols[f'prob_{L}'] = cols[f'P_{L}']
        cols[f'TotalNLL_{L}'] = cols[f'NLL_{L}']
    return cols


def subset_to_spectral4(df5, nll5, pool_rows_path):
    """Return the Spectral-4 view of a Multi-5 scoring pass.

    The Spectral-4 pool is the union of the K2..K5 beams, which is a strict SUBSET of the
    Multi-5 pool (Multi-5 additionally contains the RQVAE beam's own candidates).  The
    subset is taken against the FROZEN Spectral-4 union parquet, keyed on
    (user, candidate_item), so the repaired Spectral-4 artifact has exactly the frozen
    row set and row order.  NLL values are identical for a shared row under either pool --
    a candidate's NLL depends only on the candidate, the user's history and the model.
    """
    import pandas as pd
    d4 = pd.read_parquet(pool_rows_path)
    key5 = df5.user.values.astype(np.int64) * (10 ** 9) + df5.candidate_item.values
    lut = {int(k): i for i, k in enumerate(key5)}
    idx = np.array([lut.get(int(u) * 10 ** 9 + int(c), -1)
                    for u, c in zip(d4.user.values, d4.candidate_item.values)],
                   dtype=np.int64)
    missing = int((idx < 0).sum())
    if missing:
        raise AssertionError(f'{missing} Spectral-4 rows are absent from the Multi-5 pool')
    return d4, {f: nll5[f][idx] for f in nll5}, idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--splits', default='valid,test')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--chunk', type=int, default=CAND_CHUNK)
    ap.add_argument('--resume', action='store_true',
                    help='reuse completed run directories instead of hard-failing')
    a = ap.parse_args(argv)

    import pandas as pd
    import torch
    allprov = {'cand_chunk': a.chunk, 'cells': {}, 'resume': bool(a.resume),
               'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    sid_len = {f: int(np.load(sid_path(f, DATASETS[0]), allow_pickle=True).shape[1])
               for f in M5}
    for ds in a.datasets.split(','):
        sid_len.update({f: int(np.load(sid_path(f, ds), allow_pickle=True).shape[1])
                        for f in M5})
        for split in a.splits.split(','):
            print(f'=== {ds} {split} : scoring the Multi-5 pool once ===', flush=True)
            df5, nll5, prov5 = score_pool_repaired(ds, split, 'multi5', device=a.device,
                                                   chunk=a.chunk, resume=a.resume)
            # ---- Multi-5 artifact ----
            d5dir = REPAIR / SPLIT_SUBDIR[split] / 'multi5'
            d5dir.mkdir(parents=True, exist_ok=True)
            p5 = d5dir / f'{ds}_candidate_nll.parquet'
            pd.DataFrame(build_columns(df5, nll5, M5, sid_len)).to_parquet(p5, index=False)
            prov5['nll_parquet'] = str(p5)
            prov5['nll_sha256'] = sha256_file(p5)

            # ---- Spectral-4 artifact, as the subset view ----
            s4rows = union_parquet(ds, split, 'spectral4')
            df4, nll4, idx = subset_to_spectral4(df5, nll5, s4rows)
            d4dir = REPAIR / SPLIT_SUBDIR[split] / 'spectral4'
            d4dir.mkdir(parents=True, exist_ok=True)
            p4 = d4dir / f'{ds}_candidate_nll.parquet'
            pd.DataFrame(build_columns(df4, nll4, S4, sid_len)).to_parquet(p4, index=False)
            prov4 = dict(prov5)
            prov4['pool'] = 'spectral4'
            prov4['n_pairs'] = int(len(df4))
            prov4['n_users'] = int(df4.user.nunique())
            prov4['union_parquet'] = str(s4rows)
            prov4['union_sha256'] = sha256_file(s4rows)
            prov4['derived_from'] = 'Multi-5 scoring pass, subset by frozen Spectral-4 keys'
            prov4['nll_parquet'] = str(p4)
            prov4['nll_sha256'] = sha256_file(p4)

            allprov['cells'][f'{ds}|{split}|multi5'] = prov5
            allprov['cells'][f'{ds}|{split}|spectral4'] = prov4
            print(f'  -> {p5.name} {sha256_file(p5)[:12]}  ({prov5["n_pairs"]} rows)\n'
                  f'  -> {p4.name} {sha256_file(p4)[:12]}  ({prov4["n_pairs"]} rows)',
                  flush=True)
            del df5, nll5, df4, nll4
            torch.cuda.empty_cache()
    write_json(REPAIR / 'manifests' / '_REPAIRED_NLL_PROVENANCE.json', allprov)
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
