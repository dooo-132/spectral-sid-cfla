#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE D — candidate-level teacher-forced NLL for every candidate of a pool.

For a pool P (spectral4 or multi5) and every factorization F in the five-factor
primary pool, compute

    TotalNLL_F(u, i)   for every i in C_P(u), even when i is NOT in F's beam.

Candidate likelihood is not the same thing as retrieval presence (prompt section
22), so a candidate absent from F's beam still gets a real NLL under F.

Reuse (prompt section 19): the spectral_k3 and rqvae3 candidate NLL already exists
in `outputs/candidate_conditioned/` for the union C_K3 ∪ C_RQVAE.  Those values are
read from disk and merged in AFTER a provenance + agreement audit
(`mf_reuse_audit.py`); they are never recomputed.

Batching (prompt section 23): candidates are grouped by user and processed in
fixed-size chunks of candidate sequences inside one forward pass.  The chunk size
is a PREDECLARED constant of this run, not something that adapts to memory
pressure; on OOM the process stops and writes OOM_STOP_REPORT.md (prompt section 1).

Usage
  python3 mf_candidate_nll.py --split valid --pool spectral4 --datasets Beauty
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (CC, DATASETS, FACTORIZATIONS, FLABEL, REUSE_F, S4, M5,  # noqa: E402
                       OUT, sha256_file, sid_path, load_sid, sid_tokens_for_rows,
                       write_json)
from mf_models import (build_history_tokens, candidate_total_nll,  # noqa: E402
                       encoder_input_for, load_model_for)

# PREDECLARED constants.  Not tuned and never adapted at runtime.
#
# REUSE_F comes from mf_common: spectral_k3 and rqvae3 candidate NLL already exists
# in the frozen previous-round artifact and is reused wherever the (user, candidate)
# pair is covered; only the uncovered remainder is computed.
#
# CONFIG REVISION 1 (recorded in OOM_PREFLIGHT.md and _CONFIG_REVISIONS.json):
# the first declaration was CAND_CHUNK = 2048, which failed the section-18
# preflight outright with `torch.OutOfMemoryError` during the very first forward
# pass.  No scoring run existed at that point and no metric had been computed, so
# the constant was re-declared once, BEFORE any scoring, from the only documented
# safe operating point this project has: the previous round scored candidates in
# user batches of 256 with ~100 candidates/user (25k sequences) at a torch peak of
# ~17.8 GiB.  512 sequences is ~50x smaller than the failed declaration and ~5x
# smaller than that documented point.  It is declared here and then held fixed;
# if it OOMs the run stops (section 1).
CAND_CHUNK = 512           # candidate sequences per forward pass
BATCH_USERS = 128          # user rows per encoder pass
CONFIG_REVISION = 1
CC_RUN = CC / 'runs' / 'multi_candidate_nll_seed2025_20260914T194657'

POOLS = {'spectral4': S4, 'multi5': M5}


def union_dir(split):
    return OUT / 'validation' if split == 'valid' else OUT / 'test'


def load_union(ds, split, pool):
    import pandas as pd
    p = union_dir(split) / f'{ds}_{pool}_candidate_union.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    return df, p


def load_reuse(ds, split):
    """The frozen C_K3 ∪ C_RQVAE union and its NLL, keyed by (user, candidate)."""
    import pandas as pd
    d = CC_RUN / ('valid' if split == 'valid' else 'test')
    u = pd.read_parquet(d / f'{ds}_candidate_union.parquet')
    n = pd.read_parquet(d / f'{ds}_candidate_nll.parquet')
    if not (len(u) == len(n) and (u.user.values == n.user.values).all()
            and (u.candidate_item.values == n.candidate_item.values).all()):
        raise AssertionError(f'{ds}/{split}: frozen union and NLL are not row-aligned')
    key = u.user.values.astype(np.int64) * (10 ** 9) + u.candidate_item.values
    return key, n


@np.errstate(all='ignore')
def score_pool(ds, split, pool, device='cuda', chunk=CAND_CHUNK, verbose=True,
               borrow_s4=False):
    df, upath = load_union(ds, split, pool)
    f_list = POOLS[pool]
    users = df.user.values.astype(np.int64)
    cands = df.candidate_item.values.astype(np.int64)   # 1-based canonical item ids
    n_pairs = len(df)
    order = np.lexsort((cands, users))
    u_sorted = users[order]
    c_sorted = cands[order]
    uniq_users, starts = np.unique(u_sorted, return_index=True)
    ends = np.append(starts[1:], n_pairs)
    n_users = len(uniq_users)

    prov = {'dataset': ds, 'split': split, 'pool': pool, 'n_pairs': int(n_pairs),
            'n_users': int(n_users), 'cand_chunk': int(chunk),
            'union_parquet': str(upath), 'union_sha256': sha256_file(upath),
            'factorizations': list(f_list), 'reused': {}, 'computed': {}}

    scores = {}
    borrowed = None
    cache = {}
    import torch
    skip = set(S4) if (borrow_s4 and pool == 'multi5') else set()
    if borrow_s4 and pool == 'multi5':
        prov['skipped_because_borrowed'] = sorted(skip)
    for f in f_list:
        if f in skip:
            continue
        t0 = time.time()
        bundle = load_model_for(f, ds, device=device, cache=cache)
        sid = load_sid(f, ds)
        # canonical 1-based item id -> SID row index (0-based)
        rows = c_sorted - 1
        if rows.min() < 0 or rows.max() >= len(sid):
            raise AssertionError(
                f'{ds}/{f}: candidate item id out of range: '
                f'min={int(c_sorted.min())} max={int(c_sorted.max())} n_items={len(sid)}')
        if rows.min() < 0 or rows.max() >= len(sid):
            raise AssertionError(f'{ds}/{f}: candidate item id out of range')
        if f in REUSE_F:
            # ---- reuse the frozen values wherever the pair exists ------------
            key_cc, nll_cc = load_reuse(ds, split)
            src_key = users.astype(np.int64) * (10 ** 9) + cands
            lut = dict(zip(key_cc.tolist(), nll_cc[REUSE_F[f]].values.tolist()))
            vals = np.array([lut.get(int(k), np.nan) for k in src_key],
                            dtype=np.float64)
            have = np.isfinite(vals)
            present_col = df[f'present_{FLABEL[f]}'].values.astype(bool)
            n_in_beam_missing = int((present_col & ~have).sum())
            # The frozen artifact's union is C_sharedK3 u C_sharedRQVAE; this round's
            # pool is the INDEPENDENT beams plus the shared RQVAE beam, so gaps are
            # expected and are computed below.  Only the RQVAE column is reused, and
            # it is reused under the same model that produced it.
            prov['reused'][f] = {
                'source': str(CC_RUN / ('valid' if split == 'valid' else 'test') /
                              f'{ds}_candidate_nll.parquet'),
                'source_sha256': sha256_file(
                    CC_RUN / ('valid' if split == 'valid' else 'test') /
                    f'{ds}_candidate_nll.parquet'),
                'column': REUSE_F[f],
                'n_reused': int(have.sum()),
                'n_missing_from_frozen_union': int((~have).sum()),
                'n_missing_that_are_in_beam': n_in_beam_missing,
                'note': ('the frozen artifact union is C_K3 u C_RQVAE of the shared '
                         'beams; candidates generated by the independent spectral '
                         'beams are legitimately absent from it and are computed here'),
                'checkpoint': bundle['checkpoint'],
                'checkpoint_sha256': sha256_file(bundle['checkpoint'])}
            del key_cc, nll_cc, lut
            if have.all():
                scores[f] = vals
                if verbose:
                    print(f'  {f:12s} REUSED all {n_pairs} values from '
                          f'candidate_conditioned ({time.time()-t0:.1f}s)', flush=True)
                continue
            # ---- compute only the candidates absent from the frozen union ----
            todo = np.where(~have)[0]
            h, m = build_history_tokens(ds, split, sid_path(f, ds))
            h, m = encoder_input_for(bundle, f, h, m)
            toks = sid_tokens_for_rows(sid, rows[todo])
            owner_todo = np.repeat(np.arange(n_users), ends - starts)[todo]
            nll_new = candidate_total_nll(bundle['model'], h, m, toks, owner_todo,
                                          device=device, batch_size=chunk)
            vals[todo] = nll_new
            del toks, nll_new
            scores[f] = vals
            prov['reused'][f]['n_computed_to_fill_gap'] = int(len(todo))
            prov['reused'][f]['computed_mean_nll'] = float(np.nanmean(vals[todo]))
            if verbose:
                print(f'  {f:12s} REUSED {int(have.sum())} + computed {len(todo)} '
                      f'to cover the wider union ({time.time()-t0:.1f}s)', flush=True)
            continue

        h, m = build_history_tokens(ds, split, sid_path(f, ds))
        h, m = encoder_input_for(bundle, f, h, m)
        if len(h) != n_users:
            raise AssertionError(f'{ds}/{f}: history rows {len(h)} != users {n_users}')
        out = np.full(n_pairs, np.nan, dtype=np.float64)
        model = bundle['model']
        # One single batched call per factorization: the scorer itself blocks the
        # candidate stream at the predeclared `chunk` size, so the peak does not
        # depend on this user grouping.  `batch_users` only bounds host-side
        # preparation and the encoder rows alive at once.
        owner_all = np.repeat(np.arange(n_users), ends - starts)
        toks = sid_tokens_for_rows(sid, rows)
        nll = candidate_total_nll(model, h, m, toks, owner_all,
                                  device=device, batch_size=chunk)
        out[:] = nll
        del toks
        scores[f] = out
        fin = np.isfinite(out)
        prov['computed'][f] = {
            'checkpoint': bundle['checkpoint'],
            'checkpoint_sha256': sha256_file(bundle['checkpoint']),
            'sid': str(sid_path(f, ds)), 'sid_sha256': sha256_file(sid_path(f, ds)),
            'finite_fraction': float(fin.mean()),
            'mean_nll': float(out[fin].mean()) if fin.any() else None,
            'std_nll': float(out[fin].std()) if fin.any() else None,
            'elapsed_s': round(time.time() - t0, 1)}
        if verbose:
            print(f'  {f:12s} computed {n_pairs} NLLs  mean={out[fin].mean():.4f} '
                  f'finite={fin.mean():.6f}  ({time.time()-t0:.1f}s)', flush=True)
        if f not in ('spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5',
                     'rqvae3'):
            raise AssertionError(f)
    return df, order, scores, prov, borrowed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--pool', default='spectral4', choices=list(POOLS))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--chunk', type=int, default=CAND_CHUNK)
    ap.add_argument('--borrow-from-spectral4', action='store_true',
                    help='for pool=multi5: take the K2/K3/K4/K5 NLL columns from the '
                         'already written spectral4 file (the row sets are identical by '
                         'construction) and only compute the RQVAE gap')
    a = ap.parse_args(argv)

    outdir = Path(a.outdir) if a.outdir else union_dir(a.split)
    outdir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    allprov = {'split': a.split, 'pool': a.pool, 'cand_chunk': a.chunk,
               'reuse_run': str(CC_RUN), 'cells': {}}
    for ds in a.datasets.split(','):
        print(f'=== {ds} {a.split} {a.pool} ===', flush=True)
        t0 = time.time()
        df, order, scores, prov, borrowed = score_pool(
            ds, a.split, a.pool, device=a.device, chunk=a.chunk,
            borrow_s4=a.borrow_from_spectral4)
        # every column is written in the union's own row order; the NLL arrays are
        # re-sorted back from the lexsort order into that same row order.
        cols = {'dataset': df.dataset.values, 'split': df.split.values,
                'user': df.user.values, 'user_row': df.user_row.values,
                'candidate_item': df.candidate_item.values}
        borrowed = None
        borrow_need = None
        if a.borrow_from_spectral4 and a.pool == 'multi5':
            import pandas as _pd
            sp = outdir / f'{ds}_spectral4_candidate_nll.parquet'
            if not sp.exists():
                raise FileNotFoundError(
                    f'{sp} missing: run --pool spectral4 first')
            s4 = _pd.read_parquet(sp)
            k_new = df.user.values.astype(np.int64) * (10 ** 9) + df.candidate_item.values
            k_s4 = s4.user.values.astype(np.int64) * (10 ** 9) + s4.candidate_item.values
            lut = {int(a_): i for i, a_ in enumerate(k_s4)}
            # The multi5 pool is a SUPERSET of the spectral4 pool (it adds the
            # candidates only RQVAE's beam generated), so the four spectral columns
            # are borrowed wherever the pair exists and computed for the remainder.
            hit = np.array([lut.get(int(x), -1) for x in k_new], dtype=np.int64)
            have = hit >= 0
            safe = np.where(have, hit, 0)
            borrowed = {}
            for f in S4:
                for prefix in ('TotalNLL', 'sid_len', 'present', 'prob', 'rank'):
                    col = f'{prefix}_{FLABEL[f]}'
                    borrowed[col] = s4[col].values[safe]
            borrow_need = np.where(~have)[0]
            allprov['cells'].setdefault(ds, prov)['borrowed_from_spectral4'] = {
                'path': str(sp), 'sha256': sha256_file(sp), 'n_rows': int(len(s4)),
                'n_rows_borrowed': int(have.sum()),
                'n_rows_computed_because_rqvae_only': int(len(borrow_need)),
                'columns': [f'TotalNLL_{FLABEL[f]}' for f in S4]}
            print(f'  spectral4 borrow: {int(have.sum())} rows taken, '
                  f'{len(borrow_need)} RQVAE-only rows need all four spectral NLLs',
                  flush=True)
        if borrow_need is not None and len(borrow_need):
            # score the RQVAE-only candidates under the four spectral models
            import torch as _t
            need_rows = (df.candidate_item.values[borrow_need] - 1)
            cache = {}
            for f in S4:
                bundle = load_model_for(f, ds, device=a.device, cache=cache)
                sid = load_sid(f, ds)
                h, m = build_history_tokens(ds, a.split, sid_path(f, ds))
                h, m = encoder_input_for(bundle, f, h, m)
                toks = sid_tokens_for_rows(sid, need_rows)
                owners = df.user_row.values[borrow_need].astype(np.int64)
                v = candidate_total_nll(bundle['model'], h, m, toks, owners,
                                        device=a.device, batch_size=a.chunk)
                borrowed[f'TotalNLL_{FLABEL[f]}'][borrow_need] = v
                del toks
                _t.cuda.empty_cache()

        for f in POOLS[a.pool]:
            for prefix in ('TotalNLL', 'sid_len', 'present', 'prob', 'rank'):
                col = f'{prefix}_{FLABEL[f]}'
                if borrowed is not None and col in borrowed:
                    v = borrowed[col]
                    if prefix == 'TotalNLL' and not np.isfinite(v).all():
                        raise AssertionError(f'{ds}/{f}: borrowed NLL non-finite')
                    cols[col] = v
                elif prefix == 'TotalNLL':
                    cols[col] = scores[f][order]
                elif prefix == 'sid_len':
                    cols[col] = np.int64(load_sid(f, ds).shape[1])
                else:
                    cols[col] = df[col].values
        out = pd.DataFrame(cols)
        p = outdir / f'{ds}_{a.pool}_candidate_nll.parquet'
        out.to_parquet(p, index=False)
        prov['nll_parquet'] = str(p)
        prov['nll_sha256'] = sha256_file(p)
        prov['elapsed_s_total'] = round(time.time() - t0, 1)
        allprov['cells'][ds] = prov
        print(f'  -> {p.name} ({time.time()-t0:.1f}s)', flush=True)
        del df, out, scores
        import torch
        torch.cuda.empty_cache()
    write_json(outdir / f'_{a.split.upper()}_{a.pool.upper()}_NLL_PROVENANCE.json',
               allprov)
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
