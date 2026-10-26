#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE C + D — build the candidate union and score every candidate under both
factorizations, with no target access anywhere in the scoring path.

For each dataset and split:

  1.  Load the frozen shared model's per-factorization candidate sets (already merged
      over beams by the previous round: `cand_ids*/cand_p*` in
      `adaptive_factorization/selector/features/<dataset>_<split>.npz`).
  2.  Build `C_union(u) = C_F1(u) ∪ C_F2(u)` from generation output only.
  3.  Tokenise every candidate under BOTH SID systems and teacher-force it through the
      frozen checkpoint, giving `TotalNLL_F1(u,i)` and `TotalNLL_F2(u,i)`.
  4.  Write `candidate_union.parquet` and `candidate_nll.parquet`.

The scorer receives only (history, candidate ids, factorization).  The target is read
nowhere in this file; the evaluation join happens in a separate script.  Section 16's
prohibition — never add the target to the union — is enforced by construction: the only
inputs are the two candidate arrays and the two SID arrays.

Usage
  python3 cc_candidate_nll.py --datasets Beauty --splits valid --outdir <run dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_common import (DATASETS, FACTORIZATIONS, F1, F2, SCORER_INPUT_FIELDS,  # noqa: E402
                       build_history_bank, candidate_total_nll, config_hash,
                       frozen_checkpoint, load_frozen_shared, sid_tokens_for_items,
                       write_json)
from xt_common import load_sid, sha256_file, sid_path  # noqa: E402

SELECTOR = Path('outputs/adaptive_factorization/selector')


# ---------------------------------------------------------------------------
# union construction (no target anywhere)
# ---------------------------------------------------------------------------
def build_union(cand_ids1, cand_p1, cand_ids2, cand_p2):
    """Return (items (n,k), p1 (n,k), p2 (n,k)) for C_union, plus per-factorization ranks.

    `cand_ids*` are 0-based SID-array rows with -1 padding; `cand_p*` are the merged
    per-item probabilities (0 where absent).  Duplicates are impossible inside one
    factorization's array because it was already merged by item.
    """
    n, k1 = cand_ids1.shape
    k2 = cand_ids2.shape[1]
    kmax = k1 + k2
    items = np.full((n, kmax), -1, dtype=np.int64)
    p1 = np.zeros((n, kmax), dtype=np.float64)
    p2 = np.zeros((n, kmax), dtype=np.float64)

    for i in range(n):
        seen = {}
        for j in range(k1):
            it = int(cand_ids1[i, j])
            if it >= 0:
                seen[it] = [float(cand_p1[i, j]), 0.0]
        for j in range(k2):
            it = int(cand_ids2[i, j])
            if it < 0:
                continue
            if it in seen:
                seen[it][1] = float(cand_p2[i, j])
            else:
                seen[it] = [0.0, float(cand_p2[i, j])]
        # deterministic order: by descending best probability, then item id
        order = sorted(seen.items(), key=lambda kv: (-max(kv[1]), kv[0]))
        for slot, (it, (a, b)) in enumerate(order):
            items[i, slot] = it
            p1[i, slot] = a
            p2[i, slot] = b
    return items, p1, p2


def ranks_from_probs(items, probs):
    """1-based rank of each (user, candidate) inside one factorization; -1 if p == 0."""
    rank = np.full(items.shape, -1, dtype=np.int64)
    for i in range(items.shape[0]):
        row = probs[i]
        nz = np.where(row > 0)[0]
        if len(nz) == 0:
            continue
        # denser probability first; stable tie-break on item id
        order = sorted(nz.tolist(), key=lambda j: (-row[j], int(items[i, j])))
        for r, j in enumerate(order, start=1):
            rank[i, j] = r
    return rank


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--splits', default='valid,test')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-users', type=int, default=256)
    ap.add_argument('--featdir', default=str(SELECTOR / 'features'))
    a = ap.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    featdir = Path(a.featdir)
    prov = {'cells': {}, 'scorer_input_schema': sorted(SCORER_INPUT_FIELDS),
            'target_read_by_scorer': False, 'target_added_to_union': False}

    for ds in a.datasets.split(','):
        ck = frozen_checkpoint(ds)
        model, plan = load_frozen_shared(ck, device=a.device)
        sids = {f: load_sid(f, ds) for f in FACTORIZATIONS}
        prov['cells'].setdefault(ds, {})['checkpoint'] = str(ck)
        prov['cells'][ds]['checkpoint_sha256'] = sha256_file(ck)
        for f in FACTORIZATIONS:
            prov['cells'][ds][f'sid_{f}'] = str(sid_path(f, ds))
            prov['cells'][ds][f'sid_{f}_sha256'] = sha256_file(sid_path(f, ds))

        for split in a.splits.split(','):
            t0 = time.time()
            print(f'=== {ds} {split} ===', flush=True)
            zp = featdir / f'{ds}_{split}.npz'
            z = np.load(zp, allow_pickle=True)
            i1 = np.asarray(z['cand_ids1']); p1m = np.asarray(z['cand_p1'])
            i2 = np.asarray(z['cand_ids2']); p2m = np.asarray(z['cand_p2'])
            users = np.asarray(z['user'])
            n = len(users)

            items, up1, up2 = build_union(i1, p1m, i2, p2m)
            rank1 = ranks_from_probs(items, up1)
            rank2 = ranks_from_probs(items, up2)

            # flatten (user, candidate) pairs
            owner = np.repeat(np.arange(n), items.shape[1])
            flat_items = items.reshape(-1)
            keep = flat_items >= 0
            owner = owner[keep]; flat_items = flat_items[keep]
            n_pairs = int(keep.sum())

            bank = build_history_bank(ds, split, plan)
            nll = {}
            for f in FACTORIZATIONS:
                toks = sid_tokens_for_items(sids[f], flat_items)
                h, m = bank[f]
                nll[f] = candidate_total_nll(model, h, m, toks, owner,
                                             device=a.device, batch_size=a.batch_users)
                valid_frac = float(np.isfinite(nll[f]).mean())
                print(f'  {f:12s} scored {n_pairs} candidates '
                      f'({valid_frac:.4f} valid SIDs) in {time.time()-t0:.1f}s', flush=True)
                del toks

            # ---- write union ----
            up = out / split / f'{ds}_candidate_union.parquet'
            up.parent.mkdir(parents=True, exist_ok=True)
            _write_union(up, ds, users, owner, flat_items,
                         rank1.reshape(-1)[keep], rank2.reshape(-1)[keep],
                         up1.reshape(-1)[keep], up2.reshape(-1)[keep])

            # ---- write NLL (no target columns) ----
            np_ = out / split / f'{ds}_candidate_nll.parquet'
            _write_nll(np_, ds, users, owner, flat_items,
                       nll[F1], nll[F2], sids[F1], sids[F2])

            prov['cells'][ds][split] = {
                'n_users': int(n), 'n_pairs': n_pairs,
                'mean_union_size': float(keep.reshape(n, -1).sum(axis=1).mean()),
                'union_parquet': str(up), 'union_sha256': sha256_file(up),
                'nll_parquet': str(np_), 'nll_sha256': sha256_file(np_),
                'nll_finite_fraction_F1': float(np.isfinite(nll[F1]).mean()),
                'nll_finite_fraction_F2': float(np.isfinite(nll[F2]).mean()),
                'elapsed_s': round(time.time() - t0, 1),
            }
            print(f'  union size/user {prov["cells"][ds][split]["mean_union_size"]:.1f}  '
                  f'-> {up.name}, {np_.name}', flush=True)
        del model
        import torch
        torch.cuda.empty_cache()

    write_json(out / '_CANDIDATE_NLL_PROVENANCE.json', prov)
    print('done', flush=True)
    return 0


def _write_union(p, ds, users, owner, items, rank1, rank2, p1, p2):
    try:
        import pandas as pd
        df = pd.DataFrame({
            'dataset': ds, 'user': users[owner], 'candidate_item': items,
            'source_in_F1_beam': rank1 >= 1, 'source_in_F2_beam': rank2 >= 1,
            'rank_F1_if_present': rank1, 'rank_F2_if_present': rank2,
            'retrieval_score_F1': p1, 'retrieval_score_F2': p2})
        df.to_parquet(p, index=False)
    except Exception as e:                     # pragma: no cover
        _write_npz_fallback(p, {'owner': owner, 'items': items, 'rank1': rank1,
                                'rank2': rank2, 'p1': p1, 'p2': p2, 'users': users}, e)


def _write_nll(p, ds, users, owner, items, n1, n2, sid1, sid2):
    """Write candidate NLL in a COLUMNAR, list-free layout.

    An earlier version stored `SID_F1`/`SID_F2` as one Python list per row.  That boxes
    ~1.1M lists into an object array and dominated the wall clock (one core pinned at 94%
    with the GPU idle).  The SID is a fixed-width (L,) integer vector, so it is written as
    L scalar columns instead -- same information, no Python objects, and it round-trips.
    """
    import pandas as pd
    L1 = sid1.shape[1]; L2 = sid2.shape[1]
    safe = np.where(items >= 0, items, 0)
    cols = {
        'dataset': ds, 'user': users[owner], 'candidate_item': items,
        'TotalNLL_F1': n1, 'TotalNLL_F2': n2,
        'sid_len_F1': np.int64(L1), 'sid_len_F2': np.int64(L2),
    }
    for pos in range(L1):
        cols[f'SID_F1_{pos}'] = sid1[safe, pos]
    for pos in range(L2):
        cols[f'SID_F2_{pos}'] = sid2[safe, pos]
    pd.DataFrame(cols).to_parquet(p, index=False)


def _write_npz_fallback(p, arrays, err):       # pragma: no cover
    np.savez_compressed(str(p).replace('.parquet', '.npz'), **arrays)
    print(f'  NOTE parquet unavailable ({err}); wrote {p.with_suffix(".npz")}')


if __name__ == '__main__':
    sys.exit(main())
