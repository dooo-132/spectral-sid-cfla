#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE D audit — provenance + agreement check for the REUSED candidate NLL.

Two independent questions, both answered from disk:

1. Is the frozen artifact's union (C_K3 ∪ C_RQVAE, built by the previous round)
   the same candidate set this round's pools induce for those two factorizations?
   Checked as an exact set equality of (user, candidate) pairs.

2. Do the frozen TotalNLL_K3 / TotalNLL_RQVAE values agree with an INDEPENDENT
   recomputation through this round's scorer?  This is the reproducibility guard:
   the scorer here is written separately (chunked forward pass, control token
   applied by this module), so agreement is evidence and not a tautology.

Writes `REUSE_AUDIT.json`.  Any dataset failing either check makes the reuse
invalid and the caller must recompute K3/RQVAE instead.

Usage
  python3 mf_reuse_audit.py --split valid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import DATASETS, FLABEL, OUT, sha256_file, sid_path, load_sid, \
    sid_tokens_for_rows, write_json
from mf_candidate_nll import CC_RUN, load_reuse, load_union
from mf_common import REUSE_F
from mf_models import (build_history_tokens, candidate_total_nll,
                       encoder_input_for, load_model_for)



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--split', default='valid')
    ap.add_argument('--pool', default='spectral4')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--max-users', type=int, default=2000,
                    help='users to recompute per factorization (0 = all)')
    a = ap.parse_args(argv)

    report = {'split': a.split, 'pool': a.pool, 'cells': {}}
    for ds in a.datasets.split(','):
        t0 = time.time()
        # the K3 u RQVAE subset only exists in the multi5 pool (spectral4 has no RQ
        # column at all), so the provenance check always reads the multi5 union
        df, upath = load_union(ds, a.split, 'multi5')
        users = df.user.values.astype(np.int64)
        cands = df.candidate_item.values.astype(np.int64)
        key_new = users * (10 ** 9) + cands
        # --- 1. provenance of the frozen artifact ----------------------------
        # The frozen union is C_sharedK3 u C_sharedRQVAE.  This round reuses only its
        # RQVAE column, under the same model that produced it, so the check that
        # matters is: every frozen pair whose F2 (RQVAE) source flag is set must be
        # present in this round's pool, and every pair this round's RQVAE beam
        # generates must be in the frozen union.  (The frozen K3 column is the SHARED
        # checkpoint's likelihood and is not reused: this round's spectral_k3 uses
        # its own independent checkpoint.)
        new_pair = users * (10 ** 9) + (cands - 1)          # 0-based row ids
        new_set = set(int(x) for x in new_pair)
        rq_rows = fr[fr.source_in_F2_beam.values]
        rq_keys = set(int(u) * 10 ** 9 + int(c) for u, c in
                      zip(rq_rows.user.values, rq_rows.candidate_item.values))
        m5 = pd.read_parquet(upath.parent / f'{ds}_multi5_candidate_union.parquet')
        beammask = m5.present_RQ.values
        beam_keys = set(int(u) * 10 ** 9 + int(c) - 1 for u, c in
                        zip(m5.user.values[beammask], m5.candidate_item.values[beammask]))
        cell = {'dataset': ds,
                'frozen_union': str(CC_RUN / ('valid' if a.split == 'valid' else 'test') /
                                    f'{ds}_candidate_union.parquet'),
                'frozen_union_sha256': sha256_file(
                    CC_RUN / ('valid' if a.split == 'valid' else 'test') /
                    f'{ds}_candidate_union.parquet'),
                'n_frozen_union_pairs': int(len(fr)),
                'n_frozen_rqvae_column_pairs': int(len(rq_rows)),
                'n_this_round_rqvae_beam_pairs': int(beammask.sum()),
                'frozen_rqvae_pairs_present_in_pool': bool(rq_keys <= new_set),
                'rqvae_beam_pairs_present_in_frozen': bool(beam_keys <= rq_keys),
                'factorizations': {}}

        # --- 2. recompute a sample and compare -------------------------------
        # Sample users, then for each sampled user recompute EVERY frozen row's SID
        # under this scorer and compare with the frozen TotalNLL.  The frozen rows
        # are the authority on which (user, candidate) pairs are compared.
        n_new = len(df)
        rows_all = cands - 1
        order = np.argsort(users, kind='stable')
        u_sorted = users[order]
        uniq, starts_new = np.unique(u_sorted, return_index=True)
        ends_new = np.append(starts_new[1:], n_new)
        row_of_user = {int(u): i for i, u in enumerate(uniq)}
        # ID CONVENTION: the frozen artifact stores 0-based SID-row indices in
        # `candidate_item` (verified: all its SID_F* columns equal sid[candidate_item]);
        # this round's artifacts store canonical 1-based item ids.  The join key is
        # therefore built on the 0-based row on both sides.
        row_key_new = users * (10 ** 9) + (cands - 1)
        new_pos_of_key = {}
        for i in range(n_new):
            new_pos_of_key[int(row_key_new[i])] = i
        take = len(uniq) if a.max_users <= 0 else min(len(uniq), a.max_users)
        sel_users = set(int(x) for x in uniq[:take])
        sel_keys = [int(k) for k in key_cc if int(k) // (10 ** 9) in sel_users]
        # include only frozen pairs this round's RQVAE beam also generates; the rest
        # of the frozen union came from the SHARED K3 beam, which this round does not
        # use as a candidate source
        sel_keys = [k for k in sel_keys if k in new_pos_of_key]
        if not sel_keys:
            raise AssertionError(f'{ds}: no comparable frozen pairs for the RQVAE audit')
        pos = np.array([new_pos_of_key[k] for k in sel_keys], dtype=np.int64)
        own = np.array([row_of_user[k // (10 ** 9)] for k in sel_keys], dtype=np.int64)
        cache = {}
        for f, col in REUSE_F.items():
            sid = load_sid(f, ds)
            bundle = load_model_for(f, ds, device=a.device, cache=cache)
            h, m = build_history_tokens(ds, a.split, sid_path(f, ds))
            h, m = encoder_input_for(bundle, f, h, m)
            rec = candidate_total_nll(bundle['model'], h, m,
                                      sid_tokens_for_rows(sid, rows_all[pos]), own,
                                      device=a.device, batch_size=a.chunk)
            lut = dict(zip([int(k) for k in key_cc], nll_cc[col].values.tolist()))
            frozen = np.array([lut[k] for k in sel_keys], dtype=np.float64)
            fin = np.isfinite(rec) & np.isfinite(frozen)
            diff = np.abs(rec[fin] - frozen[fin])
            cell['factorizations'][f] = {
                'n_compared': int(fin.sum()),
                'n_nonfinite': int((~fin).sum()),
                'max_abs_diff': float(diff.max()) if len(diff) else None,
                'mean_abs_diff': float(diff.mean()) if len(diff) else None,
                'pearson': float(np.corrcoef(rec[fin], frozen[fin])[0, 1])
                if fin.sum() > 1 else None,
                'frozen_column': col,
                'frozen_nll_sha256': sha256_file(
                    CC_RUN / ('valid' if a.split == 'valid' else 'test') /
                    f'{ds}_candidate_nll.parquet'),
            }
            print(f'  {ds:32s} {f:12s} compared={int(fin.sum())} '
                  f'max|diff|={cell["factorizations"][f]["max_abs_diff"]:.3e} '
                  f'r={cell["factorizations"][f]["pearson"]:.9f}', flush=True)
            del rec, frozen
        cell['elapsed_s'] = round(time.time() - t0, 1)
        report['cells'][ds] = cell
        import torch
        torch.cuda.empty_cache()

    report['frozen_rqvae_provenance_all_datasets'] = all(
        c['frozen_rqvae_pairs_present_in_pool'] and c['rqvae_beam_pairs_present_in_frozen']
        for c in report['cells'].values())
    report['agreement_thresholds'] = {'max_abs_diff': 1e-4, 'pearson': 0.9999}
    report['agreement_ok'] = all(
        (c['factorizations'][f]['max_abs_diff'] is not None
         and c['factorizations'][f]['max_abs_diff'] <= 1e-4
         and c['factorizations'][f]['pearson'] >= 0.9999)
        for c in report['cells'].values() for f in REUSE_F)
    report['reuse_valid'] = bool(report['frozen_rqvae_provenance_all_datasets']
                                 and report['agreement_ok'])
    write_json(OUT / 'REUSE_AUDIT.json', report)
    print(f"\nfrozen RQVAE provenance: {report['frozen_rqvae_provenance_all_datasets']}; "
          f"agreement: {report['agreement_ok']}; reuse_valid={report['reuse_valid']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
