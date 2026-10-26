#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE E -- teacher-forced full-ID target NLL for EVERY tokenizer factorization.

For each dataset x family x K the script scores the ground-truth target SID
(all L = K+1 item-ID tokens; BOS never scored) under the matching checkpoint.
Evaluation only: checkpoints are loaded in eval mode, no gradient, no training,
no parameter update.

The spectral (nonneg_pearson) cells are recomputed here even though the earlier
full-ID re-audit already produced them, because every cell of the new table must
come from ONE code path; the recomputation is then checked against the audited
`teacher_forced_full_sid_nll.csv`, which doubles as a regression test.

Outputs
  full_id_nll_all_factorizations.parquet   user x factorization long table
  nll_by_position_all_factorizations.csv   mean NLL per (family, K, position)
  _TF_PROVENANCE.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import (  # noqa: E402
    CODEBOOK, DATASETS, FAMILY_LABELS, K_LIST, OUT, ckpt_path, file_sha,
    load_cell, load_sid, sid_array_path, single_rank, BEAM,
)
from src.analysis.teacher_forced import (  # noqa: E402
    load_model_from_ckpt, teacher_forced_logprobs, tokenise_sid,
)
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402


def prefix_group_sizes(sid, k):
    """Collision-group size of each item's first-k semantic prefix."""
    prefixes = sid[:, :k]
    keys = {}
    for r in prefixes:
        t = tuple(int(x) for x in r)
        keys[t] = keys.get(t, 0) + 1
    return np.array([keys[tuple(int(x) for x in r)] for r in prefixes], dtype=np.int64)


def main():
    t_start = time.time()
    prov = {'phase': 'full_id_nll_all_factorizations',
            'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'artifact_root': str(rp.ARTIFACT_ROOT),
            'codebook_size': CODEBOOK,
            'training_performed': 0,
            'evaluation_only': True,
            'scored_tokens': 'all K+1 real item-ID tokens; BOS not scored; EOS/PAD not scored',
            'device': 'cuda', 'inputs': {}, 'alignment_mode': 'canonical'}
    user_rows, pos_rows = [], []

    for ds in DATASETS:
        print(f'=== {ds} ===', flush=True)
        prov['inputs'][ds] = {}
        for family in FAMILY_LABELS:
            for k in K_LIST:
                t0 = time.time()
                sidp = sid_array_path(ds, family, k)
                ckp = ckpt_path(ds, family, k)
                sid = load_sid(ds, family, k)
                L = k + 1
                cell = load_cell(ds, family, k, 'test')
                n = cell['n_users']
                tgt = cell['target']
                rank = single_rank(cell)
                gt_tok = tokenise_sid(sid[tgt - 1], CODEBOOK)
                assert gt_tok.shape == (n, L)
                back = gt_tok - np.arange(L)[None, :] * CODEBOOK - 1
                assert np.array_equal(back, sid[tgt - 1]), 'token/SID round-trip failed'

                model, vocab = load_model_from_ckpt(str(ckp))
                expect_vocab = CODEBOOK * L + 1
                assert vocab == expect_vocab, (vocab, expect_vocab)
                dset = GenRecDataset(
                    dataset_path=str(rp.ARTIFACT_ROOT / 'data' / ds / 'test.parquet'),
                    code_path=str(sidp), mode='evaluation', max_len=20,
                    codebook_size=CODEBOOK)
                assert len(dset) == n, (len(dset), n)
                ld = GenRecDataLoader(dset, batch_size=64, shuffle=False)
                logp = np.zeros((n, L), dtype=np.float64)
                p = 0
                for b in ld:
                    hb = b['history'].numpy()
                    tb = b['target'].numpy()
                    am = b['attention_mask'].numpy()
                    codes = tb - np.arange(tb.shape[1])[None, :] * CODEBOOK - 1
                    assert np.array_equal(codes, sid[tgt[p:p + hb.shape[0]] - 1]), \
                        'dataset target codes disagree with the SID array'
                    logp[p:p + hb.shape[0]] = teacher_forced_logprobs(
                        model, hb, tb, device='cuda', attention_mask=am)
                    p += hb.shape[0]
                del model
                torch.cuda.empty_cache()

                nll = -logp
                total = nll.sum(axis=1)
                sem = nll[:, :L - 1].sum(axis=1)
                col = nll[:, L - 1]
                g = prefix_group_sizes(sid, k)
                gt_g = g[tgt - 1]
                for t in range(L):
                    pos_rows.append({
                        'dataset': ds, 'family': family, 'K': k, 'position': t,
                        'token_type': 'collision' if t == L - 1 else 'semantic',
                        'mean_nll': float(nll[:, t].mean()),
                        'mean_ce_over_uniform': float(nll[:, t].mean())})
                user_rows.append(pd.DataFrame({
                    'dataset': ds, 'user_index': np.arange(n),
                    'target_item_id': tgt, 'family': family, 'K': k, 'L': L,
                    'total_nll': total, 'semantic_nll': sem,
                    'collision_nll': col, 'mean_token_nll': nll.mean(axis=1),
                    'prefix_group_size': gt_g,
                    'rank': rank,
                    'hit5': ((rank >= 1) & (rank <= 5)).astype(np.int8),
                    'hit10': ((rank >= 1) & (rank <= 10)).astype(np.int8),
                    'hit20': ((rank >= 1) & (rank <= 20)).astype(np.int8),
                }))
                prov['inputs'][ds][f'{family}_k{k}'] = {
                    'ckpt': {'path': str(ckp.relative_to(rp.ARTIFACT_ROOT)),
                             'sha256': file_sha(ckp)},
                    'sid': {'path': str(sidp.relative_to(rp.ARTIFACT_ROOT)),
                            'sha256': file_sha(sidp)},
                    'beam': {'path': str(cell['path'].relative_to(rp.ARTIFACT_ROOT)),
                             'sha256': file_sha(cell['path'])},
                    'raw_item_id_shift': cell['raw_item_id_shift'],
                    'n_users': int(n), 'L': int(L),
                }
                print(f'  {family:8s} K={k} n={n} totalNLL={total.mean():.4f} '
                      f'sem={sem.mean():.4f} col={col.mean():.4f} '
                      f'[{time.time()-t0:.0f}s]', flush=True)

    udf = pd.concat(user_rows, ignore_index=True)
    udf.to_parquet(OUT / 'full_id_nll_all_factorizations.parquet', index=False)
    print(f'wrote full_id_nll_all_factorizations.parquet ({len(udf)} rows)')
    pdf = pd.DataFrame(pos_rows)
    pdf.to_csv(OUT / 'nll_by_position_all_factorizations.csv', index=False)
    print(f'wrote nll_by_position_all_factorizations.csv ({len(pdf)} rows)')
    prov['elapsed_s'] = round(time.time() - t_start, 1)
    (OUT / '_TF_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print('provenance written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
