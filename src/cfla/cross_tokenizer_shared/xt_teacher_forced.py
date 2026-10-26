#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1 — teacher-forced full-ID target NLL for every independent tokenizer.

For each dataset x tokenizer the ground-truth target SID (all L = 4 tokens:
3 semantic + 1 disambiguation) is scored under that tokenizer's own recommender
checkpoint.  Evaluation only: eval mode, no gradient, no parameter update.

Because these are independently trained models, raw NLL is not calibrated across
tokenizers (STEP1 §17).  The script therefore also computes, for every
(tokenizer, dataset), the mean and standard deviation of TotalNLL on the
VALIDATION split, so the test NLL can be reported as a within-model z-score.
Nothing about the calibration touches test labels.

Outputs -> step1_cross_tokenizer/
  cross_tokenizer_target_nll.parquet   per user x tokenizer
  cross_tokenizer_nll_calibration.csv  per tokenizer x dataset valid mean/std
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, CODEBOOK, DATASETS, STEP1, TOKENIZERS, ckpt_path,
                       load_sid, sha256_file, sid_path)  # noqa: E402
from src.analysis.teacher_forced import (load_model_from_ckpt,
                                         teacher_forced_logprobs, tokenise_sid)  # noqa: E402
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402


def score_split(model, sid, ds, split, sid_p, n_expected=None):
    L = sid.shape[1]
    dset = GenRecDataset(dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
                         code_path=str(sid_p), mode='evaluation',
                         max_len=20, codebook_size=CODEBOOK)
    n = len(dset)
    if n_expected is not None:
        assert n == n_expected, (n, n_expected)
    loader = GenRecDataLoader(dset, batch_size=64, shuffle=False)
    logp = np.zeros((n, L), dtype=np.float64)
    p = 0
    # targets are yielded in token space by GenRecDataset; we re-derive them from
    # the parquet order so the SID lookup can be asserted per batch.
    import pandas as _pd
    tgt_col = _pd.read_parquet(ART / 'data' / ds / f'{split}.parquet')['target'].to_numpy()
    for b in loader:
        hb = b['history'].numpy()
        tb = b['target'].numpy()
        am = b['attention_mask'].numpy()
        codes = tb - np.arange(tb.shape[1])[None, :] * CODEBOOK - 1
        sl = tgt_col[p:p + hb.shape[0]]
        assert np.array_equal(codes, sid[sl - 1]), 'dataset target codes disagree with SID array'
        logp[p:p + hb.shape[0]] = teacher_forced_logprobs(
            model, hb, tb, device='cuda', attention_mask=am)
        p += hb.shape[0]
    return -logp, tgt_col


def main():
    t0 = time.time()
    rows, cal_rows, prov = [], [], {'tokenizers': {}, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}

    for tok in TOKENIZERS:
        for ds in DATASETS:
            sid_p = sid_path(tok, ds)
            ck = ckpt_path(tok, ds)
            if not sid_p.exists() or not ck.exists():
                print(f'SKIP {tok} {ds}')
                continue
            sid = load_sid(tok, ds)
            L = sid.shape[1]
            print(f'=== {tok} {ds} L={L} ===', flush=True)
            model, vocab = load_model_from_ckpt(str(ck))
            assert vocab == L * CODEBOOK + 1, (vocab, L)
            per_split = {}
            for split in ('valid', 'test'):
                nll, tgt = score_split(model, sid, ds, split, sid_p)
                per_split[split] = (nll, tgt)
            del model
            torch.cuda.empty_cache()

            vnll, _ = per_split['valid']
            vtot = vnll.sum(axis=1)
            vmean, vstd = float(vtot.mean()), float(vtot.std(ddof=1))
            cal_rows.append({'dataset': ds, 'tokenizer': tok,
                             'label': TOKENIZERS[tok]['label'],
                             'valid_total_nll_mean': vmean,
                             'valid_total_nll_std': vstd,
                             'valid_n': int(len(vtot))})

            tnll, ttgt = per_split['test']
            tot = tnll.sum(axis=1)
            sem = tnll[:, :L - 1].sum(axis=1)
            dis = tnll[:, L - 1]
            frame = pd.DataFrame({
                'dataset': ds, 'tokenizer': tok,
                'label': TOKENIZERS[tok]['label'],
                'user_index': np.arange(len(tot)), 'target_item_id': ttgt,
                'L': L, 'total_nll': tot, 'semantic_nll': sem,
                'disambiguation_nll': dis,
                'mean_token_nll': tnll.mean(axis=1),
                'valid_total_nll_mean': vmean, 'valid_total_nll_std': vstd,
                'z_total_nll': (tot - vmean) / vstd if vstd > 0 else np.nan,
            })
            for t in range(L):
                frame[f'nll_pos{t}'] = tnll[:, t]
            rows.append(frame)
            prov['tokenizers'][f'{tok}|{ds}'] = {
                'sid_path': str(sid_p), 'sid_sha256': sha256_file(sid_p),
                'ckpt_path': str(ck), 'ckpt_sha256': sha256_file(ck),
                'L': L, 'n_test': int(len(tot)), 'n_valid': int(len(vtot)),
                'valid_total_nll_mean': vmean, 'valid_total_nll_std': vstd,
            }
            print(f'  valid mean={vmean:.4f} std={vstd:.4f} | '
                  f'test mean={tot.mean():.4f} [{time.time()-t0:.0f}s]', flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(STEP1 / 'cross_tokenizer_target_nll.parquet', index=False)
    pd.DataFrame(cal_rows).to_csv(STEP1 / 'cross_tokenizer_nll_calibration.csv', index=False)
    import json
    (STEP1 / '_tf_provenance.json').write_text(json.dumps(prov, indent=2, default=str))
    print(f'wrote cross_tokenizer_target_nll.parquet ({len(df)} rows)')
    print(f'wrote cross_tokenizer_nll_calibration.csv ({len(cal_rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
