#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B — teacher-forced target NLL under the FROZEN shared checkpoint, once per
factorization.

Used for exactly two things, both explicitly permitted by the protocol:

  * the SECONDARY mechanistic label `q_NLL = softmax(-TotalNLL)` on
    RouterTrain/RouterDev, where the target is known during router training;
  * the explanatory NLL diagnostic reported AFTER the test routing is frozen.

It is NEVER a selector input feature.  `af_router_train`/`af_eval_test` only ever
read the feature blocks `router_h` / `router_p`, and `SELECTOR_FEATURE_AUDIT.md`
records that this file is not among them.

Evaluation only: eval mode, no gradients, frozen weights.

Outputs -> <outdir>/nll/<dataset>_<factorization>_<split>.npz   (total_nll (n,))
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xt_common import ART, CODEBOOK, DATASETS, TOKENIZERS, load_sid, sha256_file, shared_vocab_plan
from xt_train_shared import ControlPrefixedDataset, build_cfg
from src.tiger.dataset import GenRecDataset
from src.tiger.dataloader import GenRecDataLoader
from src.tiger.train import TIGER
from src.analysis.teacher_forced import teacher_forced_logprobs


def load_frozen(ckpt, factorizations):
    sd = torch.load(str(ckpt), map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    vocab = int(sd['model.shared.weight'].shape[0])
    plan = shared_vocab_plan(list(factorizations))
    if plan['vocab_size'] != vocab:
        raise AssertionError(f'vocab mismatch {vocab} vs {plan["vocab_size"]}')
    m = TIGER(build_cfg(vocab))
    m.load_state_dict(sd, strict=True)
    return m, plan


def score_split(model, plan, ds, f, split, device, max_len=20, batch_size=96):
    sid = load_sid(f, ds)
    L = sid.shape[1]
    base = GenRecDataset(dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
                         code_path=str(TOKENIZERS[f]['sid'](ds)), mode='evaluation',
                         max_len=max_len, codebook_size=CODEBOOK)
    dset = ControlPrefixedDataset(base, plan['control_tokens'][f])
    loader = GenRecDataLoader(dset, batch_size=batch_size, shuffle=False)
    n = len(dset)
    nll = np.zeros((n, L), dtype=np.float64)
    import pandas as pd
    tgt_col = pd.read_parquet(ART / 'data' / ds / f'{split}.parquet')['target'].to_numpy()
    p = 0
    for b in loader:
        hb = b['history'].numpy()
        tb = b['target'].numpy()
        am = b['attention_mask'].numpy()
        codes = tb - np.arange(tb.shape[1])[None, :] * CODEBOOK - 1
        sl = tgt_col[p:p + hb.shape[0]]
        if not np.array_equal(codes, sid[sl - 1]):
            raise AssertionError(f'{ds}/{split}: target codes disagree with SID array')
        nll[p:p + hb.shape[0]] = -teacher_forced_logprobs(model, hb, tb, device=device,
                                                          attention_mask=am)
        p += hb.shape[0]
    return nll, tgt_col


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--factorizations', default='spectral_k3:rqvae3')
    ap.add_argument('--splits', default='valid,test')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)

    ds = a.dataset
    fcts = a.factorizations.split(':')
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model, plan = load_frozen(a.ckpt, fcts)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    outdir = Path(a.outdir) / 'nll'
    outdir.mkdir(parents=True, exist_ok=True)
    man_p = outdir / 'NLL_MANIFEST.json'
    man = json.loads(man_p.read_text()) if man_p.exists() else {'cells': {}, 'datasets': {}}
    man.setdefault('datasets', {})
    # each dataset is trained separately, so the checkpoint hash must be recorded
    # per dataset rather than as a single global value
    man['datasets'][ds] = {'checkpoint': str(a.ckpt),
                           'checkpoint_sha256': sha256_file(a.ckpt)}
    man['target_nll_used_as_feature'] = False
    man['nearest_use'] = ('secondary label q_NLL on RouterTrain/RouterDev, and an '
                          'explanatory post-freeze diagnostic; never a selector input')

    for split in a.splits.split(','):
        for f in fcts:
            t0 = time.time()
            nll, tgt = score_split(model, plan, ds, f, split, dev)
            tot = nll.sum(axis=1)
            out_p = outdir / f'{ds}_{f}_{split}.npz'
            np.savez_compressed(out_p, total_nll=tot, per_token_nll=nll,
                                target_item_ids=tgt, L=np.array(nll.shape[1]),
                                factorization=np.array(f), split=np.array(split))
            rec = {'n': int(len(tot)), 'mean_total_nll': float(tot.mean()),
                   'std_total_nll': float(tot.std(ddof=1)),
                   'sha256': sha256_file(out_p), 'elapsed_s': round(time.time() - t0, 1)}
            man['cells'][f'{ds}|{f}|{split}'] = rec
            man_p.write_text(json.dumps(man, indent=2))
            print(f'  {ds} {f} {split}: mean={tot.mean():.4f} '
                  f'({rec["elapsed_s"]}s)', flush=True)

    del model
    torch.cuda.empty_cache()
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
