#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B — leakage-safe task-free feature + label generation from the FROZEN shared checkpoints.

Produces, for a given dataset, split and factorization:

  1. a shared-model beam dump  (the same artifact kind as
     `step2_shared_model/beams/<ds>_<f>_test.npz`, written here for the `valid`
     split as well), giving the target-free beam statistics the selector needs;
  2. a mean-pooled frozen encoder representation of the user history, computed in
     the same inference pass.  The encoder never sees the target, so this is a
     legitimate Router-H input.

LEAKAGE DISCIPLINE
------------------
Inference uses ONLY `history` and `attention_mask`.  `labels` (the target) is read
solely to (a) reproduce the dump's `target_item_ids` field for parity with the
existing artifacts and (b) compute the teacher-forced target NLL used as a
*training label* and as an explanatory diagnostic.  It is never an input to the
encoder or the decoder.

The frozen checkpoints are opened read-only; this script never writes to
`the frozen artifact root`(artifact root) or to any prior study's output directory.

Usage
  python3 af_dump_shared.py --ckpt <shared_model.pth> --dataset Beauty \
      --splits valid,test --outdir outputs/adaptive_factorization/selector/features
"""
from __future__ import annotations

import os as _os
import pathlib as _pl
_REPO_ROOT = str(_pl.Path(__file__).resolve().parents[3])
_ART_ROOT = _os.environ.get('CFLA_ARTIFACT_ROOT') or _os.path.join(_REPO_ROOT, 'artifacts')
_UP_ROOT = _os.environ.get('CFLA_UPSTREAM_ROOT') or _ART_ROOT
import sys as _sys
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from src.utils import repo_paths as rp  # noqa: E402
import argparse
import hashlib
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

from xt_common import (ART, CODEBOOK, DATASETS, SHORT, TOKENIZERS, load_sid, sha256_file,
                       shared_vocab_plan)
from xt_train_shared import ControlPrefixedDataset, build_cfg
from src.tiger.dataset import GenRecDataset
from src.tiger.dataloader import GenRecDataLoader
from src.tiger.train import TIGER, set_seed


def load_frozen(ckpt, factorizations):
    sd = torch.load(str(ckpt), map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    vocab = int(sd['model.shared.weight'].shape[0])
    plan = shared_vocab_plan(list(factorizations))
    if plan['vocab_size'] != vocab:
        raise AssertionError(f'vocab mismatch: checkpoint {vocab} vs plan {plan["vocab_size"]}')
    model = TIGER(build_cfg(vocab))
    model.load_state_dict(sd, strict=True)
    return model, plan


def mean_pool(hidden, mask):
    """Masked mean over non-pad positions. hidden (B,T,D), mask (B,T) in {0,1}."""
    m = mask.to(hidden.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1e-6)
    return (hidden * m).sum(dim=1) / denom


def dump_one(model, plan, ds, f, split, out_npz, out_feat, beam_size, infer_size,
             max_len, seed, device, overwrite=False, skip_beams=False):
    if out_npz.exists() and out_feat.exists() and not overwrite:
        return {'skipped': True, 'beams': str(out_npz), 'features': str(out_feat)}
    if skip_beams and not out_npz.exists():
        raise FileNotFoundError(
            f'--skip-beams requested but {out_npz} does not exist; the frozen '
            f'checkpoint\'s beam dump must already be present to reuse it')

    sid = load_sid(f, ds)
    L = sid.shape[1]
    sid_to_item = {tuple(int(x) for x in r): i for i, r in enumerate(sid)}
    base = GenRecDataset(
        dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
        code_path=str(TOKENIZERS[f]['sid'](ds)), mode='evaluation',
        max_len=max_len, codebook_size=CODEBOOK)
    dset = ControlPrefixedDataset(base, plan['control_tokens'][f])
    loader = GenRecDataLoader(dset, batch_size=infer_size, shuffle=False)
    set_seed(seed)

    all_t, all_c, all_tok, all_s, all_h = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            input_ids = b['history'].to(device)
            am = b['attention_mask'].to(device)
            labels = b['target'].to(device)
            bs = input_ids.shape[0]

            # --- frozen encoder representation (target never enters) ---
            enc = model.model.get_encoder()(input_ids=input_ids, attention_mask=am)
            all_h.append(mean_pool(enc.last_hidden_state, am).detach().cpu().numpy())

            # --- target-free beam generation (skipped when reusing a frozen dump) ---
            if not skip_beams:
                out = model.model.generate(
                    input_ids=input_ids, attention_mask=am, max_length=L + 1,
                    num_beams=beam_size, num_return_sequences=beam_size,
                    return_dict_in_generate=True, output_scores=True)
                seq = out.sequences[:, 1:]
                if seq.shape[1] < L:
                    seq = torch.cat([seq, torch.zeros((seq.shape[0], L - seq.shape[1]),
                                                      dtype=seq.dtype, device=seq.device)], dim=1)
                seq = seq[:, :L]
                tok = seq.reshape(bs, beam_size, L).detach().cpu().numpy()
                if getattr(out, 'sequences_scores', None) is not None:
                    sc = out.sequences_scores.reshape(bs, beam_size).cpu().numpy()
                else:
                    sc = np.zeros((bs, beam_size))
                lab = labels.detach().cpu().numpy()

                def dec(t):
                    s = tuple(int(t[p]) - p * CODEBOOK - 1 for p in range(L))
                    if any(x < 0 or x >= CODEBOOK for x in s):
                        return -1
                    return sid_to_item.get(s, -1)

                all_t.append(np.array([dec(lab[i]) for i in range(bs)], dtype=np.int64))
                all_c.append(np.array([[dec(tok[i, j]) for j in range(beam_size)]
                                       for i in range(bs)], dtype=np.int64))
                all_tok.append(tok.astype(np.int64))
                all_s.append(sc.astype(np.float32))

    if skip_beams:
        # Reuse the frozen checkpoint's existing beam dump verbatim; only the
        # encoder representation is (re)computed.
        z = np.load(out_npz, allow_pickle=True)
        target_item_ids = np.asarray(z['target_item_ids'])
        candidate_item_ids = np.asarray(z['candidate_item_ids'])
        beam_rec = {'reused_frozen_dump': str(out_npz), 'sha256': sha256_file(out_npz)}
    else:
        target_item_ids = np.concatenate(all_t)
        candidate_item_ids = np.concatenate(all_c)
        candidate_tokens = np.concatenate(all_tok)
        sequence_scores = np.concatenate(all_s)
        beam_rec = {'sha256': sha256_file(out_npz)}
    history_repr = np.concatenate(all_h).astype(np.float32)
    elapsed = time.time() - t0

    out_feat.parent.mkdir(parents=True, exist_ok=True)
    if not skip_beams:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz, split=np.array(split), factorization=np.array(f),
            control_token=np.array(plan['control_tokens'][f]), sid_len=np.array(L),
            beam_size=np.array(beam_size), vocab_size=np.array(plan['vocab_size']),
            target_item_ids=target_item_ids, candidate_item_ids=candidate_item_ids,
            candidate_tokens=candidate_tokens, sequence_scores=sequence_scores)
    np.savez_compressed(out_feat, history_repr=history_repr,
                        split=np.array(split), factorization=np.array(f),
                        pooling=np.array('mean_nonpad_encoder_last_hidden'),
                        control_token=np.array(plan['control_tokens'][f]))

    return {
        'skipped': False,
        'beams': str(out_npz), 'features': str(out_feat),
        **beam_rec,
        'beams_reused_not_recomputed': bool(skip_beams),
        'n_rows': int(candidate_item_ids.shape[0]),
        'beam': int(candidate_item_ids.shape[1]),
        'candidate_valid_ratio': float((candidate_item_ids >= 0).mean()),
        'target_valid_ratio': float((target_item_ids >= 0).mean()),
        'feature_dim': int(history_repr.shape[1]),
        'elapsed_s': round(elapsed, 1),
        'feature_npz_sha256': sha256_file(out_feat),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--factorizations', default='spectral_k3:rqvae3')
    ap.add_argument('--splits', default='valid')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--beam_size', type=int, default=30)
    ap.add_argument('--infer_size', type=int, default=96)
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--skip-beams', action='store_true',
                    help='reuse an existing frozen beam dump verbatim and only '
                         '(re)compute the encoder history representation')
    a = ap.parse_args(argv)

    ds = a.dataset
    factorizations = a.factorizations.split(':')
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model, plan = load_frozen(a.ckpt, factorizations)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    outdir = Path(a.outdir)
    manifest_p = outdir / 'FEATURE_MANIFEST.json'
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {'cells': {}}
    manifest.setdefault('cells', {})
    manifest['checkpoint'] = str(a.ckpt)
    manifest['checkpoint_sha256'] = sha256_file(a.ckpt)
    manifest['pooling'] = 'mean over non-pad final encoder hidden states'
    manifest['target_used_as_input'] = False

    for split in a.splits.split(','):
        for f in factorizations:
            key = f'{ds}|{f}|{split}'
            print(f'=== {key} ===', flush=True)
            res = dump_one(model, plan, ds, f, split,
                           outdir / 'beams' / f'{ds}_{f}_{split}.npz',
                           outdir / 'history_repr' / f'{ds}_{f}_{split}.npz',
                           a.beam_size, a.infer_size, a.max_len, a.seed, dev,
                           overwrite=a.overwrite, skip_beams=a.skip_beams)
            print('   ', json.dumps(res), flush=True)
            manifest['cells'][key] = res
            manifest_p.parent.mkdir(parents=True, exist_ok=True)
            manifest_p.write_text(json.dumps(manifest, indent=2))
    del model
    torch.cuda.empty_cache()
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
