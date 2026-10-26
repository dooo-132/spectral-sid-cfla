#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP A — dump beam outputs for the COMPUTE-MATCHED shared Beauty checkpoint.

Same inference path as the previous study's shared-beam dumper, with one addition:
it records separately measured encoder and beam-generation latency so the
exposure-matched run can be reported with its own timing rather than borrowing the
previous study's.

Outputs -> outputs/adaptive_factorization/compute_matched/beams/<Beauty>_<f>_<split>.npz
"""
from __future__ import annotations

import argparse
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

from af_dump_shared import load_frozen, mean_pool
from xt_common import ART, CODEBOOK, TOKENIZERS, load_sid, sha256_file, shared_vocab_plan
from xt_train_shared import ControlPrefixedDataset
from src.tiger.dataset import GenRecDataset
from src.tiger.dataloader import GenRecDataLoader
from src.tiger.train import set_seed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--factorizations', default='spectral_k3:rqvae3')
    ap.add_argument('--splits', default='test,valid')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--beam_size', type=int, default=30)
    ap.add_argument('--infer_size', type=int, default=96)
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)

    ds = a.dataset
    fcts = a.factorizations.split(':')
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model, plan = load_frozen(a.ckpt, fcts)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    outdir = Path(a.outdir) / 'beams'
    outdir.mkdir(parents=True, exist_ok=True)
    man_p = Path(a.outdir) / 'BEAM_MANIFEST.json'
    man = json.loads(man_p.read_text()) if man_p.exists() else {'cells': {}}
    man['checkpoint'] = str(a.ckpt)
    man['checkpoint_sha256'] = sha256_file(a.ckpt)
    man['beam_size'] = a.beam_size
    man['infer_size'] = a.infer_size

    for split in a.splits.split(','):
        for f in fcts:
            sid = load_sid(f, ds)
            L = sid.shape[1]
            sid_to_item = {tuple(int(x) for x in r): i for i, r in enumerate(sid)}
            base = GenRecDataset(
                dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
                code_path=str(TOKENIZERS[f]['sid'](ds)), mode='evaluation',
                max_len=a.max_len, codebook_size=CODEBOOK)
            dset = ControlPrefixedDataset(base, plan['control_tokens'][f])
            loader = GenRecDataLoader(dset, batch_size=a.infer_size, shuffle=False)
            set_seed(a.seed)
            all_t, all_c, all_tok, all_s = [], [], [], []
            enc_t, gen_t, n_batches = 0.0, 0.0, 0
            t0 = time.time()
            with torch.no_grad():
                for b in loader:
                    inp = b['history'].to(dev)
                    am = b['attention_mask'].to(dev)
                    labels = b['target'].to(dev)
                    bs = inp.shape[0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_e = time.perf_counter()
                    enc = model.model.get_encoder()(input_ids=inp, attention_mask=am)
                    _ = mean_pool(enc.last_hidden_state, am)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    enc_t += time.perf_counter() - t_e
                    t_g = time.perf_counter()
                    out = model.model.generate(
                        input_ids=inp, attention_mask=am, max_length=L + 1,
                        num_beams=a.beam_size, num_return_sequences=a.beam_size,
                        return_dict_in_generate=True, output_scores=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    gen_t += time.perf_counter() - t_g
                    n_batches += 1
                    seq = out.sequences[:, 1:]
                    if seq.shape[1] < L:
                        seq = torch.cat([seq, torch.zeros((bs, L - seq.shape[1]),
                                                          dtype=seq.dtype, device=seq.device)],
                                        dim=1)
                    seq = seq[:, :L]
                    tok = seq.reshape(bs, a.beam_size, L).detach().cpu().numpy()
                    sc = (out.sequences_scores.reshape(bs, a.beam_size).cpu().numpy()
                          if getattr(out, 'sequences_scores', None) is not None
                          else np.zeros((bs, a.beam_size)))
                    lab = labels.detach().cpu().numpy()

                    def dec(t):
                        s = tuple(int(t[p]) - p * CODEBOOK - 1 for p in range(L))
                        if any(x < 0 or x >= CODEBOOK for x in s):
                            return -1
                        return sid_to_item.get(s, -1)

                    all_t.append(np.array([dec(lab[i]) for i in range(bs)], dtype=np.int64))
                    all_c.append(np.array([[dec(tok[i, j]) for j in range(a.beam_size)]
                                           for i in range(bs)], dtype=np.int64))
                    all_tok.append(tok.astype(np.int64))
                    all_s.append(sc.astype(np.float32))
            target_item_ids = np.concatenate(all_t)
            candidate_item_ids = np.concatenate(all_c)
            out_p = outdir / f'{ds}_{f}_{split}.npz'
            np.savez_compressed(
                out_p, split=np.array(split), factorization=np.array(f),
                control_token=np.array(plan['control_tokens'][f]), sid_len=np.array(L),
                beam_size=np.array(a.beam_size), vocab_size=np.array(plan['vocab_size']),
                target_item_ids=target_item_ids, candidate_item_ids=candidate_item_ids,
                candidate_tokens=np.concatenate(all_tok),
                sequence_scores=np.concatenate(all_s))
            rec = {'n': int(candidate_item_ids.shape[0]),
                   'sha256': sha256_file(out_p),
                   'candidate_valid_ratio': float((candidate_item_ids >= 0).mean()),
                   'target_valid_ratio': float((target_item_ids >= 0).mean()),
                   'encoder_seconds': round(enc_t, 3),
                   'beam_generation_seconds': round(gen_t, 3),
                   'n_batches': n_batches,
                   'elapsed_s': round(time.time() - t0, 1)}
            man['cells'][f'{ds}|{f}|{split}'] = rec
            man_p.write_text(json.dumps(man, indent=2))
            print(f'  {ds} {f} {split}: {json.dumps(rec)}', flush=True)

    del model
    torch.cuda.empty_cache()
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
