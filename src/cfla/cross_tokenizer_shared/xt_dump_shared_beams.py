#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump beam outputs from ONE shared checkpoint, once per factorization.

All beams come from the SAME checkpoint and the SAME user order; only the
control token (`<F_x>`, prepended to the encoder input) and the SID array used to
decode tokens back to items differ.  Inference only.

Convention: 0-based SID-row item ids with -1 sentinel, matching every other dump
in this study (`xt_common.load_cell` detects the convention anyway).

Outputs -> step2_shared_model/beams/<dataset>_<factorization>_<split>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, CODEBOOK, DATASETS, STEP2, TOKENIZERS, load_sid,
                       sha256_file, shared_vocab_plan)  # noqa: E402
from xt_train_shared import ControlPrefixedDataset  # noqa: E402
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402
from src.tiger.train import TIGER, set_seed  # noqa: E402

BEAMS = STEP2 / 'beams'
BEAMS.mkdir(parents=True, exist_ok=True)


def build_cfg(vocab):
    return {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
            'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': vocab,
            'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu'}


def dump_shared(ckpt, ds, factorizations, split='test', beam_size=30,
                infer_size=96, max_len=20, seed=2025, device='cuda'):
    sd = torch.load(str(ckpt), map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    vocab = sd['model.shared.weight'].shape[0]
    model = TIGER(build_cfg(vocab))
    model.load_state_dict(sd, strict=True)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    plan = shared_vocab_plan(factorizations)
    assert plan['vocab_size'] == vocab, (plan['vocab_size'], vocab)
    set_seed(seed)

    outs = {}
    for f in factorizations:
        sid = load_sid(f, ds)
        L = sid.shape[1]
        sid_to_item = {tuple(int(x) for x in r): i for i, r in enumerate(sid)}
        base = GenRecDataset(
            dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
            code_path=str(TOKENIZERS[f]['sid'](ds)), mode='evaluation',
            max_len=max_len, codebook_size=CODEBOOK)
        dset = ControlPrefixedDataset(base, plan['control_tokens'][f])
        loader = GenRecDataLoader(dset, batch_size=infer_size, shuffle=False)

        all_t, all_c, all_tok, all_s = [], [], [], []
        with torch.no_grad():
            for b in loader:
                input_ids = b['history'].to(dev)
                am = b['attention_mask'].to(dev)
                labels = b['target'].to(dev)
                bs = input_ids.shape[0]
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

        target_item_ids = np.concatenate(all_t)
        candidate_item_ids = np.concatenate(all_c)
        candidate_tokens = np.concatenate(all_tok)
        sequence_scores = np.concatenate(all_s)
        vr = float((candidate_item_ids >= 0).mean())
        tr = float((target_item_ids >= 0).mean())
        out_p = BEAMS / f'{ds}_{f}_{split}.npz'
        np.savez_compressed(
            out_p, split=np.array(split), shared_ckpt=np.array(str(ckpt)),
            factorization=np.array(f), control_token=np.array(plan['control_tokens'][f]),
            sid_len=np.array(L), beam_size=np.array(beam_size), vocab_size=np.array(vocab),
            target_item_ids=target_item_ids, candidate_item_ids=candidate_item_ids,
            candidate_tokens=candidate_tokens, sequence_scores=sequence_scores)
        outs[f] = str(out_p)
        print(f'  {ds} {f} {split}: cand_valid={vr:.4f} tgt_valid={tr:.4f} -> {out_p.name}',
              flush=True)
    del model
    torch.cuda.empty_cache()
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--factorizations', default='spectral_k3:rqvae3')
    ap.add_argument('--splits', default='test,valid')
    ap.add_argument('--infer_size', type=int, default=96)
    ap.add_argument('--beam_size', type=int, default=30)
    a = ap.parse_args(argv)
    for ds in a.datasets.split(','):
        for split in a.splits.split(','):
            print(f'=== shared {ds} {split} ===', flush=True)
            dump_shared(a.ckpt, ds, a.factorizations.split(':'), split,
                        beam_size=a.beam_size, infer_size=a.infer_size)
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
