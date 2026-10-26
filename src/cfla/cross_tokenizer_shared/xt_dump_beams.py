#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump beam outputs for the independent tokenizers (RQ-VAE / OPQ / RQ-OPQ).

INFERENCE ONLY.  No training, no gradient, no parameter update.

Convention: this reproduces the HISTORICAL dumper
(`<artifact_root>/model/dump_beam_outputs.py`), which stores the **0-based**
SID-array row index in `candidate_item_ids` / `target_item_ids` with `-1` as the
"no item" sentinel.  `xt_common.load_cell` detects the convention per file, but
keeping one convention across all new dumps makes the cross-tokenizer files
directly comparable to the historical spectral dumps.

Self-check: before dumping anything new, `--verify` re-dumps Beauty Spectral K=3
and requires element-wise identity with the historical dump.

Beam parameters match the canonical recipe: beam 30, infer 96, max_len 20,
codebook 256, sid_len = L.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, DATASETS, STEP1, TOKENIZERS, ckpt_path, load_sid,
                       sha256_file, sid_path)  # noqa: E402
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402
from src.tiger.train import TIGER, set_seed  # noqa: E402

CB = 256


def strip_module_prefix(sd):
    if all(k.startswith('module.') for k in sd):
        return {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def infer_vocab_size(sd):
    for key in ('model.shared.weight', 'model.encoder.embed_tokens.weight',
                'model.decoder.embed_tokens.weight', 'model.lm_head.weight'):
        if key in sd:
            return int(sd[key].shape[0])
    for k, v in sd.items():
        if k.endswith('shared.weight') or k.endswith('embed_tokens.weight'):
            return int(v.shape[0])
    raise ValueError('cannot infer vocab_size')


def build_config(vocab_size):
    return {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
            'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
            'vocab_size': vocab_size, 'pad_token_id': 0, 'eos_token_id': 0,
            'feed_forward_proj': 'relu'}


def build_sid_to_item(codes):
    """HISTORICAL convention: 0-based row index (row i -> item id i)."""
    mp = {}
    for idx, row in enumerate(codes):
        mp[tuple(int(x) for x in row)] = idx
    return mp


def tokens_to_sid(tokens, sid_len, codebook_size):
    tokens = [int(x) for x in tokens[:sid_len]]
    if len(tokens) < sid_len:
        return None
    out = []
    for i, t in enumerate(tokens):
        lo, hi = i * codebook_size + 1, (i + 1) * codebook_size
        if t < lo or t > hi:
            return None
        out.append(t - lo)
    return tuple(out)


def decode_tokens_to_item(tokens, sid_len, sid_to_item, codebook_size):
    sid = tokens_to_sid(tokens, sid_len, codebook_size)
    return -1 if sid is None else sid_to_item.get(sid, -1)


def dump(dataset, code_path, ckpt, output_path, split, sid_len, beam_size=30,
         infer_size=96, codebook_size=CB, max_len=20, seed=2025,
         device='cuda'):
    set_seed(seed)
    sd = strip_module_prefix(torch.load(str(ckpt), map_location='cpu'))
    vocab_size = infer_vocab_size(sd)
    model = TIGER(build_config(vocab_size))
    model.load_state_dict(sd, strict=True)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    codes = np.load(str(code_path), allow_pickle=True).astype(np.int64)
    sid_to_item = build_sid_to_item(codes)
    ds_obj = GenRecDataset(
        dataset_path=str(ART / 'data' / dataset / f'{split}.parquet'),
        code_path=str(code_path), mode='evaluation', max_len=max_len,
        codebook_size=codebook_size)
    loader = GenRecDataLoader(ds_obj, batch_size=infer_size, shuffle=False)

    all_t, all_c, all_tok, all_s, all_tr = [], [], [], [], []
    gen_max_length = sid_len + 1
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['history'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['target'].to(dev)
            bs = input_ids.shape[0]
            out = model.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=gen_max_length, num_beams=beam_size,
                num_return_sequences=beam_size, return_dict_in_generate=True,
                output_scores=True)
            seq = out.sequences[:, 1:]
            if seq.shape[1] < sid_len:
                seq = torch.cat([seq, torch.zeros((seq.shape[0], sid_len - seq.shape[1]),
                                                  dtype=seq.dtype, device=seq.device)], dim=1)
            seq = seq[:, :sid_len]
            cand_tokens = seq.reshape(bs, beam_size, sid_len)
            if getattr(out, 'sequences_scores', None) is not None:
                seq_scores = out.sequences_scores.reshape(bs, beam_size)
            else:
                seq_scores = torch.zeros((bs, beam_size), device=dev)
            try:
                tr = model.model.compute_transition_scores(
                    out.sequences, out.scores, out.beam_indices,
                    normalize_logits=True)[:, :sid_len]
                if tr.shape[1] < sid_len:
                    tr = torch.cat([tr, torch.full(
                        (tr.shape[0], sid_len - tr.shape[1]), float('nan'),
                        device=tr.device)], dim=1)
                tr = tr.reshape(bs, beam_size, sid_len)
            except Exception as exc:
                print('WARNING transition scores:', exc)
                tr = torch.full((bs, beam_size, sid_len), float('nan'), device=dev)

            labels_np = labels.detach().cpu().numpy()
            tok_np = cand_tokens.detach().cpu().numpy()
            tgt_ids = []
            cand_ids = np.full((bs, beam_size), -1, dtype=np.int64)
            for i in range(bs):
                tgt_ids.append(decode_tokens_to_item(labels_np[i], sid_len,
                                                     sid_to_item, codebook_size))
                for b in range(beam_size):
                    cand_ids[i, b] = decode_tokens_to_item(tok_np[i, b], sid_len,
                                                           sid_to_item, codebook_size)
            all_t.append(np.asarray(tgt_ids, dtype=np.int64))
            all_c.append(cand_ids)
            all_tok.append(tok_np.astype(np.int64))
            all_s.append(seq_scores.detach().cpu().numpy().astype(np.float32))
            all_tr.append(tr.detach().cpu().numpy().astype(np.float32))

    target_item_ids = np.concatenate(all_t)
    candidate_item_ids = np.concatenate(all_c)
    candidate_tokens = np.concatenate(all_tok)
    sequence_scores = np.concatenate(all_s)
    transition_logprobs = np.concatenate(all_tr)

    valid_ratio = float((candidate_item_ids >= 0).mean())
    target_ratio = float((target_item_ids >= 0).mean())
    print(f'  decode health: cand={valid_ratio:.4f} target={target_ratio:.4f} '
          f'vocab={vocab_size} n={len(target_item_ids)}')
    if valid_ratio < 0.90:
        raise RuntimeError('token layout mismatch: candidates do not decode')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, split=np.array(split), code_path=np.array(str(code_path)),
        ckpt_path=np.array(str(ckpt)), sid_len=np.array(sid_len),
        beam_size=np.array(beam_size), vocab_size=np.array(vocab_size),
        target_item_ids=target_item_ids, candidate_item_ids=candidate_item_ids,
        candidate_tokens=candidate_tokens, sequence_scores=sequence_scores,
        transition_logprobs=transition_logprobs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return candidate_item_ids


def verify_spectral_k3():
    """Re-dump Beauty Spectral K=3 and require identity with the historical dump."""
    ds = 'Beauty'
    hist = (ART / f'model/beam_dumps_{ds}_nonneg_pearson_v3' / 'test_k3_sid4.npz')
    tmp = STEP1 / '_tmp_verify_spectral_k3.npz'   # written inside the active repo only
    dump(ds, sid_path('spectral_k3', ds), ckpt_path('spectral_k3', ds), tmp,
         'test', 4)
    new = np.load(tmp, allow_pickle=True)
    old = np.load(hist, allow_pickle=True)
    a = np.asarray(new['candidate_item_ids'])
    b = np.asarray(old['candidate_item_ids'])
    ident = bool(a.shape == b.shape and np.array_equal(a, b))
    ti = bool(np.array_equal(np.asarray(new['target_item_ids']),
                             np.asarray(old['target_item_ids'])))
    print(f'VERIFY spectral_k3 Beauty: shape {a.shape} vs {b.shape} '
          f'candidates_identical={ident} targets_identical={ti}')
    tmp.unlink(missing_ok=True)
    return ident and ti


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--tokenizers', default='rqvae3')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--splits', default='test,valid')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--infer_size', type=int, default=96)
    ap.add_argument('--beam_size', type=int, default=30)
    a = ap.parse_args(argv)

    if a.verify:
        return 0 if verify_spectral_k3() else 1

    toks = a.tokenizers.split(',')
    dss = a.datasets.split(',')
    splits = a.splits.split(',')
    for tok in toks:
        for ds in dss:
            sid_p = sid_path(tok, ds)
            ck = ckpt_path(tok, ds)
            if not sid_p.exists() or not ck.exists():
                print(f'SKIP {tok} {ds}: sid={sid_p.exists()} ckpt={ck.exists()}')
                continue
            sid_len = int(load_sid(tok, ds).shape[1])
            for split in splits:
                out = STEP1 / 'beams' / f'{ds}_{tok}_{split}.npz'
                if a.skip_existing and out.exists():
                    print(f'skip existing {out.name}')
                    continue
                print(f'=== dump {tok} {ds} {split} L={sid_len} ===', flush=True)
                dump(ds, sid_p, ck, out, split, sid_len,
                     beam_size=a.beam_size, infer_size=a.infer_size)
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
