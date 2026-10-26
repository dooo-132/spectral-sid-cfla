#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump beam outputs for a PRE-EXISTING seed checkpoint, in the historical
artifact convention.

Why this exists
---------------
The S-diversity control (same family, same K, different training seed) needs
beam dumps.  Nine seed checkpoints already exist under
`<artifact_root>/experiments/seed_ensemble/ckpt/`, but only Beauty K=2 (and
Beauty K=3..5 seed 2026) had beams.  This script runs INFERENCE ONLY -- no
training, no gradient, no parameter update, no new model -- to complete the
missing beam dumps so the same diversity metrics can be computed for the
seed axis.

Convention
----------
It deliberately reproduces the HISTORICAL dumper
(`<artifact_root>/model/dump_beam_outputs.py`), which stored the **0-based**
SID-array row index in `candidate_item_ids` / `target_item_ids` with `-1` as
the "no item" sentinel (see docs/ITEM_ID_CONVENTION.md).  The copy in
`src/tiger/dump_beams.py` was later "fixed" to 1-based; using it here would mix
two id conventions inside one study, so this script keeps the historical one
and `fg_common.load_cell` canonicalises on load.

`--verify` re-dumps a cell that already has a historical dump and asserts the
two `candidate_item_ids` arrays are element-wise identical, which is the
comparability check required by the study protocol.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.tiger.dataset import GenRecDataset, DEFAULT_CODEBOOK_SIZE  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402
from src.tiger.train import TIGER, set_seed  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402


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
    raise ValueError('cannot infer vocab_size from checkpoint')


def build_config(vocab_size):
    return {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
            'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
            'vocab_size': vocab_size, 'pad_token_id': 0, 'eos_token_id': 0,
            'feed_forward_proj': 'relu'}


def build_sid_to_item(code_path):
    codes = np.load(code_path, allow_pickle=True).astype(np.int64)
    mp = {}
    dup = 0
    for idx, row in enumerate(codes):
        key = tuple(row.tolist())
        if key in mp:
            dup += 1
        mp[key] = idx                       # HISTORICAL: 0-based row index
    return mp, codes, dup


def tokens_to_sid(tokens, sid_len, codebook_size):
    tokens = [int(x) for x in tokens[:sid_len]]
    if len(tokens) < sid_len:
        return None
    sid = []
    for i, t in enumerate(tokens):
        lo, hi = i * codebook_size + 1, (i + 1) * codebook_size
        if t < lo or t > hi:
            return None
        sid.append(t - lo)
    return tuple(sid)


def decode_tokens_to_item(tokens, sid_len, sid_to_item, codebook_size):
    sid = tokens_to_sid(tokens, sid_len, codebook_size)
    return -1 if sid is None else sid_to_item.get(sid, -1)


def dump(dataset_path, code_path, ckpt_path, output_path, split, sid_len,
         dataset, beam_size=30, infer_size=96, codebook_size=DEFAULT_CODEBOOK_SIZE,
         max_len=20, seed=2025, device='cuda'):
    set_seed(seed)
    sd = strip_module_prefix(torch.load(ckpt_path, map_location='cpu'))
    vocab_size = infer_vocab_size(sd)
    model = TIGER(build_config(vocab_size))
    model.load_state_dict(sd, strict=True)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    sid_to_item, codes, dup = build_sid_to_item(code_path)
    dataset_obj = GenRecDataset(
        dataset_path=os.path.join(dataset_path, f'{split}.parquet'),
        code_path=code_path, mode='evaluation', max_len=max_len,
        codebook_size=codebook_size)
    loader = GenRecDataLoader(dataset_obj, batch_size=infer_size, shuffle=False)

    all_t, all_c, all_tok, all_s, all_tr = [], [], [], [], []
    gen_max_length = sid_len + 1
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['history'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['target'].to(dev)
            bs = input_ids.shape[0]
            outputs = model.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=gen_max_length, num_beams=beam_size,
                num_return_sequences=beam_size, return_dict_in_generate=True,
                output_scores=True)
            seq_wo_start = outputs.sequences[:, 1:]
            if seq_wo_start.shape[1] < sid_len:
                pad = torch.zeros((seq_wo_start.shape[0], sid_len - seq_wo_start.shape[1]),
                                  dtype=seq_wo_start.dtype, device=seq_wo_start.device)
                seq_wo_start = torch.cat([seq_wo_start, pad], dim=1)
            seq_wo_start = seq_wo_start[:, :sid_len]
            cand_tokens = seq_wo_start.reshape(bs, beam_size, sid_len)
            if getattr(outputs, 'sequences_scores', None) is not None:
                seq_scores = outputs.sequences_scores.reshape(bs, beam_size)
            else:
                seq_scores = torch.zeros((bs, beam_size), device=dev)
            try:
                tr = model.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices,
                    normalize_logits=True)[:, :sid_len]
                if tr.shape[1] < sid_len:
                    tr = torch.cat([tr, torch.full(
                        (tr.shape[0], sid_len - tr.shape[1]), float('nan'),
                        device=tr.device)], dim=1)
                tr = tr.reshape(bs, beam_size, sid_len)
            except Exception as exc:  # pragma: no cover
                print('WARNING transition scores:', exc)
                tr = torch.full((bs, beam_size, sid_len), float('nan'), device=dev)

            labels_np = labels.detach().cpu().numpy()
            tok_np = cand_tokens.detach().cpu().numpy()
            target_ids = []
            cand_ids = np.full((bs, beam_size), -1, dtype=np.int64)
            for i in range(bs):
                target_ids.append(decode_tokens_to_item(
                    labels_np[i], sid_len, sid_to_item, codebook_size))
                for b in range(beam_size):
                    cand_ids[i, b] = decode_tokens_to_item(
                        tok_np[i, b], sid_len, sid_to_item, codebook_size)
            all_t.append(np.asarray(target_ids, dtype=np.int64))
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
    print(f'decode health: cand={valid_ratio:.4f} target={target_ratio:.4f} '
          f'dup_sid={dup} vocab={vocab_size}')
    if valid_ratio < 0.90:
        raise RuntimeError('token layout mismatch: candidates do not decode')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, split=np.array(split), code_path=np.array(code_path),
        ckpt_path=np.array(ckpt_path), sid_len=np.array(sid_len),
        beam_size=np.array(beam_size), vocab_size=np.array(vocab_size),
        target_item_ids=target_item_ids, candidate_item_ids=candidate_item_ids,
        candidate_tokens=candidate_tokens, sequence_scores=sequence_scores,
        transition_logprobs=transition_logprobs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return candidate_item_ids


def verify_canonical():
    """Re-dump Beauty K=2 spectral canonical and compare to the historical dump."""
    ds, k, sid_len = 'Beauty', 2, 3
    code = (rp.ARTIFACT_ROOT / 'data' / ds /
            f'{ds}_t5_spectral_nonneg_pearson_v3_k{k}_sid{sid_len}.npy')
    ckpt = (rp.ARTIFACT_ROOT / 'model' / 'ckpt' /
            f'tiger_{ds}_nonneg_pearson_v3_k{k}_sid{sid_len}.pth')
    hist = (rp.ARTIFACT_ROOT / f'model/beam_dumps_{ds}_nonneg_pearson_v3' /
            f'test_k{k}_sid{sid_len}.npz')
    tmp = REPO / 'outputs' / '_tmp_verify_dump.npz'
    dump(str(rp.ARTIFACT_ROOT / 'data' / ds), str(code), str(ckpt), str(tmp),
         'test', sid_len, ds)
    new = np.load(tmp, allow_pickle=True)
    old = np.load(hist, allow_pickle=True)
    a = np.asarray(new['candidate_item_ids'])
    b = np.asarray(old['candidate_item_ids'])
    same_shape = a.shape == b.shape
    ident = bool(same_shape and np.array_equal(a, b))
    frac = float((a == b).mean()) if same_shape else 0.0
    tgt_ident = bool(np.array_equal(np.asarray(new['target_item_ids']),
                                    np.asarray(old['target_item_ids'])))
    print(f'VERIFY {ds} K={k}: shape {a.shape} vs {b.shape}  '
          f'candidate_identical={ident} match_frac={frac:.6f} '
          f'target_identical={tgt_ident}')
    tmp.unlink(missing_ok=True)
    return ident and tgt_ident


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--dataset')
    ap.add_argument('--k', type=int)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify_canonical() else 1)
    raise SystemExit('use --verify or import dump()')
