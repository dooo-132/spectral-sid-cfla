#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher-forced evaluation of the ground-truth target SID.

For each test instance and each granularity K, feed the history and the
GROUND-TRUTH target SID (K semantic codes + 1 disambiguation code, encoded in the
slot-wise token layout) and read the model's log-probability of each target token
under the correct prefix:

    log p(c_1 | history)
    log p(c_2 | history, c_1)
    ...
    log p(c_L | history, c_1..c_{L-1})        with L = K+1

Design notes
------------
* Evaluation ONLY. The checkpoint is loaded and the model put in eval mode; no
  training, no gradient, no parameter update.
* Only the L = K+1 real item-ID tokens are scored. BOS is the decoder start token
  and is not scored. PAD/EOS is the shared id 0; whether an EOS step is also
  scored is recorded separately as `training_style_nll` so the two conventions
  are never conflated.
* Token ids follow `dataset.item2code`: token = code + position*256 + 1.
* Row indexing: dataset item id i corresponds to SID-array row i-1.
"""
from __future__ import annotations

import numpy as np
import torch

from src.tiger.dataset import DEFAULT_CODEBOOK_SIZE


def load_model_from_ckpt(ckpt_path, device='cuda'):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.tiger.train import TIGER

    sd = torch.load(ckpt_path, map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    vocab = None
    for key in ('model.shared.weight', 'model.encoder.embed_tokens.weight',
                'model.lm_head.weight'):
        if key in sd:
            vocab = sd[key].shape[0]
            break
    if vocab is None:
        raise ValueError('cannot infer vocab_size from checkpoint')
    cfg = dict(num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024,
               num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=vocab,
               pad_token_id=0, eos_token_id=0, feed_forward_proj='relu')
    model = TIGER(cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, vocab


@torch.no_grad()
def teacher_forced_logprobs(model, histories, target_tokens, device='cuda',
                            batch_size=64, attention_mask=None):
    """
    histories       : (n, S) int64 flattened history token ids (padded with 0)
    target_tokens   : (n, L) int64 token ids of the ground-truth target SID
    attention_mask  : (n, S) int64, 1 where the history token is real. Taken from
                      the dataset loader; derived from `histories != 0` only as a
                      fallback (the two agree for this pipeline).
    returns         : (n, L) float64 log-probabilities of each target token

    Only the L = K+1 real item-ID tokens are scored. The decoder is fed
    BOS + target[:-1], so position 0 is conditioned on the history alone.
    """
    n, L = target_tokens.shape
    out = np.zeros((n, L), dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        inp = torch.as_tensor(histories[s:e], dtype=torch.long, device=device)
        tgt = torch.as_tensor(target_tokens[s:e], dtype=torch.long, device=device)
        B = inp.shape[0]
        bos = torch.zeros((B, 1), dtype=torch.long, device=device)
        # decoder input: BOS + target[0..L-2]
        dec_in = torch.cat([bos, tgt[:, :-1]], dim=1)
        if attention_mask is not None:
            attn = torch.as_tensor(attention_mask[s:e], dtype=torch.long, device=device)
        else:
            attn = (inp != 0).long()
        logits = model.model(input_ids=inp, attention_mask=attn,
                             decoder_input_ids=dec_in).logits      # (B, L, V)
        logp = torch.log_softmax(logits.float(), dim=-1)
        gathered = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L)
        out[s:e] = gathered.double().cpu().numpy()
    return out


def tokenise_sid(codes, codebook_size=DEFAULT_CODEBOOK_SIZE):
    """SID codes -> slot-wise tokens (same shape).

    Accepts (L,) or (n, L). Position offsets are applied along the LAST axis.
    """
    codes = np.asarray(codes, dtype=np.int64)
    n_pos = codes.shape[-1]
    return codes + np.arange(n_pos, dtype=np.int64) * codebook_size + 1
