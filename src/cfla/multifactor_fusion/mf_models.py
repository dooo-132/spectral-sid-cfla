#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model loading + batched candidate teacher-forced NLL for the multi-factorization
round.

Two model kinds are used, and NEITHER is trained or modified here:

  independent  the historical per-(dataset, K, affinity) TIGER checkpoint.
               Vocabulary 256*L+1, no control token.  Used for spectral_k2,
               spectral_k4 and spectral_k5 (the three factorizations whose
               candidate NLL does not already exist).

  shared       the frozen shared spectral_k3 + rqvae3 checkpoint.  Vocabulary
               1027 = 4*256 + 1 + 2 control tokens.  Used for spectral_k3 and
               rqvae3, carrying the factorization's control token at the front of
               the encoder input.  K3/RQVAE candidate NLL is REUSED from
               outputs/candidate_conditioned, not recomputed (prompt section 19).

Scoring is always batched (prompt section 23): candidates are grouped by user and
a whole user-batch's candidates go through the decoder in one forward pass.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import MODEL_KIND, ckpt_path, load_sid, shared_ckpt, tokenise  # noqa: E402


# ---------------------------------------------------------------------------
# config inference for the historical checkpoints
# ---------------------------------------------------------------------------
def _strip_module_prefix(sd):
    if all(k.startswith('module.') for k in sd.keys()):
        return {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict'):
            if key in ckpt:
                return _strip_module_prefix(ckpt[key])
        return _strip_module_prefix(ckpt)
    raise ValueError('checkpoint is not a dict')


def infer_cfg(sd):
    """Canonical TIGER architecture (configs/canonical.yaml) + vocab from the weights.

    num_heads is NOT inferred from `relative_attention_bias.weight`: its per-head
    dimension is `d_model // num_heads` while the projection width that matters is
    `d_kv * num_heads = 384`, so a shape-based guess silently yields 3 heads and
    every attention projection fails to load.  The architecture is a canonical
    constant of this project and is taken from the config, not guessed.
    """
    cfg = {}
    for key in ('model.shared.weight', 'shared.weight'):
        if key in sd:
            cfg['vocab_size'] = int(sd[key].shape[0])
            cfg['d_model'] = int(sd[key].shape[1])
            break
    enc, dec = [], []
    for k in sd:
        m = re.search(r'encoder\.block\.(\d+)\.', k)
        if m:
            enc.append(int(m.group(1)))
        m = re.search(r'decoder\.block\.(\d+)\.', k)
        if m:
            dec.append(int(m.group(1)))
    cfg['num_layers'] = max(enc) + 1 if enc else 4
    cfg['num_decoder_layers'] = max(dec) + 1 if dec else cfg['num_layers']
    for k, v in sd.items():
        if 'DenseReluDense.wi' in k or 'DenseReluDense.wi_0' in k:
            cfg['d_ff'] = int(v.shape[0])
            break
    # canonical architecture
    cfg.setdefault('d_model', 128)
    cfg.setdefault('d_ff', 1024)
    cfg.setdefault('num_heads', 6)
    cfg.setdefault('num_layers', 4)
    cfg.setdefault('num_decoder_layers', cfg['num_layers'])
    cfg['d_kv'] = 64
    cfg['dropout_rate'] = 0.1
    cfg['pad_token_id'] = 0
    cfg['eos_token_id'] = 0
    cfg['feed_forward_proj'] = 'relu'
    # structural guard: the attention projection width must match the checkpoint
    expect_attn = cfg['d_kv'] * cfg['num_heads']
    for k, v in sd.items():
        if k.endswith('SelfAttention.q.weight'):
            if tuple(v.shape) != (expect_attn, cfg['d_model']):
                raise AssertionError(
                    f'attention shape {tuple(v.shape)} != ({expect_attn}, {cfg["d_model"]}) '
                    f'for {k}')
            break
    return cfg


def _tiger(cfg):
    from src.tiger.train import TIGER
    return TIGER(cfg)


def load_independent(f, ds, device='cuda'):
    """Load a historical per-K checkpoint.  Returns a frozen eval model."""
    p = ckpt_path(f, ds)
    if not p.exists():
        raise FileNotFoundError(p)
    sd = _extract_state_dict(torch.load(str(p), map_location='cpu'))
    cfg = infer_cfg(sd)
    sid = load_sid(f, ds)
    expect = 256 * sid.shape[1] + 1
    if cfg['vocab_size'] != expect:
        raise AssertionError(f'{p}: vocab {cfg["vocab_size"]} != 256*{sid.shape[1]}+1')
    model = _tiger(cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return {'model': model, 'cfg': cfg, 'kind': 'independent',
            'checkpoint': str(p), 'control_token': None}


def load_shared(ds, device='cuda'):
    """Load the frozen shared spectral_k3+rqvae3 checkpoint."""
    p = shared_ckpt(ds)
    if not p.exists():
        raise FileNotFoundError(p)
    sd = _strip_module_prefix(torch.load(str(p), map_location='cpu'))
    cfg = infer_cfg(sd)
    from xt_common import shared_vocab_plan
    plan = shared_vocab_plan(['spectral_k3', 'rqvae3'])
    if cfg['vocab_size'] != plan['vocab_size']:
        raise AssertionError(f'{p}: vocab {cfg["vocab_size"]} != {plan["vocab_size"]}')
    model = _tiger(cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return {'model': model, 'cfg': cfg, 'kind': 'shared',
            'checkpoint': str(p), 'control_tokens': plan['control_tokens']}


def load_model_for(f, ds, device='cuda', cache=None):
    """Dispatch on MODEL_KIND; `cache` maps ('shared', ds) -> bundle."""
    kind = MODEL_KIND[f]
    if kind == 'shared':
        key = ('shared', ds)
        if cache is not None and key in cache:
            return cache[key]
        b = load_shared(ds, device=device)
        if cache is not None:
            cache[key] = b
        return b
    return load_independent(f, ds, device=device)


def encoder_input_for(bundle, f, history_tokens, history_mask):
    """Prepend the factorization control token when the model is shared."""
    if bundle['kind'] == 'shared':
        ctrl = bundle['control_tokens'][f]
        n = history_tokens.shape[0]
        c = np.full((n, 1), ctrl, dtype=np.int64)
        return (np.concatenate([c, history_tokens], axis=1),
                np.concatenate([np.ones((n, 1), dtype=np.int64), history_mask], axis=1))
    return history_tokens, history_mask


# ---------------------------------------------------------------------------
# batched teacher-forced candidate NLL
# ---------------------------------------------------------------------------
@torch.no_grad()
def candidate_total_nll(model, history_tokens, history_mask, cand_tokens, cand_owner,
                        device='cuda', batch_size=512, want_per_token=False):
    """TotalNLL of every candidate's own SID, teacher-forced.

    NLL = -(sum over the L SID positions of log P(token | history, previous tokens)).
    A positive number; a negative value would be a sign error.

    Parameters
        history_tokens (S, T) int64   encoder input (control token already applied)
        history_mask   (S, T) int64
        cand_tokens    (C, L) int64   slot-encoded candidate SIDs, -1 = invalid
        cand_owner     (C,)   int64   index into the history rows
        batch_size     int            hard cap on candidate SEQUENCES per forward
                                      pass (the predeclared chunk constant); user
                                      rows are grouped to respect it
    Returns
        (C,) float64, NaN where the candidate SID is invalid
    """
    C, L = cand_tokens.shape
    total = np.full(C, np.nan, dtype=np.float64)
    per_tok = np.full((C, L), np.nan, dtype=np.float64) if want_per_token else None
    valid_rows = np.where((cand_tokens >= 0).all(axis=1))[0]
    if len(valid_rows) == 0:
        return (total, per_tok) if want_per_token else total

    cand_owner = np.asarray(cand_owner, dtype=np.int64)
    owner_of = cand_owner[valid_rows]
    order = np.argsort(owner_of, kind='stable')
    rows_sorted = valid_rows[order]
    owners_sorted = owner_of[order]
    row_start = np.searchsorted(owners_sorted, np.arange(len(history_tokens) + 1))

    # Slice the candidate stream into blocks of at most `batch_size` SEQUENCES.
    # The block is the unit of memory control and is a predeclared constant, so the
    # peak does not depend on how many candidates a particular user happens to have.
    pos = 0
    total_c = len(rows_sorted)
    while pos < total_c:
        end = min(pos + batch_size, total_c)
        rows = rows_sorted[pos:end]
        own = owners_sorted[pos:end]
        pos = end
        if len(rows) == 0:
            continue
        tok = torch.as_tensor(cand_tokens[rows], dtype=torch.long, device=device)
        B = tok.shape[0]
        inp = torch.as_tensor(history_tokens[own], dtype=torch.long, device=device)
        am = torch.as_tensor(history_mask[own], dtype=torch.long, device=device)
        bos = torch.zeros((B, 1), dtype=torch.long, device=device)
        dec_in = torch.cat([bos, tok[:, :-1]], dim=1)
        logits = model.model(input_ids=inp, attention_mask=am,
                             decoder_input_ids=dec_in).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        g = -logp.gather(2, tok.unsqueeze(-1)).squeeze(-1).double().cpu().numpy()
        total[rows] = g.sum(axis=1)
        if want_per_token:
            per_tok[rows] = g
        del logits, logp, dec_in
    return (total, per_tok) if want_per_token else total


# ---------------------------------------------------------------------------
# history encoding (canonical GenRecDataset path, same as previous rounds)
# ---------------------------------------------------------------------------
def build_history_tokens(ds, split, sid_file_for_dataset, max_len=20, batch=512):
    """Canonical evaluation-mode history tokens/mask for one split.

    The history is tokenised with the SID array of the factorization whose model
    is being queried; that is how the model was trained.
    """
    from src.tiger.dataset import GenRecDataset
    from src.tiger.dataloader import GenRecDataLoader
    base = GenRecDataset(
        dataset_path=str(ART_DATA(ds, split)),
        code_path=str(sid_file_for_dataset), mode='evaluation',
        max_len=max_len, codebook_size=256)
    loader = GenRecDataLoader(base, batch_size=batch, shuffle=False)
    toks, masks = [], []
    for b in loader:
        toks.append(b['history'].numpy().astype(np.int64))
        masks.append(b['attention_mask'].numpy().astype(np.int64))
    return np.concatenate(toks), np.concatenate(masks)


def ART_DATA(ds, split):
    from mf_common import ART
    return ART / 'data' / ds / f'{split}.parquet'


def load_parquet_targets(ds, split):
    """Canonical 1-based target item id per user from the dataset parquet."""
    import pandas as pd
    from mf_common import ART
    df = pd.read_parquet(ART / 'data' / ds / f'{split}.parquet')
    return df
