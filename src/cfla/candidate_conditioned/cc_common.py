#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared primitives for the candidate-conditioned factorization study.

The important thing this module gets right is the **candidate scorer's input schema**:
it accepts a user history and a list of candidate item ids, and nothing else.  There is
no parameter, field, key or column through which a target item, target SID, target rank
or hit flag could reach the scorer.  `tests/test_candidate_nll_no_target_access.py`
asserts that schema statically and dynamically.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared'),
           str(HERE.parent / 'adaptive_factorization'),
           str(HERE.parent / 'factorization_generality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xt_common import CODEBOOK, STEP2, TOKENIZERS, load_sid, sha256_file, sid_path
from xt_train_shared import build_cfg, shared_vocab_plan
from src.tiger.dataset import GenRecDataset
from src.tiger.dataloader import GenRecDataLoader
from src.tiger.train import TIGER

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
            'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
FACTORIZATIONS = ('spectral_k3', 'rqvae3')
F1, F2 = FACTORIZATIONS
MAXLEN = 20

# The complete, closed set of fields the candidate scorer may consume.  Anything else is
# a schema violation, not a convenience.
SCORER_INPUT_FIELDS = frozenset({'user_history_tokens', 'user_history_mask',
                                 'candidate_item_ids', 'factorization'})
FORBIDDEN_FIELD_SUBSTRINGS = ('target', 'label', 'hit', 'rank', 'is_', 'y_')


# ---------------------------------------------------------------------------
# model / SID plumbing
# ---------------------------------------------------------------------------
def load_frozen_shared(ckpt, device='cuda'):
    sd = torch.load(str(ckpt), map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    vocab = int(sd['model.shared.weight'].shape[0])
    plan = shared_vocab_plan(list(FACTORIZATIONS))
    if plan['vocab_size'] != vocab:
        raise AssertionError(f'vocab mismatch: checkpoint {vocab} vs plan {plan["vocab_size"]}')
    model = TIGER(build_cfg(vocab))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, plan


def sid_tokens_for_items(sid_array, item_ids):
    """SID-array ROW indices -> (n, L) slot-encoded tokens. -1 stays -1.

    IMPORTANT CONVENTION.  The candidate arrays written by the previous round
    (`cand_ids1`, `cand_ids2`, `cand_target` in
    `adaptive_factorization/selector/features/<dataset>_{valid,test}.npz`) hold
    **0-based SID-array row indices**, not 1-based canonical item ids.  This was
    established empirically rather than assumed: for Beauty test row 0 the canonical
    target tokens are `[237, 331, 756, 769]`, and `sid[cand_target[0]] + pos*256 + 1`
    reproduces them exactly while `sid[cand_target[0] - 1] + pos*256 + 1` does not.

    The same convention holds inside the frozen dumper (`xt_train_shared` +
    `af_dump_shared`), which builds `sid_to_item = {tuple(row): index}` over
    `enumerate(sid_array)`, so 0 is a legal value and `-1` is the "no item" sentinel.
    Passing a 1-based id here silently scores the wrong item, so the validity test uses
    `>= 0` and the row index is used directly.
    """
    item_ids = np.asarray(item_ids, dtype=np.int64)
    out = np.full((len(item_ids), sid_array.shape[1]), -1, dtype=np.int64)
    valid = (item_ids >= 0) & (item_ids < len(sid_array))
    idx = np.where(valid, item_ids, 0)
    codes = sid_array[idx]
    pos = np.arange(sid_array.shape[1], dtype=np.int64)[None, :]
    toks = codes + pos * CODEBOOK + 1
    out[valid] = toks[valid]
    return out


def build_history_bank(ds, split, plan):
    """Return (history_tokens (n, maxlen), attention_mask (n, maxlen)) per factorization.

    Uses the canonical `GenRecDataset` evaluation mode, i.e. the same history encoding
    the frozen model was trained and evaluated with.
    """
    bank = {}
    for f in FACTORIZATIONS:
        base = GenRecDataset(
            dataset_path=str(STEP2.parents[2] / 'TIGER' / 'data' / ds / f'{split}.parquet')
            if False else str(_art_data(ds, split)),
            code_path=str(sid_path(f, ds)), mode='evaluation',
            max_len=MAXLEN, codebook_size=CODEBOOK)
        loader = GenRecDataLoader(base, batch_size=512, shuffle=False)
        toks, masks = [], []
        for b in loader:
            toks.append(b['history'].numpy().astype(np.int64))
            masks.append(b['attention_mask'].numpy().astype(np.int64))
        t = np.concatenate(toks); m = np.concatenate(masks)
        ctrl = plan['control_tokens'][f]
        bank[f] = (np.concatenate([np.full((len(t), 1), ctrl, dtype=np.int64), t], axis=1),
                   np.concatenate([np.ones((len(m), 1), dtype=np.int64), m], axis=1))
    return bank


def _art_data(ds, split):
    from xt_common import ART
    return ART / 'data' / ds / f'{split}.parquet'


# ---------------------------------------------------------------------------
# the scorer: history + candidates -> NLL, and nothing else
# ---------------------------------------------------------------------------
@torch.no_grad()
def candidate_total_nll(model, history_tokens, history_mask, cand_tokens, cand_owner,
                        device='cuda', batch_size=512, want_per_token=False):
    """Teacher-forced TotalNLL of every candidate's own SID under one factorization.

    Parameters (the complete schema — see SCORER_INPUT_FIELDS)
        history_tokens : (S, T) int64 encoder input, already carrying the control token
        history_mask   : (S, T) int64
        cand_tokens    : (C, L) int64 candidate SIDs, slot-encoded
        cand_owner     : (C,) int64 index into the history rows for each candidate
    Returns
        total : (C,) float64 TotalNLL summed over the L scored positions; NaN for an
                invalid SID (-1 anywhere)

    Histories are grouped into batches and each user's candidates are flattened into a
    single forward pass, so the cost is one encoder pass per user-batch and all of that
    batch's candidates decoded together.  No target information is consumed.
    """
    C, L = cand_tokens.shape
    total = np.full(C, np.nan, dtype=np.float64)
    per_tok = np.full((C, L), np.nan, dtype=np.float64) if want_per_token else None
    valid_rows = np.where((cand_tokens >= 0).all(axis=1))[0]
    if len(valid_rows) == 0:
        return (total, per_tok) if want_per_token else total

    cand_owner = np.asarray(cand_owner, dtype=np.int64)
    owner_of = cand_owner[valid_rows]                 # owner of each valid candidate
    order = np.argsort(owner_of, kind='stable')       # candidate order grouped by user
    row_start = np.searchsorted(owner_of[order], np.arange(len(history_tokens) + 1))
    n_users = len(history_tokens)

    for u0 in range(0, n_users, batch_size):
        u1 = min(u0 + batch_size, n_users)
        rows = valid_rows[order[row_start[u0]:row_start[u1]]]
        if len(rows) == 0:
            continue
        own = cand_owner[rows]
        tok = torch.as_tensor(cand_tokens[rows], dtype=torch.long, device=device)
        B = tok.shape[0]
        inp = torch.as_tensor(history_tokens[own], dtype=torch.long, device=device)
        am = torch.as_tensor(history_mask[own], dtype=torch.long, device=device)
        bos = torch.zeros((B, 1), dtype=torch.long, device=device)
        dec_in = torch.cat([bos, tok[:, :-1]], dim=1)
        logits = model.model(input_ids=inp, attention_mask=am, decoder_input_ids=dec_in).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        # NLL = -(log-probability), summed over the L scored positions
        g = -logp.gather(2, tok.unsqueeze(-1)).squeeze(-1).double().cpu().numpy()
        total[rows] = g.sum(axis=1)
        if want_per_token:
            per_tok[rows] = g
        del logits, logp
    return (total, per_tok) if want_per_token else total


# ---------------------------------------------------------------------------
# provenance / manifests
# ---------------------------------------------------------------------------
def run_id(dataset, experiment, seed, config_hash):
    return f'{dataset}_{experiment}_{seed}_{config_hash[:8]}'


def config_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def frozen_checkpoint(ds):
    return STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth'


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str))
