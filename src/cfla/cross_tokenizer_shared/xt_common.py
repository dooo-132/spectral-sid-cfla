#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared primitives for the cross-tokenizer / shared-parameter study.

This module deliberately REUSES the already-validated generic fusion and
diversity code from the factorization-generality study
(`analysis/mechanism/factorization_generality/fg_common.py`), which was checked
element-wise against the reference `src.analysis.per_user.fuse_subset`.  The only
new things here are:

  * a registry of independent tokenizer families (Spectral / RQ-VAE / RQ-OPQ /
    OPQ), each with its own SID array, recommender checkpoint and beam dump;
  * per-file item-id convention detection (two conventions exist in the audited
    repository);
  * tokenisation / round-trip checks.

Read-only with respect to the audited repository.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / 'analysis' / 'mechanism' / 'factorization_generality')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils import repo_paths as rp  # noqa: E402
from fg_common import (  # noqa: E402  (validated generic machinery)
    _grouped_probabilities, _row_softmax, canonicalize_item_ids,
    candidate_overlap, common_rank_correlation, model_name, oracle_rank,
    paired_bootstrap_chunked, paired_delta_ci, pairwise_diversity, per_user_item_probs,
    single_rank, summarize, union_fuse,
)

ART = rp.ARTIFACT_ROOT
OUT = REPO / 'outputs' / 'cross_tokenizer_shared'
STEP1 = OUT / 'step1_cross_tokenizer'
STEP2 = OUT / 'step2_shared_model'
FIGDIR = OUT / 'figures'
for _d in (OUT, STEP1, STEP2, FIGDIR, STEP1 / 'beams', STEP2 / 'beams',
           STEP2 / 'runs', STEP2 / 'configs'):
    _d.mkdir(parents=True, exist_ok=True)

CODEBOOK = 256
BEAM = 30
SEED = 2025
DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'CellPhones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
N_ITEMS = {'Beauty': 12101, 'Cell_Phones_and_Accessories': 10429,
           'Health_and_Personal_Care': 18534, 'Toys_and_Games': 11924}
N_TEST = {'Beauty': 22363, 'Cell_Phones_and_Accessories': 27879,
          'Health_and_Personal_Care': 38609, 'Toys_and_Games': 19412}


# ---------------------------------------------------------------------------
# independent tokenizer registry
# ---------------------------------------------------------------------------
def _rqopq_sid(ds, variant):
    return (ART / 'experiments' / 'rq_opq_tiger' / 'runs' / 'rqopq_collision_v1' /
            'sids' / ds / variant / f'{ds}_t5_{variant}.npy')


def _rqopq_ckpt(ds, variant):
    return (ART / 'experiments' / 'rq_opq_tiger' / 'runs' / 'rqopq_collision_v1' /
            'checkpoints' / f'{ds}_{variant}.pth')


TOKENIZERS = {
    'spectral_k3': {
        'label': 'Spectral-K3',
        'family': 'Spectral',
        'construction': 'spectral clustering of the embedding affinity into K=3 '
                        'dimension groups, one code per group, plus an appended '
                        'collision/disambiguation token',
        'semantic_len': 3, 'L': 4, 'codebook': 256, 'vocab': 1025,
        'independent': False,
        'sid': lambda ds: (ART / 'data' / ds /
                           f'{ds}_t5_spectral_nonneg_pearson_v3_k3_sid4.npy'),
        'ckpt': lambda ds: (ART / 'model' / 'ckpt' /
                            f'tiger_{ds}_nonneg_pearson_v3_k3_sid4.pth'),
        'beam': lambda ds, split: (
            (STEP1 / 'beams' / f'{ds}_spectral_k3_{split}.npz')
            if (STEP1 / 'beams' / f'{ds}_spectral_k3_{split}.npz').exists()
            else (ART / f'model/beam_dumps_{ds}_nonneg_pearson_v3' /
                  f'{split}_k3_sid4.npz')),
        'training_log': lambda ds: ART / 'model' / 'logs' /
            f'tiger_{ds}_nonneg_pearson_v3_k3_sid4.log',
    },
    'rqvae3': {
        'label': 'RQ-VAE-3layer',
        'family': 'RQ-VAE',
        'construction': 'residual-quantised VAE, num_emb_list=[256,256,256] '
                        '(3 residual layers) over the 768-d item embedding, plus '
                        'an appended collision/disambiguation token',
        'semantic_len': 3, 'L': 4, 'codebook': 256, 'vocab': 1025,
        'independent': True,
        'sid': lambda ds: ART / 'data' / ds / f'{ds}_t5_rqvae.npy',
        'ckpt': lambda ds: ART / 'model' / 'ckpt' / f'tiger_{ds}.pth',
        'beam': lambda ds, split: STEP1 / 'beams' / f'{ds}_rqvae3_{split}.npz',
        'training_log': lambda ds: ART / 'model' / 'logs' / f'tiger_{ds}.log',
        'tokenizer_ckpt': lambda ds: sorted(
            (ART / 'rqvae' / 'ckpt' / ds).glob('*/best_collision_model.pth')),
    },
    'opq3': {
        'label': 'OPQ-3',
        'family': 'OPQ',
        'construction': 'optimised product quantisation, 3 sub-quantisers over a '
                        'learned rotation, plus an appended collision token',
        'semantic_len': 3, 'L': 4, 'codebook': 256, 'vocab': 1025,
        'independent': True,
        'sid': lambda ds: _rqopq_sid(ds, 'opq3'),
        'ckpt': lambda ds: _rqopq_ckpt(ds, 'opq3'),
        'beam': lambda ds, split: STEP1 / 'beams' / f'{ds}_opq3_{split}.npz',
        'training_log': lambda ds: (ART / 'experiments' / 'rq_opq_tiger' / 'runs' /
                                    'rqopq_collision_v1' / 'logs' /
                                    f'{ds}_opq3_train.log'),
    },
    'rq1_opq2': {
        'label': 'RQ1-OPQ2',
        'family': 'RQ-OPQ',
        'construction': 'one residual-quantisation layer plus 2 OPQ sub-quantisers, '
                        'plus an appended collision token',
        'semantic_len': 3, 'L': 4, 'codebook': 256, 'vocab': 1025,
        'independent': True,
        'sid': lambda ds: _rqopq_sid(ds, 'rq1_opq2'),
        'ckpt': lambda ds: _rqopq_ckpt(ds, 'rq1_opq2'),
        'beam': lambda ds, split: STEP1 / 'beams' / f'{ds}_rq1_opq2_{split}.npz',
        'training_log': lambda ds: (ART / 'experiments' / 'rq_opq_tiger' / 'runs' /
                                    'rqopq_collision_v1' / 'logs' /
                                    f'{ds}_rq1_opq2_train.log'),
    },
}
PRIMARY = ('spectral_k3', 'rqvae3')
SECONDARY = ('opq3', 'rq1_opq2')


def sid_path(tok, ds):
    return TOKENIZERS[tok]['sid'](ds)


def ckpt_path(tok, ds):
    return TOKENIZERS[tok]['ckpt'](ds)


def beam_path(tok, ds, split='test'):
    return TOKENIZERS[tok]['beam'](ds, split)


def sha256_file(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SID arrays and tokenisation
# ---------------------------------------------------------------------------
_SID_CACHE = {}


def load_sid(tok, ds):
    key = (tok, ds)
    if key not in _SID_CACHE:
        a = np.load(sid_path(tok, ds), allow_pickle=True).astype(np.int64)
        if a.ndim != 2:
            raise ValueError(f'{sid_path(tok, ds)} is not 2-D: {a.shape}')
        _SID_CACHE[key] = a
    return _SID_CACHE[key]


def tokenise(codes, codebook=CODEBOOK):
    codes = np.asarray(codes, dtype=np.int64)
    return codes + np.arange(codes.shape[-1], dtype=np.int64) * codebook + 1


def untokenise(tokens, codebook=CODEBOOK):
    t = np.asarray(tokens, dtype=np.int64)
    return t - np.arange(t.shape[-1], dtype=np.int64) * codebook - 1


def sid_to_item_map(codes):
    return {tuple(int(x) for x in row): i + 1 for i, row in enumerate(codes)}


def roundtrip_report(tok, ds):
    """item -> SID -> token -> SID -> item must be the identity for every item."""
    codes = load_sid(tok, ds)
    tok_arr = tokenise(codes)
    back = untokenise(tok_arr)
    sid_ident = bool(np.array_equal(back, codes))
    m = sid_to_item_map(codes)
    items = [m[tuple(int(x) for x in r)] for r in codes]
    item_ident = items == list(range(1, len(codes) + 1))
    return {
        'n_items': int(len(codes)), 'L': int(codes.shape[1]),
        'codes_min': int(codes.min()), 'codes_max': int(codes.max()),
        'injective': bool(len({tuple(r) for r in codes}) == len(codes)),
        'sid_roundtrip': sid_ident, 'item_roundtrip': bool(item_ident),
        'token_min': int(tok_arr.min()), 'token_max': int(tok_arr.max()),
        'expected_n_items': N_ITEMS[ds],
        'item_universe_matches_canonical': bool(len(codes) == N_ITEMS[ds]),
    }


def prefix_uniqueness(tok, ds):
    codes = load_sid(tok, ds)
    k = TOKENIZERS[tok]['semantic_len']
    return len({tuple(r[:k]) for r in codes}) / len(codes)


# ---------------------------------------------------------------------------
# beam dumps
# ---------------------------------------------------------------------------
def detect_item_shift(z, sid, n_users=120):
    """Recover the raw-id convention of a beam dump: row = raw - shift.

    Canonical item id = raw - shift + 1.  Historical family dumps use shift 0;
    dumps written by the later "fixed" dumper use shift 1.
    """
    tok = np.asarray(z['candidate_tokens'])
    raw = np.asarray(z['candidate_item_ids'])
    L = tok.shape[2]
    pos = np.arange(L, dtype=np.int64)[None, :]
    n = min(n_users, raw.shape[0])
    scores = {}
    for shift in (0, 1):
        idx = raw[:n] - shift
        ok = tot = 0
        for i in range(n):
            m = idx[i] >= 0
            if not m.any():
                continue
            ids = idx[i][m]
            if ids.max() >= len(sid):
                continue
            rec = sid[ids] + pos * CODEBOOK + 1
            ok += int((rec == tok[i][m]).all(axis=1).sum())
            tot += int(m.sum())
        scores[shift] = (ok / tot) if tot else float('nan')
    best = max(scores, key=lambda s: scores[s] if np.isfinite(scores[s]) else -1.0)
    if not np.isfinite(scores[best]) or scores[best] < 0.99:
        raise AssertionError(f'cannot determine item-id shift: {scores}')
    return int(best), scores


def load_cell(tok, ds, split='test', beam_override=None, want_sha=False):
    """Load one tokenizer's beam dump as a fusion-compatible cell."""
    p = Path(beam_override) if beam_override is not None else beam_path(tok, ds, split)
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    raw_items = np.asarray(z['candidate_item_ids'])
    raw_target = np.asarray(z['target_item_ids'])
    sid = load_sid(tok, ds)
    shift, scores = detect_item_shift(z, sid)
    items = np.where(raw_items >= 0, raw_items - shift + 1, -1).astype(np.int64)
    target = np.where(raw_target >= 0, raw_target - shift + 1, -1).astype(np.int64)
    if target.max() > len(sid):
        raise AssertionError(f'{p}: target {target.max()} exceeds {len(sid)}')
    s = np.asarray(z['sequence_scores'], dtype=np.float64)
    probs = _row_softmax(s, raw_items >= 0)
    if not np.isfinite(probs).all():
        raise AssertionError(f'non-finite probabilities in {p}')
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError(f'beam probabilities do not sum to 1 in {p}')
    cell = {
        'tokenizer': tok, 'dataset': ds, 'split': split, 'path': p,
        'raw_item_id_shift': shift, 'shift_detection': scores,
        'items': items, 'probs': probs, 'target': target,
        'n_users': int(items.shape[0]), 'beam': int(items.shape[1]),
        'z': z,
    }
    if want_sha:
        cell['sha256'] = sha256_file(p)
    return cell


# ---------------------------------------------------------------------------
# vocab audit for the shared model
# ---------------------------------------------------------------------------
def shared_vocab_plan(factorizations):
    """Control-token ids for a factorisation-controlled shared vocabulary.

    SID tokens occupy [1, L*codebook] because token = code + pos*codebook + 1;
    token 0 is PAD/EOS.  Control tokens therefore start at L*codebook + 1.
    """
    Ls = {TOKENIZERS[f]['L'] for f in factorizations}
    if len(Ls) != 1:
        raise ValueError(f'factorizations have different L: {Ls}')
    L = Ls.pop()
    base = L * CODEBOOK + 1              # 1025 for L = 4
    plan = {f: base + i for i, f in enumerate(factorizations)}
    return {'L': L, 'sid_token_range': [1, L * CODEBOOK],
            'pad_token_id': 0, 'control_tokens': plan,
            'vocab_size': base + len(factorizations)}


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str))
