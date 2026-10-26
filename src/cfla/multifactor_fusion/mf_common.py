#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared primitives for the Multi-Factorization Candidate-Level Fusion round.

Five factorizations of the SAME item universe:

    F1 = spectral_k2   (SID length 3)
    F2 = spectral_k3   (SID length 4)   <- shared model, control token
    F3 = spectral_k4   (SID length 5)
    F4 = spectral_k5   (SID length 6)
    F5 = rqvae3        (SID length 4)   <- shared model, control token

Two candidate pools are built from generation output only:

    C_S4(u) = C_K2 ∪ C_K3 ∪ C_K4 ∪ C_K5          (Pool A, "Spectral-4")
    C_M5(u) = C_S4(u) ∪ C_RQVAE(u)               (Pool B, "MultiFactor-5")

The target is NEVER added to a union (prompt section 16) and is read only by the
evaluation scripts, after the freeze manifest exists.

Item-id conventions (docs/ITEM_ID_CONVENTION.md):
  * historical beam dumps store the 0-based SID-array row index (`raw`);
  * canonical item id = raw + 1;
  * SID row index      = raw.
"""
from __future__ import annotations

import os as _os
import pathlib as _pl
_REPO_ROOT = str(_pl.Path(__file__).resolve().parents[3])
_ART_ROOT = _os.environ.get('CFLA_ARTIFACT_ROOT') or _os.path.join(_REPO_ROOT, 'artifacts')
_UP_ROOT = _os.environ.get('CFLA_UPSTREAM_ROOT') or _ART_ROOT
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / 'analysis' / 'mechanism' / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils import repo_paths as rp  # noqa: E402

ART = rp.ARTIFACT_ROOT                      # the frozen artifact root  (READ ONLY)
OUT = REPO / 'outputs' / 'multifactor_fusion'
VAL = OUT / 'validation'
TESTDIR = OUT / 'test'
RUNS = OUT / 'runs'
STEP2 = REPO / 'outputs' / 'cross_tokenizer_shared' / 'step2_shared_model'
CC = REPO / 'outputs' / 'candidate_conditioned'
DG = REPO / 'outputs' / 'disagreement_gate'

CODEBOOK = 256
BEAM = 30                                   # historical, never increased (section 15)
SEED = 2025

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
N_ITEMS = {'Beauty': 12101, 'Cell_Phones_and_Accessories': 10429,
           'Health_and_Personal_Care': 18534, 'Toys_and_Games': 11924}
N_TEST = {'Beauty': 22363, 'Cell_Phones_and_Accessories': 27879,
          'Health_and_Personal_Care': 38609, 'Toys_and_Games': 19412}

AFFINITY = 'nonneg_pearson_v3'

# ---------------------------------------------------------------------------
# factorization registry
# ---------------------------------------------------------------------------
S4 = ('spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5')
M5 = S4 + ('rqvae3',)
FACTORIZATIONS = M5
FLABEL = {'spectral_k2': 'K2', 'spectral_k3': 'K3', 'spectral_k4': 'K4',
          'spectral_k5': 'K5', 'rqvae3': 'RQ'}

# Which model actually SCORES each factorization (the beam source is separate, see
# `beam_path`):
#   'independent' -> its own per-(dataset,K) checkpoint, no control token
#   'shared'      -> the frozen shared spectral_k3+rqvae3 checkpoint + control token
#
# spectral_k3 uses its OWN independent checkpoint, because the candidate universe of
# this round (the independent K2/K3/K4/K5 beams) is the frozen DynamicFusion's own
# universe and the reproduction audit is bit-exact only on those beams.
MODEL_KIND = {f: ('shared' if f == 'rqvae3' else 'independent')
              for f in FACTORIZATIONS}


def sid_path(f, ds):
    if f == 'rqvae3':
        return ART / 'data' / ds / f'{ds}_t5_rqvae.npy'
    k = int(f.split('_k')[1])
    return rp.sid_array(ds, AFFINITY, k)


def ckpt_path(f, ds):
    if f == 'rqvae3':
        return ART / 'model' / 'ckpt' / f'tiger_{ds}.pth'
    k = int(f.split('_k')[1])
    return rp.checkpoint(ds, AFFINITY, k)


def beam_path(f, ds, split='test'):
    """Where this factorization's CANDIDATE SET comes from.

    Established from the artifacts, not assumed:

      * spectral_k2 / spectral_k3 / spectral_k4 / spectral_k5 -> the historical
        INDEPENDENT per-(dataset, K) beam dump in the audited repository.

        This is forced by the reproduction requirement.  The frozen
        DynamicFusion's own output (`fusion_test_dynamic_<ds>_<aff>_top_items.npz`)
        stores per-user confidences that match the independent K3 dump to 1e-16 and
        only approximate the shared model's K3 beam, so the frozen baseline was
        computed on the independent beams -- and so was its candidate universe.
        Using them makes this round's Spectral-4 universe *identical* to the frozen
        DynamicFusion's own universe, and makes the phase-B reproduction bit-exact
        (max absolute metric error 0.0 on 4/4 datasets).

      * rqvae3 -> the FROZEN SHARED model's RQVAE beam (control token 1026), which is
        the beam the previous round's candidate-NLL artifact was scored from.
        Reusing that artifact (prompt section 19) requires this source.

    Which MODEL scores each factorization is a separate decision, see MODEl_KIND /
    the `SCORER` table below.
    """
    if f == 'rqvae3':
        p = (REPO / 'outputs' / 'adaptive_factorization' / 'selector' / 'features' /
             'beams' / f'{ds}_{f}_{split}.npz')
        if p.exists():
            return p
        raise FileNotFoundError(p)
    k = int(f.split('_k')[1])
    p = ART / 'model' / f'beam_dumps_{ds}_{AFFINITY}' / f'{split}_k{k}_sid{k+1}.npz'
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def beam_source_label(f):
    return 'shared_model_beam' if f == 'rqvae3' else 'independent_beam'


# Candidate NLL that is REUSED from the frozen previous-round artifact, and the
# column it lives in.
#
#   rqvae3 -> VALID reuse: the frozen values were produced by the SAME model this
#             round uses for rqvae3 (the shared checkpoint, control token 1026), so
#             every covered pair is reused and only the uncovered remainder is
#             computed.
#   spectral_k3 -> NOT reusable.  The frozen TotalNLL_F1 column is the SHARED
#             checkpoint's likelihood of the candidate SID.  This round's spectral_k3
#             is scored by the INDEPENDENT K3 checkpoint (see MODEL_KIND), because the
#             frozen DynamicFusion -- and therefore this round's candidate universe --
#             was built on the independent canonical beams.  Reusing the shared
#             model's numbers would mix two different models' likelihoods inside one
#             factorization's zNLL scaler, which is not a legitimate reuse and would
#             silently break the cross-K fairness control.  spectral_k3 is therefore
#             computed under its own checkpoint.  `mf_reuse_audit.py` still uses the
#             frozen column as a reproducibility check of the shared scorer on the
#             pairs the frozen union covers.
REUSE_F = {'rqvae3': 'TotalNLL_F2'}


def shared_ckpt(ds):
    return STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth'


def structure_json(ds):
    return (ART / 'model' / f'beam_dumps_{ds}_{AFFINITY}' /
            f'structure_weights_{ds}_{AFFINITY}.json')


def dynamic_json(ds, split='test'):
    """Historical DynamicFusion output for a split.

    `test` -> `fusion_test_dynamic_<ds>_<aff>.json` (the frozen method output).
    `valid` -> `fusion_valid_dynamic_grid_<ds>_<aff>.json` (the VALIDATION gamma/eta
    grid search that selected the frozen values; it recorded `best_gamma`/`best_eta`
    rather than `gamma`/`eta`, which `dynamic_params()` normalises).
    """
    if split == 'test':
        return (ART / 'model' / f'beam_dumps_{ds}_{AFFINITY}' /
                f'fusion_test_dynamic_{ds}_{AFFINITY}.json')
    return (ART / 'model' / f'beam_dumps_{ds}_{AFFINITY}' /
            f'fusion_valid_dynamic_grid_{ds}_{AFFINITY}.json')


def dynamic_params(ds):
    """The FROZEN (gamma, eta) of the old DynamicFusion, read from the test output.

    These are historical constants: this round never re-searches them (prompt
    section 12).  `best_gamma`/`best_eta` in the validation grid file are the same
    values, recorded under the grid-search key names.
    """
    j = read_json(dynamic_json(ds, 'test'))
    return float(j['gamma']), float(j['eta'])


# ---------------------------------------------------------------------------
# hashing / io
# ---------------------------------------------------------------------------
def sha256_file(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))


def read_json(p):
    return json.loads(Path(p).read_text())


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_dir(dataset, experiment, seed, chash):
    return RUNS / f'{dataset}_{experiment}_{seed}_{chash[:8]}'


def hard_fail_if_exists(p):
    if Path(p).exists():
        raise SystemExit(f'HARD FAIL: run path already exists: {p}')


# ---------------------------------------------------------------------------
# SID arrays / tokenisation
# ---------------------------------------------------------------------------
_SID_CACHE = {}


def load_sid(f, ds):
    key = (f, ds)
    if key not in _SID_CACHE:
        a = np.load(sid_path(f, ds), allow_pickle=True).astype(np.int64)
        if a.ndim != 2:
            raise ValueError(f'{sid_path(f, ds)} is not 2-D: {a.shape}')
        if len(a) != N_ITEMS[ds]:
            raise ValueError(f'{sid_path(f, ds)}: {len(a)} rows != {N_ITEMS[ds]} items')
        _SID_CACHE[key] = a
    return _SID_CACHE[key]


def tokenise(codes):
    codes = np.asarray(codes, dtype=np.int64)
    return codes + np.arange(codes.shape[-1], dtype=np.int64) * CODEBOOK + 1


def sid_tokens_for_rows(sid_array, rows):
    """SID-array ROW indices (0-based, -1 = none) -> (n, L) slot-encoded tokens."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.full((len(rows), sid_array.shape[1]), -1, dtype=np.int64)
    ok = (rows >= 0) & (rows < len(sid_array))
    if ok.any():
        idx = np.where(ok, rows, 0)
        out[ok] = tokenise(sid_array[idx])[ok]
    return out


# ---------------------------------------------------------------------------
# beam dumps
# ---------------------------------------------------------------------------
def detect_item_shift(z, sid, n_users=120):
    """Row index of a dump = raw - shift.  Historical dumps: shift 0."""
    tok = np.asarray(z['candidate_tokens'])
    raw = np.asarray(z['candidate_item_ids'])
    pos = np.arange(tok.shape[2], dtype=np.int64)[None, :]
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


def row_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def load_beam_cell(f, ds, split='test', canonical=False):
    """Load one factorization's beam dump.

    Returns dict with
      items      (n, beam) int64  raw ids (0-based SID rows) or 1-based if canonical
      probs      (n, beam) float64 row-softmax of sequence_scores over valid beams
      target     (n,)      int64  same convention as items
      in_beam    (n, beam) bool
    """
    p = beam_path(f, ds, split)
    z = np.load(p, allow_pickle=True)
    sid = load_sid(f, ds)
    shift, scores = detect_item_shift(z, sid)
    raw = np.asarray(z['candidate_item_ids'])
    raw_t = np.asarray(z['target_item_ids'])
    if canonical:
        items = np.where(raw >= 0, raw - shift + 1, -1).astype(np.int64)
        target = np.where(raw_t >= 0, raw_t - shift + 1, -1).astype(np.int64)
    else:
        items = np.where(raw >= 0, raw - shift, -1).astype(np.int64)
        target = np.where(raw_t >= 0, raw_t - shift, -1).astype(np.int64)
    valid = items >= 0
    sq = np.asarray(z['sequence_scores'], dtype=np.float64)
    probs = row_softmax(np.where(valid, sq, -1e30)) * valid
    if not np.isfinite(probs).all():
        raise AssertionError(f'non-finite probabilities in {p}')
    return {'factorization': f, 'dataset': ds, 'split': split, 'path': p,
            'raw_item_id_shift': shift, 'shift_detection': scores,
            'items': items, 'probs': probs, 'target': target,
            'valid': valid, 'n_users': items.shape[0], 'beam': items.shape[1],
            'sid_len': int(sid.shape[1]), 'sha256': sha256_file(p)}


def merge_by_item(items_row, probs_row, valid_row):
    """Within ONE factorization, merge duplicate item ids by summed probability.

    Returns (uniq_items, uniq_probs) with uniq_items ascending.
    """
    it = items_row[valid_row]
    pr = probs_row[valid_row]
    if len(it) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    order = np.argsort(it, kind='stable')
    it = it[order]; pr = pr[order]
    uniq, start = np.unique(it, return_index=True)
    sums = np.add.reduceat(pr, start)
    return uniq, sums


def split_user_ids(ds, split):
    """Canonical 1-based `user` column of the split, in parquet row order.

    This is the ONLY user-id convention used in this round's artifacts: beam dumps
    carry no user column, so the row order of `data/<ds>/<split>.parquet` defines
    it, exactly as the previous rounds' artifacts do.
    """
    import pandas as pd
    df = pd.read_parquet(ART / 'data' / ds / f'{split}.parquet', columns=['user'])
    return df['user'].values.astype(np.int64)


def build_union(cells, canonical=False, user_ids=None):
    """Union over factorizations from generation output only.

    cells : ordered dict {factorization: cell} for ONE split.

    Returns dict with
      items   (n, U)   int64  union (pooled, sorted by best prob desc then item)
      probs   (n, U, F) float64 per-factorization merged probability (0 if absent)
      rank    (n, U, F) int64  1-based rank inside that factorization's beam (-1 absent)
      present (n, U, F) bool
      target  (n,)     int64
      f_order tuple of factorization names
    """
    fs = list(cells.keys())
    n = cells[fs[0]]['n_users']
    for f in fs[1:]:
        if cells[f]['n_users'] != n:
            raise AssertionError('user count mismatch across factorizations')
    # user counts are identical by construction (canonical splits)
    beam_max = max(cells[f]['beam'] for f in fs)
    F = len(fs)
    U = beam_max * F
    items = np.full((n, U), -1, dtype=np.int64)
    probs = np.zeros((n, U, F), dtype=np.float64)
    rank = np.full((n, U, F), -1, dtype=np.int64)

    for i in range(n):
        acc = {}
        for fi, f in enumerate(fs):
            c = cells[f]
            uniq, sums = merge_by_item(c['items'][i], c['probs'][i], c['valid'][i])
            if len(uniq) == 0:
                continue
            order = np.argsort(-sums, kind='stable')
            uniq = uniq[order]; sums = sums[order]
            for r, (it, s) in enumerate(zip(uniq.tolist(), sums.tolist()), start=1):
                slot = acc.get(it)
                if slot is None:
                    slot = len(acc)
                    acc[it] = slot
                    items[i, slot] = it
                probs[i, slot, fi] = s
                rank[i, slot, fi] = r
        # Deterministic order: descending best probability, then item id.
        # COPIED first: `items[i, order]` is a fancy-indexed copy, but assigning into
        # `items[i, :m]` while `items[i, ...]` is still needed would corrupt the
        # reordering (the same applies to probs/rank).
        if acc:
            m = len(acc)
            order = sorted(range(m), key=lambda j: (-probs[i, j].max(), items[i, j]))
            order = np.asarray(order, dtype=np.int64)
            items[i, :m] = items[i, order].copy()
            probs[i, :m] = probs[i, order].copy()
            rank[i, :m] = rank[i, order].copy()
    target = cells[fs[0]]['target']
    if user_ids is None:
        user_ids = np.arange(1, n + 1, dtype=np.int64)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    if len(user_ids) != n:
        raise AssertionError(f'user_ids length {len(user_ids)} != n_users {n}')
    return {'items': items, 'probs': probs, 'rank': rank,
            'present': rank >= 1, 'target': target, 'f_order': tuple(fs),
            'n_users': n, 'user': user_ids}
