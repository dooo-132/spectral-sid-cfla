#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_m5_candidate_nll.py -- the missing M5 CFLA input producer.

WHY THIS FILE EXISTS
--------------------
`build_m5_rq5.py` emits the M5 CFLA rows with the status
`BLOCKED_NO_CANDIDATE_NLL` and names this file as the unblock command:

    cd the development workspace && .venv_cuda_compatible/bin/python \\
      outputs/final_hyperparam_efficiency/drivers/score_m5_candidate_nll.py \\
      --datasets <ds,...> --splits valid,test \\
      --factorizations spectral_k2,spectral_k3,spectral_k4,spectral_k5,rqvae5 \\
      --device cuda

CFLA (`logsumexp_F(-zNLL_F)`, zero trainable parameters) needs, for every
candidate of the five-arm M5 union and EVERY factorization F:

    NLL_F(u, i)  = TotalNLL of item i's own SID under F's own checkpoint,
                   teacher-forced on user u's own history,
    (mu_F, sigma_F)  fitted on the VALID split only.

This driver produces exactly those two things:

    outputs/final_hyperparam_efficiency/m5_rq5/candidate_nll/<dataset>/
        valid_m5_candidate_nll.parquet
        test_m5_candidate_nll.parquet
        <split>_m5_candidate_nll.metadata.json
        <split>_m5_candidate_nll.status.json
    outputs/final_hyperparam_efficiency/m5_rq5/candidate_nll/
        validation_nll_scalers.json

WHAT IS REUSED (nothing certified is re-implemented)
----------------------------------------------------
* the NLL scorer `mf_models.candidate_total_nll` (teacher-forced, full SID
  including the collision slot, own SID, own history) -- UNCHANGED;
* the loaders `mf_models.load_model_for` / `load_independent` / `load_shared`,
  `mf_models.encoder_input_for` and `mf_models.build_history_tokens` -- the
  RQ-VAE5 arm is registered by PATCHING `mf_common`'s in-memory registry
  (sid_path / ckpt_path / MODEL_KIND); no checkpoint-loading code is written
  here;
* `fusion_audit.build_union` (via `build_m5_rq5.load_scope_inputs` /
  `set_nk` / `build_scope_union`) so the union is bit-identical to the one the
  M5 metrics are computed on;
* `build_m5_rq5.load_scope_inputs`, which EMPIRICALLY measures the item-id
  convention of every beam dump by token reconstruction and applies the
  `de_canonicalize_view` (-1) adapter to the already-1-based RQ-VAE5 dumps so
  that `build_union`'s `+1` restores the canonical id;
* the audited row-alignment writer `write_scores_back_to_original_rows` +
  `assert_permutation` (sha256-pinned to BUG_FIX_AUDIT.md) and the full
  verification wrapper `build_m5_rq5.write_back_and_verify`, which re-scores a
  seeded sample of rows DIRECTLY, in the original row order, and requires an
  exact match.

HARD RULES ENFORCED HERE
------------------------
* no writes outside `outputs/final_hyperparam_efficiency/`;
* `sys.dont_write_bytecode = True` before anything is imported, so no `.pyc`
  can be dropped into the read-only historical roots;
* the GPU is probed with `nvidia-smi` only (never a CUDA context) and the
  driver REFUSES to run while a training job or foreign compute app owns the
  device, unless `--allow-busy-gpu` is passed -- and that fact is then recorded
  in the output metadata;
* a finished cell (parquet + `completed` status + matching input hashes) is
  SKIPPED, never rewritten; an output path that exists without a completed
  status is a HARD FAIL;
* NaN, shape mismatch, non-permutation index, out-of-range item id, missing
  input, ambiguous item-id convention and a failed post-write identity check
  are all hard failures -- there is no `except Exception: pass` anywhere.

Usage
    # validation only, never touches the GPU, writes nothing
    python3 score_m5_candidate_nll.py --check
    python3 score_m5_candidate_nll.py --check --check-models

    # the real thing (after the queue has finished and the device is idle)
    python3 score_m5_candidate_nll.py --datasets <ds,...> --splits valid,test \\
        --factorizations spectral_k2,spectral_k3,spectral_k4,spectral_k5,rqvae5 \\
        --device cuda
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
import sys as _sys

# The historical roots are READ-ONLY.  Importing `fusion_audit` / `phase5_fusion`
# / `mf_common` from them must not drop a `__pycache__/*.pyc` anywhere.  Refuse
# bytecode caching before anything at all is imported.
_sys.dont_write_bytecode = True

import argparse                                                        # noqa: E402
import hashlib                                                         # noqa: E402
import json                                                            # noqa: E402
import os                                                              # noqa: E402
import time                                                            # noqa: E402
from datetime import datetime, timezone                                # noqa: E402
from pathlib import Path                                               # noqa: E402

import numpy as np                                                     # noqa: E402

# ===========================================================================
# roots and constants
# ===========================================================================
REPO = Path(_REPO_ROOT + '')
ART_ROOT = Path(_ART_ROOT + '')                      # READ-ONLY
UPSTREAM_ROOT = Path(_UP_ROOT + '')      # READ-ONLY
OUT = rp.OUTPUT_ROOT / 'final_hyperparam_efficiency'        # only writable root
DRIVERS = Path(__file__).resolve().parent
MF_DIR = REPO / 'src' / 'cfla' / 'multifactor_fusion'
CAND = OUT / 'm5_rq5' / 'candidate_nll'
CAND_HISTORY = CAND / 'history'
CAND_RUNS = CAND / 'runs'
SCALERS = CAND / 'validation_nll_scalers.json'

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
FACTORIZATIONS = ['spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5',
                  'rqvae5']
# Column label per factorization, in the repaired round's `NLL_<L>` / `P_<L>`
# naming.  `RQ5` is new: the historical label `RQ` belongs to the RQ-VAE3 arm.
LABEL = {'spectral_k2': 'K2', 'spectral_k3': 'K3', 'spectral_k4': 'K4',
         'spectral_k5': 'K5', 'rqvae5': 'RQ5'}
SCOPE = 'm5'
DEFAULT_AFFINITY = 'nonneg_pearson_v3'
DEFAULT_CHUNK = 512                    # same predeclared constant as PHASE A
DEFAULT_GPU_MEM_LIMIT_MIB = 1000.0
DEFAULT_VERIFY_SAMPLE = 20000          # rows re-scored directly as the identity check
VERIFY_SEED = 20260912
# Calibrated tolerance for the DIRECT re-score identity check.  The bulk pass and
# the direct pass batch the same rows differently and the scorer is float32, so they
# agree only to ~1e-6 (see analysis/mechanism/final_ablation/fa_independent.py:94,
# "float32 batching noise ~2e-5").  1e-4 is the tolerance this project already uses
# for float32 NLL comparisons elsewhere; it is ~1e4 times smaller than the ~1 nat
# gap a genuine row-permutation error produces, and the driver PROVES on every call
# that the historical buggy write-back is still rejected at this value.
DEFAULT_VERIFY_ATOL = 1e-4
STATUS_COMPLETED = 'completed'

# Every column this driver writes, in order.  The parquet's ROW ORDER is the
# union's own row order: user-major, and within a user ascending union position,
# i.e. exactly `np.where(valid_mask)` of `fusion_audit.build_union`.  `user_row`
# is the 0-based row of the dataset split and the index into the history built
# by `build_history_tokens`; `user` is the canonical 1-based id (`user_row + 1`)
# used by the repaired round's parquets.
BASE_COLS = ['dataset', 'split', 'user', 'user_row', 'pos', 'candidate_item']


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_file(p) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def human_size(n) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if n < 1024 or unit == 'GiB':
            return f'{n:.1f}{unit}'
        n /= 1024.0
    return f'{n:.1f}GiB'


# ===========================================================================
# certified machinery, imported from the frozen trees
# ===========================================================================
def load_certified():
    """Import the M5 driver + the certified multi-factorization modules.

    `build_m5_rq5` is imported as a module (its own `sys.dont_write_bytecode`
    runs at import time).  It gives us `load_scope_inputs` (empirical item-id
    convention detection + the RQ-VAE5 `-1` adapter), `set_nk`,
    `build_scope_union`, the pinned `assert_permutation` /
    `write_scores_back_to_original_rows`, `write_back_and_verify`, and the GPU
    guard helpers -- none of which is re-implemented here.
    """
    if str(DRIVERS) not in _sys.path:
        _sys.path.insert(0, str(DRIVERS))
    if str(MF_DIR) not in _sys.path:
        _sys.path.insert(0, str(MF_DIR))
    import build_m5_rq5 as bm
    import mf_common as mc
    import mf_models as mm
    return bm, mc, mm


def register_rqvae5_arm(bm, mc, mm):
    """Register the RQ-VAE5 arm in the in-memory factorization registry.

    `mf_common` knows `spectral_k2..k5` and the historical `rqvae3`; the arm
    trained this round is `rqvae5`.  Its checkpoint and SID array live under
    this round's writable root, at the paths `build_m5_rq5.py` already uses.

    Nothing is re-implemented: `mf_models.load_model_for('rqvae5', ds)` then
    dispatches to the certified `load_independent`, which resolves the
    checkpoint through `ckpt_path`, the SID array through `load_sid`
    (`N_ITEMS`-checked, cached) and the vocabulary through `infer_cfg`.

    The patches are in-memory only; `mf_common.py` on disk is untouched.
    """
    orig_sid, orig_ckpt = mc.sid_path, mc.ckpt_path

    def sid_path_patched(f, ds):
        if f == 'rqvae5':
            return bm.rqvae5_sid_path(ds)
        return orig_sid(f, ds)

    def ckpt_path_patched(f, ds):
        if f == 'rqvae5':
            return bm.rqvae5_ckpt_path(ds)
        return orig_ckpt(f, ds)

    mc.sid_path = sid_path_patched
    mc.ckpt_path = ckpt_path_patched
    # `load_model_for` resolves MODEL_KIND from mf_models' globals; `load_independent`
    # resolves ckpt_path from mf_models' globals (`load_sid` is the same function
    # object as mf_common.load_sid, so it already sees the patched sid_path).
    mc.MODEL_KIND['rqvae5'] = 'independent'
    mm.MODEL_KIND['rqvae5'] = 'independent'
    mm.ckpt_path = ckpt_path_patched
    return {'registered_factorization': 'rqvae5',
            'kind': 'independent',
            'sid_path': str(bm.rqvae5_sid_path('<ds>')),
            'checkpoint': str(bm.rqvae5_ckpt_path('<ds>')),
            'patch_surface': ['mf_common.sid_path', 'mf_common.ckpt_path',
                              'mf_common.MODEL_KIND', 'mf_models.ckpt_path',
                              'mf_models.MODEL_KIND'],
            'note': 'in-memory only; the module sources are never modified'}


# ===========================================================================
# gpu guard -- nvidia-smi only, never a CUDA context
# ===========================================================================
def gpu_state(bm, mem_limit_mib):
    info = bm.probe_gpu()
    procs = bm.scan_host_processes()
    busy, reasons = bm.assess_contention(info, mem_limit_mib, procs)
    return info, procs, busy, reasons


def report_gpu(bm, info, busy, reasons, mem_limit_mib):
    bm.report_gpu_state(info, busy, reasons, mem_limit_mib)
    try:
        import torch
        print(f'  this process holds a CUDA context: {torch.cuda.is_initialized()}')
    except Exception as e:                                    # pragma: no cover
        print(f'  torch not importable in this process: {type(e).__name__}: {e}')


# ===========================================================================
# union rows
# ===========================================================================
def union_rows(bm, fa, p5, ds, affinity, split):
    """The M5 union's own rows, exactly as the M5 metrics see them.

    Returns (record, U) where `record` carries n_users / n_pairs / provenance,
    or (record, None) when a required input is missing (the caller decides).
    """
    dumps, prov = bm.load_scope_inputs(fa, ds, affinity, split, SCOPE)
    if dumps is None:
        return {'dataset': ds, 'split': split, 'status': 'MISSING_INPUT',
                'missing_input': str(prov)}, None

    nk = bm.set_nk(fa, p5, SCOPE)
    if nk != len(bm.ARMS[SCOPE]):
        raise AssertionError(f'N_K {nk} != {len(bm.ARMS[SCOPE])} arms')
    U = bm.build_scope_union(fa, dumps, SCOPE)
    items_pad, score_tensor, target, target_pos, valid_mask = U
    if items_pad.shape[:2] != valid_mask.shape:
        raise AssertionError(f'{ds}/{split}: union shape mismatch')
    if score_tensor.shape != (items_pad.shape[0], items_pad.shape[1], nk):
        raise AssertionError(
            f'{ds}/{split}: score_tensor {score_tensor.shape} != '
            f'{(items_pad.shape[0], items_pad.shape[1], nk)}')

    rows_i, rows_p = np.where(valid_mask)
    cands = np.asarray(items_pad[rows_i, rows_p], dtype=np.int64)   # canonical 1-based
    users = rows_i.astype(np.int64)            # 0-based user row == history row
    n_users = int(items_pad.shape[0])
    n_pairs = int(cands.size)
    if n_pairs == 0:
        raise AssertionError(f'{ds}/{split}: the union is empty')
    if cands.min() < 1:
        raise AssertionError(f'{ds}/{split}: non-canonical candidate id '
                             f'{int(cands.min())} (canonical ids are 1-based)')
    keys = pair_keys(users, cands)
    if len(np.unique(keys)) != n_pairs:
        raise AssertionError(f'{ds}/{split}: duplicate (user, candidate) pairs in '
                             f'the union -- the row identity check would be vacuous')

    # ------------------------------------------------------------------
    # Per-arm SOURCE RANK of every union candidate (§12 `source_rank_if_available`).
    #
    # `dumps[m]['candidate_item_ids']` is (n_users, k) in arm m's own id
    # convention and in beam-rank order.  `build_union` canonicalises every arm
    # with `+1`, so a canonical item id equals `dump_ids[u, j] + 1`.  The join is
    # a sorted-key searchsorted, i.e. an exact integer equality test -- never a
    # float tolerance.  Rank is 1-based; 0 means "this arm did not propose it".
    # ------------------------------------------------------------------
    BIG = 10 ** 7
    arm_ranks = []
    for m, d in enumerate(dumps):
        arr = np.asarray(d['candidate_item_ids'], dtype=np.int64)
        if arr.shape[0] != n_users:
            raise AssertionError(
                f'{ds}/{split}: arm {m} dump has {arr.shape[0]} user rows but the '
                f'union has {n_users} -- the rank join would be misaligned')
        r = np.zeros(n_pairs, dtype=np.int64)
        uu, jj = np.where(arr >= 0)
        if uu.size:
            ak = uu.astype(np.int64) * BIG + (arr[uu, jj] + 1)
            o = np.argsort(ak, kind='stable')
            ak_s, rk_s = ak[o], (jj[o] + 1).astype(np.int64)
            pos = np.clip(np.searchsorted(ak_s, keys), 0, ak_s.size - 1)
            hit = ak_s[pos] == keys
            r[hit] = rk_s[pos][hit]
        arm_ranks.append(r)
    n_ranked = int(sum(int((r > 0).sum()) for r in arm_ranks))
    if n_ranked == 0:
        raise AssertionError(
            f'{ds}/{split}: the per-arm source-rank join attributed NO union '
            f'candidate to any arm -- the id convention of `candidate_item_ids` '
            f'is not the one `build_union` canonicalises')

    rec = {'dataset': ds, 'split': split, 'status': 'OK',
           'affinity': affinity, 'scope': SCOPE, 'n_users': n_users,
           'n_pairs': n_pairs, 'union_shape': list(items_pad.shape),
           'n_users_target_in_union': int((target_pos >= 0).sum()),
           'mean_union_size': float(valid_mask.sum(1).mean()),
           'arm_names': list(bm.ARMS[SCOPE]), 'n_arms': nk,
           'source_rank_attributed': n_ranked,
           'source_rank_total_slots': int(n_pairs * len(dumps)),
           'provenance': prov,
           # `de_canonicalize_view` shifted the already-1-based RQ-VAE5 ids down by
           # one so `build_union`'s `canonicalize_item_ids(+1)` restores them.
           'rqvae5_adapter': 'de_canonicalize_view (-1) then build_union (+1)'}
    return rec, (U, cands, users, arm_ranks)


def pair_keys(users, cands):
    """(user_row, canonical item id) -> one exact integer key.

    Extracted from `union_rows` so the duplicate guard is directly testable
    (regression check E).  Item ids are < 2e4 here, so the 1e7 stride cannot
    collide; the comparison is integer equality, never a float tolerance.
    """
    users = np.asarray(users, dtype=np.int64)
    cands = np.asarray(cands, dtype=np.int64)
    if users.min() < 0:
        raise AssertionError('user rows must be non-negative')
    if cands.min() < 1:
        raise AssertionError('candidate ids must be canonical (1-based)')
    return users * (10 ** 7) + cands


def union_parts(U):
    """Unpack `union_rows`' return value into (union 5-tuple, per-arm ranks).

    `U` is `(union_5tuple, cands, users, arm_ranks)` where the 5-tuple is
    `(items_pad, score_tensor, target, target_pos, valid_mask)` -- the same
    contract `build_m5_rq5.pu_for` unpacks.  This helper exists so that no call
    site can repeat the historical `np.asarray(U[0])` mistake.
    """
    if not isinstance(U, tuple) or len(U) != 4:
        raise AssertionError(
            f'union_parts expects union_rows\' 4-tuple '
            f'(union5, cands, users, arm_ranks); got '
            f'{type(U).__name__} of length {len(U) if hasattr(U, "__len__") else "?"}')
    union5, _cands, _users, arm_ranks = U
    if len(union5) != 5:
        raise AssertionError(
            f'the union tuple must be (items_pad, score_tensor, target, '
            f'target_pos, valid_mask); got {len(union5)} elements')
    return union5, arm_ranks


def convention_audit(rec, required_match):
    """Refuse to score a dump whose item-id convention is not empirically settled.

    `build_m5_rq5.load_scope_inputs` already hard-fails on an AMBIGUOUS dump
    (neither convention >= 0.999, or the wrong one above 0.01).  This adds the
    stricter requirement used by this driver: the WINNING convention must
    reconstruct the SID codes at `--require-convention-match` (default 1.0) on
    every valid beam slot, and the RQ-VAE5 arm's adapter must have measured
    `canonical_shift == 0` (i.e. the dump arrived already 1-based).
    """
    out = {}
    for p in rec['provenance']:
        m = p.get('item_id_convention_measured')
        if m is None:
            raise AssertionError(
                f"{rec['dataset']}/{rec['split']}/{p['arm']}: no empirical "
                f"item-id convention measurement -- refusing to score")
        name = m['convention']
        # `verify_item_id_convention` stores the two measured match rates under
        # `as_zero_based_row` / `as_canonical_one_based` and names the winner in
        # `convention`; the winner's measurement is what must reconstruct exactly.
        rate = m['as_' + name]['match_rate']
        if rate < required_match:
            raise AssertionError(
                f"{rec['dataset']}/{rec['split']}/{p['arm']}: the measured "
                f"convention {name!r} reconstructs only {rate:.6f} of valid beam "
                f"slots (required >= {required_match}); refusing to score a dump "
                f"whose ids are not settled")
        if p['arm'] == 'rqvae5' and m['canonical_shift'] != 0:
            raise AssertionError(
                f"{rec['dataset']}/{rec['split']}/rqvae5: the adapter assumes an "
                f"already-canonical dump but the measurement says {name!r} "
                f"(canonical shift {m['canonical_shift']})")
        out[p['arm']] = {'convention': name, 'match_rate': rate,
                         'canonical_shift': m['canonical_shift'],
                         'n_valid_slots': m['n_valid_slots'],
                         'n_padded_slots': m['n_padded_slots']}
    return out


# ===========================================================================
# scoring one (dataset, split)
# ===========================================================================
def get_model(bm, mm, ds, f, device, cache):
    """`load_model_for` (certified loader) with a per-dataset residency cache."""
    key = (ds, f)
    if key not in cache:
        print(f'    loading model {f} for {ds} (device={device}) ...', flush=True)
        t0 = time.time()
        cache[key] = mm.load_model_for(f, ds, device=device, cache=None)
        print(f'      {Path(cache[key]["checkpoint"]).name} '
              f'kind={cache[key]["kind"]} '
              f'vocab={cache[key]["cfg"]["vocab_size"]} '
              f'({time.time() - t0:.1f}s)', flush=True)
    return cache[key]


def write_back_and_verify_calibrated(bm, scores_sorted, order, users, candidates,
                                     score_fn, what='scores', verify_all=True,
                                     sample_seed=20260912, atol=1e-4):
    """`bm.write_back_and_verify` with a CALIBRATED tolerance on the direct re-score.

    WHY THIS EXISTS (added 2026-10-13, final paper completion round)
    ---------------------------------------------------------------
    `build_m5_rq5.write_back_and_verify` compares the directly re-scored rows to the
    written column with `!=`, i.e. exact float equality.  That is correct for a
    deterministic scorer but not for this one: the bulk pass scores all n rows in
    chunks of `--chunk`, while the direct re-score scores only the sampled rows, so
    the two see different batch compositions.  The model runs in float32, so the two
    agree only to ~1e-6 -- the round's own note in
    `analysis/mechanism/final_ablation/fa_independent.py:94` documents "float32
    batching noise ~2e-5".  Exact equality therefore rejects a CORRECT write-back.

    Observed on the real run (2026-10-13, Beauty/valid/spectral_k2):
        30/20000 rows differed, max |diff| = 1.04e-6
    which is float32 round-off, not a mis-ordered stream: a genuine permutation error
    puts a DIFFERENT ROW's value in a cell, and those differ by order 1 nat.

    WHAT IS KEPT EXACT
    ------------------
    Every structural check stays exact and sha-pinned:
      * `assert_permutation` on `order` (injectivity + range);
      * `write_scores_back_to_original_rows` (the repaired scatter);
      * `out[order] == scores_sorted` -- the scatter identity, atol = 0;
      * the pair-identity dict check, atol = 0;
      * uniqueness of the (user, candidate) keys.
    Only the numeric comparison of the independently re-scored sample is given `atol`.

    THE TOLERANCE IS PROVED ADMISSIBLE ON EVERY CALL
    ------------------------------------------------
    A tolerance is only legitimate if it would still catch the defect it replaced
    exactness for.  So the historical buggy write-back `scores_sorted[order]` is
    simulated and MUST exceed `atol` on a large number of verified rows.  If it does
    not, the call hard-fails: the tolerance would have bought nothing.
    """
    scores_sorted = np.asarray(scores_sorted, dtype=np.float64)
    order = bm.assert_permutation(order, len(scores_sorted))     # EXACT, unchanged
    users = np.asarray(users)
    candidates = np.asarray(candidates)
    if users.shape != candidates.shape or users.shape[0] != len(order):
        raise AssertionError(
            f'{what}: users/candidates shape {users.shape}/{candidates.shape} do not '
            f'match order length {len(order)}')
    if not np.isfinite(scores_sorted).all():
        raise AssertionError(f'{what}: non-finite value in the sorted score stream')
    if not callable(score_fn):
        raise AssertionError(f'{what}: score_fn is required')

    key_sorted = list(zip(users[order].tolist(), candidates[order].tolist()))
    if len(set(key_sorted)) != len(key_sorted):
        raise AssertionError(f'{what}: duplicate (user, candidate) pairs in the stream')
    truth = dict(zip(key_sorted, scores_sorted.tolist()))

    out = bm.write_scores_back_to_original_rows(scores_sorted, order)
    for i, key in enumerate(zip(users.tolist(), candidates.tolist())):
        want = truth[key]
        if out[i] != want:                                       # EXACT, unchanged
            raise AssertionError(
                f'{what}: row {i} holds {out[i]!r} but the stream carried {want!r} '
                f'for its own pair (user={key[0]}, candidate={key[1]}) -- the '
                f'write-back is not the inverse of the sort')

    n = len(order)
    if verify_all or n <= 20000:
        idx = np.arange(n)
    else:
        idx = np.sort(np.random.default_rng(sample_seed).choice(
            n, size=20000, replace=False))
    want_direct = np.asarray(score_fn(users[idx], candidates[idx]), dtype=np.float64)
    got = out[idx]
    if not np.isfinite(want_direct).all():
        raise AssertionError(f'{what}: the direct re-score returned a non-finite value')

    diff = np.abs(got - want_direct)
    worst = float(diff.max()) if diff.size else 0.0
    bad = np.flatnonzero(diff > atol)
    if bad.size:
        i = int(idx[bad[0]])
        raise AssertionError(
            f'{what}: {bad.size}/{len(idx)} verified rows differ from the direct '
            f're-score by more than the calibrated atol={atol:g}; row {i} names '
            f'(user={int(users[i])}, candidate={int(candidates[i])}), written '
            f'{got[bad[0]]!r}, direct {want_direct[bad[0]]!r} (|d|={diff[bad[0]]:.3e}). '
            f'A permutation error moves a DIFFERENT row\'s value here, so a gap this '
            f'large is not round-off.')

    # --- the tolerance must still catch the historical defect -----------------
    buggy = scores_sorted[order]                 # `scores_sorted[order]`, the old bug
    is_involution = bool(np.array_equal(order[order], np.arange(n)))
    buggy_bad = int((np.abs(buggy[idx] - want_direct) > atol).sum())
    if not is_involution and buggy_bad == 0:
        raise AssertionError(
            f'{what}: atol={atol:g} is too loose -- the historical buggy write-back '
            f'`scores_sorted[order]` would pass it on all {len(idx)} verified rows, '
            f'so the tolerance has destroyed the check it replaced')

    return out, {'verify_mode': 'all' if (verify_all or n <= 20000)
                 else 'seeded-sample-20000',
                 'n_verified': int(len(idx)), 'atol': float(atol),
                 'max_abs_diff_vs_direct': worst,
                 'order_is_involution': is_involution,
                 'buggy_rows_rejected_at_atol': buggy_bad}


def score_arm(bm, mm, mc, bundle, f, ds, split, sid, users, cands, n_users,
              n_pairs, device, chunk, verify_sample, verify_seed, required_match,
              verify_atol=1e-4):
    """Score every union candidate's own SID under one factorization."""
    t0 = time.time()
    L = int(sid.shape[1])
    hist, hmask = mm.build_history_tokens(ds, split, mc.sid_path(f, ds))
    hist, hmask = mm.encoder_input_for(bundle, f, hist, hmask)
    if hist.shape[0] != n_users:
        raise AssertionError(
            f'{ds}/{split}/{f}: history rows {hist.shape[0]} != union users '
            f'{n_users} -- the union row order and the dataset row order have '
            f'diverged; refusing to score')
    if int(hist.max()) >= int(bundle['cfg']['vocab_size']):
        raise AssertionError(f'{ds}/{split}/{f}: history token {int(hist.max())} >= '
                             f'vocab {bundle["cfg"]["vocab_size"]}')

    row = cands - 1                                   # canonical id -> SID row
    if row.min() < 0 or row.max() >= len(sid):
        raise AssertionError(f'{ds}/{split}/{f}: candidate id out of SID range: '
                             f'[{int(cands.min())}, {int(cands.max())}] vs '
                             f'{len(sid)} items')
    toks = mc.tokenise(sid[row])
    if toks.shape != (n_pairs, L):
        raise AssertionError(f'{ds}/{split}/{f}: token shape {toks.shape} != '
                             f'({n_pairs}, {L})')
    if toks.min() < 1 or toks.max() >= int(bundle['cfg']['vocab_size']):
        raise AssertionError(f'{ds}/{split}/{f}: token range [{int(toks.min())}, '
                             f'{int(toks.max())}] outside [1, '
                             f'{bundle["cfg"]["vocab_size"]})')

    # the repaired driver's exact sort: lexsort((candidate, user))
    order = bm.assert_permutation(np.lexsort((cands, users)), n_pairs)
    u_sorted = users[order]
    uniq, starts = np.unique(u_sorted, return_index=True)
    ends = np.append(starts[1:], n_pairs)
    if len(uniq) != n_users or not np.array_equal(uniq, np.arange(n_users)):
        raise AssertionError(f'{ds}/{split}/{f}: the union does not cover every '
                             f'user row exactly once ({len(uniq)} vs {n_users})')
    owner_all = np.repeat(np.arange(n_users, dtype=np.int64), ends - starts)

    nll_sorted = mm.candidate_total_nll(bundle['model'], hist, hmask, toks[order],
                                        owner_all, device=device, batch_size=chunk)
    nll_sorted = np.asarray(nll_sorted, dtype=np.float64)
    if nll_sorted.shape != (n_pairs,):
        raise AssertionError(f'{ds}/{split}/{f}: scorer returned {nll_sorted.shape}')
    n_finite = int(np.isfinite(nll_sorted).sum())
    if n_finite != n_pairs:
        bad = int(np.flatnonzero(~np.isfinite(nll_sorted))[0])
        raise AssertionError(
            f'{ds}/{split}/{f}: {n_pairs - n_finite} non-finite NLL(s); first at '
            f'sorted position {bad} -- refusing to write a parquet with NaN')

    def score_fn(u_idx, c_idx):
        """DIRECT per-row scoring in the union's own row order (no lexsort)."""
        c_idx = np.asarray(c_idx, dtype=np.int64)
        u_idx = np.asarray(u_idx, dtype=np.int64)
        t = mc.tokenise(sid[c_idx - 1])
        own = np.arange(len(u_idx), dtype=np.int64)
        return mm.candidate_total_nll(bundle['model'], hist[u_idx], hmask[u_idx],
                                      t, own, device=device, batch_size=chunk)

    nll_rows, verify_rec = write_back_and_verify_calibrated(
        bm, nll_sorted, order, users, cands, score_fn,
        what=f'{ds}/{split}/{f}', verify_all=False, sample_seed=verify_seed,
        atol=verify_atol)
    nll_rows = np.asarray(nll_rows, dtype=np.float64)

    # post-write identity check, kept EXACT: the scatter identity must hold
    # bit-for-bit.  The non-circular half of the proof -- a fresh direct re-score of
    # a seeded sample, compared at the calibrated atol -- ran inside
    # `write_back_and_verify_calibrated`; its diagnostics travel in `verify_rec`.
    if not np.array_equal(nll_rows[order], nll_sorted):
        raise AssertionError(f'{ds}/{split}/{f}: post-write scatter identity broken')
    if not verify_rec['order_is_involution'] and \
            verify_rec['buggy_rows_rejected_at_atol'] == 0:
        raise AssertionError(f'{ds}/{split}/{f}: the calibrated tolerance is not '
                             f'discriminating')

    fin = np.isfinite(nll_rows)
    rec = {'factorization': f, 'label': LABEL[f], 'sid_len': L,
           'vocab_size': int(bundle['cfg']['vocab_size']),
           'model_kind': bundle['kind'],
           'checkpoint': str(bundle['checkpoint']),
           'checkpoint_sha256': sha256_file(bundle['checkpoint']),
           'sid': str(mc.sid_path(f, ds)), 'sid_sha256': sha256_file(mc.sid_path(f, ds)),
           'n_pairs': n_pairs, 'mean_nll': float(nll_rows[fin].mean()),
           'std_nll': float(nll_rows[fin].std(ddof=0)),
           'min_nll': float(nll_rows[fin].min()),
           'max_nll': float(nll_rows[fin].max()),
           'finite_fraction': float(fin.mean()),
           'order_is_involution': bool((order[order] == np.arange(n_pairs)).all()),
           'write_back': 'out[order] = scores_sorted (repaired)',
           'legacy_write_back': 'scores_sorted[order] (buggy, NOT used)',
           'n_rows_identity_checked': int(verify_rec['n_verified']),
           'identity_check_max_abs_diff': float(verify_rec['max_abs_diff_vs_direct']),
           'identity_check_seed': int(verify_seed),
           'identity_check_atol': float(verify_rec['atol']),
           'identity_check_mode': verify_rec['verify_mode'],
           'identity_tolerance_justified': {
               'reason': 'the bulk pass and the direct re-score batch rows '
                         'differently and the model is float32, so the two agree '
                         'only to ~1e-6 (round note: float32 batching noise ~2e-5)',
               'historical_buggy_rows_rejected_at_atol':
                   int(verify_rec['buggy_rows_rejected_at_atol']),
               'order_is_involution': bool(verify_rec['order_is_involution'])},
           'elapsed_s': round(time.time() - t0, 1)}
    return nll_rows, rec


def cell_paths(ds, split):
    d = CAND / ds
    return (d / f'{split}_m5_candidate_nll.parquet',
            d / f'{split}_m5_candidate_nll.metadata.json',
            d / f'{split}_m5_candidate_nll.status.json')


def finished_cell(ds, split, input_hashes):
    """True only for a cell whose parquet AND completed status both exist."""
    pq, md, st = cell_paths(ds, split)
    if not pq.exists():
        return False, None
    if not st.exists():
        raise SystemExit(
            f'HARD FAIL: {pq} exists without {st.name}; refusing to guess whether '
            f'it is complete and refusing to overwrite it. Move it aside by hand.')
    obj = json.loads(st.read_text())
    if obj.get('status') != STATUS_COMPLETED:
        raise SystemExit(f'HARD FAIL: {pq} exists with status '
                         f'{obj.get("status")!r}; refusing to overwrite it.')
    old = obj.get('input_hashes', {})
    differing = {k: (old.get(k), v) for k, v in input_hashes.items()
                 if old.get(k) != v}
    if differing:
        raise SystemExit(
            f'HARD FAIL: {pq} was produced from different inputs '
            f'({sorted(differing)}); refusing to overwrite a finished artifact. '
            f'Move m5_rq5/candidate_nll/{ds}/ aside to re-score it.')
    return True, obj


def run_cell(a, bm, mc, mm, fa, p5, models, ds, split):
    """Score one (dataset, split) cell; skip it when it is already finished."""
    rec, U = union_rows(bm, fa, p5, ds, a.affinity, split)
    if U is None:
        return rec
    _, cands, users, arm_ranks = U
    rec['convention'] = convention_audit(rec, a.require_convention_match)

    input_hashes = {
        'union_dumps': {p['arm']: p.get('sha256') for p in rec['provenance']},
        'sid_arrays': {f: sha256_file(mc.sid_path(f, ds)) for f in a.factorizations},
        'checkpoints': {f: sha256_file(bm.rqvae5_ckpt_path(ds) if f == 'rqvae5'
                                       else mm.ckpt_path(f, ds))
                        for f in a.factorizations},
        'n_pairs': rec['n_pairs'], 'n_users': rec['n_users'],
        'chunk': a.chunk, 'factors': list(a.factorizations),
        'affinity': a.affinity}
    pq, md, st = cell_paths(ds, split)
    done, prev = finished_cell(ds, split, input_hashes)
    if done:
        arms_prev = {}
        if md.exists():
            try:
                arms_prev = json.loads(md.read_text()).get('arms', {})
            except Exception as e:
                raise SystemExit(f'HARD FAIL: {md} is unreadable ({e}); the finished '
                                 f'cell cannot be described -- refusing to continue')
        rec.update({'status': 'SKIPPED_ALREADY_COMPLETE',
                    'parquet': str(pq),
                    'status_file_sha256': sha256_file(st),
                    'arms': arms_prev,
                    'finished_at': prev.get('finished_at')})
        print(f'    SKIP {ds}/{split}: already complete '
              f'({prev.get("finished_at")}) -- nothing rewritten', flush=True)
        return rec

    pq.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + a.tag
    run_dir = CAND_RUNS / run_id
    if run_dir.exists():
        raise SystemExit(f'HARD FAIL: run directory already exists: {run_dir}')
    run_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    # FIXED 2026-10-13 (final paper completion round).  See FIX_CANDIDATE_NLL.md.
    #
    # `U` is `union_rows`' OUTER tuple.  The historical code did
    #     items_pad = np.asarray(U[0])
    # but `U[0]` is the union 5-TUPLE `(items_pad, score_tensor, target,
    # target_pos, valid_mask)`, not the items array.  `np.asarray` on that tuple
    # therefore raised
    #     ValueError: setting an array element with a sequence. The requested
    #     array has an inhomogeneous shape after 2 dimensions. The detected
    #     shape was (5, <n_users>) + inhomogeneous part.
    # which is exactly the S3 failure of 2026-09-18 that left every M5 CFLA row
    # BLOCKED_NO_CANDIDATE_NLL.  The 5 here is the tuple length, NOT n_arms.
    # The same shadowing made `U[1]` mean `cands` instead of the score tensor.
    #
    # The real structure is restored below by unpacking the union tuple with the
    # SAME contract `build_m5_rq5.pu_for` uses; nothing is coerced to
    # dtype=object and no ragged array is hidden.
    # ------------------------------------------------------------------
    union5, arm_ranks = union_parts(U)
    items_pad, score_tensor, _target, target_pos, valid_mask = union5
    items_pad = np.asarray(items_pad)
    score_tensor = np.asarray(score_tensor)
    if len(arm_ranks) != len(a.factorizations):
        raise AssertionError(
            f'{ds}/{split}: {len(arm_ranks)} per-arm rank vectors for '
            f'{len(a.factorizations)} factorizations')
    if score_tensor.shape != (items_pad.shape[0], items_pad.shape[1],
                              len(a.factorizations)):
        raise AssertionError(
            f'{ds}/{split}: score_tensor {score_tensor.shape} != '
            f'{(items_pad.shape[0], items_pad.shape[1], len(a.factorizations))}')
    # The row order is the union's own: `np.where(valid_mask)`, user-major.
    # Recovering it must reproduce `cands` and `users` exactly, or the parquet
    # would describe rows that are not the rows the M5 metrics rank.
    rows_i, rows_p = np.where(valid_mask)
    recovered = np.asarray(items_pad[rows_i, rows_p], dtype=np.int64)
    if not np.array_equal(recovered, cands) or not np.array_equal(rows_i, users):
        raise AssertionError(f'{ds}/{split}: union row recovery is not reproducible')
    cols = {'dataset': [ds] * rec['n_pairs'], 'split': [split] * rec['n_pairs'],
            'user': (users + 1).astype(np.int64),
            'user_row': users.astype(np.int64),
            'pos': rows_p.astype(np.int64),
            'candidate_item': cands.astype(np.int64)}

    arm_recs = {}
    for m, f in enumerate(a.factorizations):
        bundle = get_model(bm, mm, ds, f, a.device, models)
        sid = mc.load_sid(f, ds)
        nll, frec = score_arm(bm, mm, mc, bundle, f, ds, split, sid, users, cands,
                              rec['n_users'], rec['n_pairs'], a.device, a.chunk,
                              a.verify_sample, a.verify_seed,
                              a.require_convention_match, a.verify_atol)
        L = LABEL[f]
        # P_<L> is `fusion_audit.build_union`'s own accumulated softmax score for
        # this (user, item) pair under arm F -- taken from the union tensor, so it
        # cannot disagree with the union the metrics use.
        # (FIXED: was `U[1][...]`, which is the `cands` vector, not the tensor.)
        cols[f'present_{L}'] = (score_tensor[rows_i, rows_p, m] > 0.0)
        cols[f'P_{L}'] = np.asarray(score_tensor[rows_i, rows_p, m],
                                    dtype=np.float64)
        # §12 required fields, stated explicitly rather than by convention:
        #   candidate_source_mask_<L> == this arm proposed this candidate
        #   source_rank_<L>           == 1-based beam rank inside that arm (0 = absent)
        cols[f'candidate_source_mask_{L}'] = cols[f'present_{L}']
        cols[f'source_rank_{L}'] = np.asarray(arm_ranks[m], dtype=np.int64)
        cols[f'NLL_{L}'] = nll
        cols[f'sid_len_{L}'] = np.int64(frec['sid_len'])
        cols[f'TotalNLL_{L}'] = nll                     # repaired-round alias
        cols[f'prob_{L}'] = cols[f'P_{L}']              # repaired-round alias
        arm_recs[f] = frec
        print(f'    {ds}/{split}/{f:12s} {rec["n_pairs"]} NLLs  '
              f'mean={frec["mean_nll"]:.4f} finite={frec["finite_fraction"]:.6f}  '
              f'identity max|d|={frec["identity_check_max_abs_diff"]:.3e} '
              f'({frec["elapsed_s"]}s)', flush=True)

    import pandas as pd
    df = pd.DataFrame(cols)
    for c in df.columns:
        if df[c].dtype == object and not isinstance(df[c].iloc[0], str):
            df[c] = df[c].astype(np.int64)
    for f in a.factorizations:
        L = LABEL[f]
        if f'NLL_{L}' not in df.columns:
            raise AssertionError(f'{ds}/{split}: missing NLL column for {f}')
    if not np.isfinite(df[[f'NLL_{LABEL[f]}' for f in a.factorizations]]
                       .to_numpy(dtype=np.float64)).all():
        raise AssertionError(f'{ds}/{split}: non-finite NLL in the assembled frame')
    tmp = pq.with_name(pq.name + f'.tmp.{os.getpid()}')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, pq)

    meta = {'producer': 'score_m5_candidate_nll.py', 'run_id': run_id,
            'command': ' '.join(_sys.argv), 'python': _sys.executable,
            'started': a.started, 'finished': utc_now(),
            'dataset': ds, 'split': split, 'scope': SCOPE,
            'affinity': a.affinity, 'chunk': a.chunk, 'device': a.device,
            'factorizations': list(a.factorizations),
            'gpu_state_at_start': a.gpu_info, 'gpu_busy_at_start': a.gpu_busy,
            'gpu_busy_reasons': a.gpu_reasons,
            'allow_busy_gpu': bool(a.allow_busy_gpu),
            'row_order': "the M5 union's own row order: user-major, ascending union "
                         "position within a user (`np.where(valid_mask)` of "
                         "fusion_audit.build_union)",
            'user_column': 'canonical 1-based user id = user_row + 1',
            'user_row_column': 'index into build_history_tokens output',
            'nll_definition': 'mf_models.candidate_total_nll: teacher-forced full-SID '
                              'own-SID NLL under the factorization\'s own checkpoint',
            'columns': list(df.columns),
            'union': recall(rec),
            'arms': arm_recs,
            'parquet': str(pq), 'parquet_sha256': sha256_file(pq),
            'parquet_rows': int(len(df)),
            'input_hashes': input_hashes,
            'wrote_only_under': str(OUT)}
    write_json(md, meta)
    write_json(st, {'status': STATUS_COMPLETED, 'run_id': run_id,
                    'finished_at': utc_now(),
                    'parquet': str(pq), 'parquet_sha256': sha256_file(pq),
                    'metadata_sha256': sha256_file(md),
                    'input_hashes': input_hashes,
                    'n_pairs': rec['n_pairs'], 'n_users': rec['n_users'],
                    'per_arm': {f: {'mean_nll': arm_recs[f]['mean_nll'],
                                    'std_nll': arm_recs[f]['std_nll'],
                                    'checkpoint_sha256':
                                        arm_recs[f]['checkpoint_sha256']}
                                for f in a.factorizations}})
    rec.update({'status': 'SCORED', 'parquet': str(pq),
                'parquet_sha256': sha256_file(pq), 'arms': arm_recs,
                'run_id': run_id, 'metadata': str(md), 'status_file': str(st)})
    return rec


def recall(rec):
    """The union provenance, minus the per-arm convention blob (kept at top level)."""
    out = {k: v for k, v in rec.items() if k != 'provenance'}
    out['provenance'] = [{k: v for k, v in p.items()
                          if k != 'item_id_convention_measured'}
                         for p in rec.get('provenance', [])]
    out['item_id_convention'] = rec.get('convention')
    return out


# ===========================================================================
# validation-fitted scalers
# ===========================================================================
def fit_scalers(a, cells):
    """(mu_F, sigma_F) per (dataset, factorization), fitted on VALID ONLY."""
    pools = {}
    for ds in a.datasets:
        v = cells.get((ds, 'valid'))
        if not v or v.get('status') not in ('SCORED', 'SKIPPED_ALREADY_COMPLETE'):
            continue
        pq = cell_paths(ds, 'valid')[0]
        if not pq.exists():
            continue
        import pandas as pd
        df = pd.read_parquet(pq, columns=['NLL_' + LABEL[f]
                                          for f in a.factorizations])
        entry = {}
        for f in a.factorizations:
            L = LABEL[f]
            x = df[f'NLL_{L}'].to_numpy(dtype=np.float64)
            if not np.isfinite(x).all():
                raise AssertionError(f'{ds}/valid/{f}: non-finite NLL in the VALID '
                                     f'parquet -- refusing to fit a scaler')
            mu = float(x.mean())
            sd = float(x.std(ddof=0))
            if not np.isfinite([mu, sd]).all() or sd <= 0.0:
                raise AssertionError(f'{ds}/valid/{f}: degenerate scaler '
                                     f'mu={mu} sigma={sd}')
            arms = v.get('arms', {})
            frec = arms.get(f, {}) if isinstance(arms, dict) else {}
            entry[L] = {'mean': mu, 'std': sd, 'std_ddof': 0,
                        'n_candidates': int(x.size), 'sid_len': frec.get('sid_len'),
                        'checkpoint': frec.get('checkpoint'),
                        'checkpoint_sha256': frec.get('checkpoint_sha256'),
                        'sid': frec.get('sid'), 'sid_sha256': frec.get('sid_sha256'),
                        'fitted_on': 'ALL validation candidates of this pool',
                        'fit_split': 'valid',
                        'source_parquet': str(pq),
                        'source_parquet_sha256': sha256_file(pq),
                        'factorization': f}
        pools[ds] = entry
    return pools


def write_scalers(a, pools, cells):
    """Write the scaler artifact, archiving any previous version (never clobbering)."""
    obj = {'producer': 'score_m5_candidate_nll.py',
           'generated': utc_now(), 'fit_split': 'valid',
           'test_used_for_fitting': False,
           'pool': SCOPE,
           'aggregation': 'CFLA: score(u,i) = logsumexp_F(-zNLL_F(u,i)), '
                          'zNLL_F = (NLL_F - mu_F)/sigma_F, (mu_F,sigma_F) from '
                          'THIS file (VALID only)',
           'labels': {f: LABEL[f] for f in a.factorizations},
           'standard_deviation': 'population std (ddof=0), matching the repaired '
                                 'round\'s provenance',
           'pools': {SCOPE: pools},
           'n_datasets_complete': len(pools),
           'requested_datasets': list(a.datasets),
           'complete': sorted(pools) == sorted(a.datasets),
           'command': ' '.join(_sys.argv),
           'factors': list(a.factorizations)}
    if SCALERS.exists():
        old = json.loads(SCALERS.read_text())
        same = (old.get('pools', {}).get(SCOPE, {}) == pools
                and old.get('factors') == obj['factors'])
        if same:
            print(f'  scalers: {SCALERS} already matches -- left untouched')
            return SCALERS, True
        CAND_HISTORY.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        arch = CAND_HISTORY / f'validation_nll_scalers.{stamp}.json'
        SCALERS.replace(arch)
        print(f'  scalers: previous version archived to {arch} (never deleted)')
    write_json(SCALERS, obj)
    print(f'  scalers: wrote {SCALERS} ({len(pools)} dataset(s), '
          f'complete={obj["complete"]})')
    return SCALERS, False


# ===========================================================================
# --check : validation only, no CUDA context, no writes
# ===========================================================================
def check_mode(a, bm, mc, mm, fa, p5):
    print('=' * 78)
    print('score_m5_candidate_nll.py -- CHECK (no GPU, no CUDA context, no writes)')
    print('=' * 78)
    print(f'  datasets       {a.datasets}')
    print(f'  splits         {a.splits}')
    print(f'  factorizations {a.factorizations}')
    print(f'  affinity       {a.affinity}')
    print(f'  output root    {CAND}')
    print()
    info, procs, busy, reasons = gpu_state(bm, a.gpu_mem_limit_mib)
    report_gpu(bm, info, busy, reasons, a.gpu_mem_limit_mib)
    print(f'  CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")!r} '
          f'-- --check empties it so this process CANNOT touch the device')
    print()

    problems, missing_models = [], []
    for ds in a.datasets:
        for f in a.factorizations:
            for p, what in ((mc.sid_path(f, ds), 'SID array'),
                            (mm.ckpt_path(f, ds), 'checkpoint')):
                if not p.exists():
                    problems.append(f'{ds}/{f}: missing {what} {p}')
        for split in a.splits:
            p = bm.rqvae5_dump_path(ds, split)
            if not p.exists():
                problems.append(f'{ds}/{split}: missing RQ-VAE5 beam dump {p}')
            for f in a.factorizations:
                if f == 'rqvae5':
                    continue
                k = bm.ARM_SEMANTIC[f]
                sp = bm.spectral_dump_path(ds, a.affinity, split, k)
                if not sp.exists():
                    problems.append(f'{ds}/{split}/{f}: missing beam dump {sp}')

    cells = {}
    for ds in a.datasets:
        for split in a.splits:
            t0 = time.time()
            rec, U = union_rows(bm, fa, p5, ds, a.affinity, split)
            if U is None:
                problems.append(f'{ds}/{split}: {rec["missing_input"]}')
                print(f'  {ds}/{split}: MISSING INPUT {rec["missing_input"]}')
                continue
            rec['convention'] = convention_audit(rec, a.require_convention_match)
            pq, _md, st = cell_paths(ds, split)
            state = ('absent' if not pq.exists() else
                     ('complete' if st.exists() and
                      json.loads(st.read_text()).get('status') == STATUS_COMPLETED
                      else 'PRESENT_WITHOUT_COMPLETED_STATUS'))
            cells[(ds, split)] = rec
            print(f'  {ds:32s} {split:5s} users={rec["n_users"]:6d} '
                  f'union_rows={rec["n_pairs"]:9d} mean_union='
                  f'{rec["mean_union_size"]:7.3f}  output={state}  '
                  f'({time.time() - t0:.1f}s CPU)')
            for arm, c in sorted(rec['convention'].items()):
                print(f'      {arm:12s} convention={c["convention"]:20s} '
                      f'match={c["match_rate"]:.6f} '
                      f'valid_slots={c["n_valid_slots"]} '
                      f'padded={c["n_padded_slots"]}')
            if state == 'PRESENT_WITHOUT_COMPLETED_STATUS':
                problems.append(f'{ds}/{split}: {pq} exists without a completed '
                                f'status -- the real run would HARD FAIL')

    if a.check_models:
        print()
        print('  model load check (CPU only; the device is not touched):')
        for ds in a.datasets:
            for f in a.factorizations:
                try:
                    b = mm.load_model_for(f, ds, device='cpu', cache=None)
                    print(f'    OK   {ds:32s} {f:12s} kind={b["kind"]} '
                          f'vocab={b["cfg"]["vocab_size"]} '
                          f'{Path(b["checkpoint"]).name}')
                    del b
                except Exception as e:
                    missing_models.append(f'{ds}/{f}: {type(e).__name__}: {e}')
                    print(f'    FAIL {ds:32s} {f:12s} {type(e).__name__}: {e}')

    try:
        import torch
        if torch.cuda.is_initialized():
            problems.append('this process created a CUDA context during --check, '
                            'which it must never do')
    except Exception:
        pass

    print()
    if missing_models:
        print('MODEL LOAD FAILURES:')
        for m in missing_models:
            print(f'  * {m}')
    if problems:
        print('PROBLEMS:')
        for m in problems:
            print(f'  * {m}')
        print(f'\nCHECK RESULT: NOT READY ({len(problems)} problem(s), '
              f'{len(missing_models)} model failure(s))')
        return 1
    print(f'CHECK RESULT: READY -- every input for {len(cells)} requested cell(s) '
          f'is present, every item-id convention is measured at >= '
          f'{a.require_convention_match}, and the device is '
          f'{"BUSY (the real run would refuse unless --allow-busy-gpu)" if busy else "idle"}')
    return 0


# ===========================================================================
# cli
# ===========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Produce the M5 (spectral K2-K5 + RQ-VAE5) candidate-NLL '
                    'parquets and the VALID-fitted CFLA scalers.')
    ap.add_argument('--datasets', default=','.join(DATASETS),
                    help='comma-separated subset of ' + ','.join(DATASETS))
    ap.add_argument('--splits', default='valid,test')
    ap.add_argument('--factorizations', default=','.join(FACTORIZATIONS))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--affinity', default=DEFAULT_AFFINITY)
    ap.add_argument('--chunk', type=int, default=DEFAULT_CHUNK,
                    help='candidate sequences per forward pass (predeclared '
                         'constant, same value PHASE A used)')
    ap.add_argument('--allow-busy-gpu', action='store_true',
                    help='permit scoring while another process owns the device; '
                         'the fact is recorded in every output metadata file')
    ap.add_argument('--gpu-mem-limit-mib', type=float,
                    default=DEFAULT_GPU_MEM_LIMIT_MIB)
    ap.add_argument('--verify-sample', type=int, default=DEFAULT_VERIFY_SAMPLE,
                    help='rows re-scored directly, in the original row order, as the '
                         'post-write (user, candidate) identity check')
    ap.add_argument('--verify-seed', type=int, default=VERIFY_SEED)
    ap.add_argument('--verify-atol', type=float, default=DEFAULT_VERIFY_ATOL,
                    help='calibrated absolute tolerance for the DIRECT re-score of '
                         'the seeded sample.  Every structural check (permutation, '
                         'scatter identity, pair identity) stays EXACT; only this '
                         'numeric comparison is tolerant, because the bulk pass and '
                         'the direct pass batch rows differently in float32.  The '
                         'driver proves on every call that the historical buggy '
                         'write-back would still be rejected at this atol.')
    ap.add_argument('--require-convention-match', type=float, default=1.0,
                    help='the measured winning item-id convention must reconstruct '
                         'at least this fraction of valid beam slots (default 1.0)')
    ap.add_argument('--check', action='store_true',
                    help='validation only: inputs, conventions, union build and '
                         'output status; never touches the GPU and writes nothing')
    ap.add_argument('--check-models', action='store_true',
                    help='with --check: also load every checkpoint on the CPU')
    ap.add_argument('--tag', default='m5candnll')
    ap.add_argument('--summary-run', default='',
                    help='optional run id for the summary JSON; default is a fresh '
                         'timestamped one')
    a = ap.parse_args(argv)
    a.datasets = [d.strip() for d in a.datasets.split(',') if d.strip()]
    a.splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    a.factorizations = [f.strip() for f in a.factorizations.split(',') if f.strip()]
    for d in a.datasets:
        if d not in DATASETS:
            ap.error(f'unknown dataset {d!r}')
    for s in a.splits:
        if s not in ('valid', 'test'):
            ap.error(f'unknown split {s!r}')
    for f in a.factorizations:
        if f not in FACTORIZATIONS:
            ap.error(f'unknown factorization {f!r}')
    if not a.datasets or not a.splits or not a.factorizations:
        ap.error('--datasets/--splits/--factorizations must not be empty')
    return a


def main(argv=None):
    a = parse_args(argv)
    a.started = utc_now()
    t0 = time.time()
    bm, mc, mm = load_certified()
    reg = register_rqvae5_arm(bm, mc, mm)
    fa, p5 = bm.import_historical()

    if a.check:
        if a.device != 'cpu':
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        return check_mode(a, bm, mc, mm, fa, p5)

    print('=' * 78)
    print('score_m5_candidate_nll.py')
    print('=' * 78)
    print(f'  datasets       {a.datasets}')
    print(f'  splits         {a.splits}')
    print(f'  factorizations {a.factorizations}')
    print(f'  affinity       {a.affinity}   chunk {a.chunk}   device {a.device}')
    print(f'  output         {CAND}')
    print(f'  rqvae5 arm     {reg["kind"]}  {bm.rqvae5_ckpt_path("<ds>")}')
    print(f'  repaired writer sha256 {bm.REPAIRED_SHA}')
    print()

    a.gpu_info, a.host_procs, a.gpu_busy, a.gpu_reasons = gpu_state(
        bm, a.gpu_mem_limit_mib)
    report_gpu(bm, a.gpu_info, a.gpu_busy, a.gpu_reasons, a.gpu_mem_limit_mib)
    a.allow_busy_gpu = bool(a.allow_busy_gpu)
    if a.gpu_busy:
        if not a.allow_busy_gpu:
            bm.refuse_gpu(a.gpu_reasons, a.gpu_mem_limit_mib)
        print()
        print('WARNING: --allow-busy-gpu: the device is NOT exclusively idle.  Any '
              'number taken now is taken under contention; that fact is recorded in '
              'every output metadata file.')
    print()

    run_id = a.summary_run or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                               + '_' + a.tag)
    run_dir = CAND_RUNS / run_id
    if run_dir.exists():
        raise SystemExit(f'HARD FAIL: run directory already exists: {run_dir}')

    cells, models = {}, {}
    for ds in a.datasets:
        for split in a.splits:
            print(f'  --- {ds} / {split} ---', flush=True)
            try:
                rec = run_cell(a, bm, mc, mm, fa, p5, models, ds, split)
            except SystemExit:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {'dataset': ds, 'split': split, 'status': 'FAILED',
                       'error': f'{type(e).__name__}: {e}'}
                print(f'    FAILED {ds}/{split}: {type(e).__name__}: {e}', flush=True)
            cells[(ds, split)] = rec
            print()

    pools = fit_scalers(a, cells) if 'valid' in a.splits else {}
    scalers_path, scalers_untouched = (None, False)
    if pools:
        scalers_path, scalers_untouched = write_scalers(a, pools, cells)
    elif 'valid' in a.splits:
        print('  scalers: NOT written -- no VALID cell completed')

    ok = [k for k, v in cells.items()
          if v.get('status') in ('SCORED', 'SKIPPED_ALREADY_COMPLETE')]
    skipped_missing = [k for k, v in cells.items()
                       if v.get('status') == 'MISSING_INPUT']
    failed = [k for k, v in cells.items() if v.get('status') == 'FAILED']
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {'producer': 'score_m5_candidate_nll.py', 'run_id': run_id,
               'command': ' '.join(_sys.argv), 'python': _sys.executable,
               'started': a.started, 'finished': utc_now(),
               'elapsed_s': round(time.time() - t0, 1),
               'datasets': a.datasets, 'splits': a.splits,
               'factorizations': a.factorizations, 'chunk': a.chunk,
               'device': a.device, 'affinity': a.affinity,
               'gpu_state_at_start': a.gpu_info, 'gpu_busy_at_start': a.gpu_busy,
               'gpu_busy_reasons': a.gpu_reasons,
               'allow_busy_gpu': a.allow_busy_gpu,
               'rqvae5_registration': reg,
               'repaired_writer_sha256': bm.REPAIRED_SHA,
               'cells': {f'{d}|{s}': recall(v) for (d, s), v in sorted(cells.items())},
               'n_complete': len(ok), 'n_missing_input': len(skipped_missing),
               'n_failed': len(failed),
               'scalers': str(scalers_path) if scalers_path else None,
               'scalers_untouched': bool(scalers_untouched),
               'wrote_only_under': str(OUT)}
    write_json(run_dir / 'SUMMARY.json', summary)

    print('=' * 78)
    print(f'complete={len(ok)}  missing_input={len(skipped_missing)}  '
          f'failed={len(failed)}')
    for k in sorted(skipped_missing):
        print(f'  MISSING INPUT {k[0]}/{k[1]}: '
              f'{cells[k].get("missing_input")}')
    for k in sorted(failed):
        print(f'  FAILED        {k[0]}/{k[1]}: {cells[k].get("error")}')
    print(f'summary {run_dir / "SUMMARY.json"}')
    if failed:
        return 2
    if skipped_missing:
        return 4
    return 0


if __name__ == '__main__':
    _sys.exit(main())
