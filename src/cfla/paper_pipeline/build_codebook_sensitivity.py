#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 12-16 -- Beauty-only codebook sensitivity study.

WHAT THIS DRIVER ANSWERS (and what it refuses to answer)
========================================================
Three separable results, kept separate in every table it writes:

  (a) RUNNABLE NOW, Beauty, frozen inputs only, CPU only
      * `spectral_k5` single-model codebook sensitivity: 6 codebooks x K2-K5 of
        frozen Spectral SIDs + checkpoints + beam dumps, re-scored with the audited
        dump evaluator.
      * per-codebook multi-factorization fusion over the pool K2..K5:
        `uniform`, `best_single` (K chosen on VALID), and the historical
        structure+confidence dynamic fusion ((gamma, eta) chosen on VALID).
      * at the canonical C=256 only, the matched 2-factor pair
        `Spectral-K5(256) + RQ-VAE5(256)` (both beam dumps exist).

  (b) BLOCKED -- the matched `Spectral-K5(C)` vs `RQ-VAE5(C)` pairing for
      C != 256.  `RQ-VAE5(C)` does not exist: only the C=256 arm was trained this
      round.  Explicit blocked rows are emitted, naming the missing artifacts and
      the exact command that would produce them.  Nothing is approximated.

  (c) IMPLEMENTED, GPU-PENDING -- CFLA (Cross-Factorization Likelihood
      Aggregation) across the tested codebooks.  CFLA needs teacher-forced
      candidate NLL over the K2..K5 union, i.e. 4 T5 forward passes per candidate
      per codebook, which requires the GPU.  The device is currently owned by a
      training job, so this driver REFUSES to run that stage and emits explicit
      `PENDING_GPU_BUSY` rows instead.  The exact command is recorded in
      `RUN_METADATA.json` and printed by `--dry-run`.

SCIENTIFIC RULES ENFORCED HERE
==============================
 1. Selection is ALWAYS on VALID (item-level NDCG@20).  TEST is only ever
    reported.  No codebook was selected on TEST, and no codebook was selected at
    all: the codebook study is *descriptive sensitivity*, never "the optimal
    codebook is C".
 2. Item ids in the historical dumps are 0-based and canonicalise to +1 on load.
    The 5+1 campaign dumps (RQ-VAE5) are ALREADY 1-based.  The convention of every
    dump is therefore *measured* per dump by exact candidate-token reconstruction
    (below), never assumed; a naive `+1` on both sides of a cross-family fusion
    would silently destroy every cross-family item match.
 3. The vocabulary rule `stride = max(semantic_codebook, disambiguation_alphabet)`,
    `vocab = stride * L + 1` holds for C >= 256 and FAILS for C in {32,64,128}.
    The embedded `vocab_size` is read from each frozen dump and the stride is
    cross-checked against the SID array itself; `C*L+1` is never assumed.
 4. Per-row scores are written back through the repaired helper
    `write_scores_back_to_original_rows` from
    `experiments/multifactor_fusion/mf_candidate_nll_repaired.py`, plus an
    independent spot-check that each value belongs to its own (user, candidate)
    row.  `--self-test` pins this with a NON-INVOLUTION permutation, so the
    historical `scores_sorted[order]` bug cannot pass.
 5. NaN, shape mismatch, missing file, non-permutation index and failed
    independent cross-checks are loud failures.  No bare `except`.

USAGE
=====
  python3 OUT/drivers/build_codebook_sensitivity.py --self-test
  python3 OUT/drivers/build_codebook_sensitivity.py --dry-run
  python3 OUT/drivers/build_codebook_sensitivity.py --stage cpu
  python3 OUT/drivers/build_codebook_sensitivity.py --stage cfla-gpu   # refuses while busy
  python3 OUT/drivers/build_codebook_sensitivity.py --stage cfla-gpu --limit-users 256

Read-only with respect to `the frozen artifact root` and
`upstream TIGER`: this driver only ever reads them.  Everything it
writes goes under `outputs/final_hyperparam_efficiency/codebook/`, and an existing
artifact is NEVER overwritten (a later run is diverted to a fresh `runs/<stamp>/`).
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
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# READ-ONLY GUARD
# This driver imports historical modules that live inside the read-only roots
# (`src/legacy_audited/fusion_audit.py`, `src/legacy_audited/phase5_fusion.py`).
# A plain import makes CPython WRITE `__pycache__/<mod>.cpython-*.pyc` next to the
# source, i.e. it modifies a root this round must never modify.  Suppressing
# bytecode writing makes every historical import side-effect free.
# (Incident 2026-09-17 13:46:57: the first run of this driver created
#  a bytecode cache for `phase5_fusion` inside the read-only root before
#  this guard existed.  It was NOT deleted -- deletion is equally forbidden in the
#  read-only root -- and is reported in RUN_METADATA.json: read_only_root_incident.)
# ---------------------------------------------------------------------------
sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# roots (read-only vs writable)
# ---------------------------------------------------------------------------
REPO = Path(_REPO_ROOT + "")
ART_ROOT = Path(_ART_ROOT + "")                    # READ-ONLY historical root
UPSTREAM_ROOT = Path(_UP_ROOT + "")    # READ-ONLY historical root
FE = rp.OUTPUT_ROOT / "final_hyperparam_efficiency"
SPEC = FE / "spectral_codebook"
OUT = FE / "codebook"                                       # the ONLY writable root
DRIVERS = Path(__file__).resolve().parent
MF = REPO / "src/cfla/multifactor_fusion"
PHASE5 = REPO / "src/legacy_audited/phase5_fusion.py"
FUSION_AUDIT = REPO / "src/legacy_audited/fusion_audit.py"
REPAIRED = MF / "mf_candidate_nll_repaired.py"

CST = timezone(timedelta(hours=8))

DATASET = "Beauty"
MATCHED_CODEBOOKS = [32, 64, 128, 256, 512]      # MATCHED_CODEBOOK_SET.json
EXTRA_CODEBOOK = 1024                            # checkpoint-only, no K5 dump
KS = [2, 3, 4, 5]
BEAM = 30
METRICS = ["Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20"]
N_BOOT = 5000
RNG_SEED = 20260912

REQUIRED_OUTPUTS = [
    "RQ5_SINGLE_CODEBOOK_RESULTS.csv",
    "MATCHED_FUSION_CODEBOOK_RESULTS.csv",
    "BEAUTY_MATCHED_CODEBOOK_RESULTS.csv",
    "BEAUTY_CODEBOOK_PLOT_SOURCE.csv",
    "RUN_METADATA.json",
]
# extra evidence files (data only -- no figure is ever rendered by this driver)
EXTRA_OUTPUTS = [
    "CODEBOOK_FUSION_BOOTSTRAP.csv",
    "DYNAMIC_FUSION_VALID_GRID.csv",
    "CELL_frozen vocabulary audit.csv",
    "FROZEN_METHOD4_REPRODUCIBILITY.csv",
    "CODEBOOK_SENSITIVITY_REPORT.md",
    "_PROVENANCE.json",
]

RQVAE5_UNBLOCK_CMD = (
    "cd {fe} && ../../.venv_cuda_compatible/bin/python train_tokenizer_5plus1.py "
    "--dataset Beauty --arm rqvae5 --codebook {{C}}  &&  "
    "../../.venv_cuda_compatible/bin/python train_t5_5plus1.py "
    "--dataset Beauty --arm rqvae5 --codebook {{C}}"
).format(fe=FE)

CFLA_CMD = (
    "cd {repo} && {py} {driver} --stage cfla-gpu"
).format(repo=REPO, py=REPO / ".venv_cuda_compatible/bin/python",
         driver=DRIVERS / "build_codebook_sensitivity.py")


# ===========================================================================
# small utilities
# ===========================================================================
def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(CST).strftime("%Y%m%dT%H%M%S%z")


def sha256_file(path, cache=None) -> str:
    p = Path(path)
    if cache is not None and str(p) in cache:
        return cache[str(p)]
    if not p.exists():
        raise FileNotFoundError(f"sha256: missing file {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    if cache is not None:
        cache[str(p)] = d
    return d


def read_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing JSON input: {p}")
    return json.loads(p.read_text())


def write_json_new(path, obj):
    p = Path(path)
    if p.exists():
        raise SystemExit(f"HARD FAIL: refusing to overwrite existing artifact {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def write_csv_new(path, columns, rows):
    p = Path(path)
    if p.exists():
        raise SystemExit(f"HARD FAIL: refusing to overwrite existing artifact {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return p


def write_text_new(path, text):
    p = Path(path)
    if p.exists():
        raise SystemExit(f"HARD FAIL: refusing to overwrite existing artifact {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def fmt(v):
    """Full-precision float cell, empty for a missing value (never a placeholder)."""
    if v is None:
        return ""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        raise SystemExit(f"HARD FAIL: non-finite metric about to be written: {v}")
    return repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)


# ===========================================================================
# lazy imports of the historical libraries
# ===========================================================================
def import_fusion_audit():
    for p in (str(REPO), str(FUSION_AUDIT.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fusion_audit  # noqa: E402
    return fusion_audit


def import_phase5():
    """gamma/eta grids, select_on_valid, grid_ndcg20, paired_bootstrap, N_BOOT.

    The historical helper used to create a `canonical/` directory at import time;
    that import-time write was removed when the helper was moved into this release.
    """
    for p in (str(REPO), str(PHASE5.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if str(PHASE5.parent) not in sys.path:
        sys.path.insert(0, str(PHASE5.parent))
    import phase5_fusion  # noqa: E402
    return phase5_fusion


def import_repaired():
    """The repaired row-alignment writer + permutation guard (audited fix)."""
    if str(MF) not in sys.path:
        sys.path.insert(0, str(MF))
    import mf_candidate_nll_repaired as rep  # noqa: E402
    return rep


# ===========================================================================
# row alignment: the one place scores are scattered back
# ===========================================================================
def rows_lexsort(users, cands):
    """`order[k]` = ORIGINAL row index of lexsorted position k (sorted by (user, cand))."""
    order = np.lexsort((np.asarray(cands, dtype=np.int64),
                        np.asarray(users, dtype=np.int64)))
    import_repaired().assert_permutation(order, len(order))
    return order


def scatter_scores(scores_sorted, order):
    """Repaired write-back: `out[order[k]] == scores_sorted[k]`; never `scores_sorted[order]`."""
    rep = import_repaired()
    order = rep.assert_permutation(order, len(scores_sorted))
    return rep.write_scores_back_to_original_rows(np.asarray(scores_sorted), order)


def assert_rows_own_their_values(users, cands, values, direct_scorer,
                                 n_sample=64, seed=RNG_SEED, label="rows",
                                 atol=1e-9):
    """Post-write assertion: every sampled value belongs to its own (user, candidate) row.

    `direct_scorer(row_indices)` must recompute the score of EXACTLY those rows by an
    independent path (no permutation involved) and return a same-length array.  A
    row-permuted write-back is caught here because the comparison is per (user, candidate).
    """
    users = np.asarray(users, dtype=np.int64)
    cands = np.asarray(cands, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if not (len(users) == len(cands) == n):
        raise SystemExit(f"HARD FAIL [{label}]: row vector length mismatch "
                         f"{len(users)}/{len(cands)}/{n}")
    if n == 0:
        raise SystemExit(f"HARD FAIL [{label}]: no rows to check")
    if np.isnan(values).any():
        raise SystemExit(f"HARD FAIL [{label}]: NaN in written values "
                         f"({int(np.isnan(values).sum())} rows)")
    rng = np.random.default_rng(seed)
    k = min(int(n_sample), n)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    direct = np.asarray(direct_scorer(idx), dtype=np.float64)
    if direct.shape != idx.shape:
        raise SystemExit(f"HARD FAIL [{label}]: direct scorer returned {direct.shape} "
                         f"for {idx.shape} rows")
    diff = np.abs(direct - values[idx])
    worst = float(diff.max()) if len(diff) else 0.0
    if worst > atol:
        j = int(np.argmax(diff))
        raise SystemExit(
            f"HARD FAIL [{label}]: row-ownership violated on {int((diff > atol).sum())}/{k} "
            f"sampled rows (max |delta| = {worst:.3e}); first bad row {int(idx[j])} "
            f"(user={int(users[idx[j]])}, cand={int(cands[idx[j]])}): "
            f"written={values[idx[j]]!r} direct={direct[j]!r}"
        )
    return {"label": label, "n_rows": int(n), "n_sampled": int(k),
            "max_abs_delta": worst, "atol": float(atol)}


# ===========================================================================
# input index: SPECTRAL_CODEBOOK_INVENTORY.csv is the index, not a guess
# ===========================================================================
def load_inventory():
    p = SPEC / "SPECTRAL_CODEBOOK_INVENTORY.csv"
    if not p.exists():
        raise SystemExit(f"HARD FAIL: missing input index {p}")
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if len(rows) != 96:
        raise SystemExit(f"HARD FAIL: inventory has {len(rows)} rows, expected 96")
    return rows


def _bool(s):
    return str(s).strip().lower() == "true"


def beauty_cells(inv):
    """Beauty cells from the inventory, keyed by (codebook, K)."""
    cells = {}
    for r in inv:
        if r["dataset"] != DATASET:
            continue
        cb, k = int(r["codebook_size"]), int(r["K"])
        d = dict(
            dataset=DATASET, codebook=cb, k=k,
            semantic_length=int(r["semantic_length"]),
            full_sid_length=int(r["full_sid_length"]),
            sid_path=Path(r["SID_path"]), sid_sha256=r["SID_sha256"],
            ckpt_path=Path(r["checkpoint"]), ckpt_sha256=r["checkpoint_sha256"],
            valid_available=_bool(r["valid_available"]),
            test_available=_bool(r["test_available"]),
            beam_available=_bool(r["beam_available"]),
            codebook_is_swept=_bool(r["codebook_is_swept"]),
            selection_metric=r["selection_metric"],
            scientific_status=r["scientific_status"],
        )
        if d["full_sid_length"] != k + 1:
            raise SystemExit(f"HARD FAIL: inventory row cb{cb} k{k} has full_sid_length "
                             f"{d['full_sid_length']} != K+1")
        # The beam-dump directory is derived from the INVENTORY checkpoint path
        # (same parent model/ dir), and its existence is asserted below.
        d["beam_dir"] = d["ckpt_path"].parent.parent / f"beam_dumps_abspearson_v3_cb{cb}"
        cells[(cb, k)] = d
    if len(cells) != 24:
        raise SystemExit(f"HARD FAIL: expected 24 Beauty cells, found {len(cells)}")
    return cells


# ===========================================================================
# dump validation: convention, stride, vocab -- all measured
# ===========================================================================
_DUMP_CACHE = {}


def load_dump(path):
    p = Path(path)
    if str(p) in _DUMP_CACHE:
        return _DUMP_CACHE[str(p)]
    if not p.exists():
        raise SystemExit(f"HARD FAIL: missing beam dump {p}")
    z = np.load(p, allow_pickle=True)
    need = {"split", "code_path", "ckpt_path", "sid_len", "beam_size", "vocab_size",
            "target_item_ids", "candidate_item_ids", "candidate_tokens",
            "sequence_scores"}
    missing = need - set(z.keys())
    if missing:
        raise SystemExit(f"HARD FAIL: {p} lacks keys {sorted(missing)}")
    _DUMP_CACHE[str(p)] = z
    return z


def reconstruct_rate(sid, tok, row_index, valid):
    """Fraction of valid candidate tokens reproduced by `code + stride*pos + 1`.

    `stride` is derived from the SID array itself: semantic codes live in
    [0, C), the disambiguation column may exceed C, and the generator sizes the
    token stride to the larger of the two.
    """
    L = sid.shape[1]
    stride = max(int(sid[:, :L - 1].max()) + 1, int(sid[:, L - 1].max()) + 1)
    ok = []
    for pos in range(L):
        lhs = sid[row_index[valid], pos] + stride * pos + 1
        rhs = tok[:, :, pos][valid]
        ok.append(float((lhs == rhs).mean()))
    return min(ok), float(np.mean(ok)), stride


def measure_dump(path, cell=None, sid_sha_cache=None):
    """Read the embedded vocab/stride and MEASURE the item-id convention."""
    z = load_dump(path)
    L = int(z["sid_len"])
    vocab = int(z["vocab_size"])
    if (vocab - 1) % L != 0:
        raise SystemExit(f"HARD FAIL: {path}: vocab {vocab} is not of the form "
                         f"stride*{L}+1")
    stride_embedded = (vocab - 1) // L
    beam = int(z["beam_size"])
    if beam != BEAM:
        raise SystemExit(f"HARD FAIL: {path}: beam {beam} != {BEAM}")

    code_path = Path(str(z["code_path"]))
    ckpt_path = Path(str(z["ckpt_path"]))
    if not code_path.exists():
        raise SystemExit(f"HARD FAIL: {path} points at a missing SID array {code_path}")
    sid = np.load(code_path, allow_pickle=True).astype(np.int64)
    if sid.ndim != 2 or sid.shape[1] != L:
        raise SystemExit(f"HARD FAIL: {code_path} shape {sid.shape} != (n, {L})")
    n_items = sid.shape[0]

    cand = np.asarray(z["candidate_item_ids"], dtype=np.int64)
    tok = np.asarray(z["candidate_tokens"], dtype=np.int64)
    if tok.shape[:2] != cand.shape or tok.shape[2] != L:
        raise SystemExit(f"HARD FAIL: {path}: candidate_tokens {tok.shape} does not "
                         f"match candidate_item_ids {cand.shape} with L={L}")

    # --- item-id convention, measured by exact token reconstruction -------------
    candidates = {}
    for name, row_index in (("raw_0based", cand), ("canonical_1based", cand - 1)):
        valid = (row_index >= 0) & (row_index < n_items)
        if int(valid.sum()) == 0:
            candidates[name] = (0.0, 0.0, 0, 0)
            continue
        worst, mean, stride_sid = reconstruct_rate(sid, tok, row_index, valid)
        candidates[name] = (worst, mean, int(valid.sum()), stride_sid)
    exact = [n for n, v in candidates.items() if v[0] == 1.0]
    if len(exact) != 1:
        raise SystemExit(
            f"HARD FAIL: {path}: item-id convention is ambiguous "
            f"({ {n: round(v[0], 6) for n, v in candidates.items()} }); "
            "exactly one convention must reproduce 100% of candidate tokens")
    convention = exact[0]
    stride_sid = candidates[convention][3]
    if stride_sid != stride_embedded:
        raise SystemExit(
            f"HARD FAIL: {path}: embedded stride {stride_embedded} != SID-derived "
            f"stride {stride_sid} (vocab {vocab}, L {L})")
    if not np.isfinite(sid.astype(np.float64)).all():
        raise SystemExit(f"HARD FAIL: {code_path} contains non-finite codes")

    sem_max = int(sid[:, :L - 1].max())
    dis_max = int(sid[:, L - 1].max())
    dis_alpha = dis_max + 1
    semantic_measured = sem_max + 1
    out = {
        "dump": str(path), "split": str(z["split"]), "sid_len": L, "beam_size": beam,
        "vocab_size_embedded": vocab, "stride_embedded": stride_embedded,
        "stride_sid_derived": stride_sid, "semantic_codebook_measured": semantic_measured,
        "disambiguation_alphabet": dis_alpha,
        "disambiguation_exceeds_semantic": bool(dis_alpha > semantic_measured),
        "vocab_formula_C_times_L_plus_1": L * semantic_measured + 1,
        "vocab_formula_holds": bool(vocab == L * semantic_measured + 1),
        "item_id_convention": convention,
        "reconstruction_worst_pos_rate": candidates[convention][0],
        "reconstruction_mean_rate": candidates[convention][1],
        "n_valid_candidates": candidates[convention][2],
        "n_users": int(cand.shape[0]),
        "code_path": str(code_path), "ckpt_path": str(ckpt_path),
        "sid_sha256": sha256_file(code_path, sid_sha_cache),
        "n_items_sid": int(n_items),
    }
    if cell is not None:
        if code_path != cell["sid_path"]:
            raise SystemExit(f"HARD FAIL: {path} was dumped from {code_path} but the "
                             f"inventory names {cell['sid_path']}")
        if ckpt_path != cell["ckpt_path"]:
            raise SystemExit(f"HARD FAIL: {path} was dumped from {ckpt_path} but the "
                             f"inventory names {cell['ckpt_path']}")
        if L != cell["full_sid_length"]:
            raise SystemExit(f"HARD FAIL: {path}: L={L} != inventory "
                             f"{cell['full_sid_length']}")
        if semantic_measured != cell["codebook"]:
            raise SystemExit(
                f"HARD FAIL: {path}: measured semantic alphabet {semantic_measured} != "
                f"the inventory codebook {cell['codebook']}; the stride derivation "
                "would be unsound")
    return out


def dump_path(cell, split):
    return cell["beam_dir"] / f"{split}_k{cell['k']}_sid{cell['full_sid_length']}.npz"


def canonical_candidates(z, convention):
    """Canonicalise dumped item ids to the repository's canonical 1-based space."""
    cand = np.asarray(z["candidate_item_ids"], dtype=np.int64)
    if convention == "canonical_1based":
        return np.where(cand >= 0, cand, -1)
    if convention == "raw_0based":
        return np.where(cand >= 0, cand + 1, -1)
    raise SystemExit(f"HARD FAIL: unknown item-id convention {convention!r}")


def canonical_targets(z, convention):
    tgt = np.asarray(z["target_item_ids"], dtype=np.int64)
    if convention == "canonical_1based":
        return np.where(tgt >= 0, tgt, -1)
    if convention == "raw_0based":
        return np.where(tgt >= 0, tgt + 1, -1)
    raise SystemExit(f"HARD FAIL: unknown item-id convention {convention!r}")


def verify_target_convention(meta, parquet_targets):
    """The canonical target of every user must equal the parquet target, in order.

    This pins BOTH the convention and the dump's user ordering in one check, and it
    is what makes the GPU stage's history-token row alignment legitimate.
    """
    z = load_dump(meta["dump"])
    tgt = canonical_targets(z, meta["item_id_convention"])
    if len(tgt) != len(parquet_targets):
        raise SystemExit(f"HARD FAIL: {meta['dump']}: {len(tgt)} rows vs "
                         f"{len(parquet_targets)} parquet users")
    if not np.array_equal(tgt, parquet_targets):
        bad = int((tgt != parquet_targets).sum())
        raise SystemExit(f"HARD FAIL: {meta['dump']}: canonical targets differ from the "
                         f"parquet targets for {bad}/{len(tgt)} users -> convention or "
                         "user ordering is wrong")
    return {"n_users": int(len(tgt)), "exact_match": True}


# ===========================================================================
# evaluators (audited code paths)
# ===========================================================================
def score_single_dump(path, convention, per_user=False):
    """Single-model item-level metrics from one frozen beam dump.

    Byte-for-byte the audited convention of
    `spectral_codebook/recover_spectral_metrics.py:score_single_dump` (per-user
    duplicate merge by summing softmax probabilities; rank = first position in a
    stable descending sort), with the item-id convention measured instead of assumed.
    """
    fa = import_fusion_audit()
    z = load_dump(path)
    cand = canonical_candidates(z, convention)
    pr = fa._row_softmax(np.asarray(z["sequence_scores"], dtype=np.float64), cand >= 0)
    tgt = canonical_targets(z, convention)
    n = cand.shape[0]
    hit = {5: np.zeros(n), 10: np.zeros(n), 20: np.zeros(n)}
    nd = {5: np.zeros(n), 10: np.zeros(n), 20: np.zeros(n)}
    for i in range(n):
        acc = {}
        for j in range(cand.shape[1]):
            it = int(cand[i, j])
            if it >= 0:
                acc[it] = acc.get(it, 0.0) + float(pr[i, j])
        if not acc:
            continue
        items = np.fromiter(acc.keys(), dtype=np.int64)
        vals = np.fromiter(acc.values(), dtype=np.float64)
        ranked = items[np.argsort(-vals, kind="stable")]
        pos = np.where(ranked == int(tgt[i]))[0]
        r = int(pos[0]) + 1 if len(pos) else 0
        for k in (5, 10, 20):
            if 0 < r <= k:
                hit[k][i] = 1.0
                nd[k][i] = 1.0 / np.log2(r + 1)
    out = {f"Recall@{k}": float(hit[k].mean()) for k in (5, 10, 20)}
    out.update({f"NDCG@{k}": float(nd[k].mean()) for k in (5, 10, 20)})
    out["n_users"] = int(n)
    if per_user:
        out["_pu"] = {**{f"Recall@{k}": hit[k] for k in (5, 10, 20)},
                      **{f"NDCG@{k}": nd[k] for k in (5, 10, 20)}}
    return out


def build_union_generic(dumps, conventions):
    """Mirror of `fusion_audit.build_union`, generic in the number of factorizations.

    Same semantics, same accumulation order; the CPU stage asserts element-wise
    equality with the historical `fusion_audit.build_union` for the 4-factor case.
    Item ids are canonicalised with each dump's MEASURED convention.
    """
    fa = import_fusion_audit()
    m_factors = len(dumps)
    shapes = {np.asarray(d["candidate_item_ids"]).shape for d in dumps}
    if len(shapes) != 1:
        raise SystemExit(f"HARD FAIL: union inputs have different candidate shapes "
                         f"{sorted(shapes)}")
    for d, c in zip(dumps, conventions):
        if len(np.asarray(d["target_item_ids"])) != np.asarray(
                d["candidate_item_ids"]).shape[0]:
            raise SystemExit("HARD FAIL: target/candidate row mismatch in a union input")
    n, B = np.asarray(dumps[0]["candidate_item_ids"]).shape
    target = canonical_targets(dumps[0], conventions[0])
    cand = [canonical_candidates(d, c) for d, c in zip(dumps, conventions)]
    probs = []
    for d, c in zip(dumps, cand):
        s = np.asarray(d["sequence_scores"], dtype=np.float64)
        probs.append(fa._row_softmax(s, c >= 0))
    umax = m_factors * B
    items_pad = np.full((n, umax), -1, dtype=np.int64)
    score_tensor = np.zeros((n, umax, m_factors), dtype=np.float64)
    for i in range(n):
        pos = {}
        p = 0
        for m in range(m_factors):
            ci = cand[m][i]
            for j in range(ci.shape[0]):
                it = int(ci[j])
                if it < 0:
                    continue
                if it not in pos:
                    pos[it] = p
                    items_pad[i, p] = it
                    p += 1
                score_tensor[i, pos[it], m] += probs[m][i, j]
    target_pos = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        w = np.where(items_pad[i] == target[i])[0]
        if len(w):
            target_pos[i] = w[0]
    valid_mask = items_pad >= 0
    if not np.isfinite(score_tensor).all():
        raise SystemExit("HARD FAIL: non-finite fused score tensor")
    return items_pad, score_tensor, target, target_pos, valid_mask


def per_user_ndcg_from_scores(fused, target_pos, valid_mask, topk=(5, 10, 20)):
    """Per-user Recall/NDCG for an arbitrary per-row score matrix.

    Ranking rule identical to `fusion_audit.per_user_ndcg` (rank = #strictly-greater
    valid candidates + 1); the CPU stage asserts exact agreement for score matrices
    produced by `per_user_ndcg`, so this cannot drift from the audited evaluator.
    """
    if fused.shape != valid_mask.shape:
        raise SystemExit(f"HARD FAIL: fused {fused.shape} != mask {valid_mask.shape}")
    if not np.isfinite(fused).all():
        raise SystemExit("HARD FAIL: non-finite fused scores")
    n = fused.shape[0]
    rows = np.where(target_pos >= 0)[0]
    out = {}
    ranks = np.full(n, -1, dtype=np.int64)
    if len(rows):
        ts = fused[rows, target_pos[rows]]
        r = ((fused[rows] > ts[:, None]) & valid_mask[rows]).sum(axis=1) + 1
        ranks[rows] = r
    for k in topk:
        hit = (ranks >= 1) & (ranks <= k)
        nd = np.zeros(n)
        nd[hit] = 1.0 / np.log2(ranks[hit] + 1)
        rc = np.zeros(n)
        rc[hit] = 1.0
        out[f"NDCG@{k}"] = nd
        out[f"Recall@{k}"] = rc
    out["_rank"] = ranks.astype(np.float64)
    out["_target_present"] = (target_pos >= 0).astype(np.float64)
    return out


def summarize_pu(pu):
    return {f"{m}@{k}": float(pu[f"{m}@{k}"].mean())
            for m in ("Recall", "NDCG") for k in (5, 10, 20)}


def structure_q_codebook(cb):
    """Q_K (K=2..5) from the per-codebook structure weights JSON (read, not recomputed)."""
    p = ART_ROOT / f"model/beam_dumps_abspearson_v3_cb{cb}" / \
        f"structure_weights_abspearson_v3_cb{cb}.json"
    obj = read_json(p)
    recs = sorted(obj["records"], key=lambda r: int(r["k"]))
    ks = [int(r["k"]) for r in recs]
    if ks != KS:
        raise SystemExit(f"HARD FAIL: {p} has K records {ks}, expected {KS}")
    return np.array([float(r["Q"]) for r in recs], dtype=np.float64), str(p), ks


# ===========================================================================
# GPU exclusivity
# ===========================================================================
def gpu_compute_apps():
    """Read-only `nvidia-smi` query: which processes own the device. No CUDA context."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi unavailable: {exc}",
                "procs": []}
    if out.returncode != 0:
        return {"available": False, "reason": f"nvidia-smi rc={out.returncode}: "
                                             f"{out.stderr.strip()}", "procs": []}
    procs = []
    for line in out.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            procs.append({"pid": int(parts[0]), "process_name": parts[1],
                          "used_memory": parts[2]})
    return {"available": True, "reason": "", "procs": procs}


def assert_gpu_exclusive(allow_pids=()):
    """Refuse to run a GPU stage while any other process owns the device."""
    st = gpu_compute_apps()
    if not st["available"]:
        raise SystemExit(f"HARD FAIL: GPU ownership cannot be established: {st['reason']}")
    others = [p for p in st["procs"]
              if p["pid"] != os.getpid() and p["pid"] not in set(allow_pids)]
    if others:
        detail = "; ".join(f"pid {p['pid']} ({p['process_name']}, {p['used_memory']})"
                           for p in others)
        raise SystemExit(
            "REFUSING TO RUN THE GPU STAGE: the device is owned by another job "
            f"({detail}).\nThis stage needs the GPU for teacher-forced "
            "cross-factorization scoring; running it now would contend with the "
            "training job and would mis-measure nothing but contention.\n"
            f"Re-run when the device is free:\n  {CFLA_CMD}\n"
            "Meanwhile every CPU-only result is already written under "
            f"{OUT}.")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"HARD FAIL: torch is required for the GPU stage: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("HARD FAIL: no process owns the device but "
                         "torch.cuda.is_available() is False; the GPU stage cannot run")
    return {"device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "owners_checked": st["procs"]}


# ===========================================================================
# CFLA stage (GPU) -- implemented, refused while the device is busy
# ===========================================================================
def build_history_tokens_for_stride(split, sid_path, stride, sid_len, max_len=20,
                                    batch=512):
    """Canonical evaluation-mode history tokens for a codebook cell (GPU stage only).

    The history is tokenised with the SAME SID array the cell's model was trained
    with and the MEASURED stride (never `256`), which is the `item2code` rule
    `token = code + position * stride + 1`.  A stride smaller than a code's value
    raises inside `item2code` rather than being silently folded.

    Shape: `GenRecDataset` flattens each history ITEM into its `sid_len` offset
    tokens, so the returned width is `max_len * sid_len` (120 for the 5+1 design),
    not `max_len`.  Both are asserted here.
    """
    if str(MF) not in sys.path:
        sys.path.insert(0, str(MF))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.tiger.dataset import GenRecDataset
    from src.tiger.dataloader import GenRecDataLoader
    ds_path = ART_ROOT / "data" / DATASET / f"{split}.parquet"
    if not ds_path.exists():
        raise SystemExit(f"HARD FAIL: missing dataset parquet {ds_path}")
    base = GenRecDataset(dataset_path=str(ds_path), code_path=str(sid_path),
                         mode="evaluation", max_len=max_len,
                         codebook_size=int(stride))
    vocab = int(stride) * int(sid_len) + 1
    loader = GenRecDataLoader(base, batch_size=batch, shuffle=False)
    toks, masks = [], []
    for b in loader:
        toks.append(b["history"].numpy().astype(np.int64))
        masks.append(b["attention_mask"].numpy().astype(np.int64))
    toks = np.concatenate(toks)
    masks = np.concatenate(masks)
    want = int(max_len) * int(sid_len)
    if toks.shape[1] != want:
        raise SystemExit(f"HARD FAIL: history width {toks.shape[1]} != max_len*"
                         f"sid_len = {want}")
    if masks.shape != toks.shape:
        raise SystemExit(f"HARD FAIL: mask shape {masks.shape} != tokens "
                         f"{toks.shape}")
    if toks.min() < 0 or int(toks.max()) >= vocab:
        raise SystemExit(f"HARD FAIL: history token range [{int(toks.min())}, "
                         f"{int(toks.max())}] outside [0, {vocab}) for stride {stride}")
    return toks, masks


def load_tiger_checkpoint(path, stride, sid_len, device="cuda"):
    """Load a frozen TIGER checkpoint, checking its vocabulary against the measured one."""
    if str(MF) not in sys.path:
        sys.path.insert(0, str(MF))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch
    from mf_models import _tiger, infer_cfg
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"HARD FAIL: missing checkpoint {p}")
    obj = torch.load(str(p), map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit(f"HARD FAIL: {p} is not a state dict")
    # FIXED 2026-10-13 (final paper completion round, section 23).
    #
    # The old loader stripped a `model.` prefix whenever EVERY key had one:
    #     sd = {k[len("model."):]: v ...} if all(k.startswith("model.") for k in obj)
    # That is a guess about the FILE, but what matters is what the MODULE expects.
    # `_tiger(cfg)` is a wrapper whose parameters really are named `model.*`, so on
    # the S4 run every key was stripped and `load_state_dict(strict=True)` died with
    # "Missing key(s) in state_dict: model.shared.weight, ..." -- which is exactly
    # how `--stage cfla-gpu` failed at 2026-09-18 00:38.
    #
    # Now both spellings are offered and the one the module ACCEPTS is used.  No
    # prefix is ever removed silently: strict=True on both attempts, and if neither
    # fits it is a hard failure with the real missing/unexpected keys.
    raw_sd = dict(obj)
    stripped = ({k[len("model."):]: v for k, v in obj.items()}
                if all(k.startswith("model.") for k in obj) else None)
    cfg = infer_cfg(raw_sd)
    expect = int(stride) * int(sid_len) + 1
    if int(cfg["vocab_size"]) != expect:
        raise SystemExit(f"HARD FAIL: {p}: vocab {cfg['vocab_size']} != measured "
                         f"stride*sid_len+1 = {expect}")
    model = _tiger(cfg)
    attempts = [("raw", raw_sd)]
    if stripped is not None:
        attempts.append(("model.-stripped", stripped))
    errs = {}
    for how, sd in attempts:
        try:
            model.load_state_dict(sd, strict=True)
            break
        except RuntimeError as e:
            errs[how] = str(e).splitlines()[0][:160]
    else:
        raise SystemExit(f"HARD FAIL: {p}: neither the raw nor the model.-prefixed "
                         f"state dict loads with strict=True: {errs}")
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model, cfg


def union_row_space(items_pad):
    """(user, candidate) rows of a union tensor: flat indices of every valid slot."""
    users, slots = np.nonzero(items_pad >= 0)
    cands = items_pad[users, slots]
    if len(users) == 0:
        raise SystemExit("HARD FAIL: empty union")
    return users.astype(np.int64), cands.astype(np.int64), slots.astype(np.int64)


def stage_cfla_gpu(args):
    """CFLA across the tested codebooks: teacher-forced z-NLL LSE over the K2..K5 pool.

    score(u,i) = logsumexp_F( -(NLL_F(u,i) - mu_F) / sigma_F )
    (mu_F, sigma_F) are fitted on ALL VALIDATION candidates of that (codebook, F),
    exactly as `mf_core.fit_scalers` does, and are then applied unchanged to TEST.
    """
    gpu = assert_gpu_exclusive(allow_pids=args.allow_pid or ())
    if str(MF) not in sys.path:
        sys.path.insert(0, str(MF))
    import torch
    from mf_models import candidate_total_nll

    # The (mu, sigma) scalers are fitted on VALID and applied to TEST, so VALID must
    # be scored first in the same invocation.  Refuse an ordering that would need a
    # scaler that was never fitted (silently reusing a scaler from another codebook is
    # exactly the kind of cross-cell leak this study must not have).
    if "test" in args.splits and "valid" not in args.splits:
        raise SystemExit(
            "HARD FAIL: --splits test requires --splits valid as well; the z-NLL "
            "scalers (mu_F, sigma_F) are fitted on VALID and applied to TEST.")
    splits_ordered = [s for s in ("valid", "test") if s in args.splits]

    inv = load_inventory()
    cells = beauty_cells(inv)
    gpu_dir = OUT / "gpu_runs" / stamp()
    gpu_dir.mkdir(parents=True, exist_ok=False)
    codebooks_present = [c for c in MATCHED_CODEBOOKS
                         if all(dump_path(cells[(c, k)], "valid").exists()
                                and dump_path(cells[(c, k)], "test").exists()
                                for k in KS)]
    todo = [c for c in (args.codebooks or codebooks_present) if c in codebooks_present]
    completeness = ("SMOKE_FIRST_N_USERS" if args.limit_users else "FULL_UNIVERSE")
    print(f"[cfla-gpu] device={gpu['device']} torch={gpu['torch']}")
    print(f"[cfla-gpu] codebooks={todo}  splits={splits_ordered}  "
          f"completeness={completeness}  out={gpu_dir}")
    if args.limit_users:
        print(f"[cfla-gpu] SMOKE RUN: only the first {args.limit_users} users; these "
              "numbers are NOT the study result and are never folded into the plot "
              "source")

    scalers, rows, boot_rows, audit = {}, [], [], []
    for cb in todo:
        metas = {}
        for k in KS:
            cell = cells[(cb, k)]
            for split in ("valid", "test"):
                dp = dump_path(cell, split)
                metas[(k, split)] = measure_dump(dp, cell)
        dumps = {}
        for split in ("valid", "test"):
            dumps[split] = [load_dump(dump_path(cells[(cb, k)], split)) for k in KS]
        conv = {split: [metas[(k, split)]["item_id_convention"] for k in KS]
                for split in ("valid", "test")}
        unions = {split: build_union_generic(dumps[split], conv[split])
                  for split in ("valid", "test")}
        hist = {}
        for split in splits_ordered:
            for k in KS:
                # Each factorization has its OWN model and therefore its OWN history
                # token space: the stride is per (codebook, K) and is MEASURED.
                hist[(k, split)] = build_history_tokens_for_stride(
                    split, cells[(cb, k)]["sid_path"],
                    metas[(k, split)]["stride_embedded"],
                    metas[(k, split)]["sid_len"])

        for split in splits_ordered:
            U = unions[split]
            items_pad, score_tensor, target, target_pos, valid_mask = U
            users, cands, slots = union_row_space(items_pad)
            if args.limit_users:
                keep = users < int(args.limit_users)
                users, cands, slots = users[keep], cands[keep], slots[keep]
            n_users = int(users.max()) + 1 if len(users) else 0
            # Evaluate ONLY the rows that were actually scored: a smoke run leaves the
            # remaining rows unscored, and scoring them as -1e30 would rank every one
            # of their candidates first (an inflated fake metric).
            view = slice(0, n_users)
            items_v = items_pad[view]
            score_v = score_tensor[view]
            tpos_v = target_pos[view]
            vmask_v = valid_mask[view]
            order = rows_lexsort(users, cands)
            nll_cols = {}
            for m, k in enumerate(KS):
                model, cfg = load_tiger_checkpoint(
                    cells[(cb, k)]["ckpt_path"], metas[(k, split)]["stride_embedded"],
                    metas[(k, split)]["sid_len"], device="cuda")
                sid = np.load(cells[(cb, k)]["sid_path"], allow_pickle=True).astype(np.int64)
                cand_rows = cands[order]                       # canonical 1-based item id
                # FIXED 2026-10-13 (section 22): `candidate_total_nll` takes
                # SLOT-ENCODED decoder tokens, not raw SID codes.  The old line was
                #     cand_tokens = sid[cand_rows - 1]
                # i.e. raw codes were teacher-forced, so the CFLA column would have
                # scored a different token namespace from the one the model was
                # trained on.  `mf_models.candidate_total_nll` uses the tokens both
                # as decoder input and as a gather index, so this is silent, not a
                # crash:     token = code + position * stride + 1
                _stride_k = int(metas[(k, split)]["stride_embedded"])
                _L_k = int(metas[(k, split)]["sid_len"])
                cand_codes = sid[cand_rows - 1]                # SID row = item id - 1
                if (cand_codes < 0).any():
                    raise SystemExit("HARD FAIL: negative SID code")
                if cand_codes.shape[1] != _L_k:
                    raise SystemExit("HARD FAIL: candidate token width != dump sid_len")
                cand_tokens = cand_codes + (
                    np.arange(_L_k, dtype=np.int64) * _stride_k + 1)
                if cand_tokens.min() < 1 or cand_tokens.max() >= int(cfg["vocab_size"]):
                    raise SystemExit(
                        f"HARD FAIL: slot-encoded candidate tokens "
                        f"[{int(cand_tokens.min())}, {int(cand_tokens.max())}] outside "
                        f"[1, {int(cfg['vocab_size'])})")
                h_tok, h_mask = hist[(k, split)]
                h_tok, h_mask = h_tok[:n_users], h_mask[:n_users]
                # FIXED 2026-10-13: `candidate_total_nll` takes the TIGER WRAPPER
                # and calls `model.model(...)` itself (mf_models.py:233).  Passing
                # `model.model` therefore double-unwrapped and raised
                # "AttributeError: 'T5ForConditionalGeneration' object has no
                # attribute 'model'".  Pass the wrapper.
                nll_sorted = candidate_total_nll(
                    model, h_tok, h_mask, cand_tokens, users[order],
                    device="cuda", batch_size=args.batch_size)
                if np.isnan(nll_sorted).any():
                    raise SystemExit(f"HARD FAIL: NaN NLL cb{cb} K{k} {split}")
                nll = scatter_scores(nll_sorted, order)

                def _direct(idx, _m=model, _sid=sid, _h=h_tok, _hm=h_mask,
                            _L=_L_k, _S=_stride_k):
                    # Same slot encoding as the stream above -- the ownership
                    # verifier is only meaningful if both sides speak one
                    # namespace (section 22).
                    _codes = _sid[cands[idx] - 1]
                    _enc = _codes + (np.arange(_L, dtype=np.int64) * _S + 1)
                    tl = torch.as_tensor(_enc, dtype=torch.long, device="cuda")
                    own = users[idx]
                    inp = torch.as_tensor(_h[own], dtype=torch.long, device="cuda")
                    am = torch.as_tensor(_hm[own], dtype=torch.long, device="cuda")
                    bos = torch.zeros((len(idx), 1), dtype=torch.long, device="cuda")
                    dec = torch.cat([bos, tl[:, :-1]], dim=1)
                    lg = _m.model(input_ids=inp, attention_mask=am,
                                  decoder_input_ids=dec).logits
                    lp = torch.log_softmax(lg.float(), dim=-1)
                    return (-lp.gather(2, tl.unsqueeze(-1)).squeeze(-1)
                            .sum(dim=1).double().cpu().numpy())

                # FIXED 2026-10-13: the default atol=1e-9 is far below the
                # float32 batching noise of this scorer.  The bulk stream and the
                # direct re-score batch the same rows differently and agree only to
                # ~6e-6 (same defect and same reasoning as
                # score_m5_candidate_nll.write_back_and_verify_calibrated).  1e-4 is
                # the tolerance the rest of the project uses for float32 NLL
                # comparisons and is ~1e4x smaller than a genuine row permutation.
                audit.append(assert_rows_own_their_values(
                    users, cands, nll, _direct, n_sample=args.check_rows,
                    label=f"cfla cb{cb} K{k} {split}", atol=1e-4))
                nll_cols[k] = nll
                del model
                torch.cuda.empty_cache()

            if split == "valid":
                for k in KS:
                    x = nll_cols[k]
                    sd = float(x.std())
                    if sd < 1e-12:
                        raise SystemExit(f"HARD FAIL: degenerate NLL spread cb{cb} K{k}")
                    scalers[(cb, k)] = {"mu": float(x.mean()), "sigma": sd,
                                        "n_rows": int(len(x))}
            Z = []
            for k in KS:
                sc = scalers[(cb, k)]
                Z.append(-(nll_cols[k] - sc["mu"]) / sc["sigma"])
            # FIXED 2026-10-13: was `np.full(..., np.nan, ...)`, which left every
            # PADDED slot NaN (union_row_space covers only items_pad >= 0; the union
            # is ~109 of umax = 5*30 = 150 slots, so padding always exists).  The
            # mask-aware check on the next line passes, but
            # `per_user_ndcg_from_scores` opens with an UNMASKED
            # `np.isfinite(fused).all()` and hard-failed every time.
            # Padding is never read: that function only ever touches `fused[rows]`
            # for rows with a target and gates every rank on `valid_mask[rows]`.
            # 0.0 is what the CPU path already leaves there (its dense matrix has
            # zero padding), so this also makes the two paths consistent.  The
            # strict finiteness guard is preserved for genuinely bad VALID scores,
            # because line below always writes every valid slot.
            fused = np.zeros(items_v.shape, dtype=np.float64)
            fused[users, slots] = _logsumexp_stack(Z)
            if np.isnan(fused[vmask_v]).any():
                raise SystemExit(f"HARD FAIL: unscored rows remained in the CFLA score "
                                 f"matrix (cb{cb} {split})")
            pu_lse = per_user_ndcg_from_scores(fused, tpos_v, vmask_v)
            pu_uni = import_fusion_audit().per_user_ndcg(
                items_v, score_v, tpos_v, vmask_v,
                np.ones((items_v.shape[0], len(KS))) / len(KS))
            for name, pu in (("cfla_lse", pu_lse), ("uniform", pu_uni)):
                rows.append({"dataset": DATASET, "codebook": cb, "method": name,
                             "split": split, "n_users": int(items_v.shape[0]),
                             "completeness": completeness,
                             **summarize_pu(pu)})
            for metric in METRICS:
                r = import_phase5().paired_bootstrap(pu_lse[metric], pu_uni[metric])
                boot_rows.append({"dataset": DATASET, "codebook": cb, "split": split,
                                  "metric": metric, "comparison": "cfla_lse_vs_uniform",
                                  "completeness": completeness,
                                  "a_value": float(pu_lse[metric].mean()),
                                  "b_value": float(pu_uni[metric].mean()),
                                  "delta": r["delta"], "ci_lo": r["ci_lo"],
                                  "ci_hi": r["ci_hi"], "p_value": r["p_value"],
                                  "n_boot": N_BOOT, "seed": RNG_SEED,
                                  "n_users": r["n_users"]})
            print(f"[cfla-gpu] cb{cb} {split}: "
                  f"cfla N@20={float(pu_lse['NDCG@20'].mean()):.6f} "
                  f"uniform N@20={float(pu_uni['NDCG@20'].mean()):.6f}")

    write_csv_new(gpu_dir / "CFLA_CODEBOOK_RESULTS.csv",
                  ["dataset", "codebook", "method", "split", "n_users", "completeness"]
                  + METRICS, rows)
    write_csv_new(gpu_dir / "CFLA_BOOTSTRAP.csv",
                  ["dataset", "codebook", "split", "metric", "comparison",
                   "completeness", "a_value", "b_value", "delta", "ci_lo", "ci_hi",
                   "p_value", "n_boot", "seed", "n_users"], boot_rows)
    write_json_new(gpu_dir / "RUN_METADATA.json", {
        "stage": "cfla-gpu", "generated": now_iso(), "gpu": gpu,
        "codebooks": todo, "splits": splits_ordered,
        "completeness": completeness,
        "smoke_limited_users": int(args.limit_users),
        "definition": "score(u,i) = logsumexp_F(-(NLL_F(u,i)-mu_F)/sigma_F); "
                      "(mu_F,sigma_F) fitted on ALL VALID candidates and applied to TEST",
        "scalers": {f"cb{cb}_k{k}": v for (cb, k), v in scalers.items()},
        "row_alignment_audit": audit,
        "selection": "no codebook selected; TEST never used to choose anything",
        "fold_into_plot_source": (
            "only rows with completeness == 'FULL_UNIVERSE' are folded into "
            "BEAUTY_CODEBOOK_PLOT_SOURCE.csv by a later `--stage cpu` run"),
    })
    print(f"[cfla-gpu] wrote {gpu_dir}")


def _logsumexp_stack(parts):
    stack = np.stack(parts, axis=0)
    m = stack.max(axis=0)
    return m + np.log(np.exp(stack - m).sum(axis=0))


# ===========================================================================
# CPU stage
# ===========================================================================
def validate_all(verify_hashes=True, verify_ckpt=False, verbose=False):
    """Validate every input path, hash, dump header, stride and convention. CPU only."""
    t0 = time.time()
    inv = load_inventory()
    cells = beauty_cells(inv)
    sha_cache = {}
    meta, problems = {}, []
    for cb in sorted({c["codebook"] for c in cells.values()}):
        for k in KS:
            cell = cells[(cb, k)]
            if not cell["beam_dir"].exists():
                raise SystemExit(f"HARD FAIL: missing beam-dump directory "
                                 f"{cell['beam_dir']}")
            if not cell["ckpt_path"].exists():
                raise SystemExit(f"HARD FAIL: missing checkpoint {cell['ckpt_path']}")
            if not cell["sid_path"].exists():
                raise SystemExit(f"HARD FAIL: missing SID array {cell['sid_path']}")
            if verify_hashes:
                if sha256_file(cell["sid_path"], sha_cache) != cell["sid_sha256"]:
                    raise SystemExit(f"HARD FAIL: SID sha256 mismatch vs inventory: "
                                     f"{cell['sid_path']}")
                if sha256_file(cell["ckpt_path"], sha_cache) != cell["ckpt_sha256"]:
                    raise SystemExit(f"HARD FAIL: checkpoint sha256 mismatch vs "
                                     f"inventory: {cell['ckpt_path']}")
            for split, flag in (("valid", cell["valid_available"]),
                                ("test", cell["test_available"])):
                p = dump_path(cell, split)
                if flag and not p.exists():
                    raise SystemExit(f"HARD FAIL: inventory says {split} dump available "
                                     f"for cb{cb} k{k} but {p} is missing")
                if not flag and p.exists():
                    raise SystemExit(f"HARD FAIL: inventory says {split} dump missing "
                                     f"for cb{cb} k{k} but {p} exists (index is stale)")
                if p.exists():
                    meta[(cb, k, split)] = measure_dump(p, cell, sha_cache)
            if verbose:
                print(f"  validated cb{cb:<5d} K{k}  "
                      f"stride={meta[(cb, k, 'valid')]['stride_embedded'] if (cb,k,'valid') in meta else '-'}")
    if verify_ckpt:
        if str(MF) not in sys.path:
            sys.path.insert(0, str(MF))
        import torch
        from mf_models import infer_cfg
        for (cb, k), cell in sorted(cells.items()):
            obj = torch.load(str(cell["ckpt_path"]), map_location="cpu")
            sd = {kk[len("model."):]: v for kk, v in obj.items()} if all(
                kk.startswith("model.") for kk in obj) else dict(obj)
            cfg = infer_cfg(sd)
            ref = None
            for split in ("valid", "test"):
                if (cb, k, split) in meta:
                    ref = meta[(cb, k, split)]
                    break
            if ref is None:
                continue
            if int(cfg["vocab_size"]) != ref["vocab_size_embedded"]:
                raise SystemExit(f"HARD FAIL: {cell['ckpt_path']} vocab "
                                 f"{cfg['vocab_size']} != embedded dump vocab "
                                 f"{ref['vocab_size_embedded']}")
            del obj
    if verbose:
        print(f"  validated {len(meta)} dumps in {time.time() - t0:.1f}s")
    return inv, cells, meta, sha_cache, problems


def _parquet_targets(split):
    path = ART_ROOT / "data" / DATASET / f"{split}.parquet"
    if not path.exists():
        raise SystemExit(f"HARD FAIL: missing {path}")
    import pandas as pd
    df = pd.read_parquet(path)
    if "target" not in df.columns:
        raise SystemExit(f"HARD FAIL: {path} has no 'target' column")
    return df["target"].to_numpy(dtype=np.int64)


def cpu_stage(args):
    inv, cells, meta, sha_cache, _ = validate_all(
        verify_hashes=not args.skip_hashes, verify_ckpt=not args.skip_ckpt_check)
    fa = import_fusion_audit()
    p5 = import_phase5()

    # --- target-convention / user-order check (one per split is enough) ---------
    order_checks = []
    for split in ("valid", "test"):
        pq = _parquet_targets(split)
        order_checks.append({**verify_target_convention(meta[(256, 5, split)], pq),
                             "split": split})

    # --- per-codebook single-model metrics (K2..K5) ----------------------------
    singles = {}
    for cb in sorted({c["codebook"] for c in cells.values()}):
        for k in KS:
            for split in ("valid", "test"):
                key = (cb, k, split)
                if key not in meta:
                    continue
                m = score_single_dump(meta[key]["dump"], meta[key]["item_id_convention"],
                                      per_user=True)
                singles[key] = m

    # --- cross-check the K=5 numbers against the recovered metrics CSV ---------
    k5_csv = SPEC / "SPECTRAL_K5_CODEBOOK_METRICS.csv"
    k5_rows = {(int(r["codebook_size"]), int(r["K"])): r
               for r in csv.DictReader(k5_csv.open(encoding="utf-8"))}
    k5_crosscheck, k5_max_delta = [], 0.0
    for cb in MATCHED_CODEBOOKS:
        for split in ("valid", "test"):
            if (cb, 5, split) not in singles:
                continue
            for m in METRICS:
                mine = singles[(cb, 5, split)][m]
                theirs = float(k5_rows[(cb, 5)][f"{split}_{m}"])
                d = abs(mine - theirs)
                k5_max_delta = max(k5_max_delta, d)
                k5_crosscheck.append({"codebook": cb, "split": split, "metric": m,
                                      "recomputed": mine, "recovered_csv": theirs,
                                      "abs_delta": d})
                if d > 1e-12:
                    raise SystemExit(
                        f"HARD FAIL: my dump evaluator disagrees with "
                        f"SPECTRAL_K5_CODEBOOK_METRICS.csv at cb{cb} {split} {m}: "
                        f"{mine!r} vs {theirs!r} (|d|={d:.3e})")

    # --- RQ-VAE5(C=256) single-model, recomputed from the frozen dump ----------
    rq_cell = {
        "codebook": 256,
        "beams": {"valid": FE / "runs_5plus1/Beauty/rqvae5/beams/valid_k5_sid6.npz",
                  "test": FE / "runs_5plus1/Beauty/rqvae5/beams/test_k5_sid6.npz"},
        "sid": FE / "runs_5plus1/Beauty/rqvae5/sid/Beauty_t5_rqvae5_base.npy",
        "ckpt": FE / "runs_5plus1/Beauty/rqvae5/t5/best_model.pth",
        "results_csv": {
            "valid": FE / "single_5plus1/VALID_RESULTS_FINAL.csv",
            "test": FE / "single_5plus1/TEST_RESULTS_FINAL.csv"},
    }
    rq_meta, rq_single, rq_cross = {}, {}, []
    for split, p in rq_cell["beams"].items():
        if not p.exists():
            raise SystemExit(f"HARD FAIL: missing RQ-VAE5 beam dump {p}")
        rq_meta[split] = measure_dump(p)
        rq_single[split] = score_single_dump(p, rq_meta[split]["item_id_convention"],
                                            per_user=True)
        csv_rows = [r for r in csv.DictReader(rq_cell["results_csv"][split].open())
                    if r["dataset"] == DATASET and r["method"] == "rqvae5"]
        if len(csv_rows) != 1:
            raise SystemExit(f"HARD FAIL: expected exactly 1 Beauty rqvae5 row in "
                             f"{rq_cell['results_csv'][split]}, found {len(csv_rows)}")
        for m in METRICS:
            theirs = float(csv_rows[0][m])
            mine = rq_single[split][m]
            d = abs(mine - theirs)
            rq_cross.append({"split": split, "metric": m, "recomputed": mine,
                             "single_5plus1_csv": theirs, "abs_delta": d})
            if d > 1e-12:
                raise SystemExit(f"HARD FAIL: RQ-VAE5 C=256 recomputation disagrees "
                                 f"with single_5plus1 at {split} {m}: {mine!r} vs "
                                 f"{theirs!r}")
    if rq_meta["valid"]["stride_embedded"] != 256:
        raise SystemExit("HARD FAIL: RQ-VAE5 C=256 stride != 256")
    # the RQ-VAE5 dump's own convention must also survive the parquet target check
    for split in ("valid", "test"):
        order_checks.append({**verify_target_convention(rq_meta[split],
                                                        _parquet_targets(split)),
                             "split": split, "cell": f"rqvae5_256_{split}"})

    # --- per-codebook K2..K5 pool fusion (uniform / best-single / dynamic) -----
    pool_rows, boot_rows, grid_rows, union_checks, cell_audit = [], [], [], [], []
    frozen_checks = []
    fusion = {}
    for cb in MATCHED_CODEBOOKS:
        if not all((cb, k, s) in meta for k in KS for s in ("valid", "test")):
            continue
        unions, dumps, convs = {}, {}, {}
        for split in ("valid", "test"):
            dumps[split] = [load_dump(meta[(cb, k, split)]["dump"]) for k in KS]
            convs[split] = [meta[(cb, k, split)]["item_id_convention"] for k in KS]
            unions[split] = build_union_generic(dumps[split], convs[split])
            if cb in (32, 256):     # audited-equality spot check (documented)
                if any(c != "raw_0based" for c in convs[split]):
                    raise SystemExit("HARD FAIL: the audited-equality check assumes the "
                                     "historical 0-based dumps")
                ref = fa.build_union(dumps[split])
                for got, exp, nm in zip(unions[split], ref,
                                        ("items_pad", "score_tensor", "target",
                                         "target_pos", "valid_mask")):
                    if not np.array_equal(np.asarray(got), np.asarray(exp)):
                        raise SystemExit(f"HARD FAIL: build_union_generic != "
                                         f"fusion_audit.build_union at cb{cb} {split} "
                                         f"({nm})")
                union_checks.append({"codebook": cb, "split": split,
                                     "elements": ["items_pad", "score_tensor", "target",
                                                  "target_pos", "valid_mask"],
                                     "equal": True})
        items_pad, score_tensor, target, target_pos, valid_mask = unions["valid"]
        n_valid = items_pad.shape[0]
        q_raw, q_path, q_ks = structure_q_codebook(cb)
        qn = fa.zscore(q_raw)
        conf = {s: fa.confidence_matrix(dumps[s]) for s in ("valid", "test")}
        cell_audit.append({"codebook": cb, "structure_json": q_path, "q_raw":
                           [float(x) for x in q_raw], "q_norm": [float(x) for x in qn]})

        # uniform
        uni = {}
        for split in ("valid", "test"):
            ip, st, tg, tp, vm = unions[split]
            uni[split] = fa.per_user_ndcg(ip, st, tp, vm,
                                          np.ones((ip.shape[0], len(KS))) / len(KS))
        # best single K, chosen on VALID NDCG@20 ONLY
        k_star = max(KS, key=lambda k: singles[(cb, k, "valid")]["NDCG@20"])
        bs = {}
        for split in ("valid", "test"):
            w = np.zeros((unions[split][0].shape[0], len(KS)))
            w[:, KS.index(k_star)] = 1.0
            bs[split] = fa.per_user_ndcg(unions[split][0], unions[split][1],
                                         unions[split][3], unions[split][4], w)
        # dynamic structure+confidence fusion, (gamma, eta) chosen on VALID ONLY
        best, table = p5.select_on_valid(unions["valid"], qn, conf["valid"])
        for t in table:
            grid_rows.append({"dataset": DATASET, "codebook": cb,
                              "split": "valid", **t})
        dyn = {"valid": fa.per_user_ndcg(
            unions["valid"][0], unions["valid"][1], unions["valid"][3],
            unions["valid"][4],
            fa.weights_from(best["gamma"], best["eta"], qn, conf["valid"]))}
        dyn["test"] = fa.per_user_ndcg(
            unions["test"][0], unions["test"][1], unions["test"][3], unions["test"][4],
            fa.weights_from(best["gamma"], best["eta"], qn, conf["test"]))

        # exact-agreement check: per_user_ndcg vs per_user_ndcg_from_scores
        ip, st, _tg, tp, vm = unions["test"]
        w = np.ones((ip.shape[0], len(KS))) / len(KS)
        direct = per_user_ndcg_from_scores(
            np.einsum("nik,nk->ni", st, w), tp, vm)
        if not np.array_equal(direct["NDCG@20"], uni["test"]["NDCG@20"]):
            raise SystemExit(f"HARD FAIL: per_user_ndcg_from_scores != per_user_ndcg "
                             f"at cb{cb} test")
        # row space + repaired write-back, exercised on a real union (CPU analogue)
        users, cands, slots = union_row_space(ip)
        order = rows_lexsort(users, cands)
        col = st[:, :, 0]
        vals_sorted = col[users[order], slots[order]]
        back = scatter_scores(vals_sorted, order)
        assert np.array_equal(back, col[users, slots])
        cell_audit.append(assert_rows_own_their_values(
            users, cands, back,
            lambda idx: col[users[idx], slots[idx]],
            n_sample=512, label=f"cpu union cb{cb} test"))

        # frozen historical Method-4 JSON cross-check (reproducible at its own gamma/eta)
        fj_v = read_json(ART_ROOT / f"model/beam_dumps_abspearson_v3_cb{cb}"
                                f"/fusion_valid_dynamic_abspearson_v3_cb{cb}.json")
        fj_t = read_json(ART_ROOT / f"model/beam_dumps_abspearson_v3_cb{cb}"
                                f"/fusion_test_dynamic_abspearson_v3_cb{cb}.json")
        if [float(x) for x in fj_v["q_norm"]] != [float(x) for x in qn]:
            raise SystemExit(f"HARD FAIL: cb{cb} q_norm differs from the frozen "
                             "structure weights JSON")
        rec = {}
        max_delta = 0.0
        for split, fj, cf in (("valid", fj_v, conf["valid"]), ("test", fj_t, conf["test"])):
            pu = fa.per_user_ndcg(unions[split][0], unions[split][1], unions[split][3],
                                  unions[split][4],
                                  fa.weights_from(fj_v["best_gamma"], fj_v["best_eta"],
                                                  qn, cf))
            rec[split] = pu
            for m in METRICS:
                max_delta = max(max_delta, abs(float(pu[m].mean())
                                               - float(fj["metrics"][m])))
        # A mismatch here is NOT an input error and must not abort the run: the
        # historical Method-4 JSONs are inputs whose dump files were partly
        # REPLACED after they were written (`run_beauty_codebook_method4.sh` skips
        # the fusion step when the JSON already exists).  Report it loudly with the
        # mtime evidence instead of inventing agreement or discarding the cell.
        json_paths = [ART_ROOT / f"model/beam_dumps_abspearson_v3_cb{cb}"
                              f"/fusion_{s}_dynamic_abspearson_v3_cb{cb}.json"
                      for s in ("valid", "test")]
        json_mtime = max(p.stat().st_mtime for p in json_paths)
        newer = []
        for k in KS:
            for split in ("valid", "test"):
                dp = Path(meta[(cb, k, split)]["dump"])
                if dp.stat().st_mtime > json_mtime:
                    newer.append({"input": str(dp),
                                  "mtime": datetime.fromtimestamp(
                                      dp.stat().st_mtime, CST).isoformat(timespec="seconds")})
        reproduced = bool(max_delta <= 1e-9 and not newer)
        m4_check = {
            "codebook": cb,
            "status": ("REPRODUCED" if reproduced else
                       "STALE_JSON_INPUT_REPLACED" if newer else "MISMATCH"),
            "max_abs_delta_over_12_cells": float(max_delta),
            "tolerance": 1e-9,
            "json_mtime": datetime.fromtimestamp(json_mtime,
                                                 CST).isoformat(timespec="seconds"),
            "inputs_newer_than_json": newer,
            "note": ("frozen Method-4 JSON re-derived bit-exactly at its own "
                     "(gamma, eta)" if reproduced else
                     "the frozen Method-4 JSON predates the beam dumps currently on "
                     "disk, so it cannot be re-derived and is EXCLUDED from the "
                     "usable data (see CODEBOOK_BLOCKER.md caveat C1 for cb256)"),
        }
        frozen_checks.append(m4_check)
        if not reproduced:
            print(f"  [cpu] !! cb{cb}: frozen Method-4 JSON NOT reproducible "
                  f"(max|d|={max_delta:.3e}); {len(newer)} dump(s) newer than the "
                  "JSON -> stale artifact, excluded (recorded, not raised)")

        fusion[cb] = {"uniform": uni, "best_single": bs, "k_star": int(k_star),
                      "dynamic": dyn, "gamma": best["gamma"], "eta": best["eta"],
                      "valid_grid_ndcg20": best["valid_NDCG@20"],
                      "frozen_method4_valid": fj_v, "frozen_method4_test": fj_t,
                      "method4_reproduced": reproduced,
                      "method4_check": m4_check,
                      "n_users": int(n_valid), "q_raw": q_raw,
                      "n_union_rows": {s: int(unions[s][4].sum())
                                       for s in ("valid", "test")}}

        # paired user-level bootstrap (5000 resamples, seed 20260912)
        for split in ("valid", "test"):
            for metric in METRICS:
                for a_name, a_pu, b_name, b_pu in (
                        ("uniform", uni[split], f"best_single_K{k_star}", bs[split]),
                        ("dynamic", dyn[split], "uniform", uni[split]),
                        ("dynamic", dyn[split], f"best_single_K{k_star}", bs[split])):
                    r = p5.paired_bootstrap(a_pu[metric], b_pu[metric])
                    boot_rows.append({
                        "dataset": DATASET, "codebook": cb, "split": split,
                        "metric": metric, "comparison": f"{a_name}_vs_{b_name}",
                        "a_value": float(a_pu[metric].mean()),
                        "b_value": float(b_pu[metric].mean()),
                        "delta": r["delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "p_value": r["p_value"], "n_boot": N_BOOT, "seed": RNG_SEED,
                        "n_users": r["n_users"]})

        for method, pu in (("uniform", uni), ("best_single", bs), ("dynamic", dyn)):
            for split in ("valid", "test"):
                pool_rows.append({
                    "dataset": DATASET, "codebook": cb, "method": method,
                    "pool": "K2_K3_K4_K5", "split": split,
                    "n_users": int(n_valid), **{m: float(pu[split][m].mean())
                                                for m in METRICS},
                    "selection_source": ("VALID item-level NDCG@20 (best K)"
                                         if method == "best_single" else
                                         "VALID item-level NDCG@20 (gamma,eta grid)"
                                         if method == "dynamic" else "fixed (no selection)"),
                    "selected_k": int(k_star) if method in ("best_single",) else "",
                    "selected_gamma": best["gamma"] if method == "dynamic" else "",
                    "selected_eta": best["eta"] if method == "dynamic" else "",
                    "status": "MEASURED", "provenance": "frozen Beauty beam dumps"})
        print(f"  [cpu] cb{cb:<5d} best_single=K{k_star} "
              f"uniform_N20={float(uni['test']['NDCG@20'].mean()):.6f} "
              f"dyn_N20={float(dyn['test']['NDCG@20'].mean()):.6f} "
              f"(g={best['gamma']},e={best['eta']})")

    # --- matched pair at C=256 (both sides exist) -----------------------------
    matched = {}
    mp = FE / "runs_5plus1/Beauty/rqvae5/beams"
    for split in ("valid", "test"):
        sp_dump = meta[(256, 5, split)]
        rq_dump = rq_meta[split]
        if sp_dump["stride_embedded"] != rq_dump["stride_embedded"]:
            raise SystemExit("HARD FAIL: matched pair has different strides")
        dumps = [load_dump(sp_dump["dump"]), load_dump(rq_dump["dump"])]
        convs = [sp_dump["item_id_convention"], rq_dump["item_id_convention"]]
        ip, st, tg, tp, vm = build_union_generic(dumps, convs)
        # cross-family alignment diagnostic (why the measured convention matters)
        a = canonical_candidates(dumps[0], convs[0])
        b = canonical_candidates(dumps[1], convs[1])
        naive = [np.where(np.asarray(d["candidate_item_ids"]) >= 0,
                          np.asarray(d["candidate_item_ids"]) + 1, -1) for d in dumps]
        correct_pairs = sum(len(set(a[i][a[i] > 0]) & set(b[i][b[i] > 0]))
                            for i in range(a.shape[0]))
        naive_pairs = sum(len(set(naive[0][i][naive[0][i] > 0])
                             & set(naive[1][i][naive[1][i] > 0]))
                          for i in range(a.shape[0]))
        w = np.ones((ip.shape[0], 2)) / 2.0
        uni = fa.per_user_ndcg(ip, st, tp, vm, w)
        k_star = max(("spectral_k5", "rqvae5"),
                     key=lambda nm: rq_single["valid"]["NDCG@20"]
                     if nm == "rqvae5" else singles[(256, 5, "valid")]["NDCG@20"])
        wb = np.zeros((ip.shape[0], 2))
        wb[:, 0 if k_star == "spectral_k5" else 1] = 1.0
        bs = fa.per_user_ndcg(ip, st, tp, vm, wb)
        matched[split] = {"uniform": uni, "best_single": bs, "k_star": k_star,
                          "n_users": int(ip.shape[0]),
                          "mean_union_size": float(vm.sum(axis=1).mean()),
                          "cross_family_pairs_correct_convention": int(correct_pairs),
                          "cross_family_pairs_naive_plus1": int(naive_pairs)}
        print(f"  [cpu] matched C=256 {split}: uniform_N20="
              f"{float(uni['NDCG@20'].mean()):.6f} best_single={k_star} "
              f"(N20={float(bs['NDCG@20'].mean()):.6f}) union_size="
              f"{float(vm.sum(axis=1).mean()):.2f}")

    # --- GPU-pending CFLA rows + blocked manifest ------------------------------
    cfla_measured = _load_cfla_gpu_results()
    blocked = _blocked_manifest(cells, rq_cell, meta)

    results = {
        "singles": singles, "fusion": fusion, "matched": matched,
        "rq_single": rq_single, "rq_meta": rq_meta, "meta": meta, "cells": cells,
        "k5_crosscheck": k5_crosscheck, "k5_max_delta": k5_max_delta,
        "rq_cross": rq_cross, "order_checks": order_checks,
        "union_checks": union_checks, "cell_audit": cell_audit,
        "pool_rows": pool_rows, "boot_rows": boot_rows, "grid_rows": grid_rows,
        "cfla_measured": cfla_measured, "blocked": blocked,
        "sha_cache": sha_cache, "inv": inv, "frozen_checks": frozen_checks,
    }
    return results


def _load_cfla_gpu_results():
    """Fold in any completed CFLA GPU run (never computed here, only read back).

    Only FULL-UNIVERSE rows are accepted: a `--limit-users` smoke run is a wiring
    check, not a measurement, and must never appear as a plotted number.
    """
    root = OUT / "gpu_runs"
    out = {}
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "CFLA_CODEBOOK_RESULTS.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r["method"] != "cfla_lse":
                continue
            if r.get("completeness", "") != "FULL_UNIVERSE":
                print(f"[cpu] ignoring {d.name}: completeness="
                      f"{r.get('completeness', '<missing>')} (smoke run, not folded in)")
                continue
            out[(int(r["codebook"]), r["split"])] = {
                "value": {m: float(r[m]) for m in METRICS},
                "n_users": int(r["n_users"]), "run_dir": str(d),
                "results_sha256": sha256_file(f)}
    return out


def _blocked_manifest(cells, rq_cell, meta):
    """Explicit blocked/pending rows: missing artifacts named, nothing approximated."""
    missing_rq = []
    for cb in MATCHED_CODEBOOKS:
        if cb == 256:
            continue
        arm_dir = FE / "runs_5plus1/Beauty" / f"rqvae5_cb{cb}"
        missing_rq.append({
            "codebook": cb,
            "missing_tokenizer_dir": str(arm_dir / "tokenizer"),
            "missing_sid": str(arm_dir / "sid" / "Beauty_t5_rqvae5_base.npy"),
            "missing_t5_ckpt": str(arm_dir / "t5" / "best_model.pth"),
            "missing_valid_beam": str(arm_dir / "beams" / "valid_k5_sid6.npz"),
            "missing_test_beam": str(arm_dir / "beams" / "test_k5_sid6.npz"),
            "blocker": "RQ-VAE5(C) was never trained for this codebook; only the "
                       "C=256 arm exists (this round's arm).",
            "cost": "1 tokenizer + 1 T5 recommender per codebook (~4 GPU-h per "
                    "codebook, ~16 GPU-h for the four missing codebooks)",
            "unblock_command": RQVAE5_UNBLOCK_CMD.replace("{C}", str(cb)),
        })
    pending_cfla = []
    for cb in MATCHED_CODEBOOKS:
        if all((cb, k, s) in meta for k in KS for s in ("valid", "test")):
            pending_cfla.append({
                "codebook": cb,
                "status": "PENDING_GPU_BUSY",
                "reason": "CFLA needs teacher-forced cross-factorization candidate NLL "
                          "(4 T5 forward passes per union candidate per codebook); the "
                          "device is owned by the running training job.",
                "unblock_command": CFLA_CMD + f" --codebooks {cb}",
            })
    return {"rqvae5_missing_codebooks": missing_rq, "cfla_pending": pending_cfla,
            "blocked_datasets": RQVAE5_UNBLOCK_CMD}


# ===========================================================================
# output writing
# ===========================================================================
def resolve_run_dir(force=None, create=True):
    """Canonical output dir on a first run; a fresh `runs/<stamp>/` if it is taken."""
    if create:
        OUT.mkdir(parents=True, exist_ok=True)
    if force:
        p = Path(force)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p, False
    if any((OUT / f).exists() for f in REQUIRED_OUTPUTS + EXTRA_OUTPUTS):
        p = OUT / "runs" / stamp()
        if create:
            p.mkdir(parents=True, exist_ok=False)
        return p, True
    return OUT, False


def _value_of(pu, m):
    """Metric value from either a per-user array dict or a plain scalar dict."""
    v = pu[m]
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v)
    return float(np.asarray(v, dtype=np.float64).mean())


def _metric_rows_from_pu(pu, dataset, codebook, method, pool, split, status,
                         selection_source, provenance, provenance_sha, note="",
                         extra=None):
    rows = []
    for m in METRICS:
        r = {"dataset": dataset, "codebook": codebook, "method": method, "pool": pool,
             "metric": m, "value": fmt(_value_of(pu, m)), "split": split,
             "status": status, "selection_source": selection_source,
             "provenance": provenance, "provenance_sha256": provenance_sha, "note": note}
        if extra:
            r.update(extra)
        rows.append(r)
    return rows


def write_outputs(res, args):
    run_dir, diverted = resolve_run_dir(args.run_dir)
    if diverted:
        print(f"[warn] canonical outputs already exist -> this run writes to {run_dir} "
              "(existing artifacts are never overwritten)")
    singles, fusion, matched = res["singles"], res["fusion"], res["matched"]
    cells, meta = res["cells"], res["meta"]
    sha = res["sha_cache"]
    k5_sha = sha256_file(SPEC / "SPECTRAL_K5_CODEBOOK_METRICS.csv", sha)
    inv_sha = sha256_file(SPEC / "SPECTRAL_CODEBOOK_INVENTORY.csv", sha)
    plt_rows = []

    # ---------------- RQ5_SINGLE_CODEBOOK_RESULTS.csv -------------------------
    rq_cols = ["dataset", "method", "codebook", "semantic_positions", "full_sid_length",
               "stride", "vocab_size", "split", "status", "n_users"] + METRICS + \
              ["selection_metric", "checkpoint", "checkpoint_sha256", "sid_path",
               "sid_sha256", "beam_dump", "beam_dump_sha256", "item_id_convention",
               "missing_artifacts", "unblock_command", "provenance", "note"]
    rq_rows = []
    for cb in MATCHED_CODEBOOKS:
        for split in ("valid", "test"):
            if cb == 256:
                m = res["rq_single"][split]
                mt = res["rq_meta"][split]
                rq_rows.append({
                    "dataset": DATASET, "method": "rqvae5", "codebook": 256,
                    "semantic_positions": 5, "full_sid_length": 6,
                    "stride": mt["stride_embedded"], "vocab_size": mt["vocab_size_embedded"],
                    "split": split, "status": "MEASURED", "n_users": m["n_users"],
                    **{k: fmt(m[k]) for k in METRICS},
                    "selection_metric": "item-level VALID NDCG@20 (early_stop 10)",
                    "checkpoint": str(Path(str(rq_meta_ckpt()))),
                    "checkpoint_sha256": sha256_file(rq_meta_ckpt(), sha),
                    "sid_path": str(Path(str(res["rq_meta"][split]["code_path"]))),
                    "sid_sha256": res["rq_meta"][split]["sid_sha256"],
                    "beam_dump": res["rq_meta"][split]["dump"],
                    "beam_dump_sha256": sha256_file(res["rq_meta"][split]["dump"], sha),
                    "item_id_convention": mt["item_id_convention"],
                    "missing_artifacts": "",
                    "unblock_command": "",
                    "provenance": "runs_5plus1/Beauty/rqvae5/beams/"
                                  f"{split}_k5_sid6.npz (re-scored) + "
                                  "single_5plus1/*_RESULTS_FINAL.csv (cross-checked, "
                                  f"max |d|={max(x['abs_delta'] for x in res['rq_cross'] if x['split'] == split):.2e})",
                    "note": "frozen 5+1 arm of this round; the ONLY codebook RQ-VAE5 "
                            "exists at"})
                plt_rows += _metric_rows_from_pu(
                    m, DATASET, 256, "rqvae5_single", "single", split, "MEASURED",
                    "frozen (no selection)",
                    res["rq_meta"][split]["dump"],
                    sha256_file(res["rq_meta"][split]["dump"], sha),
                    "RQ-VAE5 C=256; prefix arm, not a codebook sweep point")
            else:
                arm_dir = FE / "runs_5plus1/Beauty" / f"rqvae5_cb{cb}"
                miss = "; ".join([str(arm_dir / "sid" / "Beauty_t5_rqvae5_base.npy"),
                                  str(arm_dir / "t5" / "best_model.pth"),
                                  str(arm_dir / "beams" / f"{split}_k5_sid6.npz")])
                rq_rows.append({
                    "dataset": DATASET, "method": "rqvae5", "codebook": cb,
                    "semantic_positions": 5, "full_sid_length": 6, "stride": "",
                    "vocab_size": "", "split": split,
                    "status": "BLOCKED_MISSING_RQVAE5_ARM", "n_users": "",
                    **{k: "" for k in METRICS},
                    "selection_metric": "", "checkpoint": "", "checkpoint_sha256": "",
                    "sid_path": "", "sid_sha256": "", "beam_dump": "",
                    "beam_dump_sha256": "", "item_id_convention": "",
                    "missing_artifacts": miss,
                    "unblock_command": RQVAE5_UNBLOCK_CMD.replace("{C}", str(cb)),
                    "provenance": "MATCHED_CODEBOOK_SET.json (matched sizes) + "
                                  "CODEBOOK_BLOCKER.md",
                    "note": "NOT approximated: the RQ-VAE5(C) arm was never trained"})
                for m in METRICS:
                    plt_rows.append({
                        "dataset": DATASET, "codebook": cb, "method": "rqvae5_single",
                        "pool": "single", "metric": m, "value": "", "split": split,
                        "status": "BLOCKED_MISSING_RQVAE5_ARM",
                        "selection_source": "n/a (blocked)",
                        "provenance": "missing: " + miss,
                        "provenance_sha256": "",
                        "note": "no RQ-VAE5 arm at this codebook; blocked, not "
                                "estimated"})

    # ---------------- spectral-K5 single sensitivity (plot source) ------------
    for cb in MATCHED_CODEBOOKS:
        for split in ("valid", "test"):
            s = singles.get((cb, 5, split))
            if s is None:
                continue
            plt_rows += _metric_rows_from_pu(
                s, DATASET, cb, "spectral_k5_single", "single", split, "MEASURED",
                "frozen (no selection)", meta[(cb, 5, split)]["dump"],
                sha256_file(meta[(cb, 5, split)]["dump"], sha),
                "descriptive codebook sensitivity; no codebook was selected")
    # K2..K4 singles (context for best-single / uniform rows)
    for cb in MATCHED_CODEBOOKS:
        for k in (2, 3, 4):
            for split in ("valid", "test"):
                s = singles.get((cb, k, split))
                if s is None:
                    continue
                plt_rows += _metric_rows_from_pu(
                    s, DATASET, cb, f"spectral_k{k}_single", "single", split,
                    "MEASURED", "frozen (no selection)", meta[(cb, k, split)]["dump"],
                    sha256_file(meta[(cb, k, split)]["dump"], sha),
                    "pool member of the K2..K5 fusion")

    # ---------------- pool fusion (uniform / best-single / dynamic) -----------
    beauty_rows = []
    for cb, f in sorted(fusion.items()):
        for method, key, sel in (
                ("uniform_K2_K5", "uniform", "fixed (no selection)"),
                ("best_single_K2_K5", "best_single",
                 f"VALID item-level NDCG@20 -> K={f['k_star']}"),
                ("dynamic_structure_confidence_K2_K5", "dynamic",
                 f"VALID item-level NDCG@20 grid -> gamma={f['gamma']}, "
                 f"eta={f['eta']}")):
            for split in ("valid", "test"):
                pu = f[key][split]
                prov = f"{cb} codebook beam dumps K2..K5 + structure_weights json"
                plt_rows += _metric_rows_from_pu(
                    pu, DATASET, cb, method, "K2_K3_K4_K5", split, "MEASURED", sel,
                    prov, "")
                beauty_rows.append({
                    "dataset": DATASET, "codebook": cb, "method": method,
                    "pool": "K2_K3_K4_K5", "split": split,
                    "n_users": f["n_users"], **{m: fmt(float(pu[m].mean()))
                                                for m in METRICS},
                    "selection_source": sel, "status": "MEASURED",
                    "provenance": prov, "note": ""})
        # frozen historical Method-4 rows: usable only where the frozen JSON is
        # re-derivable from the dumps currently on disk
        m4_status = ("MEASURED_CROSSCHECK" if f["method4_reproduced"]
                     else "FROZEN_JSON_STALE_DUMP_EXCLUDED")
        m4_note = ("historical Method-4 artifact, re-derived bit-exactly"
                   if f["method4_reproduced"] else
                   "DO NOT PLOT: the frozen JSON predates the dumps on disk "
                   f"({len(f['method4_check']['inputs_newer_than_json'])} replaced "
                   f"input(s)); max|delta| = "
                   f"{f['method4_check']['max_abs_delta_over_12_cells']:.3e}")
        for split, key in (("valid", "frozen_method4_valid"),
                           ("test", "frozen_method4_test")):
            fr = f[key]
            for m in METRICS:
                plt_rows.append({
                    "dataset": DATASET, "codebook": cb, "method":
                        "method4_frozen_dynamic_fusion", "pool": "K2_K3_K4_K5",
                    "metric": m, "value": fmt(float(fr["metrics"][m])), "split": split,
                    "status": m4_status,
                    "selection_source": f"frozen JSON (gamma={fr.get('best_gamma', fr.get('gamma'))}, "
                                        f"eta={fr.get('best_eta', fr.get('eta'))})",
                    "provenance": str(ART_ROOT / f"model/beam_dumps_abspearson_v3_cb{cb}"
                                              f"/fusion_{split}_dynamic_abspearson_v3_cb{cb}.json"),
                    "provenance_sha256": "",
                    "note": m4_note})
            beauty_rows.append({
                "dataset": DATASET, "codebook": cb,
                "method": "method4_frozen_dynamic_fusion", "pool": "K2_K3_K4_K5",
                "split": split, "n_users": f["n_users"],
                **{m: fmt(float(fr["metrics"][m])) for m in METRICS},
                "selection_source": "frozen JSON (VALID-selected gamma/eta)",
                "status": m4_status,
                "provenance": str(ART_ROOT / f"model/beam_dumps_abspearson_v3_cb{cb}"
                                          f"/fusion_{split}_dynamic_abspearson_v3_cb{cb}.json"),
                "note": m4_note})

        # CFLA rows: measured if a GPU run exists, else PENDING_GPU_BUSY
        for split in ("valid", "test"):
            got = res["cfla_measured"].get((cb, split))
            if got:
                plt_rows += _metric_rows_from_pu(
                    got["value"], DATASET, cb, "cfla_lse_K2_K5", "K2_K3_K4_K5", split,
                    "MEASURED", "no selection (mu,sigma fitted on VALID)",
                    got["run_dir"], got["results_sha256"],
                    "logsumexp_F(-(NLL_F-mu_F)/sigma_F)")
                beauty_rows.append({
                    "dataset": DATASET, "codebook": cb, "method": "cfla_lse_K2_K5",
                    "pool": "K2_K3_K4_K5", "split": split,
                    "n_users": got["n_users"],
                    **{m: fmt(got["value"][m]) for m in METRICS},
                    "selection_source": "no selection (mu,sigma fitted on VALID)",
                    "status": "MEASURED", "provenance": got["run_dir"], "note": ""})
            else:
                for m in METRICS:
                    plt_rows.append({
                        "dataset": DATASET, "codebook": cb, "method": "cfla_lse_K2_K5",
                        "pool": "K2_K3_K4_K5", "metric": m, "value": "", "split": split,
                        "status": "PENDING_GPU_BUSY",
                        "selection_source": "n/a (GPU stage not run)",
                        "provenance": "not computed: " + CFLA_CMD,
                        "provenance_sha256": "",
                        "note": "CFLA needs GPU teacher-forced cross-factorization "
                                "scoring; the device is owned by the training job"})
                beauty_rows.append({
                    "dataset": DATASET, "codebook": cb, "method": "cfla_lse_K2_K5",
                    "pool": "K2_K3_K4_K5", "split": split, "n_users": "",
                    **{m: "" for m in METRICS},
                    "selection_source": "n/a (GPU stage not run)",
                    "status": "PENDING_GPU_BUSY",
                    "provenance": "not computed: " + CFLA_CMD,
                    "note": "implemented and ready; refused while the GPU is busy"})

    # ---------------- matched pair (C=256 measured, rest blocked) -------------
    matched_rows = []
    for cb in MATCHED_CODEBOOKS:
        if cb == 256:
            for split in ("valid", "test"):
                mp = matched[split]
                for method, key, sel in (
                        ("uniform_spectralK5_rqvae5", "uniform", "fixed (no selection)"),
                        ("best_single", "best_single",
                         f"VALID item-level NDCG@20 -> {mp['k_star']}")):
                    pu = mp[key]
                    prov = ("beam_dumps_abspearson_v3_cb256/valid|test_k5_sid6.npz + "
                            "runs_5plus1/Beauty/rqvae5/beams/valid|test_k5_sid6.npz")
                    plt_rows += _metric_rows_from_pu(
                        pu, DATASET, 256, f"matched_{method}", "spectral_k5+rqvae5",
                        split, "MEASURED", sel, prov, "")
                    matched_rows.append({
                        "dataset": DATASET, "method": method, "codebook": 256,
                        "pool": "spectral_k5+rqvae5", "split": split,
                        "n_users": mp["n_users"], **{m: fmt(float(pu[m].mean()))
                                                     for m in METRICS},
                        "selection_source": sel, "status": "MEASURED",
                        "spectral_dump": meta[(256, 5, split)]["dump"],
                        "rqvae5_dump": res["rq_meta"][split]["dump"],
                        "blocker": "", "unblock_command": "", "provenance": prov,
                        "note": f"mean union size {mp['mean_union_size']:.2f}; "
                                f"cross-family matched pairs with the MEASURED "
                                f"conventions {mp['cross_family_pairs_correct_convention']} "
                                f"vs {mp['cross_family_pairs_naive_plus1']} under a naive "
                                "uniform +1"})
                for m in METRICS:
                    plt_rows.append({
                        "dataset": DATASET, "codebook": 256,
                        "method": "matched_cfla_lse_spectralK5_rqvae5", "pool":
                            "spectral_k5+rqvae5", "metric": m, "value": "",
                        "split": split, "status": "PENDING_GPU_BUSY",
                        "selection_source": "n/a (GPU stage not run)",
                        "provenance": "not computed: " + CFLA_CMD,
                        "provenance_sha256": "",
                        "note": "CFLA over the matched pair also needs the GPU"})
        else:
            arm_dir = FE / "runs_5plus1/Beauty" / f"rqvae5_cb{cb}"
            miss = "; ".join([str(arm_dir / "sid" / "Beauty_t5_rqvae5_base.npy"),
                              str(arm_dir / "t5" / "best_model.pth"),
                              str(arm_dir / "beams" / "valid_k5_sid6.npz")])
            for split in ("valid", "test"):
                for method in ("uniform_spectralK5_rqvae5", "best_single",
                               "cfla_lse_spectralK5_rqvae5"):
                    matched_rows.append({
                        "dataset": DATASET, "method": method, "codebook": cb,
                        "pool": "spectral_k5+rqvae5", "split": split, "n_users": "",
                        **{m: "" for m in METRICS},
                        "selection_source": "n/a (blocked)", "status":
                            "BLOCKED_MISSING_RQVAE5_ARM",
                        "spectral_dump": str(dump_path(cells[(cb, 5)], split)),
                        "rqvae5_dump": str(arm_dir / "beams" / f"{split}_k5_sid6.npz"),
                        "blocker": "RQ-VAE5(C) does not exist for C != 256; the "
                                   "matched pairing has no second side",
                        "unblock_command": RQVAE5_UNBLOCK_CMD.replace("{C}", str(cb)),
                        "provenance": "MATCHED_CODEBOOK_SET.json",
                        "note": "blocked, not approximated"})
                    for m in METRICS:
                        plt_rows.append({
                            "dataset": DATASET, "codebook": cb,
                            "method": f"matched_{method}", "pool":
                                "spectral_k5+rqvae5", "metric": m, "value": "",
                            "split": split, "status": "BLOCKED_MISSING_RQVAE5_ARM",
                            "selection_source": "n/a (blocked)",
                            "provenance": "missing: " + miss,
                            "provenance_sha256": "",
                            "note": "matched pair incomplete: RQ-VAE5(C) never trained"})

    # ---------------- 1024: not in the matched set, no K5 dump ---------------
    for split in ("valid", "test"):
        for m in METRICS:
            plt_rows.append({
                "dataset": DATASET, "codebook": EXTRA_CODEBOOK,
                "method": "spectral_k5_single", "pool": "single", "metric": m,
                "value": "", "split": split,
                "status": "NOT_IN_MATCHED_SET_NO_K5_DUMP",
                "selection_source": "n/a",
                "provenance": "MATCHED_CODEBOOK_SET.json: "
                              "spectral_k5_codebooks_checkpoint_only=[1024]; "
                              f"missing {dump_path(cells[(EXTRA_CODEBOOK, 5)], split)}",
                "provenance_sha256": "",
                "note": "checkpoint only, no K5 beam dump; excluded from the matched set"})
    plt_rows.append({
        "dataset": DATASET, "codebook": EXTRA_CODEBOOK,
        "method": "method4_frozen_dynamic_fusion", "pool": "K2_K3_K4_K5",
        "metric": "NDCG@20", "value": "", "split": "test",
        "status": "FROZEN_JSON_NOT_REPRODUCIBLE",
        "selection_source": "n/a",
        "provenance": str(ART_ROOT / "model/beam_dumps_abspearson_v3_cb1024/"
                                  "fusion_test_dynamic_abspearson_v3_cb1024.json"),
        "provenance_sha256": "",
        "note": "the frozen cb1024 fusion JSON names a K5 dump "
                "(test_k5_sid6.npz) that no longer exists, so its metrics cannot be "
                "re-derived; its K2-K4 dumps were also rewritten AFTER the JSON "
                "(2026-07-22 vs 2026-07-20). Reported, not used."})

    plot_cols = ["dataset", "codebook", "method", "pool", "metric", "value", "split",
                 "status", "selection_source", "provenance", "provenance_sha256", "note"]
    write_csv_new(run_dir / "BEAUTY_CODEBOOK_PLOT_SOURCE.csv", plot_cols, plt_rows)
    write_csv_new(run_dir / "RQ5_SINGLE_CODEBOOK_RESULTS.csv", rq_cols, rq_rows)
    mcols = ["dataset", "method", "codebook", "pool", "split", "n_users"] + METRICS + \
            ["selection_source", "status", "spectral_dump", "rqvae5_dump", "blocker",
             "unblock_command", "provenance", "note"]
    write_csv_new(run_dir / "MATCHED_FUSION_CODEBOOK_RESULTS.csv", mcols, matched_rows)
    write_csv_new(run_dir / "BEAUTY_MATCHED_CODEBOOK_RESULTS.csv",
                  ["dataset", "codebook", "method", "pool", "split", "n_users"] +
                  METRICS + ["selection_source", "status", "provenance", "note"],
                  _beauty_sorted(beauty_rows))
    write_csv_new(run_dir / "CODEBOOK_FUSION_BOOTSTRAP.csv",
                  ["dataset", "codebook", "split", "metric", "comparison", "a_value",
                   "b_value", "delta", "ci_lo", "ci_hi", "p_value", "n_boot", "seed",
                   "n_users"], res["boot_rows"])
    write_csv_new(run_dir / "DYNAMIC_FUSION_VALID_GRID.csv",
                  ["dataset", "codebook", "split", "gamma", "eta",
                   "valid_NDCG@20", "valid_Recall@20"], res["grid_rows"])
    write_csv_new(run_dir / "CELL_frozen vocabulary audit.csv",
                  ["dataset", "codebook", "K", "split", "full_sid_length",
                   "vocab_size_embedded", "stride_embedded", "stride_sid_derived",
                   "semantic_codebook_measured", "disambiguation_alphabet",
                   "vocab_formula_C_times_L_plus_1", "vocab_formula_holds",
                   "item_id_convention", "reconstruction_worst_pos_rate",
                   "n_valid_candidates", "n_users", "sid_sha256", "dump",
                   "code_path", "ckpt_path"],
                  _cell_rows(res))
    write_csv_new(run_dir / "FROZEN_METHOD4_REPRODUCIBILITY.csv",
                  ["dataset", "codebook", "status", "max_abs_delta_over_12_cells",
                   "tolerance", "json_mtime", "n_inputs_newer_than_json",
                   "inputs_newer_than_json", "note"],
                  [{"dataset": DATASET, "codebook": c["codebook"],
                    "status": c["status"],
                    "max_abs_delta_over_12_cells": fmt(
                        c["max_abs_delta_over_12_cells"]),
                    "tolerance": c["tolerance"], "json_mtime": c["json_mtime"],
                    "n_inputs_newer_than_json": len(c["inputs_newer_than_json"]),
                    "inputs_newer_than_json": "; ".join(
                        f"{x['input']} ({x['mtime']})" for x in
                        c["inputs_newer_than_json"]),
                    "note": c["note"]} for c in res["frozen_checks"]])
    meta_obj = _metadata(res, run_dir, diverted, args, k5_sha, inv_sha)
    write_json_new(run_dir / "RUN_METADATA.json", meta_obj)
    write_json_new(run_dir / "_PROVENANCE.json", {
        "driver": str(DRIVERS / "build_codebook_sensitivity.py"),
        "driver_sha256": sha256_file(DRIVERS / "build_codebook_sensitivity.py"),
        "generated": now_iso(), "inputs": meta_obj["inputs_sha256"],
        "outputs": sorted([p.name for p in run_dir.iterdir() if p.is_file()]),
    })
    write_text_new(run_dir / "CODEBOOK_SENSITIVITY_REPORT.md",
                   _report(res, run_dir, meta_obj))
    print(f"[cpu] wrote {len(plt_rows)} plot-source rows -> {run_dir}")
    return run_dir


def rq_meta_ckpt():
    return FE / "runs_5plus1/Beauty/rqvae5/t5/best_model.pth"


def _beauty_sorted(rows):
    return sorted(rows, key=lambda r: (r["codebook"], r["method"], r["split"]))


def _cell_rows(res):
    rows = []
    for (cb, k, split), m in sorted(res["meta"].items()):
        rows.append({"dataset": DATASET, "codebook": cb, "K": k, "split": split,
                     "full_sid_length": m["sid_len"],
                     "vocab_size_embedded": m["vocab_size_embedded"],
                     "stride_embedded": m["stride_embedded"],
                     "stride_sid_derived": m["stride_sid_derived"],
                     "semantic_codebook_measured": m["semantic_codebook_measured"],
                     "disambiguation_alphabet": m["disambiguation_alphabet"],
                     "vocab_formula_C_times_L_plus_1":
                         m["vocab_formula_C_times_L_plus_1"],
                     "vocab_formula_holds": m["vocab_formula_holds"],
                     "item_id_convention": m["item_id_convention"],
                     "reconstruction_worst_pos_rate": fmt(
                         m["reconstruction_worst_pos_rate"]),
                     "n_valid_candidates": m["n_valid_candidates"],
                     "n_users": m["n_users"], "sid_sha256": m["sid_sha256"],
                     "dump": m["dump"], "code_path": m["code_path"],
                     "ckpt_path": m["ckpt_path"]})
    return rows


def _metadata(res, run_dir, diverted, args, k5_sha, inv_sha):
    cells, meta = res["cells"], res["meta"]
    inputs = {
        "SPECTRAL_CODEBOOK_INVENTORY.csv": {"path": str(SPEC / "SPECTRAL_CODEBOOK_INVENTORY.csv"),
                                            "sha256": inv_sha},
        "MATCHED_CODEBOOK_SET.json": {
            "path": str(SPEC / "MATCHED_CODEBOOK_SET.json"),
            "sha256": sha256_file(SPEC / "MATCHED_CODEBOOK_SET.json", res["sha_cache"])},
        "CODEBOOK_BLOCKER.md": {
            "path": str(SPEC / "CODEBOOK_BLOCKER.md"),
            "sha256": sha256_file(SPEC / "CODEBOOK_BLOCKER.md", res["sha_cache"])},
        "SPECTRAL_K5_CODEBOOK_METRICS.csv": {
            "path": str(SPEC / "SPECTRAL_K5_CODEBOOK_METRICS.csv"), "sha256": k5_sha},
        "SPECTRAL_K5_CODEBOOK_CROSSCHECK.csv": {
            "path": str(SPEC / "SPECTRAL_K5_CODEBOOK_CROSSCHECK.csv"),
            "sha256": sha256_file(SPEC / "SPECTRAL_K5_CODEBOOK_CROSSCHECK.csv",
                                  res["sha_cache"])},
        "single_5plus1/VALID_RESULTS_FINAL.csv": {
            "path": str(FE / "single_5plus1/VALID_RESULTS_FINAL.csv"),
            "sha256": sha256_file(FE / "single_5plus1/VALID_RESULTS_FINAL.csv",
                                  res["sha_cache"])},
        "single_5plus1/TEST_RESULTS_FINAL.csv": {
            "path": str(FE / "single_5plus1/TEST_RESULTS_FINAL.csv"),
            "sha256": sha256_file(FE / "single_5plus1/TEST_RESULTS_FINAL.csv",
                                  res["sha_cache"])},
        "fusion_audit.py": {"path": str(FUSION_AUDIT),
                            "sha256": sha256_file(FUSION_AUDIT, res["sha_cache"])},
        "phase5_fusion.py": {"path": str(PHASE5),
                             "sha256": sha256_file(PHASE5, res["sha_cache"])},
        "mf_candidate_nll_repaired.py": {
            "path": str(REPAIRED), "sha256": sha256_file(REPAIRED, res["sha_cache"])},
        "build_codebook_sensitivity.py": {
            "path": str(DRIVERS / "build_codebook_sensitivity.py"),
            "sha256": sha256_file(DRIVERS / "build_codebook_sensitivity.py",
                                  res["sha_cache"])},
    }
    for split in ("valid", "test"):
        pq = ART_ROOT / "data" / DATASET / f"{split}.parquet"
        inputs[f"data/Beauty/{split}.parquet"] = {
            "path": str(pq), "sha256": sha256_file(pq, res["sha_cache"])}
    for (cb, k), cell in sorted(cells.items()):
        if (cb, k, "valid") not in meta and (cb, k, "test") not in meta:
            continue
        inputs[f"ckpt_cb{cb}_k{k}"] = {"path": str(cell["ckpt_path"]),
                                       "sha256": cell["ckpt_sha256"]}
        inputs[f"sid_cb{cb}_k{k}"] = {"path": str(cell["sid_path"]),
                                      "sha256": cell["sid_sha256"]}
    for key, m in sorted(meta.items()):
        inputs[f"dump_cb{key[0]}_k{key[1]}_{key[2]}"] = {
            "path": m["dump"], "sha256": sha256_file(m["dump"], res["sha_cache"])}
    rqm = res["rq_meta"]
    for split in ("valid", "test"):
        inputs[f"dump_rqvae5_256_{split}"] = {
            "path": rqm[split]["dump"], "sha256": sha256_file(rqm[split]["dump"],
                                                              res["sha_cache"])}
    inputs["sid_rqvae5_256"] = {"path": str(rqm["valid"]["code_path"]),
                                "sha256": rqm["valid"]["sid_sha256"]}
    inputs["ckpt_rqvae5_256"] = {"path": str(rq_meta_ckpt()),
                                 "sha256": sha256_file(rq_meta_ckpt(),
                                                       res["sha_cache"])}
    gpu = gpu_compute_apps()
    return {
        "phase": "PHASE 12-16 Beauty-only codebook sensitivity",
        "generated": now_iso(),
        "driver": str(DRIVERS / "build_codebook_sensitivity.py"),
        "driver_sha256": sha256_file(DRIVERS / "build_codebook_sensitivity.py",
                                     res["sha_cache"]),
        "python": sys.version,
        "output_dir": str(run_dir),
        "diverted_from_canonical": bool(diverted),
        "dataset_scope": "Beauty only (CODEBOOK_BLOCKER.md: the Spectral codebook "
                         "study exists only for Beauty)",
        "matched_codebooks": MATCHED_CODEBOOKS,
        "matched_set_stop_rule": read_json(SPEC / "MATCHED_CODEBOOK_SET.json")[
            "stop_rule_reason"],
        "selection_statement": {
            "codebook_selected_on_test": False,
            "codebook_selected_at_all": False,
            "statement": "No codebook was selected on TEST, and no codebook was "
                         "selected at all. The codebook axis is reported as "
                         "descriptive sensitivity, never as 'the optimal codebook is C'. "
                         "The only VALID-side selections are the fusion pool member K "
                         "(best_single) and the dynamic-fusion (gamma, eta) grid.",
            "valid_selection_metric": "item-level VALID NDCG@20",
        },
        "vocabulary_rule": {
            "rule": "stride = max(semantic_codebook, disambiguation_alphabet); "
                    "vocab = stride*L + 1",
            "holds_for_C_ge_256": True,
            "fails_for_C_in": [32, 64, 128],
            "measured_per_cell": "CELL_frozen vocabulary audit.csv (embedded vocab read from every "
                                 "frozen dump, stride cross-checked against the SID array)",
            "example": "C=32,K=2: 32 semantic codes, 453 disambiguation codes, "
                       "embedded vocab 1360 (not 97)",
        },
        "item_id_convention": {
            "historical_codebook_dumps": "raw_0based (canonical = raw + 1)",
            "rqvae5_5plus1_dumps": "canonical_1based (already +1)",
            "method": "measured per dump by exact candidate-token reconstruction "
                      "(must be 100% on valid candidates, else HARD FAIL)",
            "consequence": "the matched Spectral(0-based) + RQ-VAE5(1-based) fusion "
                           "matches items correctly only because each side's convention "
                           "is measured; applying +1 to both would match "
                           f"{res['matched']['test']['cross_family_pairs_naive_plus1']} "
                           "cross-family pairs instead of "
                           f"{res['matched']['test']['cross_family_pairs_correct_convention']}",
        },
        "target_and_user_order_check": res["order_checks"],
        "audited_union_equality_check": res["union_checks"],
        "row_alignment_audit": res["cell_audit"],
        "frozen_method4_reproducibility": res["frozen_checks"],
        "crosscheck_discrepancies": [
            {"kind": "FROZEN_METHOD4_JSON_STALE",
             "codebook": c["codebook"], "status": c["status"],
             "max_abs_delta_over_12_cells": c["max_abs_delta_over_12_cells"],
             "inputs_newer_than_json": c["inputs_newer_than_json"],
             "consequence": "the frozen Method-4 codebook-fusion numbers for this "
                            "codebook are NOT reproducible from the artifacts on "
                            "disk and are excluded from the usable data; they were "
                            "not used to compute any row of this study",
             "mechanism": "run_beauty_codebook_method4.sh skips the fusion step when "
                          "the JSON already exists, so a re-run of the cell (new "
                          "checkpoint + new beam dumps) leaves a stale fusion JSON"}
            for c in res["frozen_checks"] if c["status"] != "REPRODUCED"],
        "k5_metrics_crosscheck": {
            "source": str(SPEC / "SPECTRAL_K5_CODEBOOK_METRICS.csv"),
            "max_abs_delta_vs_recomputed": float(res["k5_max_delta"]),
            "tolerance": 1e-12,
            "verdict": "PASS",
        },
        "rqvae5_single_crosscheck": {
            "max_abs_delta_vs_single_5plus1": float(max(x["abs_delta"]
                                                        for x in res["rq_cross"])),
            "tolerance": 1e-12, "verdict": "PASS"},
        "gpu_ownership_now": gpu,
        "read_only_root_incident": {
            "what": "the first invocation of this driver (before the "
                    "`sys.dont_write_bytecode = True` guard) caused CPython to write a "
                    "bytecode cache file inside the read-only historical root",
            "path": "a bytecode cache file inside the read-only historical root",
                        "cause": "`import phase5_fusion` (the required reuse of the historical "
                     "gamma/eta + paired_bootstrap helpers) writes __pycache__ next to "
                     "the source unless bytecode writing is disabled",
            "action_taken": "the .pyc was NOT deleted (deleting is equally forbidden in "
                            "a read-only root); the guard was added so no further write "
                            "can happen; no other file under ART_ROOT / upstream TIGER was "
                            "created or modified (`find -newermt` over both roots "
                            "returns only this one .pyc)",
            "operator_decision_needed": "whether to remove that single .pyc",
        },
        "blocked_manifest": res["blocked"],
        "cfla_status": {
            "stage": "cfla-gpu",
            "implemented": True,
            "run": bool(res["cfla_measured"]),
            "definition": "score(u,i)=logsumexp_F(-(NLL_F(u,i)-mu_F)/sigma_F), "
                          "(mu_F,sigma_F) fitted on ALL VALID candidates of that "
                          "(codebook, factorization); TEST never used to fit or select",
            "exact_command": CFLA_CMD,
            "blocked_because": "the GPU is owned by a running training job "
                               "(GPU RULE); the stage refuses to contend and exits "
                               "with a clear message",
            "gpu_owners_at_write_time": gpu["procs"],
            "work_estimate_from_measured_union_sizes": {
                f"cb{cb}": {
                    "union_rows": f["n_union_rows"],
                    "teacher_forced_sequences_4_factors":
                        {s: 4 * n for s, n in f["n_union_rows"].items()},
                } for cb, f in sorted(res["fusion"].items())
            },
            "note_on_estimate": "row counts are MEASURED here; the GPU stage scores "
                                "4 factorizations per union row, so the sequence count "
                                "is 4x. No FLOP figure is invented.",
        },
        "inputs_sha256": inputs,
    }


def _report(res, run_dir, meta_obj):
    L = []
    L.append("# Beauty codebook sensitivity -- PHASE 12-16\n")
    L.append(f"generated {now_iso()} by "
             f"`drivers/build_codebook_sensitivity.py`\n")
    L.append("## Status\n")
    L.append("* (a) RUNNABLE, done here: Spectral-K5(C) single-model sensitivity and "
             "the K2-K5 pool fusion (uniform / best-single / dynamic) for "
             "C in {32,64,128,256,512}; the matched `Spectral-K5(256)+RQ-VAE5(256)` "
             "pair (both sides exist).\n"
             "* (b) BLOCKED: `RQ-VAE5(C)` for C != 256 -- never trained; explicit "
             "blocked rows name the missing files and the exact command.\n"
             "* (c) IMPLEMENTED, GPU-PENDING: CFLA across codebooks (needs "
             "teacher-forced cross-factorization NLL).\n")
    L.append("**No codebook was selected on TEST, and no codebook was selected at "
             "all**: the codebook axis is descriptive sensitivity.\n")
    L.append("## Measured Beauty numbers (TEST, item-level)\n")
    L.append("| C | spectral K5 N@20 | uniform K2-K5 N@20 | best single K | best N@20 | "
             "dynamic (g,e) N@20 |\n|---|---|---|---|---|---|\n")
    for cb, f in sorted(res["fusion"].items()):
        s = res["singles"][(cb, 5, "test")]
        L.append(f"| {cb} | {s['NDCG@20']:.6f} | "
                 f"{float(f['uniform']['test']['NDCG@20'].mean()):.6f} | "
                 f"K{f['k_star']} | {float(f['best_single']['test']['NDCG@20'].mean()):.6f} | "
                 f"({f['gamma']},{f['eta']}) "
                 f"{float(f['dynamic']['test']['NDCG@20'].mean()):.6f} |\n")
    if res["matched"]:
        m = res["matched"]["test"]
        L.append(f"\nMatched C=256 pair (Spectral-K5 + RQ-VAE5, both frozen): "
                 f"uniform TEST N@20 {float(m['uniform']['NDCG@20'].mean()):.6f}, "
                 f"best single {m['k_star']} "
                 f"{float(m['best_single']['NDCG@20'].mean()):.6f}, "
                 f"mean union size {m['mean_union_size']:.2f}.\n")
    L.append("\n## Provenance checks that passed\n")
    L.append(f"* dump evaluator vs `SPECTRAL_K5_CODEBOOK_METRICS.csv`: max |delta| = "
             f"{res['k5_max_delta']:.2e}\n")
    L.append(f"* RQ-VAE5 C=256 vs `single_5plus1/*_RESULTS_FINAL.csv`: max |delta| = "
             f"{max(x['abs_delta'] for x in res['rq_cross']):.2e}\n")
    L.append("* `build_union_generic` == `fusion_audit.build_union` element-wise at "
             f"C=32 and C=256\n")
    repro = [c for c in res["frozen_checks"] if c["status"] == "REPRODUCED"]
    stale = [c for c in res["frozen_checks"] if c["status"] != "REPRODUCED"]
    L.append(f"* the frozen Method-4 fusion JSONs reproduce bit-exactly at their own "
             f"(gamma, eta) for C in {[c['codebook'] for c in repro]}\n")
    if stale:
        L.append(f"* **NOT reproducible** for C in {[c['codebook'] for c in stale]} "
                 "-- their beam dumps were replaced AFTER the JSON was written "
                 "(checkpoint + dump mtimes prove it), so those frozen fusion numbers "
                 "are STALE and are excluded (status "
                 "`FROZEN_JSON_STALE_DUMP_EXCLUDED`, see "
                 "FROZEN_METHOD4_REPRODUCIBILITY.csv). They are not used anywhere in "
                 "this study.\n")
    L.append("* repaired row-alignment write-back + per-row ownership assertion on a "
             "real union row space\n")
    L.append("\n## Blocked / pending\n")
    L.append(f"* `RQ-VAE5(C)` for C in "
             f"{[b['codebook'] for b in res['blocked']['rqvae5_missing_codebooks']]}: "
             "missing\n")
    for b in res["blocked"]["rqvae5_missing_codebooks"]:
        L.append(f"  * C={b['codebook']}: `{b['missing_t5_ckpt']}` -> "
                 f"`{b['unblock_command']}`\n")
    L.append(f"* CFLA GPU stage: `{CFLA_CMD}`\n")
    L.append("\n## Caveat carried from the recovery round\n")
    L.append("Training logs are NOT a usable metric source (two appended runs per log; "
             "test logged only on improving epochs). Every metric here comes from the "
             "frozen beam dumps.\n")
    return "".join(L)


# ===========================================================================
# self-test: the row-alignment unit test (non-involution)
# ===========================================================================
def self_test():
    rep = import_repaired()
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
              (f": {detail}" if detail else ""))

    print("=" * 74)
    print("test_codebook_row_alignment (non-involution fixture)")
    print("=" * 74)

    # --- fixture: a permutation that is NOT an involution ---------------------
    n = 39
    order = None
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        cand = rng.permutation(n)
        if not np.array_equal(cand[cand], np.arange(n)):
            order = cand
            break
    if order is None:
        raise SystemExit("FAIL: could not build a non-involution fixture")
    chk("fixture_is_not_an_involution", not np.array_equal(order[order], np.arange(n)),
        f"order != order^-1 on {n} rows (seed found deterministically)")
    rep.assert_permutation(order, n)
    chk("order_is_a_valid_permutation", True, f"{n} rows")

    # --- independent per-row truth: a pure function of (user, candidate) ------
    users = np.repeat(np.arange(13, dtype=np.int64), 3)
    cands = np.array([(u * 37 + j * 11) % 97 for u, j in zip(users, np.tile(
        np.arange(3), 13))], dtype=np.int64)

    def truth(u, c):
        return np.sin(u * 0.7) + 0.5 * np.cos(c * 1.3) + u * c * 1e-3

    order = rows_lexsort(users, cands)
    chk("real_row_space_order_is_not_an_involution",
        not np.array_equal(order[order], np.arange(len(order))),
        f"{len(order)} rows, order != order^-1")
    scores_sorted = truth(users[order], cands[order])
    out = scatter_scores(scores_sorted, order)
    expected = truth(users, cands)
    worst = float(np.abs(out - expected).max())
    chk("write_back_maps_every_row_to_its_own_pair", worst < 1e-12,
        f"max|out-expected| = {worst:.1e}")
    buggy = scores_sorted[order]
    bug_delta = float(np.abs(buggy - expected).max())
    chk("historical_buggy_form_is_rejected", bug_delta > 1e-3,
        f"scores_sorted[order] differs from truth by max {bug_delta:.3f}")
    chk("write_back_is_the_inverse_of_the_sort",
        np.array_equal(out[order], scores_sorted), "out[order] == scores_sorted")

    # --- the production assertion must CATCH the bug (negative control) -------
    try:
        assert_rows_own_their_values(users, cands, buggy,
                                     lambda idx: truth(users[idx], cands[idx]),
                                     n_sample=len(users), label="negative-control")
        caught = False
    except SystemExit:
        caught = True
    chk("permuted_write_back_is_caught_by_the_assertion", caught,
        "assert_rows_own_their_values raised on the buggy scatter")

    # --- and must PASS on the repaired scatter --------------------------------
    audit = assert_rows_own_their_values(users, cands, out,
                                        lambda idx: truth(users[idx], cands[idx]),
                                        n_sample=len(users), label="positive-control")
    chk("repaired_write_back_passes_the_assertion", audit["max_abs_delta"] < 1e-12,
        f"max|delta| = {audit['max_abs_delta']:.1e}")

    # --- permutation guard rejects a non-permutation --------------------------
    try:
        rep.assert_permutation(np.array([0, 1, 1, 3]), 4)
        rejected = False
    except AssertionError:
        rejected = True
    chk("non_permutation_index_is_rejected", rejected, "duplicate index refused")

    # --- realistic block: lexsort round trip on a union-shaped row space ------
    rng = np.random.default_rng(RNG_SEED)
    nu, per = 250, 50
    u2 = np.repeat(np.arange(nu, dtype=np.int64), per)
    c2 = np.array([(int(u) * 101 + j * 7) % 4001 for u, j in
                   zip(u2, np.tile(np.arange(per), nu))], dtype=np.int64)
    o2 = rows_lexsort(u2, c2)
    s2 = truth(u2[o2] % 13, c2[o2] % 97)
    back2 = scatter_scores(s2, o2)
    exp2 = truth(u2 % 13, c2 % 97)
    chk("lexsort_round_trip_on_a_union_shaped_block",
        np.array_equal(back2, exp2) and not np.array_equal(o2[o2], np.arange(len(o2))),
        f"{len(o2)} rows, {nu} users, exact inversion")

    # --- the PRODUCTION plumbing on the REAL cb32 Beauty union ----------------
    # Same sequence the GPU stage runs (union -> row space -> lexsort -> per-factor
    # candidate tokens -> scatter -> ownership assertion); only the T5 forward pass
    # is replaced by a deterministic row-local CPU pseudo-scorer, so this validates
    # everything about the CFLA stage that does not need the GPU.
    try:
        cells = beauty_cells(load_inventory())
        cb = 32
        dumps, convs, sids = [], [], []
        for k in KS:
            p = dump_path(cells[(cb, k)], "valid")
            m = measure_dump(p, cells[(cb, k)])
            dumps.append(load_dump(p))
            convs.append(m["item_id_convention"])
            sids.append(np.load(cells[(cb, k)]["sid_path"],
                                allow_pickle=True).astype(np.int64))
        items_pad, _st, _tg, _tp, valid_mask = build_union_generic(dumps, convs)
        users, cands, slots = union_row_space(items_pad)
        order = rows_lexsort(users, cands)

        def pseudo(idx, _tok):
            t = _tok[idx].astype(np.float64)
            return (t * np.arange(1, t.shape[1] + 1)).sum(axis=1) + (users[idx] % 7) * 0.5

        for m, k in enumerate(KS):
            tok = sids[m][cands - 1]

            def scorer(idx, _tok=tok):
                return pseudo(idx, _tok)

            audit = assert_rows_own_their_values(
                users, cands, scatter_scores(scorer(order), order), scorer,
                n_sample=256, label=f"self-test cb{cb} K{k} real union valid")
            chk(f"real_union_write_back_cb{cb}_K{k}", audit["max_abs_delta"] == 0.0,
                f"{audit['n_rows']} rows, {audit['n_sampled']} sampled, "
                f"max|delta| = {audit['max_abs_delta']:.1e}")
            try:
                assert_rows_own_their_values(users, cands, scorer(order)[order], scorer,
                                             n_sample=256, label="neg")
                caught_real = False
            except SystemExit:
                caught_real = True
            chk(f"real_union_permuted_write_back_caught_cb{cb}_K{k}", caught_real,
                "buggy scatter rejected on the real row space")
    except SystemExit as exc:
        chk("real_union_plumbing", False, f"could not run: {exc}")

    bad = [c for c in checks if not c[1]]
    print("-" * 74)
    print(f"  {'PASS' if not bad else 'FAIL'}  {len(checks) - len(bad)}/{len(checks)} checks")
    if bad:
        raise SystemExit(1)
    return {"checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks]}


# ===========================================================================
# dry-run: validate everything, compute nothing, write nothing
# ===========================================================================
def dry_run(args):
    print("=" * 74)
    print("DRY RUN -- CPU-only validation, no GPU, no writes, no metrics")
    print("=" * 74)
    inv, cells, meta, sha_cache, _ = validate_all(
        verify_hashes=not args.skip_hashes, verify_ckpt=not args.skip_ckpt_check,
        verbose=True)
    print(f"\n[1] inventory: {len(inv)} rows read from "
          f"{SPEC / 'SPECTRAL_CODEBOOK_INVENTORY.csv'}")
    print(f"    Beauty cells: {len(cells)} (6 codebooks x K2-K5)")
    n_cellkeys = {(cb, k) for (cb, k, _s) in meta}
    n_both = sum(1 for cb, k in n_cellkeys
                 if (cb, k, "valid") in meta and (cb, k, "test") in meta)
    print(f"    dumps measured: {len(meta)} over {len(n_cellkeys)} of 24 cells "
          f"({n_both} with both splits; the cb1024/K5 cell has none)")
    print("[2] per-cell measured vocab/stride (the C*L+1 trap):")
    for (cb, k, split), m in sorted(meta.items()):
        if split != "valid":
            continue
        print(f"    cb{cb:<5d} K{k} L={m['sid_len']} vocab={m['vocab_size_embedded']:<5d} "
              f"stride={m['stride_embedded']:<5d} (C*L+1={m['vocab_formula_C_times_L_plus_1']}"
              f" holds={m['vocab_formula_holds']}) conv={m['item_id_convention']}")
    print("[3] references to read (not recomputed):")
    for p in (SPEC / "MATCHED_CODEBOOK_SET.json", SPEC / "CODEBOOK_BLOCKER.md",
              SPEC / "SPECTRAL_K5_CODEBOOK_METRICS.csv",
              SPEC / "SPECTRAL_K5_CODEBOOK_CROSSCHECK.csv", FUSION_AUDIT, PHASE5,
              REPAIRED, FE / "single_5plus1/VALID_RESULTS_FINAL.csv",
              FE / "single_5plus1/TEST_RESULTS_FINAL.csv"):
        print(f"    {'OK ' if Path(p).exists() else 'MISSING'} {p}")
    print("[4] what the CPU stage would compute:")
    n_single = sum(1 for key in meta if key[2] in ("valid", "test"))
    print(f"    - single-model metrics for {n_single} (codebook, K, split) cells from "
          "frozen dumps")
    pool_cbs = [cb for cb in MATCHED_CODEBOOKS
                if all((cb, k, s) in meta for k in KS for s in ("valid", "test"))]
    print("    - K2-K5 pool fusion (uniform / best-single-on-VALID / dynamic "
          f"gamma,eta-on-VALID) for {pool_cbs}")
    print("    - matched Spectral-K5+RQ-VAE5 fusion at C=256 (both sides exist)")
    blocked = _blocked_manifest(cells, {}, meta)["rqvae5_missing_codebooks"]
    print(f"    - {len(blocked)} blocked RQ-VAE5(C) codebooks "
          f"{[b['codebook'] for b in blocked]} with explicit missing paths")
    print("    - paired user-level bootstrap (5000 resamples, seed 20260912)")
    print("[5] GPU stage that will NOT run now:")
    st = gpu_compute_apps()
    if st["available"]:
        for p in st["procs"]:
            print(f"    device owner: pid {p['pid']} {p['process_name']} "
                  f"({p['used_memory']})")
        if st["procs"]:
            print("    -> `--stage cfla-gpu` would REFUSE (GPU RULE)")
    else:
        print(f"    ownership unknown: {st['reason']}")
    print(f"    CFLA across codebooks command: {CFLA_CMD}")
    print("[6] outputs the CPU stage would write (never overwriting an existing file):")
    run_dir, diverted = resolve_run_dir(args.run_dir, create=False)
    for f in REQUIRED_OUTPUTS + EXTRA_OUTPUTS:
        print(f"    {run_dir / f}")
    if diverted:
        print("    (canonical outputs already exist -> fresh runs/ directory)")
    print("\nDRY RUN OK -- every input path validated, no metric computed, nothing written.")
    return {"cells": len(cells), "dumps": len(meta)}


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="PHASE 12-16 Beauty-only codebook sensitivity driver")
    ap.add_argument("--stage", choices=["cpu", "cfla-gpu"], default="cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate all CPU-side inputs, print the plan, write nothing, "
                         "never touch the GPU")
    ap.add_argument("--self-test", action="store_true",
                    help="run the row-alignment unit test (non-involution)")
    ap.add_argument("--run-dir", default=None,
                    help="explicit output directory (must not already contain outputs)")
    ap.add_argument("--skip-hashes", action="store_true",
                    help="skip sha256 verification against the inventory (not advised)")
    ap.add_argument("--skip-ckpt-check", action="store_true",
                    help="skip the CPU checkpoint-vocab validation in --dry-run")
    ap.add_argument("--codebooks", type=int, nargs="*", default=None,
                    help="cfla-gpu: subset of codebooks")
    ap.add_argument("--splits", nargs="*", default=["valid", "test"],
                    choices=["valid", "test"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit-users", type=int, default=0,
                    help="cfla-gpu: only the first N users (smoke run)")
    ap.add_argument("--check-rows", type=int, default=64,
                    help="cfla-gpu: rows re-scored independently by the ownership check")
    ap.add_argument("--allow-pid", type=int, nargs="*", default=None,
                    help="cfla-gpu: pids allowed to hold the device")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.dry_run:
        dry_run(args)
        return 0
    if args.stage == "cfla-gpu":
        stage_cfla_gpu(args)
        return 0
    res = cpu_stage(args)
    run_dir = write_outputs(res, args)
    print(f"[cpu] canonical deliverables: "
          f"{'yes' if run_dir == OUT else f'no (diverted to {run_dir})'}")
    bad = [c for c in res["frozen_checks"] if c["status"] != "REPRODUCED"]
    if bad:
        print("=" * 74)
        print("EXIT 2 = all results were written successfully; this is a REPORTED "
              "DISCREPANCY, not a driver failure.")
        for c in bad:
            print(f"  FROZEN_METHOD4_JSON_STALE cb{c['codebook']}: "
                  f"max|delta|={c['max_abs_delta_over_12_cells']:.3e}, "
                  f"{len(c['inputs_newer_than_json'])} dump(s) newer than the JSON")
        print(f"  details: {run_dir / 'FROZEN_METHOD4_REPRODUCIBILITY.csv'} and "
              f"{run_dir / 'RUN_METADATA.json'} -> crosscheck_discrepancies")
        print("=" * 74)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
