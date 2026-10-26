#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_m5_rq5.py -- PHASE 8-10: the NEW M5 (five-factorization multi-granularity
fusion), its VALID/TEST ablation, and the paired user-level bootstrap.

WHAT THIS DRIVER IS
-------------------
The "new M5" fuses FIVE factorizations of the same item universe:

    spectral_k2  (SID length 3 = 2 semantic + 1 disambiguation)  frozen historical dump
    spectral_k3  (SID length 4 = 3 semantic + 1 disambiguation)  frozen historical dump
    spectral_k4  (SID length 5 = 4 semantic + 1 disambiguation)  frozen historical dump
    spectral_k5  (SID length 6 = 5 semantic + 1 disambiguation)  frozen historical dump
    rqvae5       (SID length 6 = 5 semantic + 1 disambiguation)  NEW this round

This is the historical M5 with the RQ-VAE3 arm replaced by the newly trained
RQ-VAE5 arm.  The arms keep their **NATIVE** SID lengths -- spectral K2 = 3,
K3 = 4, K4 = 5, K5 = 6, RQ-VAE5 = 6 -- so the M5 pool is a
`native_length_pool` with MIXED token budgets.  Only the K5 and RQ-VAE5 arms are
L = 6.

**The M5 is NOT a length-matched construction and must never be described as
one.**  The campaign's length-matched fairness claim belongs to the single-model
table (4 families x 4 datasets at L = 6, `single_5plus1/`), where every compared
model spends exactly 6 tokens per item.  Comparing this M5 against a single L = 6
model confounds two effects at once: (a) five models instead of one, and
(b) a different per-item token budget.  Every row this driver writes therefore
carries `pool`, `arm_sid_lengths`, `mixed_token_budgets` and
`length_matched_L6 = False`; the per-arm lengths are read from the dump
(`sid_len` + `candidate_tokens.shape[2]`) and asserted, never assumed.

A second, clearly separated `spectral4` scope runs the SAME machinery on the four
frozen spectral arms only.  That scope is the historical ablation verbatim, so it
doubles as the reproduction check of this driver against the frozen
`frozen_audit/PHASE5_FUSION.json`.

It is a pure CPU consumer of frozen artifacts.  It never trains, never loads a
checkpoint, never creates a CUDA context, and never writes outside
`outputs/final_hyperparam_efficiency/`.

CONTRACTS READ BEFORE WRITING THIS FILE (not re-derived here)
-------------------------------------------------------------
* `outputs/final_hyperparam_efficiency/HARNESS_HANDOFF.md` sections 3, 4, 5
  (invariants: read-only historical roots; selection on item-level VALID NDCG@20;
  0-based dumps canonicalised +1; token layout `code + p*stride + 1`).
* `outputs/final_hyperparam_efficiency/FINAL_HYPERPARAM_EFFICIENCY_REPORT.md`
  (the `m5_rq5/{VALID,TEST,BOOTSTRAP}.csv` deliverable contract).
* `the frozen artifact root/fusion_audit.py` -- the historical fusion library.
  `load_split`, `build_union`, `structure_q`, `zscore`, `confidence_matrix`,
  `weights_from`, `per_user_ndcg`, `summarize`, `canonicalize_item_ids` are
  REUSED, not re-implemented.  The only thing this driver does to that module is
  set its module-global `N_K`, because `build_union`/`confidence_matrix` read it
  to size the factorization axis; 4 is kept for the `spectral4` scope and 5 is
  set for the `m5` scope.  That override is recorded in RUN_METADATA.json.
* `src/legacy_audited/phase5_fusion.py` -- the
  historical fusion driver template.  `GAMMA_GRID`, `ETA_GRID`, `N_BOOT`,
  `RNG_SEED`, `METRICS`, `uniform_weights`, `single_k_weights`, `select_on_valid`
  and `paired_bootstrap` are imported from it verbatim.
* `outputs/multifactor_nll_repair/BUG_FIX_AUDIT.md` -- the row-alignment repair.
  Any per-row score column this driver writes goes through
  `write_scores_back_to_original_rows` + `assert_permutation` and is additionally
  verified by a `(user, candidate)` identity check.  Those two helpers are loaded
  from `experiments/multifactor_fusion/mf_candidate_nll_repaired.py` *by source
  extraction* (sha256 pinned to the value published in the audit), so the
  certified text is reused without executing that module's import-time side
  effects inside trees this round may not write to.

SELECTION RULE (hard requirement)
---------------------------------
Every selection -- the best single arm, the confidence-only eta, the (gamma, eta)
grid point -- is made on the **VALID** split using **item-level NDCG@20**.  TEST is
never used for any selection.  The legacy token-prefix metric is never computed.
The historical `phase5_fusion.py` picked `best_single` on TEST; that is the one
historical behaviour this driver deliberately does NOT reproduce, and the
difference is called out in RUN_METADATA.json's reproduction check.

GPU RULE
--------
A T5 training job owns the device.  The driver refuses any GPU stage and exits 3
with a diagnosis when the device is busy (running `model/main.py` /
`train_t5_5plus1.py`, foreign compute apps, or `memory.used` over the threshold).
Nothing in the delivered pipeline needs the GPU: see CFLA below.

CFLA
----
CFLA = Cross-Factorization Likelihood Aggregation.  Zero trainable parameters.
`zNLL_F(u,i) = (NLL_F(u,i) - mu_F) / sigma_F` with `(mu_F, sigma_F)` fitted on
VALID only, `score(u,i) = logsumexp_F(-zNLL_F(u,i))`.  It is NOT inference-free:
it costs `M * U` teacher-forced candidate scorings.  The M5 CFLA needs the
candidate TotalNLL of every union candidate under the **RQ-VAE5** checkpoint,
which does not exist on disk; those rows are emitted with an explicit
`BLOCKED_NO_CANDIDATE_NLL` status plus the exact command that would produce the
input.  No number is estimated for them.
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
# from them would drop a `__pycache__/*.pyc` next to their sources, i.e. write into
# a tree this round may only read.  Refuse bytecode caching before anything is
# imported, so the guarantee is enforced by the interpreter and not by luck.
_sys.dont_write_bytecode = True

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ===========================================================================
# roots and constants
# ===========================================================================
REPO = Path(_REPO_ROOT + '')
ART_ROOT = Path(_ART_ROOT + '')                       # READ-ONLY historical
UPSTREAM_ROOT = Path(_UP_ROOT + '')     # READ-ONLY historical
OUT = rp.OUTPUT_ROOT / 'final_hyperparam_efficiency'       # only writable root
M5DIR = OUT / 'm5_rq5'
DRIVERS = Path(__file__).resolve().parent

HIST_PHASE5 = REPO / 'src' / 'legacy_audited' / 'phase5_fusion.py'
HIST_PHASE5_JSON = ART_ROOT / 'frozen_audit' / 'PHASE5_FUSION.json'
HIST_FUSION_AUDIT = REPO / 'src' / 'legacy_audited' / 'fusion_audit.py'
REPAIRED_NLL = REPO / 'src' / 'cfla' / 'multifactor_fusion' / 'mf_candidate_nll_repaired.py'
MF_MODELS = REPO / 'src' / 'cfla' / 'multifactor_fusion' / 'mf_models.py'

# sha256 published in outputs/multifactor_nll_repair/BUG_FIX_AUDIT.md
EXPECTED_SHA = {
    str(REPAIRED_NLL):
        '7480beb64750dbdbcc2507f655708160129207577ac3830d5fb7d27eb07ad21c',
}

CANONICAL_AFFINITY = 'nonneg_pearson_v3'
DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
SCOPES = ('m5', 'spectral4')
ARMS = {
    'm5': ('spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5', 'rqvae5'),
    'spectral4': ('spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5'),
}
ARM_LEN = {'spectral_k2': 3, 'spectral_k3': 4, 'spectral_k4': 5, 'spectral_k5': 6,
           'rqvae5': 6}
ARM_SEMANTIC = {'spectral_k2': 2, 'spectral_k3': 3, 'spectral_k4': 4,
                'spectral_k5': 5, 'rqvae5': 5}
STRATEGIES = ('best_single', 'uniform', 'structure_aware', 'confidence_only',
              'valid_grid_search', 'cfla')
METRIC_KEYS = ['Recall@5', 'Recall@10', 'Recall@20',
               'NDCG@5', 'NDCG@10', 'NDCG@20']

# status vocabulary -- every non-OK row names the artifact that is missing
ST_OK = 'OK'
ST_BLOCKED_NO_NLL = 'BLOCKED_NO_CANDIDATE_NLL'
ST_BLOCKED_NO_Q = 'BLOCKED_NO_STRUCTURE_Q_FOR_RQVAE5'
ST_PENDING = 'PENDING_MISSING_INPUT'
ST_NOT_IN_SCOPE = 'NOT_IN_SCOPE_FOR_THIS_DELIVERABLE'
# Scope decision, not a blocker: the RQ-VAE arm is excluded from structure-aware
# weighting by operator decision (2026-09-17).  Kept distinct from the BLOCKED_*
# statuses so that "we chose not to" can never be misread as "we could not".
ST_EXCLUDED_BY_DECISION = 'EXCLUDED_BY_OPERATOR_DECISION_RQVAE_NOT_STRUCTURE_AWARE'
ST_NA = 'NOT_APPLICABLE_NO_RQVAE5_COUNTERPART'

STRIDE = 256            # canonical 5+1 token stride (token = code + p*stride + 1)
# How well the token-reconstruction test must separate the two candidate item-id
# conventions before the driver will trust a dump (see verify_item_id_convention).
MIN_CONV_MATCH = 0.999   # the detected convention must match at least this well
MAX_OTHER_MATCH = 0.010  # the rejected convention must be near zero
PAIR_KEY_STRIDE = 10 ** 7   # key packing for (user, item) pair sets
DEFAULT_GPU_MEM_LIMIT_MIB = 1000.0
TRAINING_PATTERNS = ('model/main.py', 'train_t5_5plus1.py', 'train_t5_5plus1')

VALID_FIELDS = [
    'run_id', 'dataset', 'split', 'affinity', 'scope', 'strategy', 'status',
    'pool', 'arms', 'arm_sid_lengths', 'arm_semantic_lengths', 'n_arms',
    'mixed_token_budgets', 'length_matched_L6',
    'best_single_arm', 'best_single_selection_split',
    'selected_gamma', 'selected_eta', 'selection_split',
    'n_users', 'n_users_target_in_union', 'mean_union_size',
    'mean_candidates_per_arm',
] + METRIC_KEYS + ['missing_input', 'producing_command', 'notes']

BOOT_FIELDS = [
    'run_id', 'dataset', 'affinity', 'scope', 'comparison', 'strategy_a',
    'strategy_b', 'metric', 'a_value', 'b_value', 'delta', 'pct_change',
    'ci_lo', 'ci_hi', 'p_value', 'n_boot', 'seed', 'n_users',
    'ci95_excludes_zero', 'status', 'missing_input', 'notes',
]


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


# ===========================================================================
# the repaired row-alignment helpers, reused by SOURCE EXTRACTION
# ===========================================================================
def load_repaired_helpers(path: Path = REPAIRED_NLL):
    """Return the certified `assert_permutation` + `write_scores_back_to_original_rows`.

    The helpers come from the audited source text itself (parsed with `ast`; only
    the two function definitions are executed) -- not re-typed and not imported as
    a module, because importing `mf_candidate_nll_repaired` drags in `mf_common`
    and that module's own run-directory handling.  The source sha256 is checked
    against the value published in BUG_FIX_AUDIT.md, so a silent edit of the
    repair is a hard failure here.
    """
    digest = sha256_file(path)
    if digest is None:
        raise FileNotFoundError(f'missing repaired helper source: {path}')
    expected = EXPECTED_SHA.get(str(path))
    if expected is not None and digest != expected:
        raise AssertionError(
            f'{path}: sha256 {digest} != {expected} published in BUG_FIX_AUDIT.md')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wanted = {'assert_permutation', 'write_scores_back_to_original_rows'}
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    got = {n.name for n in nodes}
    if got != wanted:
        raise AssertionError(f'{path}: expected {sorted(wanted)}, found {sorted(got)}')
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['assert_permutation'], ns['write_scores_back_to_original_rows'], digest


assert_permutation, write_scores_back_to_original_rows, REPAIRED_SHA = \
    load_repaired_helpers()


def write_back_and_verify(scores_sorted, order, users, candidates, score_fn,
                          what='scores', verify_all=True, sample_seed=20260912):
    """Scatter lexsorted per-row scores back and PROVE every row got its own value.

    `order = np.lexsort((candidates, users))`, so `scores_sorted[k]` was computed
    for the pair `(users[order[k]], candidates[order[k]])`.  The write-back is
    `out[order] = scores_sorted` -- never `scores_sorted[order]`, which is the
    inverse map and coincides with the correct one only when `order` is an
    involution.

    Two checks run, and the second is the one that catches the historical defect:

    1. *scatter*: a dict keyed by the `(user, candidate)` PAIR is built from the
       sorted stream and every original row must hold the value that stream carried
       for the pair that row names.  This catches a wrong inverse map.
    2. *stream order*: `score_fn(users, candidates)` is applied DIRECTLY, in the
       original row order, to every row (or to a seeded sample when
       `verify_all=False`) and must reproduce `out` exactly.  This is the only
       non-circular check of the producer's ordering: check 1 alone cannot see a
       mis-ordered stream, because it takes that stream as ground truth.  Passing
       `scores_sorted[order]` -- the historical defect -- fails here on every row
       whose `order` is not an involution, which is the point.

    `score_fn` is therefore a REQUIRED argument: a per-row score column is never
    written on trust.
    """
    scores_sorted = np.asarray(scores_sorted, dtype=np.float64)
    order = assert_permutation(order, len(scores_sorted))
    users = np.asarray(users)
    candidates = np.asarray(candidates)
    if users.shape != candidates.shape or users.shape[0] != len(order):
        raise AssertionError(
            f'{what}: users/candidates shape {users.shape}/{candidates.shape} do '
            f'not match order length {len(order)}')
    if not np.isfinite(scores_sorted).all():
        raise AssertionError(f'{what}: non-finite value in the sorted score stream')
    if not callable(score_fn):
        raise AssertionError(f'{what}: score_fn is required -- a per-row score column '
                             f'is never written without an independent identity check')

    key_sorted = list(zip(users[order].tolist(), candidates[order].tolist()))
    if len(set(key_sorted)) != len(key_sorted):
        raise AssertionError(f'{what}: duplicate (user, candidate) pairs in the stream')
    truth = dict(zip(key_sorted, scores_sorted.tolist()))

    out = write_scores_back_to_original_rows(scores_sorted, order)
    for i, key in enumerate(zip(users.tolist(), candidates.tolist())):
        want = truth[key]
        if out[i] != want:
            raise AssertionError(
                f'{what}: row {i} holds {out[i]!r} but the stream carried {want!r} '
                f'for its own pair (user={key[0]}, candidate={key[1]}) -- the '
                f'write-back is not the inverse of the sort')

    n = len(order)
    if verify_all or n <= 20000:
        idx = np.arange(n)
        mode = 'all'
    else:
        idx = np.random.default_rng(sample_seed).choice(n, size=20000, replace=False)
        idx = np.sort(idx)
        mode = 'seeded-sample-20000'
    want_direct = np.asarray(score_fn(users[idx], candidates[idx]), dtype=np.float64)
    got = out[idx]
    bad = np.flatnonzero(got != want_direct)
    if bad.size:
        i = int(idx[bad[0]])
        raise AssertionError(
            f'{what}: {bad.size}/{len(idx)} verified rows hold a value that does not '
            f'belong to them; row {i} names (user={int(users[i])}, '
            f'candidate={int(candidates[i])}), was written {got[bad[0]]!r}, but the '
            f'scorer returns {want_direct[bad[0]]!r} for that pair -- the sorted '
            f'stream was not in `order` (scores_original[order] = scores_sorted was '
            f'required, not scores_sorted[order])')
    return out


# ===========================================================================
# GPU guard (read-only; nvidia-smi only, never a CUDA context)
# ===========================================================================
GPU_QUERY = 'index,name,driver_version,memory.used,memory.total,utilization.gpu'


def scan_host_processes():
    """Live processes whose cmdline looks like a training job (read-only /proc)."""
    me = os.getpid()
    found = []
    proc = Path('/proc')
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            raw = (entry / 'cmdline').read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        except OSError:
            continue
        cmd = raw.replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
        if not cmd:
            continue
        for pat in TRAINING_PATTERNS:
            if pat in cmd:
                found.append({'pid': pid, 'pattern': pat, 'cmdline': cmd[:400]})
                break
    return found


def probe_gpu():
    """Read GPU state with nvidia-smi only -- never creates a CUDA context."""
    info = {'nvidia_smi_ok': False, 'error': None, 'gpus': [], 'compute_apps': []}
    try:
        r = subprocess.run(
            ['nvidia-smi', f'--query-gpu={GPU_QUERY}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        info['error'] = f'{type(e).__name__}: {e}'
        return info
    if r.returncode != 0:
        info['error'] = f'nvidia-smi exit {r.returncode}: {r.stderr.strip()[:300]}'
        return info
    info['nvidia_smi_ok'] = True
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        idx, name, drv, mused, mtot, util = parts[:6]
        info['gpus'].append({'index': int(idx), 'name': name, 'driver_version': drv,
                             'memory_used_MiB': float(mused),
                             'memory_total_MiB': float(mtot),
                             'util_pct': float(util)})
    try:
        r2 = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
        if r2.returncode == 0:
            for line in r2.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3 and parts[0].isdigit():
                    info['compute_apps'].append(
                        {'pid': int(parts[0]), 'process_name': parts[1],
                         'used_memory_MiB': float(parts[2])})
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        info['compute_apps_error'] = f'{type(e).__name__}: {e}'
    return info


def assess_contention(gpu_info, mem_limit_mib, host_procs):
    """Return (busy, reasons[]).  Unknown == busy."""
    reasons = []
    if not gpu_info.get('nvidia_smi_ok'):
        reasons.append('nvidia-smi unavailable or failed ({}), so device idleness '
                       'cannot be established'.format(gpu_info.get('error')))
        return True, reasons
    if not gpu_info['gpus']:
        reasons.append('nvidia-smi reported no GPUs')
    for g in gpu_info['gpus']:
        if g['memory_used_MiB'] >= mem_limit_mib:
            reasons.append(f"GPU {g['index']} ({g['name']}): memory.used="
                           f"{g['memory_used_MiB']:.0f} MiB >= limit "
                           f"{mem_limit_mib:.0f} MiB")
    for a in gpu_info.get('compute_apps', []):
        reasons.append(f"foreign compute app pid={a['pid']} ({a['process_name']}) "
                       f"holds {a['used_memory_MiB']:.0f} MiB on the device")
    for p in host_procs:
        reasons.append(f"running training process pid={p['pid']} matched "
                       f"'{p['pattern']}': {p['cmdline']}")
    return (len(reasons) > 0), reasons


def report_gpu_state(gpu_info, busy, reasons, mem_limit_mib):
    print('GPU GUARD (read-only; nvidia-smi only, no CUDA context created)')
    if not gpu_info.get('nvidia_smi_ok'):
        print(f"  nvidia-smi unavailable: {gpu_info.get('error')}")
    for g in gpu_info.get('gpus', []):
        print(f"  GPU {g['index']} {g['name']} driver {g['driver_version']} "
              f"memory.used={g['memory_used_MiB']:.0f}/{g['memory_total_MiB']:.0f} MiB "
              f"util={g['util_pct']:.0f}%")
    print(f'  busy={busy}  (memory.used limit {mem_limit_mib:.0f} MiB)')
    for r in reasons:
        print(f'   * {r}')


def refuse_gpu(reasons, mem_limit_mib):
    print()
    print('=' * 78)
    print('REFUSING TO RUN A GPU STAGE: the device is not exclusively idle.')
    print('=' * 78)
    print('A T5 training job owns the GPU.  This driver will not contend for it.')
    print()
    for r in reasons:
        print(f'  * {r}')
    print()
    print(f'Thresholds: --gpu-mem-limit-mib {mem_limit_mib:.0f}')
    print()
    print('The delivered pipeline of this driver is CPU-only and does NOT need the')
    print('device; only the (currently blocked) candidate-NLL / CFLA stage would.')
    print('Re-run with --allow-busy-gpu only if you intend to take GPU numbers')
    print('under contention; that fact is then recorded in RUN_METADATA.json.')
    print('=' * 78)
    sys.exit(3)


# ===========================================================================
# input registry
# ===========================================================================
def spectral_dump_path(dataset, affinity, split, k):
    return (ART_ROOT / 'model' / f'beam_dumps_{dataset}_{affinity}' /
            f'{split}_k{k}_sid{k+1}.npz')


def rqvae5_dump_path(dataset, split):
    return OUT / 'runs_5plus1' / dataset / 'rqvae5' / 'beams' / f'{split}_k5_sid6.npz'


def rqvae5_sid_path(dataset):
    return OUT / 'runs_5plus1' / dataset / 'rqvae5' / 'sid' / f'{dataset}_t5_rqvae5_base.npy'


def rqvae5_ckpt_path(dataset):
    return OUT / 'runs_5plus1' / dataset / 'rqvae5' / 't5' / 'best_model.pth'


def rqvae5_nll_path(dataset, split):
    """The candidate-NLL file CFLA would need.  Verified absent, never assumed."""
    return OUT / 'm5_rq5' / 'candidate_nll' / dataset / f'{split}_m5_candidate_nll.parquet'


def rqvae5_dim_groups_path(dataset):
    """The artifact `structure_aware` / `valid_grid_search` would need for RQVAE5."""
    return (OUT / 'runs_5plus1' / dataset / 'rqvae5' / 'sid' /
            f'{dataset}_t5_rqvae5_base_dim_groups.npy')


def structure_json_path(dataset, affinity=CANONICAL_AFFINITY):
    return (ART_ROOT / 'model' / f'beam_dumps_{dataset}_{affinity}' /
            f'structure_weights_{dataset}_{affinity}.json')


def cfla_producing_command(datasets):
    """The EXACT command that would produce the missing M5 candidate NLL.

    It is not runnable today, and the driver says so instead of pretending:
    `experiments/multifactor_fusion/mf_candidate_nll_repaired.py` iterates over
    `mf_common.FACTORIZATIONS`, whose fifth entry is `rqvae3` (the historical
    RQ-VAE3 arm).  There is no `rqvae5` entry, and `mf_common.py` lives in a
    read-only root, so a thin registering driver must exist first.
    """
    py = REPO / '.venv_cuda_compatible' / 'bin' / 'python'
    return (f"cd {REPO} && {py} {DRIVERS}/score_m5_candidate_nll.py "
            f"--datasets {','.join(datasets)} --splits valid,test "
            f"--factorizations spectral_k2,spectral_k3,spectral_k4,spectral_k5,rqvae5 "
            f"--device cuda   "
            f"[helper does not exist yet: it must register the rqvae5 arm (SID "
            f"{rqvae5_sid_path('<ds>')}, checkpoint {rqvae5_ckpt_path('<ds>')}, beams "
            f"{rqvae5_dump_path('<ds>', '<split>')}) and call "
            f"mf_models.candidate_total_nll, the certified scorer in {MF_MODELS}. "
            f"It NEEDS THE GPU and must not run while training owns the device.]")


# ===========================================================================
# input validation
# ===========================================================================
class Report:
    """Collects validation findings without swallowing anything."""

    def __init__(self):
        self.rows = []
        self.fatal = []
        self.files = []

    def add(self, dataset, scope, split, role, status, detail, path=None):
        self.rows.append({'dataset': dataset, 'scope': scope, 'split': split,
                          'role': role, 'status': status, 'detail': detail,
                          'path': str(path) if path is not None else ''})

    def fail(self, msg):
        self.fatal.append(msg)

    def ok(self):
        return not self.fatal


def load_arm_dump(path: Path, expect_len: int, expect_semantic: int, label: str):
    """Load one beam dump, failing loudly on a missing file or a shape mismatch."""
    if not path.exists():
        raise FileNotFoundError(f'{label}: missing beam dump {path}')
    d = np.load(path, allow_pickle=True)
    for key in ('candidate_item_ids', 'candidate_tokens', 'sequence_scores',
                'target_item_ids', 'sid_len', 'vocab_size'):
        if key not in d.files:
            raise AssertionError(f'{label}: {path} has no key {key!r}')
    sid_len = int(d['sid_len'])
    if sid_len != expect_len:
        raise AssertionError(
            f'{label}: {path} records sid_len={sid_len}, expected {expect_len} '
            f'({expect_semantic} semantic + 1 disambiguation)')
    ct = np.asarray(d['candidate_tokens'])
    if ct.ndim != 3 or ct.shape[2] != sid_len:
        raise AssertionError(
            f'{label}: candidate_tokens shape {ct.shape} != (n, beam, {sid_len})')
    ci = np.asarray(d['candidate_item_ids'])
    if ci.ndim != 2 or ci.shape != ct.shape[:2]:
        raise AssertionError(
            f'{label}: candidate_item_ids {ci.shape} vs candidate_tokens {ct.shape}')
    ss = np.asarray(d['sequence_scores'], dtype=np.float64)
    if ss.shape != ci.shape:
        raise AssertionError(
            f'{label}: sequence_scores {ss.shape} != candidate_item_ids {ci.shape}')
    if not np.isfinite(ss).all():
        raise AssertionError(f'{label}: sequence_scores contain NaN/Inf')
    vocab = int(d['vocab_size'])
    expect_vocab = STRIDE * sid_len + 1
    if vocab != expect_vocab:
        raise AssertionError(
            f'{label}: {path} records vocab_size={vocab}, expected '
            f'stride*L+1 = {STRIDE}*{sid_len}+1 = {expect_vocab}')
    return d


def verify_item_id_convention(dump, sid_path: Path, label: str):
    """Empirically establish the item-id convention of a dump.  Never assumed.

    Asserts the canonical token layout `token = sid[row][slot] + 256*slot + 1`
    first (so a mismatch cannot be blamed on the layout), then measures how well
    each candidate id convention reproduces the decoded codes.
    """
    if not sid_path.exists():
        raise FileNotFoundError(f'{label}: missing SID array {sid_path}')
    sid = np.load(sid_path)
    ct = np.asarray(dump['candidate_tokens'], dtype=np.int64)
    ci = np.asarray(dump['candidate_item_ids'], dtype=np.int64)
    if sid.ndim != 2 or sid.shape[1] != ct.shape[2]:
        raise AssertionError(f'{label}: SID {sid.shape} vs candidate_tokens {ct.shape}')
    L = ct.shape[2]
    valid = ci >= 0                      # padded beam slots carry no candidate
    t = ct - 1
    posof = t // STRIDE
    pos_bad = (posof != np.arange(L)[None, None, :]) & valid[:, :, None]
    if pos_bad.any():
        raise AssertionError(
            f'{label}: candidate_tokens do not follow token = code + p*{STRIDE} + 1 '
            f'on {int(pos_bad.sum())} VALID slots')
    codes = (t % STRIDE)[valid]
    ids = ci[valid]
    res = {'sid_path': str(sid_path), 'sid_shape': list(sid.shape), 'sid_len': L,
           'n_candidate_slots': int(ci.size), 'n_valid_slots': int(valid.sum()),
           'n_padded_slots': int((~valid).sum()),
           'layout_verified': f'token == code + {STRIDE}*slot + 1 on every valid slot',
           'mismatch_note': 'rows whose candidate_item_ids < 0 are beam padding; '
                            'they carry no candidate and are excluded from BOTH the '
                            'layout assertion and the denominator of the '
                            'id-convention match rate. Counting them in the '
                            'denominator would score a pad-heavy dump below the '
                            'ambiguity threshold even when every real candidate '
                            'reconstructs exactly.'}
    for name, shift in (('as_zero_based_row', 0), ('as_canonical_one_based', -1)):
        idx = ids + shift
        ok = (idx >= 0) & (idx < sid.shape[0])
        idxc = np.clip(idx, 0, sid.shape[0] - 1)
        m = (sid[idxc] == codes).all(axis=1) & ok
        res[name] = {'match_rate': float(m.mean()), 'n_mismatch': int((~m).sum()),
                     'n_compared': int(m.size)}
    zero = res['as_zero_based_row']['match_rate']
    one = res['as_canonical_one_based']['match_rate']
    # HARD FAIL if the dump is not unambiguously one convention or the other.
    # Detecting the winner by `>` alone would accept a dump that matches NEITHER
    # convention at any usable rate, and the whole five-arm union would then be
    # built on a mis-mapped arm while every downstream metric still looked plain.
    if zero >= MIN_CONV_MATCH and one <= MAX_OTHER_MATCH:
        res['convention'] = 'zero_based_row'
        res['canonical_shift'] = +1
    elif one >= MIN_CONV_MATCH and zero <= MAX_OTHER_MATCH:
        res['convention'] = 'canonical_one_based'
        res['canonical_shift'] = 0
    else:
        raise AssertionError(
            f'{label}: item-id convention is AMBIGUOUS for {sid_path.name} -- '
            f'zero-based-row match {zero:.6f} (needs >= {MIN_CONV_MATCH}), '
            f'canonical-one-based match {one:.6f} (needs >= {MIN_CONV_MATCH}, other '
            f'<= {MAX_OTHER_MATCH}). Refusing to guess: a wrong convention shifts '
            f'the arm by one item and silently collapses the cross-arm union.')
    return res


def _pair_keys(ids, canonical_shift):
    """Pack valid (user, canonical item) pairs into sortable integers."""
    ids = np.asarray(ids, dtype=np.int64)
    n, B = ids.shape
    a = ids + canonical_shift
    u = np.repeat(np.arange(n, dtype=np.int64), B).reshape(n, B)
    m = a >= 0
    return (u[m] * PAIR_KEY_STRIDE + a[m]).astype(np.int64)


def cross_family_guard(spectral_dumps, raw_rqvae5_ids, label):
    """Matched (user, item) pairs shared by the spectral block and the RQ-VAE5 block.

    Independent end-to-end guard against the "+1 on both families" defect: under
    the correct normalisation the two families share a large set of candidates;
    shifting the RQ-VAE5 arm by one item collapses that overlap to a small residue
    (measured on Beauty/test: 172054 vs 31113, a 5.5x drop) while every fused
    metric still looks plausible.  The guard asserts the correct normalisation
    wins, and records both counts.
    """
    spec = None
    for d in spectral_dumps:
        k = _pair_keys(np.asarray(d['candidate_item_ids'], dtype=np.int64), +1)
        spec = k if spec is None else np.union1d(spec, k)
    correct = _pair_keys(raw_rqvae5_ids, 0)     # already canonical 1-based
    naive = _pair_keys(raw_rqvae5_ids, +1)      # the defect
    n_correct = int(np.intersect1d(spec, correct).size)
    n_naive = int(np.intersect1d(spec, naive).size)
    out = {'spectral_block_pairs': int(spec.size),
           'rqvae5_pairs': int(correct.size),
           'matched_pairs_correct_normalisation': n_correct,
           'matched_pairs_naive_plus1_on_both': n_naive,
           'ratio_correct_over_naive': (float(n_correct) / n_naive) if n_naive else None}
    if n_correct <= n_naive:
        raise AssertionError(
            f'{label}: the correct normalisation does not increase the cross-family '
            f'match over the naive +1 control ({n_correct} vs {n_naive}) -- the '
            f'convention detection or the RQ-VAE5 adapter is wrong; refusing to '
            f'fuse')
    return out


# ===========================================================================
# union construction (historical code, N_K widened for the m5 scope)
# ===========================================================================
def de_canonicalize_view(dump):
    """Re-express an ALREADY-1-based dump in the 0-based form `build_union` expects.

    `fusion_audit.build_union` canonicalises every dump with `+1`, because the
    frozen historical dumps store the 0-based SID-array row index.  The RQ-VAE5
    dumps written this round store the canonical 1-based id already (verified
    empirically by `verify_item_id_convention`; the measurement is recorded in
    RUN_METADATA.json).  Feeding them in unchanged would shift the fifth arm's
    items by one relative to the other four and silently corrupt the cross-arm
    union join, so the ids are shifted back by one here and restored by
    `build_union` itself.
    """
    out = {k: dump[k] for k in dump.files}
    for key in ('candidate_item_ids', 'target_item_ids'):
        a = np.asarray(dump[key], dtype=np.int64)
        out[key] = np.where(a >= 0, a - 1, -1)
    return out


def load_scope_inputs(fa, dataset, affinity, split, scope):
    """Return (dumps, provenance) for one scope, or (None, missing_path)."""
    dumps, prov = [], []
    raw_rqvae5_ids = None
    for arm in ARMS[scope]:
        if arm == 'rqvae5':
            path = rqvae5_dump_path(dataset, split)
            if not path.exists():
                return None, path
            d = load_arm_dump(path, ARM_LEN[arm], ARM_SEMANTIC[arm],
                              f'{dataset}/{split}/{arm}')
            conv = verify_item_id_convention(d, rqvae5_sid_path(dataset),
                                             f'{dataset}/{split}/{arm}')
            if conv['canonical_shift'] != 0:
                raise AssertionError(
                    f'{dataset}/{split}/{arm}: the RQ-VAE5 adapter assumes an '
                    f'already-canonical dump, but the measurement says '
                    f'{conv["convention"]} (canonical shift {conv["canonical_shift"]})')
            raw_rqvae5_ids = np.asarray(d['candidate_item_ids'], dtype=np.int64)
            d = de_canonicalize_view(d)
            prov.append({'arm': arm, 'path': str(path), 'sha256': sha256_file(path),
                         'sid_len': ARM_LEN[arm], 'semantic_len': ARM_SEMANTIC[arm],
                         'source': 'this_round',
                         'item_id_convention_measured': conv,
                         'adapter': 'de_canonicalize_view (-1) so that '
                                    'build_union(+1) restores the canonical id'})
        else:
            k = ARM_SEMANTIC[arm]
            path = spectral_dump_path(dataset, affinity, split, k)
            if not path.exists():
                return None, path
            d = load_arm_dump(path, ARM_LEN[arm], k, f'{dataset}/{split}/{arm}')
            entry = {'arm': arm, 'path': str(path), 'sha256': sha256_file(path),
                     'sid_len': ARM_LEN[arm], 'semantic_len': k,
                     'source': 'frozen_historical',
                     'adapter': 'none; build_union applies '
                                'canonicalize_item_ids(+1)'}
            # EVERY dump is measured, never assumed -- including the four historical
            # arms, so the 0-based claim is evidence for this dataset/split and not
            # a restatement of fusion_audit.py's header comment.
            entry['item_id_convention_measured'] = verify_item_id_convention(
                d, ART_ROOT / 'data' / dataset /
                f'{dataset}_t5_spectral_{affinity}_k{ARM_SEMANTIC[arm]}_sid'
                f'{ARM_LEN[arm]}.npy',
                f'{dataset}/{split}/{arm}')
            prov.append(entry)
        dumps.append(d)

    if scope == 'm5' and raw_rqvae5_ids is not None:
        guard = cross_family_guard(dumps[:-1], raw_rqvae5_ids,
                                   f'{dataset}/{split}')
        prov[-1]['cross_family_guard'] = guard
        print(f'    cross-family guard ({split}): '
              f'matched(correct)={guard["matched_pairs_correct_normalisation"]} '
              f'matched(naive +1)={guard["matched_pairs_naive_plus1_on_both"]} '
              f'ratio={guard["ratio_correct_over_naive"]:.2f}')

    base = fa.canonicalize_item_ids(dumps[0]['target_item_ids'])
    for arm, d in zip(ARMS[scope][1:], dumps[1:]):
        other = fa.canonicalize_item_ids(d['target_item_ids'])
        if not np.array_equal(base, other):
            bad = int((base != other).sum())
            raise AssertionError(
                f'{dataset}/{split}/{scope}: arm {arm} target ids disagree with '
                f'{ARMS[scope][0]} on {bad}/{base.size} users -- user order or '
                f'item-id convention mismatch; refusing to fuse')
    return dumps, prov


def set_nk(fa, p5, scope):
    """Widen the factorization axis to this scope's arm count, in BOTH namespaces.

    `fusion_audit.build_union` / `confidence_matrix` size their factorization axis
    from `fusion_audit.N_K`; `phase5_fusion.uniform_weights` /
    `single_k_weights` size the weight matrix from `phase5_fusion.N_K`, which
    `from fusion_audit import ... N_K ...` bound to the value 4 at import time.  A
    five-arm scope therefore needs both names to be set -- patching only one
    silently produces a 4-column weight matrix against a 5-column score tensor
    (caught here by numpy's broadcast error, not by a wrong number).
    """
    nk = len(ARMS[scope])
    if fa.N_K != nk:
        fa.N_K = nk
    if getattr(p5, 'N_K', None) != nk:
        p5.N_K = nk
    return nk


def build_scope_union(fa, dumps, scope):
    if fa.N_K != len(ARMS[scope]):
        raise AssertionError(
            f'fusion_audit.N_K={fa.N_K} but the {scope} scope has '
            f'{len(ARMS[scope])} arms; set_nk() must run first')
    return fa.build_union(dumps)


def scope_confidence(fa, dumps, scope):
    if fa.N_K != len(ARMS[scope]):
        raise AssertionError(
            f'fusion_audit.N_K={fa.N_K} but the {scope} scope has '
            f'{len(ARMS[scope])} arms; set_nk() must run first')
    return fa.confidence_matrix(dumps)


POOL_LABEL = {'m5': 'native_length_pool',
              'spectral4': 'native_length_pool_spectral4'}


def arm_length_label(scope):
    return '|'.join(f'{a}:{ARM_LEN[a]}' for a in ARMS[scope])


def arm_semantic_label(scope):
    return '|'.join(f'{a}:{ARM_SEMANTIC[a]}' for a in ARMS[scope])


# ===========================================================================
# strategy evaluation
# ===========================================================================
def one_hot_weights(n, idx, k):
    w = np.zeros((n, k), dtype=np.float64)
    w[:, idx] = 1.0
    return w


def pu_for(fa, U, w):
    """`fusion_audit.per_user_ndcg` on a union tuple: (items, score, target_pos, mask)."""
    items_pad, score_tensor, _target, target_pos, valid_mask = U
    return fa.per_user_ndcg(items_pad, score_tensor, target_pos, valid_mask, w)


# bound by `import_historical()`; needed by the blocked-CFLA row builder and by the
# metadata builder, which must report the certified N_BOOT/seed constants.
_P5 = None


def weights_for(fa, p5, scope, strat, n, q_norm, conf, detail):
    k = len(ARMS[scope])
    if strat == 'uniform':
        return p5.uniform_weights(n)
    if strat == 'best_single':
        return one_hot_weights(n, ARMS[scope].index(detail['best_single_arm']), k)
    if strat == 'confidence_only':
        return fa.weights_from(0.0, detail['selected_eta'], q_norm, conf)
    if strat == 'structure_aware':
        return fa.weights_from(1.0, 0.0, q_norm, conf)
    if strat == 'valid_grid_search':
        return fa.weights_from(detail['selected_gamma'], detail['selected_eta'],
                               q_norm, conf)
    raise AssertionError(f'unknown strategy {strat!r}')


def evaluate_strategies(fa, p5, unions, confs, q_norm, scope, strategy_status):
    """Compute every strategy the inputs allow, for BOTH splits.

    Returns {strategy: {'status', 'missing_input', 'producing_command',
                        'detail', 'per_user': {'valid': pu|None, 'test': pu|None}}}
    plus '_single' with the per-arm vectors.
    A strategy whose certified inputs do not exist comes back with its status set
    and `per_user == {'valid': None, 'test': None}` -- never with a number.
    """
    U_v, U_t = unions['valid'], unions['test']
    n_v, n_t = U_v[0].shape[0], U_t[0].shape[0]
    k = len(ARMS[scope])
    out = {}

    def blocked(name):
        st, missing, cmd = strategy_status[name]
        out[name] = {'status': st, 'missing_input': missing, 'producing_command': cmd,
                     'detail': {}, 'per_user': {'valid': None, 'test': None}}

    singles = {'valid': {}, 'test': {}}
    for m, arm in enumerate(ARMS[scope]):
        singles['valid'][arm] = pu_for(fa, U_v, one_hot_weights(n_v, m, k))
        singles['test'][arm] = pu_for(fa, U_t, one_hot_weights(n_t, m, k))
    valid_ndcg20 = {a: float(singles['valid'][a]['NDCG@20'].mean())
                    for a in ARMS[scope]}
    best_arm = max(ARMS[scope], key=lambda a: valid_ndcg20[a])
    out['_single'] = {'per_arm': singles, 'valid_NDCG@20': valid_ndcg20,
                      'best_arm': best_arm}

    out['best_single'] = {
        'status': ST_OK, 'missing_input': '', 'producing_command': '',
        'per_user': {'valid': singles['valid'][best_arm],
                     'test': singles['test'][best_arm]},
        'detail': {'best_single_arm': best_arm, 'selection_split': 'valid',
                   'valid_NDCG@20_by_arm': valid_ndcg20,
                   'best_single_valid_NDCG@20': valid_ndcg20[best_arm]}}

    out['uniform'] = {
        'status': ST_OK, 'missing_input': '', 'producing_command': '',
        'per_user': {'valid': pu_for(fa, U_v, p5.uniform_weights(n_v)),
                     'test': pu_for(fa, U_t, p5.uniform_weights(n_t))},
        'detail': {}}

    # `weights_from(gamma, eta, q_norm, conf)` computes `gamma*q_norm + eta*conf`.
    # With gamma == 0 the Q term is identically zero, but Python still evaluates
    # `q_norm[None, :]`, so a placeholder is needed when the five-arm pool has no
    # Q.  `0.0 * q` is exactly 0.0 for every finite q, so passing a zero vector
    # here is EXACT, not an approximation or a substituted scientific value --
    # nothing about this strategy depends on Q.  When a real q_norm exists it is
    # used, as the historical driver does.
    q_conf = np.zeros(len(ARMS[scope])) if q_norm is None else q_norm
    tab = []
    for e in p5.ETA_GRID:
        pu = pu_for(fa, U_v, fa.weights_from(0.0, e, q_conf, confs['valid']))
        tab.append({'eta': float(e), 'valid_NDCG@20': float(pu['NDCG@20'].mean())})
    best_eta = max(tab, key=lambda r: r['valid_NDCG@20'])['eta']
    out['confidence_only'] = {
        'status': ST_OK, 'missing_input': '', 'producing_command': '',
        'per_user': {'valid': pu_for(fa, U_v, fa.weights_from(0.0, best_eta, q_conf,
                                                              confs['valid'])),
                     'test': pu_for(fa, U_t, fa.weights_from(0.0, best_eta, q_conf,
                                                             confs['test']))},
        'detail': {'selected_eta': best_eta, 'selection_split': 'valid',
                   'eta_grid': [float(e) for e in p5.ETA_GRID],
                   'valid_NDCG@20_by_eta': tab,
                   'q_term': 'gamma=0 so the structure term is exactly zero; '
                             + ('a zero placeholder vector is passed because the '
                                'RQ-VAE5 arm has no Q (exact: 0.0*q == 0.0)'
                                if q_norm is None else
                                'the real VALID z-scored Q vector is passed')}}

    if q_norm is None:
        blocked('structure_aware')
        blocked('valid_grid_search')
    else:
        out['structure_aware'] = {
            'status': ST_OK, 'missing_input': '', 'producing_command': '',
            'per_user': {'valid': pu_for(fa, U_v, fa.weights_from(1.0, 0.0, q_norm,
                                                              confs['valid'])),
                         'test': pu_for(fa, U_t, fa.weights_from(1.0, 0.0, q_norm,
                                                             confs['test']))},
            'detail': {'gamma': 1.0, 'eta': 0.0,
                       'q_norm': [float(x) for x in np.asarray(q_norm).ravel()]}}
        best, table = p5.select_on_valid(U_v, q_norm, confs['valid'])
        out['valid_grid_search'] = {
            'status': ST_OK, 'missing_input': '', 'producing_command': '',
            'per_user': {'valid': pu_for(fa, U_v, fa.weights_from(best['gamma'],
                                                              best['eta'], q_norm,
                                                              confs['valid'])),
                         'test': pu_for(fa, U_t, fa.weights_from(best['gamma'],
                                                             best['eta'], q_norm,
                                                             confs['test']))},
            'detail': {'selected_gamma': best['gamma'],
                       'selected_eta': best['eta'],
                       'selection_split': 'valid',
                       'valid_NDCG@20': best['valid_NDCG@20'],
                       'gamma_grid': [float(g) for g in p5.GAMMA_GRID],
                       'eta_grid': [float(e) for e in p5.ETA_GRID],
                       'valid_grid': table}}

    blocked('cfla')
    return out


# ===========================================================================
# CFLA (candidate-NLL aggregation)
# ===========================================================================
def cfla_scores_from_nll(nll_by_arm, scalers, arms):
    """`logsumexp_F(-zNLL_F)` -- the certified CFLA aggregation.

    `nll_by_arm` maps arm -> per-row TotalNLL in the SAME row order; `scalers` maps
    arm -> (mu, sigma) fitted on VALID.  Kept here so the blocked status is a
    statement about missing INPUTS, not about missing implementation.
    """
    Z = np.empty((len(arms),) + np.asarray(nll_by_arm[arms[0]]).shape, dtype=np.float64)
    for i, arm in enumerate(arms):
        mu, sigma = scalers[arm]
        if not np.isfinite([mu, sigma]).all() or sigma <= 0:
            raise AssertionError(f'CFLA: invalid VALID-fitted scaler for {arm}: '
                                 f'mu={mu} sigma={sigma}')
        Z[i] = (np.asarray(nll_by_arm[arm], dtype=np.float64) - mu) / sigma
    if not np.isfinite(Z).all():
        raise AssertionError('CFLA: non-finite zNLL')
    m = (-Z).max(axis=0)
    return m + np.log(np.exp(-Z - m).sum(axis=0))


def cfla_rows_blocked(run_id, dataset, affinity, scope, status, missing, command):
    return [{'run_id': run_id, 'dataset': dataset, 'affinity': affinity,
             'scope': scope, 'comparison': 'cfla_vs_uniform', 'strategy_a': 'cfla',
             'strategy_b': 'uniform', 'metric': k, 'a_value': '', 'b_value': '',
             'delta': '', 'pct_change': '', 'ci_lo': '', 'ci_hi': '', 'p_value': '',
             'n_boot': _P5.N_BOOT, 'seed': _P5.RNG_SEED, 'n_users': '',
             'ci95_excludes_zero': '', 'status': status,
             'missing_input': str(missing),
             'notes': 'no CFLA number and no bootstrap: the per-arm candidate NLL of '
                      'the five-arm union does not exist. ' + command[:300]}
            for k in METRIC_KEYS]


# ===========================================================================
# bootstrap
# ===========================================================================
def bootstrap_pairs(p5, run_id, dataset, affinity, scope, res):
    """Paired user-level bootstrap: 5000 resamples, seed 20260912.

    Every row records n_boot, seed, ci_lo, ci_hi, p_value, n_users and whether the
    95% CI excludes zero (hard requirement 4).
    """
    uni_t = res['uniform']['per_user']['test']
    best_arm = res['_single']['best_arm']
    bs_arm_t = res['_single']['per_arm']['test'][best_arm]
    src = [
        ('uniform_vs_best_single', uni_t, bs_arm_t,
         {'a': 'uniform', 'b': f'best_single:{best_arm}'}),
        ('confidence_only_vs_uniform', res['confidence_only']['per_user']['test'],
         uni_t, {'a': 'confidence_only', 'b': 'uniform'}),
    ]
    for name in ('valid_grid_search', 'structure_aware'):
        r = res.get(name)
        if r is not None and r['per_user']['test'] is not None:
            src.append((f'{name}_vs_uniform', r['per_user']['test'], uni_t,
                        {'a': name, 'b': 'uniform'}))
    rows = []
    for name, a, b, ref in src:
        for key in METRIC_KEYS:
            r = p5.paired_bootstrap(a[key], b[key])
            lo, hi = float(r['ci_lo']), float(r['ci_hi'])
            pct = r.get('pct_change')
            rows.append({
                'run_id': run_id, 'dataset': dataset, 'affinity': affinity,
                'scope': scope, 'comparison': name, 'strategy_a': ref['a'],
                'strategy_b': ref['b'], 'metric': key,
                'a_value': repr(float(a[key].mean())),
                'b_value': repr(float(b[key].mean())),
                'delta': repr(float(r['delta'])),
                'pct_change': '' if pct is None else repr(float(pct)),
                'ci_lo': repr(lo), 'ci_hi': repr(hi),
                'p_value': repr(float(r['p_value'])), 'n_boot': p5.N_BOOT,
                'seed': p5.RNG_SEED, 'n_users': int(r['n_users']),
                'ci95_excludes_zero': bool(lo > 0.0 or hi < 0.0),
                'status': ST_OK, 'missing_input': '',
                'notes': r.get('note', '') or ''})
    return rows


# ===========================================================================
# csv / json writers
# ===========================================================================
def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _same_measurement(a, b):
    """Two rows for the same cell agree on everything except which run made them."""
    keys = (set(a) | set(b)) - {'run_id'}
    return all(a.get(k, '') == b.get(k, '') for k in keys)


def merge_aggregate(dest: Path, fields, key_fields, run_id):
    """Rebuild the contract-level aggregate from EVERY run directory.

    `m5_rq5/{VALID,TEST,BOOTSTRAP}.csv` is a derived union of all runs, so a later
    run (e.g. the one that picks Toys_and_Games up) can ADD rows without disturbing
    an earlier run's output.  The previous aggregate is archived under
    `m5_rq5/history/` before it is replaced, and a merge that would DROP a
    previously present row is a hard failure rather than a silent overwrite.
    """
    runs_root = M5DIR / 'runs'
    all_rows, seen = [], set()
    # An intentional revision (e.g. an operator decision that changes a cell's
    # STATUS) is not a conflict: it is recorded in SUPERSEDED.json with a reason,
    # and the superseded run directories are excluded from the union.  The run
    # directories themselves are never deleted, so the old measurement stays on
    # disk and auditable.
    sup_path = M5DIR / 'SUPERSEDED.json'
    superseded = {}
    if sup_path.exists():
        try:
            superseded = json.loads(sup_path.read_text()).get('superseded', {})
        except Exception:
            superseded = {}
    sources = sorted(p for p in runs_root.glob('*')
                     if p.is_dir() and p.name not in superseded)
    if superseded:
        print(f'  superseded run(s) excluded from the merge: '
              f'{", ".join(sorted(superseded))}')
    by_key = {}
    for rd in sources:
        for row in read_csv_rows(rd / dest.name):
            key = tuple(row.get(k, '') for k in key_fields)
            if key in by_key:
                if not _same_measurement(by_key[key], row):
                    raise AssertionError(
                        f'{dest.name}: key {key} was measured differently by two '
                        f'runs ({by_key[key].get("run_id")} and '
                        f'{row.get("run_id")}); refusing to merge conflicting '
                        f'measurements')
                continue          # a deterministic re-measurement of the same cell
            by_key[key] = row
            seen.add(key)
            all_rows.append(row)
    all_rows.sort(key=lambda r: (r.get('dataset', ''), r.get('scope', ''),
                                 r.get('strategy', r.get('comparison', '')),
                                 r.get('metric', ''), r.get('run_id', '')))
    if dest.exists():
        old = read_csv_rows(dest)
        old_keys = {tuple(r.get(k, '') for k in key_fields) for r in old}
        new_keys = {tuple(r.get(k, '') for k in key_fields) for r in all_rows}
        lost = old_keys - new_keys
        if lost:
            raise AssertionError(
                f'{dest}: the merge would drop {len(lost)} existing row(s), e.g. '
                f'{sorted(lost)[0]}; refusing to overwrite')
        hist = M5DIR / 'history'
        hist.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        dest.replace(hist / f'{dest.stem}.{stamp}.{run_id}{dest.suffix}')
    tmp = dest.with_name(dest.name + f'.tmp.{os.getpid()}')
    write_csv(tmp, fields, all_rows)
    os.replace(tmp, dest)
    return len(all_rows), len(sources)


# ===========================================================================
# metadata
# ===========================================================================
def skipped_entries(args, per_dataset):
    """Explicit list of what was NOT computed and the artifact that is missing."""
    out = []
    for ds in DATASETS:
        if ds in args.datasets:
            for scope, st in per_dataset.get(ds, {}).get('scopes', {}).items():
                if st.get('status') == ST_PENDING:
                    out.append({'dataset': ds, 'scope': scope,
                                'what': 'all strategies', 'status': ST_PENDING,
                                'reason': 'the RQ-VAE5 beam dump does not exist yet '
                                          '(its T5 training is still queued); '
                                          're-running the driver after it lands picks '
                                          'the dataset up and adds its rows without '
                                          'disturbing earlier output',
                                'missing_input': st.get('missing_input', '')})
            continue
        out.append({'dataset': ds, 'scope': 'm5', 'what': 'all strategies',
                    'status': 'NOT_REQUESTED_THIS_RUN',
                    'reason': 'dataset not selected by --datasets for this run',
                    'missing_input': str(rqvae5_dump_path(ds, 'valid'))})
    out.append({
        'what': 'cfla on the m5 scope', 'status': ST_BLOCKED_NO_NLL,
        'reason': 'CFLA needs NLL_F(u,i): the TotalNLL of every union candidate under '
                  'EACH of the five checkpoints, with (mu_F, sigma_F) fitted on '
                  'VALID. The four spectral arms have repaired historical NLL, but '
                  'the RQ-VAE5 arm was trained this round and no candidate-NLL run '
                  'has scored the union under it. Fabricating it is forbidden, so '
                  'the CFLA rows carry the status.',
        'missing_input': ' ; '.join(str(rqvae5_nll_path(d, s)) for d in args.datasets
                                    for s in ('valid', 'test')),
        'producing_command': cfla_producing_command(args.datasets)})
    out.append({
        'what': 'structure_aware / valid_grid_search on the m5 scope',
        'status': ST_BLOCKED_NO_Q,
        'reason': 'both strategies exponentiate the per-arm structure weight Q, so '
                  'the five-arm pool needs a Q for the RQ-VAE5 arm. Q is '
                  'a*z(Sep) + b*z(Bal) + c*z(Unique) and Sep/Bal need the groups '
                  'into which the 768 embedding dimensions are partitioned by the '
                  'factorization; an RQ-VAE level is a learned residual code over '
                  'the FULL embedding, not a subset of dimensions, so Sep and Bal '
                  'are undefined for that arm and no certified script can emit the '
                  'file. The four-arm spectral4 scope, where Q does exist, is '
                  'reported alongside.',
        'missing_input': ' ; '.join(str(rqvae5_dim_groups_path(d))
                                    for d in args.datasets),
        'producing_command': 'none: this needs a methodology decision (how to define '
                             'dimension groups for an RQ-VAE arm), not a command'})
    out.append({
        'what': 'cfla on the spectral4 control scope', 'status': ST_NOT_IN_SCOPE,
        'reason': 'out of scope for this deliverable, NOT a missing input: the '
                  'repaired spectral4 candidate NLL and the VALID-fitted scalers '
                  'both exist, so a four-arm CFLA is computable today. It is left '
                  'out because the deliverable is the five-arm M5 and a second pool '
                  'needs its own candidate-NLL plumbing.',
        'available_inputs': [
            str(REPO / 'outputs' / 'multifactor_nll_repair' / 'validation' /
                'spectral4' / '<ds>_candidate_nll.parquet'),
            str(REPO / 'outputs' / 'multifactor_nll_repair' / 'test' / 'spectral4' /
                '<ds>_candidate_nll.parquet'),
            str(REPO / 'outputs' / 'multifactor_nll_repair' / 'validation' /
                'validation_nll_scalers.json') + ' (block pools.spectral4)'],
        'producing_command': 'extend a CFLA driver over those parquets; the '
                             'aggregation is logsumexp_F(-zNLL_F) with (mu_F, '
                             'sigma_F) from the spectral4 scaler block'})
    out.append({
        'what': 'the m5 pool with affinity=abspearson_v3', 'status': ST_NA,
        'reason': 'the RQ-VAE5 arm built this round has a single beam set with no '
                  'affinity variant, so a five-arm pool cannot be formed for '
                  'abspearson_v3 without mixing two conventions. The historical M5 '
                  'canonical affinity is nonneg_pearson_v3, which is what this run '
                  'uses; the abspearson_v3 spectral dumps were validated for '
                  'existence but are not fused with an affinity-free arm.',
        'missing_input': str(OUT / 'runs_5plus1' / '<ds>' / 'rqvae5' / 'beams' /
                             'valid_k5_sid6.npz') +
                             ' (illustrative: the arm has no affinity axis at all)'})
    out.append({
        'what': 'legacy token-prefix metric', 'status': 'NEVER_COMPUTED',
        'reason': 'HARNESS_HANDOFF section 5 invariant 2: item-level metrics only, '
                  'never --eval_mode token.'})
    out.append({
        'what': 'test-set selection of any kind', 'status': 'NEVER_PERFORMED',
        'reason': 'every selection in this driver is made on VALID item-level '
                  'NDCG@20.'})
    return out


def reproduction_check(args, test_rows):
    if not HIST_PHASE5_JSON.exists():
        return {'reference': str(HIST_PHASE5_JSON), 'available': False}
    hist = json.loads(HIST_PHASE5_JSON.read_text())
    comp = {}
    for row in test_rows:
        if row['scope'] != 'spectral4' or row['status'] != ST_OK:
            continue
        for ref in hist['fusion']:
            if (ref['dataset'] == row['dataset']
                    and ref['affinity'] == args.affinity
                    and ref['strategy'] == row['strategy']):
                mine = float(row['NDCG@20'])
                theirs = float(ref['NDCG@20'])
                comp[f'{row["dataset"]}|{row["strategy"]}'] = {
                    'mine_NDCG@20': repr(mine), 'historical_NDCG@20': repr(theirs),
                    'abs_diff': abs(mine - theirs)}
    det = {k: v for k, v in comp.items() if not k.endswith('best_single')}
    return {
        'reference': str(HIST_PHASE5_JSON),
        'reference_sha256': sha256_file(HIST_PHASE5_JSON),
        'scope': 'spectral4 (the historical four-arm fusion)',
        'comparison': comp,
        'all_deterministic_strategies_reproduce': bool(det) and all(
            v['abs_diff'] < 1e-12 for v in det.values()),
        'note': 'best_single is the one strategy EXPECTED to differ: the historical '
                'driver selected the arm on TEST (a leak), this driver selects it on '
                'VALID as hard requirement 1 demands. uniform / per-arm singles / '
                'valid_grid_search / structure_aware / confidence_only are '
                'deterministic given the frozen dumps and must match exactly.'}


def build_metadata(args, run_id, run_dir, report, gpu_info, busy, reasons,
                   per_dataset, valid_rows, test_rows, boot_rows, t0):
    return {
        'run_id': run_id,
        'run_dir': str(run_dir),
        'generated': utc_now(),
        'phase': 'PHASE 8-10 -- new M5 (Spectral K2-K5 + RQ-VAE5) + CFLA + bootstrap',
        'driver': str(Path(__file__).resolve()),
        'driver_sha256': sha256_file(Path(__file__).resolve()),
        'elapsed_s': round(time.time() - t0, 2),
        'datasets': args.datasets,
        'scopes': args.scopes,
        'affinity': args.affinity,
        'selection_metric': 'item-level VALID NDCG@20 (never TEST, never token-prefix)',
        'seeds': {'bootstrap_seed': _P5.RNG_SEED, 'n_boot': _P5.N_BOOT,
                  'gamma_grid': [float(g) for g in _P5.GAMMA_GRID],
                  'eta_grid': [float(e) for e in _P5.ETA_GRID]},
        'lengths': {s: {'arms': list(ARMS[s]), 'sid_lengths': arm_length_label(s),
                        'semantic_lengths': arm_semantic_label(s),
                        'all_len_6': all(ARM_LEN[a] == 6 for a in ARMS[s]),
                        'mixed_token_budgets': len({ARM_LEN[a] for a in ARMS[s]}) > 1}
                    for s in args.scopes},
        'pool_labeling': {
            'pool_values_used': POOL_LABEL,
            'm5_pool_is_length_matched': False,
            'measured_arm_lengths': dict(ARM_LEN),
            'native_length_fact': 'each arm keeps the SID length it was trained at: '
                                  'spectral K2 -> 3, K3 -> 4, K4 -> 5, K5 -> 6, '
                                  'RQ-VAE5 -> 6. There are zero `*_k[234]_sid6*.npy` '
                                  'files, so a five-arm pool in which every model '
                                  'spends 6 tokens per item is NOT CONSTRUCTIBLE '
                                  'from the frozen artifacts.',
            'caveat': 'This M5 fuses FIVE models at MIXED SID lengths (token budgets '
                      '3/4/5/6/6). Comparing it with a single length-matched L=6 '
                      'model confounds two effects at once: (a) more models and '
                      '(b) different per-item token budgets. The campaign '
                      'length-matched fairness claim applies ONLY to the '
                      'single-model table (single_5plus1/, 4 families x 4 datasets '
                      'at L=6) and NEVER to the M5.',
            'enforcement': 'every VALID/TEST row carries pool, arm_sid_lengths, '
                           'arm_semantic_lengths, mixed_token_budgets and '
                           'length_matched_L6=False; each arm length is read from '
                           'the dump (sid_len + candidate_tokens.shape[2]) and '
                           'asserted on load by `load_arm_dump`, so an arm cannot '
                           'silently change length.'},
        'item_id_convention': {
            'canonical': 'item id = SID-array row index + 1',
            'frozen_historical_dumps': '0-based SID-array row index (pre-fix dumper); '
                                       'canonicalised +1 inside '
                                       'fusion_audit.build_union',
            'rqvae5_dumps_this_round': 'ALREADY canonical 1-based (measured, not '
                                       'assumed); shifted -1 by de_canonicalize_view '
                                       'so that build_union(+1) restores it',
            'token_layout_verified': 'token = sid[row][slot] + 256*slot + 1 asserted '
                                     'on every loaded dump before use',
            'detection_is_enforced': 'the token-reconstruction test must separate the '
                                     'two conventions by at least '
                                     f'{MIN_CONV_MATCH} vs {MAX_OTHER_MATCH}; '
                                     'otherwise the driver raises instead of '
                                     'picking the larger of two bad matches',
            'cross_family_guard': 'per split, the (user, item) pairs shared by the '
                                  'spectral block and the RQ-VAE5 block are counted '
                                  'in canonical space under the correct '
                                  'normalisation and under the naive +1-on-both '
                                  'control; the driver raises unless the correct '
                                  'one wins. Counts are recorded under '
                                  'per_dataset[<ds>].scopes.m5.provenance_*'
                                  '[-1].cross_family_guard.',
            'cross_arm_check': 'all arms must agree on the canonicalised target id '
                               'for every user, else the driver raises'},
        'gpu_guard': {
            'busy_at_start': bool(busy), 'reasons': reasons,
            'nvidia_smi_ok': gpu_info.get('nvidia_smi_ok'),
            'memory_used_MiB': [g['memory_used_MiB'] for g in gpu_info.get('gpus', [])],
            'host_training_processes': scan_host_processes(),
            'gpu_stages_requested': bool(args.gpu_stages),
            'cuda_context_created': False,
            'note': 'this pipeline is CPU-only. The only GPU stage is the '
                    'candidate-NLL scoring CFLA would need, which refuses to run '
                    'while the device is busy.'},
        'inputs': report.files,
        'reference_sources': {
            'fusion_audit.py': {'path': str(HIST_FUSION_AUDIT),
                                'sha256': sha256_file(HIST_FUSION_AUDIT)},
            'phase5_fusion.py': {'path': str(HIST_PHASE5),
                                 'sha256': sha256_file(HIST_PHASE5),
                                 'reused': ['GAMMA_GRID', 'ETA_GRID', 'N_BOOT',
                                            'RNG_SEED', 'METRICS', 'uniform_weights',
                                            'single_k_weights', 'select_on_valid',
                                            'grid_ndcg20', 'paired_bootstrap']},
            'mf_candidate_nll_repaired.py': {'path': str(REPAIRED_NLL),
                                             'sha256': sha256_file(REPAIRED_NLL),
                                             'published_sha256': REPAIRED_SHA,
                                             'reused': ['assert_permutation',
                                                        'write_scores_back_to_original_rows']},
            'mf_models.py': {'path': str(MF_MODELS),
                             'sha256': sha256_file(MF_MODELS),
                             'reused': 'candidate_total_nll (the certified NLL '
                                       'scorer; not callable until the rqvae5 '
                                       'factorization is registered)'},
            'read_only_roots_untouched': {
                'roots': [str(ART_ROOT), str(UPSTREAM_ROOT)],
                'mechanism': 'sys.dont_write_bytecode = True is set before the '
                             'historical modules are imported, so no '
                             '__pycache__ entry can be added under them; '
                             'phase5_fusion.py also does CANON.mkdir(exist_ok=True) '
                             'at import time and the driver refuses to import it '
                             'unless that directory already exists.',
                'note': 'a pre-existing '
                        'frozen_audit/__pycache__/phase5_fusion.cpython-'
                        '312.pyc (mtime 2026-09-17 13:46:57 +0800) was created by '
                        'the parallel PHASE 11 driver before this one first ran '
                        '(2026-09-17 13:49:57 +0800). Nothing was deleted or '
                        'modified: files under the read-only roots are left exactly '
                        'as found.'},
            'N_K_override': {
                'modules': [str(HIST_FUSION_AUDIT), str(HIST_PHASE5)],
                'historical_value': 4, 'm5_value': 5,
                'why': 'build_union/confidence_matrix size the factorization axis '
                       'from fusion_audit.N_K, and uniform_weights/single_k_weights '
                       'size the weight matrix from phase5_fusion.N_K, which the '
                       'historical driver bound BY VALUE at its own import time. '
                       'Both names are set per scope by set_nk(); the spectral4 '
                       'scope keeps the historical 4. No source file is modified.',
                'set_per_scope': True}},
        'row_alignment': {
            'helper_source': str(REPAIRED_NLL), 'helper_sha256': REPAIRED_SHA,
            'rule': 'scores_original[order] = scores_sorted, NEVER '
                    'scores_sorted[order]',
            'identity_assertion': 'after the write-back every row is re-scored '
                                  'DIRECTLY in the original row order by score_fn '
                                  'and must match; plus a pair-keyed consistency '
                                  'check of the scatter itself. The direct re-score '
                                  'is what catches the historical defect -- the '
                                  'pair-keyed check alone takes the (possibly '
                                  'mis-ordered) stream as ground truth.',
            'test': str(DRIVERS / 'test_m5_row_alignment.py') +
                    ' (non-involution fixture)'},
        'skipped': skipped_entries(args, per_dataset),
        'per_dataset': per_dataset,
        'reproduction_check_spectral4': reproduction_check(args, test_rows),
        'counts': {'valid_rows': len(valid_rows), 'test_rows': len(test_rows),
                   'bootstrap_rows': len(boot_rows)},
    }


# ===========================================================================
# orchestration
# ===========================================================================
def import_historical():
    """Import the historical fusion library and the phase5 template, unmodified."""
    global _FA, _P5
    for p in (str(REPO), str(HIST_PHASE5.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fusion_audit as fa
    import phase5_fusion as p5
    _FA, _P5 = fa, p5
    return fa, p5


def validate_only(args, report: Report):
    """Every input path this driver would touch, checked and hashed."""
    files = []
    for dataset in args.datasets:
        for scope in args.scopes:
            for split in ('valid', 'test'):
                for arm in ARMS[scope]:
                    if arm == 'rqvae5':
                        p = rqvae5_dump_path(dataset, split)
                        extra = [('sid', rqvae5_sid_path(dataset)),
                                 ('ckpt', rqvae5_ckpt_path(dataset))]
                    else:
                        k = ARM_SEMANTIC[arm]
                        p = spectral_dump_path(dataset, args.affinity, split, k)
                        extra = []
                    st = 'FOUND' if p.exists() else ST_PENDING
                    report.add(dataset, scope, split, f'beam:{arm}', st,
                               f'{p.stat().st_size} bytes' if p.exists()
                               else 'missing beam dump', p)
                    for role, e in extra:
                        st2 = ('FOUND' if e.exists() else ST_PENDING) \
                            if arm == 'rqvae5' else 'FOUND'
                        report.add(dataset, scope, split, f'{role}:{arm}', st2,
                                   'supporting artifact' if e.exists()
                                   else 'missing supporting artifact', e)
                        files.append(e)
                    files.append(p)
        q = structure_json_path(dataset, args.affinity)
        report.add(dataset, '-', '-', 'structure_q', 'FOUND' if q.exists() else 'MISSING',
                   'structure weight Q for the four spectral arms', q)
        files.append(q)
        dg = rqvae5_dim_groups_path(dataset)
        report.add(dataset, 'm5', '-', 'structure_q:rqvae5',
                   'FOUND' if dg.exists() else ST_BLOCKED_NO_Q,
                   'dimension groups for the RQ-VAE5 arm (needed by structure_aware / '
                   'valid_grid_search on the m5 scope)', dg)
        files.append(dg)
        for split in ('valid', 'test'):
            nll = rqvae5_nll_path(dataset, split)
            report.add(dataset, 'm5', split, 'candidate_nll:rqvae5',
                       'FOUND' if nll.exists() else ST_BLOCKED_NO_NLL,
                       'cross-factorization candidate NLL for CFLA', nll)
            files.append(nll)
    for p in (HIST_FUSION_AUDIT, HIST_PHASE5, HIST_PHASE5_JSON, REPAIRED_NLL,
              MF_MODELS):
        files.append(p)
    report.files = [{'path': str(p), 'exists': p.exists(),
                     'sha256': sha256_file(p) if p.exists() else None}
                    for p in files]
    return report


def row_for(run_id, dataset, affinity, scope, strat, split, r, common):
    d = r['detail']
    pu = r['per_user'][split]
    row = dict(common)
    row.update({
        'run_id': run_id, 'dataset': dataset, 'split': split, 'affinity': affinity,
        'scope': scope, 'strategy': strat, 'status': r['status'],
        'best_single_arm': d.get('best_single_arm', ''),
        'best_single_selection_split':
            ('valid' if strat == 'best_single' and r['status'] == ST_OK else ''),
        'selected_gamma': d.get('selected_gamma', d.get('gamma', '')),
        'selected_eta': d.get('selected_eta', d.get('eta', '')),
        'selection_split': d.get('selection_split', ''),
        'missing_input': r['missing_input'],
        'producing_command': r['producing_command'],
        'notes': ''})
    if pu is not None:
        row.update({k: repr(float(pu[k].mean())) for k in METRIC_KEYS})
    else:
        row.update({k: '' for k in METRIC_KEYS})
    return row


def common_detail(U, dumps):
    return {
        'n_users': int(U[0].shape[0]),
        'n_users_target_in_union': int((U[3] >= 0).sum()),
        'mean_union_size': float((U[0] >= 0).sum(1).mean()),
        'mean_candidates_per_arm': float(np.mean(
            [np.asarray(d['candidate_item_ids'] >= 0).sum(1).mean() for d in dumps]))}


def print_table(rows, title):
    out = [f'{title} rows ({len(rows)}):',
           f'  {"dataset":32s} {"scope":10s} {"strategy":20s} {"status":28s} '
           f'{"NDCG@20":>11s} {"Recall@20":>11s} {"n_users":>7s}']
    for r in rows:
        n20, r20 = r.get('NDCG@20', ''), r.get('Recall@20', '')
        out.append(f'  {r["dataset"]:32s} {r["scope"]:10s} {r["strategy"]:20s} '
                   f'{r["status"]:28s} '
                   f'{(f"{float(n20):.6f}" if n20 else "-"):>11s} '
                   f'{(f"{float(r20):.6f}" if r20 else "-"):>11s} '
                   f'{str(r.get("n_users", "")):>7s}')
    return '\n'.join(out)


def run(args) -> int:
    t0 = time.time()
    fa, p5 = import_historical()

    # An intentional revision of an already-recorded cell (a STATUS change, not a
    # re-measurement) must be declared, never inferred: the merge guard refuses
    # conflicting measurements and that guard must stay strict. Declaring the
    # superseded run ids here records WHY the aggregate changed while leaving every
    # superseded run directory on disk, untouched and auditable.
    if getattr(args, 'supersede', ''):
        ids = [x.strip() for x in args.supersede.split(',') if x.strip()]
        if not ids:
            raise SystemExit('--supersede given but no run ids parsed')
        if not getattr(args, 'supersede_reason', ''):
            raise SystemExit('--supersede requires --supersede-reason')
        sp = M5DIR / 'SUPERSEDED.json'
        cur = json.loads(sp.read_text()) if sp.exists() else {'superseded': {}}
        cur.setdefault('superseded', {})
        for i in ids:
            if not (M5DIR / 'runs' / i).is_dir():
                raise SystemExit(f'--supersede: no such run directory: {i}')
            cur['superseded'][i] = {
                'reason': args.supersede_reason,
                'at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'superseded_by_tag': args.tag,
            }
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(cur, indent=2) + '\n')
        print(f'recorded {len(ids)} superseded run(s) in {sp}')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + args.tag
    if args.dry_run:
        run_id += '_DRYRUN'
    run_dir = M5DIR / 'runs' / run_id

    gpu_info = probe_gpu()
    host_procs = scan_host_processes()
    busy, reasons = assess_contention(gpu_info, args.gpu_mem_limit_mib, host_procs)
    print('=' * 78)
    print(f'build_m5_rq5.py  run_id={run_id}')
    print(f'  datasets {args.datasets}')
    print(f'  scopes   {args.scopes}')
    print(f'  affinity {args.affinity}')
    print(f'  mode     {"DRY-RUN (full CPU pipeline, writes nothing)" if args.dry_run else "RUN"}')
    print('=' * 78)
    report_gpu_state(gpu_info, busy, reasons, args.gpu_mem_limit_mib)
    print()

    # The delivered pipeline is CPU-only, so a busy device does not block it.  A GPU
    # stage (candidate NLL / CFLA) would, and --gpu-stages is the only way to ask
    # for one; that path refuses outright while the device is busy.
    if args.gpu_stages and busy:
        if args.allow_busy_gpu:
            print('WARNING: --gpu-stages with --allow-busy-gpu on a BUSY device; any '
                  'number taken now measures contention.  Recorded in RUN_METADATA.')
        else:
            refuse_gpu(reasons, args.gpu_mem_limit_mib)

    report = Report()
    validate_only(args, report)
    found = sum(1 for r in report.rows if r['status'] == 'FOUND')
    print(f'input validation: {len(report.rows)} checks, {found} found')
    if not report.ok():
        for m in report.fatal:
            print(f'  FATAL {m}')
        return 2

    valid_rows, test_rows, boot_rows = [], [], []
    per_dataset = {}

    for dataset in args.datasets:
        print()
        print('-' * 78)
        print(f'DATASET {dataset}')
        print('-' * 78)
        ds_info = {'dataset': dataset, 'scopes': {}}

        # A dataset whose five-arm pool cannot be formed is reported as PENDING for
        # EVERY scope, and nothing is measured for it in this run.  That is
        # deliberate: the campaign must be able to re-run the driver once the
        # missing arm lands, and the aggregate merge refuses two DIFFERENT
        # measurements of one cell.  Measuring the spectral4 control now and the m5
        # pool later would collide on the spectral4 cells, so a pending dataset is
        # left entirely to the run that can do it properly.
        pending_missing = None
        if 'm5' in args.scopes:
            for split in ('valid', 'test'):
                if not rqvae5_dump_path(dataset, split).exists():
                    pending_missing = rqvae5_dump_path(dataset, split)
                    break
        if pending_missing is not None:
            print(f'  PENDING_MISSING_INPUT: {pending_missing} '
                  f'(RQ-VAE5 training still queued) -- every scope of {dataset} is '
                  f'deferred to the run that can complete it')
            for scope in args.scopes:
                for sp in ('valid', 'test'):
                    tgt = valid_rows if sp == 'valid' else test_rows
                    for strat in STRATEGIES:
                        tgt.append(status_row(
                            run_id, dataset, sp, args.affinity, scope, strat,
                            ST_PENDING, pending_missing, '',
                            'dataset pending: the RQ-VAE5 arm of this dataset is not '
                            'built yet; re-run the driver to add this dataset '
                            'without disturbing earlier output'))
                ds_info['scopes'][scope] = {'status': ST_PENDING,
                                            'missing_input': str(pending_missing)}
            per_dataset[dataset] = ds_info
            continue

        q_raw = fa.structure_q(dataset, args.affinity)
        q_norm_s4 = None if q_raw is None else fa.zscore(q_raw)
        if q_raw is None:
            print('  structure_q: MISSING -> structure_aware / valid_grid_search '
                  'blocked on every scope')
        else:
            print(f'  structure_q (spectral arms, k=2..5): '
                  f'{np.asarray(q_raw).round(6).tolist()}')

        for scope in args.scopes:
            print(f'  scope {scope}: arms={ARMS[scope]} '
                  f'lengths={arm_length_label(scope)}')

            dumps_v, prov_v = load_scope_inputs(fa, dataset, args.affinity, 'valid',
                                                scope)
            if dumps_v is None:
                print(f'    PENDING_MISSING_INPUT: {prov_v} (RQ-VAE5 training queued)')
                for split in ('valid', 'test'):
                    tgt = valid_rows if split == 'valid' else test_rows
                    for strat in STRATEGIES:
                        tgt.append(status_row(
                            run_id, dataset, split, args.affinity, scope, strat,
                            ST_PENDING, prov_v, '',
                            'dataset pending: the RQ-VAE5 arm of this dataset is not '
                            'built yet; re-run to add its rows'))
                ds_info['scopes'][scope] = {'status': ST_PENDING,
                                            'missing_input': str(prov_v)}
                continue

            dumps_t, prov_t = load_scope_inputs(fa, dataset, args.affinity, 'test',
                                                scope)
            if dumps_t is None:
                print(f'    PENDING_MISSING_INPUT: {prov_t}')
                for split in ('valid', 'test'):
                    tgt = valid_rows if split == 'valid' else test_rows
                    for strat in STRATEGIES:
                        tgt.append(status_row(
                            run_id, dataset, split, args.affinity, scope, strat,
                            ST_PENDING, prov_t, '',
                            'test-split beams missing; re-run to add its rows'))
                ds_info['scopes'][scope] = {'status': ST_PENDING,
                                            'missing_input': str(prov_t)}
                continue

            set_nk(fa, p5, scope)
            unions = {'valid': build_scope_union(fa, dumps_v, scope),
                      'test': build_scope_union(fa, dumps_t, scope)}
            confs = {'valid': scope_confidence(fa, dumps_v, scope),
                     'test': scope_confidence(fa, dumps_t, scope)}
            for nm, U in unions.items():
                if not np.isfinite(U[1]).all():
                    raise AssertionError(f'{dataset}/{scope}/{nm}: union scores NaN')
            print(f'    users valid/test {unions["valid"][0].shape[0]}/'
                  f'{unions["test"][0].shape[0]}   '
                  f'mean union size (test) '
                  f'{float((unions["test"][0] >= 0).sum(1).mean()):.2f}')

            q_norm = q_norm_s4 if scope == 'spectral4' else None
            nll_v = rqvae5_nll_path(dataset, 'valid')
            nll_t = rqvae5_nll_path(dataset, 'test')
            if scope == 'm5':
                strategy_status = {
                    'structure_aware': (
                        ST_EXCLUDED_BY_DECISION, 'operator decision 2026-09-17',
                        'operator decision (2026-09-17): the RQ-VAE arm does NOT '
                        'participate in structure-aware weighting. This is a scope '
                        'decision, not a missing input. No certified structure Q can '
                        'exist for the RQ-VAE5 arm anyway: '
                        f'{rqvae5_dim_groups_path(dataset)} is undefined because an '
                        'RQ-VAE level is a learned residual code over the FULL '
                        'embedding, not a partition of the 768 embedding dimensions '
                        'that compute_structure_weights.py needs for Sep/Bal. The '
                        'structure-aware ablation is therefore reported on the '
                        'four-arm spectral4 control pool, where Q IS defined and '
                        'which reproduces the frozen historical ablation exactly.'),
                    'valid_grid_search': (
                        ST_EXCLUDED_BY_DECISION, 'operator decision 2026-09-17',
                        'excluded with structure_aware: the (gamma, eta) grid '
                        'multiplies Q by gamma, so gamma != 0 is inadmissible for a '
                        'pool containing the RQ-VAE arm. On the admissible slice '
                        'gamma = 0 the grid degenerates to eta-only, i.e. exactly the '
                        'confidence_only row reported above, so no M5 information is '
                        'lost. Note the four-arm control selects gamma = 0 on EVERY '
                        'dataset, so the structure term is empirically inert there '
                        'as well.'),
                    'cfla': (ST_BLOCKED_NO_NLL, f'{nll_v} ; {nll_t}',
                             cfla_producing_command(args.datasets))}
            else:
                strategy_status = {
                    'cfla': (ST_NOT_IN_SCOPE,
                             'the deliverable is the five-arm M5; CFLA on the '
                             'four-arm spectral4 control pool is out of scope for '
                             'this run even though its inputs exist '
                             '(outputs/multifactor_nll_repair/{validation,test}/'
                             'spectral4/*_candidate_nll.parquet plus the '
                             'validation_nll_scalers.json spectral4 block)',
                             'not a missing-input blocker: run a CFLA driver over '
                             'the repaired spectral4 candidate-NLL parquets if the '
                             'four-arm control is wanted')}

            res = evaluate_strategies(fa, p5, unions, confs, q_norm, scope,
                                      strategy_status)

            cv = common_detail(unions['valid'], dumps_v)
            ct = common_detail(unions['test'], dumps_t)
            for scope_base in (cv, ct):
                scope_base.update({'pool': POOL_LABEL[scope],
                                   'arms': '|'.join(ARMS[scope]),
                                   'n_arms': len(ARMS[scope]),
                                   'arm_sid_lengths': arm_length_label(scope),
                                   'arm_semantic_lengths': arm_semantic_label(scope),
                                   'mixed_token_budgets': True,
                                   'length_matched_L6': False})
            for strat in STRATEGIES:
                r = res.get(strat)
                if r is None:
                    continue
                valid_rows.append(row_for(run_id, dataset, args.affinity, scope,
                                          strat, 'valid', r, cv))
                test_rows.append(row_for(run_id, dataset, args.affinity, scope,
                                         strat, 'test', r, ct))

            boot_rows.extend(bootstrap_pairs(p5, run_id, dataset, args.affinity,
                                             scope, res))
            if scope == 'm5':
                boot_rows.extend(cfla_rows_blocked(
                    run_id, dataset, args.affinity, scope, ST_BLOCKED_NO_NLL,
                    f'{nll_v} ; {nll_t}', cfla_producing_command(args.datasets)))

            print('    best_single on VALID: '
                  + ', '.join(f'{a}={v:.6f}' for a, v in
                              res['_single']['valid_NDCG@20'].items())
                  + f'  -> {res["_single"]["best_arm"]}')
            KEY20 = 'NDCG@20'
            for strat in ('uniform', 'confidence_only', 'valid_grid_search',
                          'structure_aware', 'cfla'):
                r = res[strat]
                v = r['per_user']['valid']
                t = r['per_user']['test']
                vs = '-' if v is None else format(float(v[KEY20].mean()), '.6f')
                ts = '-' if t is None else format(float(t[KEY20].mean()), '.6f')
                print(f'    {strat:20s} {r["status"]:28s} '
                      f'valid_NDCG@20={vs:>9s} test_NDCG@20={ts:>9s}')

            ds_info['scopes'][scope] = {
                'status': ST_OK, 'arms': list(ARMS[scope]),
                'arm_sid_lengths': arm_length_label(scope),
                'q_raw': None if q_raw is None else [float(x) for x in np.asarray(q_raw)],
                'q_norm': None if q_norm is None else [float(x) for x in np.asarray(q_norm)],
                'provenance_valid': prov_v,
                'provenance_test': prov_t,
                'valid_NDCG@20_by_arm': res['_single']['valid_NDCG@20'],
                'best_single_arm_selected_on_valid': res['_single']['best_arm'],
                'strategies': {s: {'status': res[s]['status'],
                                   'missing_input': res[s]['missing_input'],
                                   'detail': {k: v for k, v in
                                              res[s]['detail'].items()
                                              if k != 'valid_grid'},
                                   'valid': None if res[s]['per_user']['valid'] is None
                                   else {k: repr(float(res[s]['per_user']['valid'][k].mean()))
                                         for k in METRIC_KEYS},
                                   'test': None if res[s]['per_user']['test'] is None
                                   else {k: repr(float(res[s]['per_user']['test'][k].mean()))
                                         for k in METRIC_KEYS}}
                               for s in res if not s.startswith('_')}}
            if scope == 'spectral4':
                ds_info['scopes'][scope]['valid_grid_table_size'] = len(
                    res['valid_grid_search']['detail']['valid_grid'])

        per_dataset[dataset] = ds_info
        print(f'  done {dataset}  ({time.time() - t0:.1f}s elapsed)')

    if args.dry_run:
        print()
        print('=' * 78)
        print('DRY-RUN: every input was validated and the whole CPU pipeline ran. '
              'Nothing was written.')
        print('A real run would create:')
        print(f'  {M5DIR}/runs/{run_id}/VALID.csv       ({len(valid_rows)} rows)')
        print(f'  {M5DIR}/runs/{run_id}/TEST.csv        ({len(test_rows)} rows)')
        print(f'  {M5DIR}/runs/{run_id}/BOOTSTRAP.csv   ({len(boot_rows)} rows)')
        print(f'  {M5DIR}/runs/{run_id}/BOOTSTRAP.json')
        print(f'  {M5DIR}/runs/{run_id}/RUN_METADATA.json')
        print(f'  {M5DIR}/VALID.csv, TEST.csv, BOOTSTRAP.csv, BOOTSTRAP.json, '
              f'RUN_METADATA.json  (aggregate merged over every run directory)')
        print('=' * 78)
        print()
        print(print_table(valid_rows, 'VALID'))
        print()
        print(print_table(test_rows, 'TEST'))
        print()
        print('bootstrap rows:')
        for r in boot_rows[:12]:
            print(f'  {r["dataset"]:32s} {r["comparison"]:32s} {r["metric"]:9s} '
                  f'{r["status"]:26s} delta={r["delta"] or "-"} '
                  f'p={r["p_value"] or "-"}')
        print(f'  ... {len(boot_rows)} rows total')
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)   # a run dir is never reused
    write_csv(run_dir / 'VALID.csv', VALID_FIELDS, valid_rows)
    write_csv(run_dir / 'TEST.csv', VALID_FIELDS, test_rows)
    write_csv(run_dir / 'BOOTSTRAP.csv', BOOT_FIELDS, boot_rows)
    write_json(run_dir / 'BOOTSTRAP.json',
               {'run_id': run_id, 'n_boot': p5.N_BOOT, 'seed': p5.RNG_SEED,
                'fields': BOOT_FIELDS, 'rows': boot_rows})
    write_json(run_dir / 'RUN_METADATA.json',
               build_metadata(args, run_id, run_dir, report, gpu_info, busy, reasons,
                              per_dataset, valid_rows, test_rows, boot_rows, t0))

    nv, ns = merge_aggregate(M5DIR / 'VALID.csv', VALID_FIELDS,
                             ('dataset', 'split', 'scope', 'strategy'), run_id)
    merge_aggregate(M5DIR / 'TEST.csv', VALID_FIELDS,
                    ('dataset', 'split', 'scope', 'strategy'), run_id)
    nb, _ = merge_aggregate(M5DIR / 'BOOTSTRAP.csv', BOOT_FIELDS,
                            ('dataset', 'scope', 'comparison', 'metric'), run_id)
    write_json(M5DIR / 'BOOTSTRAP.json',
               {'aggregate': True, 'generated': utc_now(),
                'runs': [p.name for p in sorted((M5DIR / 'runs').glob('*'))
                         if p.is_dir()],
                'n_boot': p5.N_BOOT, 'seed': p5.RNG_SEED, 'fields': BOOT_FIELDS,
                'rows': read_csv_rows(M5DIR / 'BOOTSTRAP.csv')})
    write_json(M5DIR / 'RUN_METADATA.json',
               {'aggregate': True, 'generated': utc_now(),
                'note': 'merge of every run directory under m5_rq5/runs/; each run '
                        'directory is immutable and keeps its own RUN_METADATA.json',
                'runs': {p.name: json.loads((p / 'RUN_METADATA.json').read_text())
                         for p in sorted((M5DIR / 'runs').glob('*'))
                         if (p / 'RUN_METADATA.json').exists()}})

    print()
    print(f'wrote {run_dir.relative_to(OUT)}  (VALID {len(valid_rows)} rows, '
          f'TEST {len(test_rows)} rows, BOOTSTRAP {len(boot_rows)} rows)')
    print(f'aggregate {M5DIR.relative_to(OUT)}/VALID.csv: {nv} rows from {ns} run(s); '
          f'BOOTSTRAP.csv: {nb} rows')
    return 0


def status_row(run_id, dataset, split, affinity, scope, strategy, status, missing,
               command, notes):
    row = {k: '' for k in VALID_FIELDS}
    row.update({'run_id': run_id, 'dataset': dataset, 'split': split,
                'affinity': affinity, 'scope': scope, 'strategy': strategy,
                'status': status, 'missing_input': str(missing),
                'producing_command': command, 'notes': notes,
                'pool': POOL_LABEL[scope],
                'arms': '|'.join(ARMS[scope]), 'n_arms': len(ARMS[scope]),
                'arm_sid_lengths': arm_length_label(scope),
                'arm_semantic_lengths': arm_semantic_label(scope),
                'mixed_token_budgets': True, 'length_matched_L6': False})
    return row


# ===========================================================================
# cli
# ===========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='PHASE 8-10: new M5 (Spectral K2-K5 + RQ-VAE5) fusion, '
                    'VALID/TEST ablation, paired user-level bootstrap.')
    ap.add_argument('--datasets', default=','.join(DATASETS),
                    help='comma-separated subset of ' + ','.join(DATASETS))
    ap.add_argument('--scopes', default='m5,spectral4',
                    help='comma-separated subset of {m5,spectral4}')
    ap.add_argument('--affinity', default=CANONICAL_AFFINITY)
    ap.add_argument('--dry-run', action='store_true',
                    help='do ALL the CPU-only work and validate every input path, '
                         'then write nothing')
    ap.add_argument('--check', action='store_true',
                    help='validation only (fast preflight); exit 0 if every required '
                         'beam dump for the selected datasets exists')
    ap.add_argument('--gpu-stages', action='store_true',
                    help='permit the GPU-only candidate-NLL/CFLA stage (refuses while '
                         'the device is busy)')
    ap.add_argument('--allow-busy-gpu', action='store_true')
    ap.add_argument('--gpu-mem-limit-mib', type=float,
                    default=DEFAULT_GPU_MEM_LIMIT_MIB)
    ap.add_argument('--tag', default='m5rq5')
    ap.add_argument('--supersede', default='',
                    help='comma-separated run ids to EXCLUDE from the aggregate '
                         'merge as an intentional revision. Requires '
                         '--supersede-reason. The run directories are kept on disk; '
                         'the exclusion is recorded in m5_rq5/SUPERSEDED.json.')
    ap.add_argument('--supersede-reason', default='',
                    help='why those runs are superseded (recorded verbatim).')
    a = ap.parse_args(argv)
    a.datasets = [d.strip() for d in a.datasets.split(',') if d.strip()]
    a.scopes = [s.strip() for s in a.scopes.split(',') if s.strip()]
    for d in a.datasets:
        if d not in DATASETS:
            ap.error(f'unknown dataset {d!r}')
    for s in a.scopes:
        if s not in SCOPES:
            ap.error(f'unknown scope {s!r}')
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check:
        report = Report()
        validate_only(args, report)
        missing = [r for r in report.rows
                   if r['role'].startswith('beam:') and r['status'] != 'FOUND']
        for r in missing:
            print(f'MISSING {r["dataset"]:32s} {r["scope"]:10s} {r["split"]:5s} '
                  f'{r["role"]:22s} {r["path"]}')
        print(f'checked {len(report.rows)} paths, {len(missing)} missing')
        return 0 if not missing else 1
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
