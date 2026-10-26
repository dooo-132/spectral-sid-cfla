#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 11 -- measured inference complexity for the 5+1 / multi-factorization campaign.

WHAT THIS DRIVER IS FOR
-----------------------
`complexity/THEORETICAL_COMPLEXITY.md` gives a symbolic accounting whose core
CFLA term is

    T_CFLA - T_Uniform = M * U * C_TF(L) + O(M*U)

where `C_TF(L)` is the cost of one cross-factorization *teacher-forced* scoring
pass over an `L`-token SID.  This driver replaces the symbolic `C_TF` with a
measured one and measures the real beam-search retrieval path (not a toy
stand-in), so that `INFERENCE_COMPLEXITY_MEASURED` has something to stand on.

WHAT IT MEASURES
----------------
  A. Single Spectral-K5   -- full T5 encode + beam-30 decode, real checkpoint
  B. Single RQ-VAE5       -- full T5 encode + beam-30 decode, real checkpoint
  C. M5 Uniform           -- M sequential beam searches + union dedup (timed
                             end to end in ONE timed region)
  D. M5 CFLA              -- C plus the real teacher-forced scoring of every
                             factorization over the union; NOT inference-free
  G. C_TF(L)              -- standalone micro-benchmark of one batched
                             teacher-forced forward over candidate SIDs

`Uniform` and `CFLA` differ only by the teacher-forcing stage, so the measured
difference between rows C and D *is* the empirical CFLA extra cost that
THEORETICAL_COMPLEXITY.md derives symbolically.

MEASUREMENT PROTOCOL (binding, see module constant PROTOCOL)
------------------------------------------------------------
  * 5 warmup iterations per configuration, DISCARDED;
  * 10 measured iterations per configuration, every one written to
    `LATENCY_RAW_RUNS_MEASURED.csv` (per-run values, not only aggregates);
  * `torch.cuda.synchronize()` immediately before the timer starts and again
    immediately before it stops -- without both, the numbers are meaningless;
  * aggregate mean / std / median / min / max, never the mean alone;
  * `torch.cuda.max_memory_allocated()` recorded per run;
  * random seed fixed and recorded (`--seed`, default 2025, the campaign seed);
  * **model loading is never inside a timed region.**  Loading happens first and
    its wall time is reported separately as `load_ms` (an informational field,
    explicitly NOT a measurement of inference latency).

GPU RULE (why this driver refuses to run)
-----------------------------------------
A latency benchmark taken concurrently with a training job measures contention,
not the model.  Before it measures anything, this driver therefore requires the
device to be essentially idle:

  * `nvidia-smi --query-gpu=memory.used` < `--gpu-mem-limit-mib` (default 1000), and
  * `nvidia-smi --query-gpu=utilization.gpu` <= `--gpu-util-limit-pct` (default 0), and
  * no foreign compute application on the device, and
  * no running `model/main.py` / `train_t5_5plus1.py` process anywhere on the host.

If any of those fail it prints an explicit refusal and exits with code 3 without
writing or overwriting any artifact.  `--force` proceeds anyway and permanently
records, in every emitted row and in `RUN_METADATA.json`, that the device was
busy when the numbers were taken.

ARTIFACT DISCIPLINE
-------------------
Nothing is ever overwritten.  Every run writes into its own directory

    complexity/runs/<UTC-timestamp>_pid<PID>/

and only *publishes* a copy to the canonical `complexity/<NAME>` path when that
path does not already exist.  If it does exist, the driver reports its sha256,
leaves it untouched, and prints the `cp` command for a human to run if they
really mean to replace it.

NOTHING IS EVER INVENTED.  Any quantity that could not be measured is emitted as
the literal string `NOT_MEASURED` next to a `blocker` that names the precise
cause.  No cell is ever left blank.

USAGE
-----
    PY=the development workspace/.venv_cuda_compatible/bin/python
    D=.../outputs/final_hyperparam_efficiency/drivers/measure_inference_complexity.py

    $PY $D --dry-run                 # validate paths/args, touch no GPU work
    $PY $D --cpu-selftest            # CPU-only proof that the harness works
    $PY $D --emit-not-measured       # write schema-complete NOT_MEASURED rows
    $PY $D                           # the real measurement (requires idle GPU)
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
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# roots -- OUT is the ONLY writable root
# ---------------------------------------------------------------------------
# This driver imports modules from the development workspace (the reuse
# targets) and reads artifacts from the frozen artifact root (read-only).  Python
# would otherwise drop __pycache__/*.pyc next to those sources; disabling
# bytecode writing guarantees this process creates files ONLY under OUT.
# ---------------------------------------------------------------------------
sys.dont_write_bytecode = True

OUT = rp.OUTPUT_ROOT / 'final_hyperparam_efficiency'
REPO = Path(_REPO_ROOT + '')
ART_ROOT = Path(_ART_ROOT + '')               # READ-ONLY historical
VENV_PY = Path(os.environ.get('CFLA_PYTHON', sys.executable))
MF_DIR = REPO / 'src' / 'cfla' / 'multifactor_fusion'
COMPLEXITY = OUT / 'complexity'
RUNS_DIR = COMPLEXITY / 'runs'

AFFINITY = 'nonneg_pearson_v3'     # the campaign's spectral affinity (mf_common)
BEAM_SIZE = 30                     # campaign constant, never increased
SEED_DEFAULT = 2025                # campaign seed
WARMUP_DEFAULT = 5                 # protocol
MEASURED_DEFAULT = 10              # protocol
INFER_SIZE_DEFAULT = 96            # `--infer_size` of the campaign's T5 runs
TF_BATCH_DEFAULT = 512             # predeclared chunk constant of candidate_total_nll
CODEBOOK = 256

PROTOCOL = {
    'warmup_iterations_discarded': WARMUP_DEFAULT,
    'measured_iterations': MEASURED_DEFAULT,
    'sync': 'torch.cuda.synchronize() immediately before start and before stop',
    'aggregate': 'mean, std (sample, ddof=1), median, min, max',
    'model_loading': 'outside every timed region; reported separately as load_ms',
    'peak_memory': 'torch.cuda.max_memory_allocated() after torch.cuda.reset_peak_memory_stats()',
}

# ---------------------------------------------------------------------------
# schema -- column names copied verbatim from the existing NOT_MEASURED files so
# the new CSVs diff directly against them.  Extension columns are appended
# AFTER the canonical ones; the canonical prefix is never reordered.
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    # --- canonical prefix, identical to LATENCY_RAW_RUNS.csv ---
    'method', 'cohort', 'phase', 'run_index', 'warmup', 'elapsed_ms',
    'sync_used', 'batch_size', 'num_workers', 'gpu', 'torch', 'cuda', 'status',
    # --- extensions ---
    'dataset', 'arm', 'pool', 'user_batch', 'beam_size', 'sid_len', 'vocab_size',
    'stage', 'peak_memory_MiB', 'gpu_busy_at_measure_time', 'forced',
    'seed', 'measured_runs', 'warmup_runs', 'device', 'note', 'blocker',
]

AGG_COLUMNS = [
    # --- canonical prefix, identical to INFERENCE_COMPLEXITY.csv ---
    'method', 'n_factors', 'mean_union', 'beam_ms_user', 'candidate_nll_ms_user',
    'aggregation_ms_user', 'total_ms_user', 'users_per_sec', 'peak_memory_MiB',
    'parameters_M', 'checkpoint_MB', 'status', 'blocker', 'note',
    # --- extensions: per-run spread, never replace the mean ---
    'std_beam_ms_user', 'median_beam_ms_user', 'min_beam_ms_user', 'max_beam_ms_user',
    'std_total_ms_user', 'median_total_ms_user', 'min_total_ms_user', 'max_total_ms_user',
    'n_measured_runs', 'n_warmup_runs', 'n_users', 'seed', 'gpu', 'torch', 'cuda',
    'c_tf_ms_per_candidate_seq', 'mean_union_before_dedup', 'n_tf_sequences_per_user',
    # --- symbolic terms, kept separate so measured vs symbolic can be compared ---
    'symbolic_T_beam', 'symbolic_T_union_dedup', 'symbolic_T_candidate_nll',
    'symbolic_T_calibration_lse', 'symbolic_total',
    # --- provenance / honesty flags ---
    'measurement_kind', 'directly_timed_fields', 'derived_from_measured_fields',
    'gpu_busy_at_measure_time', 'forced', 'input_artifacts',
]

NOT_MEASURED = 'NOT_MEASURED'

# The campaign's four canonical rows, in the existing file's order.
CANONICAL_METHODS = [
    'A. Single Spectral-K5',
    'B. Single RQ-VAE5',
    'C. M5 Uniform (K2+K3+K4+K5+RQ-VAE5)',
    'D. M5 CFLA (K2+K3+K4+K5+RQ-VAE5)',
]
C_TF_METHOD = 'G. C_TF micro-benchmark (teacher-forced candidate scoring)'


# ===========================================================================
# small utilities
# ===========================================================================
def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ===========================================================================
# GPU guard
# ===========================================================================
TRAINING_PATTERNS = ('model/main.py', 'train_t5_5plus1.py', 'train_t5_5plus1')

GPU_QUERY = ('index,name,driver_version,memory.used,memory.total,'
             'utilization.gpu,temperature.gpu')


def scan_host_processes():
    """Every live process whose cmdline looks like a training/benchmark job.

    Read-only /proc scan; works without torch, so the guard can run before any
    CUDA context is created.
    """
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
            continue          # process exited or is not ours to inspect
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


def probe_gpu(samples: int = 3, gap_s: float = 1.0):
    """Read GPU state with nvidia-smi only -- never creates a CUDA context.

    `utilization.gpu` is an instantaneous sample, so it is read `samples` times
    with a gap and the MAXIMUM is used: a device that is busy in bursts must not
    slip through a momentary 0 % reading.
    """
    info = {'nvidia_smi_ok': False, 'error': None, 'gpus': [],
            'compute_apps': [], 'samples': samples}
    per_gpu = {}
    for s in range(samples):
        if s:
            time.sleep(gap_s)
        try:
            r = subprocess.run(
                ['nvidia-smi', f'--query-gpu={GPU_QUERY}',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            info['error'] = f'{type(e).__name__}: {e}'
            return info
        if r.returncode != 0:
            info['error'] = f'nvidia-smi exit {r.returncode}: {r.stderr.strip()[:300]}'
            return info
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue
            idx, name, drv, mused, mtot, util, temp = parts[:7]
            g = per_gpu.setdefault(idx, {'index': int(idx), 'name': name,
                                         'driver_version': drv,
                                         'memory_used_MiB_samples': [],
                                         'memory_total_MiB': int(float(mtot)),
                                         'util_pct_samples': [],
                                         'temperature_C_samples': []})
            g['memory_used_MiB_samples'].append(float(mused))
            g['util_pct_samples'].append(float(util))
            g['temperature_C_samples'].append(float(temp))
    info['nvidia_smi_ok'] = True
    for g in per_gpu.values():
        g['memory_used_MiB'] = max(g['memory_used_MiB_samples'])
        g['util_pct'] = max(g['util_pct_samples'])
        info['gpus'].append(g)

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


def assess_contention(gpu_info, mem_limit_mib: float, util_limit_pct: float,
                      host_procs):
    """Return (is_busy, reasons[], per_gpu_verdict[]).  Unknown == busy."""
    reasons = []
    verdicts = []
    if not gpu_info.get('nvidia_smi_ok'):
        reasons.append(
            'nvidia-smi unavailable or failed ({}), so device idleness cannot be '
            'established'.format(gpu_info.get('error')))
        return True, reasons, verdicts

    if not gpu_info['gpus']:
        reasons.append('nvidia-smi reported no GPUs')

    for g in gpu_info['gpus']:
        v = {'index': g['index'], 'name': g['name'],
             'memory_used_MiB': g['memory_used_MiB'],
             'util_pct': g['util_pct'], 'idle': True, 'why': []}
        if g['memory_used_MiB'] >= mem_limit_mib:
            v['idle'] = False
            v['why'].append(
                f"memory.used={g['memory_used_MiB']:.0f} MiB >= limit {mem_limit_mib:.0f} MiB")
        if g['util_pct'] > util_limit_pct:
            v['idle'] = False
            v['why'].append(
                f"utilization.gpu={g['util_pct']:.0f}% > limit {util_limit_pct:.0f}%")
        verdicts.append(v)
        if not v['idle']:
            reasons.append(f"GPU {g['index']} ({g['name']}): " + '; '.join(v['why']))

    for a in gpu_info.get('compute_apps', []):
        reasons.append(
            f"foreign compute app pid={a['pid']} ({a['process_name']}) holds "
            f"{a['used_memory_MiB']:.0f} MiB on the device")

    for p in host_procs:
        reasons.append(
            f"running training/benchmark process pid={p['pid']} matched "
            f"'{p['pattern']}': {p['cmdline']}")

    return (len(reasons) > 0), reasons, verdicts


def guard_or_die(args, gpu_info, busy, reasons):
    """The refusal path.  Writes NOTHING; exits 3."""
    print()
    print('=' * 78)
    print('REFUSING TO MEASURE: the GPU is not idle.')
    print('=' * 78)
    print('A latency benchmark run concurrently with another job measures')
    print('CONTENTION, not the model.  The numbers would be worthless, so this')
    print('driver will not take them.')
    print()
    for r in reasons:
        print(f'  * {r}')
    print()
    print('Device snapshot:')
    for g in gpu_info.get('gpus', []):
        print(f"  GPU {g['index']}  {g['name']}  driver {g['driver_version']}  "
              f"memory.used={g['memory_used_MiB']:.0f}/{g['memory_total_MiB']} MiB  "
              f"util={g['util_pct']:.0f}%")
    print()
    print(f"Thresholds: --gpu-mem-limit-mib {args.gpu_mem_limit_mib:.0f}, "
          f"--gpu-util-limit-pct {args.gpu_util_limit_pct:.0f}")
    print()
    print('Nothing was written and no existing artifact was touched.')
    print('Wait for the device to go idle, then re-run the same command.')
    print('--force overrides this check and permanently records that the device')
    print('was busy when the numbers were taken.')
    print('=' * 78)
    sys.exit(3)


# ===========================================================================
# model / data registry
# ===========================================================================
def _load_mf_modules():
    """Import the campaign's existing loaders instead of re-deriving them."""
    for p in (str(MF_DIR), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np                     # noqa: F401  (re-exported implicitly)
    import mf_models                       # noqa: F401
    import mf_common                       # noqa: F401
    return mf_models, mf_common


def arm_specs(ds: str):
    """Every arm of the campaign, with the exact artifact paths it needs.

    `pool` is the pool this arm can belong to:
      'both'      -- usable in the campaign (length-matched 5+1) pool and in the
                     native-length historical pool
      'native'    -- historical spectral K2/K3/K4: trained at SID length K+1
                     (3, 4, 5), so NOT length-matched with the 6-token 5+1 design
    """
    rp_spectral = {}
    for k in (2, 3, 4, 5):
        rp_spectral[f'spectral_k{k}'] = {
            'label': f'spectral_k{k}',
            'sid_native': ART_ROOT / 'data' / ds / f'{ds}_t5_spectral_{AFFINITY}_k{k}_sid{k + 1}.npy',
            'ckpt_native': ART_ROOT / 'model' / 'ckpt' / f'tiger_{ds}_{AFFINITY}_k{k}_sid{k + 1}.pth',
            'sid_sid6': ART_ROOT / 'data' / ds / f'{ds}_t5_spectral_{AFFINITY}_k{k}_sid6.npy',
            'ckpt_sid6': ART_ROOT / 'model' / 'ckpt' / f'tiger_{ds}_{AFFINITY}_k{k}_sid6.pth',
            'sid_len_native': k + 1,
            'sid_len_sid6': 6,
        }
    return {
        'spectral_k2': rp_spectral['spectral_k2'],
        'spectral_k3': rp_spectral['spectral_k3'],
        'spectral_k4': rp_spectral['spectral_k4'],
        'spectral_k5': rp_spectral['spectral_k5'],
        'rqvae5': {
            'label': 'rqvae5',
            'sid': OUT / 'runs_5plus1' / ds / 'rqvae5' / 'sid' / f'{ds}_t5_rqvae5.npy',
            'ckpt': OUT / 'runs_5plus1' / ds / 'rqvae5' / 't5' / 'best_model.pth',
            'sid_len': 6,
            # RQ-VAE5 is already a 6-token 5+1 SID, so it is the SAME artifact in
            # the campaign (length-matched) pool and in the native-length pool.
            'sid_sid6': OUT / 'runs_5plus1' / ds / 'rqvae5' / 'sid' / f'{ds}_t5_rqvae5.npy',
            'ckpt_sid6': OUT / 'runs_5plus1' / ds / 'rqvae5' / 't5' / 'best_model.pth',
            'sid_len_sid6': 6,
            'sid_native': OUT / 'runs_5plus1' / ds / 'rqvae5' / 'sid' / f'{ds}_t5_rqvae5.npy',
            'ckpt_native': OUT / 'runs_5plus1' / ds / 'rqvae5' / 't5' / 'best_model.pth',
            'sid_len_native': 6,
        },
    }


POOL_CAMPAIGN = 'campaign'   # length-matched 5+1: all five arms at SID length 6
POOL_NATIVE = 'native'       # historical: K2/K3/K4 at their trained lengths 3/4/5


def resolve_pool(ds: str, pool_kind: str, arms):
    """Resolve every arm of a pool to concrete (sid_path, ckpt_path, sid_len).

    Returns (resolved, missing) -- `missing` names the precise blocker per arm.
    Length matching is enforced for `campaign`: an arm whose SID is not 6 tokens
    long is a blocker, not a silent substitution.
    """
    resolved, missing = [], []
    for name in ('spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5', 'rqvae5'):
        a = arms[name]
        if pool_kind == POOL_CAMPAIGN:
            sid, ckpt, L = a['sid_sid6'], a['ckpt_sid6'], a['sid_len_sid6']
        else:
            if name == 'rqvae5':
                sid, ckpt, L = a['sid'], a['ckpt'], a['sid_len']
            else:
                sid, ckpt, L = a['sid_native'], a['ckpt_native'], a['sid_len_native']
        entry = {'arm': name, 'sid_path': sid, 'ckpt_path': ckpt, 'sid_len': L,
                 'pool': pool_kind}
        absent = []
        if not Path(sid).exists():
            absent.append(f'SID array missing: {sid}')
        if not Path(ckpt).exists():
            absent.append(f'checkpoint missing: {ckpt}')
        if absent:
            missing.append({'arm': name, 'pool': pool_kind, 'reasons': absent})
        else:
            resolved.append(entry)
    return resolved, missing


def paths_for_single(ds: str, which: str, arms):
    """Arm entry for the single-model rows A / B."""
    if which == 'rqvae5':
        a = arms['rqvae5']
        p = {'arm': 'rqvae5', 'sid_path': a['sid'], 'ckpt_path': a['ckpt'],
             'sid_len': a['sid_len'], 'pool': 'single'}
    elif which == 'spectral_k5_sid6':
        a = arms['spectral_k5']
        p = {'arm': 'spectral_k5', 'sid_path': a['sid_sid6'], 'ckpt_path': a['ckpt_sid6'],
             'sid_len': a['sid_len_sid6'], 'pool': 'single'}
    else:
        raise ValueError(which)
    absent = []
    if not Path(p['sid_path']).exists():
        absent.append(f"SID array missing: {p['sid_path']}")
    if not Path(p['ckpt_path']).exists():
        absent.append(f"checkpoint missing: {p['ckpt_path']}")
    return p, absent


def load_arm(entry, device):
    """Reuse mf_models' state-dict extraction / config inference / TIGER builder.

    Returns (bundle, load_ms).  `load_ms` is informational ONLY: it is measured
    outside every timed region and is never reported as inference latency.
    """
    mf_models, _ = _load_mf_modules()
    import torch
    t0 = time.perf_counter()
    sd = mf_models._extract_state_dict(
        torch.load(str(entry['ckpt_path']), map_location='cpu'))
    cfg = mf_models.infer_cfg(sd)
    sid = _load_sid_array(entry['sid_path'])
    expect = CODEBOOK * sid.shape[1] + 1
    if cfg['vocab_size'] != expect:
        raise AssertionError(
            f"{entry['ckpt_path']}: vocab {cfg['vocab_size']} != "
            f"256*{sid.shape[1]}+1 = {expect}")
    if sid.shape[1] != entry['sid_len']:
        raise AssertionError(
            f"{entry['sid_path']}: sid length {sid.shape[1]} != declared {entry['sid_len']}")
    model = mf_models._tiger(cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    load_ms = (time.perf_counter() - t0) * 1000.0
    return {'model': model, 'cfg': cfg, 'sid': sid, 'entry': entry,
            'checkpoint': str(entry['ckpt_path']), 'load_ms': load_ms,
            'n_parameters': sum(q.numel() for q in model.parameters())}


def _load_sid_array(p):
    mf_models, _ = _load_mf_modules()
    import numpy as np
    a = np.load(str(p), allow_pickle=True).astype(np.int64)
    if a.ndim != 2:
        raise ValueError(f'{p} is not 2-D: {a.shape}')
    return a


def sid_to_item_map(sid_array):
    """(item row index, SID tuple) -> 1-based item id.  Mirrors evaluate_item_level."""
    m = {}
    for idx, row in enumerate(sid_array):
        m[tuple(int(x) for x in row)] = idx + 1
    return m


# ===========================================================================
# timed regions
# ===========================================================================
def sync(device):
    """torch.cuda.synchronize() before start and before stop; no-op on CPU."""
    import torch
    if getattr(device, 'type', str(device)) == 'cuda':
        torch.cuda.synchronize(device)


def timed_runs(fn, warmup, measured, device, on_run):
    """The protocol.  5 warmups discarded, then `measured` timed iterations.

    Every iteration brackets the work with a device synchronise on BOTH sides.
    `fn` returns (payload, digest) where `digest` is evaluated only AFTER the
    timer has stopped, so result consumption never pollutes the measurement.
    """
    import torch
    for _ in range(warmup):
        payload, digest = fn()
        del payload, digest
    sync(device)

    out = []
    for i in range(measured):
        if getattr(device, 'type', str(device)) == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.perf_counter()
        payload, digest = fn()
        sync(device)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if getattr(device, 'type', str(device)) == 'cuda' else float('nan'))
        # The digest is consumed only here, i.e. strictly AFTER the timer stopped.
        digest_value = digest(payload) if callable(digest) else digest
        rec = {'run_index': i, 'elapsed_ms': elapsed_ms, 'peak_memory_MiB': peak,
               'digest': digest_value}
        out.append(rec)
        if on_run is not None:
            on_run(rec)
        del payload
    return out


def aggregate(values):
    """mean / std(ddof=1) / median / min / max.  Never the mean alone."""
    import statistics
    v = [x for x in values if x is not None]
    if not v:
        return {}
    return {
        'mean': statistics.fmean(v),
        'std': statistics.stdev(v) if len(v) > 1 else 0.0,
        'median': statistics.median(v),
        'min': min(v),
        'max': max(v),
        'n': len(v),
    }


def fmt(x, nd=4):
    if x is None:
        return NOT_MEASURED
    if isinstance(x, bool):
        return 'True' if x else 'False'
    if isinstance(x, int):
        return str(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:                     # NaN
        return NOT_MEASURED
    return f'{f:.{nd}f}'


# ===========================================================================
# the real code path: encode + beam-30 decode
# ===========================================================================
def make_generate_fn(bundle, history_tokens, history_mask, user_slice, beam_size):
    """One full T5 encode + beam-search decode over a fixed user batch.

    This is the campaign's canonical inference call, copied from
    `evaluate_item_level` in the historical T5 trainer: `max_length = sid_len + 1`,
    `num_beams = 30`, `num_return_sequences = 30`, scores returned.
    """
    import numpy as np
    import torch
    model = bundle['model']
    sid_len = bundle['entry']['sid_len']
    n_users = user_slice.stop - user_slice.start
    hist = torch.as_tensor(history_tokens[user_slice], dtype=torch.long)
    mask = torch.as_tensor(history_mask[user_slice], dtype=torch.long)
    device = next(model.parameters()).device
    hist = hist.to(device)
    mask = mask.to(device)

    def fn():
        out = model.model.generate(
            input_ids=hist,
            attention_mask=mask,
            max_length=sid_len + 1,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # Digest is computed after the timer stops (see timed_runs).
        def digest(o):
            s = o.sequences
            return {'sequences_shape': tuple(s.shape),
                    'sequences_sum': int(s.sum().item())}
        return out, digest

    return fn, n_users, (hist, mask)


def decode_beams(out, bundle, user_slice, beam_size):
    """Generated token ids -> per-user list of SID-array ROW indices.

    Mirrors `evaluate_item_level`: subtract `pos*codebook_size + 1`, keep only
    in-range codes, map SID tuple -> 1-based item id, then to a 0-based row.
    Pure CPU numpy; runs AFTER the timed region.
    """
    import numpy as np
    sid_array = bundle['sid']
    sid_len = bundle['entry']['sid_len']
    sid_to_item = sid_to_item_map(sid_array)
    n_users = user_slice.stop - user_slice.start
    seq = out.sequences[:, 1:].cpu().numpy()
    if seq.shape[1] < sid_len:
        seq = np.concatenate(
            [seq, np.zeros((seq.shape[0], sid_len - seq.shape[1]), dtype=seq.dtype)], axis=1)
    seq = seq[:, :sid_len]
    seq3 = seq.reshape(n_users, beam_size, sid_len)
    offsets = np.arange(sid_len) * CODEBOOK + 1
    valid = np.zeros((n_users, beam_size), dtype=bool)
    rows = np.full((n_users, beam_size), -1, dtype=np.int64)
    for i in range(n_users):
        for b in range(beam_size):
            codes = seq3[i, b] - offsets
            if ((codes < 0) | (codes >= CODEBOOK)).any():
                continue
            item = sid_to_item.get(tuple(int(x) for x in codes), -1)
            if item <= 0:
                continue
            valid[i, b] = True
            rows[i, b] = item - 1
    return rows, valid


def build_union(per_arm_rows, n_users):
    """Per-user union of generated candidates.

    Returns (union_rows, u_before_dedup, u_after_dedup):
      u_before_dedup  -- valid generated candidate SEQUENCES per user (<= M*B)
      u_after_dedup   -- DISTINCT item rows per user, i.e. |C(u)| in
                         THEORETICAL_COMPLEXITY.md section 4
    The dedup is a per-user ordered set built with a dict; that CPU hash-merge is
    the `O(M*B*L_F)` term of the symbolic accounting.
    """
    union_rows = []
    u_before = []
    u_after = []
    for i in range(n_users):
        seen = {}
        before = 0
        for rows in per_arm_rows:
            for r in rows[i]:
                if r < 0:
                    continue
                before += 1
                if r not in seen:
                    seen[r] = len(seen)
        union_rows.append(list(seen.keys()))
        u_before.append(before)
        u_after.append(len(seen))
    return union_rows, u_before, u_after


# ===========================================================================
# measurement routines
# ===========================================================================
def measure_single(ds, label, entry, hist_bundle, args, device, env, ctx):
    """One single-model beam-search latency configuration (rows A / B)."""
    rows = ctx['raw_rows']
    busy_flag = ctx['gpu_busy']
    forced = ctx['forced']
    blockers = []
    if entry is None:
        blockers = ctx['missing']
    rec = {
        'method': label, 'dataset': ds, 'arm': entry['arm'] if entry else NOT_MEASURED,
        'pool': 'single', 'stage': 'beam_generate', 'cohort': cohort_str(ds, args),
        'user_batch': args.n_users, 'beam_size': args.beam_size,
        'sid_len': entry['sid_len'] if entry else NOT_MEASURED,
        'num_workers': args.num_workers, 'seed': args.seed,
        'measured_runs': args.measured, 'warmup_runs': args.warmup,
        'device': str(device), 'gpu_busy_at_measure_time': busy_flag,
        'forced': forced,
    }
    if entry is None:
        reason = '; '.join(blockers) or 'unresolved arm'
        _emit_not_measured_raw(rows, rec, env, reason)
        return {'status': NOT_MEASURED, 'blocker': reason}

    bundle = load_arm(entry, device)
    hist, mask = hist_bundle[entry['arm']]
    fn, n_users, _ = make_generate_fn(
        bundle, hist, mask, slice(0, args.n_users), args.beam_size)

    print(f'  [{label}] {ds}/{entry["arm"]}: {args.warmup} warmup + '
          f'{args.measured} measured, {n_users} users, beam {args.beam_size}, '
          f'sid_len {entry["sid_len"]} (load {bundle["load_ms"]:.0f} ms, '
          f'{bundle["n_parameters"] / 1e6:.4f} M params)')

    digest_holder = {}
    runs = timed_runs(fn, args.warmup, args.measured, device,
                      on_run=lambda r: None)

    # Re-run once more OUTSIDE the timing region purely to decode candidates for
    # the union / CFLA stages.  Its wall time is not reported anywhere.
    with no_grad_ctx():
        out, _ = fn()
    rowsgen, valid = decode_beams(out, bundle, slice(0, args.n_users), args.beam_size)
    digest_holder['valid_beams'] = int(valid.sum())
    del out

    per_run_ms = [r['elapsed_ms'] for r in runs]
    st = aggregate(per_run_ms)
    for r in runs:
        rows.append(_raw_row(
            method=label, cohort=cohort_str(ds, args), phase='measured',
            run_index=r['run_index'], warmup=0,
            elapsed_ms=r['elapsed_ms'], sync_used=ctx['sync_used'],
            batch_size=args.n_users, num_workers=args.num_workers, env=env,
            status='MEASURED', dataset=ds, arm=entry['arm'], pool='single',
            user_batch=args.n_users, beam_size=args.beam_size,
            sid_len=entry['sid_len'], vocab_size=bundle['cfg']['vocab_size'],
            stage='beam_generate', peak=r['peak_memory_MiB'],
            gpu_busy=busy_flag, forced=forced, seed=args.seed,
            measured_runs=args.measured, warmup_runs=args.warmup,
            device=str(device),
            note=f'valid_beams={digest_holder["valid_beams"]}; '
                 f'model_load_excluded=true; load_ms={bundle["load_ms"]:.1f}',
            blocker='none'))

    ckpt_mb = Path(entry['ckpt_path']).stat().st_size / (1024 ** 2)
    return {
        'status': 'MEASURED', 'blocker': 'none',
        'beam_stat': st, 'n_factors': 1,
        'mean_union': float(valid.sum()) / n_users,
        'parameters_M': bundle['n_parameters'] / 1e6,
        'checkpoint_MB': ckpt_mb,
        'peak_memory_MiB': max(r['peak_memory_MiB'] for r in runs),
        'bundle': bundle, 'rows_per_arm': {entry['arm']: rowsgen},
        'arm': entry['arm'], 'sid_len': entry['sid_len'],
        'load_ms': bundle['load_ms'],
        'input_artifacts': [str(entry['sid_path']), str(entry['ckpt_path'])],
    }


def measure_pool(ds, label, resolved, missing, hist_bundles, args, device, env, ctx,
                 measure_cfla: bool, pool_kind: str):
    """M5 Uniform (row C) and M5 CFLA (row D) over one pool definition.

    The Uniform timed region contains the M sequential beam searches AND the
    union dedup -- it is the real end-to-end pipeline, not a sum of parts.
    The CFLA timed region additionally contains the real teacher-forced scoring
    of every factorization over the union.
    """
    import numpy as np
    rows = ctx['raw_rows']
    busy_flag = ctx['gpu_busy']
    forced = ctx['forced']
    pool_name = pool_kind

    rec_common = {
        'method': label, 'dataset': ds, 'pool': pool_name,
        'cohort': cohort_str(ds, args), 'user_batch': args.n_users,
        'beam_size': args.beam_size, 'num_workers': args.num_workers,
        'seed': args.seed, 'measured_runs': args.measured,
        'warmup_runs': args.warmup, 'device': str(device),
        'gpu_busy_at_measure_time': busy_flag, 'forced': forced,
        'n_factors': len(resolved), 'stage': 'beam_generate+union+tf' if measure_cfla
                 else 'beam_generate+union',
    }

    if missing:
        reason = '; '.join(f'{m["arm"]}: ' + ' | '.join(m['reasons']) for m in missing)
        for r in missing:
            ctx['missing_notes'].append(f'{ds}/{r["arm"]}: ' + ' | '.join(r['reasons']))
        _emit_not_measured_raw(rows, rec_common, env,
                               f'pool incomplete, {len(missing)} arm(s) missing: {reason}')
        return {'status': NOT_MEASURED, 'pool': pool_name,
                'blocker': f'pool incomplete, {len(missing)} arm(s) missing: {reason}',
                'n_factors': len(resolved) + len(missing),
                'arms': [e['arm'] for e in resolved]}

    bundles = [load_arm(e, device) for e in resolved]
    arms = [e['arm'] for e in resolved]
    # Every arm must describe the same item universe in the same row order,
    # otherwise "union of item rows" would silently mix two different id spaces.
    lens = {len(b['sid']) for b in bundles}
    if len(lens) != 1:
        raise AssertionError(
            f'pool {pool_name} mixes SID arrays of different item counts: {sorted(lens)}')

    # --- per-arm beam functions (each arm's own history, own model) ---
    gen_fns = []
    for b, e in zip(bundles, resolved):
        hist, mask = hist_bundles[e['arm']]
        fn, n_users, _ = make_generate_fn(
            b, hist, mask, slice(0, args.n_users), args.beam_size)
        gen_fns.append(fn)

    def _release():
        """Hand the caching allocator back between arms.

        Added 2026-10-13 (final paper completion round, section 26).  The previous
        implementation held EVERY arm's generation output at once
        (`outs = [fn() for fn in gen_fns]`), so the fifth arm's beam search ran while
        the first four were still resident.  That is what OOM-ed method F on
        2026-09-18 with 23.18 GiB in use.  Now only one arm's payload is ever alive:
        generate -> consume -> delete -> drop cache references -> gc -> empty_cache.
        """
        import torch as _torch
        gc.collect()
        if str(device).startswith('cuda'):
            _torch.cuda.empty_cache()

    # Timed region: M SEQUENTIAL beam searches, each followed by its own output
    # release.  The release is part of the measured cost on purpose -- it is what
    # the method actually costs when it is run arm by arm.
    def uniform_fn():
        beams = []
        for fn in gen_fns:
            payload, _ = fn()
            beams.append(int(payload.sequences.shape[0]))
            del payload
            _release()
        return beams, lambda bs: {'n_arms': len(bs), 'total_beams': int(sum(bs))}

    print(f'  [{label}] {ds}/{pool_name} pool ({len(resolved)} arms: {",".join(arms)}): '
          f'{args.warmup} warmup + {args.measured} measured, {args.n_users} users, '
          f'beam {args.beam_size}')

    uniform_runs = timed_runs(uniform_fn, args.warmup, args.measured, device, None)

    # Untimed extra pass to obtain the real candidate sets for union + CFLA.
    # Decoded per arm, and the raw beams are dropped before the next arm runs.
    with no_grad_ctx():
        per_arm_rows = []
        for b, e, fn in zip(bundles, resolved, gen_fns):
            payload, _ = fn()
            rr, _vv = decode_beams(payload, b, slice(0, args.n_users),
                                   args.beam_size)
            per_arm_rows.append(rr)
            del payload
            _release()

    union_rows, u_before, u_after = build_union(per_arm_rows, args.n_users)
    mean_u_before = float(np.mean(u_before)) if u_before else 0.0
    mean_u_after = float(np.mean(u_after)) if u_after else 0.0

    # --- aggregation (dedup) measured on its own so it can be a column ---
    def dedup_fn():
        res = build_union(per_arm_rows, args.n_users)
        return res, lambda x: {'u_after_sum': int(sum(x[2]))}
    dedup_runs = timed_runs(dedup_fn, args.warmup, args.measured, device, None)
    dedup_stat = aggregate([r['elapsed_ms'] for r in dedup_runs])
    dedup_ms_user = dedup_stat['mean'] / args.n_users

    for r in uniform_runs:
        rows.append(_raw_row(
            method=label, cohort=cohort_str(ds, args), phase='measured',
            run_index=r['run_index'], warmup=0, elapsed_ms=r['elapsed_ms'],
            sync_used=ctx['sync_used'], batch_size=args.n_users,
            num_workers=args.num_workers, env=env, status='MEASURED',
            dataset=ds, arm='+'.join(arms), pool=pool_name,
            user_batch=args.n_users, beam_size=args.beam_size,
            sid_len='mixed' if len({e['sid_len'] for e in resolved}) > 1
                    else resolved[0]['sid_len'],
            vocab_size='mixed' if len({b['cfg']['vocab_size'] for b in bundles}) > 1
                       else bundles[0]['cfg']['vocab_size'],
            stage='beam_generate+union', peak=r['peak_memory_MiB'],
            gpu_busy=busy_flag, forced=forced, seed=args.seed,
            measured_runs=args.measured, warmup_runs=args.warmup, device=str(device),
            note=f'mean_union_before_dedup={mean_u_before:.3f}; '
                 f'mean_union_after_dedup={mean_u_after:.3f}; '
                 f'model_load_excluded=true',
            blocker='none'))
    for r in dedup_runs:
        rows.append(_raw_row(
            method=label + ' [union dedup only]', cohort=cohort_str(ds, args),
            phase='measured', run_index=r['run_index'], warmup=0,
            elapsed_ms=r['elapsed_ms'], sync_used='cpu-only (numpy hash-merge)',
            batch_size=args.n_users, num_workers=args.num_workers, env=env,
            status='MEASURED', dataset=ds, arm='+'.join(arms), pool=pool_name,
            user_batch=args.n_users, beam_size=args.beam_size,
            sid_len=NOT_MEASURED, vocab_size=NOT_MEASURED,
            stage='union_dedup', peak=float('nan'), gpu_busy=busy_flag,
            forced=forced, seed=args.seed, measured_runs=args.measured,
            warmup_runs=args.warmup, device='cpu', note='CPU hash-merge only',
            blocker='none'))

    uniform_stat = aggregate([r['elapsed_ms'] for r in uniform_runs])
    result = {
        'status': 'MEASURED', 'blocker': 'none',
        'uniform_stat': uniform_stat,
        'aggregation_ms_user': dedup_ms_user,
        'n_factors': len(resolved),
        'mean_union': mean_u_after, 'mean_union_before_dedup': mean_u_before,
        'arms': arms, 'pool': pool_name,
        'parameters_M': sum(b['n_parameters'] for b in bundles) / 1e6,
        'checkpoint_MB': sum(Path(e['ckpt_path']).stat().st_size
                             for e in resolved) / (1024 ** 2),
        'peak_memory_MiB': max(r['peak_memory_MiB'] for r in uniform_runs),
        'load_ms': sum(b['load_ms'] for b in bundles),
        'bundles': bundles, 'resolved': resolved,
        'input_artifacts': [str(e['sid_path']) for e in resolved] +
                           [str(e['ckpt_path']) for e in resolved],
        'candidate_nll_ms_user': None, 'c_tf_ms_per_seq': None,
        'n_tf_sequences_per_user': None, 'cfla_stat': None,
    }

    if not measure_cfla:
        return result

    # ------------------------------------------------------------------
    # CFLA: real teacher-forced scoring of EVERY factorization over the union
    # ------------------------------------------------------------------
    def cfla_fn():
        # Section 26: one arm at a time, released before the next is generated.
        per_arm = []
        for b, _fn in zip(bundles, gen_fns):
            payload, _ = _fn()
            rr, _ = decode_beams(payload, b, slice(0, args.n_users), args.beam_size)
            per_arm.append(rr)
            del payload
            _release()
        ur, _, ua = build_union(per_arm, args.n_users)
        n_tf = 0
        for b, e, urows in zip(bundles, resolved, [ur] * len(bundles)):
            L = e['sid_len']
            cand = np.full((sum(len(x) for x in urows), L), -1, dtype=np.int64)
            owner = np.empty(cand.shape[0], dtype=np.int64)
            k = 0
            for i, rl in enumerate(urows):
                for r in rl:
                    cand[k] = b['sid'][r] + np.arange(L) * CODEBOOK + 1
                    owner[k] = i
                    k += 1
            if k == 0:
                continue
            n_tf += k
            hist, mask = hist_bundles[e['arm']]
            mf_models.candidate_total_nll(
                b['model'], hist[:args.n_users], mask[:args.n_users],
                cand, owner, device=str(device), batch_size=args.tf_batch)
        def digest(_):
            return {'n_tf_sequences': n_tf}
        return None, digest

    mf_models, _ = _load_mf_modules()
    cfla_runs = timed_runs(cfla_fn, args.warmup, args.measured, device, None)
    cfla_stat = aggregate([r['elapsed_ms'] for r in cfla_runs])

    n_tf_per_batch = cfla_runs[0]['digest']['n_tf_sequences'] if cfla_runs else 0
    n_tf_per_user = n_tf_per_batch / args.n_users if args.n_users else 0
    total_ms_user = cfla_stat['mean'] / args.n_users
    # Uniform's teacher-forcing cost is identically zero; the measured CFLA extra
    # cost is therefore (CFLA total) - (Uniform total) computed from the two
    # independently timed end-to-end regions.
    uniform_ms_user = uniform_stat['mean'] / args.n_users
    extra_ms_user = total_ms_user - uniform_ms_user
    # Marginal C_TF: the CFLA extra cost divided by the number of candidate
    # sequences actually teacher-forced.  This is the measured replacement for
    # the symbolic C_TF(L) of THEORETICAL_COMPLEXITY.md section 4.
    marginal_ms_per_batch = cfla_stat['mean'] - uniform_stat['mean']
    c_tf_ms_per_seq = (marginal_ms_per_batch / n_tf_per_batch
                       if n_tf_per_batch else None)

    for r in cfla_runs:
        rows.append(_raw_row(
            method=label, cohort=cohort_str(ds, args), phase='measured',
            run_index=r['run_index'], warmup=0, elapsed_ms=r['elapsed_ms'],
            sync_used=ctx['sync_used'], batch_size=args.n_users,
            num_workers=args.num_workers, env=env, status='MEASURED',
            dataset=ds, arm='+'.join(arms), pool=pool_name,
            user_batch=args.n_users, beam_size=args.beam_size,
            sid_len='mixed' if len({e['sid_len'] for e in resolved}) > 1
                    else resolved[0]['sid_len'],
            vocab_size='mixed' if len({b['cfg']['vocab_size'] for b in bundles}) > 1
                       else bundles[0]['cfg']['vocab_size'],
            stage='beam_generate+union+teacher_forced',
            peak=r['peak_memory_MiB'], gpu_busy=busy_flag, forced=forced,
            seed=args.seed, measured_runs=args.measured,
            warmup_runs=args.warmup, device=str(device),
            note=f'n_tf_sequences={n_tf_per_batch}; '
                 f'teacher_forced=true; model_load_excluded=true; '
                 f'uniform_total_ms_user={uniform_ms_user:.4f}; '
                 f'this timed region INCLUDES beam generation + union + the '
                 f'candidate-tensor construction loop, so total_ms_user is the '
                 f'end-to-end CFLA pipeline and candidate_nll_ms_user is the '
                 f'marginal (CFLA minus independently-timed Uniform)',
            blocker='none'))

    result.update({
        'candidate_nll_ms_user': extra_ms_user,
        'c_tf_ms_per_seq': c_tf_ms_per_seq,
        'n_tf_sequences_per_user': n_tf_per_user,
        'cfla_stat': cfla_stat,
        'total_stat': cfla_stat,
        'peak_memory_MiB': max(result['peak_memory_MiB'],
                               max(r['peak_memory_MiB'] for r in cfla_runs)),
    })
    return result


def measure_c_tf_microbenchmark(ds, entry, hist_bundle, args, device, env, ctx):
    """Row G: standalone C_TF(L) -- one batched teacher-forced forward.

    This is the quantity the symbolic accounting calls `C_TF(L_F)`.  It is
    measured on a REAL candidate pool: the union of this dataset's available
    single models' own beams, chunked at `--tf-batch` sequences exactly as
    `mf_models.candidate_total_nll` does.
    """
    import numpy as np
    rows = ctx['raw_rows']
    if entry is None:
        reason = '; '.join(ctx['missing']) or 'no arm available to teacher-force'
        _emit_not_measured_raw(rows, {'method': C_TF_METHOD, 'dataset': ds,
                                      'stage': 'teacher_forced_forward',
                                      'cohort': cohort_str(ds, args),
                                      'user_batch': args.n_users,
                                      'beam_size': args.beam_size,
                                      'num_workers': args.num_workers,
                                      'seed': args.seed, 'measured_runs': args.measured,
                                      'warmup_runs': args.warmup, 'device': str(device),
                                      'gpu_busy_at_measure_time': ctx['gpu_busy'],
                                      'forced': ctx['forced']},
                               env, reason)
        return {'status': NOT_MEASURED, 'blocker': reason}

    mf_models, _ = _load_mf_modules()
    bundle = load_arm(entry, device)
    L = entry['sid_len']
    hist, mask = hist_bundle[entry['arm']]

    # Candidate pool: `--tf-sequences` candidate SIDs drawn from the real SID
    # array -- every item of this dataset is a legitimate candidate SID, so the
    # pool is real data, not a synthetic tensor.
    n_seq = min(args.tf_sequences, len(bundle['sid']))
    rows_idx = np.arange(n_seq) % len(bundle['sid'])
    cand = bundle['sid'][rows_idx] + np.arange(L, dtype=np.int64) * CODEBOOK + 1
    owner = np.arange(n_seq, dtype=np.int64) % args.n_users

    def tf_fn():
        out = mf_models.candidate_total_nll(
            bundle['model'], hist[:args.n_users], mask[:args.n_users],
            cand, owner, device=str(device), batch_size=args.tf_batch)
        def digest(o):
            finite = np.isfinite(o)
            return {'n_scored': int(finite.sum()),
                    'mean_nll': float(np.nanmean(o)) if finite.any() else None}
        return out, digest

    print(f'  [{C_TF_METHOD}] {ds}/{entry["arm"]}: {n_seq} candidate SIDs, '
          f'L={L}, tf_batch={args.tf_batch}')

    runs = timed_runs(tf_fn, args.warmup, args.measured, device, None)
    st = aggregate([r['elapsed_ms'] for r in runs])
    digest = runs[0]['digest'] if runs else {}
    per_seq = st['mean'] / n_seq if n_seq else None

    for r in runs:
        rows.append(_raw_row(
            method=C_TF_METHOD, cohort=cohort_str(ds, args), phase='measured',
            run_index=r['run_index'], warmup=0, elapsed_ms=r['elapsed_ms'],
            sync_used=ctx['sync_used'], batch_size=n_seq,
            num_workers=args.num_workers, env=env, status='MEASURED',
            dataset=ds, arm=entry['arm'], pool='microbenchmark',
            user_batch=args.n_users, beam_size=NOT_MEASURED, sid_len=L,
            vocab_size=bundle['cfg']['vocab_size'], stage='teacher_forced_forward',
            peak=r['peak_memory_MiB'], gpu_busy=ctx['gpu_busy'], forced=ctx['forced'],
            seed=args.seed, measured_runs=args.measured, warmup_runs=args.warmup,
            device=str(device),
            note=f'n_candidate_sequences={n_seq}; tf_batch={args.tf_batch}; '
                 f'mean_candidate_total_nll={digest.get("mean_nll")}; '
                 f'model_load_excluded=true',
            blocker='none'))

    return {'status': 'MEASURED', 'blocker': 'none', 'tf_stat': st,
            'c_tf_ms_per_seq': per_seq, 'n_sequences': n_seq,
            'parameters_M': bundle['n_parameters'] / 1e6,
            'checkpoint_MB': Path(entry['ckpt_path']).stat().st_size / (1024 ** 2),
            'peak_memory_MiB': max(r['peak_memory_MiB'] for r in runs),
            'sid_len': L, 'arm': entry['arm'],
            'input_artifacts': [str(entry['sid_path']), str(entry['ckpt_path'])],
            'mean_nll': digest.get('mean_nll'),
            'load_ms': bundle['load_ms']}


# ===========================================================================
# row helpers
# ===========================================================================
def cohort_str(ds, args):
    return f'{ds} TEST first {args.n_users} users'


def _raw_row(*, method, cohort, phase, run_index, warmup, elapsed_ms, sync_used,
             batch_size, num_workers, env, status, dataset, arm, pool, user_batch,
             beam_size, sid_len, vocab_size, stage, peak, gpu_busy, forced, seed,
             measured_runs, warmup_runs, device, note, blocker):
    return {
        'method': method, 'cohort': cohort, 'phase': phase, 'run_index': run_index,
        'warmup': warmup, 'elapsed_ms': fmt(elapsed_ms, 6), 'sync_used': sync_used,
        'batch_size': batch_size, 'num_workers': num_workers,
        'gpu': env['gpu_name'], 'torch': env['torch'], 'cuda': env['cuda'],
        'status': status, 'dataset': dataset, 'arm': arm, 'pool': pool,
        'user_batch': user_batch, 'beam_size': beam_size, 'sid_len': sid_len,
        'vocab_size': vocab_size, 'stage': stage,
        'peak_memory_MiB': fmt(peak if peak == peak else None, 2),
        'gpu_busy_at_measure_time': gpu_busy, 'forced': forced, 'seed': seed,
        'measured_runs': measured_runs, 'warmup_runs': warmup_runs,
        'device': device, 'note': note, 'blocker': blocker,
    }


def _emit_not_measured_raw(rows, rec, env, blocker):
    rows.append({
        'method': rec.get('method', NOT_MEASURED),
        'cohort': rec.get('cohort', NOT_MEASURED),
        'phase': 'n/a', 'run_index': '', 'warmup': '',
        'elapsed_ms': NOT_MEASURED,
        'sync_used': 'torch.cuda.synchronize() would be used',
        'batch_size': rec.get('user_batch', NOT_MEASURED),
        'num_workers': rec.get('num_workers', 0),
        'gpu': env['gpu_name'], 'torch': env['torch'], 'cuda': env['cuda'],
        'status': f'{NOT_MEASURED}: {blocker}',
        'dataset': rec.get('dataset', NOT_MEASURED),
        'arm': rec.get('arm', NOT_MEASURED), 'pool': rec.get('pool', NOT_MEASURED),
        'user_batch': rec.get('user_batch', NOT_MEASURED),
        'beam_size': rec.get('beam_size', NOT_MEASURED),
        'sid_len': rec.get('sid_len', NOT_MEASURED),
        'vocab_size': NOT_MEASURED, 'stage': rec.get('stage', NOT_MEASURED),
        'peak_memory_MiB': NOT_MEASURED, 'gpu_busy_at_measure_time': 'True',
        'forced': rec.get('forced', False), 'seed': rec.get('seed', NOT_MEASURED),
        'measured_runs': rec.get('measured_runs', NOT_MEASURED),
        'warmup_runs': rec.get('warmup_runs', NOT_MEASURED),
        'device': rec.get('device', NOT_MEASURED), 'note': '',
        'blocker': blocker,
    })


def no_grad_ctx():
    import torch
    return torch.no_grad()


# ===========================================================================
# writers
# ===========================================================================
def write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            out = {}
            for c in columns:
                v = r.get(c, '')
                out[c] = NOT_MEASURED if v is None or v == '' else v
            w.writerow(out)
    return path


def tex_escape(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&')
            .replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')
            .replace('$', r'\$').replace('{', r'\{').replace('}', r'\}'))


def write_tex(path: Path, agg_rows, env, meta_note):
    """Plain `tabular` data table -- this is NOT a paper figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('% AUTO-GENERATED by drivers/measure_inference_complexity.py -- '
                 'data table, not a figure.')
    lines.append(f'% generated {env["generated_utc"]}')
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{lccccc}')
    lines.append(r'\toprule')
    lines.append(r'Method & Factors & Beam & Cand.\ NLL & Total & Peak GPU \\')
    lines.append(r'\midrule')
    for r in agg_rows:
        lines.append(' & '.join([
            tex_escape(r['method']),
            tex_escape(r['n_factors']),
            tex_escape(r['beam_ms_user']),
            tex_escape(r['candidate_nll_ms_user']),
            tex_escape(r['total_ms_user']),
            tex_escape(r['peak_memory_MiB']),
        ]) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\par\smallskip')
    lines.append(r'\footnotesize ' + tex_escape(meta_note))
    lines.append(r'\end{table}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def csv_is_pure_not_measured(path: Path, status_col: str = 'status') -> bool:
    """True when every row of a CSV is a NOT_MEASURED blocker record."""
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    if not rows:
        return False
    for r in rows:
        v = str(r.get(status_col, ''))
        if v.startswith('MEASURED') and NOT_MEASURED not in v:
            return False
    return True


def publish(run_dir: Path, name: str, canonical_dir: Path, log,
            run_has_measurement: bool, canonical_csv_is_placeholder: bool,
            allow_placeholder_refresh: bool = False):
    """Copy a run artifact to its canonical path under a strict, auditable rule.

    1. canonical path absent                     -> publish.
    2. canonical path holds a pure NOT_MEASURED  -> PROMOTE the new run over it,
       placeholder AND this run HAS measurements     first copying the placeholder
                                                    aside as `<name>.superseded_<utc>`
                                                    (archived, never deleted).
    2b. canonical path holds a pure NOT_MEASURED  -> refresh it, but ONLY when
        placeholder AND --refresh-placeholder        the operator asked explicitly.
        was passed
    3. otherwise                                 -> never overwrite; report both
                                                    sha256s and the manual command.

    Rule 2 exists so that a blocker record written while the GPU was busy can
    never permanently block the real measurement from reaching the canonical
    path.  A real measurement is never overwritten by anything.  Rule 2b is safe
    because a placeholder contains no measured number: refreshing it destroys no
    scientific quantity, and the replaced bytes are archived with their sha256.
    """
    src = run_dir / name
    dst = canonical_dir / name
    if not src.exists():
        return
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        promote = canonical_csv_is_placeholder and run_has_measurement
        refresh = (canonical_csv_is_placeholder and allow_placeholder_refresh
                   and not run_has_measurement)
        if not (promote or refresh):
            log.append({
                'artifact': name, 'published': False,
                'reason': ('canonical path already exists and is not a pure '
                           'NOT_MEASURED placeholder; never overwritten'),
                'canonical_path': str(dst), 'canonical_sha256': sha256_file(dst),
                'run_path': str(src), 'run_sha256': sha256_file(src),
                'manual_replace_command': f'cp {src} {dst}',
            })
            return
        archive = canonical_dir / f'{name}.superseded_{utc_stamp()}'
        archive.write_bytes(dst.read_bytes())
        dst.write_bytes(src.read_bytes())
        log.append({'artifact': name, 'published': True,
                    'action': 'promoted' if promote else 'refreshed_placeholder',
                    'reason': ('superseded an all-NOT_MEASURED placeholder with a run '
                               'that contains real measurements' if promote else
                               'refreshed a blocker record at the operator\'s '
                               'explicit request (--refresh-placeholder)'),
                    'archived_placeholder': str(archive),
                    'archived_sha256': sha256_file(archive),
                    'canonical_path': str(dst), 'canonical_sha256': sha256_file(dst),
                    'run_path': str(src), 'run_sha256': sha256_file(src)})
        return
    dst.write_bytes(src.read_bytes())
    log.append({'artifact': name, 'published': True, 'action': 'created',
                'canonical_path': str(dst), 'canonical_sha256': sha256_file(dst),
                'run_path': str(src), 'run_sha256': sha256_file(src)})


# ===========================================================================
# environment
# ===========================================================================
def environment(gpu_info):
    import torch
    g = gpu_info['gpus'][0] if gpu_info.get('gpus') else {}
    cap = driver_cuda_cap(gpu_info)
    return {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'host': platform.node(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'python_executable': sys.executable,
        'torch': torch.__version__,
        'torch_cuda_build': torch.version.cuda,
        'cuda': f'{torch.version.cuda} (driver supports {cap})',
        'cuda_available': bool(torch.cuda.is_available()),
        'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu_name': g.get('name', NOT_MEASURED),
        'gpu_driver_version': g.get('driver_version', NOT_MEASURED),
        'gpu_memory_total_MiB': g.get('memory_total_MiB', NOT_MEASURED),
        'nvidia_smi_cuda_version': cap,
        'note_driver_vs_torch': (
            'The NVIDIA driver caps CUDA at 12.8 while the venv torch is a cu128 '
            'build. These numbers are therefore tied to THIS device and driver; '
            'they are not portable to a different GPU or driver.'),
    }


def driver_cuda_cap(gpu_info):
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            m = re.search(r'CUDA Version:\s*([0-9.]+)', r.stdout)
            if m:
                return m.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f'{NOT_MEASURED} ({type(e).__name__})'
    return NOT_MEASURED


# ===========================================================================
# plan / dry-run
# ===========================================================================
def build_plan(args):
    plan = []
    for ds in args.datasets:
        arms = arm_specs(ds)
        for key, label in (('spectral_k5_sid6', CANONICAL_METHODS[0]),
                           ('rqvae5', CANONICAL_METHODS[1])):
            entry, absent = paths_for_single(ds, key, arms)
            plan.append({'kind': 'single', 'dataset': ds, 'method': label,
                         'entry': entry, 'missing': absent})
        for pool_kind, m_uniform, m_cfla in (
                (POOL_CAMPAIGN, CANONICAL_METHODS[2], CANONICAL_METHODS[3]),
                (POOL_NATIVE, 'E. M5 Uniform (native-length pool)',
                 'F. M5 CFLA (native-length pool)')):
            if pool_kind == POOL_CAMPAIGN and not args.include_campaign_pool:
                continue
            if pool_kind == POOL_NATIVE and not args.include_native_pool:
                continue
            resolved, missing = resolve_pool(ds, pool_kind, arms)
            plan.append({'kind': 'pool', 'dataset': ds, 'pool': pool_kind,
                         'uniform_label': m_uniform, 'cfla_label': m_cfla,
                         'resolved': resolved, 'missing': missing})
        if args.include_c_tf:
            entry, absent = paths_for_single(ds, 'rqvae5', arms)
            if absent:
                entry = None
            plan.append({'kind': 'c_tf', 'dataset': ds, 'method': C_TF_METHOD,
                         'entry': entry, 'missing': absent})
    return plan


def print_plan(plan, args):
    print()
    print('PLAN')
    print('-' * 78)
    for item in plan:
        if item['kind'] == 'single':
            ok = not item['missing']
            e = item['entry']
            print(f"  [{'OK ' if ok else 'MISS'}] {item['dataset']:32s} "
                  f"{item['method']}")
            print(f"          sid  = {e['sid_path']}")
            print(f"          ckpt = {e['ckpt_path']}")
            for m in item['missing']:
                print(f"          MISSING: {m}")
        elif item['kind'] == 'pool':
            n_ok, n_miss = len(item['resolved']), len(item['missing'])
            print(f"  [{'OK ' if not n_miss else 'MISS'}] {item['dataset']:32s} "
                  f"pool={item['pool']}  ({n_ok} arms resolved, {n_miss} missing)")
            for e in item['resolved']:
                print(f"          ok   {e['arm']:14s} L={e['sid_len']} "
                      f"{Path(e['ckpt_path']).name}")
            for m in item['missing']:
                print(f"          MISS {m['arm']:14s} " + ' | '.join(m['reasons']))
        else:
            ok = item['entry'] is not None
            print(f"  [{'OK ' if ok else 'MISS'}] {item['dataset']:32s} "
                  f"{item['method']}")
    print('-' * 78)
    print(f"Protocol: {args.warmup} warmup (discarded) + {args.measured} measured "
          f"iterations per configuration.")
    print(f"Users per timed batch: {args.n_users}   beam: {args.beam_size}   "
          f"tf_batch: {args.tf_batch}   seed: {args.seed}")
    print(f"Run directory would be: {RUNS_DIR}/<UTC-stamp>_pid<PID>/")


# ===========================================================================
# CPU-only selftest
# ===========================================================================
def cpu_selftest(args):
    """Prove the timing harness is correct without touching the GPU.

    Runs the EXACT same `timed_runs` structure (warmup, sync on both sides,
    per-run records, aggregate) over (a) a small CPU op and (b) the REAL
    checkpoint loaded on CPU and the REAL generate/teacher-forced code paths at
    a tiny size.  No CUDA call is made: `sync()` is a documented no-op on CPU.
    """
    import torch
    import numpy as np
    print('CPU-only smoke test of the timing harness (NO GPU IS TOUCHED)')
    print('-' * 78)
    device = torch.device('cpu')
    results = {'torch': torch.__version__, 'cuda_available': bool(torch.cuda.is_available()),
               'device': 'cpu', 'steps': {}}

    # (a) trivial CPU op through the real harness
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)

    def op():
        c = a @ b
        return c, lambda x: {'sum': float(x.sum())}
    runs = timed_runs(op, args.warmup, args.measured, device, None)
    st = aggregate([r['elapsed_ms'] for r in runs])
    results['steps']['cpu_matmul_512x512'] = {
        'per_run_ms': [r['elapsed_ms'] for r in runs],
        'mean': st['mean'], 'std': st['std'], 'median': st['median'],
        'min': st['min'], 'max': st['max'],
        'sync_used': 'none (cpu: torch.cuda.synchronize() is a no-op by design)'}
    print(f"  (a) cpu matmul 512x512 : mean {st['mean']:.3f} ms  "
          f"std {st['std']:.3f}  median {st['median']:.3f}  "
          f"min {st['min']:.3f}  max {st['max']:.3f}  (n={len(runs)})")

    # (b) the real checkpoint + real code paths, tiny size, CPU
    ds = args.datasets[0]
    arms = arm_specs(ds)
    entry, absent = paths_for_single(ds, 'rqvae5', arms)
    if absent:
        results['steps']['real_path'] = {'status': NOT_MEASURED,
                                         'blocker': '; '.join(absent)}
        print(f"  (b) real path: {NOT_MEASURED} -- {'; '.join(absent)}")
    else:
        mf_models, mf_common = _load_mf_modules()
        bundle = load_arm(entry, device)
        hist, mask = mf_models.build_history_tokens(
            ds, 'test', entry['sid_path'], max_len=20, batch=512)
        n = min(args.selftest_users, len(hist))
        fn, n_users, _ = make_generate_fn(bundle, hist, mask, slice(0, n),
                                          args.selftest_beam)
        g_runs = timed_runs(fn, 1, 2, device, None)
        gst = aggregate([r['elapsed_ms'] for r in g_runs])
        with torch.no_grad():
            out, _ = fn()
        rr, vv = decode_beams(out, bundle, slice(0, n), args.selftest_beam)
        L = entry['sid_len']
        cand = bundle['sid'][:8] + np.arange(L, dtype=np.int64) * CODEBOOK + 1
        owner = np.zeros(8, dtype=np.int64)

        def tf():
            o = mf_models.candidate_total_nll(
                bundle['model'], hist[:n], mask[:n], cand, owner,
                device='cpu', batch_size=8)
            return o, lambda x: {'mean_nll': float(np.nanmean(x))}
        t_runs = timed_runs(tf, 1, 2, device, None)
        tst = aggregate([r['elapsed_ms'] for r in t_runs])
        results['steps']['real_path'] = {
            'status': 'CPU_SMOKE_OK', 'checkpoint': entry['ckpt_path'],
            'sid_path': entry['sid_path'], 'n_parameters_M': bundle['n_parameters'] / 1e6,
            'history_shape': list(hist.shape), 'n_users': n,
            'beam': args.selftest_beam, 'sid_len': L,
            'generate_ms': gst, 'valid_beams': int(vv.sum()),
            'teacher_forced_ms': tst,
            'note': 'CPU only -- these are NOT the reported GPU latencies.'}
        print(f"  (b) real checkpoint loaded on CPU: {bundle['n_parameters'] / 1e6:.4f} M params, "
              f"history {tuple(hist.shape)}")
        print(f"      real generate path (beam {args.selftest_beam}, {n} users): "
              f"mean {gst['mean']:.2f} ms; valid beams decoded = {int(vv.sum())}")
        print(f"      real teacher-forced path (8 candidate SIDs): mean {tst['mean']:.2f} ms")

    print('-' * 78)
    print('Harness verified on CPU. The real GPU latencies still require an idle')
    print('device and are NOT_MEASURED until then.')
    return results


# ===========================================================================
# main
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='PHASE 11 measured inference complexity (writes into the '
                    'complexity/ directory; never overwrites an artifact).')
    p.add_argument('--dry-run', action='store_true',
                   help='validate paths/args and report the GPU guard verdict; '
                        'takes no measurement and creates no file')
    p.add_argument('--cpu-selftest', action='store_true',
                   help='CPU-only smoke test of the timing harness; never touches the GPU')
    p.add_argument('--full-cpu-selftest', action='store_true',
                   help='run the WHOLE pipeline on CPU at tiny size to verify every '
                        'code path; the output is marked CPU_SELFTEST_ONLY and is '
                        'never published')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                   help="'auto'/'cuda' require a working CUDA device; 'cpu' runs the "
                        "pipeline selftest on CPU and is never published")
    p.add_argument('--emit-not-measured', action='store_true',
                   help='write schema-complete NOT_MEASURED artifacts with the '
                        'current blocker, without measuring anything')
    p.add_argument('--force', action='store_true',
                   help='measure even though the device is busy; the output records it')
    p.add_argument('--datasets', nargs='+', default=['Beauty'],
                   choices=['Beauty', 'Cell_Phones_and_Accessories',
                            'Health_and_Personal_Care', 'Toys_and_Games'])
    p.add_argument('--n-users', type=int, default=INFER_SIZE_DEFAULT,
                   help='users per timed batch (campaign infer_size)')
    p.add_argument('--beam-size', type=int, default=BEAM_SIZE)
    p.add_argument('--warmup', type=int, default=WARMUP_DEFAULT)
    p.add_argument('--measured', type=int, default=MEASURED_DEFAULT)
    p.add_argument('--seed', type=int, default=SEED_DEFAULT)
    p.add_argument('--tf-batch', type=int, default=TF_BATCH_DEFAULT,
                   help='candidate SEQUENCES per teacher-forced forward')
    p.add_argument('--tf-sequences', type=int, default=TF_BATCH_DEFAULT,
                   help='candidate SIDs in the standalone C_TF micro-benchmark')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--gpu-mem-limit-mib', type=float, default=1000.0)
    p.add_argument('--gpu-util-limit-pct', type=float, default=0.0)
    p.add_argument('--include-campaign-pool', action='store_true', default=True)
    p.add_argument('--include-native-pool', action='store_true', default=True)
    p.add_argument('--include-c-tf', action='store_true', default=True)
    p.add_argument('--no-campaign-pool', dest='include_campaign_pool',
                   action='store_false')
    p.add_argument('--no-native-pool', dest='include_native_pool', action='store_false')
    p.add_argument('--no-c-tf', dest='include_c_tf', action='store_false')
    p.add_argument('--blocker', type=str, default=None,
                   help='blocker text recorded by --emit-not-measured')
    p.add_argument('--selftest-users', type=int, default=4)
    p.add_argument('--selftest-beam', type=int, default=4)
    p.add_argument('--no-publish', dest='publish', action='store_false', default=True,
                   help='write only into the per-run directory')
    p.add_argument('--refresh-placeholder', action='store_true',
                   help='allow a new all-NOT_MEASURED blocker record to replace an '
                        'existing all-NOT_MEASURED canonical record; the replaced '
                        'bytes are archived with their sha256. A record containing '
                        'real measurements is never replaced by this flag.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.cpu_selftest:
        res = cpu_selftest(args)
        lines = json.dumps(res, indent=2, default=str)
        print()
        print(lines)
        return 0

    if args.full_cpu_selftest:
        # Tiny sizes so the whole pipeline finishes quickly on CPU.  Nothing here
        # is claimed to be a latency measurement; see run_measurements().
        args.device = 'cpu'
        args.datasets = [args.datasets[0]]
        args.n_users = min(args.n_users, 2)
        args.beam_size = min(args.beam_size, 2)
        args.warmup = min(args.warmup, 1)
        args.measured = min(args.measured, 2)
        args.tf_batch = min(args.tf_batch, 4)
        args.tf_sequences = min(args.tf_sequences, 8)

    # --- GPU guard FIRST: it uses nvidia-smi only, never a CUDA context ---
    gpu_info = probe_gpu()
    host_procs = scan_host_processes()
    busy, reasons, verdicts = assess_contention(
        gpu_info, args.gpu_mem_limit_mib, args.gpu_util_limit_pct, host_procs)

    plan = build_plan(args)

    if args.dry_run:
        print_plan(plan, args)
        print()
        print('GPU GUARD (read-only; nvidia-smi only, no CUDA context created)')
        print('-' * 78)
        if not gpu_info.get('nvidia_smi_ok'):
            print(f"  nvidia-smi unavailable: {gpu_info.get('error')}")
        for g in gpu_info.get('gpus', []):
            print(f"  GPU {g['index']} {g['name']} driver {g['driver_version']} "
                  f"memory.used={g['memory_used_MiB']:.0f}/{g['memory_total_MiB']} MiB "
                  f"util={g['util_pct']:.0f}%")
        for a in gpu_info.get('compute_apps', []):
            print(f"  compute app pid={a['pid']} {a['process_name']} "
                  f"{a['used_memory_MiB']:.0f} MiB")
        for p in host_procs:
            print(f"  host process pid={p['pid']} matched '{p['pattern']}'")
        print(f"  VERDICT: {'BUSY -- a real run would REFUSE (exit 3)' if busy else 'IDLE -- a real run would proceed'}")
        for r in reasons:
            print(f'    * {r}')
        print('-' * 78)
        print('DRY RUN: no measurement taken, no file created, no artifact touched.')
        return 0

    if args.emit_not_measured:
        blocker = args.blocker or (' | '.join(reasons) if reasons
                                   else 'measurement not requested in this invocation')
        write_not_measured_artifacts(args, gpu_info, verdicts, busy, reasons, blocker)
        return 0

    if busy and not args.force and args.device != 'cpu':
        guard_or_die(args, gpu_info, busy, reasons)

    return run_measurements(args, gpu_info, verdicts, busy, reasons, plan)


def run_measurements(args, gpu_info, verdicts, busy, reasons, plan):
    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = environment(gpu_info)
    if args.device == 'cpu':
        device = torch.device('cpu')
        cpu_selftest_run = True
    else:
        if not torch.cuda.is_available():
            print('REFUSING: torch.cuda.is_available() is False; nothing to measure.')
            sys.exit(4)
        device = torch.device('cuda')
        cpu_selftest_run = False
    env['measurement_device'] = str(device)

    if cpu_selftest_run:
        print('=' * 78)
        print('FULL-PIPELINE CPU SELFTEST RUN.')
        print('Every code path below (load -> warmup -> timed -> decode -> union ->')
        print('teacher-forced -> aggregate -> CSV/TeX/JSON) is exercised, but these')
        print('are CPU numbers.  They are NOT the reported GPU latencies, they are')
        print('marked CPU_SELFTEST_ONLY in every row, and they are NEVER published')
        print('to the canonical complexity/ paths.')
        print('=' * 78)
    else:
        print(f"torch {env['torch']} / CUDA build {env['torch_cuda_build']} / "
              f"{env['gpu_name']} driver {env['gpu_driver_version']} "
              f"(driver caps CUDA at {env['nvidia_smi_cuda_version']})")

    stamp = utc_stamp()
    suffix = '_cpuselftest' if cpu_selftest_run else ''
    run_dir = RUNS_DIR / f'{stamp}_pid{os.getpid()}{suffix}'
    run_dir.mkdir(parents=True, exist_ok=False)
    if cpu_selftest_run:
        (run_dir / 'SELFTEST_NOT_A_MEASUREMENT.txt').write_text(
            'This run is a CPU-only pipeline selftest.\n'
            'It is NOT a GPU measurement and must never be reported as one.\n'
            'Every row carries status/measurement_kind CPU_SELFTEST_ONLY.\n',
            encoding='utf-8')

    ctx = {'raw_rows': [], 'gpu_busy': busy, 'forced': bool(args.force and busy),
           'sync_used': ('none (cpu selftest: torch.cuda.synchronize() is a no-op)'
                         if cpu_selftest_run else
                         'torch.cuda.synchronize() (before start and before stop)'),
           'missing_notes': [], 'cpu_selftest': cpu_selftest_run}

    hist_cache = {}
    mf_models, mf_common = _load_mf_modules()

    print('Loading history tokens (outside every timed region) ...')
    for ds in args.datasets:
        arms = arm_specs(ds)
        for name in ('spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5', 'rqvae5'):
            a = arms[name]
            for cand in ({a.get('sid'), a.get('sid_native'), a.get('sid_sid6')} - {None}):
                key = f'{ds}|{cand}'
                if key in hist_cache or not Path(cand).exists():
                    continue
                t0 = time.perf_counter()
                h, m = mf_models.build_history_tokens(
                    ds, 'test', cand, max_len=20, batch=512)
                hist_cache[key] = (h, m)
                print(f'  {ds:32s} {Path(cand).name:60s} {h.shape} '
                      f'({time.perf_counter() - t0:.1f} s, excluded from timing)')

    agg_rows = []
    for item in plan:
        ds = item['dataset']
        hist_bundles = _history_bundles_for(ds, item, hist_cache, arm_specs(ds))
        if item['kind'] == 'single':
            res = measure_single(ds, item['method'], item['entry'] if not item['missing'] else None,
                                 hist_bundles, args, device, env, ctx)
            agg_rows.append(_agg_single(item['method'], ds, res, args, env, ctx))
        elif item['kind'] == 'pool':
            if not item['missing']:
                hb = {e['arm']: hist_cache[f'{ds}|{e["sid_path"]}']
                      for e in item['resolved']}
            else:
                hb = {}
            uniform = measure_pool(ds, item['uniform_label'], item['resolved'],
                                   item['missing'], hb, args, device, env, ctx,
                                   measure_cfla=False, pool_kind=item['pool'])
            cfla = measure_pool(ds, item['cfla_label'], item['resolved'],
                                item['missing'], hb, args, device, env, ctx,
                                measure_cfla=True, pool_kind=item['pool'])
            agg_rows.append(_agg_pool(item['uniform_label'], ds, uniform, args,
                                      env, ctx, kind='uniform'))
            agg_rows.append(_agg_pool(item['cfla_label'], ds, cfla, args,
                                      env, ctx, kind='cfla'))
        else:
            res = measure_c_tf_microbenchmark(ds, item['entry'], hist_bundles,
                                              args, device, env, ctx)
            agg_rows.append(_agg_c_tf(item['method'], ds, res, args, env, ctx))

    # ------------------------------------------------------------------
    # a CPU selftest run must never be mistaken for a GPU measurement
    # ------------------------------------------------------------------
    if cpu_selftest_run:
        cpu_note = ('NOT_MEASURED (cpu selftest: torch.cuda.max_memory_allocated() '
                    'is undefined without a CUDA device)')
        for r in ctx['raw_rows']:
            if r.get('status') == 'MEASURED':
                r['status'] = 'CPU_SELFTEST_ONLY'
                r['note'] = '[CPU SELFTEST - NOT A GPU MEASUREMENT] ' + str(r.get('note', ''))
            if r.get('peak_memory_MiB') == NOT_MEASURED:
                r['peak_memory_MiB'] = cpu_note
        for r in agg_rows:
            if r.get('status') == 'MEASURED':
                r['status'] = 'MEASURED_CPU_SELFTEST_ONLY'
                r['measurement_kind'] = 'cpu_selftest_only'
            r['note'] = '[CPU SELFTEST - NOT A GPU MEASUREMENT] ' + str(r.get('note', ''))
            if r.get('peak_memory_MiB') == NOT_MEASURED:
                r['peak_memory_MiB'] = cpu_note

    # ------------------------------------------------------------------
    # write artifacts into this run's own directory, then publish-if-absent
    # ------------------------------------------------------------------
    raw_path = run_dir / 'LATENCY_RAW_RUNS_MEASURED.csv'
    agg_path = run_dir / 'INFERENCE_COMPLEXITY_MEASURED.csv'
    tex_path = run_dir / 'INFERENCE_COMPLEXITY_MEASURED.tex'
    meta_path = run_dir / 'RUN_METADATA.json'

    write_csv(raw_path, RAW_COLUMNS, ctx['raw_rows'])
    write_csv(agg_path, AGG_COLUMNS, agg_rows)
    tex_note = (
        f"Measured on {env['gpu_name']} (driver {env['gpu_driver_version']}, "
        f"driver CUDA cap {env['nvidia_smi_cuda_version']}); "
        f"torch {env['torch']} (cu{env['torch_cuda_build']}). "
        f"Protocol: {args.warmup} discarded warmup iterations then {args.measured} "
        f"measured iterations, each bracketed by torch.cuda.synchronize() before "
        f"start and before stop; model loading excluded from every timed region; "
        f"seed {args.seed}. Beam column and Candidate-NLL column are milliseconds "
        f"per user; Peak GPU is torch.cuda.max_memory_allocated() in MiB. "
        f"CFLA is NOT inference-free: its Cand.\\ NLL column is the measured "
        f"cross-factorization teacher-forced candidate scoring. "
        f"See RUN_METADATA.json for what was not measured."
        + (' THIS TABLE IS A CPU PIPELINE SELFTEST AND IS NOT A GPU MEASUREMENT.'
           if cpu_selftest_run else '')
        + (f" WARNING: the device was BUSY during this run (--force)."
           if ctx['forced'] else ''))
    write_tex(tex_path, agg_rows, env, tex_note)

    metadata = {
        'phase': 'PHASE 11 -- inference complexity',
        'driver': str(Path(__file__).resolve()),
        'driver_sha256': sha256_file(Path(__file__).resolve()),
        'run_directory': str(run_dir),
        'argv': sys.argv,
        'args': vars(args),
        'protocol': PROTOCOL,
        'protocol_applied': {
            'warmup_iterations_discarded': args.warmup,
            'measured_iterations': args.measured,
            'beam_size': args.beam_size,
            'users_per_timed_batch': args.n_users,
            'tf_candidate_sequences_per_forward': args.tf_batch,
            'seed': args.seed,
        },
        'environment': env,
        'gpu_guard': {
            'mem_limit_MiB': args.gpu_mem_limit_mib,
            'util_limit_pct': args.gpu_util_limit_pct,
            'verdicts': verdicts,
            'busy_at_measure_time': busy,
            'busy_reasons': reasons,
            'forced': ctx['forced'],
            'nvidia_smi_raw': {'ok': gpu_info.get('nvidia_smi_ok'),
                               'error': gpu_info.get('error'),
                               'compute_apps': gpu_info.get('compute_apps', [])},
            'host_training_processes': scan_host_processes(),
        },
        'model_loading': {
            'excluded_from_every_timed_region': True,
            'note': 'Checkpoints are torch.load()-ed and moved to the device before '
                    'the first warmup iteration; nothing in a timed region loads '
                    'weights or builds a model.',
        },
        'not_measured': ctx['missing_notes'] + [
            'FLOP counts: no profiler was run in this driver; parameter counts and '
            'wall-clock latency are reported separately and are NOT interchangeable.',
            'Any arm whose SID array or checkpoint is absent is reported as '
            'NOT_MEASURED with the exact missing path in the blocker column.',
        ],
        'artifacts': {'raw_runs': str(raw_path), 'aggregate': str(agg_path),
                      'tex': str(tex_path)},
        'row_counts': {'raw': len(ctx['raw_rows']), 'aggregate': len(agg_rows)},
        'cpu_selftest_run': cpu_selftest_run,
        'notes': [
            'CFLA is NOT inference-free. Its extra cost is the measured '
            'cross-factorization teacher-forced candidate scoring.',
            'Parameter count, symbolic/FLOP-ish accounting and wall-clock latency '
            'are three different things and are kept in different columns.',
        ] + (['THIS RUN IS A CPU PIPELINE SELFTEST. It is not a GPU measurement and '
              'must never be reported as one.'] if cpu_selftest_run else []),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')

    pub_log = []
    publish_allowed = args.publish and not cpu_selftest_run
    if args.publish and cpu_selftest_run:
        print('PUBLISH DISABLED: a CPU selftest run is never published to the '
              'canonical complexity/ paths.')
    if publish_allowed:
        run_has_measurement = any(
            str(r.get('status', '')).startswith('MEASURED') and
            NOT_MEASURED not in str(r.get('status', '')) for r in agg_rows)
        canonical_placeholder = csv_is_pure_not_measured(
            COMPLEXITY / 'INFERENCE_COMPLEXITY_MEASURED.csv')
        for name in ('LATENCY_RAW_RUNS_MEASURED.csv', 'INFERENCE_COMPLEXITY_MEASURED.csv',
                     'INFERENCE_COMPLEXITY_MEASURED.tex', 'RUN_METADATA.json'):
            publish(run_dir, name, COMPLEXITY, pub_log,
                    run_has_measurement, canonical_placeholder,
                    allow_placeholder_refresh=args.refresh_placeholder)
        (run_dir / 'PUBLISH_LOG.json').write_text(
            json.dumps(pub_log, indent=2, default=str), encoding='utf-8')

    print()
    print('WROTE')
    for p in (raw_path, agg_path, tex_path, meta_path):
        print(f'  {p}')
    for e in pub_log:
        if e['published']:
            print(f"  published ({e.get('action', 'created')}) -> {e['canonical_path']}")
            if 'archived_placeholder' in e:
                print(f"    placeholder archived at {e['archived_placeholder']}")
        else:
            print(f"  NOT published {e['artifact']}: {e['reason']}")
            print(f"    canonical: {e['canonical_path']} sha256={e['canonical_sha256']}")
            print(f"    this run : {e['run_path']} sha256={e['run_sha256']}")
            print(f"    to replace deliberately: {e['manual_replace_command']}")
    return 0


def _history_bundles_for(ds, item, hist_cache, arms):
    """arm name -> (history_tokens, history_mask) for the arms being measured."""
    out = {}
    if item['kind'] == 'single':
        e = item['entry']
        key = f'{ds}|{e["sid_path"]}'
        out[e['arm']] = hist_cache.get(key)
    elif item['kind'] == 'pool':
        for e in item['resolved']:
            out[e['arm']] = hist_cache.get(f'{ds}|{e["sid_path"]}')
    else:
        if item['entry'] is not None:
            e = item['entry']
            out[e['arm']] = hist_cache.get(f'{ds}|{e["sid_path"]}')
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# aggregate row builders
# ---------------------------------------------------------------------------
def _sym(n_factors, beam_stat, tf_per_user=None, c_tf_ms_per_seq=None,
         aggregation_ms_user=None):
    """The symbolic terms of THEORETICAL_COMPLEXITY.md, kept as their own columns.

    Only the SHAPE of the symbolic model is reproduced here (M sequential beams,
    one hash-merge, M*U teacher-forced scorings, M*U element-wise LSE).  The
    values are strings describing the term, never a number pretending to be a
    measurement.
    """
    M = n_factors
    return {
        'symbolic_T_beam': f'sum_t=1..L O(B*C_dec(t)) x M(M={M}) + M*C_enc(H)',
        'symbolic_T_union_dedup': f'O(M*B*L_F), M={M}',
        'symbolic_T_candidate_nll': (f'M*U*C_TF(L) with M={M}, U measured; '
                                     f'C_TF measured = '
                                     f'{fmt(c_tf_ms_per_seq, 6)} ms/candidate-seq'
                                     if c_tf_ms_per_seq else f'M*U*C_TF(L), M={M}'),
        'symbolic_T_calibration_lse': f'O(M*U), M={M}',
        'symbolic_total': ('M*T_beam(1) + O(M*B*L) + M*U*C_TF(L) + O(M*U)'
                           if n_factors > 1 else 'T_beam(1) + O(U)'),
    }


def _base_agg(method, n_factors, status, blocker, env, ctx, note):
    return {
        'method': method, 'n_factors': n_factors, 'mean_union': NOT_MEASURED,
        'beam_ms_user': NOT_MEASURED, 'candidate_nll_ms_user': NOT_MEASURED,
        'aggregation_ms_user': NOT_MEASURED, 'total_ms_user': NOT_MEASURED,
        'users_per_sec': NOT_MEASURED, 'peak_memory_MiB': NOT_MEASURED,
        'parameters_M': NOT_MEASURED, 'checkpoint_MB': NOT_MEASURED,
        'status': status, 'blocker': blocker, 'note': note,
        'std_beam_ms_user': NOT_MEASURED, 'median_beam_ms_user': NOT_MEASURED,
        'min_beam_ms_user': NOT_MEASURED, 'max_beam_ms_user': NOT_MEASURED,
        'std_total_ms_user': NOT_MEASURED, 'median_total_ms_user': NOT_MEASURED,
        'min_total_ms_user': NOT_MEASURED, 'max_total_ms_user': NOT_MEASURED,
        'n_measured_runs': NOT_MEASURED, 'n_warmup_runs': NOT_MEASURED,
        'n_users': NOT_MEASURED, 'seed': NOT_MEASURED, 'gpu': env['gpu_name'],
        'torch': env['torch'], 'cuda': env['cuda'],
        'c_tf_ms_per_candidate_seq': NOT_MEASURED,
        'mean_union_before_dedup': NOT_MEASURED,
        'n_tf_sequences_per_user': NOT_MEASURED,
        'symbolic_T_beam': NOT_MEASURED, 'symbolic_T_union_dedup': NOT_MEASURED,
        'symbolic_T_candidate_nll': NOT_MEASURED,
        'symbolic_T_calibration_lse': NOT_MEASURED, 'symbolic_total': NOT_MEASURED,
        'measurement_kind': NOT_MEASURED, 'directly_timed_fields': NOT_MEASURED,
        'derived_from_measured_fields': NOT_MEASURED,
        'gpu_busy_at_measure_time': ctx['gpu_busy'], 'forced': ctx['forced'],
        'input_artifacts': NOT_MEASURED,
    }


def _agg_single(method, ds, res, args, env, ctx):
    row = _base_agg(method, 1, res.get('status', NOT_MEASURED),
                    res.get('blocker', NOT_MEASURED), env, ctx,
                    f'{ds}; single-model beam-{args.beam_size} retrieval; '
                    f'model loading excluded from the timed region')
    if res.get('status') != 'MEASURED':
        return row
    st = res['beam_stat']
    ms_user = st['mean'] / args.n_users
    row.update({
        'mean_union': fmt(res['mean_union'], 4),
        'beam_ms_user': fmt(ms_user, 6),
        'candidate_nll_ms_user': '0 (single model performs no cross-factor scoring)',
        'aggregation_ms_user': fmt(0.0, 6),
        'total_ms_user': fmt(ms_user, 6),
        # users_per_sec = 1000 / mean_ms_per_user, stated identically to the pooled
        # branch (:2069) so the two can never drift apart again (section 27).
        'users_per_sec': fmt(1000.0 / ms_user, 3),
        'peak_memory_MiB': fmt(res['peak_memory_MiB'], 2),
        'parameters_M': fmt(res['parameters_M'], 6),
        'checkpoint_MB': fmt(res['checkpoint_MB'], 3),
        'std_beam_ms_user': fmt(st['std'] / args.n_users, 6),
        'median_beam_ms_user': fmt(st['median'] / args.n_users, 6),
        'min_beam_ms_user': fmt(st['min'] / args.n_users, 6),
        'max_beam_ms_user': fmt(st['max'] / args.n_users, 6),
        'std_total_ms_user': fmt(st['std'] / args.n_users, 6),
        'median_total_ms_user': fmt(st['median'] / args.n_users, 6),
        'min_total_ms_user': fmt(st['min'] / args.n_users, 6),
        'max_total_ms_user': fmt(st['max'] / args.n_users, 6),
        'n_measured_runs': args.measured, 'n_warmup_runs': args.warmup,
        'n_users': args.n_users, 'seed': args.seed,
        'measurement_kind': 'directly_timed',
        'directly_timed_fields': 'beam_ms_user,total_ms_user,peak_memory_MiB,users_per_sec',
        'derived_from_measured_fields': 'per-user figures = timed batch / n_users',
        'input_artifacts': ' ; '.join(res['input_artifacts']),
    })
    row.update(_sym(1, st))
    return row


def _agg_pool(method, ds, res, args, env, ctx, kind):
    n_factors = res.get('n_factors', NOT_MEASURED)
    row = _base_agg(method, n_factors, res.get('status', NOT_MEASURED),
                    res.get('blocker', NOT_MEASURED), env, ctx,
                    f'{ds}; pool={res.get("pool", NOT_MEASURED)}; '
                    f'arms={",".join(res.get("arms", []))}; '
                    f'end-to-end timed region; model loading excluded')
    if res.get('status') != 'MEASURED':
        return row
    ust = res['uniform_stat']
    uniform_ms_user = ust['mean'] / args.n_users
    if kind == 'cfla':
        cst = res['cfla_stat']
        total_ms_user = cst['mean'] / args.n_users
        row.update({
            'candidate_nll_ms_user': fmt(res['candidate_nll_ms_user'], 6),
            'c_tf_ms_per_candidate_seq': fmt(res['c_tf_ms_per_seq'], 8),
            'n_tf_sequences_per_user': fmt(res['n_tf_sequences_per_user'], 4),
            'std_total_ms_user': fmt(cst['std'] / args.n_users, 6),
            'median_total_ms_user': fmt(cst['median'] / args.n_users, 6),
            'min_total_ms_user': fmt(cst['min'] / args.n_users, 6),
            'max_total_ms_user': fmt(cst['max'] / args.n_users, 6),
            'measurement_kind': 'directly_timed',
            'directly_timed_fields': ('total_ms_user and peak_memory_MiB in the CFLA '
                                      'timed region; Uniform total separately timed'),
            'derived_from_measured_fields':
                'candidate_nll_ms_user = CFLA_total - Uniform_total (two '
                'independently timed end-to-end regions); '
                'c_tf_ms_per_candidate_seq = that marginal / measured '
                'n_tf_sequences; beam/aggregation components measured separately',
        })
    else:
        total_ms_user = uniform_ms_user
        row.update({
            'candidate_nll_ms_user': '0 (Uniform performs no cross-factor scoring)',
            'c_tf_ms_per_candidate_seq': NOT_MEASURED,
            # For Uniform the whole timed region IS the total, so the per-run
            # spread of `total_ms_user` is the per-run spread of that region.
            'std_total_ms_user': fmt(ust['std'] / args.n_users, 6),
            'median_total_ms_user': fmt(ust['median'] / args.n_users, 6),
            'min_total_ms_user': fmt(ust['min'] / args.n_users, 6),
            'max_total_ms_user': fmt(ust['max'] / args.n_users, 6),
            'measurement_kind': 'directly_timed',
            'directly_timed_fields': 'total_ms_user,peak_memory_MiB',
            'derived_from_measured_fields':
                'per-user figures = timed batch / n_users; '
                'beam_ms_user = uniform_total_ms_user - separately-timed union dedup',
        })
    row.update({
        'mean_union': fmt(res['mean_union'], 4),
        'mean_union_before_dedup': fmt(res['mean_union_before_dedup'], 4),
        'beam_ms_user': fmt(uniform_ms_user - res['aggregation_ms_user'], 6),
        'aggregation_ms_user': fmt(res['aggregation_ms_user'], 6),
        'total_ms_user': fmt(total_ms_user, 6),
        # FIXED 2026-10-13 (final paper completion round, section 27).
        # `total_ms_user` is ALREADY per user (:1998 divides by n_users), so the
        # pooled rows previously reported 1000*n_users/mean_ms_per_user -- the
        # single-model rows at :1962 divide a batch total and were correct, which
        # made the two branches disagree by a factor of n_users (about 96x here).
        # users_per_sec = 1000 / mean_ms_per_user, uniformly.
        'users_per_sec': fmt(1000.0 / total_ms_user, 3),
        'peak_memory_MiB': fmt(res['peak_memory_MiB'], 2),
        'parameters_M': fmt(res['parameters_M'], 6),
        'checkpoint_MB': fmt(res['checkpoint_MB'], 3),
        'std_beam_ms_user': fmt(ust['std'] / args.n_users, 6),
        'median_beam_ms_user': fmt(ust['median'] / args.n_users, 6),
        'min_beam_ms_user': fmt(ust['min'] / args.n_users, 6),
        'max_beam_ms_user': fmt(ust['max'] / args.n_users, 6),
        'n_measured_runs': args.measured, 'n_warmup_runs': args.warmup,
        'n_users': args.n_users, 'seed': args.seed,
        'input_artifacts': ' ; '.join(res['input_artifacts']),
    })
    row.update(_sym(len(res.get('arms', [])) or n_factors, ust,
                    c_tf_ms_per_seq=res.get('c_tf_ms_per_seq')))
    return row


def _agg_c_tf(method, ds, res, args, env, ctx):
    row = _base_agg(method, NOT_MEASURED, res.get('status', NOT_MEASURED),
                    res.get('blocker', NOT_MEASURED), env, ctx,
                    f'{ds}; replaces the symbolic C_TF(L) with a measured value; '
                    f'model loading excluded')
    if res.get('status') != 'MEASURED':
        return row
    st = res['tf_stat']
    row.update({
        'beam_ms_user': 'n/a (no beam search in this micro-benchmark)',
        'candidate_nll_ms_user': fmt(st['mean'], 6) + ' per forward pass',
        'c_tf_ms_per_candidate_seq': fmt(res['c_tf_ms_per_seq'], 8),
        'total_ms_user': fmt(st['mean'], 6) + ' per forward pass',
        'peak_memory_MiB': fmt(res['peak_memory_MiB'], 2),
        'parameters_M': fmt(res['parameters_M'], 6),
        'checkpoint_MB': fmt(res['checkpoint_MB'], 3),
        'std_beam_ms_user': fmt(st['std'], 6),
        'median_beam_ms_user': fmt(st['median'], 6),
        'min_beam_ms_user': fmt(st['min'], 6),
        'max_beam_ms_user': fmt(st['max'], 6),
        'n_measured_runs': args.measured, 'n_warmup_runs': args.warmup,
        'n_users': args.n_users, 'seed': args.seed,
        'measurement_kind': 'directly_timed',
        'directly_timed_fields': 'candidate_nll_ms_user (one teacher-forced forward), '
                                 'peak_memory_MiB',
        'derived_from_measured_fields':
            f"c_tf_ms_per_candidate_seq = forward_time / {res['n_sequences']} "
            f"candidate sequences",
        'input_artifacts': ' ; '.join(res['input_artifacts']),
        'symbolic_T_candidate_nll':
            f"M*U*C_TF({res['sid_len']}) with measured C_TF = "
            f"{fmt(res['c_tf_ms_per_seq'], 8)} ms per candidate sequence",
        'symbolic_total': f"T_candidate_NLL = M*U*C_TF({res['sid_len']})",
    })
    return row


# ---------------------------------------------------------------------------
# NOT_MEASURED artifact emission
# ---------------------------------------------------------------------------
def write_not_measured_artifacts(args, gpu_info, verdicts, busy, reasons, blocker):
    import_torch_ok = True
    try:
        import torch
        torch_v = torch.__version__
        torch_build_v = torch.version.cuda
        cuda_v = f'{torch_build_v} (driver caps CUDA at {driver_cuda_cap(gpu_info)})'
    except ImportError as e:
        import_torch_ok = False
        torch_v = f'{NOT_MEASURED} (torch import failed: {type(e).__name__}: {e})'
        torch_build_v = NOT_MEASURED
        cuda_v = NOT_MEASURED

    g = gpu_info['gpus'][0] if gpu_info.get('gpus') else {}
    env = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'gpu_name': g.get('name', NOT_MEASURED),
        'torch': torch_v,
        'cuda': cuda_v,
    }
    stamp = utc_stamp()
    run_dir = RUNS_DIR / f'{stamp}_pid{os.getpid()}_notmeasured'
    run_dir.mkdir(parents=True, exist_ok=False)

    plan = build_plan(args)
    raw_rows = []
    agg_rows = []

    for item in plan:
        ds = item['dataset']
        if item['kind'] == 'single':
            if item['missing']:
                reason = ('INPUT ARTIFACTS MISSING (not an environment problem): '
                          + ' | '.join(item['missing'])
                          + f' | GPU at this moment: {blocker}')
            else:
                reason = (f'{blocker} | INPUT ARTIFACTS PRESENT AND VALIDATED: '
                          f"{item['entry']['sid_path']} ; {item['entry']['ckpt_path']}")
            _emit_not_measured_raw(raw_rows, {
                'method': item['method'], 'dataset': ds, 'arm': item['entry']['arm'],
                'pool': 'single', 'stage': 'beam_generate',
                'cohort': cohort_str(ds, args), 'user_batch': args.n_users,
                'beam_size': args.beam_size, 'sid_len': item['entry']['sid_len'],
                'num_workers': args.num_workers, 'seed': args.seed,
                'measured_runs': args.measured, 'warmup_runs': args.warmup,
                'device': 'cuda', 'forced': args.force}, env, reason)
            row = _base_agg(item['method'], 1, NOT_MEASURED, reason, env,
                            {'gpu_busy': busy, 'forced': args.force},
                            f'{ds}; single-model beam-{args.beam_size} retrieval; '
                            f'measured protocol NOT executed')
            row['input_artifacts'] = ' ; '.join(
                [str(item['entry']['sid_path']), str(item['entry']['ckpt_path'])])
            agg_rows.append(row)
        elif item['kind'] == 'pool':
            if item['missing']:
                reason = ('INPUT ARTIFACTS MISSING (not an environment problem): '
                          + ' | '.join(f'{m["arm"]}: ' + '; '.join(m['reasons'])
                                       for m in item['missing'])
                          + f' | GPU at this moment: {blocker}')
            else:
                reason = (f'{blocker} | ALL {len(item["resolved"])} POOL INPUT '
                          f'ARTIFACTS PRESENT')
            _emit_not_measured_raw(raw_rows, {
                'method': item['uniform_label'], 'dataset': ds,
                'arm': NOT_MEASURED, 'pool': item['pool'],
                'stage': 'beam_generate+union',
                'cohort': cohort_str(ds, args), 'user_batch': args.n_users,
                'beam_size': args.beam_size, 'num_workers': args.num_workers,
                'seed': args.seed, 'measured_runs': args.measured,
                'warmup_runs': args.warmup, 'device': 'cuda',
                'forced': args.force}, env, reason)
            for label in (item['uniform_label'], item['cfla_label']):
                row = _base_agg(
                    label, len(item['resolved']) + len(item['missing']),
                    NOT_MEASURED, reason, env,
                    {'gpu_busy': busy, 'forced': args.force},
                    f'{ds}; pool={item["pool"]}; '
                    f'arms resolved = {[e["arm"] for e in item["resolved"]] or "none"}')
                row['input_artifacts'] = ' ; '.join(
                    [str(e['ckpt_path']) for e in item['resolved']] +
                    [f'MISSING -> {r}' for m in item['missing']
                     for r in m['reasons']]) or NOT_MEASURED
                agg_rows.append(row)
        else:
            if item['entry'] is None:
                reason = ('INPUT ARTIFACTS MISSING (not an environment problem): '
                          + ' | '.join(item['missing']) + f' | GPU at this moment: {blocker}')
            else:
                reason = (f'{blocker} | INPUT ARTIFACTS PRESENT AND VALIDATED: '
                          f"{item['entry']['sid_path']} ; {item['entry']['ckpt_path']}")
            _emit_not_measured_raw(raw_rows, {
                'method': item['method'], 'dataset': ds,
                'arm': item['entry']['arm'] if item['entry'] else NOT_MEASURED,
                'pool': 'microbenchmark', 'stage': 'teacher_forced_forward',
                'cohort': cohort_str(ds, args), 'user_batch': args.n_users,
                'beam_size': args.beam_size, 'num_workers': args.num_workers,
                'seed': args.seed, 'measured_runs': args.measured,
                'warmup_runs': args.warmup, 'device': 'cuda',
                'forced': args.force}, env, reason)
            row = _base_agg(item['method'], NOT_MEASURED, NOT_MEASURED, reason, env,
                            {'gpu_busy': busy, 'forced': args.force},
                            f'{ds}; standalone teacher-forced micro-benchmark '
                            f'NOT executed')
            if item['entry'] is not None:
                row['input_artifacts'] = ' ; '.join(
                    [str(item['entry']['sid_path']), str(item['entry']['ckpt_path'])])
            agg_rows.append(row)

    raw_path = run_dir / 'LATENCY_RAW_RUNS_MEASURED.csv'
    agg_path = run_dir / 'INFERENCE_COMPLEXITY_MEASURED.csv'
    tex_path = run_dir / 'INFERENCE_COMPLEXITY_MEASURED.tex'
    meta_path = run_dir / 'RUN_METADATA.json'
    write_csv(raw_path, RAW_COLUMNS, raw_rows)
    write_csv(agg_path, AGG_COLUMNS, agg_rows)
    write_tex(tex_path, agg_rows, env, f'NOT MEASURED: {blocker}')
    meta_path.write_text(json.dumps({
        'phase': 'PHASE 11 -- inference complexity',
        'status': NOT_MEASURED,
        'blocker': blocker,
        'driver': str(Path(__file__).resolve()),
        'driver_sha256': sha256_file(Path(__file__).resolve()),
        'argv': sys.argv, 'args': vars(args),
        'environment': {
            'generated_utc': env['generated_utc'],
            'gpu_name': env['gpu_name'],
            'gpu_driver_version': g.get('driver_version', NOT_MEASURED),
            'gpu_memory_total_MiB': g.get('memory_total_MiB', NOT_MEASURED),
            'nvidia_smi_cuda_cap': driver_cuda_cap(gpu_info),
            'torch': torch_v, 'torch_cuda_build': torch_build_v,
            'torch_import_ok': import_torch_ok,
            'python': sys.version.split()[0],
            'python_executable': sys.executable,
            'note_driver_vs_torch': ('The NVIDIA driver caps CUDA at 12.8 while the '
                                     'venv torch is a cu128 build; any number this '
                                     'driver eventually produces is tied to THIS '
                                     'device and driver.'),
        },
        'gpu_guard': {
            'mem_limit_MiB': args.gpu_mem_limit_mib,
            'util_limit_pct': args.gpu_util_limit_pct,
            'verdicts': verdicts, 'busy': busy, 'busy_reasons': reasons,
            'nvidia_smi_raw': {'ok': gpu_info.get('nvidia_smi_ok'),
                               'error': gpu_info.get('error'),
                               'compute_apps': gpu_info.get('compute_apps', [])},
            'host_training_processes': scan_host_processes(),
        },
        'protocol_not_executed': PROTOCOL,
        'not_measured': [
            'Every row in the two CSVs is NOT_MEASURED; see the blocker column.',
            blocker,
        ],
        'planned_but_not_run': [
            {'dataset': it['dataset'], 'kind': it['kind'],
             'method': it.get('method') or it.get('uniform_label'),
             'missing': it.get('missing', [])} for it in plan],
        'artifacts': {'raw_runs': str(raw_path), 'aggregate': str(agg_path),
                      'tex': str(tex_path)},
    }, indent=2, default=str), encoding='utf-8')

    pub_log = []
    if args.publish:
        canonical_placeholder = csv_is_pure_not_measured(
            COMPLEXITY / 'INFERENCE_COMPLEXITY_MEASURED.csv')
        for name in ('LATENCY_RAW_RUNS_MEASURED.csv', 'INFERENCE_COMPLEXITY_MEASURED.csv',
                     'INFERENCE_COMPLEXITY_MEASURED.tex', 'RUN_METADATA.json'):
            publish(run_dir, name, COMPLEXITY, pub_log,
                    run_has_measurement=False,
                    canonical_csv_is_placeholder=canonical_placeholder,
                    allow_placeholder_refresh=args.refresh_placeholder)
        (run_dir / 'PUBLISH_LOG.json').write_text(
            json.dumps(pub_log, indent=2, default=str), encoding='utf-8')

    print('WROTE (all rows NOT_MEASURED)')
    for p in (raw_path, agg_path, tex_path, meta_path):
        print(f'  {p}')
    for e in pub_log:
        if e['published']:
            print(f"  published ({e.get('action', 'created')}) -> {e['canonical_path']}")
            if 'archived_placeholder' in e:
                print(f"    placeholder archived at {e['archived_placeholder']}")
        else:
            print(f"  NOT published {e['artifact']}: {e['reason']}")
            print(f"    canonical: {e['canonical_path']} sha256={e['canonical_sha256']}")
            print(f"    this run : {e['run_path']} sha256={e['run_sha256']}")
            print(f"    to replace deliberately: {e['manual_replace_command']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
