#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUN_METADATA.json writer shared by every entry point.

Records enough to reproduce or at least explain a run later: git state, the
exact command, a config hash, the environment (GPU/driver/CUDA/torch), host, and
the hashes of the inputs and outputs.

The historical runs did NOT record any of this, which is why answering "did the
hardware change the result?" required a full retrain. This utility exists so
that never happens again.

Usage
-----
    from src.utils.run_metadata import RunMetadata
    run = RunMetadata(stage='sid_generation', config=cfg, inputs=[emb_path])
    ...
    run.finish(outputs=[sid_dir])   # writes RUN_METADATA.json
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import repo_paths as rp


def _git(args, cwd):
    try:
        out = subprocess.run(['git'] + args, cwd=str(cwd), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _hash_file(p: Path):
    p = Path(p)
    if p.is_file():
        h = hashlib.sha256()
        with p.open('rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
        return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}
    if p.is_dir():
        entries = []
        for f in sorted(x for x in p.rglob('*') if x.is_file()):
            entries.append(f'{f.relative_to(p)}:{_hash_file(f)["sha256"]}')
        digest = hashlib.sha256('\n'.join(entries).encode()).hexdigest()
        return {'path': str(p), 'sha256_tree': digest, 'n_files': len(entries)}
    return {'path': str(p), 'missing': True}


def config_hash(config) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def environment() -> dict:
    env = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
    }
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env['cuda'] = torch.version.cuda
            env['gpu_count'] = torch.cuda.device_count()
            env['gpu_name'] = torch.cuda.get_device_name(0)
    except Exception as e:                      # pragma: no cover
        env['torch_error'] = str(e)
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version',
                              '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            env['nvidia_smi'] = out.stdout.strip()
    except Exception:
        pass
    return env


class RunMetadata:
    def __init__(self, stage: str, config: dict | None = None, inputs=None,
                 out_dir: Path | None = None):
        self.stage = stage
        self.config = config or {}
        self.inputs = [Path(p) for p in (inputs or [])]
        self.out_dir = Path(out_dir) if out_dir else rp.OUTPUT_ROOT / 'logs'
        self.t0 = time.time()
        self.data = {
            'stage': stage,
            'git_commit': _git(['rev-parse', 'HEAD'], rp.REPO_ROOT),
            'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD'], rp.REPO_ROOT),
            'git_dirty': bool(_git(['status', '--porcelain'], rp.REPO_ROOT)),
            'command': ' '.join(sys.argv),
            'config': self.config,
            'config_hash': config_hash(self.config),
            'environment': environment(),
            'artifact_root': str(rp.ARTIFACT_ROOT),
            'output_root': str(rp.OUTPUT_ROOT),
            'start_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(self.t0)),
        }

    def finish(self, outputs=None, extra: dict | None = None) -> Path:
        self.data['end_timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        self.data['elapsed_seconds'] = round(time.time() - self.t0, 3)
        self.data['input_artifacts'] = [_hash_file(p) for p in self.inputs]
        self.data['output_artifacts'] = [_hash_file(Path(p)) for p in (outputs or [])]
        if extra:
            self.data['extra'] = extra
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / 'RUN_METADATA.json'
        p.write_text(json.dumps(self.data, indent=2, default=str))
        return p
