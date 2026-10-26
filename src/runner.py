#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified experiment runner.

    python3 -m src.runner --config configs/experiments/<name>.yaml --stage sid
    python3 -m src.runner --config configs/experiments/<name>.yaml --stage all

Stages: sid | train | beam | evaluate | fusion | all

The runner ONLY executes what the config declares. It never edits source
files, never uses sed/heredoc rewriting, never copies or restores backups,
never writes into ARTIFACT_ROOT, and never searches hyper-parameters.

Every stage writes under outputs/runs/<run_id>/ :
    config.yaml      immutable snapshot of the config actually used
    RUN_METADATA.json
    sid/  checkpoint/  beam/  metrics/  logs/
    manifest.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils import repo_paths as rp                                   # noqa: E402
from src.utils.experiment_config import (                                # noqa: E402
    ConfigError, load_config, run_dir, run_id, validate,
)
from src.utils.run_metadata import RunMetadata, config_hash              # noqa: E402

STAGES = ('sid', 'train', 'beam', 'evaluate', 'fusion', 'all')
REGISTRY = rp.OUTPUT_ROOT / 'registry.jsonl'


# ---------------------------------------------------------------- registry --
def registry_append(entry: dict) -> None:
    """Concurrency-safe append: one atomic line per run."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, default=str) + '\n'
    with REGISTRY.open('a') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------- run layout ---
def ensure_run_layout(cfg: dict) -> Path:
    d = run_dir(cfg)
    for sub in ('sid', 'checkpoint', 'beam', 'metrics', 'logs'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    snap = d / 'config.yaml'
    if not snap.exists():
        # immutable snapshot of what was ACTUALLY used
        import yaml
        src = Path(cfg['_source_path'])
        shutil.copy2(src, snap)
        meta = d / 'config_snapshot.json'
        meta.write_text(json.dumps({
            'source_config_path': str(src),
            'source_config_sha256': cfg['_source_sha256'],
            'snapshot_sha256': hashlib.sha256(snap.read_bytes()).hexdigest(),
            'config_hash': config_hash(cfg),
        }, indent=2))
    return d


def manifest(d: Path) -> Path:
    lines = []
    for f in sorted(x for x in d.rglob('*') if x.is_file()):
        if f.name == 'manifest.sha256':
            continue
        h = hashlib.sha256()
        with f.open('rb') as fh:
            for c in iter(lambda: fh.read(1 << 20), b''):
                h.update(c)
        lines.append(f'{h.hexdigest()}  {f.relative_to(d)}')
    p = d / 'manifest.sha256'
    p.write_text('\n'.join(lines) + '\n')
    return p


# ----------------------------------------------------------------- stages ---
def stage_sid(cfg: dict, d: Path, resume: bool) -> dict:
    from src.sid.generator import generate
    sid_cfg = {
        'dataset': cfg['experiment']['dataset'],
        'affinity': cfg['sid']['affinity'],
        'k': cfg['sid']['k'],
        'codebook_size': cfg['sid']['codebook_size'],
        'seed': cfg['experiment'].get('seed', 2025),
        'pca_dim': cfg['sid'].get('pca_dim', 64),
        'alignment_mode': cfg['sid'].get('alignment_mode', 'canonical'),
    }
    res = generate(sid_cfg, verbose=True)
    # mirror the artifacts into this run's tree
    dst = d / 'sid'
    for name in ('sid.npy', 'base_sid.npy', 'dim_groups.npy', 'dim_labels.npy',
                 'metadata.json', 'manifest.sha256'):
        src = Path(res['out_dir']) / name
        if src.exists():
            shutil.copy2(src, dst / name)
    return {'sid_dir': str(res['out_dir']), 'copied_to': str(dst)}


def stage_train(cfg: dict, d: Path, resume: bool) -> dict:
    """Thin wrapper around src.tiger.train: pass flags, never patch source."""
    ds = cfg['experiment']['dataset']
    seed = cfg['experiment'].get('seed', 2025)
    k = cfg['sid']['k']
    L = k + 1
    sid_dir = Path(stage_sid(cfg, d, resume)['sid_dir'])
    code_path = sid_dir / 'sid.npy'
    ckpt = d / 'checkpoint' / 'model.pth'
    log = d / 'logs' / 'train.log'

    if resume and ckpt.exists():
        print(f'[runner] resume: keeping existing checkpoint {ckpt}')
        return {'checkpoint': str(ckpt), 'resumed': True}

    m = cfg.get('model', {})
    g = cfg.get('generation', {})
    argv = [
        sys.executable, '-m', 'src.tiger.train',
        '--dataset_path', str(rp.ARTIFACT_ROOT / 'data' / ds),
        '--code_path', str(code_path),
        '--vocab_size', str(cfg['sid']['vocab_size']),
        '--codebook_size', str(cfg['sid']['codebook_size']),
        '--max_len', str(m.get('max_len', 20)),
        '--batch_size', str(m.get('batch_size', 256)),
        '--lr', str(m.get('learning_rate', 1e-4)),
        '--num_epochs', str(m.get('max_epochs', 200)),
        '--early_stop', str(m.get('patience', 10)),
        '--infer_size', str(m.get('infer_size', 96)),
        '--beam_size', str(g.get('beam_size', 30)),
        '--eval_mode', str(cfg.get('evaluation', {}).get('mode', 'item')),
        '--device', 'cuda',
        '--seed', str(seed),
        '--log_path', str(log),
        '--save_path', str(ckpt),
    ]
    print('[runner] train:', ' '.join(argv[2:8]), '...')
    p = subprocess.run(argv, cwd=str(REPO))
    if p.returncode != 0:
        raise RuntimeError(f'training failed with exit code {p.returncode}')
    return {'checkpoint': str(ckpt), 'resumed': False}


def stage_beam(cfg: dict, d: Path, resume: bool) -> dict:
    ds = cfg['experiment']['dataset']
    k, L = cfg['sid']['k'], cfg['sid']['k'] + 1
    sid_dir = Path(stage_sid(cfg, d, resume)['sid_dir'])
    ckpt = d / 'checkpoint' / 'model.pth'
    if not ckpt.exists():
        raise RuntimeError('no checkpoint; run --stage train first')
    outs = {}
    for split in ('valid', 'test'):
        out = d / 'beam' / f'{split}.npz'
        argv = [
            sys.executable, '-m', 'src.tiger.dump_beams',
            '--dataset_path', str(rp.ARTIFACT_ROOT / 'data' / ds),
            '--code_path', str(sid_dir / 'sid.npy'),
            '--ckpt_path', str(ckpt),
            '--output_path', str(out),
            '--split', split, '--sid_len', str(L),
            '--codebook_size', str(cfg['sid']['codebook_size']),
            '--beam_size', str(cfg.get('generation', {}).get('beam_size', 30)),
            '--device', 'cuda',
        ]
        p = subprocess.run(argv, cwd=str(REPO))
        if p.returncode != 0:
            raise RuntimeError(f'beam dump failed for {split}')
        outs[split] = str(out)
    return outs


def _load_dump(path):
    import numpy as np
    return np.load(path, allow_pickle=True)


def stage_evaluate(cfg: dict, d: Path, resume: bool) -> dict:
    import numpy as np
    from src.fusion.common import (_row_softmax, canonicalize_item_ids,
                                   per_user_ndcg, summarize)
    beam = d / 'beam' / 'test.npz'
    if not beam.exists():
        raise RuntimeError('no test beam dump; run --stage beam first')
    z = _load_dump(beam)
    items = canonicalize_item_ids(z['candidate_item_ids'])
    tgt = canonicalize_item_ids(z['target_item_ids'])
    sc = np.asarray(z['sequence_scores'], dtype=np.float64)
    pr = _row_softmax(sc, items >= 0)
    n, B = items.shape
    U = np.full((n, B), -1, np.int64)
    st = np.zeros((n, B, 1))
    for i in range(n):
        pos, p = {}, 0
        for j in range(B):
            it = int(items[i, j])
            if it < 0:
                continue
            if it not in pos:
                pos[it] = p
                U[i, p] = it
                p += 1
            st[i, pos[it], 0] += pr[i, j]
    tp = np.full(n, -1, np.int64)
    for i in range(n):
        w = np.where(U[i] == tgt[i])[0]
        if len(w):
            tp[i] = w[0]
    metrics = summarize(per_user_ndcg(U, st, tp, U >= 0, np.ones((n, 1))))
    out = d / 'metrics' / 'test_metrics.json'
    out.write_text(json.dumps(metrics, indent=2))
    return metrics


def stage_fusion(cfg: dict, d: Path, resume: bool) -> dict:
    """Uniform fusion needs >1 granularity. With a single K declared in this
    config the fused score equals that model's score; the cross-K ensemble is
    built by referencing several run_ids (see docs/EXPERIMENT_PROTOCOL.md)."""
    m = stage_evaluate(cfg, d, resume)
    out = d / 'metrics' / 'fusion_metrics.json'
    out.write_text(json.dumps({'method': cfg.get('fusion', {}).get('method', 'uniform'),
                               'n_models': 1, 'metrics': m}, indent=2))
    return {'method': 'uniform', 'n_models': 1, 'metrics': m}


STAGE_FN = {
    'sid': stage_sid, 'train': stage_train, 'beam': stage_beam,
    'evaluate': stage_evaluate, 'fusion': stage_fusion,
}
ORDER = ('sid', 'train', 'beam', 'evaluate', 'fusion')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python3 -m src.runner')
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage', default='all', choices=list(STAGES))
    ap.add_argument('--resume', action='store_true',
                    help='reuse an existing checkpoint; only valid for the SAME '
                         'run_id and config hash')
    ap.add_argument('--dry-run', action='store_true',
                    help='validate the config and print the plan, do nothing')
    args = ap.parse_args(argv)

    try:
        cfg = validate(load_config(args.config))
    except ConfigError as e:
        print(f'[runner] CONFIG ERROR\n{e}', file=sys.stderr)
        return 2

    rid = run_id(cfg)
    d = run_dir(cfg)
    print(f'[runner] run_id   {rid}')
    print(f'[runner] run_dir  {d}')
    print(f'[runner] dataset  {cfg["experiment"]["dataset"]}  '
          f'K={cfg["sid"]["k"]}  L={cfg["sid"]["sid_length"]}  '
          f'vocab={cfg["sid"]["vocab_size"]}  '
          f'alignment={cfg["sid"]["alignment_mode"]}')
    if args.dry_run:
        print('[runner] dry-run: config valid, nothing executed')
        return 0

    d = ensure_run_layout(cfg)
    stages = ORDER if args.stage == 'all' else (args.stage,)
    run = RunMetadata(stage=f'runner:{args.stage}', config=cfg,
                      inputs=[rp.item_embedding(cfg['experiment']['dataset'])],
                      out_dir=d)
    results, failed = {}, None
    try:
        for s in stages:
            print(f'[runner] ===== stage {s} =====')
            results[s] = STAGE_FN[s](cfg, d, args.resume)
    except Exception as e:                                   # noqa: BLE001
        failed = f'{type(e).__name__}: {e}'
        print(f'[runner] FAILED at stage {s}: {failed}', file=sys.stderr)

    man = manifest(d)
    meta = run.finish(outputs=[d], extra={'stages': list(stages),
                                          'results': results,
                                          'failure': failed})
    mpath = d / 'metrics' / 'test_metrics.json'
    registry_append({
        'run_id': rid, 'dataset': cfg['experiment']['dataset'],
        'method': cfg['sid'].get('method', 'spectral'),
        'k': cfg['sid']['k'], 'seed': cfg['experiment'].get('seed', 2025),
        'alignment_mode': cfg['sid'].get('alignment_mode', 'canonical'),
        'git_sha': run.data.get('git_commit'),
        'config_sha': cfg['_source_sha256'],
        'status': 'completed' if failed is None else 'failed',
        'failure_reason': failed,
        'metrics_path': str(mpath) if mpath.exists() else None,
        'run_dir': str(d), 'manifest': str(man),
    })
    print(f'[runner] registry  {REGISTRY}')
    if failed:
        return 1
    print('[runner] DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
