#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provenance for the cross-tokenizer / shared-parameter study.

Writes
  outputs/cross_tokenizer_shared/_PROVENANCE.json
  outputs/cross_tokenizer_shared/step1_cross_tokenizer/_PROVENANCE.json
  outputs/cross_tokenizer_shared/step2_shared_model/_PROVENANCE.json   (if step 2 ran)

Records git state, environment, every input artifact with its sha256, and every
output with its sha256, plus an explicit statement of what was and was not run.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, DATASETS, OUT, PRIMARY, SECONDARY, STEP1, STEP2,
                       TOKENIZERS, beam_path, ckpt_path, sha256_file,
                       sid_path)  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]


def git(*a):
    return subprocess.run(['git', *a], cwd=str(REPO_ROOT),
                          capture_output=True, text=True).stdout.strip()


def env_block():
    import torch
    return {
        'host': platform.node(), 'platform': platform.platform(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                        else None),
        'cuda_version': getattr(torch.version, 'cuda', None),
    }


def hash_outputs(root):
    out = {}
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and p.name != '_PROVENANCE.json':
            out[str(p.relative_to(root))] = {
                'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    return out


def input_block():
    b = {'tokenizers': {}, 'split_sizes': {}, 'data_files': {}}
    for tok in TOKENIZERS:
        b['tokenizers'][tok] = {}
        for ds in DATASETS:
            sid_p, ck = sid_path(tok, ds), ckpt_path(tok, ds)
            b['tokenizers'][tok][ds] = {
                'sid_path': str(sid_p), 'sid_sha256': sha256_file(sid_p),
                'recommender_ckpt': str(ck), 'ckpt_sha256': sha256_file(ck),
                'readonly': True,
            }
    for ds in DATASETS:
        for split in ('train', 'valid', 'test'):
            p = ART / 'data' / ds / f'{split}.parquet'
            b['data_files'][f'{ds}/{split}.parquet'] = {
                'path': str(p), 'sha256': sha256_file(p)}
    # every beam dump this study uses or produced, with its split made explicit
    b['beams'] = {}
    for tok in TOKENIZERS:
        for ds in DATASETS:
            for split in ('test', 'valid'):
                pth = beam_path(tok, ds, split)
                if pth.exists():
                    b['beams'][f'{ds}|{tok}|{split}'] = {
                        'path': str(pth), 'sha256': sha256_file(pth),
                        'dataset': ds, 'tokenizer': tok, 'split': split,
                        'L': TOKENIZERS[tok]['L'],
                        'codebook': TOKENIZERS[tok]['codebook'],
                    }
    # rqvae tokenizer checkpoints
    b['rqvae_tokenizer_ckpts'] = {}
    for ds in DATASETS:
        found = TOKENIZERS['rqvae3'].get('tokenizer_ckpt')
        if found:
            f = found(ds)
            b['rqvae_tokenizer_ckpts'][ds] = [str(x) for x in f][-1:]
    return b


def shared_models_block():
    runs = {}
    base = STEP2 / 'runs'
    if not base.exists():
        return runs
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        ck = d / 'shared_model.pth'
        entry = {'run_dir': str(d),
                 'checkpoint': str(ck), 'checkpoint_sha256': sha256_file(ck),
                 'train_log': str(d / 'train_log.json'),
                 'config': ''}
        cfg = STEP2 / 'configs' / f"{d.name.split('_shared')[0]}_shared.yaml"
        if cfg.exists():
            raw = cfg.read_text()
            entry['config'] = json.loads(raw)
            entry['config_hash'] = hashlib.sha256(
                json.dumps(json.loads(raw), sort_keys=True).encode()).hexdigest()
        tl = d / 'train_log.json'
        if tl.exists():
            j = json.loads(tl.read_text())
            entry['epochs_completed'] = len(j.get('epochs', []))
            entry['best_epoch'] = j.get('best_epoch')
            entry['best_macro_valid_ndcg20'] = j.get('best_macro_valid_ndcg20')
            entry['train_size_total'] = j.get('train_size_total')
            entry['params'] = j.get('params')
        runs[d.name] = entry
    return runs


def main():
    step1 = {
        'study': 'step1_cross_tokenizer',
        'primary_comparison': list(PRIMARY),
        'secondary': list(SECONDARY),
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git': {'branch': git('branch', '--show-current'),
                'sha': git('rev-parse', 'HEAD'),
                'dirty': bool(git('status', '--porcelain'))},
        'environment': env_block(),
        'artifact_root_readonly': str(ART),
        'inputs': input_block(),
        'governance': {
            'new_recommender_models_trained_for_step1': 0,
            'beam_dumps_generated_inference_only': True,
            'historical_artifacts_modified': False,
            'test_set_selection': False,
            'hyperparameters_tuned': 0,
            'paper_read_or_written': False,
        },
        'outputs': hash_outputs(STEP1),
    }
    (STEP1 / '_PROVENANCE.json').write_text(json.dumps(step1, indent=2, default=str))
    print(f'wrote step1/_PROVENANCE.json ({len(step1["outputs"])} outputs)')

    step2 = {
        'study': 'step2_shared_model',
        'factorizations': list(PRIMARY),
        'shared_architecture': 'one encoder, one decoder, one embedding table, one '
                               'output head; only a reserved control token and the '
                               'factorization-specific SID sequence differ',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git': step1['git'], 'environment': step1['environment'],
        'shared_models': shared_models_block(),
        'governance': {
            'factorization_specific_decoder_or_head': False,
            'target_information_in_control_token': False,
            'test_used_for_selection': False,
            'historical_split_modified': False,
        },
        'outputs': hash_outputs(STEP2),
    }
    (STEP2 / '_PROVENANCE.json').write_text(json.dumps(step2, indent=2, default=str))
    print(f'wrote step2/_PROVENANCE.json ({len(step2["outputs"])} outputs, '
          f'{len(step2["shared_models"])} shared runs)')

    root = {
        'study': 'cross_tokenizer_shared',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git': step1['git'], 'environment': step1['environment'],
        'step1': 'step1_cross_tokenizer/_PROVENANCE.json',
        'step2': 'step2_shared_model/_PROVENANCE.json',
        'step1_governance': step1['governance'],
        'step2_governance': step2['governance'],
        'artifacts': hash_outputs(OUT),
    }
    (OUT / '_PROVENANCE.json').write_text(json.dumps(root, indent=2, default=str))
    print(f'wrote _PROVENANCE.json ({len(root["artifacts"])} artifacts)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
