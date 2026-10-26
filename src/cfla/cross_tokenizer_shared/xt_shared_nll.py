#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — teacher-forced NLL under the SHARED checkpoint, and Only-factorization
NLL alignment.

Because both factorizations are scored by the SAME checkpoint with the SAME
parameters and the SAME sequence length, the raw TotalNLL values are directly
comparable (no cross-model calibration problem).  A validation-split mean/std is
still recorded per factorization so a normalized variant can be reported.

Outputs -> step2_shared_model/
  shared_target_nll.parquet    per user x factorization
  shared_nll_alignment.csv     Only-F groups vs argmin TotalNLL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, CODEBOOK, DATASETS, PRIMARY, STEP2, TOKENIZERS,
                       load_sid, shared_vocab_plan)  # noqa: E402
from xt_train_shared import ControlPrefixedDataset  # noqa: E402
from src.analysis.teacher_forced import (load_model_from_ckpt,
                                         teacher_forced_logprobs)  # noqa: E402
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402

N_PERM = 1000
PERM_SEED = 20260912


def score(model, ckpt_vocab, ds, f, split, ctrl):
    sid = load_sid(f, ds)
    L = sid.shape[1]
    base = GenRecDataset(dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
                         code_path=str(TOKENIZERS[f]['sid'](ds)), mode='evaluation',
                         max_len=20, codebook_size=CODEBOOK)
    dset = ControlPrefixedDataset(base, ctrl)
    loader = GenRecDataLoader(dset, batch_size=64, shuffle=False)
    n = len(dset)
    logp = np.zeros((n, L), dtype=np.float64)
    p = 0
    for b in loader:
        hb = b['history'].numpy()
        tb = b['target'].numpy()
        am = b['attention_mask'].numpy()
        logp[p:p + hb.shape[0]] = teacher_forced_logprobs(
            model, hb, tb, device='cuda', attention_mask=am)
        p += hb.shape[0]
    return -logp


def permutation_acc(argmin, m, observed):
    rng = np.random.default_rng(PERM_SEED)
    n = len(argmin)
    if n == 0:
        return float('nan'), float('nan')
    accs = np.empty(N_PERM)
    for i in range(N_PERM):
        accs[i] = (rng.integers(0, m, size=n) == argmin).mean()
    return float(accs.mean()), float((accs >= observed).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--factorizations', default=':'.join(PRIMARY))
    ap.add_argument('--datasets', default=','.join(DATASETS))
    a = ap.parse_args(argv)
    facs = a.factorizations.split(':')

    frames, cal_rows, align_rows = [], [], []
    for ds in a.datasets.split(','):
        ckpt = STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth'
        if not ckpt.exists():
            print(f'  {ds}: shared checkpoint missing ({ckpt})')
            continue
        plan = shared_vocab_plan(facs)
        model, vocab = load_model_from_ckpt(str(ckpt))
        print(f'=== {ds} shared ckpt vocab={vocab} ===', flush=True)
        per = {}
        for f in facs:
            ctrl = plan['control_tokens'][f]
            vnll = score(model, vocab, ds, f, 'valid', ctrl)
            tnll = score(model, vocab, ds, f, 'test', ctrl)
            per[f] = tnll
            vt = vnll.sum(axis=1)
            vmean, vstd = float(vt.mean()), float(vt.std(ddof=1))
            cal_rows.append({'dataset': ds, 'factorization': f,
                             'valid_total_nll_mean': vmean,
                             'valid_total_nll_std': vstd, 'valid_n': int(len(vt))})
            L = tnll.shape[1]
            tot = tnll.sum(axis=1)
            frames.append(pd.DataFrame({
                'dataset': ds, 'factorization': f, 'user_index': np.arange(len(tot)),
                'L': L, 'total_nll': tot,
                'semantic_nll': tnll[:, :L - 1].sum(axis=1),
                'disambiguation_nll': tnll[:, L - 1],
                'mean_token_nll': tnll.mean(axis=1),
                'valid_total_nll_mean': vmean, 'valid_total_nll_std': vstd,
                'z_total_nll': (tot - vmean) / vstd if vstd > 0 else np.nan}))
            print(f'  {f}: valid mean={vmean:.4f} test mean={tot.mean():.4f}', flush=True)
        del model
        torch.cuda.empty_cache()

        # ---- Only-F alignment ----
        gp = STEP2 / 'shared_only_groups.parquet'
        if gp.exists() and len(facs) == 2:
            g = pd.read_parquet(gp)
            g = g[g['dataset'] == ds]
            if len(g):
                mat = np.stack([per[facs[0]].sum(axis=1), per[facs[1]].sum(axis=1)], axis=1)
                argmin = mat.argmin(axis=1)
                grp = g['group'].to_numpy()
                for gi, lab in ((0, f'Only-{facs[0]}'), (1, f'Only-{facs[1]}')):
                    sel = grp == lab
                    if sel.sum() == 0:
                        continue
                    acc = float((argmin[sel] == gi).mean())
                    nm, pv = permutation_acc(argmin[sel], 2, acc)
                    align_rows.append({
                        'dataset': ds, 'group': lab, 'only_factorization': facs[gi],
                        'n_users': int(sel.sum()),
                        'mean_nll_own': float(mat[sel, gi].mean()),
                        'mean_nll_other': float(mat[sel, 1 - gi].mean()),
                        'median_nll_own': float(np.median(mat[sel, gi])),
                        'median_nll_other': float(np.median(mat[sel, 1 - gi])),
                        'own_minus_other_mean': float(mat[sel, gi].mean() - mat[sel, 1 - gi].mean()),
                        'argmin_match_accuracy': acc,
                        'permutation_null_mean': nm, 'permutation_p_value': pv,
                        'random_baseline': 0.5,
                        'global_cheaper_factorization': facs[int(mat[:, 0].mean() > mat[:, 1].mean())],
                        'global_baseline_accuracy': float(
                            (argmin[sel] == int(mat[:, 0].mean() > mat[:, 1].mean())).mean()),
                    })
                sel_any = (grp == f'Only-{facs[0]}') | (grp == f'Only-{facs[1]}')
                own = np.where(grp == f'Only-{facs[0]}', 0, 1)
                if sel_any.sum():
                    acc = float((argmin[sel_any] == own[sel_any]).mean())
                    nm, pv = permutation_acc(argmin[sel_any], 2, acc)
                    align_rows.append({
                        'dataset': ds, 'group': 'BOTH_Only_groups',
                        'only_factorization': 'pooled', 'n_users': int(sel_any.sum()),
                        'mean_nll_own': float(mat[sel_any, own[sel_any]].mean()),
                        'mean_nll_other': float(mat[sel_any, 1 - own[sel_any]].mean()),
                        'median_nll_own': float(np.median(mat[sel_any, own[sel_any]])),
                        'median_nll_other': float(np.median(mat[sel_any, 1 - own[sel_any]])),
                        'own_minus_other_mean': float(
                            mat[sel_any, own[sel_any]].mean() - mat[sel_any, 1 - own[sel_any]].mean()),
                        'argmin_match_accuracy': acc,
                        'permutation_null_mean': nm, 'permutation_p_value': pv,
                        'random_baseline': 0.5,
                        'global_cheaper_factorization': facs[int(mat[:, 0].mean() > mat[:, 1].mean())],
                        'global_baseline_accuracy': float(
                            (argmin[sel_any] == int(mat[:, 0].mean() > mat[:, 1].mean())).mean()),
                    })

    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(
            STEP2 / 'shared_target_nll.parquet', index=False)
        print('wrote shared_target_nll.parquet')
    pd.DataFrame(cal_rows).to_csv(STEP2 / 'shared_nll_calibration.csv', index=False)
    pd.DataFrame(align_rows).to_csv(STEP2 / 'shared_nll_alignment.csv', index=False)
    print(f'wrote shared_nll_alignment.csv ({len(align_rows)} rows)')
    if align_rows:
        print(pd.DataFrame(align_rows)[
            ['dataset', 'group', 'n_users', 'mean_nll_own', 'mean_nll_other',
             'argmin_match_accuracy', 'permutation_null_mean',
             'permutation_p_value']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
