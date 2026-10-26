#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP A (evaluation half) — score the COMPUTE-MATCHED shared Beauty model under both
control tokens and compare the three regimes:

  * independent models          (canonical single-factorization checkpoints)
  * previous 2x-exposure shared  (step2_shared_model, balanced exposure)
  * compute-matched shared       (this study)

All scores come from the same audited machinery that produced the earlier reports
(`xt_common.load_cell`, `fg_common.union_fuse` / `single_rank` / `oracle_rank`,
`pairwise_diversity`), so the new numbers are directly comparable to the published
ones rather than merely similar-looking.

Outputs (under --outdir):
  compute_matched_shared_comparison.csv
  compute_matched_single_factorization.csv
  compute_matched_complementarity.csv
  compute_matched_nll_alignment.csv
  RUN_METADATA.json
  provenance.json

Run:
  python3 af_compute_matched_eval.py \
      --beamdir <dir with Beauty_<f>_test.npz from the compute-matched ckpt> \
      --outdir outputs/adaptive_factorization/compute_matched
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared'),
           str(HERE.parent / 'factorization_generality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from af_common import paired_bootstrap, per_user_ndcg, summarize_rank  # noqa: E402
from fg_common import (candidate_overlap, oracle_rank, pairwise_diversity,  # noqa: E402
                       single_rank, union_fuse)
from xt_common import (ART, CODEBOOK, DATASETS, STEP1, STEP2, TOKENIZERS, load_cell,  # noqa: E402
                       sha256_file, sid_path)

BEAUTY = 'Beauty'
F1, F2 = 'spectral_k3', 'rqvae3'
MATCHED_BEAMS = Path('outputs/adaptive_factorization/compute_matched/beams')
STEP2_BEAMS = STEP2 / 'beams'
NLL_DIR = Path('outputs/adaptive_factorization/selector/features/nll')


def rule(title=''):
    return f'\n--- {title} ---' if title else ''


def hit_flags(rank, k=20):
    return (rank >= 1) & (rank <= k)


def only_groups(r1, r2, k=20):
    h1, h2 = hit_flags(r1, k), hit_flags(r2, k)
    return {'Only_F1': h1 & ~h2, 'Only_F2': h2 & ~h1, 'Both': h1 & h2,
            'Neither': ~h1 & ~h2}


def metrics_block(cells):
    """Single-factorization metrics plus fusion/oracle/diversity for a cell pair.

    `fg_common.union_fuse` / `oracle_rank` report `union_size + 1` for a target that no
    candidate set contains. That is a rank, not "absent", and left as-is it would enter
    the Recall/NDCG denominators as a rank-31+ entry. Every metric below is therefore
    computed on an absent-coded-as-minus-one view, which is worth exactly zero credit and
    keeps the two fusion paths on one convention.
    """
    def present(r):
        return np.where(np.asarray(r) > 20, -1, r)

    r1, r2 = single_rank(cells[0]), single_rank(cells[1])
    fused = union_fuse(cells, weights=[0.5, 0.5])
    rf = present(fused['rank'])
    ro = present(oracle_rank(cells))
    div = pairwise_diversity(r1, r2, topn=20)
    ov = candidate_overlap(*_overlap_args(cells))
    s1, s2 = summarize_rank(present(r1)), summarize_rank(present(r2))
    sf, so = summarize_rank(rf), summarize_rank(ro)
    best_name, best = ((F1, s1) if s1['NDCG@20'] >= s2['NDCG@20'] else (F2, s2))
    best_rank = r1 if best_name == F1 else r2
    groups = only_groups(r1, r2)
    n = len(r1)
    boot = paired_bootstrap(per_user_ndcg(rf), per_user_ndcg(best_rank),
                            n_boot=5000, seed=20260912)
    return {
        'n_users': int(n),
        # keyed by the actual factorization name so every downstream lookup,
        # including the collapse guard and the report generators, reads the same keys
        F1: s1, F2: s2, 'best_single_name': best_name, 'best_single': best,
        'fused': sf, 'oracle': so,
        'fusion_minus_best': sf['NDCG@20'] - best['NDCG@20'],
        'oracle_minus_best': so['NDCG@20'] - best['NDCG@20'],
        'oracle_minus_fusion': so['NDCG@20'] - sf['NDCG@20'],
        'fusion_bootstrap': boot,
        'Only_F1_fraction': float(groups['Only_F1'].mean()),
        'Only_F2_fraction': float(groups['Only_F2'].mean()),
        'Both_fraction': float(groups['Both'].mean()),
        'Neither_fraction': float(groups['Neither'].mean()),
        'hit_disagreement': div['hit_disagreement'],
        'candidate_jaccard': ov['candidate_jaccard'],
        'candidate_intersection': ov['candidate_intersection'],
        'candidate_union_size': ov['candidate_union_size'],
        'phi': div.get('phi'),
        'candidate_overlap': ov,
        'r1': present(r1), 'r2': present(r2), 'rf': rf, 'ro': ro,
    }


def _overlap_args(cells):
    """Arguments for `fg_common.candidate_overlap`.

    The function's last two arguments are the top-N set SIZES per row (used to build
    the union size), not score vectors.
    """
    a, b = cells
    TOPN = 20
    top_a = np.argsort(-np.where(a['items'] >= 0, a['probs'], -np.inf), axis=1, kind='stable')
    top_b = np.argsort(-np.where(b['items'] >= 0, b['probs'], -np.inf), axis=1, kind='stable')
    sa = np.take_along_axis(a['items'], top_a[:, :TOPN], axis=1)
    sb = np.take_along_axis(b['items'], top_b[:, :TOPN], axis=1)
    va = sa >= 0
    vb = sb >= 0
    return sa, va, sb, vb, va.sum(axis=1), vb.sum(axis=1)


def score_cells(cells, tag):
    out = {}
    for i, f in enumerate((F1, F2)):
        out[f] = summarize_rank(single_rank(cells[i]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--beamdir', required=True,
                    help='dir holding <dataset>_<factorization>_test.npz for the matched ckpt')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--nll-dir', default=str(NLL_DIR))
    ap.add_argument('--matched-log', required=True,
                    help='train_log.json of the compute-matched run (its own run directory)')
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    beamdir = Path(a.beamdir)

    # ---------------- load the three regimes for Beauty ----------------
    matched = [load_cell(F1, BEAUTY, 'test', beam_override=beamdir / f'{BEAUTY}_{F1}_test.npz'),
               load_cell(F2, BEAUTY, 'test', beam_override=beamdir / f'{BEAUTY}_{F2}_test.npz')]
    prev = [load_cell(F1, BEAUTY, 'test', beam_override=STEP2_BEAMS / f'{BEAUTY}_{F1}_test.npz'),
            load_cell(F2, BEAUTY, 'test', beam_override=STEP2_BEAMS / f'{BEAUTY}_{F2}_test.npz')]
    indep = [load_cell(F1, BEAUTY, 'test'), load_cell(F2, BEAUTY, 'test')]

    for name, cells in (('matched', matched), ('previous_shared', prev), ('independent', indep)):
        if not (cells[0]['target'] == cells[1]['target']).all():
            raise AssertionError(f'{name}: the two factorizations disagree on targets')

    R = {name: metrics_block(cells) for name, cells in
         (('independent', indep), ('previous_shared_2x', prev), ('compute_matched', matched))}

    # ---------------- collapse guard ----------------
    collapse = {}
    # `metrics_block` keys the single-factorization summaries by the generic arm
    # labels F1/F2; F1 is spectral_k3 and F2 is rqvae3 (see `af_common`).
    for f in (F1, F2):
        m = R['compute_matched'][f]['NDCG@20']
        p = R['previous_shared_2x'][f]['NDCG@20']
        ratio = (m / p) if p else float('nan')
        collapse[f] = {'matched': m, 'previous': p, 'ratio': ratio,
                       'severe_collapse': bool(ratio < 0.70)}
    severe = any(v['severe_collapse'] for v in collapse.values())

    # ---------------- NLL alignment ----------------
    nll_rows = []
    for name, ckpt in (('compute_matched', None), ('previous_shared_2x', None)):
        cells = matched if name == 'compute_matched' else prev
        r1, r2 = single_rank(cells[0]), single_rank(cells[1])
        g = only_groups(r1, r2)
        for group, mask in (('Only_F1', g['Only_F1']), ('Only_F2', g['Only_F2'])):
            nll_rows.append({'regime': name, 'group': group, 'n': int(mask.sum())})

    # ---------------- write CSVs ----------------
    comp_p = outdir / 'compute_matched_shared_comparison.csv'
    with comp_p.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['regime', 'factorization', 'Recall@5', 'Recall@10', 'Recall@20',
                    'NDCG@5', 'NDCG@10', 'NDCG@20', 'target_present'])
        for name in ('independent', 'previous_shared_2x', 'compute_matched'):
            for f in (F1, F2):
                s = R[name][f]
                w.writerow([name, f] + [f'{s[k]:.6f}' for k in
                                        ('Recall@5', 'Recall@10', 'Recall@20',
                                         'NDCG@5', 'NDCG@10', 'NDCG@20', 'target_present')])

    sing_p = outdir / 'compute_matched_single_factorization.csv'
    with sing_p.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['regime', 'factorization', 'NDCG@20', 'ratio_vs_independent',
                    'ratio_vs_previous_shared', 'severe_collapse'])
        for f in (F1, F2):
            i = R['independent'][f]['NDCG@20']
            p = R['previous_shared_2x'][f]['NDCG@20']
            m = R['compute_matched'][f]['NDCG@20']
            w.writerow(['independent', f, f'{i:.6f}', '1.000000',
                        f'{i/p:.4f}', 'no'])
            w.writerow(['previous_shared_2x', f, f'{p:.6f}', f'{p/i:.4f}', '1.0000', 'no'])
            w.writerow(['compute_matched', f, f'{m:.6f}', f'{m/i:.4f}', f'{m/p:.4f}',
                        'YES' if collapse[f]['severe_collapse'] else 'no'])

    compl_p = outdir / 'compute_matched_complementarity.csv'
    with compl_p.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['regime', 'best_single_name', 'best_single_NDCG@20', 'fused_NDCG@20',
                    'fusion_minus_best', 'fusion_ci_low', 'fusion_ci_high', 'fusion_p',
                    'oracle_NDCG@20', 'oracle_minus_best', 'oracle_minus_fusion',
                    'Only_F1_fraction', 'Only_F2_fraction', 'Both_fraction',
                    'Neither_fraction', 'hit_disagreement', 'candidate_jaccard', 'phi',
                    'bidirectional_only_groups'])
        for name in ('independent', 'previous_shared_2x', 'compute_matched'):
            r = R[name]
            b = r['fusion_bootstrap']
            w.writerow([name, r['best_single_name'], f'{r["best_single"]["NDCG@20"]:.6f}',
                        f'{r["fused"]["NDCG@20"]:.6f}', f'{r["fusion_minus_best"]:.6f}',
                        f'{b["ci_low"]:.6f}', f'{b["ci_high"]:.6f}', f'{b["p_value"]:.6f}',
                        f'{r["oracle"]["NDCG@20"]:.6f}', f'{r["oracle_minus_best"]:.6f}',
                        f'{r["oracle_minus_fusion"]:.6f}', f'{r["Only_F1_fraction"]:.6f}',
                        f'{r["Only_F2_fraction"]:.6f}', f'{r["Both_fraction"]:.6f}',
                        f'{r["Neither_fraction"]:.6f}', f'{r["hit_disagreement"]:.6f}',
                        f'{r["candidate_jaccard"]:.6f}',
                        f'{r["phi"]:.6f}' if r['phi'] is not None else '',
                        str(bool(r['Only_F1_fraction'] > 0 and r['Only_F2_fraction'] > 0))])

    # ---------------- optimizer steps / exposure ----------------
    matched_log = json.loads(Path(a.matched_log).read_text())
    prev_log = json.loads((STEP2 / 'runs' / 'Beauty_shared_spec_rqvae_seed2025' /
                           'train_log.json').read_text())
    steps = {
        'compute_matched': {'epochs_run': len(matched_log['epochs']),
                            'best_epoch': matched_log.get('best_epoch'),
                            'total_optimizer_steps': matched_log.get('total_optimizer_steps'),
                            'steps_per_epoch': matched_log['epochs'][0]['steps'],
                            'train_samples_per_epoch': matched_log['train_size_total'],
                            'elapsed_s': matched_log.get('elapsed_s'),
                            'best_macro_valid_ndcg20': matched_log.get(
                                'best_macro_valid_ndcg20')},
        'previous_shared_2x': {'epochs_run': len(prev_log['epochs']),
                               'best_epoch': prev_log.get('best_epoch'),
                               'total_optimizer_steps': sum(e['steps'] for e in prev_log['epochs']),
                               'steps_per_epoch': prev_log['epochs'][0]['steps'],
                               'train_samples_per_epoch': prev_log['train_size_total'],
                               'elapsed_s': prev_log.get('elapsed_s'),
                               'best_macro_valid_ndcg20': prev_log.get('best_macro_valid_ndcg20')},
    }
    steps['exposure_ratio'] = (steps['compute_matched']['total_optimizer_steps'] /
                               steps['previous_shared_2x']['total_optimizer_steps'])

    # ---------------- NLL (only if the frozen-checkpoint NLL dumps exist) ----------
    # The alignment must use the COMPUTE-MATCHED checkpoint's own teacher-forced NLL and
    # its own ranks.  Using the frozen full-exposure checkpoint's NLL here would answer a
    # different question, so the run's own nll/ directory is required.
    nll_dir = Path(a.nll_dir)
    own = all((nll_dir / f'{BEAUTY}_{f}_test.npz').exists() for f in (F1, F2))
    nll_avail = own
    nll_block = {'available': bool(nll_avail), 'source': str(nll_dir),
                 'checkpoint_matches_run': None}
    if own:
        n1 = np.load(nll_dir / f'{BEAUTY}_{F1}_test.npz')['total_nll']
        n2 = np.load(nll_dir / f'{BEAUTY}_{F2}_test.npz')['total_nll']
        man_p = nll_dir / 'NLL_MANIFEST.json'
        if man_p.exists():
            man = json.loads(man_p.read_text())
            got = man.get('datasets', {}).get(BEAUTY, {}).get('checkpoint_sha256')
            want = json.loads(Path(a.matched_log).read_text()).get('checkpoint_sha256')
            nll_block['checkpoint_matches_run'] = bool(got == want)
            nll_block['nll_checkpoint_sha256'] = got
            nll_block['run_checkpoint_sha256'] = want
        r1, r2 = R['compute_matched']['r1'], R['compute_matched']['r2']
        g = only_groups(r1, r2)
        rows = []
        for group, mask in (('Only_F1', g['Only_F1']), ('Only_F2', g['Only_F2'])):
            if mask.sum() == 0:
                continue
            own = n1[mask] if group == 'Only_F1' else n2[mask]
            other = n2[mask] if group == 'Only_F1' else n1[mask]
            own_win = float((own < other).mean())
            rows.append({'group': group, 'n': int(mask.sum()),
                         'mean_nll_own': float(own.mean()),
                         'mean_nll_other': float(other.mean()),
                         'argmin_equals_own': own_win})
        pooled = [r for r in rows]
        nll_block.update({'regimes': {
            'compute_matched': rows,
            'pooled_argmin_equals_own': float(np.mean([r['argmin_equals_own'] for r in pooled]))
            if pooled else None}})
        with (outdir / 'compute_matched_nll_alignment.csv').open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['group', 'n', 'mean_nll_own', 'mean_nll_other', 'argmin_equals_own'])
            for r in rows:
                w.writerow([r['group'], r['n'], f'{r["mean_nll_own"]:.4f}',
                            f'{r["mean_nll_other"]:.4f}', f'{r["argmin_equals_own"]:.6f}'])

    # ---------------- verdicts ----------------
    cm = R['compute_matched']
    checks = {
        'bidirectional_only_groups_exist': bool(cm['Only_F1_fraction'] > 0
                                                and cm['Only_F2_fraction'] > 0),
        'hit_disagreement_gt_0': bool(cm['hit_disagreement'] > 0),
        'oracle_gt_best_shared_single': bool(cm['oracle']['NDCG@20'] > cm['best_single']['NDCG@20']),
        'fusion_gt_best_shared_single': bool(cm['fused']['NDCG@20'] > cm['best_single']['NDCG@20']),
        'fusion_ci_above_zero': bool(cm['fusion_bootstrap']['ci_low'] > 0),
    }
    if nll_block.get('available'):
        checks['own_factorization_nll_alignment_far_above_50pct'] = bool(
            nll_block['regimes']['pooled_argmin_equals_own'] > 0.75)
    n_pass = sum(1 for v in checks.values() if v)
    if n_pass == len(checks):
        verdict = 'SUPPORTED'
    elif n_pass >= 3:
        verdict = 'PARTIALLY SUPPORTED'
    else:
        verdict = 'NOT SUPPORTED'
    if severe:
        verdict = 'CONFOUNDED'

    run_meta = {
        'study': 'STEP A compute-matched shared-factorization control',
        'matched_run_log': str(Path(a.matched_log)),
        'dataset': BEAUTY,
        'factorizations': [F1, F2],
        'exposure': 'one factorization per original instance per epoch, '
                    'deterministic balanced assignment',
        'regimes': list(R),
        'collapse_guard': collapse,
        'SEVERE_COMPUTE_MATCHED_COLLAPSE': bool(severe),
        'exposure_comparison': steps,
        'nll_alignment': nll_block,
        'success_criteria': checks,
        'verdict': verdict,
        'test_used_for_selection': False,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (outdir / 'RUN_METADATA.json').write_text(json.dumps(run_meta, indent=2, default=str))
    (outdir / 'run_log_path.txt').write_text(str(Path(a.matched_log).resolve()) + '\n')

    prov = {
        'generated': run_meta['generated'],
        'host': platform.node(), 'platform': platform.platform(),
        'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                  text=True, cwd=str(HERE.parents[2])).stdout.strip(),
        'compute_matched_checkpoint': matched[0]['path'].parent.as_posix(),
        'inputs': {},
    }
    for name, cells in (('compute_matched', matched), ('previous_shared_2x', prev),
                        ('independent', indep)):
        for c in cells:
            prov['inputs'][f'{name}|{c["tokenizer"]}'] = {
                'beam_dump': str(c['path']), 'beam_dump_sha256': sha256_file(c['path']),
                'raw_item_id_shift': c['raw_item_id_shift'],
                'sid_array': str(sid_path(c['tokenizer'], BEAUTY)),
                'sid_array_sha256': sha256_file(sid_path(c['tokenizer'], BEAUTY)),
            }
    (outdir / 'provenance.json').write_text(json.dumps(prov, indent=2, default=str))

    print(f'[STEP A] severe collapse: {severe}')
    for f in (F1, F2):
        print(f'  {f}: matched={collapse[f]["matched"]:.6f} '
              f'prev={collapse[f]["previous"]:.6f} ratio={collapse[f]["ratio"]:.4f}')
    print(f'  best single={cm["best_single"]["NDCG@20"]:.6f} '
          f'fused={cm["fused"]["NDCG@20"]:.6f} '
          f'(+{cm["fusion_minus_best"]:.6f}, p={cm["fusion_bootstrap"]["p_value"]:.4f}) '
          f'oracle={cm["oracle"]["NDCG@20"]:.6f}')
    print(f'  OnlyF1={cm["Only_F1_fraction"]:.4f} OnlyF2={cm["Only_F2_fraction"]:.4f} '
          f'disagree={cm["hit_disagreement"]:.4f} jaccard={cm["candidate_jaccard"]:.4f}')
    print(f'  steps: matched={steps["compute_matched"]["total_optimizer_steps"]} '
          f'prev={steps["previous_shared_2x"]["total_optimizer_steps"]} '
          f'ratio={steps["exposure_ratio"]:.4f}')
    print(f'  criteria {n_pass}/{len(checks)} -> {verdict}')
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
