#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE D — Partition-family control: Spectral vs Even vs Random.

No training was necessary: historical SID arrays, checkpoints and beam dumps
exist for all three families x 4 datasets x K=2,3,4,5 (verified by the inventory
below, written to PARTITION_ARTIFACT_INVENTORY.md).

Every family is evaluated with the SAME uniform item-level fusion rule and the
SAME canonical model/eval/beam/seed configuration. Only the SID partition method
differs. No family gets its own fusion parameters.

Outputs -> outputs/mechanism/phaseD_partition_control/
    PARTITION_ARTIFACT_INVENTORY.md
    single_k_metrics.csv            per family / dataset / K
    fusion_results.csv              best single, +K, all 2- and 3-model subsets
    partition_family_results.csv    (A) fusion - best single, per family
    partition_family_complementarity.csv  (D) cross-family hit disagreement,
                                          candidate overlap, oracle
    phaseD_bootstrap.csv            paired user bootstrap, 5000 resamples
    _PROVENANCE.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.per_user import (                                       # noqa: E402
    K_LIST, build_view, fuse_subset, oracle_ranks, paired_bootstrap,
    per_user_ndcg_vector, summarize_ranks, topn,
)
from src.utils import repo_paths as rp                                    # noqa: E402

OUT = REPO / 'outputs' / 'mechanism' / 'phaseD_partition_control'
DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
FAMILIES = {'spectral': 'nonneg_pearson_v3',
            'even': 'evensplit_v3',
            'random': 'randomsplit_v3'}
TOPN = 20


def inventory():
    rows = []
    for fam, variant in FAMILIES.items():
        for ds in DATASETS:
            for k in K_LIST:
                L = k + 1
                sid = (rp.ARTIFACT_ROOT / 'data' / ds /
                       f'{ds}_t5_spectral_{variant}_k{k}_sid{L}.npy')
                if variant != 'nonneg_pearson_v3':
                    sid = rp.ARTIFACT_ROOT / 'data' / ds / f'{ds}_t5_{variant}_k{k}_sid{L}.npy'
                ck = (rp.ARTIFACT_ROOT / 'model/ckpt' /
                      f'tiger_{ds}_{variant}_k{k}_sid{L}.pth')
                bm = (rp.ARTIFACT_ROOT / f'model/beam_dumps_{ds}_{variant}' /
                      f'test_k{k}_sid{L}.npz')
                rows.append({'family': fam, 'dataset': ds, 'K': k,
                             'sid': sid.exists(), 'checkpoint': ck.exists(),
                             'beam': bm.exists(),
                             'beam_path': str(bm.relative_to(rp.ARTIFACT_ROOT))})
    df = pd.DataFrame(rows)
    lines = ['# PARTITION_ARTIFACT_INVENTORY.md', '',
             'Inventory of historical artifacts for the partition-family control.',
             'Checked BEFORE any training. If a cell had been missing, only that',
             'cell would have been trained with the canonical recipe.', '',
             '| family | dataset | K | SID | checkpoint | beam |',
             '|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        lines.append(f"| {r['family']} | {r['dataset']} | {r['K']} | "
                     f"{'✅' if r['sid'] else '❌'} | "
                     f"{'✅' if r['checkpoint'] else '❌'} | "
                     f"{'✅' if r['beam'] else '❌'} |")
    missing = df[(~df.sid) | (~df.checkpoint) | (~df.beam)]
    lines += ['', f"**Complete cells: {len(df) - len(missing)} / {len(df)}.**"]
    if len(missing):
        lines += ['', '## Missing', '', '| family | dataset | K |', '|---|---|---|']
        for _, r in missing.iterrows():
            lines.append(f"| {r['family']} | {r['dataset']} | {r['K']} |")
    else:
        lines += ['', '**No training is required for PHASE D.** The historical',
                  'artifacts cover all three families at all four granularities on',
                  'all four datasets.']
    (OUT / 'PARTITION_ARTIFACT_INVENTORY.md').write_text('\n'.join(lines) + '\n')
    return df, missing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv, missing = inventory()
    print(f'inventory: {len(inv) - len(missing)}/{len(inv)} cells complete')
    if len(missing):
        print('INCOMPLETE -> see PARTITION_ARTIFACT_INVENTORY.md')

    prov = {'phase': 'D_partition_family_control', 'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'families': FAMILIES, 'fusion': 'uniform_item_level_average',
            'alignment_mode': 'canonical', 'inventory_complete': bool(len(missing) == 0),
            'inputs': {}}

    single_rows, fusion_rows, fam_rows, comp_rows, boot_rows = [], [], [], [], []

    for fam, variant in FAMILIES.items():
        for ds in DATASETS:
            print(f'=== {fam} / {ds} ===', flush=True)
            view = build_view(ds, variant=variant, ks=tuple(K_LIST))
            ks = view['ks']
            n = view['n_users']
            tgt = view['target']
            per_user = {k: view['models'][k]['per_user'] for k in ks}
            ranks = {k: view['models'][k]['ranks'] for k in ks}
            prov['inputs'][f'{fam}|{ds}'] = {
                str(k): {'path': str(view['models'][k]['path'].relative_to(rp.ARTIFACT_ROOT)),
                         'sha256': view['models'][k]['sha256']} for k in ks}

            # ---- single-K ----
            for k in ks:
                s = summarize_ranks(ranks[k])
                single_rows.append({'family': fam, 'dataset': ds, 'K': k,
                                    'n_users': n, **s})
            best_k = max(ks, key=lambda k: summarize_ranks(ranks[k])['NDCG@20'])
            best_single = summarize_ranks(ranks[best_k])

            # ---- subsets ----
            subsets = [(k,) for k in ks] + list(itertools.combinations(ks, 2)) + \
                      list(itertools.combinations(ks, 3)) + [tuple(ks)]
            fus = {}
            for S in subsets:
                r = ranks[S[0]] if len(S) == 1 else fuse_subset(view, S)[1]
                fus[S] = r
                s = summarize_ranks(r)
                fusion_rows.append({'family': fam, 'dataset': ds,
                                    'subset': '+'.join(map(str, S)),
                                    'n_models': len(S),
                                    'is_full': tuple(S) == tuple(ks), **s})
            full = fus[tuple(ks)]
            full_s = summarize_ranks(full)
            orc = summarize_ranks(oracle_ranks(view))

            # ---- (A) fusion - best single ----
            for nn in (10, 20):
                fam_rows.append({
                    'family': fam, 'dataset': ds, 'topN': nn,
                    'best_single_K': best_k,
                    'BestSingle_NDCG': best_single[f'NDCG@{nn}'],
                    'Fusion_NDCG': full_s[f'NDCG@{nn}'],
                    'Oracle_NDCG': orc[f'NDCG@{nn}'],
                    'fusion_minus_best_single': full_s[f'NDCG@{nn}'] - best_single[f'NDCG@{nn}'],
                    'oracle_minus_fusion': orc[f'NDCG@{nn}'] - full_s[f'NDCG@{nn}'],
                    'unique_prefix_fraction': np.nan,
                })

            # ---- (D) within-family pairwise hit disagreement / candidate overlap ----
            for a, b in itertools.combinations(ks, 2):
                ha = (ranks[a] >= 1) & (ranks[a] <= TOPN)
                hb = (ranks[b] >= 1) & (ranks[b] <= TOPN)
                both = int((ha & hb).sum()); oa = int((ha & ~hb).sum())
                ob = int((~ha & hb).sum()); ne = int((~ha & ~hb).sum())
                union = both + oa + ob
                js = []
                for i in range(n):
                    sa = set(topn(per_user[a][i], TOPN)); sb = set(topn(per_user[b][i], TOPN))
                    js.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0)
                comp_rows.append({
                    'comparison_type': 'within_family', 'family_a': fam, 'family_b': fam,
                    'dataset': ds, 'K_a': a, 'K_b': b, 'topN': TOPN,
                    'P_disagree': (oa + ob) / n,
                    'jaccard_hitset': both / union if union else np.nan,
                    'candidate_jaccard_mean': float(np.mean(js)),
                })

            # ---- (D) cross-family comparison at the same K ----
            for other in FAMILIES:
                if other == fam:
                    continue
                ov = build_view(ds, variant=FAMILIES[other], ks=tuple(K_LIST))
                for k in ks:
                    ra, rb = ranks[k], ov['models'][k]['ranks']
                    ha = (ra >= 1) & (ra <= TOPN); hb = (rb >= 1) & (rb <= TOPN)
                    both = int((ha & hb).sum()); oa = int((ha & ~hb).sum())
                    ob = int((~ha & hb).sum())
                    union = both + oa + ob
                    comp_rows.append({
                        'comparison_type': 'cross_family',
                        'family_a': fam, 'family_b': other,
                        'dataset': ds, 'K_a': k, 'K_b': k, 'topN': TOPN,
                        'P_disagree': (oa + ob) / n,
                        'jaccard_hitset': both / union if union else np.nan,
                        'candidate_jaccard_mean': np.nan,
                    })

            # ---- (B) family-vs-family bootstrap on the full fusion ----
            for other in FAMILIES:
                if other == fam or fam > other:
                    continue
                ov = build_view(ds, variant=FAMILIES[other], ks=tuple(K_LIST))
                of = fuse_subset(ov, tuple(K_LIST))[1]
                for nn in (20,):
                    r = paired_bootstrap(per_user_ndcg_vector(full, nn),
                                         per_user_ndcg_vector(of, nn))
                    boot_rows.append({
                        'dataset': ds, 'comparison': f'{fam}_vs_{other}',
                        'topN': nn, 'a_value': float(per_user_ndcg_vector(full, nn).mean()),
                        'b_value': float(per_user_ndcg_vector(of, nn).mean()),
                        'family_a': fam, 'family_b': other, **r, 'n_boot': 5000,
                        'n_users': n,
                        'significant_95': bool(r['ci_lo'] > 0 or r['ci_hi'] < 0)})

            print(f'  best single K={best_k}  fusion N@20={full_s["NDCG@20"]:.6f}  '
                  f'oracle N@20={orc["NDCG@20"]:.6f}', flush=True)

    for name, rows in [('single_k_metrics.csv', single_rows),
                       ('fusion_results.csv', fusion_rows),
                       ('partition_family_results.csv', fam_rows),
                       ('partition_family_complementarity.csv', comp_rows),
                       ('phaseD_bootstrap.csv', boot_rows)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
        print(f'  wrote {name} ({len(rows)} rows)')

    import subprocess
    prov['git_sha'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                     capture_output=True, text=True).stdout.strip()
    prov['training_performed'] = 0
    (OUT / '_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
