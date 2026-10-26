#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FINAL_SPORTS_CONFIRMATION_REPORT.md (harness sections 42-43) and
development_vs_sports.csv (section 37).

The four confirmatory hypotheses were fixed before Sports test was read:

    H1  LSE > Uniform                          on NDCG@20
    H2  Frozen Learned Gate > Uniform          on NDCG@20
    H3  Frozen Learned Gate > Frozen LSE       on NDCG@20   (stronger, secondary)

Claim policy (section 33), applied verbatim:
    Gate > Uniform and 95% CI > 0        -> ZERO-SHOT SPORTS CONFIRMATION: SUPPORTED
    Gate > Uniform but CI overlaps zero  -> PARTIALLY SUPPORTED
    Gate <= Uniform                      -> NOT SUPPORTED

Usage
  python3 sc_final_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# SC_OUT redirects the round-4 tree for smoke-testing, matching sc_pipeline.py,
# sc_gate_eval.py, sc_provenance.py and sc_dev_vs_sports.py.
OUT = Path(os.environ.get('SC_OUT') or (REPO / 'outputs' / 'sports_confirmation'))
DEV = REPO / 'outputs' / 'disagreement_gate'
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
DIDS = list(SHORT)


def rd(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def f(x, nd=6):
    try:
        return f'{float(x):.{nd}f}'
    except (TypeError, ValueError):
        return str(x)


def verdict_primary(boot, tres):
    """Exactly the pre-registered claim policy."""
    p = [r for r in boot if r['comparison'] == 'Gate_vs_Uniform' and r['metric'] == 'NDCG@20']
    g = [r for r in tres if r['method'] == 'Gate']
    u = [r for r in tres if r['method'] == 'Uniform']
    if not (p and g and u):
        return 'NOT ESTABLISHED', 'missing test artefacts'
    d = float(p[0]['delta']); lo = float(p[0]['ci_low']); hi = float(p[0]['ci_high'])
    det = (f'Gate - Uniform = {d:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}], '
           f'p={float(p[0]["p_value"]):.4f}')
    if d > 0 and lo > 0:
        return 'SUPPORTED', det
    if d > 0:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_lse(boot):
    p = [r for r in boot if r['comparison'] == 'LSE_vs_Uniform' and r['metric'] == 'NDCG@20']
    if not p:
        return 'NOT SUPPORTED', 'missing'
    d = float(p[0]['delta']); lo = float(p[0]['ci_low'])
    det = f'LSE - Uniform = {d:+.6f}, CI [{lo:+.6f}, {float(p[0]["ci_high"]):+.6f}], p={float(p[0]["p_value"]):.4f}'
    if d > 0 and lo > 0:
        return 'SUPPORTED', det
    if d > 0:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def verdict_gate_vs_lse(boot):
    p = [r for r in boot if r['comparison'] == 'Gate_vs_LSE' and r['metric'] == 'NDCG@20']
    if not p:
        return 'NOT SUPPORTED', 'missing'
    d = float(p[0]['delta']); lo = float(p[0]['ci_low'])
    det = f'Gate - LSE = {d:+.6f}, CI [{lo:+.6f}, {float(p[0]["ci_high"]):+.6f}], p={float(p[0]["p_value"]):.4f}'
    if d > 0 and lo > 0:
        return 'SUPPORTED', det
    if d > 0:
        return 'PARTIALLY SUPPORTED', det
    return 'NOT SUPPORTED', det


def main(argv=None):
    ap = argparse.ArgumentParser()
    a = ap.parse_args(argv)
    tres = rd(OUT / 'test' / 'test_results.csv')
    vres = rd(OUT / 'valid' / 'validation_metrics.csv')
    boot = rd(OUT / 'test' / 'bootstrap.csv')
    sub = rd(OUT / 'test' / 'subgroup_results.csv')
    mech = rd(OUT / 'test' / 'mechanism_diagnostics.csv')
    # Refuse to emit verdicts from incomplete artefacts.  The verdict helpers below
    # degrade gracefully to 'NOT SUPPORTED' / 'NOT ESTABLISHED' when a table is empty,
    # which is right for a *missing comparison* but catastrophic here: a missing or
    # truncated artefact would be published as "the frozen method failed on Sports".
    # A missing artefact is not a scientific result, so fail loudly instead.
    need = {'test/test_results.csv': tres, 'valid/validation_metrics.csv': vres,
            'test/bootstrap.csv': boot, 'test/subgroup_results.csv': sub,
            'test/mechanism_diagnostics.csv': mech}
    empty = [k for k, v in need.items() if not v]
    if empty:
        raise SystemExit(f'refusing to write verdicts: missing or empty artefacts {empty}')
    wanted = {('Gate_vs_Uniform', 'NDCG@20'), ('Gate_vs_LSE', 'NDCG@20'),
              ('LSE_vs_Uniform', 'NDCG@20')}
    have = {(r.get('comparison'), r.get('metric')) for r in boot}
    if not wanted <= have:
        raise SystemExit(f'refusing to write verdicts: bootstrap.csv lacks '
                         f'{sorted(wanted - have)}')
    for m in ('Uniform', 'LSE', 'Gate'):
        if not any(r.get('method') == m for r in tres):
            raise SystemExit(f'refusing to write verdicts: test_results.csv lacks {m}')
    man = json.loads((OUT / 'SPORTS_CONFIRMATION_FREEZE_MANIFEST.json').read_text()) \
        if (OUT / 'SPORTS_CONFIRMATION_FREEZE_MANIFEST.json').exists() else {}
    # filter to directories: an unfiltered glob would let a stray file sort first and
    # silently empty the training sections (sc_provenance.py filters the same way)
    truns = sorted(p for p in (OUT / 'runs').glob('*') if p.is_dir()) \
        if (OUT / 'runs').exists() else []
    if len(truns) != 1:
        raise SystemExit(f'refusing to write report: expected exactly one run directory, '
                         f'found {len(truns)}')
    tlog = json.loads((truns[0] / 'train_log.json').read_text()) if \
        (truns[0] / 'train_log.json').exists() else {}
    if not tlog:
        raise SystemExit('refusing to write report: run has no train_log.json')
    # development_vs_sports.csv is the pristine-vs-development comparison (section K);
    # publishing the report without it would silently drop that section
    if not rd(OUT / 'development_vs_sports.csv'):
        raise SystemExit('refusing to write report: development_vs_sports.csv missing '
                         'or empty - run sc_dev_vs_sports.py first')
    gman = json.loads((DEV / 'GATE_FREEZE_MANIFEST.json').read_text()) \
        if (DEV / 'GATE_FREEZE_MANIFEST.json').exists() else {}
    dman = json.loads((DEV / 'GATE_FREEZE_MANIFEST.json').read_text()) \
        if (DEV / 'GATE_FREEZE_MANIFEST.json').exists() else {}

    vA, rA = verdict_primary(boot, tres)
    vB, rB = verdict_lse(boot)
    vC, rC = verdict_gate_vs_lse(boot)
    vD, rD = verdict_gate_vs_lse(boot)   # gate vs LSE on Sports

    def get(rows, method, key='NDCG@20'):
        for r in rows:
            if r.get('method') == method:
                return float(r[key])
        return float('nan')

    def mechv(name):
        for r in mech:
            if r.get('metric') == name:
                return float(r['value'])
        return float('nan')

    L = []
    L.append('# FINAL_SPORTS_CONFIRMATION_REPORT.md')
    L.append('')
    L.append('Independent confirmatory evaluation of the frozen candidate-conditioned '
             'factorization method on `Sports_and_Outdoors`, a dataset that took no part in '
             'method development.')
    L.append('')
    L.append('No parameter was tuned on Sports: the gate weights, the 46-feature schema, '
             'the scoring formulas, the beam size, the union rule and the gate architecture '
             'are all the frozen development artifacts. The only Sports-specific decisions '
             'are the two the protocol pre-registered (unseen-domain one-hot `[0,0,0,0]` and '
             'the global domain-transfer feature scaler), and both were fixed before the '
             'test split was read. The base shared T5 was trained on Sports, as the protocol '
             'requires, under the frozen recipe and with validation-only checkpoint '
             'selection.')
    L.append('')

    # ---- A ----
    L.append('## A. Pristine-status audit')
    L.append('')
    L.append('`PRISTINE_STATUS_AUDIT.md`. A case-insensitive search for '
             '`Sports_and_Outdoors` across every historical output matches exactly two '
             'files, both written in the immediately preceding round and both being the '
             '*confirmatory-dataset plan* sections of reports, not method inputs. Sports '
             'never contributed to mechanism discovery, factorization-generality, the '
             'cross-tokenizer study, the user router, the candidate-NLL design or the '
             'disagreement-gate development. No Sports SID was generated, no Sports '
             'embedding loaded, no Sports model trained and no Sports metric computed before '
             'this round.')
    L.append('')
    L.append('```')
    L.append('PRISTINE FOR METHOD CONFIRMATION')
    L.append('```')
    L.append('')
    L.append('Disclosed caveat: the agent knew a reserve dataset existed ~9 minutes before '
             'the final gate retrain (the plan file is timestamped 21:18:48, the gate '
             'checkpoint 21:27:46). That knowledge could not shape the method, because the '
             'architecture, features, loss, `lambda_g` grid and selection rule were fixed by '
             'specification and selected on the four development datasets\' RouterDev only.')
    L.append('')

    # ---- B ----
    L.append('## B. Sports dataset')
    L.append('')
    L.append('| quantity | value |')
    L.append('|---|---|')
    L.append('| users | 35,598 (one row per user in each of train/valid/test) |')
    L.append('| items | 18,357 |')
    L.append('| RQ-VAE-3layer SID | pre-existing, (18357, 4), injective |')
    L.append('| Spectral-K3 SID | **generated this round**, (18357, 4), injective |')
    L.append(f'| shared checkpoint | trained this round, sha `{str(man.get("shared_checkpoint_sha256"))[:32]}…` |')
    L.append('')

    # ---- C ----
    L.append('## C. New Spectral-K3 SID')
    L.append('')
    L.append('The pre-existing Sports spectral array is K=2/L=3, so it cannot serve as the '
             'K=3 arm of a structurally matched pair. A K=3 array was generated with the '
             'repository\'s audited generator, called verbatim '
             '(`sc_generate_sid.py`); only the write target differs, because the artifact '
             'root is read-only.')
    L.append('')
    L.append('Before Sports was touched, the identical call sequence was run on **Beauty** '
             'and reproduced the audited K=3 array **byte for byte** '
             '(`base_sid` and final SID both element-wise identical). Sanity: rows 18,357, '
             'length 4, codes in [0,255], final SID injective, item→SID→item 100%. Full '
             'detail in `SPORTS_SID_AUDIT.md`.')
    L.append('')

    # ---- D ----
    L.append('## D. Shared-model training')
    L.append('')
    L.append('One shared T5, Spectral-K3 + RQ-VAE-3layer, trained by the audited trainer '
             '`xt_train_shared.main` **called unchanged**, with the frozen recipe read off '
             'the historical shared run:')
    L.append('')
    if tlog:
        c = tlog.get('config', {})
        L.append('| setting | value |')
        L.append('|---|---|')
        for k in ('exposure', 'seed', 'num_epochs', 'batch_size', 'infer_size', 'lr',
                  'early_stop', 'beam_size', 'max_len'):
            if k in c:
                L.append(f'| {k} | {c[k]} |')
        L.append('| capacity | 4+4 layers, d_model 128, d_ff 1024, 6 heads, d_kv 64, dropout 0.1 |')
        L.append(f'| vocabulary | {tlog.get("vocab_size")} (control tokens '
                 f'{tlog.get("plan", {}).get("control_tokens")}) |')
        L.append(f'| parameters | {tlog.get("params")} |')
        L.append('')
        ep = tlog.get('epochs', [])
        L.append(f'Ran {len(ep)} epochs; best epoch {tlog.get("best_epoch")} with '
                 f'macro-average validation NDCG@20 '
                 f'**{tlog.get("best_macro_valid_ndcg20", float("nan")):.6f}**. '
                 f'Selection used the macro average across the two factorizations, exactly '
                 f'as in the historical shared study. **Test was never read for selection.**')
        L.append('')
        b = tlog.get('best_epoch')
        if b:
            e = ep[b - 1]
            L.append(f'At the selected epoch: Spectral-K3 validation NDCG@20 '
                     f'{e["valid_ndcg20"]["spectral_k3"]:.6f}, RQ-VAE-3layer '
                     f'{e["valid_ndcg20"]["rqvae3"]:.6f}.')
            L.append('')
    L.append(f'Run directory: `{truns[0].name if truns else "n/a"}` with config, '
             'RUN_METADATA, train_log, checkpoint hash, validation metrics and both '
             'PRE_RUN/POST_RUN manifests.')
    L.append('')

    # ---- E ----
    L.append('## E. Validation-only calibration')
    L.append('')
    L.append('| item | source | used Sports validation? |')
    L.append('|---|---|---|')
    L.append('| candidate NLL scaler (mu/sigma per factorization) | ALL Sports validation '
             'candidates | YES — pre-defined unsupervised calibration (section 18) |')
    L.append('| shared-model checkpoint selection | macro validation NDCG@20 | YES — frozen '
             'pipeline definition (section 13) |')
    L.append('| gate weights | four development datasets | **NO** |')
    L.append('| gate architecture | specification | **NO** |')
    L.append('| 46-feature schema | frozen | **NO** |')
    L.append('| `lambda_g` | frozen at 0.0 | **NO** |')
    L.append('| domain-transfer feature scaler | pooled development RouterTrain rows | **NO** |')
    L.append('')
    L.append('This split of responsibilities is exactly what section 24 permits and '
             'forbids; nothing on the second list was touched.')
    L.append('')

    # ---- F ----
    L.append('## F. Frozen confirmation protocol')
    L.append('')
    if man:
        L.append('`SPORTS_CONFIRMATION_FREEZE_MANIFEST.json` was written before the test '
                 'split was read and records:')
        L.append('')
        L.append('```')
        L.append(f'git SHA                     : {man.get("git_sha")}')
        L.append(f'Sports spectral SID sha256  : {str(man.get("sports_spectral_k3_sid_sha256"))[:32]}…')
        L.append(f'RQ-VAE SID sha256           : {str(man.get("sports_rqvae_sid_sha256"))[:32]}…')
        L.append(f'shared checkpoint sha256    : {str(man.get("shared_checkpoint_sha256"))[:32]}…')
        L.append(f'frozen gate checkpoint      : {str(man.get("frozen_gate_checkpoint_sha256"))[:32]}…')
        L.append(f'frozen gate name            : {man.get("frozen_gate_name")}')
        L.append(f'feature schema sha256       : {str(man.get("feature_schema_sha256"))[:32]}…')
        L.append(f'feature count               : {man.get("feature_count")}')
        L.append(f'unseen-domain one-hot rule  : {man.get("sports_unseen_domain_onehot_rule")}')
        L.append(f'global scaler sha256        : {str(man.get("global_feature_scaler_sha256"))[:32]}…')
        L.append(f'global scaler built from     : {man.get("global_feature_scaler_built_from")}')
        L.append(f'beam size                   : {man.get("beam_size")}')
        L.append(f'union rule                  : {man.get("candidate_union_rule")}')
        L.append(f'created                     : {man.get("created_local")}')
        L.append(f'sports_test_read            : {man.get("sports_test_read")}')
        L.append('```')
    L.append('')
    L.append('The VALID-only metrics were frozen with it:')
    L.append('')
    L.append('| method | Recall@20 | NDCG@20 |')
    L.append('|---|---|---|')
    for r in vres:
        L.append(f'| {r["method"]} | {f(r["Recall@20"])} | {f(r["NDCG@20"])} |')
    L.append('')

    # ---- G ----
    L.append('## G. Sports test — primary result')
    L.append('')
    L.append('| method | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |')
    L.append('|---|---|---|---|---|---|---|')
    for r in tres:
        L.append(f'| {r["method"]} | {f(r["Recall@5"])} | {f(r["Recall@10"])} | '
                 f'{f(r["Recall@20"])} | {f(r["NDCG@5"])} | {f(r["NDCG@10"])} | '
                 f'{f(r["NDCG@20"])} |')
    L.append('')
    L.append('Full per-user ranks are preserved in `test/per_user_ranks.npz`.')
    L.append('')

    # ---- H ----
    L.append('## H. Uniform vs LSE vs frozen Gate')
    L.append('')
    L.append('Paired user-level bootstrap, 5000 resamples, canonical seed 20260912:')
    L.append('')
    L.append('| comparison | metric | delta | 95% CI | p |')
    L.append('|---|---|---|---|---|')
    for b in boot:
        L.append(f'| {b["comparison"]} | {b["metric"]} | {f(b["delta"], 8)} | '
                 f'[{f(b["ci_low"], 8)}, {f(b["ci_high"], 8)}] | {f(b["p_value"], 6)} |')
    L.append('')
    L.append('Pre-registered hypotheses:')
    L.append('')
    L.append('| hypothesis | verdict | evidence |')
    L.append('|---|---|---|')
    L.append(f'| H1: LSE > Uniform | **{vB}** | {rB} |')
    L.append(f'| H2: frozen Gate > Uniform | **{vA}** | {rA} |')
    L.append(f'| H3 (secondary): frozen Gate > frozen LSE | **{vC}** | {rC} |')
    L.append('')

    # ---- I ----
    L.append('## I. Bootstrap')
    L.append('')
    L.append(f'{len(boot)} comparisons are tabulated in `test/bootstrap.csv`, covering '
             'NDCG@5/10/20 for Gate-vs-Uniform, Gate-vs-LSE and LSE-vs-Uniform. The claim '
             'policy of harness section 33 was applied to the NDCG@20 Gate-vs-Uniform row '
             'only, and its outcome is reproduced verbatim in the verdict block.')
    L.append('')

    # ---- J ----
    L.append('## J. Mechanism diagnostics')
    L.append('')
    L.append('Computed after the primary result was frozen.')
    L.append('')
    L.append('| metric | value |')
    L.append('|---|---|')
    for r in mech:
        L.append(f'| {r["metric"]} | {f(r["value"])} |')
    L.append('')
    L.append('Subgroup behaviour (evaluation only; it did **not** alter the method):')
    L.append('')
    L.append('| group | n | method | NDCG@20 | Recall@20 |')
    L.append('|---|---|---|---|---|')
    for r in sub:
        L.append(f'| {r["group"]} | {r["n"]} | {r["method"]} | {f(r["NDCG@20"])} | '
                 f'{f(r["Recall@20"])} |')
    L.append('')
    L.append('This is the test of whether the development datasets\' heterogeneity '
             'reproduces: candidate NLL helped the Only-F1/Only-F2 groups and hurt Both on '
             'the four development datasets. The Sports table above is the independent '
             'reading of the same phenomenon.')
    L.append('')

    # ---- K ----
    L.append('## K. Development vs Sports comparison')
    L.append('')
    L.append('`development_vs_sports.csv`. Sports is reported **separately** and is never '
             'pooled into the development averages (section 38).')
    L.append('')
    dv = rd(OUT / 'development_vs_sports.csv')
    if dv:
        L.append('| dataset | role | Uniform->LSE | Uniform->Gate | LSE->Gate | '
                 'target-vs-nontarget AUC | union coverage | Only-F1 | Only-F2 | Both |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for r in dv:
            L.append(f'| {r["dataset"]} | {r["role"]} | {f(r["uniform_to_lse"], 6)} | '
                     f'{f(r["uniform_to_gate"], 6)} | {f(r["lse_to_gate"], 6)} | '
                     f'{f(r["target_vs_nontarget_auc"], 4)} | {f(r["union_coverage"], 4)} | '
                     f'{f(r["only_f1_fraction"], 4)} | {f(r["only_f2_fraction"], 4)} | '
                     f'{f(r["both_fraction"], 4)} |')
    L.append('')

    # ---- L ----
    L.append('## L. Limitations')
    L.append('')
    L.append('* **One confirmatory dataset, one seed.** Sports is a single Amazon category; '
             'the frozen gate was trained on four other Amazon categories, so this is a '
             'domain-transfer test within one dataset family, not a cross-domain one.')
    L.append('* **The shared model is trained on Sports**, as the protocol requires, so '
             'Sports is a confirmation of the *method pipeline* rather than of the specific '
             'development checkpoints. The gate, however, is strictly zero-shot.')
    L.append('* **Sports validation is used** for the candidate-NLL scaler and for '
             'checkpoint selection. Both are pre-defined steps of the frozen pipeline, but '
             'they do mean Sports is not untouched end-to-end.')
    L.append('* **The unseen-domain one-hot had to be imputed** as `[0,0,0,0]`, so the gate '
             'receives a feature pattern it never saw in training for that block.')
    L.append('* **The domain-transfer scaler** pools four datasets\' RouterTrain values; if '
             'Sports feature distributions differ systematically, the scaled inputs the '
             'gate sees are shifted, and no Sports-specific correction was permitted.')
    L.append('* `Oracle` is the in-union ceiling and is not a method.')
    L.append('* If the frozen method fails on Sports, that failure is reported as-is and '
             'was not repaired and re-tested.')
    L.append('')

    # ---- M ----
    L.append('## M. Training performed')
    L.append('')
    L.append('| what | detail |')
    L.append('|---|---|')
    L.append('| Spectral-K3 SID | generated, not trained (deterministic pipeline) |')
    if tlog:
        L.append(f'| Sports shared T5 | 1 model, {len(tlog.get("epochs", []))} epochs, '
                 f'best epoch {tlog.get("best_epoch")} |')
    L.append('| frozen gate | **not retrained** — loaded from '
             f'`{man.get("frozen_gate_run_directory", "n/a")}` |')
    L.append('| base T5 of the development datasets | not modified |')
    L.append('')
    L.append('No Sports test observation influenced any training decision.')
    L.append('')

    # ---- verdicts ----
    L.append('## Verdicts')
    L.append('')
    L.append('```')
    L.append(f'PRISTINE_SPORTS_CONFIRMATION: {vA}')
    L.append(f'CANDIDATE_LIKELIHOOD_GENERALISES_TO_SPORTS: {vB}')
    L.append(f'FROZEN_CANDIDATE_GATE_GENERALISES_ZERO_SHOT: {vA}')
    L.append(f'FROZEN_GATE_BEATS_FROZEN_LSE_ON_SPORTS: {vD}')
    L.append('```')
    L.append('')
    L.append('| verdict | rule outcome |')
    L.append('|---|---|')
    L.append(f'| PRISTINE_SPORTS_CONFIRMATION | {rA} |')
    L.append(f'| CANDIDATE_LIKELIHOOD_GENERALISES_TO_SPORTS | {rB} |')
    L.append(f'| FROZEN_CANDIDATE_GATE_GENERALISES_ZERO_SHOT | {rA} |')
    L.append(f'| FROZEN_GATE_BEATS_FROZEN_LSE_ON_SPORTS | {rD} |')
    L.append('')

    (OUT / 'FINAL_SPORTS_CONFIRMATION_REPORT.md').write_text('\n'.join(L) + '\n')
    print(f'wrote {OUT/"FINAL_SPORTS_CONFIRMATION_REPORT.md"}')
    print(f'  {vA}\n  {vB}\n  {vC}\n  {vD}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
