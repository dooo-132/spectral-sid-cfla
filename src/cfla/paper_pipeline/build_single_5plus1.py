#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PART IV (partial) -- length-matched 5+1 single-model TEST/VALID metrics.

Three of the six 5+1 families already exist as frozen artifacts for all four
development datasets and need no new training:
    Even Split K=5 + collision      (L = 6)
    Random Split K=5 + collision    (L = 6)
    Spectral K=5 + collision        (L = 6)   nonneg_pearson_v3 and abspearson_v3

They are re-scored here from the frozen beam dumps with the audited evaluator
(validated to ~1e-17 against `outputs/results/tables/single_granularity.csv`).

The three quantization families (RQ-VAE5, RQ1-OPQ4, OPQ5) do NOT exist at depth 5
and are recorded with status NOT_GENERATED.

Writes single_5plus1/{VALID_RESULTS.csv,TEST_RESULTS.csv}.
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
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(_REPO_ROOT + "")
TIGER = Path(_ART_ROOT + "")
OUT = rp.OUTPUT_ROOT / "final_hyperparam_efficiency/single_5plus1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from src.fusion.common import canonicalize_item_ids, _row_softmax  # noqa: E402

DATASETS = ["Beauty", "Cell_Phones_and_Accessories",
            "Health_and_Personal_Care", "Toys_and_Games"]
MET = ["Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20"]

EXISTING = [  # family key, display, affinity label
    ("evensplit_v3", "Even Split K5", "n/a"),
    ("randomsplit_v3", "Random Split K5", "n/a"),
    ("nonneg_pearson_v3", "Spectral K5", "nonneg_pearson_v3"),
    ("abspearson_v3", "Spectral K5 (abs)", "abspearson_v3"),
]
PLANNED = [("rqvae5", "RQ-VAE5"), ("rq1_opq4", "RQ1-OPQ4"), ("opq5", "OPQ5")]


def score(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    cand = canonicalize_item_ids(z["candidate_item_ids"])
    pr = _row_softmax(np.asarray(z["sequence_scores"], dtype=np.float64), cand >= 0)
    tgt = canonicalize_item_ids(z["target_item_ids"])
    n = cand.shape[0]
    hit = {k: np.zeros(n) for k in (5, 10, 20)}
    nd = {k: np.zeros(n) for k in (5, 10, 20)}
    for i in range(n):
        acc: dict[int, float] = {}
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
    return {**{f"Recall@{k}": float(hit[k].mean()) for k in (5, 10, 20)},
            **{f"NDCG@{k}": float(nd[k].mean()) for k in (5, 10, 20)},
            "n_users": n,
            "mean_beam_candidates_per_user": float((cand >= 0).sum(1).mean())}


def main() -> None:
    for split in ("valid", "test"):
        rows = []
        for ds in DATASETS:
            for fam, disp, aff in EXISTING:
                dump = TIGER / f"model/beam_dumps_{ds}_{fam}/{split}_k5_sid6.npz"
                if not dump.exists():
                    rows.append({"dataset": ds, "method": disp, "family": fam,
                                 "affinity": aff, "semantic_length": 5,
                                 "full_sid_length": 6, "codebook_size": 256,
                                 "split": split, "status": "DUMP_MISSING",
                                 "source_dump": str(dump)})
                    continue
                r = score(dump)
                rows.append({"dataset": ds, "method": disp, "family": fam,
                             "affinity": aff, "semantic_length": 5,
                             "full_sid_length": 6, "codebook_size": 256,
                             "split": split, "status": "MEASURED_FROM_FROZEN_DUMP",
                             "source_dump": str(dump), **r})
            for fam, disp in PLANNED:
                rows.append({"dataset": ds, "method": disp, "family": fam,
                             "affinity": "n/a", "semantic_length": 5,
                             "full_sid_length": 6, "codebook_size": 256,
                             "split": split, "status": "NOT_GENERATED",
                             "source_dump": "",
                             "reason": "requires a new 5-level tokenizer + T5; "
                                       "blocked by frozen_state.md B1 (no usable GPU)"})
        fields = ["dataset", "method", "family", "affinity", "semantic_length",
                  "full_sid_length", "codebook_size", "split", "status",
                  "source_dump", "n_users", "mean_beam_candidates_per_user",
                  "reason"] + MET
        fields = [f for f in fields if any(f in r for r in rows)]
        with open(OUT / f"{split.upper()}_RESULTS.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {split.upper()}_RESULTS.csv  rows={len(rows)}")

    import pandas as pd
    t = pd.read_csv(OUT / "TEST_RESULTS.csv")
    m = t[t.status == "MEASURED_FROM_FROZEN_DUMP"]
    print(f"\nlength-matched 5+1 singles, TEST, measured: {len(m)} rows "
          f"({m.method.nunique()} methods x {m.dataset.nunique()} datasets)")
    print(m[["dataset", "method", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10",
             "NDCG@20"]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
