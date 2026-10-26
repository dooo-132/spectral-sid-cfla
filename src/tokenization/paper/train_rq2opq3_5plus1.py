#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 2B' -- RQ2-OPQ3 + 1 disambiguation token (5 semantic, L=6).

Authorised by the operator after OPQ5 was proven impossible (768 % 5 != 0).
Uses `drivers/generate_rq2_opq3_sid.py`, which is the historical
`generate_rq_opq_sid.py` with exactly four changed lines:
  * CLI choices += "rq2_opq3"
  * base-SID width 3 -> method-derived (5 for this arm)
  * metadata description
  * (plus pure additions: a 2-level RQ function, an optional level-1 centroid dump)

Everything else -- k-means hyper-parameters, OPQ factory construction, codebook
size, collision-token handling, item-id handling, output layout -- is byte-identical
to the historical implementation.

Usage: python3 train_rq2opq3_5plus1.py --dataset Beauty
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
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(_REPO_ROOT + "")
TIGER = Path(_ART_ROOT + "")
OUT = rp.OUTPUT_ROOT / "final_hyperparam_efficiency"
RUNS = OUT / "runs_5plus1"
DRIVER = OUT / "drivers/generate_rq2_opq3_sid.py"
VENV_PY = Path(_os.environ.get("CFLA_PYTHON", sys.executable))
ARM = "rq2_opq3"
EXPECTED_FACTORY = "OPQ3,IVF1,PQ3x8"
DATASETS = ["Beauty", "Cell_Phones_and_Accessories",
            "Health_and_Personal_Care", "Toys_and_Games"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def derive_vocab(a, C):
    L = a.shape[1]
    mx = [int(a[:, p].max()) for p in range(L)]
    disamb = mx[-1] + 1
    stride = max(C, disamb)
    vocab = stride * L + 1
    max_token = max(mx[p] + p * stride + 1 for p in range(L))
    return {"sid_len": L, "per_position_max_code": mx, "semantic_codebook": C,
            "disambiguation_alphabet": disamb, "stride": stride,
            "vocab_size": vocab, "max_token": max_token,
            "max_token_lt_vocab": max_token < vocab}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--arm", default=ARM, choices=[ARM],
                    help="supervisor interface compatibility; only rq2_opq3")
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds, C = a.dataset, a.codebook

    run_dir = RUNS / ds / ARM
    sid_dir = run_dir / "sid"
    sid_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "train.log"
    data_dir = TIGER / f"data/{ds}"
    if not (data_dir / "item_emb.parquet").exists():
        print(f"FATAL: {data_dir}/item_emb.parquet missing"); return 2

    cmd = [VENV_PY, str(DRIVER), "--dataset_dir", str(data_dir),
           "--method", ARM, "--codebook_size", str(C),
           "--add_collision_token", "--output_dir", str(sid_dir)]
    if a.dry_run:
        print("DRY-RUN rq2_opq3:\n  " + " ".join(map(str, cmd))); return 0

    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        print(f"FATAL: rq2_opq3 generation exit {rc}"); return rc

    sid = sid_dir / f"{ds}_t5_{ARM}.npy"
    meta = sid_dir / "metadata.json"
    if not sid.exists() or not meta.exists():
        print(f"FATAL: missing outputs ({sid.exists()=}, {meta.exists()=})"); return 4

    m = json.loads(meta.read_text())
    factory = str(m.get("opq_factory", ""))
    import numpy as np
    arr = np.load(sid)
    n_unique = len({tuple(int(x) for x in r) for r in arr})
    info = {"sid_path": str(sid), "sid_sha256": sha256(sid),
            "n_items": int(arr.shape[0]), "shape": list(arr.shape),
            "semantic_positions": arr.shape[1] - 1, "full_sid_length": arr.shape[1],
            "has_nan": bool(np.isnan(arr.astype(float)).any()),
            "min_code": int(arr.min()), "max_code": int(arr.max()),
            "injective": bool(n_unique == arr.shape[0]), "n_unique_sids": int(n_unique),
            "collision_rate": float(1 - n_unique / arr.shape[0]),
            "row_alignment": "row i == dataset item id i+1 (docs/ITEM_ID_CONVENTION.md)",
            "opq_factory": factory, "rq_levels": m.get("rq_levels"),
            "method_provenance_ok": factory == EXPECTED_FACTORY,
            "method_provenance_note": (
                "real faiss OPQ confirmed via the recorded factory string; "
                "unlike OPQ5 the factory is constructible at M=3 (768 % 3 == 0)")}
    info.update(derive_vocab(arr, C))
    info["ok"] = bool(info["injective"] and not info["has_nan"] and info["min_code"] >= 0
                      and info["max_token_lt_vocab"] and info["full_sid_length"] == 6
                      and info["semantic_positions"] == 5 and info["method_provenance_ok"])
    (run_dir / "SID_VALIDATION.json").write_text(json.dumps(info, indent=2) + "\n")
    (run_dir / "RUN_METADATA.json").write_text(json.dumps({
        "phase": "PHASE 2B' RQ2-OPQ3 + disambiguation (operator-authorised new driver)",
        "dataset": ds, "arm": ARM, "codebook_size": C,
        "generator": str(DRIVER), "generator_sha256": sha256(DRIVER),
        "generator_template": ("src/tokenization/paper/"
                               "generate_rq_opq_sid.py"),
        "changed_historical_lines": 4,
        "recipe": {"rq_levels": 2, "opq_codes": 3, "disambiguation": 1,
                   "codebook_size": C, "rq_niter": m.get("rq_niter"),
                   "seed": m.get("seed"), "opq_factory": factory},
        "sid_validation": info, "runtime_seconds": round(time.time() - t0, 1),
        "source_metadata": str(meta),
        "wrote_only_under": str(OUT)}, indent=2) + "\n")

    row = {"dataset": ds, "arm": ARM, "codebook": C, **info}
    vcsv = OUT / "tokenizers_5plus1/SID_VALIDATION.csv"
    new = not vcsv.exists()
    with open(vcsv, "a", newline="", encoding="utf-8") as fh:
        cols = [k for k in row if k != "per_position_max_code"] + ["per_position_max_code"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**row, "per_position_max_code": json.dumps(info["per_position_max_code"])})

    print(f"[PHASE3] {ds}/{ARM}: shape={info['shape']} injective={info['injective']} "
          f"factory={factory} stride={info['stride']} vocab={info['vocab_size']} ok={info['ok']}")
    print("PHASE2_OK" if info["ok"] else "PHASE2_FAIL")
    return 0 if info["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
