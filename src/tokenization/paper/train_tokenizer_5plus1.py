#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 2/3 -- train a canonical 5+1 tokenizer for one dataset, one arm.

Arms
    rqvae5     RQ-VAE, 5 residual layers, codebook C, + 1 disambiguation token
    opq5       5 OPQ semantic positions (faiss OPQ rotation + balanced k-means),
               + 1 disambiguation token
    rq1_opq4   1 RQ residual position + 4 OPQ/spectral positions
               (`generate_rq_residual_spectral_sid.py` sets
                spectral_token_num = num_sid_tokens - 1), + 1 disambiguation token

All three run the HISTORICAL implementations unmodified: they are invoked
read-only from the frozen artifact root/rqvae with PYTHONPATH pointing there, and
every output path is redirected into this round's writable root.

The OPQ generators emit only the semantic columns; `append_collision_token.py`
adds the 6th.  `generate_code.py` (RQ-VAE) already appends it internally.

Usage
    python3 train_tokenizer_5plus1.py --dataset Beauty --arm rqvae5
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
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(_REPO_ROOT + "")
TIGER = Path(_ART_ROOT + "")
RQVAE_DIR = REPO / "src/tokenization/legacy_rqvae"
OUT = rp.OUTPUT_ROOT / "final_hyperparam_efficiency"
RUNS = OUT / "runs_5plus1"
VENV_PY = Path(_os.environ.get("CFLA_PYTHON", sys.executable))
RECIPE = json.loads((REPO / "configs/paper/RQ5_RECIPE_FINAL.json").read_text())

DATASETS = ["Beauty", "Cell_Phones_and_Accessories",
            "Health_and_Personal_Care", "Toys_and_Games"]
ARMS = ["rqvae5", "opq5", "rq1_opq4"]
SEMANTIC = 5
FULL_LEN = 6


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd, log, env=None):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n(cwd={cwd})\n")
        fh.flush()
        return subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def derive_vocab(a, semantic_codebook: int) -> dict:
    """PHASE 15 -- derive the stride from the data; never assume C*L+1."""
    L = a.shape[1]
    per_pos_max = [int(a[:, p].max()) for p in range(L)]
    disamb = per_pos_max[-1] + 1
    stride = max(semantic_codebook, disamb)
    vocab = stride * L + 1
    max_token = max(per_pos_max[p] + p * stride + 1 for p in range(L))
    return {"sid_len": L, "per_position_max_code": per_pos_max,
            "semantic_codebook": semantic_codebook, "disambiguation_alphabet": disamb,
            "stride": stride, "vocab_size": vocab, "max_token": max_token,
            "max_token_lt_vocab": max_token < vocab}


def validate_sid(p: Path, semantic_codebook: int) -> dict:
    import numpy as np
    a = np.load(p)
    L = a.shape[1]
    n_unique = len({tuple(int(x) for x in r) for r in a})
    info = {"sid_path": str(p), "sid_sha256": sha256(p), "n_items": int(a.shape[0]),
            "shape": list(a.shape), "semantic_positions": L - 1, "full_sid_length": L,
            "has_nan": bool(np.isnan(a.astype(float)).any()),
            "min_code": int(a.min()), "max_code": int(a.max()),
            "injective": bool(n_unique == a.shape[0]), "n_unique_sids": int(n_unique),
            "collision_rate": float(1 - n_unique / a.shape[0]),
            "row_alignment": "row i == dataset item id i+1 (docs/ITEM_ID_CONVENTION.md)"}
    info.update(derive_vocab(a, semantic_codebook))
    info["ok"] = (info["injective"] and not info["has_nan"] and info["min_code"] >= 0
                  and info["max_token_lt_vocab"] and L == FULL_LEN
                  and info["semantic_positions"] == SEMANTIC)
    return info


def find_rqvae5_ckpt(ds: str) -> Path | None:
    d = RUNS / ds / "rqvae5/tokenizer"
    c = sorted(d.rglob("best_collision_model.pth")) if d.exists() else []
    return c[-1] if c else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=RECIPE["tokenizer_training"]["epochs"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds, arm, C = a.dataset, a.arm, a.codebook

    run_dir = RUNS / ds / (arm if C == 256 else f"{arm}_cb{C}")
    tok_dir = run_dir / "tokenizer"
    sids = run_dir / "sid"
    tok_dir.mkdir(parents=True, exist_ok=True)
    sids.mkdir(parents=True, exist_ok=True)
    log = run_dir / "train.log"
    item_emb = TIGER / f"data/{ds}/item_emb.parquet"
    if not item_emb.exists():
        print(f"FATAL: {item_emb} missing"); return 2

    env = dict(os.environ, PYTHONPATH=str(RQVAE_DIR), OMP_NUM_THREADS="8")
    t = RECIPE["tokenizer_training"]
    base = sids / f"{ds}_t5_{arm}_base.npy"
    final = sids / f"{ds}_t5_{arm}.npy"
    t0 = time.time()

    # ------------------------------------------------------------ arm: rqvae5
    if arm == "rqvae5":
        cmd = [VENV_PY, str(RQVAE_DIR / "main.py"),
               "--data_path", str(item_emb), "--ckpt_dir", str(tok_dir),
               "--device", "cuda:0", "--epochs", str(a.epochs),
               "--batch_size", str(t["batch_size"]), "--lr", str(t["lr"]),
               "--e_dim", str(t["e_dim"]), "--num_emb_list", *[str(C)] * SEMANTIC,
               "--sk_epsilons", *[str(x) for x in RECIPE["sk_epsilons"]["value"]],
               "--layers", *[str(x) for x in t["layers"]],
               "--kmeans_iters", str(t["kmeans_iters"]), "--sk_iters", str(t["sk_iters"]),
               "--weight_decay", str(t["weight_decay"]), "--beta", str(t["beta"]),
               "--quant_loss_weight", str(t["quant_loss_weight"]),
               "--warmup_epochs", str(t["warmup_epochs"]), "--eval_step", str(t["eval_step"]),
               "--num_workers", str(t["num_workers"]), "--save_limit", str(t["save_limit"])]
        if a.dry_run:
            print("DRY-RUN rqvae5 train:\n  " + " ".join(map(str, cmd))); return 0
        rc = run(cmd, RQVAE_DIR, log, env)
        if rc:
            print(f"FATAL: RQ-VAE training exit {rc}"); return rc
        cks = sorted(tok_dir.rglob("best_collision_model.pth"))
        if not cks:
            print("FATAL: no best_collision_model.pth"); return 3
        src = (RQVAE_DIR / "generate_code.py").read_text()
        txt = re.sub(r'^dataset\s*=\s*["\'].*?["\']', f'dataset = "{ds}"', src, count=1, flags=re.M)
        txt = re.sub(r'^ckpt_path\s*=\s*f?["\'].*?["\']', f'ckpt_path = "{cks[-1]}"', txt, count=1, flags=re.M)
        txt = re.sub(r'^output_file\s*=\s*f?["\'].*?["\']', f'output_file = "{base}"', txt, count=1, flags=re.M)
        gen = run_dir / "generate_code_local.py"; gen.write_text(txt)
        rc = run([VENV_PY, str(gen)], run_dir, log, env)
        if rc or not base.exists():
            print(f"FATAL: SID generation exit {rc}"); return rc or 4

    # -------------------------------------------------------------- arm: opq5
    elif arm == "opq5":
        cmd = [VENV_PY, str(RQVAE_DIR / "generate_spectral_opq_balanced_sid.py"),
               "--item_emb_path", str(item_emb), "--output_path", str(base),
               "--num_sid_tokens", str(SEMANTIC), "--codebook_size", str(C),
               "--metric", "abs_pearson", "--pca_dim", "64", "--opq_niter", "50",
               "--balanced_iter", "8", "--top_m", "64", "--max_combinations", "20000",
               "--random_state", "2025"]
        if a.dry_run:
            print("DRY-RUN opq5:\n  " + " ".join(map(str, cmd))); return 0
        rc = run(cmd, RQVAE_DIR, log, env)
        if rc or not base.exists():
            print(f"FATAL: OPQ5 generation exit {rc}"); return rc or 4
        rc = run([VENV_PY, str(RQVAE_DIR / "append_collision_token.py"),
                  "--input_path", str(base), "--output_path", str(final),
                  "--codebook_size", str(C)], RQVAE_DIR, log, env)
        if rc or not final.exists():
            print(f"FATAL: collision append exit {rc}"); return rc or 6

    # ---------------------------------------------------------- arm: rq1_opq4
    else:
        ck = find_rqvae5_ckpt(ds)
        if ck is None:
            print(f"FATAL: rq1_opq4 needs the {ds} RQ-VAE5 checkpoint; "
                  f"run --arm rqvae5 first"); return 7
        cmd = [VENV_PY, str(RQVAE_DIR / "generate_rq_residual_spectral_sid.py"),
               "--ckpt_path", str(ck), "--item_emb_path", str(item_emb),
               "--output_path", str(base), "--num_sid_tokens", str(SEMANTIC),
               "--codebook_size", str(C), "--metric", "abs_pearson",
               "--pca_dim", "0", "--top_m", "64", "--max_combinations", "4096",
               "--batch_size", "256", "--num_workers", "4", "--device", "cuda:0",
               "--random_state", "2025"]
        if a.dry_run:
            print("DRY-RUN rq1_opq4:\n  " + " ".join(map(str, cmd))); return 0
        rc = run(cmd, RQVAE_DIR, log, env)
        if rc or not base.exists():
            print(f"FATAL: RQ1-OPQ4 generation exit {rc}"); return rc or 4
        rc = run([VENV_PY, str(RQVAE_DIR / "append_collision_token.py"),
                  "--input_path", str(base), "--output_path", str(final),
                  "--codebook_size", str(C)], RQVAE_DIR, log, env)
        if rc or not final.exists():
            print(f"FATAL: collision append exit {rc}"); return rc or 6

    if arm == "rqvae5":
        shutil.copy2(base, final)
    runtime = round(time.time() - t0, 1)

    # Method-provenance gate: an OPQ-family arm MUST have really used faiss OPQ.
    # The historical generator silently falls back to PCA whitening when
    # faiss.OPQMatrix raises, which produced valid-looking but mislabelled SIDs
    # on 2026-09-16 (see tokenizers_5plus1/OPQ_5PLUS1_BLOCKER.md).
    rotate_method = ""
    if arm in ("opq5", "rq1_opq4"):
        for cand in sids.glob(f"{ds}_t5_{arm}_base_rotate_method.txt"):
            rotate_method = cand.read_text().strip()
        if rotate_method != "faiss_opq":
            print(f"FATAL: {ds}/{arm} did not run real OPQ "
                  f"(rotate_method={rotate_method!r}); refusing to validate")
            return 11

    # ------------------------------------------------------------ PHASE 3
    info = validate_sid(final, C)
    info["rotate_method"] = rotate_method or "n/a"
    info["method_provenance_ok"] = (rotate_method == "faiss_opq") if arm in ("opq5", "rq1_opq4") else True
    info["ok"] = bool(info["ok"] and info["method_provenance_ok"])
    (run_dir / "SID_VALIDATION.json").write_text(json.dumps(info, indent=2) + "\n")
    (run_dir / "RUN_METADATA.json").write_text(json.dumps({
        "phase": "PHASE 2/3 canonical 5+1 tokenizer", "dataset": ds, "arm": arm,
        "codebook_size": C, "semantic_positions": SEMANTIC, "full_sid_length": FULL_LEN,
        "recipe_source": "configs/paper/RQ5_RECIPE_FINAL.json",
        "recipe_invariants": RECIPE["invariants"],
        "tokenizer_checkpoint": str(find_rqvae5_ckpt(ds)) if arm != "rqvae5" else
                                str(sorted(tok_dir.rglob("best_collision_model.pth"))[-1]),
        "sid_validation": info, "runtime_seconds": runtime,
        "historical_sources": {"tokenizer": str(RQVAE_DIR / "main.py"),
                               "opq": str(RQVAE_DIR / "generate_spectral_opq_balanced_sid.py"),
                               "rq_opq": str(RQVAE_DIR / "generate_rq_residual_spectral_sid.py"),
                               "collision": str(RQVAE_DIR / "append_collision_token.py")},
        "wrote_only_under": str(OUT)}, indent=2) + "\n")

    row = {"dataset": ds, "arm": arm, "codebook": C, **info}
    vcsv = OUT / "tokenizers_5plus1/SID_VALIDATION.csv"
    new = not vcsv.exists()
    with open(vcsv, "a", newline="", encoding="utf-8") as fh:
        cols = [k for k in row if k != "per_position_max_code"] + ["per_position_max_code"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**row, "per_position_max_code": json.dumps(info["per_position_max_code"])})

    print(f"[PHASE3] {ds}/{arm}: shape={info['shape']} injective={info['injective']} "
          f"stride={info['stride']} vocab={info['vocab_size']} ok={info['ok']}")
    print("PHASE2_OK" if info["ok"] else "PHASE2_FAIL")
    return 0 if info["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
