#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE 4 -- train the canonical T5 recommender for one (dataset, arm) 5+1 model.

Runs the historical T5 trainer unmodified (read-only input root,
every output redirected into this round's writable root), then dumps VALID and
TEST beams with the historical beam dumper, then scores them with the
audited evaluator.

Checkpoint selection is the historical rule: item-level VALIDATION NDCG@20 with
early-stop patience 10. `--eval_mode item` is passed explicitly; the legacy
token-prefix metric is never used.

Vocabulary and stride come from the sibling `SID_VALIDATION.json` produced by
PHASE 2/3 -- never from the failed `codebook*L+1` assumption (PHASE 15).

Usage
    python3 train_t5_5plus1.py --dataset Beauty --arm rqvae5
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
from pathlib import Path

import numpy as np

REPO = Path(_REPO_ROOT + "")
ART_ROOT = Path(_ART_ROOT + "")
MODEL_DIR = REPO / "src" / "tiger"
OUT = rp.OUTPUT_ROOT / "final_hyperparam_efficiency"
RUNS = OUT / "runs_5plus1"
VENV_PY = Path(_os.environ.get("CFLA_PYTHON", sys.executable))
sys.path.insert(0, str(REPO))
from src.fusion.common import canonicalize_item_ids, _row_softmax  # noqa: E402

ARMS = ["rqvae5", "opq5", "rq1_opq4", "rq2_opq3"]
DATASETS = ["Beauty", "Cell_Phones_and_Accessories",
            "Health_and_Personal_Care", "Toys_and_Games"]
MET = ["Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20"]

RECIPE = dict(  # configs/canonical.yaml, verbatim
    num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024, num_heads=6,
    d_kv=64, dropout_rate=0.1, batch_size=256, lr=1e-4, num_epochs=200,
    early_stop=10, max_len=20, seed=2025, infer_size=96, beam_size=30,
    feed_forward_proj="relu", pad_token_id=0, eos_token_id=0,
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n(cwd={cwd})\n")
        fh.flush()
        return subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def score_dump(p: Path) -> dict:
    z = np.load(p, allow_pickle=True)
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
            "n_users": int(n),
            "mean_candidates_per_user": float((cand >= 0).sum(1).mean())}


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds, arm = a.dataset, a.arm
    arm_dir = arm if a.codebook == 256 else f"{arm}_cb{a.codebook}"

    tok_dir = RUNS / ds / arm_dir
    sid_val_p = tok_dir / "SID_VALIDATION.json"
    if not sid_val_p.exists():
        print(f"FATAL: {sid_val_p} missing -- run PHASE 2 for {ds}/{arm_dir} first")
        return 2
    sv = json.loads(sid_val_p.read_text())
    if not sv.get("ok"):
        print(f"FATAL: SID validation not ok for {ds}/{arm_dir}: "
              f"{ {k: sv[k] for k in ('injective','max_token_lt_vocab','shape')} }")
        return 3
    sid = Path(sv["sid_path"])
    stride, vocab, sid_len = int(sv["stride"]), int(sv["vocab_size"]), int(sv["sid_len"])
    max_tok = int(sv["max_token"])
    # PHASE 15 hard assertion before any training
    assert max_tok < vocab, f"max token {max_tok} >= vocab {vocab}"
    assert sid_len == 6, f"expected L=6, got {sid_len}"

    t5 = tok_dir / "t5"
    t5.mkdir(parents=True, exist_ok=True)
    log = tok_dir / "t5.log"
    ckpt = t5 / "best_model.pth"

    common = ["--dataset_path", str(ART_ROOT / f"data/{ds}"), "--code_path", str(sid),
              "--codebook_size", str(stride), "--vocab_size", str(vocab),
              "--seed", str(RECIPE["seed"]), "--max_len", str(RECIPE["max_len"]),
              "--beam_size", str(RECIPE["beam_size"]),
              "--infer_size", str(RECIPE["infer_size"]),
              "--batch_size", str(RECIPE["batch_size"]),
              "--num_layers", str(RECIPE["num_layers"]),
              "--num_decoder_layers", str(RECIPE["num_decoder_layers"]),
              "--d_model", str(RECIPE["d_model"]), "--d_ff", str(RECIPE["d_ff"]),
              "--num_heads", str(RECIPE["num_heads"]), "--d_kv", str(RECIPE["d_kv"]),
              "--dropout_rate", str(RECIPE["dropout_rate"]),
              "--device", "cuda", "--eval_mode", "item"]

    train_cmd = [VENV_PY, str(MODEL_DIR / "main.py"), *common,
                 "--mode", "train", "--lr", str(RECIPE["lr"]),
                 "--num_epochs", str(RECIPE["num_epochs"]),
                 "--early_stop", str(RECIPE["early_stop"]),
                 "--save_path", str(ckpt), "--log_path", str(log)]
    if a.dry_run:
        print("DRY-RUN t5 train:\n  " + " ".join(map(str, train_cmd))); return 0

    if ckpt.exists() and ckpt.stat().st_size > 1_000_000:
        print(f"[1/3] SKIP training -- checkpoint already present "
              f"({ckpt.stat().st_size} bytes): {ckpt}")
    else:
        print(f"[1/3] training T5 {ds}/{arm_dir} (stride={stride} vocab={vocab})")
        rc = run(train_cmd, MODEL_DIR, log)
        if rc or not ckpt.exists():
            print(f"FATAL: T5 training exit {rc}, ckpt exists={ckpt.exists()}"); return rc or 4

    # beams for both splits. dump_beam_outputs.py accepts only a SUBSET of the
    # training flags (it has no --vocab_size/--batch_size/--num_layers/... and no
    # --eval_mode), so build its argv separately rather than reusing `common`.
    beams = tok_dir / "beams"
    beams.mkdir(exist_ok=True)
    dump_common = ["--dataset_path", str(ART_ROOT / f"data/{ds}"), "--code_path", str(sid),
                   "--codebook_size", str(stride), "--seed", str(RECIPE["seed"]),
                   "--max_len", str(RECIPE["max_len"]),
                   "--beam_size", str(RECIPE["beam_size"]),
                   "--infer_size", str(RECIPE["infer_size"]),
                   "--device", "cuda"]
    results = {}
    for split in ("valid", "test"):
        out_npz = beams / f"{split}_k5_sid6.npz"
        cmd = [VENV_PY, str(MODEL_DIR / "dump_beam_outputs.py"), *dump_common,
               "--ckpt_path", str(ckpt), "--split", split,
               "--sid_len", str(sid_len), "--output_path", str(out_npz)]
        print(f"[2/3] dumping {split} beams")
        rc = run(cmd, MODEL_DIR, log)
        if rc or not out_npz.exists():
            print(f"FATAL: {split} beam dump exit {rc}"); return rc or 5
        results[split] = score_dump(out_npz)

    # metrics
    for split in ("valid", "test"):
        append_csv(OUT / f"single_5plus1/{split.upper()}_RESULTS_FINAL.csv", {
            "dataset": ds, "method": arm, "codebook": a.codebook,
            "semantic_positions": 5, "full_sid_length": sid_len,
            "stride": stride, "vocab_size": vocab, "split": split,
            "source": "new_5plus1", "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256(ckpt), "sid_path": str(sid),
            "sid_sha256": sv["sid_sha256"], **results[split]})

    (t5 / "RUN_METADATA.json").write_text(json.dumps({
        "phase": "PHASE 4 canonical T5 recommender", "dataset": ds, "arm": arm_dir,
        "codebook_size": a.codebook, "stride": stride, "vocab_size": vocab,
        "sid_len": sid_len, "sid_path": str(sid), "sid_sha256": sv["sid_sha256"],
        "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
        "selection_metric": "item-level VALID NDCG@20 (early_stop 10)",
        "eval_mode": "item (legacy token-prefix metric NOT used)",
        "test_used_for_selection": False,
        "recipe": RECIPE, "valid_metrics": results["valid"],
        "test_metrics": results["test"],
        "recipe_source": "configs/canonical.yaml",
        "wrote_only_under": str(OUT)}, indent=2) + "\n")

    print(f"[3/3] VALID N@20={results['valid']['NDCG@20']:.6f}  "
          f"TEST N@20={results['test']['NDCG@20']:.6f}")
    print("PHASE4_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
