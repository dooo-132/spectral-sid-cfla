#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Item semantic embeddings for SID construction (Amazon Reviews 2014 metadata).

The SID construction pipeline (``src/sid/``) consumes ``item_emb.parquet``: one
768-dimensional sentence embedding per item, produced from the item metadata by
a sentence-transformer (``sentence-t5-base`` in the paper).

Protocol (upstream TIGER convention, unchanged):

1. read the item metadata file ``<raw-dir>/meta_<Dataset>.json.gz`` (one JSON
   object per line) and keep the metadata of items that occur in
   ``item_mapping.npy`` (the ASIN -> 1-based item-id map written by
   ``process_interactions.py``);
2. build one text per item: ``Title: ... Brand: ... Categories: ... Price: ...``;
3. encode every text with ``sentence-transformers/sentence-t5-base`` and store
   ``{"itemID": <1-based id>, "embedding": <768 floats>}`` rows;
4. write ``<out-dir>/item_emb.parquet`` with rows ordered by 1-based item id.

Usage
-----
    python3 -m src.data_prep.make_item_embeddings --dataset Beauty \
        --raw-dir /path/to/raw --model-dir /path/to/sentence-t5-base

The real path needs ``sentence-transformers`` and a local copy of the
``sentence-t5-base`` weights (published separately; see THIRD_PARTY_NOTICES.md).
``--synthetic-smoke`` instead writes *deterministic synthetic* embeddings (a
seeded RNG, no model, no metadata) so that the downstream pipeline can be
smoke-tested on CPU without the dataset or the model.  Synthetic output is
labelled as such in the printed log.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EMB_DIM_DEFAULT = 768


def parse(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as g:
        for line in g:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                import ast
                yield ast.literal_eval(line)


def load_item_mapping(dataset_dir: Path) -> dict:
    """ASIN -> 1-based item id (written by process_interactions.py)."""
    path = dataset_dir / "item_mapping.npy"
    if not path.exists():
        raise SystemExit(
            f"FATAL: {path} not found. Run scripts/prepare_data.sh --stage "
            f"interactions first (it writes item_mapping.npy).")
    return np.load(path, allow_pickle=True).item()


def write_metadata_json(dataset: str, raw_dir: Path, out_dir: Path) -> Path:
    """Copy only the metadata fields used for the text of mapped items."""
    item_mapping = load_item_mapping(out_dir)
    valid_asins = set(item_mapping.keys())
    metadata_out_path = out_dir / f"{dataset}_metadata.json"
    with open(metadata_out_path, "w", encoding="utf-8") as f:
        for metadata in parse(raw_dir / f"meta_{dataset}.json.gz"):
            asin = metadata.get("asin")
            if asin is None or asin not in valid_asins:
                continue
            selected = {
                "asin": asin,
                "title": metadata.get("title") or None,
                "price": metadata.get("price") or None,
                "salesRank": metadata.get("salesRank") or None,
                "brand": metadata.get("brand") or None,
                "categories": metadata.get("categories") or None,
            }
            f.write(json.dumps(selected, ensure_ascii=False) + "\n")
    print(f"{dataset}_metadata.json saved: {metadata_out_path}")
    return metadata_out_path


def build_semantic_texts(dataset: str, out_dir: Path):
    """One text per 1-based item id, in item-id order."""
    item_mapping = load_item_mapping(out_dir)
    asin_to_meta = {}
    with open(out_dir / f"{dataset}_metadata.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta = json.loads(line)
                asin_to_meta[meta["asin"]] = meta

    id_to_asin = {v: k for k, v in item_mapping.items()}
    max_item_id = max(id_to_asin.keys())
    semantics, item_ids = [], []
    for item_id in range(1, max_item_id + 1):
        metadata = asin_to_meta.get(id_to_asin.get(item_id), {})
        semantic = ""
        if metadata.get("title"):
            semantic += f"Title: {metadata['title']}. "
        if metadata.get("brand"):
            semantic += f"Brand: {metadata['brand']}. "
        if metadata.get("categories"):
            semantic += f"Categories: {metadata['categories']}. "
        if metadata.get("price"):
            semantic += f"Price: {metadata['price']}. "
        semantics.append(semantic if semantic else "No metadata.")
        item_ids.append(item_id)
    return item_ids, semantics


def encode_with_model(semantics, model_dir: Path, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "FATAL: sentence-transformers is not installed. Install it from "
            "requirements.txt (`pip install sentence-transformers`) or use "
            "--synthetic-smoke for a CPU-only pipeline test.")
    if not model_dir.exists():
        raise SystemExit(
            f"FATAL: sentence-transformer model directory not found: {model_dir}\n"
            f"Pass --model-dir, or set CFLA_SENTENCE_T5_DIR. The model weights "
            f"are NOT shipped with this release (see THIRD_PARTY_NOTICES.md).")
    model = SentenceTransformer(str(model_dir))
    print("Encoding item semantics...")
    return model.encode(list(semantics), batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True,
                        normalize_embeddings=False)


def synthetic_embeddings(item_ids, dim: int, seed: int):
    """Deterministic SYNTHETIC embeddings (no metadata, no model)."""
    rng = np.random.default_rng(seed)
    n = len(item_ids)
    latent = rng.normal(size=(n, max(4, dim // 8)))
    proj = rng.normal(size=(latent.shape[1], dim))
    x = latent @ proj + 0.1 * rng.normal(size=(n, dim))
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    return x.astype(np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build item_emb.parquet for SID construction.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-dir", default=os.environ.get("CFLA_RAW_DATA",
                                                        str(REPO / "raw")),
                    help="directory holding meta_<Dataset>.json.gz")
    ap.add_argument("--out-dir", default=None,
                    help="dataset directory (holds item_mapping.npy); default "
                         "$CFLA_ARTIFACT_ROOT/data/<Dataset>")
    ap.add_argument("--model-dir",
                    default=os.environ.get("CFLA_SENTENCE_T5_DIR",
                                           str(REPO / "sentence-t5-base")),
                    help="local sentence-t5-base checkpoint directory")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=EMB_DIM_DEFAULT,
                    help="synthetic-smoke only: embedding dimension")
    ap.add_argument("--synthetic-smoke", action="store_true",
                    help="write deterministic SYNTHETIC embeddings (no model, "
                         "no metadata; for CPU smoke tests only)")
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args(argv)

    raw_dir = Path(a.raw_dir)
    if a.out_dir:
        out_dir = Path(a.out_dir)
    else:
        sys.path.insert(0, str(REPO))
        from src.utils import repo_paths as rp
        out_dir = rp.ARTIFACT_ROOT / "data" / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    item_mapping = load_item_mapping(out_dir)
    print("Number of mapped items:", len(item_mapping))

    if a.synthetic_smoke:
        item_ids = sorted(int(v) for v in item_mapping.values())
        embeddings = synthetic_embeddings(item_ids, a.emb_dim, a.seed)
        print(f"[synthetic] {len(item_ids)} SYNTHETIC {a.emb_dim}-d embeddings "
              f"(seed {a.seed}); no real metadata or model was used")
    else:
        meta_path = raw_dir / f"meta_{a.dataset}.json.gz"
        if not meta_path.exists():
            print(f"FATAL: missing metadata file: {meta_path}\n"
                  f"Download the official Amazon Reviews 2014 metadata file "
                  f"for {a.dataset} and pass --raw-dir.", file=sys.stderr)
            return 2
        write_metadata_json(a.dataset, raw_dir, out_dir)
        item_ids, semantics = build_semantic_texts(a.dataset, out_dir)
        print("Number of semantic texts:", len(semantics))
        embeddings = encode_with_model(semantics, Path(a.model_dir),
                                       a.batch_size)

    item_emb_df = pd.DataFrame(
        [{"itemID": int(i),
          "embedding": np.asarray(e, dtype=np.float32).tolist()}
         for i, e in zip(item_ids, embeddings)])
    out_path = out_dir / "item_emb.parquet"
    item_emb_df.to_parquet(out_path, index=False)
    print("Item embeddings DataFrame shape:", item_emb_df.shape)
    print("Saved:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
