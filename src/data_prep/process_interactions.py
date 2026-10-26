#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw Amazon Reviews 2014 -> preprocessed splits + user/item id mappings.

This reproduces the upstream TIGER preprocessing protocol:

1. read ``reviews_<Dataset>_5.json.gz`` line by line and write a "strict" JSON
   copy (one review object per line);
2. map ``reviewerID`` -> 1-based user id and ``asin`` -> 1-based item id, saving
   ``user_mapping.npy`` / ``item_mapping.npy``;
3. build per-user chronological item sequences (sorted by ``unixReviewTime``);
4. leave-one-out splits: ``train = seq[:-2]``, ``valid = seq[:-1]``,
   ``test = seq``; each split row is ``history`` (all but the last id) plus
   ``target`` (the last id);
5. write ``train.parquet``, ``valid.parquet``, ``test.parquet`` with columns
   ``user`` (1-based), ``history`` (list of 1-based ids), ``target`` (1-based).

Usage
-----
    python3 -m src.data_prep.process_interactions --dataset Beauty \
        --raw-dir /path/to/raw --out-dir /path/to/artifacts/data/Beauty

The real path needs the official Amazon Reviews 2014 files (never shipped).
``--synthetic-smoke`` generates a tiny **synthetic** reviews file (labelled
synthetic) so this stage can be exercised on CPU without the dataset.
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
DATASETS = ["Beauty", "Cell_Phones_and_Accessories",
            "Health_and_Personal_Care", "Toys_and_Games",
            "Sports_and_Outdoors"]


def parse(path):
    """Yield one JSON line per review, exactly as the upstream notebook did."""
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


def write_synthetic_raw(raw_dir: Path, dataset: str, n_users: int = 200,
                        n_items: int = 400, n_interactions: int = 12,
                        seed: int = 2025) -> Path:
    """Write a synthetic reviews_<Dataset>_5.json.gz (labelled synthetic).

    Only used by --synthetic-smoke so the preprocessing code can be executed
    end-to-end on a CPU without downloading the real 2014 dataset.  The size is
    chosen so downstream stages are numerically exercisable: the codebook is
    256 and the per-subspace PCA target is 64, both below the number of items.
    """
    rng = np.random.default_rng(seed)
    asins = [f'SYN{i:04d}' for i in range(n_items)]
    path = raw_dir / f'reviews_{dataset}_5.json.gz'
    raw_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        t = 1_000_000
        for u in range(n_users):
            for a in rng.choice(len(asins), size=n_interactions, replace=False):
                t += 1
                fh.write(json.dumps({
                    'reviewerID': f'SYNUSER{u:04d}',
                    'asin': asins[int(a)],
                    'unixReviewTime': t,
                    'overall': float(rng.integers(1, 6)),
                }) + '\n')
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Raw Amazon-Reviews-2014 reviews -> preprocessed splits '
                    '+ user/item id mappings (upstream TIGER protocol).')
    ap.add_argument('--dataset', required=True, choices=DATASETS)
    ap.add_argument('--raw-dir', default=os.environ.get(
        'CFLA_RAW_DATA', str(REPO / 'raw')),
        help='directory holding reviews_<Dataset>_5.json.gz')
    ap.add_argument('--out-dir', default=None,
                    help='output dataset directory; default: '
                         '<repo>/artifacts/data/<Dataset> '
                         '(i.e. $CFLA_ARTIFACT_ROOT/data/<Dataset>)')
    ap.add_argument('--synthetic-smoke', action='store_true',
                    help='generate a tiny SYNTHETIC raw file first (labelled '
                         'synthetic; for smoke-testing this script only)')
    a = ap.parse_args(argv)

    if a.out_dir is None:
        sys.path.insert(0, str(REPO))
        from src.utils import repo_paths as rp
        a.out_dir = str(rp.ARTIFACT_ROOT / 'data' / a.dataset)
    raw_dir = Path(a.raw_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_name = a.dataset
    raw_file = raw_dir / f'reviews_{dataset_name}_5.json.gz'
    if a.synthetic_smoke:
        raw_file = write_synthetic_raw(raw_dir, dataset_name)
        print(f'[synthetic] wrote {raw_file}')
    if not raw_file.exists():
        print(f'FATAL: missing raw reviews file: {raw_file}\n'
              f'Download the official Amazon Reviews 2014 5-core file for '
              f'{dataset_name} and pass --raw-dir (see THIRD_PARTY_NOTICES.md).',
              file=sys.stderr)
        return 2

    # ---- "strict" JSON copy (one review per line) --------------------------
    f = open(out / f'{dataset_name}.json', 'w')
    for l in parse(raw_file):
        f.write(l + '\n')
    f.close()

    data = open(out / f'{dataset_name}.json', 'r')
    print('Number of lines:', sum(1 for _ in data))
    data.seek(0)
    data.readline()
    data.close()

    # ---- id mappings + (user, item, timestamp) triples ---------------------
    userID_mapping = {}
    itemID_mapping = {}
    userIDs, itemIDs, timestamps = [], [], []

    data = open(out / f'{dataset_name}.json', 'r')
    for line in data:
        review = json.loads(line.strip())
        userID = review['reviewerID']
        itemID = review['asin']
        timestamp = review['unixReviewTime']

        if userID not in userID_mapping:
            userID_mapping[userID] = len(userID_mapping) + 1
        if itemID not in itemID_mapping:
            itemID_mapping[itemID] = len(itemID_mapping) + 1

        userIDs.append(userID_mapping[userID])
        itemIDs.append(itemID_mapping[itemID])
        timestamps.append(timestamp)

    np.save(out / 'user_mapping.npy', userID_mapping)
    print('user_num:', len(userID_mapping))
    np.save(out / 'item_mapping.npy', itemID_mapping)
    print('item_num:', len(itemID_mapping))

    # ---- per-user chronological sequences ----------------------------------
    user_item_mapping = {}
    for userID, itemID, timestamp in zip(userIDs, itemIDs, timestamps):
        user_item_mapping.setdefault(userID, []).append((itemID, timestamp))
    for userID in user_item_mapping:
        user_item_mapping[userID].sort(key=lambda x: x[1])
        user_item_mapping[userID] = [item[0] for item in user_item_mapping[userID]]

    # ---- leave-one-out splits (upstream TIGER protocol) --------------------
    train_data, val_data, test_data = {}, {}, {}
    for userID, item_sequence in user_item_mapping.items():
        train_data[userID] = item_sequence[:-2]
        val_data[userID] = item_sequence[:-1]
        test_data[userID] = item_sequence

    def prepare_data(data_dict):
        rows = []
        for userID, item_sequence in data_dict.items():
            history = item_sequence[:-1]
            target = item_sequence[-1]
            rows.append({'user': userID, 'history': history, 'target': target})
        return pd.DataFrame(rows)

    train_df = prepare_data(train_data)
    val_df = prepare_data(val_data)
    test_df = prepare_data(test_data)
    print('Training data shape:', train_df.shape)
    print('Validation data shape:', val_df.shape)
    print('Testing data shape:', test_df.shape)

    train_df.to_parquet(out / 'train.parquet', index=False)
    val_df.to_parquet(out / 'valid.parquet', index=False)
    test_df.to_parquet(out / 'test.parquet', index=False)
    print('Data saved to parquet files.')
    print('OUTPUT_DIR', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
