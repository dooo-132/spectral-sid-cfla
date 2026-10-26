# DATA FORMAT

**No dataset or model artifact is included in this repository.** Everything the code
needs at run time is an *input artifact* that you prepare locally and point to with
environment variables (see `docs/ENVIRONMENT.md`):

| variable | meaning | default |
|---|---|---|
| `CFLA_ARTIFACT_ROOT` | input artifacts (parquet splits, mappings, item embeddings, SID arrays, checkpoints, beam dumps) | `<repo>/artifacts` |
| `CFLA_OUTPUT_ROOT` | root for all new artifacts written by a run | `<repo>/outputs` |
| `CFLA_UPSTREAM_ROOT` | fallback root consulted when a file is missing from `CFLA_ARTIFACT_ROOT` | `$CFLA_ARTIFACT_ROOT` |
| `CFLA_RAW_DATA` | raw Amazon Reviews 2014 files | `<repo>/raw` |

## Layout produced by `scripts/prepare_data.sh`

```
${CFLA_ARTIFACT_ROOT}/data/<Dataset>/
├── train.parquet  valid.parquet  test.parquet     # user, history, target
├── item_mapping.npy  user_mapping.npy             # 1-based ids
├── <Dataset>.json  <Dataset>_metadata.json         # intermediate strict-JSON copies
└── item_emb.parquet                               # itemID, embedding (SID input)
```

`<Dataset>` is one of `Beauty`, `Cell_Phones_and_Accessories`,
`Health_and_Personal_Care`, `Toys_and_Games`, `Sports_and_Outdoors`.

## Parquet splits

| column | type | meaning |
|---|---|---|
| `user` | int | user id, **1-based** |
| `history` | list[int] | **1-based** item ids, chronological, left-truncated to `max_len` items |
| `target` | int | **1-based** next item id |

Protocol (inherited from TIGER): the same users appear in all three splits; the
splits are consecutive chronological segments of each user's sequence.
`valid.target == test.history[-1]`, so validation and test are adjacent exposures,
not independent populations. Validation-based selection is therefore mildly
optimistic; CFLA's calibration statistics are fitted on validation candidates only.

## SID arrays

An SID array is an `int64` `.npy` file of shape `(n_items, L)`; row `i` is the SID of
dataset item `i + 1`; codes are in `[0, codebook_size)`. Tokens use the slot-wise
layout `token = code + position*codebook_size + 1`, so the model vocabulary is
`vocab_size = codebook_size * L + 1` (see `docs/SID_FORMAT.md`). This format is used
for the paper-owned spectral and dimension-split SIDs and must also be used for
externally built RQ-VAE / RQ-OPQ SID arrays.

## Beam dumps

`src/tiger/dump_beams.py` writes one `.npz` per (dataset, factorization, split):

| key | shape | dtype | meaning |
|---|---|---|---|
| `candidate_item_ids` | `(n_users, beam)` | int64 | **0-based** item id, `-1` = invalid |
| `candidate_tokens` | `(n_users, beam, L)` | int64 | slot-wise tokens, `0` = pad |
| `sequence_scores` | `(n_users, beam)` | float32 | **length-normalised log-prob** `Σ logp / L` |
| `transition_logprobs` | `(n_users, beam, L)` | float32 | per-token log-probs |
| `target_item_ids` | `(n_users,)` | int64 | **0-based** target |
| `split`, `sid_len`, `beam_size`, `vocab_size` | scalar | | metadata |

### `sequence_scores` are log-probabilities, not probabilities

They are ≤ 0 and equal to `Σ transition_logprobs / L`. Any entropy/confidence
computation must softmax first (see `src/fusion/common.py::_row_softmax`).

## CFLA pool config

`configs/cfla_pool.example.yaml` lists, for every factorization, the SID array, the
trained checkpoint and the valid/test beam dumps, plus `dataset_dir`,
`codebook_size` and `max_len`. All factorizations must share the same dataset, the
same user order and the same split; the runner checks user alignment and vocabulary
consistency.
