# DATA FORMAT

**No dataset or model artifact is included in this repository.** Everything the
code needs at run time is an *input artifact* that you prepare locally and point
to with environment variables (see `docs/ENVIRONMENT.md`):

| variable | meaning | default |
|---|---|---|
| `CFLA_ARTIFACT_ROOT` | input artifact root (preprocessed data, SID arrays, checkpoints, beam dumps) | `<repo>/artifacts` |
| `CFLA_OUTPUT_ROOT` | root for **all new artifacts** written by a run | `<repo>/outputs` |
| `CFLA_RAW_DATA` | directory with the raw Amazon Reviews 2014 files | `<repo>/raw` |

The code never writes into `CFLA_ARTIFACT_ROOT`.

## Expected layout under `CFLA_ARTIFACT_ROOT`

```
<ARTIFACT_ROOT>/
├── data/<Dataset>/
│   ├── train.parquet, valid.parquet, test.parquet      # columns: user, history, target
│   ├── item_mapping.npy, user_mapping.npy              # ASIN -> 1-based item id, etc.
│   ├── item_emb.parquet                                # SID-construction input (itemID, embedding)
│   ├── <Dataset>_t5_spectral_<affinity>_k<K>_sid<L>.npy          # full SID array
│   ├── <Dataset>_t5_spectral_<affinity>_k<K>_sid<L>_base_sid.npy # K semantic codes
│   ├── <Dataset>_t5_spectral_<affinity>_k<K>_sid<L>_dim_groups.npy
│   └── <Dataset>_t5_spectral_<affinity>_k<K>_sid<L>_dim_labels.npy
├── model/ckpt/tiger_<Dataset>_<affinity>_k<K>_sid<L>.pth
└── model/beam_dumps_<Dataset>_<affinity>/<split>_k<K>_sid<L>.npz
```

`<affinity>` is `nonneg_pearson_v3` for the canonical spectral runs
(`--affinity nonneg_pearson`; `_v3` is the generator version suffix).
`<Dataset>` is one of `Beauty`, `Cell_Phones_and_Accessories`,
`Health_and_Personal_Care`, `Toys_and_Games`, `Sports_and_Outdoors`.

## Parquet splits

| column | type | meaning |
|---|---|---|
| `user` | int | user id, **1-based** |
| `history` | list[int] | **1-based** item ids, chronological |
| `target` | int | **1-based** next item id |

Protocol (inherited from TIGER): the **same users** appear in all three splits;
the splits are consecutive chronological segments of each user's sequence.
`valid.target == test.history[-1]`, i.e. valid and test are *adjacent
exposures*, not independent populations. Validation-based selection is
therefore mildly optimistic; the canonical uniform-fusion baseline requires no
selection at all, and CFLA's calibration statistics are fitted on validation
candidates only.

## SID arrays

`<Dataset>_t5_spectral_<affinity>_k<K>_sid<L>.npy` is an `int64` array of shape
`(n_items, L)`, `L = K + 1`; row `i` is the SID of dataset item `i + 1`; codes
are in `[0, 255]`. See `docs/SID_FORMAT.md` for the flat token layout
(`token = code + position*256 + 1`, `vocab = 256*L + 1`) and
`docs/ITEM_ID_CONVENTION.md` for the item-id spaces.

## Beam dump `.npz` keys

| key | shape | dtype | meaning |
|---|---|---|---|
| `candidate_item_ids` | `(n_users, beam)` | int64 | **0-based** item id, `-1` = invalid |
| `candidate_tokens` | `(n_users, beam, L)` | int64 | flat tokens, `0` = pad |
| `sequence_scores` | `(n_users, beam)` | float32 | **length-normalised log-prob**: `Σ token_logprob / L` |
| `transition_logprobs` | `(n_users, beam, L)` | float32 | per-token log-probs |
| `target_item_ids` | `(n_users,)` | int64 | **0-based** target |
| `split`, `sid_len`, `beam_size`, `vocab_size` | scalar | | metadata |
| `ckpt_path`, `code_path` | scalar | str | provenance |

### `sequence_scores` are log-probabilities, not probabilities

`sequence_scores == Σ transition_logprobs / L`; they are ≤ 0. **Any entropy /
confidence computation must softmax first** — feeding raw log-probs into
`p*log(p)` yields NaN and silently corrupts every ranking. See
`src/fusion/common.py::_row_softmax`.

## Regenerating artifacts

New beam dumps are written under `CFLA_OUTPUT_ROOT/beam_dumps/` by
`src/tiger/dump_beams.py`, which
* requires `--codebook_size` and defaults it to the canonical 256;
* asserts `vocab_size == sid_len * codebook_size + 1`;
* raises if the candidate decode rate drops below 0.90 (silent-failure guard).

New SID arrays are written under `CFLA_OUTPUT_ROOT/sids/...` by
`scripts/build_sid.sh`; see `docs/SID_FORMAT.md`.
