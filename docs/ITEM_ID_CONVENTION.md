# ITEM ID CONVENTION

> **This is the single most error-prone aspect of this codebase.** A one-off
> mistake here silently produced wrong collision statistics and made the
> token/decoder path look broken. Read this before touching any item id.

## The four id spaces

| Name | Meaning | Range | Where it appears |
|---|---|---|---|
| **dataset item id** | item id as stored in the parquet `history`/`target` columns and used as the key of `item2code` | **1-based**, `1..N` | `data/*/{train,valid,test}.parquet` |
| **SID array row index** | 0-based row into `<dataset>_t5_spectral_*.npy` | `0..N-1` | the SID `.npy` arrays |
| **canonical internal item id** | the id used inside `src/` | **1-based**, `1..N` | all `src/` code |
| **raw dump item id** | the id written into historical beam-dump `.npz` files | **0-based**, `0..N-1`, with `-1` = "no item" | `model/beam_dumps_*/*.npz` |

## The relationships

```
dataset item id  ==  SID array row index + 1
canonical item id == dataset item id
raw dump item id  == canonical item id - 1        (for >= 0)
canonical item id == raw dump item id + 1         (for raw dump id >= 0)
-1                == "no item decoded" sentinel in raw dumps
```

## Why the raw dumps are 0-based

Every historical beam dump was produced by the pre-fix
`dump_beam_outputs.build_sid_to_item`, which stored the **0-based row index**
of the SID array (`mp[key] = idx`). Verified three independent ways on all four
datasets:

1. `dump.target_item_ids + 1 == parquet target` for **100%** of users
   (Beauty 22,363 / Cell Phones 27,879 / Health 38,609 / Toys 19,412).
2. `candidate_item_ids.max() == n_items - 1`.
3. Reconstructing tokens as `sid_array[dump_item_id][pos] + 256*pos + 1`
   reproduces `candidate_tokens` for **100.00%** of candidates, whereas
   `sid_array[dump_item_id - 1]` reproduces only ~13%.

## What this does and does not affect

| | Affected? |
|---|---|
| Recall / NDCG (ranking metrics) | **No** — the shift is uniform over candidates *and* targets, so ranks are unchanged. All 48 canonical metrics reproduce the audit to 6 decimals after canonicalisation. |
| Prefix / collision analysis | **Yes** — must canonicalise or you count the wrong item. |
| Joining dumps to SID arrays | **Yes**. |
| Reporting an item id | **Yes**. |

## Rule for new code

Always pass raw-dump ids through `src.fusion.common.canonicalize_item_ids`
before using them, and always write canonical ids into new artifacts.

```python
from src.fusion.common import canonicalize_item_ids
cand = canonicalize_item_ids(dump["candidate_item_ids"])   # 0-based -> 1-based
target = canonicalize_item_ids(dump["target_item_ids"])
```

`dumps produced from now on should record 1-based ids`; if you regenerate them,
update this document and `tests/` accordingly.

## One known edge case

Exactly **one user per dataset** has canonical target item `1`, which the 0-based
dumps encode as the sentinel `0` and therefore lose. That user contributes 0 to
every metric. The effect is ≤ 1/19,412 and is documented rather than patched
(the raw dumps are immutable). See `test_no_zero_metric` for the guard.
