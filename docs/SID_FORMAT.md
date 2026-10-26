# SID FORMAT

## Layout

A Semantic ID is a fixed-length sequence of integer **codes**:

```
SID(item) = [c_0, c_1, ..., c_{L-1}]        each c in [0, 255]
```

stored as row `item-1` of `data/<dataset>/<dataset>_t5_spectral_<affinity>_k<K>_sid<L>.npy`
with dtype `int64` and shape `(n_items, L)`.

* `K` = number of spectral dimension groups.
* `L = K + 1` (`K` semantic tokens + 1 collision-disambiguation token).

| K | L | vocab size |
|---|---|---|
| 2 | 3 | 769 |
| 3 | 4 | 1025 |
| 4 | 5 | 1281 |
| 5 | 6 | 1537 |

## Token encoding (critical)

Codes are mapped into a **single flat token vocabulary** using a *slot-wise*
layout. Token at position `p` is:

```
token = code + p * codebook_size + 1        codebook_size = 256
```

so position `p` occupies the exclusive range `[256p + 1, 256(p+1)]`:

| position | token range |
|---|---|
| 0 | 1 – 256 |
| 1 | 257 – 512 |
| 2 | 513 – 768 |
| 3 | 769 – 1024 |
| 4 | 1025 – 1280 |
| 5 | 1281 – 1536 |

Token `0` is the padding token (also used as EOS). Hence

```
vocab_size = codebook_size * L + 1 = 256 * L + 1
```

**This is NOT `256^L + 1`.** An earlier draft of the paper stated the exponent
form; it is wrong and matches nothing (it would give 16.7M–2.8e14).

## Decoding

```python
def tokens_to_sid(tokens, codebook_size=256):
    sid = []
    for p, t in enumerate(tokens):
        lo, hi = p * codebook_size + 1, (p + 1) * codebook_size
        if not (lo <= t <= hi):
            return None          # malformed / truncated beam row
        sid.append(t - lo)
    return tuple(sid)
```

`None` means "not a valid SID"; callers must treat it as "no item", never as
item 0.

## Decoding to an item

Build the inverse map once from the SID array (row `i` → item `i+1`):

```python
sid_to_item = {tuple(map(int, row)): i + 1 for i, row in enumerate(codes)}
item = sid_to_item.get(sid_tuple, -1)
```

## Collision handling

The first `K` positions are the **semantic prefix**; the last position is a
per-collision-group local index. Because that index makes the full SID unique,
the **final collision rate is identically zero by construction** — it is a
design property, not an empirical result. The informative quantity is the
**base-prefix** collision rate. In this project the largest collision group is
133 items (Health, K=2) against a codebook of 256, so one disambiguation slot
always sufficed.

## Assumptions to assert in any new code

```python
assert sid_len == K + 1
assert vocab_size == 256 * sid_len + 1
assert codes.max() < 256
assert len({tuple(r) for r in codes}) == len(codes)   # injective
```
