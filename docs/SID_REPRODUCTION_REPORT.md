# SID REPRODUCTION REPORT

**What this documents.** In the authors' environment, the spectral SID
implementation shipped in `src/sid/` was used to regenerate the canonical
spectral SID mapping from the item-embedding matrix, and its output was compared
against the frozen SID arrays used by the paper. On the Beauty category the
regeneration was **bit-identical for every K**.

**Status in this release: not re-executable as-is and not a shipped result.** The comparison script and
the frozen reference arrays are experiment artifacts and are not shipped. The
report is kept because it documents the validation the implementation received
and the canonical numerical parameters (which must not drift).

**Result: 4/4 EXACT** (Beauty, `K ∈ {2,3,4,5}`).

| K | verdict | A. exact | B. label-perm | C. partition | D. item→SID |
|---|---|---|---|---|---|
| 2 | **exact** | True | True | True | True |
| 3 | **exact** | True | True | True | True |
| 4 | **exact** | True | True | True | True |
| 5 | **exact** | True | True | True | True |

Because level A (byte equality) holds, the weaker equivalences are implied.
They are reported anyway because a clustering pipeline *can* be correct while
differing by a label permutation.

## Equivalence levels

| Level | Definition | Meaning if it holds |
|---|---|---|
| **A. exact** | `np.array_equal` on `sid.npy`, `base_sid.npy`, `dim_labels.npy` | bit-identical output |
| **B. label permutation** | same dimension partition up to a bijective relabelling of cluster ids | the clustering found the same grouping |
| **C. partition (per position)** | same item partition per base position up to relabelling | the quantization found the same item grouping |
| **D. item → SID** | B and C both hold, so the item→SID mapping agrees | downstream metrics are unaffected |

## Reproduced artifacts

For each K, all four arrays matched:

| artifact | compared against |
|---|---|
| `dim_labels.npy` (768,) | `<Dataset>_t5_spectral_nonneg_pearson_v3_k{K}_sid{K+1}_dim_labels.npy` |
| `dim_groups.npy` (K,) | `..._dim_groups.npy` |
| `base_sid.npy` (n_items, K) | `..._base_sid.npy` |
| `sid.npy` (n_items, K+1) | `..._sid{K+1}.npy` |

## Canonical parameters (must not drift)

| Parameter | Value | Why it matters |
|---|---|---|
| `affinity` | `nonneg_pearson` (= `(rho+1)/2`) | the canonical variant; `abs_pearson` is the ablation |
| `k` | 2, 3, 4, 5 | number of semantic subspaces |
| `codebook_size` | 256 | k-means centroids per subspace |
| `seed` | 2025 | drives `SpectralClustering(random_state)` and `KMeans(random_state)` |
| `pca_dim` | 64 | PCA whitening inside each subspace when its width exceeds 64 |
| `SpectralClustering` | `affinity='precomputed'`, `assign_labels='kmeans'`, `n_init=20` | part of the canonical configuration |
| `KMeans` | `n_init=30`, `max_iter=500` | part of the canonical configuration |

`random_state` and `n_init` are **not** cosmetic: they are what makes the output
reproducible. Changing them is a configuration change, not a refactor.

## Interior statistics (Beauty, K=2) as generated

| Quantity | Value |
|---|---|
| group sizes | `[397, 371]` |
| base-prefix collision rate | `0.7937360548714982` |
| number of collision groups | 1041 |
| max collision group | 78 (< 256, so one disambiguation token suffices) |
| final unique SID count | 12101 / 12101 (correctness assertion, not a performance metric) |

## What this proves and does not prove

**Proves:** the released `src/sid/` code is a complete, self-contained spectral
SID construction pipeline that reproduces the canonical artifacts bit-for-bit
from the item embeddings in the authors' environment, with no runtime source
patching.

**Does not prove:** anything about the other categories. Only Beauty was run for
this check; the other datasets have configs under `configs/experiments/` but
their canonical reproduction was not verified.
