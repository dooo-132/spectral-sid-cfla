# SID IMPLEMENTATION NOTE

`src/sid/` is a self-contained implementation of the paper's spectral Semantic-ID
construction: dimension–dimension Pearson affinity → normalized Laplacian spectral
partition → per-subspace PCA whitening + k-means (codebook 256) → deterministic
disambiguation token making the item→SID map injective.

In the authors' environment this implementation was checked against the frozen SID
arrays used for the paper: on the Beauty category the regenerated `sid.npy`,
`base_sid.npy` and `dim_labels.npy` were bit-identical for K = 2, 3, 4, 5. That check
used the authors' frozen arrays and is **not re-executable from this release** (no
data or frozen artifacts are shipped); it is reported as background for the
implementation, not as a result of this release.

The released code additionally asserts the design invariants at generation time:
output shapes, unchanged semantic prefix, injectivity of the final SID, and code
range `[0, codebook_size)`. `scripts/smoke_pipeline.sh` exercises the construction
end to end on synthetic embeddings.

Canonical parameters (must not drift for reproduction): affinity `nonneg_pearson`
(`(ρ+1)/2`), `K ∈ {2,3,4,5}`, `codebook_size=256`, `pca_dim=64`, seed `2025`,
`SpectralClustering(affinity='precomputed', assign_labels='kmeans', n_init=20)`,
`KMeans(n_init=30, max_iter=500)`.
