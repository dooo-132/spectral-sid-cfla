# Third-Party Notices

This release is an anonymous, code-only artifact. It contains no dataset,
checkpoint, beam dump or other experiment artifact. The source code builds on the
following public work; the corresponding licences and terms apply.

## TIGER — Recommender Systems with Generative Retrieval

Rajput, S., Mehta, N., Singh, A., Hulikal, R., & Ramaswamy, V. (2023).
*Recommender Systems with Generative Retrieval.* RecSys '23.
<https://arxiv.org/abs/2305.05065>

The T5 encoder–decoder architecture, the semantic-ID data plumbing and the
generative-recommendation training/inference protocol in `src/tiger/`,
`src/analysis/teacher_forced.py` and parts of `src/fusion/` follow this method.

## RQ-VAE (not bundled)

The paper's RQ-VAE factorization can be built with the public implementation of
*Autoregressive Image Generation using Residual Quantization* (CVPR '22):

* <https://github.com/kakaobrain/rq-vae-transformer>

The RQ-VAE implementation is **not included** in this release. The hyper-parameters
used for the paper's RQ-VAE5 tokenizer are recorded in
`configs/paper/RQ5_RECIPE_FINAL.json`. See that repository for its licence and usage
terms.

## RQ-OPQ / OneSearch (not bundled)

The paper's RQ-OPQ row is RQ2-OPQ3 (2 residual-quantization positions + 3 OPQ/PQ
positions + 1 disambiguation token), following the OneSearch-style construction:

* OneSearch paper: <https://arxiv.org/abs/2212.01354>
* FAISS (OPQ / product quantization): <https://github.com/facebookresearch/faiss>
* residual quantization: <https://github.com/kakaobrain/rq-vae-transformer>

No RQ-OPQ implementation is included in this release. See the linked repositories
for their licences and terms.

## Amazon Reviews 2014 dataset

The experiments use the Amazon Reviews 2014 benchmark (Beauty,
Cell_Phones_and_Accessories, Health_and_Personal_Care, Toys_and_Games,
Sports_and_Outdoors), distributed by J. McAuley and J. Leskovec:
<https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>

The dataset is **not** included in this release. Users must obtain it from the
original distributor and comply with its terms of use; the preprocessed splits and
item embeddings are inputs the user prepares locally (see `docs/DATA_FORMAT.md`).

## Python packages

The code imports PyTorch, HuggingFace Transformers, NumPy, SciPy, scikit-learn,
pandas/pyarrow, PyYAML, tqdm and (optionally) sentence-transformers. They are
distributed under their own licences. Exactly verified versions are listed in
`requirements.txt`.

## Project licence

`LICENSE` is the MIT licence with the copyright holder anonymized as
"Anonymous Authors" for double-blind review; it must be restored, together with any
upstream notices, before a public de-anonymized release.
