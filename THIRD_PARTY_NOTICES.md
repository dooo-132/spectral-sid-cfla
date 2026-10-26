# Third-Party Notices

This release is an anonymous, code-only artifact. It contains no dataset,
checkpoint, beam dump or other experiment artifact. The source code builds on
the following public work; the corresponding licences and terms apply.

## TIGER — Recommender Systems with Generative Retrieval

Rajput, S., Mehta, N., Singh, A., Hulikal, R., & Ramaswamy, V. (2023).
*Recommender Systems with Generative Retrieval.* RecSys '23.
<https://arxiv.org/abs/2305.05065>

The T5 encoder-decoder training/inference code (TIGER architecture, semantic-ID
data plumbing) in `src/tiger/`, `src/analysis/teacher_forced.py` and parts of
`src/fusion/` is derived from the TIGER method.

## Unofficial PyTorch implementation of TIGER

`src/tiger/`, `src/sid/` and the training/evaluation utilities are derived, in
part, from an unofficial PyTorch implementation of TIGER distributed under the
MIT License (Copyright (c) 2025). The reference RQ-VAE implementation used for
the RQ-VAE tokenizer (`src/tokenization/legacy_rqvae/`) also derives from it.

For anonymous review the original copyright-holder name has been withheld here;
it must be restored, together with the upstream MIT notice, before a
camera-ready public release. The same applies to `LICENSE`: the copyright line
reads "Anonymous Authors" and must be replaced with the real holder before the
paper is de-anonymised.

## OneSearch — RQ-OPQ reference

The RQ-OPQ Semantic-ID generator follows the public OneSearch code structure:
FAISS KMeans followed by residual subtraction (RQ) and an
`OPQ{M},IVF1,PQ{M}x{bits}` FAISS index factory (OPQ).
<https://arxiv.org/abs/2212.01354> (OneSearch: Large Language Models for
Product Search)

## Amazon Reviews 2014 dataset

The experiments use the Amazon Reviews 2014 benchmark
(Beauty, Cell_Phones_and_Accessories, Health_and_Personal_Care,
Toys_and_Games, Sports_and_Outdoors), distributed by J. McAuley and
J. Leskovec: <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>
(formerly at <https://jmcauley.ucsd.edu/data/amazon/2014.html>).
The dataset is **not** included in this release. Users must obtain it from the
original distributor and comply with its terms of use; the preprocessed splits
and item embeddings are inputs the user prepares locally
(see `docs/DATA_FORMAT.md`).

## Python packages

The code imports PyTorch, HuggingFace Transformers, NumPy, SciPy,
scikit-learn, pandas/pyarrow, PyYAML, tqdm, matplotlib, FAISS and
sentence-transformers. They are distributed under their own respective
licences (BSD-3-Clause for PyTorch/FAISS/NumPy/SciPy/pandas/scikit-learn,
Apache-2.0 for Transformers, MIT for tqdm/PyYAML, PSF for matplotlib,
Apache-2.0 for sentence-transformers). Exactly verified versions are listed in
`requirements.txt`.
