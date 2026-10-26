# CODE_MAPPING — paper ↔ code

This document maps each component of the paper to the code shipped **in this
release**. File/module paths are authoritative; `:line` references are approximate.
The release is the runnable from-scratch pipeline: dataset in, CFLA metrics out.
Research-support scripts that depended on the authors' frozen experiment artifacts
are deliberately not included (see `docs/RELEASE_VALIDATION.md`).

## 0. End-to-end flow

```
Amazon Reviews 2014 (user-downloaded)
  └─ scripts/prepare_data.sh   src/data_prep/process_interactions.py, make_item_embeddings.py
       └─ data/<Dataset>/{train,valid,test}.parquet, item_mapping.npy, item_emb.parquet
  └─ scripts/build_sid.sh      src/sid/         (spectral SID, K=2..5)
                               src/tokenization/dimension_split/  (even/random baseline)
       └─ <SID array>.npy  (n_items, L)
  └─ scripts/train.sh          src/tiger/train.py
       └─ one T5 checkpoint per factorization
  └─ scripts/generate_candidates.sh  src/tiger/dump_beams.py
       └─ beam dumps for valid/test
  └─ scripts/run_cfla.sh       src/cfla/run.py (core: src/cfla/cfla.py)
       └─ candidate union -> teacher-forced candidate NLL -> validation calibration
          -> log-sum-exp aggregation -> Recall/NDCG + paired bootstrap
```

## 1. Spectral SID construction (paper §3.2)

| paper element | code | notes |
|---|---|---|
| item embedding input `X ∈ R^{N×d}` | `src/sid/embedding.py` (`load_item_embedding`, `align_embeddings`) | reads `item_emb.parquet`; two alignment modes (`canonical`, `item_id`) |
| dimension–dimension Pearson affinity `A_pq=(ρ+1)/2`, `A_pp=1` | `src/sid/affinity.py` (`build_dimension_affinity`) | canonical metric `nonneg_pearson` |
| normalized Laplacian + K smallest eigenvectors + k-means on dimensions | `src/sid/spectral_partition.py` (`spectral_group_dimensions`) | `SpectralClustering(affinity='precomputed', assign_labels='kmeans', n_init=20)` |
| per-subspace quantization with codebook `C` | `src/sid/quantizer.py` (`generate_base_sid_by_group_kmeans`) | per-group PCA whitening (`pca_dim=64`), `KMeans(n_init=30, max_iter=500)` |
| disambiguation token, injective item→SID | `src/sid/collision.py` (`add_collision_token`, `collision_stats`) | deterministic per-collision-group index appended as the last token |
| artifact serialization | `src/sid/io.py` (`write_sid_artifacts`) | writes `sid.npy`, `base_sid.npy`, `dim_groups.npy`, `dim_labels.npy`, `metadata.json`, `manifest.sha256` under `CFLA_OUTPUT_ROOT` |
| CLI | `src/sid/generator.py` (`python3 -m src.sid.generator`) | `--dataset --affinity --k --codebook_size --seed --pca_dim` |
| slot-wise token layout `token = code + pos*256 + 1` | `src/tiger/dataset.py` (`item2code`, `infer_codebook_size_from_vocab`) | `vocab_size = 256*L + 1` (see `docs/SID_FORMAT.md`) |

## 2. Alternative factorizations

| factorization | code | status |
|---|---|---|
| Even / Random dimension split (paper single-model baselines) | `src/tokenization/dimension_split/generate_dimension_split_sid_v3.py`, `append_collision_token.py` | shipped (paper-owned code) |
| RQ-VAE (paper: RQ-VAE5, `L=6`) | external: <https://github.com/kakaobrain/rq-vae-transformer> | **not bundled**; recipe in `configs/paper/RQ5_RECIPE_FINAL.json` |
| RQ-OPQ (paper: RQ2-OPQ3, `L=6`) | external components: FAISS OPQ (<https://github.com/facebookresearch/faiss>) + an RQ implementation; construction reference: OneSearch (<https://arxiv.org/abs/2212.01354>) | **not bundled** |

All factorizations must produce the same SID array format (§1) so that
`train.sh` / `generate_candidates.sh` / `run_cfla.sh` accept them identically.

## 3. T5 backbone and training (paper §4.3)

| element | code | notes |
|---|---|---|
| T5 encoder–decoder 4+4 layers, `d=128`, `ffn=1024`, 6 heads, dropout 0.1 | `src/tiger/train.py` (`TIGER`, `T5Config`) | shared by training, beam search and teacher forcing |
| SID tokenizer / dataset | `src/tiger/dataset.py` (`item2code`, `process_data`, `GenRecDataset`) | histories = flattened item SID tokens, truncated/padded to `max_len` items |
| training loop, checkpoint selection | `src/tiger/train.py` (`train`, `evaluate_item_level`, `main`) | batch 256, lr 1e-4, up to 200 epochs, patience 10, seed 2025 |
| item-level validation metric (NDCG@20) | `src/tiger/train.py` (`evaluate_item_level`, `calculate_pos_index`, `ndcg_at_k`) | `--eval_mode item` is the default and the paper's metric |
| CLI | `python3 -m src.tiger.train` | `--dataset_path --code_path --vocab_size --codebook_size --device ... --num_workers` |

## 4. Candidate generation (paper §3.3)

| element | code | notes |
|---|---|---|
| beam search over SID sequences, decode to item ids | `src/tiger/dump_beams.py` (`main`, `build_sid_to_item`, `decode_tokens_to_item`) | beam size 30 in the paper; 0-based ids with `-1` sentinel |
| dump format | see `docs/DATA_FORMAT.md` | `candidate_item_ids`, `candidate_tokens`, `sequence_scores` (length-normalised log-prob), `transition_logprobs`, `target_item_ids` |
| token-layout / decode guard | `src/tiger/dump_beams.py` | raises if fewer than 90% of candidates decode to a valid item (slot-width mismatch) |

## 5. Teacher-forced candidate NLL (paper §3.3, Eq. 4)

| element | code | notes |
|---|---|---|
| teacher-force a complete SID under a factorization | `src/analysis/teacher_forced.py` (`teacher_forced_logprobs`, `tokenise_sid`, `load_model_from_ckpt`) | decoder input `BOS + SID[:-1]`; sum of `−log p` over all `L_F` positions; target-free |
| candidate scoring over the union | `src/cfla/cfla.py` (`candidate_nll`) | encodes the history in the factorization's token space; batches rows |

## 6. Calibration and aggregation (paper §3.4, Eqs. 5–6)

| element | code | notes |
|---|---|---|
| `z_F=(NLL_F−μ_F)/σ_F`, `μ_F/σ_F` from **all validation candidates** | `src/cfla/cfla.py` (`fit_scalers`, `zscore`) | fitted on the validation block, frozen for test |
| `S_CFLA=logsumexp_F(−z_F)` | `src/cfla/cfla.py` (`score_cfla`) | no trainable parameters |
| baselines on the same union | `src/cfla/cfla.py` (`score_uniform`, `score_minnll`) | Uniform-Fusion and aggregation-function ablation |
| candidate union (target never inserted) | `src/cfla/cfla.py` (`build_union`, `flat_pairs`) | per-user union with per-factorization retrieval probabilities |
| CLI / orchestration | `src/cfla/run.py` (`python3 -m src.cfla.run`) | loads config, scores valid+test, writes `metrics.json`, `per_user.csv`, `run_config.json` |

## 7. Evaluation (paper §4.1)

| element | code | notes |
|---|---|---|
| item-level Recall@K / NDCG@K, K ∈ {5,10,20} | `src/cfla/cfla.py` (`per_user_metrics`, `summarize`); `src/fusion/common.py` (`per_user_ndcg`, `summarize`) | targets outside the union contribute 0 |
| paired user-level bootstrap (5,000 resamples) | `src/cfla/cfla.py` (`paired_bootstrap`) | 95% CI, two-sided p-value; reported for CFLA vs Uniform |
| single-dump evaluation CLI | `scripts/evaluate.py` | `--dump --out` |

## 8. Configuration and runner

| element | code | notes |
|---|---|---|
| canonical constants | `configs/canonical.yaml` | SID/model/training/inference values |
| per-experiment configs | `configs/experiments/*.yaml` | 4 datasets × K=2..5, seed 2025 |
| CFLA pool config | `configs/cfla_pool.example.yaml` | per-factorization SID / checkpoint / beam-dump paths |
| RQ-VAE5 recipe | `configs/paper/RQ5_RECIPE_FINAL.json` | hyper-parameters only; tokenizer external |
| unified per-config runner | `src/runner.py` | SID → train → beam → evaluate stages for the spectral configs |
| path resolution | `src/utils/repo_paths.py` | `CFLA_ARTIFACT_ROOT`, `CFLA_OUTPUT_ROOT`, `CFLA_UPSTREAM_ROOT` |
