# CODE_MAPPING — paper ↔ code mapping

This document maps every component of the CFLA paper (Cross-Factorization Likelihood
Aggregation) to the code that implements it **in this release**, with file:line evidence
and the entry points that produce each reported number. It is written for two readers:
a reviewer who wants to check that the method described in the paper is the method in the
code, and a user who wants to re-run the pipeline.

> **Reference caveat.** Function and file names are the stable identifiers;
> the `:line` suffixes are approximate and may drift by a few lines after edits
> (e.g. after import/path headers are adjusted). Module paths were checked
> against the release tree.

Evidence policy: every mapping below was established by reading the implementation and its
call sites (imports/callers), **not** from filenames. Where a link was not executed we say
so. No training run or full experiment was reproduced for this document.

Status legend:

| status | meaning |
|---|---|
| **verified (code)** | implementation and its call sites both read; function bodies checked |
| **verified (artifact)** | a persisted artifact's shape/dtype/keys were inspected to confirm the claim |
| **verified (execution)** | the command was executed in this release environment (stated explicitly) |
| **static inference** | inferred from frozen records, not from running the code |
| **NOT LOCATED** | no code exists; only hand-made artwork or frozen records survive |

---

## 0. Two implementation generations in one release

The release contains the code of the final paper pipeline **and**, as supporting material,
the code of an earlier, smaller two-tokenizer study that shares the same primitives. A
reviewer must not confuse the two:

| | paper-final pipeline (reproduces all paper numbers) | earlier supporting study |
|---|---|---|
| location | `src/tokenization/paper/`, `src/training/`, `src/cfla/paper_pipeline/`, `src/cfla/paper_results/` | `src/cfla/multifactor_fusion/`, `src/cfla/{candidate_conditioned,final_ablation,sports_confirmation,adaptive_factorization}/` |
| factorization pool | Spectral-K2/K3/K4/K5 + **RQ-VAE5** (full SID lengths 3/4/5/6/6) | Spectral-K2/K3/K4/K5 + **RQ-VAE3** (lengths 3/4/5/6/4) |
| single-model baselines | length-matched at 5 semantic + 1 disambiguation token | not length-matched |
| what produced | **Table 1, Table 2, Table 3, Fig. 4, Fig. 5** of the manuscript | frozen-number verification, extra ablations, mechanism analyses, Sports confirmation |

The naming trap: the earlier study calls its RQ-VAE arm `rqvae3` (3 residual levels,
4-token SID), while the paper's RQ-VAE arm is `rqvae5` (5 residual levels, 6-token SID).
Both are legitimate code; only `rqvae5` is the paper's main pool. See §6.

---

## 1. Paper component → code

Paths are relative to the release root. `[…:N]` is a line number in the released file.

### 1.1 Spectral SID construction (paper §3.2, Eqs. 1–3)

| paper element | release code | evidence |
|---|---|---|
| item embedding matrix `X ∈ R^{N×d}` | `src/sid/embedding.py::load_item_embedding` [:38], `align_embeddings` [:92] | reads `data/<Dataset>/item_emb.parquet`; two row-alignment modes (`canonical`, `item_id`); the canonical mode treats row *i* as item *i+1* |
| dimension–dimension Pearson affinity `A_pq = (ρ_pq+1)/2`, `A_pp = 1` | `src/sid/affinity.py::build_dimension_affinity` [:45]; standardise [:69], `np.corrcoef` [:72], `(S+1)/2` [:74], unit diagonal [:86] | canonical metric name `nonneg_pearson` [:29] |
| normalised graph Laplacian, `K` smallest eigenvectors, k-means on dimension rows → `K` dimension groups | `src/sid/spectral_partition.py::spectral_group_dimensions` [:26]; `SpectralClustering(affinity='precomputed', assign_labels='kmeans', random_state=seed, n_init=20)` [:47–53]; partition-validity checks [:56–67] | scikit-learn's `SpectralClustering` constructs `L = I − D^{-1/2} A D^{-1/2}` and embeds with the `K` smallest eigenvectors; the precomputed affinity is our `A` |
| subspace quantisation `c_r(i) = argmin_c ||z_{i,r} − μ_{r,c}||²`, codebook `C=256` | `src/sid/quantizer.py::generate_base_sid_by_group_kmeans` [:38]; per-group standardisation [:47], optional PCA-whitening [:48] via `reduce_if_needed` [:28], `KMeans(n_clusters=C, n_init=30, max_iter=500, random_state=seed)` [:51–57] | |
| base SID = concatenation of the `K` codes | return value of `generate_base_sid_by_group_kmeans` — shape `(N,K)` [:59] | |
| local disambiguation token; injective final SID | `src/sid/collision.py::add_collision_token` [:55]; rank inside collision group [:74–78]; injectivity assertion [:82–83]; `collision_stats` [:36] reports the **base-prefix** collision rate (the meaningful statistic) | final SID length `L = K+1`; generator asserts shape and that the semantic prefix is untouched (`src/sid/generator.py` [:159–163]) |
| end-to-end SID CLI | `python -m src.sid.generator --dataset <Dataset> --affinity nonneg_pearson --k <K> --codebook_size 256 --seed 2025` (`src/sid/generator.py` parser [:65–89], pipeline [:110–204]); writes SID/base-SID/dim-groups/dim-labels + `metadata.json` + `manifest.sha256` (`src/sid/io.py::write_sid_artifacts` [:56]) | refuses to write into the read-only input root (`src/sid/io.py::assert_not_artifact_root` [:30]) |
| reproduction guarantee | `docs/SID_REPRODUCTION_REPORT.md` (bit-identical criterion A) and `tests/test_sid_construction.py` | the clean implementation reproduces the authors' frozen canonical SID arrays bit-for-bit on the Beauty dataset (`np.array_equal` on SID, base SID and dim labels) — this is the provenance evidence that the release code *is* the code that generated the paper's spectral SIDs |
| experiment configs for `K ∈ {2,3,4,5}` | `configs/experiments/<dataset>_spectral_nonneg_k<K>_seed2025.yaml` | one YAML per dataset × K |

### 1.2 The five SID factorizations (paper §3.2, §4.4)

| factorization | release code that builds it | SID shape | status |
|---|---|---|---|
| **Spectral-K2/K3/K4/K5** | `src/sid/generator.py` (above) + `configs/experiments/*spectral_nonneg_k*.yaml` | K semantic + 1 disambiguation = **3/4/5/6** | verified (code) + bit-exact reproduction (docs/SID_REPRODUCTION_REPORT.md) |
| **RQ-VAE5** (paper's RQ-VAE arm) | `src/tokenization/paper/train_tokenizer_5plus1.py` [:134–162]: trains a 5-level RQ-VAE with `--num_emb_list` repeated 5× [:139] using the tokenizer training entry point `src/tokenization/legacy_rqvae/main.py`, then encodes items with a generated copy of `src/tokenization/legacy_rqvae/generate_code.py` [:155–159] | 5 RQ codes + 1 appended collision code = **6** | verified (code + artifact: all four paper SID arrays are `(N,6)` with codes in `[0,255]`; recipe frozen in `configs/paper/RQ5_RECIPE_FINAL.json`) |
| **RQ2-OPQ3** (paper's "RQ-OPQ") | `src/tokenization/paper/train_rq2opq3_5plus1.py` + the generator `src/tokenization/paper/generate_rq2_opq3_sid.py` (`train_rq_kmeans_2level` [:202], OPQ/PQ encoding via faiss index factory `OPQ3,IVF1,PQ3x8` [:245+], collision code [:333]) | 2 RQ + 3 OPQ/PQ semantic + 1 disambiguation = **6** | verified (code + artifact: SID arrays `(N,6)`, raw semantic arrays `(N,5)`); the driver documents its exact lineage at `train_rq2opq3_5plus1.py` [:6–9, :38] |
| **Even Split** (K=5 reported) | `src/tokenization/legacy_rqvae/generate_dimension_split_sid_v3.py::make_dimension_groups` [:35–55] with `split_method="even"` (`np.array_split` of contiguous dimension ids); subspace k-means [:73–114]; collision token [:125–168] | 5 semantic + 1 = **6** | verified (code) |
| **Random Split** (K=5 reported) | same function with `split_method="random"`: `np.random.default_rng(2025).shuffle(dims)` then `array_split` [:38–41] | 5 semantic + 1 = **6** | verified (code) |
| **RQ-VAE3** (earlier study only — *not* the paper pool) | `src/tokenization/legacy_rqvae/generate_code.py` with the earlier checkpoint's `args.num_emb_list == [256,256,256]` | 3 RQ + 1 collision = **4** | verified (artifact: checkpoint args + `(N,4)` SID arrays) |

The tokenizer entry points for all three RQ-family arms are driven by
`src/tokenization/paper/train_tokenizer_5plus1.py` and
`src/tokenization/paper/train_rq2opq3_5plus1.py`; the RQ-VAE tokenizer trainer itself is
`src/tokenization/legacy_rqvae/{main.py,models/*.py,trainer.py,datasets.py,utils.py}`.

### 1.3 T5 backbone and training (paper §4.3)

| paper element | release code | evidence |
|---|---|---|
| T5 encoder–decoder, 4 enc + 4 dec layers, d=128, FFN=1024, 6 heads, dropout 0.1 | `src/tiger/train.py::TIGER` [:20] building `T5Config` [:26–38] | matches `configs/canonical.yaml` model block |
| batch 256, lr 1e-4, max history 20, seed 2025, beam 30, `eval_mode=item` | `src/tiger/train.py` argument defaults [:306–340] | |
| checkpoint selection by **item-level validation NDCG@20** | `src/tiger/train.py:444–452` (compare `avg_ndcgs['NDCG@20']`, then save); item-level evaluator `evaluate_item_level` [:160] | the legacy token-prefix evaluator is kept only as an explicitly labelled non-default (`--eval_mode token`) |
| SID tokenizer layout `token = code + position*256 + 1`, PAD/EOS `0`, vocab `L*256+1` | `src/tiger/dataset.py::item2code` [:106], `infer_codebook_size_from_vocab` [:87] | |
| training of the 24 length-matched single models | `src/training/train_t5_5plus1.py` (T5 training driver; dumps VALID and TEST beams through `src/tiger/dump_beams.py`) | one model per (dataset, factorization); 4 datasets × 6 arms = 24 |
| shared-parameter T5 (used by the mechanism analysis of Fig. 4b) | `src/cfla/cross_tokenizer_shared/xt_train_shared.py`, `xt_shared_analysis.py` | supporting analysis code; needs the authors' frozen shared checkpoint or a retrain |

### 1.4 Candidate generation and shared union (paper §3.3)

| paper element | release code | evidence |
|---|---|---|
| beam search (beam size 30) per factorization, candidates decoded back to item ids | `src/tiger/dump_beams.py::main` [:118]; `build_sid_to_item` [:65]; `tokens_to_sid`; `decode_tokens_to_item` | dump keys: `candidate_tokens`, `candidate_item_ids`, `target_item_ids`, `sequence_scores` |
| candidate sets are target-free; the ground truth is never inserted | union construction: `src/cfla/paper_pipeline/build_m5_rq5.py::build_scope_union` [:790]; NLL producer `src/cfla/paper_pipeline/score_m5_candidate_nll.py::union_rows` [:267]; general helper `src/cfla/multifactor_fusion/mf_common.py::build_union` [:368] with `merge_by_item` [:340] | candidates are read only from the dump's `candidate_item_ids`; `no target-access guard is shipped` includes a perturb-the-target invariance guard |
| item-id convention (historical dumps store 0-based SID-array rows; canonical item id = raw+1) | documented in `docs/ITEM_ID_CONVENTION.md`; enforced by `mf_common.py::detect_item_shift` [:273] and the paper-pipeline convention audit (`score_m5_candidate_nll.py::convention_audit` [:391]) | |

### 1.5 Teacher-forced candidate NLL (paper §3.3, Eq. 4)

| paper element | release code | evidence |
|---|---|---|
| teacher-force the **complete** SID `c_1…c_L` of candidate `i` under factorization `F`; sum `−log p` over the `L_F` positions | `src/analysis/teacher_forced.py::teacher_forced_logprobs` [:61]; decoder input `BOS + target[:-1]` [:83]; log-softmax + gather; slot tokens `code + pos*256 + 1` (`tokenise_sid` [:96]) | |
| batched total NLL for a candidate pool | `src/cfla/multifactor_fusion/mf_models.py::candidate_total_nll` [:183], `build_history_tokens` [:247], `encoder_input_for` [:168] | one T5 checkpoint per factorization |
| paper-pipeline producer of the candidate-NLL tables | `src/cfla/paper_pipeline/score_m5_candidate_nll.py`: `score_arm` [:555], `write_back_and_verify_calibrated` [:448], `fit_scalers` [:885] | writes per-(dataset, split, arm) candidate-NLL parquet + frozen scaler file |
| target-access guards | `no target-access guard is shipped` | scorer signature/body scan + perturb-target invariance |

### 1.6 Calibration, LSE aggregation, fusion baselines, metrics (paper §3.4, Eqs. 5–6)

| paper element | release code | evidence |
|---|---|---|
| `z_F(u,i) = (NLL_F − μ_F)/σ_F`, with `μ_F,σ_F` estimated on **all validation candidates** and frozen for test | `src/cfla/multifactor_fusion/mf_core.py::fit_scalers` [:224], `z_of` [:236], `znll_matrix` [:90], `apply_scalers` [:218]; paper pipeline: `src/cfla/paper_pipeline/score_m5_candidate_nll.py::fit_scalers` [:885] → `write_scalers` [:926], consumed by `build_m5_rq5.py` | scalers are fitted on the VALID split only |
| `S_CFLA(u,i) = logsumexp_F(−z_F(u,i))`; no trainable parameters | `src/cfla/multifactor_fusion/mf_core.py::score_lse_from_znll` [:111], `per_user_logsumexp` [:70]; paper pipeline `src/cfla/paper_pipeline/build_m5_rq5.py::cfla_scores_from_nll` [:961] | |
| **Uniform Fusion** baseline on the same union | `src/cfla/paper_pipeline/build_m5_rq5.py::evaluate_strategies` [:854] with `one_hot_weights`/`weights_for`; general `src/fusion/uniform.py::uniform_weights` [:108] | uniform = equal-weight mean of normalized generation scores |
| ranking + Recall@K / NDCG@K (K∈{5,10,20}) | `src/cfla/multifactor_fusion/mf_core.py::rank_and_metrics` [:123], `target_rank_from_scores` [:173]; canonical item-level evaluator `src/fusion/common.py::per_user_ndcg` [:204], `evaluate_uniform` [:258] | one evaluator shared by all arms |
| paired user-level bootstrap, 5000 resamples | `src/cfla/paper_pipeline/build_m5_rq5.py::bootstrap_pairs` [:997]; `src/fusion/uniform.py::paired_bootstrap` [:138] | |
| per-user diagnostics | `src/analysis/per_user.py` (`per_user_items`, `paired_bootstrap` [:197]) | supporting |

### 1.7 Exact chain that produces paper Table 1

Everything below is a released file; the right-hand column names the persisted link.
`<AR>` = the artifact root the user provides (see `src/utils/repo_paths.py`); `OUT` = the
run output root.

```
(1) SID construction
    Spectral K=2..5   : src/sid/generator.py  (config: configs/experiments/*.yaml)
    RQ-VAE5           : src/tokenization/paper/train_tokenizer_5plus1.py:134-162
    RQ2-OPQ3          : src/tokenization/paper/train_rq2opq3_5plus1.py
                        -> src/tokenization/paper/generate_rq2_opq3_sid.py
    Even / Random K=5 : src/tokenization/legacy_rqvae/generate_dimension_split_sid_v3.py
        |
        v   persisted:  <AR>/data/<Dataset>/<Dataset>_t5_<arm>.npy          (SID arrays, L per arm)
(2) T5 training, one model per (dataset, factorization)
    src/training/train_t5_5plus1.py  ->  src/tiger/train.py   (T5 + item-level selection)
        |
        v   persisted:  <OUT>/runs/<Dataset>/<arm>/t5/best_model.pth
(3) Beam-search candidate generation (beam 30)
    src/tiger/dump_beams.py   (invoked by the training driver)
        |
        v   persisted:  <AR>/model/beam_dumps_<Dataset>_<affinity>/<split>_k<K>_sid<L>.npz
                        (and the RQ-VAE5 arm's dump)
(4) Single-model baselines  (length-matched: 5 semantic + 1 disambiguation)
    src/cfla/paper_pipeline/build_single_5plus1.py:78
        |
        v   persisted:  <OUT>/single_5plus1/{TEST,VALID}_RESULTS.csv and *_FINAL.csv
(5) CFLA pipeline on the paper pool M5 = Spectral-K2+K3+K4+K5 + RQ-VAE5
    (5a) candidate NLL of every candidate under every arm + VALID-fitted zNLL scalers
         src/cfla/paper_pipeline/score_m5_candidate_nll.py
             score_arm :555, fit_scalers :885
    (5b) union -> Uniform / CFLA(LSE) / bootstrap
         src/cfla/paper_pipeline/build_m5_rq5.py
             evaluate_strategies :854, cfla_scores_from_nll :961, bootstrap_pairs :997
    (5c) fill measured CFLA cells
         src/cfla/paper_pipeline/build_m5_cfla.py::measured_cfla :244
        |
        v   persisted:  <OUT>/m5_rq5/{VALID,TEST,BOOTSTRAP}.csv
(6) Table assembly
    src/cfla/paper_results/build_table1_final.py
        sources :48-50 (single_5plus1 TEST_RESULTS + TEST_RESULTS_FINAL, m5_rq5/TEST.csv)
        main    :167
        |
        v   writes  paper/includes/table1_main_results.tex
            (in this release: docs/table1_main_results.tex is not shipped;
             the builder writes it into the output root — see README)
```

**Which links are CFLA-specific.** (1)–(4) are shared with any single-SID system. CFLA is
only (5): scoring *every* candidate under *every* factorization (not only the arm that
generated it), the validation-fitted per-factorization zNLL, and the log-sum-exp
aggregation; Uniform Fusion is the same union with equal weights.

**Other paper results chains.**

* **Table 2 (efficiency/latency)** — measurement driver
  `src/cfla/paper_pipeline/measure_inference_complexity.py::measure_single` [:729] /
  `measure_pool` [:808] (per-user latency with CUDA synchronisation, 5 warm-up + 10
  measured runs, 96 users, beam 30) → measured CSV; table builder
  `src/cfla/paper_results/make_table_efficiency.py` (input `m5_rq5/TEST.csv` :32 and the
  measured complexity CSV).
* **Table 3 / Fig. 5 (codebook sensitivity, Beauty)** — study driver
  `src/cfla/paper_pipeline/build_codebook_sensitivity.py::stage_cfla_gpu` [:864],
  union helper `build_union_generic` [:594] → frozen CSV; figure/table builder
  `src/cfla/paper_results/make_fig_codebook.py`. Codebook sizes are configured by the
  study driver (`C ∈ {32,64,128,256,512,1024}`); each size needs its own Spectral
  tokenizer + T5 checkpoints (see §5).
* **Fig. 4 (mechanism)** — analysis code
  `src/cfla/factorization_generality/` (complementarity: `fg_pairs.py`, `fg_triples.py`,
  `fg_alignment_stats.py`) and `src/cfla/cross_tokenizer_shared/` (alignment under
  independent vs shared parameters: `xt_train_shared.py`, `xt_shared_analysis.py`,
  `xt_independent_vs_shared.py`). The figure itself is hand-assembled from the analysis
  CSVs with the plotting code in `src/cfla/paper_results/ (figure builders)` / a dedicated figure script;
  the numerical claims are in `pairwise_diversity_results.csv` and
  `independent_vs_shared_factorization.csv` (frozen inputs, not shipped).

### 1.8 Paper section → code, at a glance

| paper element | code | runnable without authors' artifacts? |
|---|---|---|
| §3.2 spectral SID construction | `src/sid/*` + `configs/experiments/*` | **yes** (needs only an item-embedding parquet) |
| §3.2 RQ-VAE / RQ-OPQ / even / random tokenizers | `src/tokenization/paper/*`, `src/tokenization/legacy_rqvae/*` | needs the dataset + (for RQ) tokenizer training; runnable after data preparation |
| §3.3 candidate generation | `src/tiger/dump_beams.py` | needs trained checkpoints |
| §3.3 teacher-forced NLL | `src/analysis/teacher_forced.py`, `src/cfla/multifactor_fusion/mf_models.py` | needs checkpoints |
| §3.4 zNLL calibration + LSE (Eqs. 5–6) | `src/cfla/multifactor_fusion/mf_core.py`, `src/cfla/paper_pipeline/build_m5_rq5.py` | needs frozen NLL tables **or** a full re-run |
| §4 Table 1 | `src/cfla/paper_results/build_table1_final.py` | needs frozen result CSVs (not shipped) |
| §4 Table 2 efficiency | `src/cfla/paper_pipeline/measure_inference_complexity.py` + `src/cfla/paper_results/make_table_efficiency.py` | measurement needs checkpoints+beams |
| §4 Table 3 / Fig. 5 codebook | `src/cfla/paper_pipeline/build_codebook_sensitivity.py` + `src/cfla/paper_results/make_fig_codebook.py` | needs per-codebook checkpoints |
| §4 Fig. 4 mechanism | `src/cfla/factorization_generality/*`, `src/cfla/cross_tokenizer_shared/*` | needs frozen analysis CSVs or a full re-run |
| §4 ablations | see §3 | frozen artifacts or a full re-run |

---

## 2. Data and configuration interface

* Inputs: `CFLA_ARTIFACT_ROOT` (preprocessed splits, item embeddings/mappings, SID arrays,
  checkpoints, beam dumps) — read-only from the code's point of view.
* Outputs: `CFLA_OUTPUT_ROOT` (every new artifact a run writes).
* Defaults: `<repo>/artifacts` and `<repo>/outputs`; the release ships no data.
* Layout expected under the artifact root is documented in `src/utils/repo_paths.py`
  (module docstring) and `docs/DATA_FORMAT.md` / `docs/SID_FORMAT.md`.
* Paper hyper-parameters: `configs/canonical.yaml` (model/training/generation defaults),
  `configs/canonical_full.yaml` (full reference configuration), and one
  `configs/experiments/<dataset>_spectral_nonneg_k<K>_seed2025.yaml` per dataset × K.
  The frozen RQ-VAE5 recipe is `configs/paper/RQ5_RECIPE_FINAL.json`.

### 2.1 Entry points (all run from the release root; all import-checked in this release)

| step | command |
|---|---|
| spectral SID build | `python -m src.sid.generator --dataset <Dataset> --affinity nonneg_pearson --k <K> --codebook_size 256 --seed 2025` |
| RQ-VAE5 tokenizer + SID | `python -m src.tokenization.paper.train_tokenizer_5plus1 --dataset <Dataset> --arm rqvae5` |
| RQ2-OPQ3 tokenizer + SID | `python -m src.tokenization.paper.train_rq2opq3_5plus1 --dataset <Dataset>` |
| even/random split SID | `python -m src.tokenization.legacy_rqvae.generate_dimension_split_sid_v3 --item_emb_path <parquet> --output_path <npy> --num_sid_tokens 5 --split_method {even,random} --codebook_size 256 --pca_dim 64 --random_state 2025` |
| T5 training for the length-matched single models | `python -m src.training.train_t5_5plus1 --dataset <Dataset> --arm <arm>` |
| candidate-NLL + zNLL scalers (paper pool) | `python -m src.cfla.paper_pipeline.score_m5_candidate_nll --dataset <Dataset>` |
| M5 union + Uniform/CFLA + bootstrap | `python -m src.cfla.paper_pipeline.build_m5_rq5 --dataset <Dataset>` |
| fill measured CFLA cells | `python -m src.cfla.paper_pipeline.build_m5_cfla --dataset <Dataset>` |
| single-model baseline tables | `python -m src.cfla.paper_pipeline.build_single_5plus1 --dataset <Dataset>` |
| efficiency measurement | `python -m src.cfla.paper_pipeline.measure_inference_complexity --dataset <Dataset>` |
| codebook sensitivity (Beauty) | `python -m src.cfla.paper_pipeline.build_codebook_sensitivity --dataset Beauty` |
| Table 1 LaTeX | `python -m src.cfla.paper_results.build_table1_final` |
| Table 2 (efficiency) LaTeX | `python -m src.cfla.paper_results.make_table_efficiency` |
| Table 3 / Fig. 5 (codebook) | `python -m src.cfla.paper_results.make_fig_codebook` |

Exact flags/argument names are listed by each module's `--help`; the table above fixes the
module path so a reviewer can locate the implementation immediately.

---

## 3. Ablations and robustness studies — honest split

The **current manuscript** reports: Table 1 (main results), Fig. 4 (mechanism), Table 3 /
Fig. 5 (codebook robustness on Beauty), Table 2 (efficiency). It has **no separate
ablation section** (the appendix is empty). The earlier draft of the paper additionally had
a two-tokenizer study with ablations A/B/C/D/E/F/H and a Sports-and-Outdoors confirmation
domain; that material is **not** part of the current manuscript but the code is shipped as
*supporting* material because it is honest, useful evidence about the method.

| study | release path + entry point | runnability |
|---|---|---|
| Uniform vs zNLL vs LSE vs min-NLL aggregation (cross-factorization rescoring) | `src/cfla/candidate_conditioned/cc_rerank_methods.py` (zNLL / LSE / min-NLL rankers), `cc_candidate_nll.py`, `cc_eval_test.py` | needs the authors' frozen candidate-NLL artifacts; otherwise full re-run |
| learned gate / selector ablation (rejected variant) | `src/cfla/multifactor_fusion/mf_gate.py`, `mf_stage.py::stage_gate`; `src/cfla/candidate_conditioned/dg_*.py`; `src/cfla/adaptive_factorization/*` | needs frozen features/checkpoints; **explicitly a rejected alternative — not the method** |
| single-SID vs multi-SID comparison | `src/cfla/final_ablation/fa_cpu_ablations.py`, `fa_family.py`, `fa_independent.py`, `fa_ablation_d.py`, `fa_ablation_e.py` | needs the authors' frozen beams/NLL tables (or a full re-run) |
| shared vs independent model parameters | `src/cfla/cross_tokenizer_shared/xt_train_shared.py`, `xt_shared_analysis.py`, `xt_independent_vs_shared.py`; also `src/cfla/final_ablation/fa_independent.py` | needs the frozen shared checkpoint (or a retrain) |
| same-family vs cross-family factorization | `src/cfla/factorization_generality/fg_pairs.py`, `fg_triples.py`; `src/cfla/final_ablation/fa_family.py` | needs frozen beam dumps |
| aggregation-function ablation (uniform / zNLL / LSE / min-NLL / gate) | `src/cfla/candidate_conditioned/cc_rerank_methods.py`, `cc_preference_stability.py`; `src/cfla/final_ablation/fa_cpu_ablations.py` | frozen artifacts; `no target-access guard is shipped` guards target-free-ness |
| codebook sensitivity | `src/cfla/paper_pipeline/build_codebook_sensitivity.py` (+ `src/tokenization/legacy_rqvae/generate_spectral_opq_balanced_sid.py` only for the OPQ variant) | **paper-current**; needs per-codebook tokenizers + T5 checkpoints |
| beam budget | `src/tiger/dump_beams.py` (`--beam_size`), `src/cfla/candidate_conditioned/cc_efficiency.py`, `src/cfla/multifactor_fusion/mf_stage.py::stage_efficiency` | a new budget requires regenerating candidate dumps (needs checkpoints); frozen-budget analysis needs the authors' dumps |
| efficiency / latency | `src/cfla/paper_pipeline/measure_inference_complexity.py` (paper Table 2); later fusion-only study `src/cfla/paper_pipeline` measurement scripts | **paper-current**; needs checkpoints+beams, or frozen measurement CSVs |
| Sports-and-Outdoors confirmation | `src/cfla/sports_confirmation/{sc_pipeline,sc_gate_eval,sc_generate_sid,sc_train_shared}.py` | **removed from the current manuscript**; supporting only. Needs its own SID construction + checkpoints |
| adaptive factorization / learned selector (rejected) | `src/cfla/adaptive_factorization/af_*.py` | **rejected alternative**; not the method |
| frozen-number verification / provenance | `src/cfla/final_ablation/fa_verify.py`, `fa_provenance.py`, `src/cfla/multifactor_fusion/mf_*` (reuse audit, immutability) | static/checksum checks; needs the authors' frozen artifacts |

Runnability vocabulary used above: **runs on synthetic/small data** (unit tests and the SID
builder), **needs the authors' frozen artifacts** (result CSVs, beam dumps, NLL tables — not
shipped), **needs retraining** (tokenizer and/or T5 checkpoints must be regenerated).

---

## 4. What is required but deliberately not shipped

| artifact | needed for | how a user obtains it |
|---|---|---|
| Amazon Reviews 2014 splits (`train/valid/test.parquet`), `item_mapping.npy` | everything | the user downloads the public dataset; `src/data_prep/process_interactions.py` rebuilds the splits |
| item semantic embeddings `data/<Dataset>/item_emb.parquet` (768-d) | SID construction, RQ-VAE | `src/data_prep/make_item_embeddings.py` (sentence-T5 over item metadata) |
| SID arrays (`*_sid*.npy`, `*_t5_<arm>.npy`) | T5 training | regenerate with `src/sid/generator.py` / `src/tokenization/*` |
| tokenizer checkpoints (RQ-VAE `best_collision_model.pth`) | RQ SID encoding | retrain with `src/tokenization/legacy_rqvae/main.py` |
| T5 checkpoints (24 single models + 5 pool arms + codebook variants + shared) | candidate generation, NLL | retrain with `src/training/train_t5_5plus1.py` / `src/tiger/train.py` |
| beam dumps (`*.npz`) | union + candidate NLL | regenerate with `src/tiger/dump_beams.py` |
| candidate-NLL tables + frozen zNLL scalers | Table 1 CFLA columns | regenerate with `src/cfla/paper_pipeline/score_m5_candidate_nll.py` |
| frozen result CSVs (TEST/BOOTSTRAP) | table/figure builders | produced by the pipeline above |

**Consequence for the README:** no command can be claimed as verified end-to-end on the
paper's data. Commands that require user-provided data or retraining must be labelled as
such. The smoke path (SID construction on a small synthetic embedding matrix, unit tests)
*runs*.

---

## 5. Implementation provenance and the naming traps

**Spectral SIDs.** The clean release implementation `src/sid/*` was extracted from the
authors' original spectral generator and reproduces the frozen canonical SID arrays
bit-for-bit on Beauty (`docs/SID_REPRODUCTION_REPORT.md`, criterion A:
`np.array_equal` on SID, base SID and dimension labels; guarded by
`tests/test_sid_construction.py`). The canonical construction is:
`A_pq = (ρ_pq+1)/2` on standardised embeddings → spectral clustering of the `d` dimensions
(affinity = `A`, `k`-means label assignment) → independent k-means (C=256, PCA-64 whitening)
inside each subspace → concatenate → append local disambiguation rank → SID length `K+1`.
Parameters: `nonneg_pearson` affinity, seed 2025, `K ∈ {2,3,4,5}`.

**RQ-VAE5 vs RQ-VAE3.** The paper's RQ-VAE arm uses **5 residual levels** (`num_layers=5`,
`semantic_codebooks=[256,256,256,256,256]`, `sk_epsilons=[0,0,0,0,0.003]`, vocab 1537,
full SID length 6). The frozen recipe is shipped as `configs/paper/RQ5_RECIPE_FINAL.json`.
The earlier supporting study instead used **3 residual levels** (`num_emb_list=[256,256,256]`,
full SID length 4) and its code refers to this arm as `rqvae3`
(`src/cfla/multifactor_fusion/mf_common.py:71`, `:88–92`, `:102–136`). Both are correct
code for different pools; the paper's M5 pool is `rqvae5` (lengths 3/4/5/6/6), not `rqvae3`
(lengths 3/4/5/6/4). This is the single most important naming clarification in the release.

**RQ2-OPQ3.** The paper's "RQ-OPQ" is 2 residual-quantisation positions followed by
3 product-quantisation positions plus 1 disambiguation token (L=6). Its generator is
`src/tokenization/paper/generate_rq2_opq3_sid.py`; the driver documents that it is the
historical RQ-OPQ generator with a small, enumerated set of changes
(`train_rq2opq3_5plus1.py:6–9`) and pins the faiss index factory
`OPQ3,IVF1,PQ3x8` (`:38`). The earlier 3-token RQ-OPQ generator (RQ1-OPQ2 / OPQ3, L=4) is
*not* the paper's arm.

**Even/Random split.** `src/tokenization/legacy_rqvae/generate_dimension_split_sid_v3.py`
`make_dimension_groups` [:35–55]: `even` splits dimension ids into contiguous blocks with
`np.array_split`; `random` permutes dimension ids with `default_rng(2025).shuffle` before
`array_split`. Subspace k-means and collision handling are identical to the spectral
pipeline, so the only difference from Spectral-K is the partition itself — an important
controlled comparison.

**Data input.** `item_emb.parquet` is derived from public Amazon Reviews 2014 metadata:
per-item text from title/brand/categories/price, encoded with sentence-T5
(`src/data_prep/make_item_embeddings.py`). Dataset files are not shipped.

---

## 6. Rejected alternatives that must not be presented as the method

| rejected idea | code | why it must be labelled |
|---|---|---|
| learned multi-factorization candidate gate | `src/cfla/multifactor_fusion/mf_gate.py`, `mf_stage.py::stage_gate` | introduces trainable parameters; the paper's method is parameter-free |
| user-level selector / adaptive factorization | `src/cfla/adaptive_factorization/*` | replaces the LSE aggregation with a learned decision |
| structure-aware / confidence-weighted fusion (DynamicFusion) | `src/cfla/multifactor_fusion/mf_dynamic_fusion.py` | earlier baseline with validation-selected (γ, η); not CFLA |
| disagreement-gate round | `src/cfla/candidate_conditioned/dg_*.py` | rejected variant studied and discarded |

These files are shipped as **supporting/negative-result code only** (see
`needed`); the README must not describe them as the
method.

---

## 7. NOT LOCATED

1. **Schematic Figures 1–3** (motivation, framework, spectral-SID construction): hand-made
   artwork (draw.io templates + exported vector PDFs). No code produces them; they are not
   part of a code release.
2. **The manuscript itself and its revision history**: submitted separately; deliberately
   not shipped.
3. **The authors' frozen result tables/beam dumps/checkpoints**: intentionally excluded
   (size, anonymity, and "code-only" release scope). Every table/figure builder in this
   release therefore requires the user to re-run the pipeline; this is stated in the README.

No algorithmic component is missing: every paper equation and every reported table/figure
has at least one released implementation path listed above.

---

## 8. How to verify a claim from code (reviewer recipe)

1. **Method claim** — start at `src/sid/generator.py` (§1.1), then
   `src/analysis/teacher_forced.py` (§1.5), then
   `src/cfla/multifactor_fusion/mf_core.py::score_lse_from_znll` [:111] (§1.6). These three
   files are the whole of Eqs. 1–6.
2. **"CFLA beats Uniform on the paper's pool"** — the comparison is implemented in
   `src/cfla/paper_pipeline/build_m5_rq5.py::evaluate_strategies` [:854]: the same candidate
   union is ranked once with equal weights (`uniform`) and once with
   `logsumexp(−zNLL)` (`cfla`); no other difference between the two strategies exists in the
   code.
3. **"Calibration is validation-only"** — `src/cfla/paper_pipeline/score_m5_candidate_nll.py::
   fit_scalers` [:885] takes the VALID cells only; the test path loads the frozen scaler
   file (`write_scalers` [:926]) and never refits.
4. **"The target is never used for inference"** — candidate sets come from beam dumps only
   (`build_scope_union` [:790]); the NLL scorer signature has no target argument
   (`mf_models.py::candidate_total_nll` [:183]); `no target-access guard is shipped`
   perturbs the target and checks that no candidate score moves.
5. **"The SID construction is the one used in the paper"** — run
   `python -m pytest tests/test_sid_construction.py` (or `tests/run_tests.py`) against a
   canonical SID array; the test asserts bit-identical reproduction, and
   `docs/SID_REPRODUCTION_REPORT.md` records the criterion.
6. **A specific number in a table** — every table builder in `src/cfla/paper_results/`
   reads a named CSV and performs no fitting; the mapping from table cell to CSV column is
   documented in each builder's module docstring (e.g. `build_table1_final.py:48–50`).

---

## 9. Verification status of this document

| claim area | how verified |
|---|---|
| spectral SID construction | read `src/sid/*.py` in full; module docstrings record the extraction source; bit-exact reproduction documented in `docs/SID_REPRODUCTION_REPORT.md` and guarded by tests |
| T5/tokenizer/beam modules | `src/tiger/{dataset,dump_beams}.py` are line-identical to the authors' historical training scripts (0-line diff); `src/tiger/train.py` differs only by import style and the `--eval_mode` flag; `src/fusion/uniform.py` is line-identical to the authors' historical fusion/evaluation script |
| CFLA primitives | `src/cfla/multifactor_fusion/mf_core.py` read in full (Eqs. 3/5 bodies); `mf_common.py` read in full (registry, beam loading, union) |
| paper Table 1 chain | manuscript table header names its two source CSVs and the builder; builder source lines [:48–50] read; frozen CSV headers inspected; RQ-VAE5/RQ2-OPQ3 SID arrays inspected (`(N,6)`, codes in `[0,255]`) |
| RQ-VAE5 vs RQ-VAE3 | tokenizer checkpoints' `args.num_emb_list` inspected (3 vs 5 levels); SID shapes inspected (`(N,4)` vs `(N,6)`); paper-pool CSV reports lengths 3/4/5/6/6 |
| even/random split | generator read in full; `make_dimension_groups` body inspected |
| efficiency/codebook chains | measurement driver function bodies and the table/figure builders inspected; frozen CSV names match the manuscript table headers |
| ablations | each study's entry module and its persisted artifact names inspected; current-manuscript contents checked directly in the LaTeX sources |
| release import checks | **executed**: all 22 released modules of the paper pipeline, table builders, tokenizers and shared primitives import successfully in this release's environment (`python3 -c "importlib.import_module(...)"`), including `src.cfla.paper_pipeline.{build_m5_rq5,build_m5_cfla,score_m5_candidate_nll,measure_inference_complexity,build_codebook_sensitivity,build_single_5plus1}`, `src.cfla.paper_results.*`, `src.tokenization.paper.*`, `src.training.train_t5_5plus1`, `src.sid.generator`, `src.analysis.teacher_forced`, `src.fusion.*`, `src.tiger.{train,dump_beams}`, `src.cfla.multifactor_fusion.*`; import success is not a claim that the pipeline was executed end-to-end |

No training run, full experiment, or dataset download was performed for this document.

---

## 10. Known open issues (see release validation report for status)

1. `src/cfla/paper_pipeline/*`, `src/cfla/paper_results/*` were authored in the authors'
   experiment workspace and use an output layout (`<run>/m5_rq5/…`) that the release keeps;
   the entry points write under `CFLA_OUTPUT_ROOT`. Users must run the pipeline in the
   documented order before the table builders can read their inputs. The historical fusion
   helper is provided by `src/fusion/common.py` and the historical fusion-strategy module by
   `src/legacy_audited/phase5_fusion.py`; the paper-pipeline drivers resolve both through
   `src/utils/repo_paths.py` (see `src/cfla/paper_pipeline/build_m5_rq5.py` around
   `HIST_FUSION_AUDIT` / `HIST_PHASE5`).
2. Several supporting analysis scripts (ablations, Sports confirmation, mechanism) require
   the authors' frozen artifacts to run; this is stated in §3 and must be repeated in the
   README.
3. Unit tests that compare against real frozen arrays (`tests/test_sid_construction.py`,
   `tests/test_metrics.py`, some guards) need either the user's regenerated artifacts or the
   unit tests synthesize their SID inputs (no fixtures ship) for the spectral reproduction test; the rest run on synthetic
   data.
4. Figures 1–3 are artwork, not code (§7).
