# ENVIRONMENT

## Verified environment

The release was checked against the versions below (import checks, shell/YAML
validation and a full synthetic CPU smoke pipeline). They are pinned in
`requirements.txt`.

| component | version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.12.0 (verified CUDA 13.0 build: `2.12.0+cu130`) |
| Transformers | 5.9.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| pyarrow | 24.0.0 |
| PyYAML | 6.0.2 |
| tqdm | 4.67.3 |
| sentence-transformers | 5.5.1 (optional: item-embedding preparation) |

```bash
python3 -V                        # 3.12.x
python3 -m pip install -r requirements.txt
```

`torch` wheels are driver/CUDA specific: the pinned `torch==2.12.0` resolves to the
CPU wheel by default — install the CUDA build matching your driver from
<https://pytorch.org> for GPU training/inference. The smoke test and the synthetic
end-to-end pipeline run on CPU.

## RQ-VAE / RQ-OPQ tokenizers (not bundled)

The RQ-VAE and RQ-OPQ implementations are **not shipped**; they are external
repositories (see README §5.2–5.3). Their dependencies (e.g. FAISS for OPQ) are
installed separately according to those repositories' instructions; they are not
part of `requirements.txt` for this release.

## Path configuration

No absolute path is hard-coded; every default is relative to the repository:

| variable | meaning | default |
|---|---|---|
| `CFLA_ARTIFACT_ROOT` | input artifacts you prepare (splits, mappings, item embeddings, SID arrays, checkpoints, beam dumps); read-only for the code | `<repo>/artifacts` |
| `CFLA_OUTPUT_ROOT` | all new artifacts written by a run | `<repo>/outputs` |
| `CFLA_UPSTREAM_ROOT` | fallback root consulted when a file is missing from `CFLA_ARTIFACT_ROOT` | `$CFLA_ARTIFACT_ROOT` |
| `CFLA_RAW_DATA` | raw Amazon Reviews 2014 files | `<repo>/raw` |
| `CFLA_SENTENCE_T5_DIR` | local `sentence-t5-base` checkpoint (embedding preparation only) | `<repo>/sentence-t5-base` |
| `CFLA_PYTHON` | interpreter used by the shell wrappers | `python3` |

Expected input layout: `docs/DATA_FORMAT.md`. CFLA pool config:
`configs/cfla_pool.example.yaml`.

## CPU / restricted hosts

The T5 training and beam-dump entry points accept `--num_workers 0` (default 4) for
hosts where multiprocessing workers are unavailable; `scripts/smoke_pipeline.sh`
uses it. All commands also run on CPU for small/synthetic data.

## Sanity checks

```bash
bash scripts/check_environment.sh   # versions + resolved path roots
bash scripts/smoke_test.sh          # compile + import smoke test
bash scripts/smoke_pipeline.sh      # full synthetic end-to-end pipeline (CPU)
```
