# ENVIRONMENT

## Verified environment

The release was checked against the versions below (import + CPU-only synthetic
tests). They are pinned in `requirements.txt` because those are the versions
that were verified — a nearby version of each package should also work.

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
| matplotlib | 3.9.2 |
| faiss-cpu (optional) | 1.14.2 — only for the RQ-OPQ / RQ2-OPQ3 generator |
| sentence-transformers (optional) | 5.5.1 — only for item-embedding preparation |

Install:

```bash
python3 -V                        # must be 3.12.x
python3 -m pip install -r requirements.txt
```

`torch` wheels are driver/CUDA specific: the pinned `torch==2.12.0` resolves to
the CPU wheel by default — install the CUDA build matching your driver from
<https://pytorch.org> if you want GPU training/inference. The unit tests and the
synthetic examples need no GPU.

Hardware used for the paper: one NVIDIA RTX 4090 (24 GB class), CUDA 13.0.
T5 training and beam generation need a GPU; the SID construction, calibration,
aggregation and metric math run on CPU.

## Path configuration

All paths come from environment variables; no absolute path is hard-coded and
every default is relative to the repository:

| variable | meaning | default |
|---|---|---|
| `CFLA_ARTIFACT_ROOT` | input artifacts you prepare: preprocessed data, SID arrays, checkpoints, beam dumps (read-only for the code) | `<repo>/artifacts` |
| `CFLA_OUTPUT_ROOT` | all new artifacts produced by a run | `<repo>/outputs` |
| `CFLA_UPSTREAM_ROOT` | fallback root consulted if a file is missing from `CFLA_ARTIFACT_ROOT` | `$CFLA_ARTIFACT_ROOT` |
| `CFLA_RAW_DATA` | raw Amazon Reviews 2014 files (`reviews_*_5.json.gz`, `meta_*.json.gz`) | `<repo>/raw` |
| `CFLA_SENTENCE_T5_DIR` | local `sentence-t5-base` checkpoint (embedding preparation only) | `<repo>/sentence-t5-base` |
| `CFLA_PYTHON` | interpreter used by the shell wrappers | `python3` |

The expected input layout is documented in `docs/DATA_FORMAT.md`.

## Sanity check

```bash
bash scripts/check_environment.sh
```

prints the interpreter, package versions and resolved path roots, and fails if a
required runtime dependency is missing.
