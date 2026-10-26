# UPLOAD CHECKLIST — CFLA anonymous release

Date: 2026-10-26. Release: the `spectral-sid-cfla` directory of this bundle. Target:
the authors' GitHub repository, anonymized through Anonymous GitHub for double-blind
review.

## 1. Target repository status

* Repository: the authors' GitHub repository, supplied by them outside this
  artifact (the owner handle is deliberately not written into the shipped files).
* Default branch: `main`.
* The repository was created empty by the authors; the release is pushed as the first
  commit on `main` (no force, no overwrite of existing work).

## 2. Inventory (final tree)

| item | value |
|---|---|
| files (excluding `.git`) | **83** |
| payload size | **347,886 bytes (≈340 KiB)**; `du` block usage ≈532 KB |
| largest file | `src/tiger/train.py` (22,472 bytes) |
| Python / YAML / Markdown / shell | 38 / 17 / 13 / 10 |
| files > 5 MB | **none** |
| Git LFS required | **no** |
| data/weight/result artifacts (`npy/npz/parquet/pth/csv/zip/gz/log`) | **none** |
| test/example directories | not shipped (by authors' request) |
| RQ-VAE / RQ-OPQ implementations | not shipped — external repository links only |

Included: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`,
`THIRD_PARTY_NOTICES.md`, `configs/` (canonical + 16 experiment configs + CFLA pool
example + RQ-VAE5 recipe), `src/` (spectral SID, dimension-split baseline, T5
model/training, beam dumper, teacher-forced scoring, CFLA pipeline, metrics, utils,
runner), `scripts/` (10 stage wrappers incl. the synthetic end-to-end smoke
pipeline), `docs/` (11 documents incl. this checklist).

## 3. Pre-upload checks

| # | check | evidence | status |
|---|---|---|---|
| 1 | `.gitignore` excludes data / weights / outputs / logs / secrets | data/artifacts/outputs/ckpt, `*.pth|pt|ckpt|npy|npz|parquet|pkl`, runs/logs, `__pycache__`, venvs, `.env*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `.netrc`, `*.csv`, `*.zip`, `*.onnx`, ... ; root-anchored `/paper/` so shipped code is not ignored | **PASS** |
| 2 | no file above GitHub's ordinary size limit | largest 22,472 B | **PASS** |
| 3 | no plaintext secrets / keys / tokens | full-tree scan (`docs/ANONYMITY_AUDIT.md`) | **PASS** |
| 4 | no personal absolute paths / internal names / author identity | full-tree scan | **PASS** |
| 5 | no Git LFS dependency / submodules / executable hooks | no `.gitattributes`, no `.gitmodules`; hooks are samples only | **PASS** |
| 6 | README complete (11 sections) and command-accurate | sections Overview … License; every referenced file exists; flags checked against `argparse` | **PASS** |
| 7 | dependencies installable / accurate | `requirements.txt` pins match the verified environment | **PASS** |
| 8 | core code present and runnable from scratch | data prep → SID construction → training → candidate generation → CFLA → evaluation; validated by the synthetic end-to-end CPU pipeline | **PASS** |
| 9 | no experiment output directories / datasets / checkpoints shipped | tree contains code/config/docs/scripts only | **PASS** |
| 10 | third-party licences / attribution | `THIRD_PARTY_NOTICES.md`: TIGER, RQ-VAE (external), OneSearch/FAISS (external), Amazon Reviews 2014 terms, package licences; project `LICENSE` MIT with anonymized holder | **PASS** (holder restoration required for camera-ready) |
| 11 | smoke tests executed | `smoke_test.sh` PASSED; `smoke_pipeline.sh` PASSED (synthetic CPU end-to-end) | **PASS** |
| 12 | target repository / default branch | created empty on `main`; first push from this directory | **PASS** |

## 4. Blocking issues (P0)

**None.**

## 5. Open issues for the authors (P1)

1. **Licence holder** — `LICENSE` is MIT with `Copyright (c) 2025 Anonymous Authors`;
   restore the real holder (and upstream notices) before any de-anonymized public
   release.
2. **RQ-VAE / RQ-OPQ dependencies** — those tokenizers are not bundled; users must
   build the SID arrays with the linked repositories. This is documented in
   README §5.2–5.3 / `THIRD_PARTY_NOTICES.md`.
3. **Smoke numbers are synthetic** — `scripts/smoke_pipeline.sh` proves the pipeline
   runs on synthetic data with 3-epoch CPU models; its metrics are meaningless by
   design.
4. **Post-review de-anonymization** — restore author/affiliation/licence holder in a
   later commit only after the review period.

## 6. Git commands (no force)

```bash
cd <release-directory>
git status -sb
git log --oneline -1
git remote -v                      # origin -> the target repository
git ls-remote origin               # inspect remote refs before pushing
git push -u origin main            # no --force, no +refspec
git ls-remote origin refs/heads/main
```

## 7. Verdict

**READY WITH WARNINGS.** The artifact is code-only, anonymous, self-contained and
runnable from scratch (dataset downloaded by the user); static checks and the
synthetic end-to-end CPU pipeline pass. The warnings are the author actions in §5
(licence-holder restoration before camera-ready, external RQ tokenizers, synthetic
smoke semantics) — none blocks the double-blind upload.
