# UPLOAD CHECKLIST — CFLA anonymous release

Date: 2026-10-26. Release: the `spectral-sid-cfla` directory of this bundle
(the absolute local path is intentionally not written in this shipped document).
Target: the authors' GitHub repository, to be anonymized through Anonymous
GitHub. The repository URL / owner handle is supplied by the authors outside this
artifact and must not appear in the shipped files.

This checklist covers **local upload preparation only**. No `git push`, no GitHub
API write and no remote operation was performed.

## 1. Target repository status (checked read-only)

* A read-only request for the target repository URL returned **HTTP 404** on
  2026-10-26, i.e. the repository does not exist publicly yet (or is private).
  The handle is deliberately not repeated in this file.
* Therefore there is **no existing commit usable for Anonymous GitHub**, and no
  remote state to preserve or modify.
* Required user actions before upload:
  1. create the repository on GitHub — it must be readable by the Anonymous
     GitHub service (in practice a **public** repository; check the service's
     current instructions if you prefer private + access);
  2. make sure the **default branch is `main`** (select it explicitly if your
     account default differs);
  3. do **not** enable Git LFS (not needed — see §2) and do not let GitHub create
     a README/licence at creation time (the release already has them);
  4. push the prepared local commit (commands in §6), then submit the repository
     URL to <https://anonymous.4open.science/> to obtain the anonymized link;
  5. open the anonymized link in a private/incognito browser and confirm the
     rendered files are the anonymous ones before pasting the link into the
     submission.

## 2. Local inventory (measured on the final tree)

| item | value |
|---|---|
| absolute path | supplied to the authors separately (not written here) |
| files (excluding `.git`) | **236** |
| total size (excluding `.git`) | **≈2.9 MB** |
| largest file | `src/cfla/paper_pipeline/build_codebook_sensitivity.py` (131,856 bytes) |
| files > 5 MB | **none** |
| Git LFS required | **no** |
| inherited `.git` directory | **none** (the original ≈2.2 GB history is not copied) |
| Python files | 191 |
| YAML configs | 18 |
| Markdown docs | 13 |
| shell scripts | 10 |
| data/weight/result artifacts (`npy/npz/parquet/pth/pt/pkl/csv/zip/gz/log`) | **none** |

Included content: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`,
`THIRD_PARTY_NOTICES.md`, `configs/`, `src/` (spectral SID construction; SID
tokenizers incl. RQ-VAE5 / RQ2-OPQ3 / even-random split; T5 model and training;
beam candidate generation; teacher-forced NLL; zNLL+LSE CFLA scoring; the paper's
M5 pipeline drivers; evaluation; supporting ablation/analysis code),
`scripts/` (10 stage wrappers), `tests/` (4 CPU unit-test files),
`examples/synthetic_demo.py`, and `docs/` (11 documents including
`CODE_MAPPING.md`, `ANONYMITY_AUDIT.md`, `RELEASE_VALIDATION.md` and this file).

Excluded (code-only mode): raw review data and preprocessed splits, item
embeddings, SID arrays, model checkpoints/weights, beam dumps, candidate-NLL
caches, the paper source/PDF, all experiment results, the authors' original
`.git`, their working `outputs/` tree, internal evidence bundles, unrelated
reproduction workspaces, virtual environments, caches and logs.

## 3. Pre-upload checks

| # | check | evidence | status |
|---|---|---|---|
| 1 | `.gitignore` excludes data / weights / outputs / logs, and does not over-match shipped code | `data/`, `artifacts/`, `outputs/`, `ckpt/`, `*.pth`, `*.npy`, `*.npz`, `*.parquet`, `runs/`, `logs/`, `wandb/`, `__pycache__/`; the paper-submission rule is root-anchored as `/paper/`, so `configs/paper/` and `src/tokenization/paper/` ARE tracked | **PASS** |
| 2 | no file above GitHub's ordinary size limit | largest 132 KB | **PASS** |
| 3 | no plaintext secrets / keys / tokens | full-tree scan (`docs/ANONYMITY_AUDIT.md`) | **PASS** (none found) |
| 4 | no personal absolute paths | POSIX root/home and Windows-profile prefixes: 0 hits | **PASS** |
| 5 | no Git LFS dependency | no binaries, no LFS rules | **PASS** |
| 6 | README complete (11 sections) | Overview, Repository Structure, Environment Setup, Dataset Preparation, Semantic ID Construction, Training, Inference and CFLA, Evaluation, Ablation Studies, Reproducibility Notes, License and Acknowledgments | **PASS** |
| 7 | dependencies installable / accurate | `requirements.txt` pins match the verified environment (12 pins + 2 optional extras) | **PASS** |
| 8 | core code present | spectral SID, SID tokenizers, T5 training, beam generation, candidate NLL, zNLL+LSE, evaluation, M5 pipeline, ablation code — all present and mapped in `docs/CODE_MAPPING.md` | **PASS** |
| 9 | no experiment output directories shipped | tree contains only source / config / docs / tests / examples / scripts | **PASS** |
| 10 | third-party licences / attribution | `THIRD_PARTY_NOTICES.md`: TIGER, upstream MIT implementation, OneSearch RQ-OPQ, Amazon Reviews 2014 terms, Python packages; project `LICENSE` = MIT with anonymous holder | **PASS** (holder restoration required before camera-ready) |
| 11 | author-identity leaks | full-tree scan (`docs/ANONYMITY_AUDIT.md`) + independent re-scan | **PASS** |
| 12 | original project unmodified by this task | tracked index of the original repository unchanged against a baseline captured before release work; all writes confined to the release directory | **PASS** |
| 13 | smoke tests executed | `tests/run_tests.py` 13 tests OK; `examples/synthetic_demo.py` assertions pass; synthetic data/SID/evaluation pipeline passes (`docs/RELEASE_VALIDATION.md` §3) | **PASS** |
| 14 | target repository / default branch / existing commit | repository does not exist yet (HTTP 404); no usable commit → create repository with default branch `main`, push the prepared initial commit | **ACTION (user)** |

## 4. Blocking issues (P0)

**None** on the final tree.

## 5. Open issues for the user's decision (P1)

1. **Licence holder.** `LICENSE` says `Copyright (c) 2025 Anonymous Authors`.
   The MIT terms are the original project's licence; the holder line is
   anonymized for double-blind review and **must be restored** (together with the
   upstream MIT copyright notice in `THIRD_PARTY_NOTICES.md`) in the camera-ready
   public release. Confirm that MIT is the licence to keep.
2. **Repository visibility for Anonymous GitHub.** The service must be able to
   read the repository. If private, follow the service's current access
   instructions; the practical default is a public repository containing only
   this anonymous content.
3. **Embedded verification constants.** A few reference-only analysis scripts
   embed frozen metric values used to verify the frozen reports (clearly marked
   as the authors' frozen values); no result tables or data files are shipped.
4. **Runnability expectations.** `[needs your data]` and `[needs retraining]`
   stages cannot be re-run by a reviewer without Amazon Reviews 2014 and GPU
   training; `[reference only]` stages need the authors' frozen artifacts. The
   README states this explicitly. Decide whether you additionally want to ship a
   small anonymized result table (not recommended: it would break the current
   code-only claim and require a new audit).
5. **Post-review de-anonymization.** Restore author/affiliation/licence holder in
   a later commit only after the review period ends.

## 6. Suggested Git commands (prepared locally; nothing was pushed)

A fresh local repository has been initialised in the release directory on branch
`main` with a single initial commit, using the non-identifying identity
`Anonymous Authors <anonymous@example.com>`:

* branch: `main`
* one initial commit: "Anonymous code release for double-blind review" (see
  `git log` for the hash)
* tracked files: 236 (= every file in the tree; nothing is ignored)
* no git remote is configured; no push has been performed
* the repository was rebuilt from the clean working tree; `git fsck --unreachable`
  reports no dangling objects and no original history is present
* `.git` is kept for the push; **remove it before creating any ZIP/tarball** of the
  release (a push transfers only reachable objects, a zip would ship the whole `.git`)

Verify with:

```bash
cd <release-directory>
git status                       # clean working tree, branch main
git log --format='%h %an <%ae> %s'
git remote -v                    # empty: no remote configured
```

To upload (commands for **you** to run; the release task did not run them, and
replace the placeholders with your repository coordinates):

```bash
cd <release-directory>
git remote add origin git@github.com:<owner>/<repository>.git
git push -u origin main
```

Then:

1. on GitHub, check the pushed file list equals the local inventory (§2) and that
   the default branch is `main`;
2. submit the repository URL at <https://anonymous.4open.science/>, obtain the
   anonymous link, open it in a private browser and check that README, LICENSE
   and `docs/*` render anonymously;
3. paste the anonymous link into the submission system.

## 7. Verdict

**READY WITH WARNINGS.**

The release directory is complete, code-only, anonymous, self-contained, passes
the static and CPU smoke validation, and is ready for the authors to push to
GitHub and anonymize. The warnings are the user actions in §5 (licence-holder
restoration before camera-ready, repository visibility choice for Anonymous
GitHub, and the documented data/retraining/reproduction limits); none blocks the
double-blind upload of this artifact.
