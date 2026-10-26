# ANONYMITY AUDIT — CFLA code release

Date: 2026-10-26. Release under audit: the `spectral-sid-cfla` directory of this
bundled artifact. Target: the authors' GitHub repository, to be anonymized through
Anonymous GitHub for double-blind review.

**This audit is local and independent of any anonymization service.** Such a service
anonymizes the repository *presentation*; it does not clean file contents, commit
metadata or licensing text.

## 1. Scope and method

Every file in the release tree was scanned, including filenames, directory names,
YAML/JSON configs, shell scripts and Markdown documents. The scan checked for, and the
final tree contains none of:

* personal identity — author names, personal e-mail addresses, GitHub owner handles,
  personal homepages/social accounts, institution or lab names;
* environment identity — POSIX root/home absolute paths, Windows user-profile paths,
  cloud-GPU provider / instance information, container hostnames;
* repository identity — the original repository/project/audit/branch names, internal
  commit identifiers, dated internal run-directory names;
* credentials — private keys, cloud access keys, API tokens, passwords, cookies;
* experiment-tracking links or run identifiers;
* data or artifacts — raw review data, preprocessed splits, embeddings, SID arrays,
  checkpoints, beam dumps, predictions, result tables;
* language signal — CJK characters;
* an inherited `.git` directory (the original history is not copied).

## 2. Findings and actions

| # | category | where it was | severity | action |
|---|---|---|---|---|
| 1 | **Author GitHub handle in the licence** | the project `LICENSE` carried the author's GitHub handle as copyright holder | **P0 if shipped** | `LICENSE` is the MIT licence with `Copyright (c) 2025 Anonymous Authors`. The real holder must be restored in the camera-ready public release (also stated in `README.md` §11 and `THIRD_PARTY_NOTICES.md`). |
| 2 | **Absolute server paths** | source constants/docstrings/configs/docs referred to the authors' machine paths | P0 if shipped | Removed: all paths resolve from environment variables (`CFLA_ARTIFACT_ROOT`, `CFLA_OUTPUT_ROOT`, `CFLA_UPSTREAM_ROOT`, `CFLA_RAW_DATA`, `CFLA_SENTENCE_T5_DIR`) with repository-relative defaults. Final tree: **0 occurrences**. |
| 3 | **Internal repository/project/audit names; internal branch/commit ids** | docstrings, comments, configs, generated metadata | P1 | Neutralized or removed together with the research-support scripts that carried them. Final tree: **0 occurrences**. The public method/repository name "TIGER" remains only as citation of the third-party method (required attribution), never as the authors' internal repository. |
| 4 | **Original Git history** | the development repository `.git` (large history with internal branch names and commit messages) | P0 if copied | **Not copied.** The release was re-initialised as a fresh repository with a single anonymous commit (`Anonymous Authors <anonymous@example.com>`, UTC author date); `git fsck --lost-found` is empty. |
| 5 | **Server / cloud-GPU identity** | generated run metadata in the authors' `outputs/` | P0 if shipped | All `outputs/`, run metadata, logs and caches excluded; the final tree contains no generated run metadata or hostname. |
| 6 | **Internal audit/process reports and research-support code** | documentation and analysis scripts describing the internal repositories/audits, plus frozen-artifact study code | P1 | Excluded; the release ships only the runnable pipeline and the neutral format/protocol documents, rewritten without internal names. |
| 7 | **Internal workflow markers** | comments with internal task numbering / phase markers | P2 | Reduced to 3 generic occurrences in reference comments (no identity, institution, server or repository information); disclosed here. |
| 8 | **Non-English comments** | four inherited tokenizer/analysis files with Chinese comments | P2 | Those tokenizer implementations are **no longer shipped** (RQ-VAE / RQ-OPQ are referenced by external repository links). Final tree: **0 CJK characters**. |
| 9 | **Credentials / keys / tokens** | whole tree | — | **None found.** No private keys, cloud keys, API tokens, passwords, cookies or `.env` files. Keyword hits are false positives (e.g. the word "token" in SID-token layout code). |
| 10 | **Experiment-tracking links** | whole tree | — | **None.** `.gitignore` lists the standard tracking-directory ignore patterns only. |
| 11 | **E-mail addresses / personal pages** | whole tree | — | **None.** The only e-mail-like strings are the non-identifying placeholder `anonymous@example.com` and the `git@github.com` SSH host placeholder in the documented upload command. |
| 12 | **Data and model artifacts** | whole tree | — | **None shipped.** No `.npy`, `.npz`, `.parquet`, `.pth`, `.pt`, `.pkl`, `.csv`, `.zip`, `.gz` or `.log` files; the synthetic smoke pipeline generates its inputs at run time in a temporary directory. |
| 13 | **Third-party attribution** | `THIRD_PARTY_NOTICES.md` | — | Preserved: TIGER (Rajput et al.), the external RQ-VAE repository, OneSearch/FAISS for RQ-OPQ, Amazon Reviews 2014 dataset terms, Python package licences. The RQ-VAE/RQ-OPQ *implementations* are deliberately not bundled; only links to public repositories are given. |

## 3. Final-tree re-scan (evidence)

| check | result |
|---|---|
| files / bytes (excluding `.git`) | **83 files / 347,886 bytes** |
| absolute server paths | **0** |
| internal repository/project/audit/branch identifiers, internal commit SHAs | **0** |
| cloud-GPU provider / instance information | **0** |
| personal e-mail addresses, personal pages, institution names | **0** |
| private keys / cloud keys / API tokens / passwords | **0** |
| experiment-tracking links or run ids | **0** |
| CJK characters | **0** |
| data/weight/result artifacts | **0** |
| files > 5 MB | **0** (largest 22,472 bytes) |
| git objects with identifying content | **0** (single anonymous commit; no dangling objects) |

Note: three generic references to internal "phase"/"protocol section" numbering remain
in explanatory comments (row 7); they convey no identity, institution, server or
repository information.

## 4. Residual risks

1. **Code-style fingerprinting** — as with any code release, style could in principle
   be compared with public repositories; no file links to an author and the original
   history is not shipped.
2. **Licence holder anonymized** — `LICENSE` says "Anonymous Authors"; must be
   restored before a public de-anonymized release (MIT attribution requirement).
3. **External repository links** — `README`/`THIRD_PARTY_NOTICES` link to public
   third-party projects (RQ-VAE, FAISS, dataset page); those links are attribution,
   not author identity.
4. **Anonymous GitHub is not a content filter** — the service must be given a
   repository whose files are already anonymous; this audit must be re-run if the
   content changes before uploading.

## 5. Conclusion

**Suitable for double-blind submission: YES WITH ACTIONS.**

The release contains no author, institution, server, credential, dataset, checkpoint
or result information; the original Git history is not included; third-party
attribution is preserved; the project licence holder is anonymized with an explicit
restoration note.

Author actions (not blocking the anonymous upload):

* push this repository and submit its URL to Anonymous GitHub; verify the anonymized
  page in a private browser before pasting the link into the submission;
* restore the real copyright holder in `LICENSE` only in the camera-ready/public
  version;
* do not add author-identifying commits while the review is ongoing.
