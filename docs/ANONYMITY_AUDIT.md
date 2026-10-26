# ANONYMITY AUDIT — CFLA code release

Date: 2026-10-26. Release under audit: the `spectral-sid-cfla` directory of this
artifact bundle. Target: a repository uploaded by the authors and anonymized
through Anonymous GitHub.

**This audit is local and independent of any anonymization service.** Such a
service hides the repository owner in its viewer; it does not clean file
contents, commit metadata or licensing text.

## 1. Scan scope and method

All files in the release tree were scanned, including filenames, directory
names, YAML/JSON configs, shell scripts and Markdown documents. The scan looked
for, and the final tree was checked to contain none of:

* personal identity: author names, personal e-mail addresses, GitHub owner
  handles, personal homepages and social accounts, institution/lab names;
* environment identity: POSIX root/home absolute paths, Windows user-profile
  paths, cloud-GPU provider or instance information, container hostnames;
* repository identity: the authors' original repository/project/audit/branch
  names, internal commit identifiers and dated internal run-directory names;
* credentials: private keys, cloud access keys, API tokens, passwords and
  cookies;
* experiment-tracking links or run identifiers;
* data or artifact leakage: raw review data, preprocessed splits, embeddings,
  SID arrays, checkpoints, beam dumps, predictions and result tables;
* language signal: CJK (Chinese/Japanese/Korean) characters;
* any inherited `.git` directory (the original repository history is not copied; a fresh single-commit repository is prepared for upload and is free of original objects).

The candidate pool was audited before assembly (the authors' source project and
its frozen evidence repository), then the assembled release was re-scanned as a
whole twice: mid-assembly and on the final tree (§3–§4), and once more by an
independent reviewer (`docs/RELEASE_VALIDATION.md` §5).

Threat model: a reviewer who knows nothing about the authors must not be able to
recover (a) author identity, (b) institution, (c) the original (non-anonymous)
repository or its history, (d) server/environment identity, (e) credentials.

## 2. Findings and actions

| # | category | where it was | severity | action taken |
|---|---|---|---|---|
| 1 | **Author GitHub handle in the licence** | the project `LICENSE` carried the author's GitHub handle as copyright holder | **P0 if shipped** | `LICENSE` rewritten as MIT with `Copyright (c) 2025 Anonymous Authors`. The MIT terms are the original project's licence; only the holder line was anonymized. **Authors must restore the real holder in the camera-ready version** (also noted in `README.md` §11 and `THIRD_PARTY_NOTICES.md`). |
| 2 | **Absolute server paths** | source constants, docstrings, configs and docs referred to the authors' machine paths for the development repository, the frozen artifact root and the upstream reference checkout | P0 if shipped | Removed. Every path is now resolved from environment variables (`CFLA_ARTIFACT_ROOT`, `CFLA_UPSTREAM_ROOT`, `CFLA_OUTPUT_ROOT`; see `docs/ENVIRONMENT.md`) with repository-relative defaults. Final tree: **0 occurrences**. |
| 3 | **Internal repository / project / audit names** | docstrings, comments, configs, generated metadata: internal names of the development repository, the frozen evidence repository and its audit directory, an internal repair branch and a historical commit hash, plus two unrelated internal workspace names | P1 | Neutralized to generic phrases ("the development workspace", "the frozen artifact root", "upstream TIGER", "frozen audit"), internal branch/commit/date strings removed, unrelated workspaces excluded entirely. Final tree: **0 occurrences**. The public method name "TIGER" remains where it is a citation or a class name — that is required third-party attribution, not an identity leak. |
| 4 | **Original Git history** | the original repository's `.git` (≈2.2 GB) with internal branch names, historical commit messages and contributor identity | P0 if copied | **Not copied.** The release is a fresh tree; a new local repository with a single initial commit and an anonymous author identity is prepared by the authors (see `docs/UPLOAD_CHECKLIST.md` §6). No original commit is reused. |
| 5 | **Server / cloud-GPU identity** | generated run metadata in the original `outputs/` contained a cloud-GPU container hostname and absolute roots | P0 if shipped | All `outputs/`, `logs/`, run metadata and caches excluded from the release (code-only mode). Final tree contains **no** generated run metadata, no hostname and no provider reference. |
| 6 | **Internal audit / process reports** | documentation describing the internal repositories, audit rounds and migration history by name | P1 | Excluded from the release. The technical facts a reviewer needs (data layout, SID format, item-id convention, protocol) are kept as neutral rewritten documents: `docs/DATA_FORMAT.md`, `docs/SID_FORMAT.md`, `docs/ITEM_ID_CONVENTION.md`, `docs/EXPERIMENT_PROTOCOL.md`. |
| 7 | **Internal workflow references** | comments/strings containing internal task numbering and phase markers, internal report filenames | P2 | Most were removed; some generic workflow wording remains in reference-only analysis scripts (no identity, institution, server or repository information). Method logic untouched. |
| 8 | **Chinese-language comments** in four inherited tokenizer/analysis files (no names, only technical notes) | tokenizer reference implementation and one analysis module | P2 (language signal) | Translated to English. Final tree: **0 CJK characters**. |
| 9 | **Credentials / keys / tokens** | whole tree | — | **None found.** No private keys, cloud keys, API tokens, passwords, cookies or `.env` files. Keyword hits were false positives (e.g. the word "token" in SID-token layout code). |
| 10 | **Experiment-tracking links** | whole tree | — | **None.** No experiment-tracking URLs, run ids or sweep links. The `.gitignore` lists the standard tracking-directory ignore patterns, which reveal nothing about the authors. |
| 11 | **E-mail addresses / personal pages / social accounts** | whole tree | — | **None.** No personal homepage, Google Scholar, LinkedIn, X or university/lab page. The only e-mail string is the non-identifying placeholder `anonymous@example.com` inside the documented git-identity command. |
| 12 | **Data and model artifacts** | whole tree | — | **None shipped.** No `.npy`, `.npz`, `.parquet`, `.pth`, `.pt`, `.pkl`, `.csv`, `.zip`, `.gz`, raw review data, preprocessed interactions, embeddings, SID arrays, checkpoints, beam dumps or prediction caches. The synthetic demo generates its inputs at run time inside a temporary directory. |
| 13 | **Third-party attribution** | `THIRD_PARTY_NOTICES.md` | — | Preserved and required: TIGER (Rajput et al., RecSys'23), the upstream MIT-licensed reference implementation, the OneSearch RQ-OPQ reference, Amazon Reviews 2014 dataset terms and the Python package licences. The upstream copyright-holder name is withheld for anonymity and must be restored before the camera-ready public release. |
| 14 | **Target repository default branch / commits** | the target GitHub repository | — | No remote repository existed at preparation time (a read-only check of the target URL returned HTTP 404 on 2026-10-26), so there is no existing public commit or history that could leak identity. The authors create the repository (default branch `main`) and push the prepared local commit (`docs/UPLOAD_CHECKLIST.md` §6). |

## 3. Final-tree re-scan (evidence)

Run on the final tree (236 files, ≈2.9 MB, excluding `.git`):

| check | result |
|---|---|
| absolute server paths (POSIX root/home prefixes, Windows user-profile prefix) | **0** |
| internal repository/product handles, server identifiers, internal branch names, internal commit SHAs | **0** |
| cloud-GPU provider / instance information | **0** |
| personal e-mail addresses | **0** (only the `anonymous@example.com` placeholder) |
| private keys / cloud keys / API tokens / passwords | **0** |
| experiment-tracking links or run ids | **0** |
| CJK characters | **0** |
| generic internal workflow / directory wording in reference-only modules (task-section and phase markers, internal analysis subdirectory and report-record names) | present in reference-only analysis scripts (32 task-section + 73 phase markers, plus generic internal analysis-directory/report wording) — generic English, no author/institution/server/repository identity; disclosed in §4.2 |
| data/weight/result artifacts | **0** |
| inherited original `.git` directory | **absent** (a fresh single-commit repository is prepared separately; it contains no original history) |
| files > 5 MB | **0** (largest ≈132 KB) |

An independent reviewer re-scanned the final tree with a separate pattern list;
see `docs/RELEASE_VALIDATION.md` §5.

## 4. Residual risks (low, disclosed)

1. **Code-style fingerprinting.** As with any code release, code style can in
   principle be compared with public repositories by a determined adversary.
   Nothing in the files links to an author, and the original history is not
   shipped.
2. **Generic experimental vocabulary.** Reference-only analysis scripts keep the
   authors' project-internal naming of experiment rounds and task sections in
   comments (32 "prompt section" and 73 phase markers). These are generic
   workflow words with no identity, institution, server or repository
   information, and were left because they document the provenance of the frozen
   numbers.
3. **Licence holder anonymized.** `LICENSE` says "Anonymous Authors". This is
   correct for double-blind review but must be restored before a public
   de-anonymized release (MIT attribution requirement).
4. **Anonymous GitHub is not a content filter.** The service anonymizes the
   repository presentation; it does not rewrite files. If the content changes
   before uploading, this audit must be repeated.

## 5. Conclusion

**Suitable for double-blind submission: YES WITH ACTIONS.**

The release contains no author, institution, server, credential, dataset,
checkpoint or result information; the original Git history is not included; all
third-party attribution is preserved; the project licence holder is anonymized
with an explicit restoration note.

Actions for the authors (not blocking the anonymous upload):

* create the GitHub repository (default branch `main`) and submit its URL to
  Anonymous GitHub; verify the anonymized page in a private browser;
* restore the real copyright holder in `LICENSE` and the upstream copyright
  notice in `THIRD_PARTY_NOTICES.md` only in the camera-ready/public version;
* do not add author-identifying commits while the review is ongoing.
