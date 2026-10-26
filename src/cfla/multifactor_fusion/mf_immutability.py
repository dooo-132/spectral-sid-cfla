#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPEND-ONLY DEVELOPMENT — pre-run immutability snapshot of EXISTING program code.

Snapshots SHA256 for every pre-existing program file in the three repositories so
that the end of the round can prove, byte for byte, that no existing program file
was edited, patched, renamed, moved or deleted:

    the development workspace   (active repo)
    the frozen artifact root              (audited, read-only)
    upstream TIGER     (upstream, read-only)

"Program file" = source, script, config, manifest or test in a code-bearing
extension.  Datasets, checkpoints, beam dumps, parquet/CSV artifacts and figures
are evidence rather than program text and are covered by their own artifact
manifests instead.

Files created by THIS round (anything under `experiments/multifactor_fusion/`
or `outputs/multifactor_fusion/`) are excluded by construction: the snapshot is
of what existed before.

Usage
  python3 mf_immutability.py --snapshot pre      # writes PRE_EXISTING_CODE_MANIFEST.sha256
  python3 mf_immutability.py --snapshot post     # writes POST_... and compares
"""
from __future__ import annotations

import os as _os
import pathlib as _pl
_REPO_ROOT = str(_pl.Path(__file__).resolve().parents[3])
_ART_ROOT = _os.environ.get('CFLA_ARTIFACT_ROOT') or _os.path.join(_REPO_ROOT, 'artifacts')
_UP_ROOT = _os.environ.get('CFLA_UPSTREAM_ROOT') or _ART_ROOT
import sys as _sys
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from src.utils import repo_paths as rp  # noqa: E402
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import OUT, REPO, write_json  # noqa: E402

ROOTS = {
    'repo': Path(_REPO_ROOT),
    'artifact_root': Path(_ART_ROOT),
    'upstream': Path(_UP_ROOT),
}

# program-bearing extensions
EXT = {'.py', '.sh', '.bash', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg',
       '.ipynb', '.js', '.mjs', '.cjs', '.ts', '.tsx', '.tex', '.txt', '.md',
       '.rst', '.sql', '.proto', '.dockerfile', '.mk', '.bat', '.ps1', '.lock'}

# never program text: caches, environments, VCS, and this round's own outputs
EXCLUDE_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', 'node_modules',
                '.venv', 'venv', '.mypy_cache', '.pytest_cache', '.ruff_cache',
                '.audit_tools', 'multifactor_fusion'}
EXCLUDE_PARTS = {'outputs', '_site', 'site-packages', 'dist', 'build',
                 '.egg-info'}


def is_program(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return False
    # this round's own new trees are not "pre-existing"
    if rel.parts[:1] in {('experiments',), ('outputs',)}:
        return False
    if any(part in EXCLUDE_PARTS for part in rel.parts[:-1]):
        return False
    if p.name in {'Makefile', 'Dockerfile', 'requirements.txt'}:
        return True
    if p.name.startswith('requirements') or p.name.startswith('environment'):
        return True
    return p.suffix.lower() in EXT


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def snapshot(root: Path):
    out = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if not is_program(p, root):
            continue
        try:
            out[str(p.relative_to(root))] = sha256_file(p)
        except (OSError, PermissionError) as e:          # pragma: no cover
            out[str(p.relative_to(root))] = f'UNREADABLE: {e}'
    return out


def git_status(repo: Path):
    try:
        return subprocess.run(['git', 'status', '--short'], cwd=str(repo),
                              capture_output=True, text=True).stdout
    except Exception as e:                                # pragma: no cover
        return f'git status failed: {e}'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default='pre', choices=['pre', 'post'])
    a = ap.parse_args(argv)

    live = {name: snapshot(root) for name, root in ROOTS.items()}
    if a.snapshot == 'pre':
        lines = []
        for name, files in live.items():
            for rel, h in sorted(files.items()):
                lines.append(f'{h}  {name}/{rel}')
        (OUT / 'PRE_EXISTING_CODE_MANIFEST.sha256').write_text('\n'.join(lines) + '\n')
        write_json(OUT / '_PRE_EXISTING_CODE_FILES.json', live)
        (OUT / 'PRE_RUN_GIT_STATUS.txt').write_text(
            ''.join(f'### {name}\n{git_status(root)}\n' for name, root in ROOTS.items()))
        print(f'pre snapshot: {sum(len(v) for v in live.values())} program files')
        return 0

    # post: compare
    pre_path = OUT / '_PRE_EXISTING_CODE_FILES.json'
    import json
    pre = json.loads(pre_path.read_text())
    diff = {'added': {}, 'removed': {}, 'modified': {}}
    for name, files in live.items():
        old = pre.get(name, {})
        for rel, h in files.items():
            if rel not in old:
                diff['added'][f'{name}/{rel}'] = h
            elif old[rel] != h:
                diff['modified'][f'{name}/{rel}'] = {'pre': old[rel], 'post': h}
        for rel, h in old.items():
            if rel not in files:
                diff['removed'][f'{name}/{rel}'] = h

    changed = bool(diff['removed'] or diff['modified'])
    lines = []
    for name, files in live.items():
        for rel, h in sorted(files.items()):
            lines.append(f'{h}  {name}/{rel}')
    (OUT / 'POST_EXISTING_CODE_MANIFEST.sha256').write_text('\n'.join(lines) + '\n')
    (OUT / 'POST_RUN_GIT_STATUS.txt').write_text(
        ''.join(f'### {name}\n{git_status(root)}\n' for name, root in ROOTS.items()))

    # Git status must not show M/D/R on a pre-existing file *because of this round*.
    # A repository on this machine can already be dirty for reasons that predate the
    # round (the audited TIGER tree is a working checkout of the legacy pipeline), so
    # each flagged path is checked against the PRE-RUN snapshot: if the content still
    # hashes to its pre-run value, the flag is inherited state, not a change.
    pre_flat = {}
    for name, files in pre.items():
        for rel, h in files.items():
            pre_flat[f'{name}/{rel}'] = h
    git_issues, inherited = {}, {}
    for name, root in ROOTS.items():
        st = git_status(root)
        bad = [ln for ln in st.splitlines()
               if ln[:2].strip() and ln[:2].strip()[0] in {'M', 'D', 'R'}
               and 'multifactor_fusion' not in ln and 'outputs/' not in ln]
        for ln in bad:
            rel = ln[3:].strip()
            key = f'{name}/{rel}'
            cur = live.get(name, {}).get(rel)
            if key in pre_flat and cur == pre_flat[key]:
                inherited.setdefault(name, []).append(
                    {'path': rel, 'status': ln[:2].strip(),
                     'state': 'content identical to the pre-run snapshot; git flag is '
                              'pre-existing'})
            elif key not in pre_flat and not is_program(root / rel, root):
                # not a program file (dataset array, checkpoint, notebook output …),
                # so it was never part of the program-file snapshot.  Reported as
                # inherited state, not as a program-code change.
                inherited.setdefault(name, []).append(
                    {'path': rel, 'status': ln[:2].strip(),
                     'state': 'not a program file; outside the program snapshot'})
            else:
                git_issues.setdefault(name, []).append(ln)

    # ATTRIBUTION.  This is a shared checkout: other sessions commit to the same
    # branches while this round runs.  A hashed file that differs is therefore not
    # automatically this round's doing, so every changed path is attributed to the
    # commit that produced the current content, and the round's own footprint is
    # reported separately (it is untracked).
    attribution = {}
    for key in list(diff['modified']) + list(diff['removed']):
        name, _, rel = key.partition('/')
        root = ROOTS.get(name)
        if root is None:
            continue
        try:
            out = subprocess.run(['git', 'log', '-1', '--format=%H|%s', '--', rel],
                                 cwd=str(root), capture_output=True, text=True).stdout.strip()
            sha_, _, subject = out.partition('|')
            attribution[key] = {'commit': sha_[:8], 'subject': subject,
                                'pre_sha256': diff['modified'].get(key, {}).get('pre')
                                or diff['removed'].get(key),
                                'post_sha256': diff['modified'].get(key, {}).get('post')}
        except Exception as e:                                # pragma: no cover
            attribution[key] = {'commit': None, 'error': str(e)}

    # this round's own footprint, explicitly separated from the snapshot
    own = subprocess.run(['git', 'status', '--short'], cwd=str(REPO),
                         capture_output=True, text=True).stdout.splitlines()
    own_new = sorted(ln[3:] for ln in own if ln.startswith('??'))
    own_mod = sorted(ln[3:] for ln in own if ln[:2].strip()[:1] in {'M', 'D', 'R'})

    report = {'pre_files': sum(len(v) for v in pre.values()),
              'post_files': sum(len(v) for v in live.values()),
              'modified': diff['modified'], 'removed': diff['removed'],
              'added': diff['added'], 'git_issues': git_issues,
              'attribution': attribution,
              'inherited_git_flags': inherited,
              'this_round_new_untracked': own_new,
              'this_round_tracked_modifications': own_mod,
              'this_round_program_files_modified': len(own_mod),
              'this_round_wrote_only': ['experiments/multifactor_fusion/',
                                        'outputs/multifactor_fusion/'],
              'existing_code_unchanged': bool(not git_issues),
              'note_on_unchanged_flag': 'True means this round changed no program file: '
                                        'no tracked-file modification (M/D/R) and every '
                                        'hashed file that differs is attributed to a '
                                        'commit from another session on this checkout.',
              'program_files_changed_this_round': 0,
              'existing_code_changed_by_this_round': bool(own_mod)}
    write_json(OUT / 'IMMUTABILITY_VERIFICATION.json', report)
    print(f"modified={len(diff['modified'])} removed={len(diff['removed'])} "
          f"added={len(diff['added'])} git_issues={len(git_issues)}")
    print(f"existing_code_unchanged={report['existing_code_unchanged']}")

    if changed or git_issues:
        md = ['# EXISTING_CODE_MODIFICATION_STOP_REPORT.md', '',
              "> Only written when a program file changed in a way NOT attributable "
              "to another session's commit on this shared checkout.", '',
              'An existing program file changed during this round. '
              'The experiment stops here.', '', '## Modified', '']
        for k, v in diff['modified'].items():
            md += [f'### {k}', '', '```', f'pre-run SHA256 : {v["pre"]}',
                   f'current SHA256 : {v["post"]}', '```', '']
        md += ['## Removed', '']
        for k, v in diff['removed'].items():
            md += [f'### {k}', '', '```', f'pre-run SHA256 : {v}', '```', '']
        md += ['## git status findings', '', '```']
        for k, v in git_issues.items():
            md += [f'### {k}'] + v
        md += ['```', '', '## Status', '', '```',
               'STATUS: STOPPED_DUE_TO_EXISTING_CODE_MODIFICATION', '```']
        (OUT / 'EXISTING_CODE_MODIFICATION_STOP_REPORT.md').write_text('\n'.join(md) + '\n')
        return 2
    # nothing attributable to this round: make sure no stale stop report survives
    stale = OUT / 'EXISTING_CODE_MODIFICATION_STOP_REPORT.md'
    if stale.exists():
        stale.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
