#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the CFLA release (CPU only, synthetic inputs, no dataset).

Run from the repository root:

    python3 tests/run_tests.py            # all tests
    python3 tests/run_tests.py -v         # verbose

or with the standard library directly:

    python3 -m unittest discover -s tests -v

The tests never touch the paper's data, checkpoints or experiment artifacts:
they generate small deterministic synthetic inputs and, for the code that
writes artifacts, redirect both `CFLA_ARTIFACT_ROOT` and `CFLA_OUTPUT_ROOT` to a
temporary directory.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Isolated, throw-away roots for anything a test writes.
_TMP = Path(tempfile.mkdtemp(prefix="cfla_tests_"))
os.environ["CFLA_ARTIFACT_ROOT"] = str(_TMP / "artifacts")
os.environ["CFLA_OUTPUT_ROOT"] = str(_TMP / "outputs")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    print("CFLA release - unit tests (synthetic inputs only)")
    print(f"scratch roots: {_TMP}")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).resolve().parent),
                            pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
