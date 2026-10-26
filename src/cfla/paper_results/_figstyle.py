#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plotting style for the final paper figures.

Rules enforced here (harness section 2):
  * white background, no 3D, no gradients, no decoration
  * one font family and one font size set across every figure
  * colourblind-safe palette (Okabe-Ito)
  * every curve also distinguishable in grayscale by linestyle + marker
  * PNG >= 300 dpi, vector PDF and SVG alongside
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
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(_REPO_ROOT + "")
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette ---
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
    "black": "#000000",
}

DATASETS = [
    "Beauty",
    "Cell_Phones_and_Accessories",
    "Health_and_Personal_Care",
    "Toys_and_Games",
]
DISPLAY = {
    "Beauty": "Beauty",
    "Cell_Phones_and_Accessories": "Cell Phones",
    "Health_and_Personal_Care": "Health",
    "Toys_and_Games": "Toys",
}
# grayscale-safe: distinct colour + distinct linestyle + distinct marker
DATASET_STYLE = {
    "Beauty": dict(color=OKABE_ITO["blue"], ls="-", marker="o"),
    "Cell_Phones_and_Accessories": dict(color=OKABE_ITO["orange"], ls="--", marker="s"),
    "Health_and_Personal_Care": dict(color=OKABE_ITO["green"], ls="-.", marker="^"),
    "Toys_and_Games": dict(color=OKABE_ITO["vermillion"], ls=":", marker="D"),
}

FIGSIZE_1COL = (3.45, 2.60)
FIGSIZE_WIDE = (7.05, 2.60)
FIGSIZE_2X2 = (7.05, 4.90)

DPI = 400


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.4,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "legend.handlelength": 1.9,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.32,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.axisbelow": True,
    })


def save(fig, stem: str) -> list:
    """Write PDF + PNG(>=300dpi) + SVG for one figure; return the paths."""
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": DPI}), ("svg", {})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


def write_source_csv(rows, path: Path) -> None:
    """Write a plot-source CSV with full float precision (repr round-trip)."""
    import csv
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def audit_rows(rows: list, path: Path) -> None:
    """Write a SOURCE_AUDIT.csv (harness section 14)."""
    import csv
    fields = ["artifact", "source_file", "source_status", "dataset", "metric",
              "method", "value", "used_in"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(r[k]) if isinstance(r.get(k), float) else r.get(k, ""))
                        for k in fields})
