#!/usr/bin/env python
"""Build the publication figures from committed results.

    python scripts/make_figures.py

Figures are generated, never hand-drawn, so they cannot drift from the numbers in
results/. A missing run produces a loud skip rather than a stale figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.viz.figures import make_all  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(make_all())
