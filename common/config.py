"""Pipeline configuration: PIPELINE_ROOT and the paths derived from it.

Contract (DESIGN.md §2): all pipeline data — raw vendor files, the PIT panel,
frozen catalogue copies, run directories and the registry — lives outside the
repo at PIPELINE_ROOT. This module is the only place that reads the
PIPELINE_ROOT environment variable; everything else imports paths from here.

REPO_ROOT / CATALOGUE_SOURCE_DIR point at the repo's own checked-in
`catalogue/` — the source of truth `catalogue.py` loads from. This is a
different thing from CATALOGUE_DIR (`$PIPELINE_ROOT/catalogue/`), which
DESIGN §2 describes as frozen, hashed per-run copies of that source,
written when a run begins.
"""

import os
from pathlib import Path

PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./data")).resolve()

RAW_DIR = PIPELINE_ROOT / "raw"
PANEL_DIR = PIPELINE_ROOT / "panel"
CATALOGUE_DIR = PIPELINE_ROOT / "catalogue"
RUNS_DIR = PIPELINE_ROOT / "runs"
REGISTRY_PATH = PIPELINE_ROOT / "registry.parquet"

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_SOURCE_DIR = REPO_ROOT / "catalogue"
