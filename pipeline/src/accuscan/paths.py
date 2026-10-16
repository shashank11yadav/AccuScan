"""Repository and dataset locations."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = PIPELINE_DIR.parent
CONFIG_DIR = PIPELINE_DIR / "configs"
REPORTS_DIR = PIPELINE_DIR / "reports"
RUNS_DIR = PIPELINE_DIR / "runs"
RESOURCES_DIR = PACKAGE_DIR / "resources"
WEB_PUBLIC = REPO_ROOT / "web" / "public"
MODELS_DIR = WEB_PUBLIC / "models"
LEXICON_PATH = WEB_PUBLIC / "lexicon" / "words.txt.gz"
SAMPLES_DIR = WEB_PUBLIC / "samples"
FIXTURES_DIR = REPO_ROOT / "shared" / "fixtures"


def default_data_dir() -> Path:
    """`$DATA_DIR`, else `_datasets/` next to the repository."""
    env = os.environ.get("DATA_DIR")
    return Path(env) if env else REPO_ROOT.parent / "_datasets"
