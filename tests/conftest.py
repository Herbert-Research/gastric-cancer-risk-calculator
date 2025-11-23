"""Pytest configuration for test discovery and path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add the repository root to sys.path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure joblib has a writable temp directory to avoid permission warnings
JOBLIB_TEMP = ROOT_DIR / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_TEMP))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
