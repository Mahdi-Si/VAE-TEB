"""Pytest setup for the ``train/`` framework test suite.

Puts the repository root on ``sys.path`` so absolute imports such as
``import train.graph_model_base`` (which transitively pulls
``utils.custom_logger``) resolve when the suite is collected, and best-effort
pins the repo-root ``utils`` package so a near-empty sibling cannot shadow it.
Mirrors ``model/vae_teb_prediction/model/tests/conftest.py``.
"""
import importlib
import sys
from pathlib import Path

import pytest

# train/tests/conftest.py -> parents[0]=tests, [1]=train, [2]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# Best-effort: bind the repo-root ``utils`` package early to pin its __path__.
try:
    importlib.import_module("utils")
except Exception:
    pass


@pytest.fixture
def config_path() -> Path:
    """Absolute path to the shipped ``train/config.yaml``."""
    return Path(__file__).resolve().parents[1] / "config.yaml"
