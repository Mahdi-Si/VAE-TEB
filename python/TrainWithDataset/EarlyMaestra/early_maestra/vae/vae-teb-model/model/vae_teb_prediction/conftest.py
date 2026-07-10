"""Pytest bootstrap for everything under ``model/vae_teb_prediction``.

Without this file the tests in ``testing/`` cannot even be collected. ``testing/`` is a regular
package but this directory is not, so pytest treats *this* directory as the base directory for
those modules and prepends it to ``sys.path``. It contains its own ``model/`` and ``utils/``
subpackages, which then shadow the repo-root ``model`` and ``utils`` -- and
``testing/__init__.py`` immediately does
``from model.vae_teb_prediction.testing.base import TestRunner``, which raises
``ModuleNotFoundError: No module named 'model.vae_teb_prediction'``.

This file lives one level *above* the ``testing`` package on purpose: pytest imports it as a
top-level module (this directory has no ``__init__.py``), so it runs before
``testing/__init__.py`` and before any test module. Putting the repo root first and eagerly
binding both names into ``sys.modules`` pins their ``__path__`` for the rest of the session,
whatever pytest later prepends.

The same shadowing bites ``utils``: ``utils.style`` exists only at the repo root, while
``model/vae_teb_prediction/utils`` is a near-empty package.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# vae_teb_prediction/ -> model/ -> <repo root>
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

for _shadowed in ("model", "utils"):
    importlib.import_module(_shadowed)
