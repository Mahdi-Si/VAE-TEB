r"""S0-T07: regression guard -- the v4 additions must leave the v2/v3 suite intact.

The load-bearing claim of ``synthetic_v4`` is that it is *additive*: new ``*_v4`` modules plus a
one-line marker + fixture re-export in the shared ``conftest.py``, and nothing else. This guard
proves the shared conftest edit did not break v2/v3 collection, and registers both markers.

Fast path (always on): collect the ``-m "not v4"`` suite in a fresh subprocess and assert it
imports cleanly (catches a broken conftest / shadowed fixture). Full path (opt-in via
``SYNTH_V4_RUN_REGRESSION=1``): actually run ``-m "not v4 and not slow"`` and assert it is green.
The commands are re-run at each sprint boundary; see the module docstring for the exact form.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.v4

_TESTS_DIR = Path(__file__).resolve().parent


def test_both_markers_registered(pytestconfig) -> None:
    r"""The shared conftest still registers ``slow`` and now also ``v4`` (additive, not replaced)."""
    markers = pytestconfig.getini("markers")
    assert any(line.startswith("slow:") for line in markers), "slow marker lost"
    assert any(line.startswith("v4:") for line in markers), "v4 marker not registered"


def test_non_v4_suite_still_collects() -> None:
    r"""The ``-m "not v4"`` suite imports/collects cleanly in a fresh process (conftest is safe)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(_TESTS_DIR),
         "-m", "not v4", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        "non-v4 collection failed (the conftest edit may have broken v2/v3):\n"
        f"STDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    )


@pytest.mark.skipif(
    os.environ.get("SYNTH_V4_RUN_REGRESSION") != "1",
    reason="set SYNTH_V4_RUN_REGRESSION=1 to run the full v2/v3 green guard (slow)",
)
def test_non_v4_suite_is_green() -> None:
    r"""Opt-in: the fast (non-slow) v2/v3 suite is actually green."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(_TESTS_DIR),
         "-m", "not v4 and not slow", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, f"v2/v3 suite regressed:\n{proc.stdout[-4000:]}"
