r"""S8-T05: the Sprint-8 closeout guard.

The load-bearing claim of ``synthetic_v4`` is that it is **additive**: it adds ``*_v4`` modules /
tests / config / docs and edits *only* v4-owned files (``run_pipeline_v4.py``, ``conftest_v4.py``,
``eval_v4.py``, ``eval_runner_v4.py`` — all ``_v4``), and touches **no** existing v2/v3 or
``model_raw`` source. This guard asserts that from the working tree:

* the git porcelain status shows no **modified/deleted/renamed** file under the pipeline
  (``synthetic_v2/``) or ``model_raw/`` that is not itself v4-owned;
* (opt-in) the ``-m "not v4"`` suite stays green.

It complements the always-on S0-T07 collection guard (``test_regression_guard_v4.py``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

pytestmark = pytest.mark.v4

_TESTS_DIR = Path(__file__).resolve().parent

#: Repo-relative directory prefixes whose *existing* (v2/v3/model_raw) sources must not change.
_PROTECTED_PREFIXES = (
    "model/vae_teb_prediction/model/model_raw/",
    "model/vae_teb_prediction/model/model_experiment/synthetic_v2/",
)

#: A path is **v4-owned** (allowed to change) when its basename carries the v4 tag.
_V4_OWNED = re.compile(r"(_v4\.|_v4$|_V4\.|^SYNTHETIC_V4|^config_synth_v4)", re.IGNORECASE)

#: Worktree/index status codes that mean an existing tracked file was changed (not merely added).
_CHANGE_CODES = frozenset({"M", "D", "R", "C"})


def _porcelain_changes() -> List[Tuple[str, str]]:
    r"""Return ``[(xy, path)]`` for every porcelain entry (rename resolves to the new path)."""
    out = subprocess.run(["git", "-C", str(_TESTS_DIR), "status", "--porcelain"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"git status failed: {out.stderr}"
    changes: List[Tuple[str, str]] = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        xy, rest = line[:2], line[3:]
        path = rest.split(" -> ", 1)[1] if " -> " in rest else rest
        changes.append((xy, path.strip().strip('"')))
    return changes


def _is_protected_change(xy: str, path: str) -> bool:
    r"""True iff this porcelain entry modifies an existing, non-v4 v2/v3/model_raw source."""
    posix = path.replace("\\", "/")
    if not any(posix.startswith(p) for p in _PROTECTED_PREFIXES):
        return False
    if _V4_OWNED.search(Path(posix).name):
        return False
    # Untracked ('??') is a new file, never a modification; only real change codes count.
    if xy == "??":
        return False
    return any(c in _CHANGE_CODES for c in xy)


def test_no_v2_v3_model_raw_source_modified() -> None:
    r"""No existing v2/v3/``model_raw`` source is modified/deleted/renamed (additive-only)."""
    violations = [f"{xy} {path}" for xy, path in _porcelain_changes()
                  if _is_protected_change(xy, path)]
    assert not violations, (
        "synthetic_v4 must be additive, but these existing v2/v3/model_raw sources changed:\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.skipif(
    os.environ.get("SYNTH_V4_RUN_REGRESSION") != "1",
    reason="set SYNTH_V4_RUN_REGRESSION=1 to run the full non-v4 green closeout (slow)",
)
def test_non_v4_suite_is_green_closeout() -> None:
    r"""Opt-in: the fast (non-slow) v2/v3 suite is actually green at closeout."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(_TESTS_DIR),
         "-m", "not v4 and not slow", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, f"v2/v3 suite regressed:\n{proc.stdout[-4000:]}"
