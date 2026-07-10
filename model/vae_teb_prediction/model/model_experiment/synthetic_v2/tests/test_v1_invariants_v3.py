r"""S0-T07: v1 invariants guard for the synthetic_v3 effort.

A single fast module asserting the Section-2 non-goals, so every later sprint has ONE
command proving no v1 regression:

* the committed ``pl_module_v2`` alias still resolves to ``SeqVaeLagAttnV1`` (read
  dynamically via ``getattr`` so it is monkeypatch-sensitive -- self-tested here);
* ``config_synth_v2.yaml`` still resolves to a v1 model with today's kwargs;
* ``vae_teb_lag_attn_v1.py`` / ``vae_teb_lag_attn_trfr.py`` match pinned SHA-256 content
  hashes (the primary limb, with teeth outside a checkout), backed by a secondary
  ``git diff --quiet`` limb (skipped when not in a work tree).

Runs in well under a second; no GPU. Re-run at every sprint boundary alongside
``test_rollback_v1.py`` / ``test_v1_golden_regression.py`` / ``test_alias_build_v2.py`` /
``test_ckpt_class_guard.py``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    pl_module_v2 as plm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    run_pipeline_v2 as drv,
)

_MODEL_DIR = _REPO_ROOT / "model" / "vae_teb_prediction" / "model"
_CFG_V2 = (_MODEL_DIR / "model_experiment" / "synthetic_v2" / "config_synth_v2.yaml")

# Pinned SHA-256 content hashes of the byte-frozen v1 model modules, taken at commit
# dfb5ffcca83c622f334ade12391e406c1d1bcb4e (2026-07-09). The synthetic_v3 effort edits only
# vae_teb_lag_attn_v3.py (additively); if either of these changes, this test fails loudly and
# the change must be justified (Section 2 non-goal) before re-pinning the hash here.
_PINNED_AT_COMMIT = "dfb5ffcca83c622f334ade12391e406c1d1bcb4e"
_PINNED_HASHES = {
    "vae_teb_lag_attn_v1.py":
        "695776917cd01c5bb8ae60afa1c97525c4fa7aa96b4af68e31444edcc391af83",
    "vae_teb_lag_attn_trfr.py":
        "e104e54da0e4526b986acb582245505c06ce7c0fb94341a5a514d2784a3f847c",
}


# ---------------------------------------------------------------------------
# Alias invariant (dynamic, monkeypatch-sensitive)
# ---------------------------------------------------------------------------
def test_committed_alias_is_v1() -> None:
    """The committed ``pl_module_v2`` alias resolves to ``SeqVaeLagAttnV1``."""
    assert getattr(plm, "SeqVaeLagAttn").__name__ == "SeqVaeLagAttnV1"


def test_alias_guard_is_monkeypatch_sensitive(monkeypatch) -> None:
    """Self-test: the alias limb reads the live attribute, not a frozen import.

    Monkeypatching ``pl_module_v2.SeqVaeLagAttn`` to another class makes the invariant read a
    different name -- proving :func:`test_committed_alias_is_v1` would fail if the toggle were
    flipped, so its green is meaningful rather than vacuous.
    """
    class _Dummy:  # noqa: D401 - a stand-in whose __name__ is not the v1 class
        pass

    monkeypatch.setattr(plm, "SeqVaeLagAttn", _Dummy)
    assert getattr(plm, "SeqVaeLagAttn").__name__ != "SeqVaeLagAttnV1"


# ---------------------------------------------------------------------------
# config_synth_v2.yaml still resolves to v1
# ---------------------------------------------------------------------------
def test_config_v2_still_builds_v1() -> None:
    cfg = drv.load_config(_CFG_V2)
    assert "class" not in cfg["model"]
    model, kwargs = plm.build_model(cfg["model"], torch.device("cpu"))
    assert type(model).__name__ == "SeqVaeLagAttnV1"
    assert kwargs["d_model"] == 128
    assert kwargs["c_y"] == 87 and kwargs["c_u"] == 101


# ---------------------------------------------------------------------------
# Content-hash pin on the byte-frozen v1 modules (primary), + git diff (secondary)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", sorted(_PINNED_HASHES))
def test_protected_model_file_content_hash(filename) -> None:
    """The v1 / trfr modules match their pinned SHA-256 (byte-frozen non-goal)."""
    path = _MODEL_DIR / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == _PINNED_HASHES[filename], (
        f"{filename} changed (pinned at {_PINNED_AT_COMMIT}). The synthetic_v3 effort must "
        f"not edit it; if the change is intentional, justify it and re-pin the hash.\n"
        f"  got   {digest}\n  pinned {_PINNED_HASHES[filename]}"
    )


@pytest.mark.parametrize("filename", sorted(_PINNED_HASHES))
def test_protected_model_file_git_clean(filename) -> None:
    """Secondary limb: the module is unmodified in the work tree (skipped outside a checkout)."""
    rel = f"model/vae_teb_prediction/model/{filename}"
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "diff", "--quiet", "--", rel],
            capture_output=True,
        )
    except (OSError, FileNotFoundError):  # pragma: no cover - git absent
        pytest.skip("git not available")
    # ``git diff --quiet`` exits 1 when the file differs from the index/HEAD; 128 means the
    # path is not in a work tree / repo -> skip rather than fail.
    if proc.returncode == 128:
        pytest.skip("not in a git work tree")
    assert proc.returncode == 0, f"{rel} has uncommitted changes in the work tree"
