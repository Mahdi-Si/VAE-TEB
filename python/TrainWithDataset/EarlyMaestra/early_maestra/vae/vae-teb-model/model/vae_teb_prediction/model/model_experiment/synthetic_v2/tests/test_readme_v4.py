r"""S8-T03: the ``README_V4.md`` operator guide renders and covers the required sections."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.v4

_README = Path(__file__).resolve().parent.parent / "README_V4.md"


def test_readme_exists() -> None:
    assert _README.is_file(), "README_V4.md is missing"


def test_readme_covers_required_sections() -> None:
    r"""The README must cover goals, stages, arms, the render decision, relationships, and a runbook."""
    text = _README.read_text(encoding="utf-8")
    for heading in (
        "## Goals",
        "## Stages",
        "## Arms",
        "## Direct vs am_carrier rendering",
        "## Relationship to `synthetic_v3` and to `model_raw`",
        "## Runbook",
        "## Deferred analyses",
    ):
        assert heading in text, f"README_V4.md missing section: {heading}"


def test_readme_runbook_has_commands() -> None:
    r"""The runbook lists the per-stage commands for both the pilot and the headline sweep."""
    text = _README.read_text(encoding="utf-8")
    for token in ("--stage build", "--stage train", "--stage eval", "--stage report",
                  "--stage arms_report", "--dry-run"):
        assert token in text, f"runbook missing command: {token}"
    # Both caches referenced (direct primary + am_carrier probe).
    assert "config_synth_v4_am.yaml" in text


def test_readme_documents_v3_relationship() -> None:
    r"""The README explains the raw-vs-scattering relationship and the dropped te_scat axis."""
    text = _README.read_text(encoding="utf-8")
    assert "te_scat" in text
    assert "scattering" in text.lower()
