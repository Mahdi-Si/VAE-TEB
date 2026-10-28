"""Tests of ``hdf5_dataset/new_pipeline/external_paths.py``: where the build finds the packages
that live outside the repository, and where they go on ``sys.path``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hdf5_dataset.new_pipeline import external_paths


def _package(root: Path, name: str) -> Path:
    """Make ``root/name/__init__.py`` and return ``root``."""
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No environment variable, no edited roots, one empty container, and a private sys.path."""
    monkeypatch.delenv(external_paths.ENV_VAR, raising=False)
    monkeypatch.setattr(external_paths, "EXTERNAL_ROOTS", [])
    monkeypatch.setattr(external_paths, "CONTAINERS", [tmp_path / "container"])
    monkeypatch.setattr(sys, "path", [str(external_paths.REPO_ROOT), "/elsewhere"])
    return tmp_path


def test_the_containers_are_searched_by_folder_name(isolated):
    early = _package(isolated / "container" / "EarlyMaestra", "early_maestra")
    mimo = _package(isolated / "container" / "MIMO_Sequence_Trainer", "mimo")
    assert external_paths.find_external_roots() == {"early_maestra": early, "mimo": mimo}


def test_the_environment_variable_wins_over_the_edited_roots_and_the_search(isolated, monkeypatch):
    _package(isolated / "container" / "EarlyMaestra", "early_maestra")
    edited = _package(isolated / "edited", "early_maestra")
    chosen = _package(isolated / "chosen", "early_maestra")
    monkeypatch.setattr(external_paths, "EXTERNAL_ROOTS", [str(edited)])
    monkeypatch.setenv(external_paths.ENV_VAR, str(chosen))
    assert external_paths.find_external_roots()["early_maestra"] == chosen


def test_an_edited_root_wins_over_the_search(isolated, monkeypatch):
    _package(isolated / "container" / "EarlyMaestra", "early_maestra")
    edited = _package(isolated / "edited", "early_maestra")
    monkeypatch.setattr(external_paths, "EXTERNAL_ROOTS", [str(edited)])
    assert external_paths.find_external_roots()["early_maestra"] == edited


def test_a_missing_package_is_left_out(isolated):
    assert external_paths.find_external_roots() == {}


def test_the_roots_go_just_after_the_repository_once(isolated):
    early = _package(isolated / "container" / "EarlyMaestra", "early_maestra")
    external_paths.add_external_roots()
    external_paths.add_external_roots()
    assert sys.path == [str(external_paths.REPO_ROOT), str(early.resolve()), "/elsewhere"]
