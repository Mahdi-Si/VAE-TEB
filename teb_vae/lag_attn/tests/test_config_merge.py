r"""The ``base:`` loader is what makes "a variant is a config, not a subclass" affordable.

If the merge is wrong the failure is silent and expensive: a variant that meant to override one
architecture flag instead resurrects a whole default block, the run trains a different model than
its config describes, and nothing raises. So the merge semantics are pinned here key by key --
dicts merge, lists and scalars replace -- along with the two properties the loader adds on top of
the copied merge: relative ``base:`` resolution and a cycle guard.
"""
from __future__ import annotations

import os

import pytest
import yaml

from teb_vae.lag_attn.config import load_config, resolve_config_file


def _write(path, mapping) -> str:
    """Write ``mapping`` as YAML to ``path`` and return the path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return str(path)


def test_a_config_without_base_resolves_to_itself(tmp_path):
    config = {"general_config": {"tag": "solo", "epochs": 5}}
    assert load_config(_write(tmp_path / "solo.yaml", config)) == config


def test_a_child_inherits_every_parent_key_it_does_not_name(tmp_path):
    _write(tmp_path / "parent.yaml", {"general_config": {"tag": "base", "epochs": 5000, "lr": 0.001}})
    child = _write(tmp_path / "child.yaml", {"base": "parent.yaml", "general_config": {"epochs": 1}})

    resolved = load_config(child)

    assert resolved["general_config"] == {"tag": "base", "epochs": 1, "lr": 0.001}


def test_nested_dicts_merge_key_by_key(tmp_path):
    """The property the whole mechanism rests on.

    A wholesale replace would mean a variant touching one ``VAE_model`` key silently rebuilds the
    default architecture for every other key -- the exact failure a config-driven variant exists to
    avoid.
    """
    _write(
        tmp_path / "parent.yaml",
        {"model_config": {"VAE_model": {"d_model": 128, "d_z": 24, "max_lag": 90}}},
    )
    child = _write(
        tmp_path / "child.yaml",
        {"base": "parent.yaml", "model_config": {"VAE_model": {"d_model": 32}}},
    )

    vae = load_config(child)["model_config"]["VAE_model"]

    assert vae == {"d_model": 32, "d_z": 24, "max_lag": 90}


def test_lists_and_scalars_replace_wholesale(tmp_path):
    """A list is a value here, not a container to extend.

    ``cuda_devices: [0]`` in a smoke variant must *replace* the prod box's seven devices. An
    extending merge would launch the smoke run across every GPU named by the parent.
    """
    _write(tmp_path / "parent.yaml", {"general_config": {"cuda_devices": [0, 1, 2, 3], "lr": 0.001}})
    child = _write(
        tmp_path / "child.yaml",
        {"base": "parent.yaml", "general_config": {"cuda_devices": [0], "lr": 0.1}},
    )

    general = load_config(child)["general_config"]

    assert general["cuda_devices"] == [0]
    assert general["lr"] == 0.1


def test_the_base_key_is_consumed_and_never_reaches_the_resolved_config(tmp_path):
    """``base`` is a loader directive.

    Left in, ``validate_config`` warns it is an unknown key and it lands in the MLflow param dump
    as a config value, which it is not.
    """
    _write(tmp_path / "parent.yaml", {"general_config": {"tag": "base"}})
    child = _write(tmp_path / "child.yaml", {"base": "parent.yaml"})

    assert "base" not in load_config(child)


def test_base_resolves_relative_to_the_child_not_the_working_directory(tmp_path, monkeypatch):
    """So a config tree can be moved, copied, or run from anywhere without rewriting its links."""
    _write(tmp_path / "configs" / "parent.yaml", {"general_config": {"tag": "base"}})
    child = _write(tmp_path / "configs" / "child.yaml", {"base": "parent.yaml"})

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_config(child)["general_config"]["tag"] == "base"


def test_a_chain_of_three_merges_in_order(tmp_path):
    """The nearest ancestor wins, and inheritance is transitive."""
    _write(tmp_path / "a.yaml", {"general_config": {"tag": "a", "epochs": 5000, "seed": 42}})
    _write(tmp_path / "b.yaml", {"base": "a.yaml", "general_config": {"tag": "b", "epochs": 100}})
    c = _write(tmp_path / "c.yaml", {"base": "b.yaml", "general_config": {"tag": "c"}})

    assert load_config(c)["general_config"] == {"tag": "c", "epochs": 100, "seed": 42}


def test_a_cycle_raises(tmp_path):
    _write(tmp_path / "a.yaml", {"base": "b.yaml"})
    _write(tmp_path / "b.yaml", {"base": "a.yaml"})

    with pytest.raises(ValueError, match="circular"):
        load_config(str(tmp_path / "a.yaml"))


def test_a_config_naming_itself_as_its_base_raises(tmp_path):
    """The one-element cycle; a `seen` set that appends too late would recurse forever here."""
    self_ref = _write(tmp_path / "self.yaml", {"base": "self.yaml"})

    with pytest.raises(ValueError, match="circular"):
        load_config(self_ref)


def test_a_missing_base_raises_file_not_found(tmp_path):
    child = _write(tmp_path / "child.yaml", {"base": "nope.yaml"})

    with pytest.raises(FileNotFoundError):
        load_config(child)


def test_a_non_string_base_raises(tmp_path):
    child = _write(tmp_path / "child.yaml", {"base": ["parent.yaml"]})

    with pytest.raises(ValueError, match="non-string"):
        load_config(child)


def test_a_non_mapping_document_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level YAML mapping"):
        load_config(str(path))


def test_the_parent_file_is_not_mutated_by_a_merge(tmp_path):
    """A shared nested dict would let one variant's resolution corrupt the next one's."""
    parent_path = tmp_path / "parent.yaml"
    parent = {"model_config": {"VAE_model": {"d_model": 128}}}
    _write(parent_path, parent)
    _write(
        tmp_path / "child.yaml",
        {"base": "parent.yaml", "model_config": {"VAE_model": {"d_model": 32}}},
    )

    load_config(str(tmp_path / "child.yaml"))

    assert load_config(str(parent_path))["model_config"]["VAE_model"]["d_model"] == 128


def test_resolve_config_file_writes_the_merged_config_and_returns_its_path(tmp_path):
    """The seam the experiment driver requires: it reads a path, not a dict."""
    _write(tmp_path / "parent.yaml", {"general_config": {"tag": "base", "epochs": 5000}})
    child = _write(tmp_path / "child.yaml", {"base": "parent.yaml", "general_config": {"epochs": 1}})

    written = resolve_config_file(child, str(tmp_path / "run"))

    assert os.path.isfile(written)
    assert load_config(written)["general_config"] == {"tag": "base", "epochs": 1}


def test_resolve_config_file_creates_a_missing_output_directory(tmp_path):
    config = _write(tmp_path / "solo.yaml", {"general_config": {"tag": "solo"}})

    written = resolve_config_file(config, str(tmp_path / "does" / "not" / "exist"))

    assert os.path.isfile(written)


def test_the_resolved_file_carries_no_base_key(tmp_path):
    """Otherwise the written provenance record re-inherits when it is re-read."""
    _write(tmp_path / "parent.yaml", {"general_config": {"tag": "base"}})
    child = _write(tmp_path / "child.yaml", {"base": "parent.yaml"})

    written = resolve_config_file(child, str(tmp_path / "run"))

    assert "base" not in yaml.safe_load(open(written, encoding="utf-8"))
