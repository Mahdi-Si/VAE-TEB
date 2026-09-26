r"""Conformance of the shared clinical labelling against *this* package's configured shards.

The labelling itself is the sibling's and is tested there; it is imported rather than rewritten.
What is checked here is the one part that could differ between the two packages and would fail
silently if it did: ``CANONICAL_SUBGROUPS`` is a hardcoded eight-name table, and this package's
holdout shards are named independently of it. If the two ever disagree, every sample carries no
subgroup, every by-subgroup table is empty, and nothing raises -- an unrecognised basename is
legitimate on the pretraining split, so it can only warn.
"""
from __future__ import annotations

from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.config_schema import load_eval_overrides


def test_the_eight_canonical_stems_resolve_against_the_configured_shards() -> None:
    """Both directions: every configured shard has a subgroup, and all eight are covered."""
    shards = load_eval_overrides()["dataset_config"]["vae_test_datasets"]
    resolved = [labels.subgroup_of(path) for path in shards]

    assert None not in resolved, f"a configured shard resolved to no subgroup: {shards}"
    assert set(resolved) == set(labels.CANONICAL_SUBGROUPS)
    assert len(resolved) == len(labels.CANONICAL_SUBGROUPS)
