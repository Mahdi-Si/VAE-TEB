r"""This package's override delta: what it changes, and what it deliberately does not.

The delta is not a config and does not stand alone. Its contract is what it *becomes* when
deep-merged over the ``resolved_config.yaml`` a training run wrote beside its checkpoints: the
run's geometry, normalisation and objective survive untouched, and the merged ``eval_config``
validates. The merge itself and the loader's refusals are the shared package's, and are tested
there.

There is a second contract here the sibling's file does not have, and it is the more important
one. **This delta must equal the sibling's.** The two models exist to be compared, so a holdout
split, a Monte Carlo draw count or a bootstrap resample count that differed between them would
make every side-by-side number a comparison of two protocols rather than of two architectures --
and nothing in either run's artifacts would say so. The equality is asserted against the sibling's
committed file rather than against a copy of its values, so a change on either side fails here.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.eval import run as shared_run
from teb_vae.lag_attn_rws.eval.config_schema import (
    DEFAULT_OVERRIDES_PATH as SIBLING_OVERRIDES_PATH,
    load_eval_overrides,
    merge_eval_overrides,
    validate_eval_config,
)
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def overrides() -> dict:
    return load_eval_overrides(TRF_BINDING.overrides_path)


@pytest.fixture(scope="module")
def sibling_overrides() -> dict:
    return load_eval_overrides(SIBLING_OVERRIDES_PATH)


@pytest.fixture(scope="module")
def resolved() -> dict:
    """Stands in for a checkpoint's ``resolved_config.yaml``: fully explicit, no ``base:``."""
    return load_config(str(_DEFAULT_CONFIG))


@pytest.fixture(scope="module")
def merged(resolved) -> dict:
    return merge_eval_overrides(resolved, TRF_BINDING.overrides_path)


# =============================================================================
# Equality with the sibling's delta
# =============================================================================
#: The only keys this delta may carry that the sibling's does not: a retention cap named for an
#: analysis **only this model has**. Derived from the binding rather than written out, so the
#: exemption cannot quietly grow to cover a setting the two models share.
LOCAL_CAP_KEYS = frozenset(TRF_BINDING.extra_analyses)


def _without_local_caps(eval_config: dict) -> Tuple[dict, dict]:
    """Split an ``eval_config`` into (everything the sibling also configures, this model's caps)."""
    shared = copy.deepcopy(eval_config)
    caps = dict(shared.get("caps") or {})
    local = {name: caps.pop(name) for name in sorted(LOCAL_CAP_KEYS) if name in caps}
    shared["caps"] = caps
    return shared, local


def test_the_two_deltas_differ_only_in_their_comments_and_this_models_own_caps(
    overrides, sibling_overrides
) -> None:
    """Parsed, the two documents are equal once this model's own analysis caps are set aside.
    Whole-document, so a divergence anywhere -- the seed, the draw count, a shared cap, a verdict
    threshold, a shard, a loader field, a batch size -- fails here.

    The carve-out is only for analyses the sibling cannot have: a shared analysis name in it would
    exempt a cap that moves a number the cross-model table compares."""
    assert not (LOCAL_CAP_KEYS & set(shared_run.ANALYSIS_FUNCTIONS))

    mine, theirs = copy.deepcopy(overrides), copy.deepcopy(sibling_overrides)
    mine["eval_config"], _local = _without_local_caps(mine["eval_config"])
    theirs["eval_config"], sibling_local = _without_local_caps(theirs["eval_config"])

    assert mine == theirs
    assert sibling_local == {}, "the sibling has none of this model's analyses to cap"


# =============================================================================
# The merge
# =============================================================================
def test_the_resolved_eval_config_validates(merged) -> None:
    """The validator is what a run calls before it loads a checkpoint, so a misspelled key here
    must cost a parse rather than a model load and a first pass over the shards."""
    assert validate_eval_config(merged)["seed"] == merged["eval_config"]["seed"]


def test_the_runs_own_contract_survives_the_merge(merged, resolved) -> None:
    """The geometry, the normalisation statistics and the objective are what the run trained
    under; the delta must touch none of them."""
    assert merged["model_config"] == resolved["model_config"]
    assert merged["dataset_config"]["stat_path"] == resolved["dataset_config"]["stat_path"]
    assert merged["dataset_config"]["vae_train_datasets"] == (
        resolved["dataset_config"]["vae_train_datasets"]
    )
