r"""What this cell's binding declares, and the two ways a declaration can be wrong in silence.

``TRF_CFS_BINDING`` is a set of *declarations* rather than of code, which is exactly why it is
proved this early: everything else in the causal-feature evaluation pipeline is validated against the
conv-LSTM cell, and a binding that names nothing is the one failure that would survive all of it.

**The interesting failure is not a wrong value; it is a key that names nothing.**
``preflight.reconcile`` compares ``model_config.VAE_model[key]`` against ``model_kwargs[key]`` and
**silently skips any key absent from either side**. So a ``geometry_keys`` entry that is not both a
constructor parameter *and* a config key is a reconciliation that never happens and never says so --
the run passes, the config and the checkpoint are free to disagree about that key, and the symptom
appears later as numbers computed at a geometry nobody chose. Both halves are therefore asserted
against the class and against this package's shipped ``configs/default.yaml``.

**The mirror-image failure is a key that names something here and nothing in this architecture.** The
five conv-LSTM-only keywords would each be dropped by the experiment driver's signature sweep without
a word, leaving a config that reads correct and builds a different model -- so they are asserted
absent from the tuple *and* from the override delta.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, Set

import pytest
import yaml

from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval.binding import GEOMETRY_KEYS as CFS_GEOMETRY_KEYS
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING, ModelBinding
from teb_vae.lag_attn_cfs.eval.config_schema import DEFAULT_OVERRIDES_PATH as CFS_OVERRIDES_PATH
from teb_vae.lag_attn_cfs.model_kwargs import WARMUP_MODEL_KWARGS
from teb_vae.lag_attn_transformer_cfs.eval import binding as binding_module
from teb_vae.lag_attn_transformer_cfs.eval.binding import (
    DEFAULT_OVERRIDES_PATH,
    GEOMETRY_KEYS,
    TRF_CFS_BINDING,
)
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask

from .conftest import CONV_LSTM_ONLY_KEYS, _REPO_ROOT, tiny_warmup_kwargs

#: The constructor keys this cell reconciles against a checkpoint, written out rather than imported:
#: importing :data:`GEOMETRY_KEYS` and comparing it to itself would pass on any edit. An **ordered**
#: sequence, not a set -- the order is what the reconciliation record is built in and what a reader of
#: two runs' preflight files compares down.
#:
#: Fifteen of the twenty-two are the conv-LSTM cell's, minus ``causal_norm``, which is not a keyword
#: of this constructor at all. The seven this architecture adds are its encoders'.
TRF_CFS_GEOMETRY_KEYS = (
    "sequence_length",
    "d_model",
    "d_z",
    "horizon",
    "raw_per_step",
    "warmup_period",
    "c_y",
    "c_u",
    "use_up_st",
    "max_lag",
    "num_heads",
    "d_head",
    "horizon_attention_blocks",
    "anchor_stride",
    "lag_floor",
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: This package's shipped training config, which the second half of the geometry-key rule reads.
DEFAULT_CONFIG_PATH = (
    Path(_REPO_ROOT) / "teb_vae" / "lag_attn_transformer_cfs" / "configs" / "default.yaml"
)


def _vae_config(path: Path) -> Dict[str, Any]:
    """Return a config's ``model_config.VAE_model`` block."""
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return (loaded.get("model_config") or {}).get("VAE_model") or {}


@pytest.fixture(scope="module")
def model() -> Any:
    """One tiny conv-Transformer cfs net, for the disclosure assertions."""
    import torch

    torch.manual_seed(0)
    return SeqVaeLagAttnTrfCfs(**tiny_warmup_kwargs())


# =================================================================================================
# The geometry keys
# =================================================================================================
def test_the_geometry_keys_are_exactly_the_twenty_two_declared_here() -> None:
    assert GEOMETRY_KEYS == TRF_CFS_GEOMETRY_KEYS
    assert TRF_CFS_BINDING.geometry_keys is GEOMETRY_KEYS
    assert len(GEOMETRY_KEYS) == 22
    assert len(set(GEOMETRY_KEYS)) == len(GEOMETRY_KEYS), "a duplicate would be compared twice"


def test_every_geometry_key_is_a_parameter_of_this_constructor() -> None:
    """A key this constructor does not take can never match and would refuse every run."""
    parameters = set(inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters)

    assert set(GEOMETRY_KEYS) <= parameters, sorted(set(GEOMETRY_KEYS) - parameters)


def test_every_geometry_key_is_also_a_key_of_this_packages_shipped_config() -> None:
    """The half that a signature check alone would miss. ``preflight.reconcile`` skips a key the
    config does not declare, so a key that is a constructor parameter and nothing else is a
    reconciliation that never happens and never says so."""
    declared = set(_vae_config(DEFAULT_CONFIG_PATH))

    assert set(GEOMETRY_KEYS) <= declared, sorted(set(GEOMETRY_KEYS) - declared)


def test_the_tuple_is_the_cfs_cells_minus_causal_norm_plus_this_architectures_seven() -> None:
    """Stated as the relation rather than as two lists, because the relation is what a reader has to
    check when either cell's tuple moves."""
    encoder_keys = {
        "encoder_conv_kernels",
        "encoder_conv_dilations",
        "encoder_num_heads",
        "encoder_d_ff",
        "target_attention_blocks",
        "source_attention_blocks",
        "source_attention_window",
    }

    assert set(GEOMETRY_KEYS) == (set(CFS_GEOMETRY_KEYS) - {"causal_norm"}) | encoder_keys
    # And the order of the shared fifteen is the conv-LSTM cell's, so the two reconciliation records
    # are read down the same columns.
    shared = [key for key in GEOMETRY_KEYS if key in set(CFS_GEOMETRY_KEYS)]
    assert shared == [key for key in CFS_GEOMETRY_KEYS if key != "causal_norm"]


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS)
def test_no_conv_lstm_only_key_reaches_this_binding_or_its_delta(key: str) -> None:
    """Each would be dropped by the experiment driver's signature sweep without a word, leaving a
    config that reads correct and builds a different model. ``causal_norm`` is the sharpest case: it
    is a ``TypeError`` at this constructor, so reconciling it could only ever compare a key against
    nothing."""
    assert key not in GEOMETRY_KEYS
    assert key not in inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters
    assert key not in _vae_config(DEFAULT_OVERRIDES_PATH)


def test_the_warm_up_budget_and_its_four_tuples_are_absent_and_checked_elsewhere() -> None:
    """The budget is a config key that names no constructor parameter; the four tuples are
    constructor parameters that name no config key. Listing either here would compare a value
    against nothing and pass every run, so ``preflight`` re-resolves the budget against the
    configured shards instead -- the only comparison that can actually fail."""
    assert "causal_warmup_budget_steps" not in GEOMETRY_KEYS
    for name in WARMUP_MODEL_KWARGS:
        assert name not in GEOMETRY_KEYS

    # Named rather than described, so this test breaks if the guard is renamed or removed.
    assert callable(preflight.check_warmup_budget_matches_checkpoint)
    assert "check_warmup_budget_matches_checkpoint" in preflight.GUARD_RECOVERY


# =================================================================================================
# The rest of the declaration
# =================================================================================================
def test_the_binding_names_this_cells_classes_and_tag() -> None:
    assert isinstance(TRF_CFS_BINDING, ModelBinding)
    assert TRF_CFS_BINDING.model_cls is SeqVaeLagAttnTrfCfs
    assert TRF_CFS_BINDING.task_cls is SeqVaeLagAttnTrfCfsTask
    assert TRF_CFS_BINDING.tag == "lag_attn_trf_cfs"
    assert TRF_CFS_BINDING.tag != CFS_BINDING.tag, (
        "two cells sharing a tag land their runs in one directory, told apart only by timestamp"
    )


def test_the_overrides_path_is_this_packages_own() -> None:
    """A binding pointing at another package's delta would evaluate the right checkpoint against the
    wrong holdout split."""
    assert TRF_CFS_BINDING.overrides_path == DEFAULT_OVERRIDES_PATH
    assert DEFAULT_OVERRIDES_PATH.is_file()
    assert DEFAULT_OVERRIDES_PATH.parent.parent.name == "eval"
    assert DEFAULT_OVERRIDES_PATH.parent.parent.parent.name == "lag_attn_transformer_cfs"


def test_the_override_delta_is_the_cfs_cells_key_for_key_and_value_for_value() -> None:
    """What makes the two cells comparable is that every shared measurement is configured
    identically: the same shards, seed, batch size, Monte Carlo draw count, thresholds and caps. This
    cell registers no analysis of its own, so unlike the raw pair's deltas there is not even a cap
    here the other file does not carry -- the two must be equal outright."""
    with open(DEFAULT_OVERRIDES_PATH, encoding="utf-8") as handle:
        here = yaml.safe_load(handle)
    with open(CFS_OVERRIDES_PATH, encoding="utf-8") as handle:
        there = yaml.safe_load(handle)

    assert here == there
    # Non-vacuity: both must actually carry the blocks a difference could hide in.
    assert set(here) >= {"general_config", "dataset_config", "eval_config"}


def test_the_extra_analyses_and_headline_scalars_are_the_parents_objects() -> None:
    """Identity, not equality. This cell adds no analysis -- the encoder replacement changes what
    produces the numbers, not which numbers there are -- so a second registry could only ever come to
    differ from the one the comparison model runs under."""
    assert TRF_CFS_BINDING.extra_analyses is CFS_BINDING.extra_analyses
    assert TRF_CFS_BINDING.headline_scalars is CFS_BINDING.headline_scalars
    # Non-vacuous now that the parent's registry is filled: identity between two empty objects
    # would be satisfied by two independent empty literals.
    assert set(TRF_CFS_BINDING.extra_analyses) == {
        "warmup", "source_null", "lag_clocks", "spectral_skill",
    }
    assert TRF_CFS_BINDING.headline_scalars != ()


def test_the_two_cells_ask_the_same_questions_in_the_same_order() -> None:
    """The registry the runner selects from, resolved through each binding. Identical key sets and
    identical order, because the encoder edge must not change *which* questions are asked -- a
    cross-cell table whose two sides ran different analyses is not a comparison, and a reordering
    would make reading two ``steps.json`` files side by side an exercise in re-sorting."""
    from teb_vae.lag_attn_cfs.eval.run import merged_analysis_functions

    here = merged_analysis_functions(TRF_CFS_BINDING)
    there = merged_analysis_functions(CFS_BINDING)

    assert list(here) == list(there)
    # And the same callables, not merely the same names: two registries agreeing on names while
    # disagreeing on implementations is the exact failure the shared registry exists to prevent.
    assert here == there
    # Non-vacuity: the four cfs-only analyses are in there, and ``cross_subgroup`` still trails
    # them, because it reads the per-recording CSVs the steps above it write.
    assert {"warmup", "source_null", "lag_clocks", "spectral_skill"} <= set(here)
    assert list(here)[-1] == "cross_subgroup"


def test_every_registered_headline_path_is_keyed_all_the_way_down() -> None:
    """The paths this binding contributes resolve by key at every step, as the shared registry's
    do. A path whose last step were a list index would resolve to the wrong row the day a metric
    was added above it, and nothing in the artifact would say so."""
    for name, path in TRF_CFS_BINDING.headline_scalars:
        assert isinstance(name, str) and name
        assert path and all(isinstance(step, str) and step for step in path), name


# =================================================================================================
# The encoder disclosure
# =================================================================================================
def test_the_disclosure_reports_a_structural_zero_and_names_what_proves_it(model) -> None:
    """Not ``causal_norm: true`` under another name. The conv-LSTM cell's key reports a *setting*;
    this reports that the setting has no counterpart because the modules it would govern do not exist
    on a history path -- and a structural claim is only as good as the check behind it."""
    record = TRF_CFS_BINDING.encoder_disclosure(model)

    assert record["time_pooling_normalisers"] == 0
    assert record["time_pooling_normalisers_are_structural"] is True
    assert record["time_pooling_normalisers_proved_by"] == binding_module.TIME_POOLING_PROOF
    assert "causal_norm" not in record


def test_the_named_proof_is_a_test_that_exists() -> None:
    """A record naming a check nobody wrote is worse than one naming none: it reads as evidence."""
    path, _, function = binding_module.TIME_POOLING_PROOF.partition("::")
    source = (Path(_REPO_ROOT) / path).read_text(encoding="utf-8")

    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function in defined


def test_the_disclosure_reports_this_encoders_geometry_against_the_lag_range(model) -> None:
    record = TRF_CFS_BINDING.encoder_disclosure(model)

    assert record["target_attention_blocks"] == int(model.target_encoder.num_attention_blocks)
    assert record["source_attention_blocks"] == int(model.source_encoder.num_attention_blocks)
    assert record["source_attention_window"] == int(model.source_encoder.attention_window)
    assert record["source_receptive_field_steps"] == int(model.source_encoder.receptive_field)
    assert record["source_receptive_field_seconds"] == record["source_receptive_field_steps"] * 4.0
    # The comparison is the point rather than either number: an encoder whose reach exceeded the lag
    # range would already be doing the alignment the lag cross-attention exists to do.
    assert "furthest searched lag" in record["source_reach_vs_lag_range"]
    assert record["source_reach_is_inside_the_lag_range"] is (
        int(model.source_encoder.receptive_field) < int(model.max_lag)
    )
    assert record["n_depthwise_init"] == int(model.n_depthwise_init)


def test_an_unbounded_source_encoder_says_so_rather_than_reporting_the_sequence_length() -> None:
    """"No bound" and "a bound that happens to equal $T$" are different statements, and the arm that
    removes the window is a real one."""
    import torch

    torch.manual_seed(0)
    unbounded = SeqVaeLagAttnTrfCfs(**tiny_warmup_kwargs(source_attention_window=None))

    record = TRF_CFS_BINDING.encoder_disclosure(unbounded)

    assert record["source_attention_window"] is None
    assert record["source_receptive_field_steps"] is None
    assert record["source_receptive_field_seconds"] is None
    assert "no window" in record["source_reach_vs_lag_range"]
    assert "source_reach_is_inside_the_lag_range" not in record


def test_the_disclosure_returns_no_key_the_shared_causality_record_owns(model) -> None:
    """The shared half is not overridable, because a reader compares the two cfs models' records down
    those key names -- and on this cell the shared half includes the target domain's own geometry,
    which both encoders share."""
    record: Set[str] = set(TRF_CFS_BINDING.encoder_disclosure(model))

    assert record & preflight.SHARED_CAUSALITY_KEYS == set()


def test_the_disclosure_refuses_a_model_it_cannot_read_naming_both() -> None:
    """Through the same reader the conv-LSTM cell's disclosure uses, so the two cannot come to word
    the refusal differently."""

    class _Renamed:
        n_depthwise_init = 0

    with pytest.raises(AttributeError) as excinfo:
        TRF_CFS_BINDING.encoder_disclosure(_Renamed())

    message = str(excinfo.value)
    assert "target_encoder" in message
    assert "_Renamed" in message
