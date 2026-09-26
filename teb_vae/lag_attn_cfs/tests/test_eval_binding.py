r"""What this cell's binding declares, and the ways a declaration can be wrong in silence.

``CFS_BINDING`` is what every call site of this pipeline gets by omission, so an edit to it
changes what a run *means*.

**The interesting failure is not a wrong value; it is a key that names nothing.**
``preflight.reconcile`` compares ``model_config.VAE_model[key]`` against ``model_kwargs[key]`` and
**silently skips any key absent from either side**. So a ``geometry_keys`` entry that is not both a
constructor parameter *and* a config key is a reconciliation that never happens and never says so.
Both halves are therefore asserted against the class and against the shipped
``configs/default.yaml`` rather than against a second hand-kept list.

The rest is the merge the binding feeds: the cell's extra analyses run before the one that reads
their tables and may not take a shared name, an exclusion is by name and refuses a name that
matches nothing, and the cell's headline scalars resolve against the blocks their analyses return
and are appended after the shared block rather than mixed into it.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from teb_vae.lag_attn_cfs.eval import preflight, report_seam
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING, GEOMETRY_KEYS, ModelBinding
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

from .conftest import _REPO_ROOT


@pytest.fixture(scope="module")
def shipped_vae_config() -> Dict[str, Any]:
    """The shipped ``model_config.VAE_model`` block, read off the committed file."""
    shipped = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_cfs" / "configs" / "default.yaml"
    return yaml.safe_load(shipped.read_text(encoding="utf-8"))["model_config"]["VAE_model"]


@pytest.fixture(scope="module")
def constructor_parameters() -> frozenset:
    """Every keyword this cell's constructor accepts."""
    return frozenset(inspect.signature(SeqVaeLagAttnCfs.__init__).parameters)


# =================================================================================================
# The binding's fields
# =================================================================================================
def test_every_geometry_key_is_a_parameter_of_this_constructor(constructor_parameters) -> None:
    """A key the constructor does not accept can never match a stamped ``model_kwargs`` entry."""
    unknown = [key for key in GEOMETRY_KEYS if key not in constructor_parameters]

    assert unknown == [], (
        f"{unknown} are not parameters of SeqVaeLagAttnCfs.__init__, so they cannot appear in a "
        f"checkpoint's model_kwargs and the reconciliation would skip them forever"
    )


def test_every_geometry_key_is_also_a_shipped_config_key(shipped_vae_config) -> None:
    """The half that a constructor check alone would miss, and the one that fails silently: the
    reconciliation skips a key absent from *either* side, so a constructor parameter that no config
    names is a key that could only ever be skipped."""
    unconfigured = [key for key in GEOMETRY_KEYS if key not in shipped_vae_config]

    assert unconfigured == [], (
        f"{unconfigured} are not keys of configs/default.yaml's model_config.VAE_model block, so "
        f"reconcile() would skip them on every run and the config could contradict the checkpoint "
        f"about them without a word"
    )


def test_the_encoder_disclosure_is_this_cells_and_refuses_a_model_it_cannot_read() -> None:
    """Wired to ``preflight.cfs_encoder_disclosure``, which reports the recurrent encoder's own
    guard and nothing that belongs to the target domain -- the one-sidedness, the group delays, the
    warm-up budget, the anchor geometry and the lag support are shared by both cfs cells and are
    owned by the shared half of the causality record, so both models report them down one set of key
    names.

    An object carrying neither attribute raises naming the class rather than disclosing an empty
    block: a causality record that was complete-looking and empty is the one outcome worse than no
    record at all."""
    assert CFS_BINDING.encoder_disclosure is preflight.cfs_encoder_disclosure

    with pytest.raises(AttributeError, match="causal_norm"):
        CFS_BINDING.encoder_disclosure(object())


def test_the_extras_are_selectable_and_run_before_the_analysis_that_reads_their_tables() -> None:
    """Two properties in one registry, and the second is why the merge is not a plain update.

    ``cross_subgroup`` reads per-recording CSVs off disk and its source table names three of this
    cell's own analyses, so an extra appended *after* it would be tested on every full run and
    found absent every time -- recorded as a partial directory rather than as a run order that
    cannot work.
    """
    registry = run_module.merged_analysis_functions(CFS_BINDING)

    extras = list(CFS_BINDING.extra_analyses)
    assert extras and set(extras) <= set(registry)
    assert list(registry)[-1] == "cross_subgroup"
    positions = {name: index for index, name in enumerate(registry)}
    for name in extras:
        assert positions[name] < positions["cross_subgroup"], name
    # And the shared order above them is untouched, so two cells' summaries still line up.
    shared = [name for name in registry if name in run_module.ANALYSIS_FUNCTIONS]
    assert shared == list(run_module.ANALYSIS_FUNCTIONS)


def test_an_extra_analysis_may_not_take_a_shared_name() -> None:
    """An extra is an addition, never an override: silently replacing a shared implementation
    would leave two models reporting different things under one name."""
    shared_name = next(iter(run_module.ANALYSIS_FUNCTIONS))
    clashing = ModelBinding(
        model_cls=CFS_BINDING.model_cls,
        task_cls=CFS_BINDING.task_cls,
        tag=CFS_BINDING.tag,
        geometry_keys=CFS_BINDING.geometry_keys,
        encoder_disclosure=CFS_BINDING.encoder_disclosure,
        overrides_path=CFS_BINDING.overrides_path,
        extra_analyses={shared_name: lambda *args, **kwargs: {}},
    )

    with pytest.raises(ValueError, match=shared_name):
        run_module.merged_analysis_functions(clashing)


def test_an_excluded_analysis_leaves_the_rest_of_the_registry_in_order() -> None:
    """The removal is by name and nothing else moves, which is what keeps two cells' summaries
    readable side by side: one has a column fewer, in the same order, rather than a reordering."""
    removed = next(iter(run_module.ANALYSIS_FUNCTIONS))
    narrowed = dataclasses.replace(CFS_BINDING, excluded_analyses=(removed,))

    full = list(run_module.merged_analysis_functions(CFS_BINDING))
    reduced = list(run_module.merged_analysis_functions(narrowed))

    assert removed in full and removed not in reduced
    assert reduced == [name for name in full if name != removed]


def test_an_exclusion_naming_nothing_refuses_rather_than_doing_nothing() -> None:
    """A misspelt exclusion is the failure worth catching: the analysis still runs, its columns
    still appear, and the binding -- and every summary written from it -- says it was removed."""
    narrowed = dataclasses.replace(
        CFS_BINDING, excluded_analyses=("attentoin",)
    )
    with pytest.raises(ValueError, match="attentoin"):
        run_module.merged_analysis_functions(narrowed)


def test_every_registered_headline_path_is_keyed_all_the_way_down() -> None:
    """A path whose last step is a list index resolves to the wrong row the day a metric is added
    above it, and nothing in the artifact would say so. Each of the three analyses assembles a flat
    block of scalars for exactly this reason."""
    for name, path in CFS_BINDING.headline_scalars:
        assert path, name
        assert all(isinstance(step, str) for step in path), (name, path)
        assert path[0] in CFS_BINDING.extra_analyses, (name, path)


def test_every_registered_headline_path_resolves_against_the_blocks_the_analyses_return() -> None:
    """Verifiable before any run exists, on a stub shaped like what the three analyses return. The
    end-to-end fixture asserts the same paths against a real run's results, which is where a path
    that resolves on a stub and not in reality would fail."""
    results = {
        "warmup": {
            "headline": {
                "pred_gap_warm_lo_nats": 0.1,
                "pred_gap_warm_mid_nats": 0.2,
                "pred_gap_warm_hi_nats": 0.3,
                "source_lag_warmth_frac_st": 0.4,
                "source_lag_warmth_frac_ph": 0.05,
            },
            "geometry_guards": {"anchors_per_sample": 152.0, "target_warm_frac": 1.0},
        },
        "source_null": {
            "difference": {
                "kld_source_null_nats": 1.25,
                "coupling_minus_clock_nats": 1.75,
                "ci_lo": 1.1,
                "ci_hi": 2.4,
            },
            # The same difference resolved by lag. ``clock_excess_degenerate`` is a BOOL and the
            # stub carries it as one deliberately: the headline finiteness check exempts bools
            # explicitly, so a builder that coerced it to a float would turn "this profile has no
            # readable shape" into a 0.0 that reads as a measured share.
            "lag": {
                "clock_excess_argmax_lag_step": 33,
                "clock_excess_peak_share": 0.42,
                "clock_excess_degenerate": False,
                "clock_excess_rectified_frac": 0.07,
            },
        },
        "spectral_skill": {
            "headline": {
                f"pred_gap_{band}_nats": 0.01
                for band in ("slow_baseline", "deceleration", "variability",
                             "beat_to_beat", "unknown")
            }
        },
        # The interventional readout's block. Its first entry is a band NAME rather than a number,
        # which the stub carries deliberately: the headline builder must pass a string through
        # untouched, and a builder that coerced every value to a float would turn the one entry
        # saying WHICH lag range mattered into a NaN and resolve it as absent.
        "occlusion": {
            "headline": {
                "band": "near",
                "delta_total_nats": 14.94,
                "peak_horizon_step": 5,
                "live_fraction": 1.0,
                "n_bands": 4,
            }
        },
        # The high-KL selection's block, shaped as ``lag_high_kl.headline_block`` returns it.
        "lag_high_kl": {
            "headline": {
                "high_kl_threshold_nats": 0.92,
                "high_kl_centroid_kl_s": 118.0,
                "high_kl_total_nats": 1.3,
                "hot_lag_count": 27,
                "hot_lag_share_kl": 0.55,
                "high_kl_pred_gap_nats": 0.8,
                "high_minus_rest_pred_gap_nats": 0.4,
                "high_gain_overlap_share": 0.61,
            }
        },
        "verdicts": [],
    }

    headline = report_seam.build_headline(results, CFS_BINDING.headline_scalars)

    unresolved = sorted(
        name for name, _ in CFS_BINDING.headline_scalars if headline.get(name) is None
    )
    assert unresolved == []
    assert headline["coupling_minus_clock_nats"] == 1.75
    assert headline["anchors_per_sample"] == 152.0
    assert headline["occlusion_peak_band"] == "near"
    assert headline["clock_excess_argmax_lag_step"] == 33
    # Passed through as a bool rather than coerced: the finiteness check exempts bools, and a
    # degenerate profile reported as 0.0 would read as a share that was measured.
    assert headline["clock_excess_degenerate"] is False


def test_the_headline_block_is_unchanged_when_a_binding_registers_nothing() -> None:
    """The neutrality claim behind the field: the extras are appended, so a binding that adds none
    produces the block this pipeline produces without them -- key for key and in the same order.

    Checked against an empty tuple rather than against this cell's binding, which registers its
    own: what is being asserted is that the *mechanism* adds nothing of its own, and the test below
    asserts that this cell's scalars are appended after the shared block rather than mixed into it.
    """
    results = {"readouts": {"mc_pred_gap": 1.5}, "verdicts": []}

    without = report_seam.build_headline(results)
    with_empty = report_seam.build_headline(results, ())

    assert with_empty == without
    assert list(with_empty) == list(without)
    # And not vacuous: a registered entry does reach the block.
    extended = report_seam.build_headline(results, (("added", ("readouts", "mc_pred_gap")),))
    assert extended["added"] == 1.5


def test_this_cells_scalars_are_appended_after_the_shared_block_rather_than_mixed_in() -> None:
    """The shared block's key order is what two cells' summaries are diffed down, so an extra that
    landed inside it would shift every row below it in a comparison that is read by position."""
    results = {"readouts": {"mc_pred_gap": 1.5}, "verdicts": []}

    headline = list(report_seam.build_headline(results, CFS_BINDING.headline_scalars))
    shared = [name for name, _ in report_seam.HEADLINE_SCALARS]
    extras = [name for name, _ in CFS_BINDING.headline_scalars]

    assert len(set(extras)) == len(extras), "a duplicated name would silently shadow itself"
    assert headline[: len(shared)] == shared
    assert headline[len(shared) : len(shared) + len(extras)] == extras


def test_an_extra_headline_scalar_may_not_take_a_shared_name() -> None:
    """The extras resolve last, so a reused name would replace a shared reading with a
    cell-specific one under the shared name -- and every arm table, the acceptance gate and every
    cross-cell row reads this block *by name*, so the substitution would be invisible in the
    artifact."""
    shared_name = report_seam.HEADLINE_SCALARS[0][0]

    with pytest.raises(ValueError, match=shared_name):
        report_seam.build_headline({}, ((shared_name, ("anything", "at", "all")),))
