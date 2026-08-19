r"""The warm-up budget resolves from the shards, and every way of getting it wrong is refused.

The boundary this module resolves is the one thing about a causal dataset that cannot be inferred
from the data it serves. The coefficients inside $[0, W'_c)$ are **real float values** -- not
zeroed, not NaN -- because the writer attaches the boundary as an attribute and leaves the array
untouched; and the normalisation constants were accumulated *excluding* exactly that region, so
those values are on no defined scale. A consumer that ignores the attribute therefore trains on
pad, and nothing anywhere raises.

So the tests below are mostly refusals. Each one is a configuration or a shard that would resolve
cleanly against a weaker check and score pad, and each asserts the message names the value the
operator has to change. Where a refusal comes from mutating the committed fixture, the unmutated
copy is resolved first, so the refusal is attributable to the mutation and not to the copy.

The expected channel counts are **derived from the shard's own stored attributes**, not written
out, with one exception: the shipped row is pinned as literals, because that row is the
configuration every number this family reports is produced at and a silent change to it must fail
here rather than move a training curve.
"""
from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from hdf5_dataset.hdf5_dataset import decimated_trim_steps
from teb_vae.lag_attn_cfs.causal_warmup import (
    SOURCE_BLOCKS,
    TARGET_BLOCKS,
    StreamWarmup,
    resolve_warmup_budget,
)
from teb_vae.lag_attn_cfs.tests.conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_SHARD,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_TRIM_MINUTES,
    SHIPPED_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    causal_config,
    stored_warmup,
    without_key,
    write_variant,
)

#: What the shipped configuration must resolve to, as literals. $98$ of $102$ target channels, the
#: four dropped ones being the ``fhr_st`` scattering channels below roughly $0.008$ Hz -- the same
#: band floor both phase selections already use, which is why the budget lands on a frequency
#: boundary the pipeline independently believes in rather than on an arbitrary cut.
SHIPPED_TARGET_KEPT = 98
SHIPPED_DROPPED_WARMUP = (162, 194, 233, 278)


def rebased(path: Path = CAUSAL_SHARD) -> Dict[str, np.ndarray]:
    r"""The shard's stored warm-up in the trimmed window's coordinates.

    Computed here from the raw attribute rather than taken from the resolver, so the resolver is
    checked against the file instead of against itself.

    Args:
        path: The shard to read.

    Returns:
        ``{block: (C,) int64}`` of $W' = \max(W - \mathrm{trim}, 0)$.
    """
    _, trim = decimated_trim_steps(SHIPPED_TRIM_MINUTES)
    return {name: np.maximum(vector - trim, 0) for name, vector in stored_warmup(path).items()}


def declared_vector(blocks: Tuple[str, ...]) -> np.ndarray:
    """The rebased warm-up of one stream, blocks concatenated in the model's order."""
    values = rebased()
    return np.concatenate([values[name] for name in blocks])


# =================================================================================================
# What the budget resolves to
# =================================================================================================
def test_no_budget_resolves_to_none() -> None:
    """A two-sided run configures no warm-up guard, and must get no gate rather than an empty one."""
    assert resolve_warmup_budget(causal_config(causal_warmup_budget_steps=None)) is None
    assert resolve_warmup_budget(without_key(causal_config(), "causal_warmup_budget_steps")) is None


def test_the_shipped_budget_keeps_ninety_eight_of_one_hundred_and_two(budget) -> None:
    """The shipped row, as literals: every phase channel survives and four scattering ones do not."""
    assert budget.budget_steps == SHIPPED_BUDGET_STEPS
    assert budget.target.declared_width == CAUSAL_C_Y
    assert budget.target.kept_width == SHIPPED_TARGET_KEPT

    dropped = budget.target.dropped_index
    declared = budget.target.declared_warmup_steps
    assert tuple(declared[index] for index in dropped) == SHIPPED_DROPPED_WARMUP

    # All four come from the scattering block, so the phase block is whole -- which is the property
    # the threshold was chosen for, and the one a moved boundary would break first.
    st_start, st_stop = next(
        (start, stop) for name, start, stop in budget.target.block_spans if name == "fhr_st"
    )
    assert all(st_start <= index < st_stop for index in dropped)
    assert budget.target.block_counts() == (("fhr_st", 32, 36), ("fhr_ph", 66, 66))


def test_the_kept_set_is_exactly_the_channels_at_or_below_the_threshold(budget) -> None:
    """Derived from the shard's own attribute, so a rebuilt fixture fails rather than passes."""
    expected = np.flatnonzero(declared_vector(TARGET_BLOCKS) <= SHIPPED_BUDGET_STEPS)
    assert budget.target.keep_index == tuple(int(index) for index in expected)
    assert budget.target.declared_warmup_steps == tuple(
        int(step) for step in declared_vector(TARGET_BLOCKS)
    )


@pytest.mark.parametrize("threshold", (31, 92, 112, 134, 151, 162, 233, 278))
def test_the_budget_walks_the_channel_staircase(threshold: int) -> None:
    """Shortening the wait buys channels, one staircase step at a time.

    The whole tradeoff of this family in one assertion: the threshold is a *choice* over a measured
    staircase, not a constant, and every value of it must select the channels at or below it and
    nothing else. The floor is held at the shipped value throughout -- the pairing that ties them
    is the constructor's, not this module's, so varying one here isolates the other.
    """
    resolved = resolve_warmup_budget(causal_config(causal_warmup_budget_steps=threshold))
    assert resolved is not None
    expected = np.flatnonzero(declared_vector(TARGET_BLOCKS) <= threshold)
    assert resolved.target.keep_index == tuple(int(index) for index in expected)
    assert resolved.target.kept_width == int(expected.size)


@pytest.mark.parametrize("threshold", (1, 41, 134, 278, 10_000))
def test_the_source_is_never_gated(threshold: int) -> None:
    """No threshold shortens the source keep-index, because it is the identity by construction.

    The source's slowest channels carry the contraction envelope and reach $W' = 278$: their
    availability row is zero for $278$ of $300$ steps. Gating them would buy warm lags at the price
    of the signal the lag search exists to find, so the design keeps them and announces their
    arrival instead. A budget that started trimming the source when it moved would remove that
    silently.
    """
    resolved = resolve_warmup_budget(causal_config(causal_warmup_budget_steps=threshold))
    assert resolved is not None
    assert resolved.source.keep_index == tuple(range(CAUSAL_C_U))
    assert resolved.source.kept_width == CAUSAL_C_U
    assert resolved.source.dropped_index == ()
    assert resolved.source.warmup_steps == tuple(
        int(step) for step in declared_vector(SOURCE_BLOCKS)
    )


def test_dropping_the_source_scattering_block_narrows_the_declared_stream() -> None:
    r"""``use_up_st: false`` leaves ``up_ph`` alone, whose fastest channel already waits $41$ steps.

    Worth its own case because it is the one configuration in which the source stream has a
    non-zero minimum warm-up, and therefore the one in which the model builds a start indicator at
    all.
    """
    resolved = resolve_warmup_budget(
        causal_config(use_up_st=False, c_u=15, causal_warmup_budget_steps=SHIPPED_BUDGET_STEPS)
    )
    assert resolved is not None
    assert resolved.source.block_spans == (("up_ph", 0, 15),)
    assert min(resolved.source.warmup_steps) == int(rebased()["up_ph"].min())
    assert min(resolved.target.warmup_steps) == 0


def test_the_keep_index_is_ascending_and_the_warm_up_is_positional_against_it(budget) -> None:
    """``ChannelGate``'s gather requires the first; the adapter's mask requires the second."""
    for stream in (budget.target, budget.source):
        assert list(stream.keep_index) == sorted(set(stream.keep_index)), stream.name
        assert len(stream.warmup_steps) == stream.kept_width, stream.name
        assert stream.warmup_steps == tuple(
            stream.declared_warmup_steps[index] for index in stream.keep_index
        ), stream.name
        assert set(stream.keep_index) | set(stream.dropped_index) == set(
            range(stream.declared_width)
        ), stream.name


def test_the_derivable_quantities_are_properties_rather_than_fields() -> None:
    """Stored, they would be a second source of truth that could disagree with the keep-index."""
    stored_fields = {field.name for field in fields(StreamWarmup)}
    assert stored_fields == {"name", "block_spans", "declared_warmup_steps", "keep_index"}
    for derived in ("kept_width", "dropped_index", "warmup_steps", "declared_width", "max_warmup"):
        assert isinstance(getattr(StreamWarmup, derived), property), derived


def test_the_slowest_survivor_is_what_the_anchor_floor_must_clear(budget) -> None:
    r"""$B = \max_{c \in \mathrm{kept}} W'_c$, which the shipped threshold happens to sit exactly on.

    "Happens to" is the point: at a threshold of $151$ the same $98$ channels survive and $B$ is
    still $134$, so a floor derived from the *threshold* would be $17$ steps too high and would cost
    two tiles for nothing.
    """
    assert budget.target.max_warmup == max(budget.target.warmup_steps)
    assert budget.target.max_warmup == SHIPPED_BUDGET_STEPS
    assert SHIPPED_WARMUP_PERIOD >= budget.target.max_warmup - 1

    loose = resolve_warmup_budget(causal_config(causal_warmup_budget_steps=151))
    assert loose is not None
    assert loose.target.kept_width == SHIPPED_TARGET_KEPT
    assert loose.target.max_warmup == SHIPPED_BUDGET_STEPS


def test_the_summary_names_both_streams_and_the_threshold(budget) -> None:
    """It is the startup log's only statement of what this run is about to read."""
    summary = budget.summary()
    assert f"{SHIPPED_BUDGET_STEPS} steps" in summary
    assert "fhr_st 32/36" in summary and "fhr_ph 66/66" in summary
    assert "up_st 36/36" in summary and "up_ph 15/15" in summary


def test_the_resolver_reads_no_transform_code() -> None:
    """The two-sided reach guard measures the wrong bank and would carry kymatio into every process.

    Asserted against the module's import statements rather than against ``sys.modules``, which
    another suite in the same session may have populated for its own reasons.
    """
    from teb_vae.lag_attn_cfs import causal_warmup

    source = Path(causal_warmup.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for forbidden in ("kymatio", "teb_vae.lag_attn.channel_reach"):
        assert not any(name.startswith(forbidden) for name in imported), forbidden
    assert not any("scattering" in name for name in imported)


# =================================================================================================
# Refusals: the configuration
# =================================================================================================
def test_a_reach_budget_beside_a_warm_up_budget_is_refused_naming_both_keys() -> None:
    """The forward reach is an energy quantile of a two-sided filter and is undefined here.

    It would not merely be ignored: it resolves *delays*, and a delay is a shift, so the model
    would read every gated channel late on top of its warm-up. The failure is loud only because the
    causal widths happen to disagree with the two-sided bank's; at widths that lined up it would be
    silent.
    """
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(causal_reach_budget_s=120.0))
    message = str(error.value)
    assert "causal_warmup_budget_steps" in message
    assert "causal_reach_budget_s" in message


@pytest.mark.parametrize("trim", (2.0, 0.5, None))
def test_a_trim_that_is_not_the_loader_s_is_refused_naming_both_config_paths(trim) -> None:
    """The one failure mode no warm-fraction readout can see.

    A uniformly wrong rebase moves the anchor floor and the validity boundary by the same amount,
    so "is every scored target step warm" still reports exactly $1.0$ while the model scores pad.
    The cross-check is against the other declaration of the same geometry: the sequence length the
    network is built at is the stored length minus twice the trim.
    """
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(trim_minutes=trim))
    message = str(error.value)
    assert "dataset_config.dataloader_config.dataset_kwargs.trim_minutes" in message
    assert "model_config.VAE_model.sequence_length" in message


def test_the_matching_trim_is_accepted(config) -> None:
    """The negative control for the three refusals above: only the trim moved in them."""
    resolved = resolve_warmup_budget(config)
    assert resolved is not None
    assert resolved.trim_minutes == SHIPPED_TRIM_MINUTES


@pytest.mark.parametrize(
    "key", ("sequence_length", "horizon", "warmup_period", "anchor_stride", "c_y", "c_u")
)
def test_a_missing_geometry_key_is_refused_naming_it(config, key: str) -> None:
    """No defaults: a default here is a second declaration of the geometry that can disagree."""
    for variant in (without_key(config, key), causal_config(**{key: None})):
        with pytest.raises(ValueError, match=key):
            resolve_warmup_budget(variant)


def test_a_declared_width_that_disagrees_with_the_shards_is_refused() -> None:
    """The keep-index is positional, so a wrong width gathers the wrong channels rather than fails."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(c_y=109))
    assert "c_y=109" in str(error.value)
    assert str(CAUSAL_C_Y) in str(error.value)

    with pytest.raises(ValueError, match="c_u=58"):
        resolve_warmup_budget(causal_config(c_u=58))


def test_a_budget_that_keeps_no_channel_is_refused_naming_it() -> None:
    """A stream with zero channels builds a model that trains to completion having never read it."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(causal_warmup_budget_steps=-1))
    assert "causal_warmup_budget_steps" in str(error.value)
    assert "-1" in str(error.value)


def test_a_floor_and_stride_leaving_a_phase_with_no_anchor_are_refused() -> None:
    r"""At phase $\varphi = S - 1$ the first anchor is $F + S - 1$; it has to exist.

    Otherwise a sample drawn at that phase contributes no forecast at all and its share of the
    epoch disappears with nothing reporting it.
    """
    t_valid = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(
            causal_config(warmup_period=t_valid - SHIPPED_HORIZON + 1, causal_warmup_budget_steps=278)
        )
    message = str(error.value)
    assert "warmup_period" in message and "anchor_stride" in message
    assert f"T_valid={t_valid}" in message

    # One step lower is the last floor that works, so the boundary is where it is claimed to be.
    resolved = resolve_warmup_budget(
        causal_config(warmup_period=t_valid - SHIPPED_HORIZON, causal_warmup_budget_steps=278)
    )
    assert resolved is not None

    # A stride of one restores every floor below T_valid: the dense range is one tile at any phase.
    dense = resolve_warmup_budget(
        causal_config(
            warmup_period=t_valid - 1, anchor_stride=1, causal_warmup_budget_steps=278
        )
    )
    assert dense is not None


def test_a_config_naming_no_shards_is_refused() -> None:
    """The boundary is a property of the data; a config that names none has nothing to read it from."""
    config = causal_config()
    config["dataset_config"]["vae_train_datasets"] = []
    config["dataset_config"]["vae_test_datasets"] = []
    with pytest.raises(ValueError, match="vae_train_datasets"):
        resolve_warmup_budget(config)


def test_a_missing_shard_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """A boundary resolved from the shards that happen to exist is the wrong boundary."""
    absent = tmp_path / "not_built_yet.hdf5"
    with pytest.raises(ValueError, match="not_built_yet"):
        resolve_warmup_budget(causal_config(paths=[CAUSAL_SHARD, absent]))


# =================================================================================================
# Refusals: the shards
# =================================================================================================
def test_a_two_sided_shard_is_refused_naming_it() -> None:
    """It has no warm-up, and reading its absence as "fully valid" is the claim this family denies."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(paths=[TWO_SIDED_SHARD]))
    assert TWO_SIDED_SHARD.name in str(error.value)
    assert "two_sided" in str(error.value)


def test_a_two_sided_test_shard_beside_causal_training_shards_is_refused() -> None:
    """The motivating case for reading every shard rather than the first.

    Resolved off the training shard alone it would come out clean, and the held-out numbers would
    be produced against a channel axis that means something else.
    """
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(paths=[CAUSAL_SHARD, TWO_SIDED_SHARD]))
    assert TWO_SIDED_SHARD.name in str(error.value)


def test_an_untouched_copy_of_the_fixture_resolves(tmp_path: Path, budget) -> None:
    """The negative control for the three shard mutations below."""
    copied = write_variant(CAUSAL_SHARD, tmp_path / "copy.hdf5", lambda handle: None)
    resolved = resolve_warmup_budget(causal_config(paths=[CAUSAL_SHARD, copied]))
    assert resolved is not None
    assert resolved.target.keep_index == budget.target.keep_index


def test_a_second_shard_built_at_another_quantile_is_refused_naming_it(tmp_path: Path) -> None:
    """The quantile sets the warm-up *and* which channels survive the build, so it is a channel axis.

    $0.90$ keeps $37$ scattering channels, $0.95$ keeps $36$, $0.99$ keeps $35$ -- which is why a
    config literal could not stand in for reading this.
    """
    def retune(handle) -> None:
        handle.attrs["causal_warmup_quantile"] = np.float32(0.99)

    other = write_variant(CAUSAL_SHARD, tmp_path / "quantile_0p99.hdf5", retune)
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(paths=[CAUSAL_SHARD, other]))
    message = str(error.value)
    assert other.name in message
    assert "causal_warmup_quantile" in message


def test_a_second_shard_with_a_different_warm_up_is_refused_naming_it(tmp_path: Path) -> None:
    """The warm-up is a constant of the filter bank: two shards that disagree had two banks."""
    def shift(handle) -> None:
        vector = np.asarray(handle["fhr_ph"].attrs["causal_warmup_steps"])
        vector[7] += 1
        handle["fhr_ph"].attrs["causal_warmup_steps"] = vector

    other = write_variant(CAUSAL_SHARD, tmp_path / "shifted_warmup.hdf5", shift)
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(paths=[CAUSAL_SHARD, other]))
    message = str(error.value)
    assert other.name in message
    assert "fhr_ph" in message


def test_a_second_shard_with_a_different_width_is_refused_naming_it(tmp_path: Path) -> None:
    """A width disagreement makes every keep-index point at a different channel per file."""
    def narrow(handle) -> None:
        block = handle["up_ph"]
        data = block[:, :14, :]
        warmup = np.asarray(block.attrs["causal_warmup_steps"])[:14]
        # Both per-channel attributes are narrowed with the block. Carrying only one would trip
        # the reader's own completeness check instead, and this test would then pass without ever
        # exercising the width comparison it is named for.
        delay = np.asarray(block.attrs["causal_delay_s"])[:14]
        del handle["up_ph"]
        created = handle.create_dataset("up_ph", data=data)
        created.attrs["causal_warmup_steps"] = warmup
        created.attrs["causal_delay_s"] = delay

    other = write_variant(CAUSAL_SHARD, tmp_path / "narrow_up_ph.hdf5", narrow)
    with pytest.raises(ValueError, match=r"Mismatched 'up_ph' width"):
        resolve_warmup_budget(causal_config(paths=[CAUSAL_SHARD, other]))


def test_a_channel_with_no_valid_step_is_refused_by_the_loader_s_own_rebasing(
    tmp_path: Path,
) -> None:
    r"""The rebase is delegated, and this is the assertion that proves it.

    A local $W' = \max(W - \mathrm{trim}, 0)$ returns a perfectly ordinary integer for a channel
    whose warm-up outruns the window: the column would be served, normalise to zeros
    indistinguishable from real coefficients, and be gated in or out on a number that means
    nothing. ``hdf5_dataset``'s own rebasing raises instead, and reusing it is what buys that.
    """
    def kill(handle) -> None:
        vector = np.asarray(handle["up_ph"].attrs["causal_warmup_steps"])
        vector[3] = handle["up_ph"].shape[2]
        handle["up_ph"].attrs["causal_warmup_steps"] = vector

    dead = write_variant(CAUSAL_SHARD, tmp_path / "dead_channel.hdf5", kill)
    with pytest.raises(ValueError, match=r"'up_ph' channel 3 has no valid step"):
        resolve_warmup_budget(causal_config(paths=[dead]))
