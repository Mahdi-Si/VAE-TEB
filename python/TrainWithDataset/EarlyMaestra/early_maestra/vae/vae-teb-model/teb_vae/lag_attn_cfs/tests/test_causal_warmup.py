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
import logging
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from hdf5_dataset.hdf5_dataset import decimated_trim_steps
from teb_vae.lag_attn_cfs.causal_warmup import (
    ALIGNMENT_DELAY_FACTOR,
    GAMMATONE_ORDER,
    SOURCE_BLOCKS,
    STEP_SECONDS,
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
def test_the_source_is_never_gated_by_the_budget(threshold: int) -> None:
    """No threshold shortens the source keep-index, because the budget never looks at that stream.

    The source's slowest channels carry the contraction envelope and reach $W' = 278$: their
    availability row is zero for $278$ of $300$ steps. Gating them would buy warm lags at the price
    of the signal the lag search exists to find, so the design keeps them and announces their
    arrival instead. A budget that started trimming the source when it moved would remove that
    silently.

    Resolved with the alignment **off**, because that is the one rule that does drop source
    channels and it drops them for an unrelated reason -- a channel slower than the reference could
    only reach it by being read from a later stored step. Mixing the two here would make a passing
    assertion about the budget indistinguishable from one about the reference.
    """
    resolved = resolve_warmup_budget(
        causal_config(causal_warmup_budget_steps=threshold, causal_align_reference=None)
    )
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
    """Stored, they would be a second source of truth that could disagree with the keep-index.

    The five stored fields are the ones nothing else determines: three declared vectors read off the
    shards, the block spans they are laid out in, and the keep-index. ``align_delays`` is stored
    for a different reason -- it is a function of the *reference*, which is a run's decision and not
    recoverable from the vectors beside it.
    """
    stored_fields = {field.name for field in fields(StreamWarmup)}
    assert stored_fields == {
        "name",
        "block_spans",
        "declared_warmup_steps",
        "declared_delay_s",
        "declared_novelty_frac",
        "keep_index",
        "align_delays",
    }
    for derived in (
        "kept_width",
        "dropped_index",
        "warmup_steps",
        "declared_width",
        "max_warmup",
        "delay_s",
        "combined_steps",
        "max_align_delay",
    ):
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
    # The shipped run is aligned, so the source line states what the reference leaves standing --
    # 32 of 36 `up_st` and every one of the 15 `up_ph`, which is why that reference was chosen.
    assert "up_st 32/36" in summary and "up_ph 15/15" in summary


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


# =================================================================================================
# The channel alignment: the reference, the shifts, and what they cost
#
# Every stored channel is stale by its own composed group delay, and those delays span thirteen
# minutes across a stream -- so reading the whole vector at one step index asserts an instant its
# entries do not share. The repair is a per-channel shift onto a common reference, and the whole of
# its cost argument is the lemma below: the shifted warm-up never exceeds the reference channel's
# own. Asserted rather than assumed, because rho = W/tau is only approximately constant.
# =================================================================================================
#: What the shipped bank's reference resolves to, in seconds: the composed delay of the slowest
#: channel the budget keeps. Pinned as a literal for the reason the shipped row above is -- it is
#: the clock every aligned number in this family is stated against.
SHIPPED_REFERENCE_S = 402.1604

#: The four source channels the reference drops, by declared index, and what they are stale by.
#: They are the ``up_st`` scattering channels *below* the reference frequency, so all fifteen
#: ``up_ph`` channels survive -- which is the whole reason this reference was chosen.
ALIGN_DROPPED_SOURCE = (32, 33, 34, 35)
ALIGN_DROPPED_SOURCE_DELAY_S = (475.7, 563.2, 667.3, 791.0)


@pytest.fixture(scope="module")
def aligned():
    """The shipped configuration, resolved once. The alignment is the shipped default.

    Named rather than reusing the shared ``budget`` fixture: the assertions below are about the
    alignment specifically, so a later change to what ships must move this fixture's body rather
    than silently re-point a whole file of alignment tests at whatever replaced it.
    """
    resolved = resolve_warmup_budget(causal_config(causal_align_reference="target_max"))
    assert resolved is not None
    return resolved


@pytest.fixture(scope="module")
def unaligned():
    """The comparison arm: the same configuration with the reference removed."""
    resolved = resolve_warmup_budget(causal_config(causal_align_reference=None))
    assert resolved is not None
    return resolved


def test_no_reference_resolves_to_exactly_todays_tuples(unaligned, aligned) -> None:
    """The off setting is not "alignment with zero shift": it is the model that has no shift at all.

    Asserted against a separately-resolved budget rather than against literals, so the identity is
    over the whole object and not over the four numbers this file happens to pin.
    """
    assert unaligned.reference_delay_s is None
    for stream in (unaligned.target, unaligned.source):
        assert stream.align_delays is None, stream.name
        assert stream.max_align_delay == 0, stream.name
        assert stream.combined_steps == stream.warmup_steps, stream.name

    # The target survives both rules identically -- the reference is its own maximum -- so the one
    # keep-index the setting moves is the source's, which is the whole content of the off arm here.
    assert unaligned.target.keep_index == aligned.target.keep_index
    assert unaligned.source.keep_index == tuple(range(CAUSAL_C_U))
    assert len(aligned.source.keep_index) < CAUSAL_C_U


def test_target_max_resolves_the_shipped_reference_off_the_shards(aligned) -> None:
    """$402.1604$ s, and it is not a coincidence of this fixture.

    It is the composed delay of the filter at $0.008240$ Hz, which is simultaneously the maximum of
    ``fhr_ph``, the maximum of ``up_ph``, the channel the shipped budget already stands on, and the
    lower band edge both phase selections use.
    """
    assert aligned.reference_delay_s == pytest.approx(SHIPPED_REFERENCE_S, abs=5e-4)
    assert aligned.reference_delay_s == max(aligned.target.delay_s)
    # And it is one of the stored values exactly, not a number near them: the drop rule compares
    # delays against it with ``<=``, so an epsilon either way moves a whole harmonic family.
    assert aligned.reference_delay_s in set(aligned.target.declared_delay_s)


def test_the_scored_targets_clock_follows_the_forecast_clock(aligned) -> None:
    """The forecast-side twin of ``source_clock_delay_s``: the tau a page shifts the scored stream
    by. Physical is the fastest kept channel's own delay, input is the target's input reference,
    and stored has no single constant at all -- ``None``, not zero, because zero would draw the
    stream as though its content were stamped with the step it is stored at."""
    from dataclasses import replace

    from teb_vae.lag_attn_cfs import causal_warmup

    physical = replace(
        aligned,
        target_forecast_clock=causal_warmup.FORECAST_CLOCK_PHYSICAL,
        target_forecast_reference_s=13.3405,
    )
    assert physical.target_forecast_clock_delay_s == 13.3405

    inputs = replace(aligned, target_forecast_clock=causal_warmup.FORECAST_CLOCK_INPUT)
    assert inputs.target_forecast_clock_delay_s == aligned.reference_delay_s

    stored = replace(aligned, target_forecast_clock=causal_warmup.FORECAST_CLOCK_STORED)
    assert stored.target_forecast_clock_delay_s is None


def test_the_shifts_span_zero_to_eighty_five_on_both_streams(aligned) -> None:
    r"""Zero at the reference channel, largest at the fastest one, and never negative.

    $85$ rather than the $97$ this pinned before :data:`ALIGNMENT_DELAY_FACTOR` existed: the
    shift carries $\kappa = 0.875$, so the span shrinks by exactly that factor while the
    keep-index, which depends on $\tau_c \le \tau_{\mathrm{ref}}$ alone, does not move at all.
    """
    for stream in (aligned.target, aligned.source):
        assert stream.align_delays is not None, stream.name
        assert min(stream.align_delays) == 0, stream.name
        assert stream.max_align_delay == 85, stream.name
        assert all(shift >= 0 for shift in stream.align_delays), stream.name
        assert len(stream.align_delays) == stream.kept_width, stream.name


def test_the_quantisation_residual_stays_inside_half_a_step(aligned) -> None:
    r"""Rounding, not ceiling: both directions are causally safe, so the only criterion is the
    residual $\lvert\kappa(\tau_{\mathrm{ref}} - \tau_c) - \Delta d_c\rvert \le \Delta/2 = 2$ s.
    Measured at $1.9865$ s on the shipped bank, which is the number the alignment is priced at.

    The residual is taken against the **scaled** difference, because that is what the shift is
    rounding: measuring it against the unscaled one would report the alignment error as
    $\approx 50$ s and say nothing about the rounding.
    """
    for stream in (aligned.target, aligned.source):
        residual = max(
            abs(
                ALIGNMENT_DELAY_FACTOR * (aligned.reference_delay_s - delay)
                - STEP_SECONDS * shift
            )
            for delay, shift in zip(stream.delay_s, stream.align_delays)
        )
        assert residual <= STEP_SECONDS / 2.0, stream.name
        assert residual == pytest.approx(1.9865, abs=1e-3), stream.name


def test_the_zero_marginal_warm_up_lemma_holds_exactly_on_both_streams(aligned) -> None:
    r"""The whole cost argument, as an equality rather than a bound.

    With $W_c = \rho\tau_c$ and $\Delta d_c = \tau_{\mathrm{ref}} - \tau_c$,
    $W_c + \Delta d_c = \tau_{\mathrm{ref}} + (\rho - 1)\tau_c \le W_{\mathrm{ref}}$, with equality
    at the reference channel -- so aligning a stream to its slowest kept channel costs no warm-up
    beyond that channel's own. $\rho$ is only approximately constant (median $1.516$ on the target,
    $1.493$ on the source, over $[1.482,\,1.713]$), so the equality is measured here and not
    derived. A bank change that eroded it would move the anchor floor with nothing else saying so.
    """
    for stream in (aligned.target, aligned.source):
        assert max(stream.combined_steps) == stream.max_warmup, stream.name
        assert max(stream.combined_steps) == SHIPPED_BUDGET_STEPS, stream.name
    # And the *minimum* crosses zero, which is the change that is easy to miss: it is what builds
    # the availability adapter's start-of-record token for the first time in this family.
    for stream in (aligned.target, aligned.source):
        assert min(stream.combined_steps) == 80, stream.name
        assert min(stream.warmup_steps) == 0, stream.name


def test_the_target_keeps_every_channel_the_budget_kept(aligned, budget) -> None:
    """The reference is the maximum over those channels, so by construction none is above it."""
    assert aligned.target.keep_index == budget.target.keep_index
    assert aligned.target.kept_width == SHIPPED_TARGET_KEPT


def test_the_source_loses_the_four_channels_above_the_reference(aligned) -> None:
    """A correctness drop, not a warm-up policy: those channels can only reach the reference by a
    negative shift, i.e. by being read from a later stored step. All fifteen ``up_ph`` survive."""
    assert aligned.source.dropped_index == ALIGN_DROPPED_SOURCE
    assert aligned.source.kept_width == CAUSAL_C_U - len(ALIGN_DROPPED_SOURCE)
    assert aligned.source.kept_width == 47
    assert {
        name: kept for name, kept, _declared in aligned.source.block_counts()
    } == {"up_st": 32, "up_ph": 15}
    for index, delay in zip(ALIGN_DROPPED_SOURCE, ALIGN_DROPPED_SOURCE_DELAY_S):
        assert aligned.source.declared_delay_s[index] == pytest.approx(delay, abs=0.1)
        assert aligned.source.declared_delay_s[index] > aligned.reference_delay_s


def test_each_dropped_source_channel_is_logged_by_index_and_by_delay(caplog) -> None:
    """Which channels a run stopped reading is not recoverable from any metric it emits, and a
    summary count would not survive a channel-plan change while an index and a delay do."""
    with caplog.at_level(logging.INFO, logger="teb_vae.lag_attn_cfs.causal_warmup"):
        resolved = resolve_warmup_budget(causal_config(causal_align_reference="target_max"))
    assert resolved is not None

    lines = [
        record.getMessage()
        for record in caplog.records
        if "drops source" in record.getMessage()
    ]
    assert len(lines) == len(ALIGN_DROPPED_SOURCE)
    for index in ALIGN_DROPPED_SOURCE:
        matching = [line for line in lines if f"source channel {index}:" in line]
        assert len(matching) == 1, index
        # The delay the resolver itself dropped the channel on, not a number this file restates:
        # a log line naming a delay the channel does not have would be worse than none at all.
        assert f"{resolved.source.declared_delay_s[index]:.4f} s" in matching[0], index
        assert "negative shift" in matching[0]


def test_the_resolver_agrees_with_the_shared_alignment_rule_entry_for_entry(aligned) -> None:
    """The one duplication in this module, pinned.

    ``hdf5_dataset.causal_scattering.channel_alignment_delays`` is the canonical statement of $d_c$
    and is what the writer and the fidelity harness call; the resolver restates it because that
    module imports ``kymatio`` at module scope and importing it here would build a filter bank into
    every training process. This is what keeps the restatement from drifting.
    """
    from hdf5_dataset.causal_scattering import channel_alignment_delays

    for stream in (aligned.target, aligned.source):
        expected = channel_alignment_delays(
            np.asarray(stream.delay_s, dtype=np.float64),
            aligned.reference_delay_s,
            STEP_SECONDS,
        )
        assert tuple(int(value) for value in expected) == stream.align_delays, stream.name


def test_an_explicit_reference_is_snapped_to_the_channel_it_names() -> None:
    r"""The every-lag-warm override, and the float32 trap it would otherwise fall into.

    $150.79$ s keeps $27$ of $51$ source channels and one of fifteen ``up_ph`` -- which is why it is
    not the default. The stored delay is a ``float32``, so a config literal lands a few microseconds
    off it; compared exactly, that drops the very channel the operator named, both its harmonic
    siblings and the whole ``up_ph`` block behind them. The resolved reference is therefore the
    matched channel's own delay rather than the number typed.
    """
    resolved = resolve_warmup_budget(causal_config(causal_align_reference=150.79))
    assert resolved is not None
    assert resolved.reference_delay_s in set(resolved.target.declared_delay_s)
    assert resolved.reference_delay_s == pytest.approx(150.786, abs=1e-3)
    assert resolved.source.kept_width == 27
    assert {
        name: kept for name, kept, _declared in resolved.source.block_counts()
    } == {"up_st": 26, "up_ph": 1}
    assert min(resolved.target.align_delays) == 0


def test_a_negative_shift_cannot_be_produced_at_any_reference() -> None:
    r"""The correctness property the drop rule exists for, over every reference the shards admit.

    A channel above $\tau_{\mathrm{ref}}$ would need $d_c < 0$ -- to be read from a *later* stored
    step, i.e. from raw signal after the anchor, which destroys the one property the whole causal
    construction is built on. ``ChannelDelay`` refuses a negative entry by name, and that refusal
    must never be the thing that fires: the resolver drops such a channel before the gate is ever
    built, so what reaches the model is a shorter keep-index rather than an exception.
    """
    for reference in (402.1604, 150.79, 61.9, 20.5):
        resolved = resolve_warmup_budget(causal_config(causal_align_reference=reference))
        assert resolved is not None
        for stream in (resolved.target, resolved.source):
            assert min(stream.align_delays) >= 0, (reference, stream.name)
            # And the drop is exactly the above-reference set, not a wider or narrower one.
            assert all(
                stream.declared_delay_s[index] > resolved.reference_delay_s
                or index in _budget_dropped(stream)
                for index in stream.dropped_index
            ), (reference, stream.name)
            assert all(
                stream.declared_delay_s[index] <= resolved.reference_delay_s
                for index in stream.keep_index
            ), (reference, stream.name)


def _budget_dropped(stream: StreamWarmup) -> set:
    """Declared indices the **warm-up budget** removed, as opposed to the alignment.

    The target stream carries both rules at once and they remove different channels for different
    reasons, so the test above has to separate them: a channel dropped for waiting too long is not
    evidence about the reference it was never compared against.
    """
    return {
        index
        for index, steps in enumerate(stream.declared_warmup_steps)
        if steps > SHIPPED_BUDGET_STEPS
    }


def test_a_reference_matching_no_kept_target_channel_is_refused_naming_it() -> None:
    """A reference between two channels is a clock no channel keeps, and its residual lands on
    every channel at once rather than on any one of them as a failure."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(causal_align_reference=10.0))
    message = str(error.value)
    assert "causal_align_reference" in message
    assert "13.3" in message and "half-step" in message


def test_an_unknown_reference_mode_is_refused_naming_the_one_that_exists() -> None:
    """The string form has exactly one admissible value, and a typo must not resolve to a float."""
    with pytest.raises(ValueError, match=r"target_max"):
        resolve_warmup_budget(causal_config(causal_align_reference="source_max"))


def test_a_shard_whose_leg_alignment_disagrees_with_the_config_is_refused() -> None:
    r"""The two shard variants are identical in every width, warm-up and stored delay, so nothing
    else in this resolution can tell them apart -- and only one of them makes the stored
    ``causal_delay_s`` true of the phase blocks. Under the unaligned operator the composed delay is
    a misprediction the block misses by a median of $60.5$ steps, and shifting by a number the data
    does not have is worse than not shifting at all."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(causal_leg_alignment="none"))
    message = str(error.value)
    assert "causal_leg_alignment" in message
    assert "'envelope'" in message and "'none'" in message
    assert str(CAUSAL_SHARD) in message


def test_the_expected_leg_alignment_may_be_left_unstated(aligned) -> None:
    """``null`` is a run that does not care, which is what a comparison of the two variants is.

    The two mechanisms stay independently toggleable: the reference resolves from ``causal_delay_s``
    alone, which every causal shard stores whichever operator built its phase blocks, so an unstated
    expectation must resolve the identical shifts rather than refuse.
    """
    assert aligned.leg_alignment == "envelope"
    unstated = resolve_warmup_budget(
        causal_config(causal_align_reference="target_max", causal_leg_alignment=None)
    )
    assert unstated is not None
    assert unstated.leg_alignment == "envelope"
    assert unstated.target.align_delays == aligned.target.align_delays


def test_the_alignment_factor_agrees_with_the_bank_that_defines_it() -> None:
    r"""The resolver restates $\kappa$ rather than importing it, so pin the two together.

    ``causal_warmup`` must stay free of ``hdf5_dataset.causal_scattering``, which builds the
    two-sided filter bank at import; the cost of that isolation is a second copy of
    :data:`ALIGNMENT_DELAY_FACTOR` and of the gammatone order it is derived from. This is the test
    that makes the copy safe -- a bank rebuilt at another order fails here rather than silently
    aligning every channel against the wrong constant.
    """
    from hdf5_dataset import causal_scattering

    assert GAMMATONE_ORDER == causal_scattering.GAMMATONE_ORDER
    assert ALIGNMENT_DELAY_FACTOR == pytest.approx(
        causal_scattering.ALIGNMENT_DELAY_FACTOR, abs=0.0
    )
    assert ALIGNMENT_DELAY_FACTOR == pytest.approx(0.875, abs=0.0)


def test_the_summary_names_the_reference_and_the_leg_alignment(aligned, unaligned) -> None:
    """The startup log is the only place a run states which clock it read its channels on."""
    aligned_summary = aligned.summary()
    assert "reference 402.1604 s" in aligned_summary
    assert "leg alignment envelope" in aligned_summary
    assert "shift 0-85 steps" in aligned_summary
    assert "up_st 32/36" in aligned_summary and "up_ph 15/15" in aligned_summary

    # And the unaligned line does not grow a "shift 0-0" that would read as a mechanism running.
    assert "unaligned" in unaligned.summary()
    assert "shift" not in unaligned.summary()


# =================================================================================================
# The novelty vector the block-score split is partitioned by
# =================================================================================================
def test_the_novelty_vector_is_read_off_the_shards_at_the_declared_width(budget) -> None:
    r"""$\nu_c$ per **declared** channel, concatenated in the same block order the other two are.

    Declared rather than kept, because the model gathers it through the keep-index: a survivors-
    length vector would be positional against a width the ungated comparison arm does not have.
    Checked against the stored attribute per block rather than against a literal, so a rebuilt
    fixture moves the expectation with the data.
    """
    import h5py

    for stream, blocks in ((budget.target, TARGET_BLOCKS), (budget.source, SOURCE_BLOCKS)):
        stored: list = []
        with h5py.File(CAUSAL_SHARD, "r") as handle:
            for block in blocks:
                stored.extend(
                    float(share) for share in handle[block].attrs["causal_novelty_frac"]
                )
        assert stream.declared_novelty_frac is not None, stream.name
        assert len(stream.declared_novelty_frac) == stream.declared_width, stream.name
        assert stream.declared_novelty_frac == pytest.approx(stored), stream.name
        assert all(0.0 <= share <= 1.0 for share in stream.declared_novelty_frac), stream.name


def test_the_novelty_vector_is_absent_rather_than_defaulted_on_a_shard_without_it(
    tmp_path: Path,
) -> None:
    """A shard predating the attribute resolves, and says so by carrying ``None``.

    Absent rather than zero-filled: there is no share a missing vector could stand in as, and both
    endpoints are meaningful values. The budget itself still resolves, because the novelty is a
    readout and not a guard -- what refuses is the model mapping, which is where the column that
    needs it is emitted.
    """

    def _strip(handle) -> None:
        for block in TARGET_BLOCKS + SOURCE_BLOCKS:
            del handle[block].attrs["causal_novelty_frac"]

    legacy = write_variant(CAUSAL_SHARD, tmp_path / "no_novelty.hdf5", _strip)
    resolved = resolve_warmup_budget(causal_config(paths=[legacy]))

    assert resolved is not None
    assert resolved.target.declared_novelty_frac is None
    assert resolved.source.declared_novelty_frac is None
    # Everything else about that shard resolves exactly as the committed one does.
    assert resolved.target.keep_index == resolve_warmup_budget(causal_config()).target.keep_index


def test_a_novelty_vector_of_the_wrong_width_is_refused_naming_the_block(tmp_path: Path) -> None:
    """Short by one on one block and long by one on another concatenates to the declared width.

    Every length downstream would then be correct and every channel after the join would carry its
    neighbour's novelty, so the check is per block rather than on the concatenation.
    """

    def _shorten(handle) -> None:
        vector = np.asarray(handle["fhr_st"].attrs["causal_novelty_frac"])
        handle["fhr_st"].attrs["causal_novelty_frac"] = vector[:-1]

    mangled = write_variant(CAUSAL_SHARD, tmp_path / "short_novelty.hdf5", _shorten)
    with pytest.raises(ValueError, match="fhr_st.*causal_novelty_frac"):
        resolve_warmup_budget(causal_config(paths=[mangled]))


# =================================================================================================
# The forecast clock
# =================================================================================================
@pytest.fixture(scope="module")
def physical_clock():
    """The shipped aligned configuration scored on the physical clock, resolved once."""
    resolved = resolve_warmup_budget(
        causal_config(causal_target_forecast_clock="physical")
    )
    assert resolved is not None
    return resolved


def test_the_absent_and_stored_clocks_resolve_to_exactly_todays_object(aligned) -> None:
    """``stored`` and the absent key are one case, and neither carries a shift vector at all.

    Whole-object identity against a separately resolved budget, so a run written before the key
    existed resolves byte for byte what it did then -- including its ``model_kwargs``.
    """
    stored = resolve_warmup_budget(causal_config(causal_target_forecast_clock="stored"))
    assert stored is not None
    assert stored == resolve_warmup_budget(causal_config())
    assert stored.target_forecast_clock == "stored"
    assert stored.target_forecast_shift is None
    assert stored.target_forecast_reference_s is None
    assert stored.max_forecast_advance == 0
    assert stored == aligned

    from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    assert "target_forecast_shift" not in warmup_model_kwargs(stored, SeqVaeLagAttnCfs)


def test_the_physical_clock_advances_onto_the_fastest_kept_channel(physical_clock) -> None:
    r"""$s_c = \mathrm{round}(\kappa(\tau_c - \tau_{\min})/\Delta) \ge 0$, entry for entry.

    The reference is the fastest kept channel's own stored delay -- whose shift is exactly zero --
    so the advance of the slowest channel equals the input alignment's largest delay: both are
    $\mathrm{round}(\kappa(\tau_{\max} - \tau_{\min})/\Delta)$.
    """
    shifts = physical_clock.target_forecast_shift
    assert shifts is not None
    assert physical_clock.target_forecast_clock == "physical"

    reference = physical_clock.target_forecast_reference_s
    assert reference == min(physical_clock.target.delay_s)
    assert reference in set(physical_clock.target.declared_delay_s)

    expected = tuple(
        int(round(ALIGNMENT_DELAY_FACTOR * (delay - reference) / STEP_SECONDS))
        for delay in physical_clock.target.delay_s
    )
    assert shifts == expected
    assert min(shifts) == 0
    assert physical_clock.max_forecast_advance == max(shifts)
    assert physical_clock.max_forecast_advance == physical_clock.target.max_align_delay

    # The clock changes the question, never the streams: every channel tuple is the shipped one.
    stored = resolve_warmup_budget(causal_config())
    assert physical_clock.target.keep_index == stored.target.keep_index
    assert physical_clock.source.keep_index == stored.source.keep_index


def test_the_input_clock_negates_the_alignment_delays(aligned) -> None:
    r"""$s_c = -d_c$: the scored element delayed exactly as the encoder input is."""
    resolved = resolve_warmup_budget(
        causal_config(causal_target_forecast_clock="input")
    )
    assert resolved is not None
    assert resolved.target_forecast_clock == "input"
    assert resolved.target_forecast_shift == tuple(
        -delay for delay in aligned.target.align_delays
    )
    assert resolved.max_forecast_advance == 0
    assert resolved.target_forecast_reference_s is None


def test_the_input_clock_is_refused_against_an_unaligned_target() -> None:
    """With no $d_c$ there is no input clock to score on, and the refusal names both keys."""
    with pytest.raises(ValueError, match="causal_target_forecast_clock.*causal_align_reference"):
        resolve_warmup_budget(
            causal_config(
                causal_align_reference=None, causal_target_forecast_clock="input"
            )
        )


def test_an_unknown_forecast_clock_is_refused_naming_the_three() -> None:
    """The refusal lists every clock the resolver knows, so the fix is a copy rather than a search."""
    with pytest.raises(ValueError, match="physical.*input.*stored"):
        resolve_warmup_budget(causal_config(causal_target_forecast_clock="aligned"))


def test_the_advance_shrinks_the_feasible_floor_and_stride_pairings(physical_clock) -> None:
    r"""A pairing the stored clock admits is refused when the advance eats its last anchor.

    The ceiling is $T_{\mathrm{valid}}$ less the largest advance, so the same floor and stride
    must clear a shorter span -- and the refusal names the clock that shortened it.
    """
    t_valid = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON
    ceiling = t_valid - physical_clock.max_forecast_advance
    floor = ceiling - SHIPPED_HORIZON + 1

    # Admitted on the stored clock...
    assert resolve_warmup_budget(causal_config(warmup_period=floor)) is not None
    # ...refused on the physical one, naming the clock.
    with pytest.raises(ValueError, match="physical"):
        resolve_warmup_budget(
            causal_config(
                warmup_period=floor, causal_target_forecast_clock="physical"
            )
        )
    # One step lower is the last floor that works, so the boundary is where it is claimed to be.
    assert (
        resolve_warmup_budget(
            causal_config(
                warmup_period=floor - 1, causal_target_forecast_clock="physical"
            )
        )
        is not None
    )


def test_the_kwargs_carry_the_shift_only_when_a_clock_is_configured(physical_clock) -> None:
    """The vector reaches the constructor under its own name, and only when it exists."""
    from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    mapped = warmup_model_kwargs(physical_clock, SeqVaeLagAttnCfs)
    assert mapped["target_forecast_shift"] == physical_clock.target_forecast_shift
