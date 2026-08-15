r"""The shared collection pass and the two tables every later analysis reads.

Four properties carry this suite, and each of them is a way the pipeline could report a complete
set of plausible numbers while measuring something else.

**The tables must be a census of what was scored.** One row per scored segment, one row per
contributing anchor, and the two must agree with the readouts computed in the same pass. A table
one row short is not a visible failure -- it is a slightly different population, reported with the
same confidence.

**A segment that measured nothing must not read as a segment that measured zero.** The per-sample
mean clamps its denominator to $1$, so an unscored segment's columns come out as exactly ``0.0``.
Averaged into a summed-$1470$-coefficient block score of hundreds of nats, that pulls the headline
toward zero and shrinks ``pred_gap``, and nothing else in the output moves.

**The anchor column must be the decimated step.** This model decodes a *gathered* set of anchors
out of $T_{\mathrm{valid}}$, so a table keyed on a row's position in that set would join silently
and wrongly against every other time axis in the run -- and on a dense pass the two differ by
exactly the anchor floor, which is a plausible-looking offset rather than an obvious one.

**Reuse must be safe or refused.** Skipping the forward pass is what makes an offline re-run
possible; reusing another checkpoint's rows under this run's summary is what makes it dangerous,
and only one of the two may be silent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.collect import (
    COLLECTION_FILENAME,
    HORIZON_STATISTICS,
    NORMALIZED_BLOCKS,
    PER_ANCHOR_FILENAME,
    PER_ANCHOR_KEY,
    PER_SAMPLE_FILENAME,
    RETAINED_QUANTITIES,
    VECTORS_FILENAME,
    Collection,
    RetentionPlan,
    TablesProvenanceMismatch,
    check_per_anchor_key,
    collect_tables,
    load_collection,
    load_or_collect,
    write_collection,
)
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    VECTOR_READOUTS,
    evaluate,
    expected_anchors_per_sample,
    horizon_residual_sums,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

from .conftest import make_stub_batch

#: Any seed; every pass below fixes the global stream too, because the model's own
#: reparameterisation draws from it and a reference pass must see the same latent.
_SEED = 7


class _StubLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)
        # Present so the retention plan can draw over the whole index space; a bare list is
        # enough, since nothing indexes it.
        self.dataset = list(range(sum(len(batch.guid) for batch in self._batches)))

    def __iter__(self):
        return iter(self._batches)


def _labelled_batch(
    guids: List[str],
    *,
    seed: int = 0,
    class_code: int = 1,
    epoch_offset: float = -36000.0,
    onsets: Optional[List[float]] = None,
    shard: str = "acidosis_cs.hdf5",
):
    """A stub batch carrying the identity columns a real batch does.

    The identity is what the tables are *about*: a stub batch without ``guid``, ``epoch`` or a
    ``target`` to recover the class from exercises only the fallback branches, which is how a
    collection pass can look correct while producing a table nobody can group by.

    Args:
        guids: One recording identifier per sample; also sets the batch size.
        seed: Seed for the signals.
        class_code: The clinical class code the ``target`` encodes, scaled by ``weight`` exactly
            as the shard writer stores it.
        epoch_offset: First sample's ``epoch``; later samples step away from it so the
            per-anchor key stays unique.
        onsets: ``time_from_labor_onset`` per sample, NaN included. Defaults to a finite value.
        shard: The source file basename, from which the subgroup is recovered.

    Returns:
        The batch.
    """
    batch = make_stub_batch(batch=len(guids), seed=seed)
    batch.guid = list(guids)
    batch.epoch = torch.tensor(
        [epoch_offset + 1200.0 * index for index in range(len(guids))], dtype=torch.float32
    )
    batch.target = float(class_code) * batch.weight
    batch.cs_label = torch.ones(len(guids), dtype=torch.uint8)
    batch.bg_label = torch.zeros(len(guids), dtype=torch.uint8)
    batch.time_from_labor_onset = torch.tensor(
        [0.0] * len(guids) if onsets is None else onsets, dtype=torch.float32
    )
    batch.source_file_basename = [shard] * len(guids)
    return batch


@pytest.fixture
def trained_task(task, perturb_posterior):
    """A tiny task whose posterior has been moved off the prior.

    At initialisation the delta heads are zero, so the KL is exactly zero and base and full are
    bitwise identical -- every column of both tables would be a structural constant.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    return module


def _collect(trained_task, batches, *, caps=None, n_total=None) -> Collection:
    """Run one collection pass over stub batches under a fixed seed."""
    torch.manual_seed(_SEED)
    return collect_tables(
        trained_task,
        _StubLoader(batches),
        eval_config={
            "seed": _SEED,
            "caps": caps or {},
            "prior_shuffle_min_nats": 1.0,
            "min_active_dims": 2,
            "clock_margin_min_nats": None,
        },
        num_samples=1,
        n_total=n_total,
        perm_generator=torch.Generator().manual_seed(_SEED),
        mc_generator=torch.Generator().manual_seed(_SEED),
    )


# =================================================================================================
# The per-sample table
# =================================================================================================
def test_the_per_sample_table_holds_one_row_per_scored_sample(trained_task):
    """The row count is the promise every table-driven analysis rests on: a table one row short
    is a different population reported with the same confidence."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])

    assert len(collection.per_sample) == collection.results["n_samples"]
    assert collection.per_sample["guid"].nunique() == collection.results["n_recordings"]


def test_every_row_carries_the_identity_a_clinical_question_is_asked_in(trained_task):
    """Attached once here rather than per analysis: the class and the subgroup are properties of
    the sample, so a by-class number is a ``groupby`` on an existing column."""
    frame = _collect(
        trained_task,
        # Two recordings, two segments each: a batch holding one recording has no stranger in it
        # for the permutation control to borrow a source from and is excluded whole, which would
        # leave every assertion below reading an empty table.
        [_labelled_batch(["a", "a", "b", "b"], class_code=3, shard="hie_cs.hdf5")],
    ).per_sample

    for name in (
        "guid", "epoch", "clinical_class", "subgroup", "cs_label", "bg_label",
        "time_from_labor_onset", "source_file_basename", "n_anchors", "n_segments_in_guid",
    ):
        assert name in frame.columns, name
    assert set(frame["clinical_class"]) == {"hie"}
    assert set(frame["subgroup"]) == {"hie_cs"}
    # Counted over every segment the recording contributed, which is the denominator of "how much
    # of this recording was measurable".
    assert list(frame["n_segments_in_guid"]) == [2, 2, 2, 2]


def test_the_readout_columns_reach_the_table_beside_the_identity(trained_task):
    """The table is what a later analysis re-derives the headline from, so every scalar readout
    the pass computed has to be on it -- including this cell's own."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])
    frame = collection.per_sample

    for name in collection.results["readouts"]:
        assert name in frame.columns, name
        assert frame[name].notna().any(), name
    # The four this cell alone has: the availability-clock arm, its difference against the
    # coupling readout, and the two geometry guards.
    for name in (
        "kld_source_null", "coupling_minus_clock", "anchors_per_sample", "target_warm_frac",
    ):
        assert name in frame.columns, name


def test_a_short_per_sample_column_raises_rather_than_misaligning(trained_task):
    """The batch is the authority on how many samples there are. A column that redefines it puts
    one sample's guid beside another sample's numbers, and the output stays plausible."""
    batch = _labelled_batch(["a", "b"])
    batch.epoch = torch.tensor([-1000.0])

    with pytest.raises(ValueError, match="holds 1 value"):
        _collect(trained_task, [batch])


# =================================================================================================
# Zero-anchor accounting
# =================================================================================================
def test_a_segment_that_scored_no_anchors_is_nan_rather_than_zero(trained_task):
    r"""Its per-sample mean divides by a denominator clamped to $1$, so an empty numerator reads
    as exactly ``0.0`` -- a fabricated score. NaN is the representation that makes every
    downstream ``mean()`` skip it without being told to.
    """
    batch = _labelled_batch(["a", "b"])
    batch.weight[0] = 0.0
    batch.target = 2.0 * batch.weight

    collection = _collect(trained_task, [batch])
    frame = collection.per_sample

    assert int(frame["n_anchors"][0]) == 0
    assert int(frame["n_anchors"][1]) > 0, "the other sample must still score"
    assert np.isnan(frame["nll_full_block"][0]), "not 0.0"
    assert np.isfinite(frame["nll_full_block"][1])
    # It is still a row: the table is a census of what the loader yielded, not of what scored.
    assert len(frame) == 2
    # And the other table's representation of the same fact: no anchor rows at all.
    assert "a" not in set(collection.per_anchor["guid"])
    assert "b" in set(collection.per_anchor["guid"])


def test_the_zero_anchor_exclusions_are_counted_per_recording_and_per_subgroup(trained_task):
    """Counted, not merely dropped: a run where this is large measured far less than its segment
    count suggests, and nothing else in the output says so."""
    batch = _labelled_batch(["a", "b"], shard="hie_no_cs.hdf5")
    batch.weight[0] = 0.0

    record = _collect(trained_task, [batch]).record

    assert record["n_segments_excluded_zero_anchors"] == 1
    assert record["excluded_zero_anchors"]["per_guid"] == {"a": 1}
    assert record["excluded_zero_anchors"]["per_subgroup"] == {"hie_no_cs": 1}


def test_every_reported_column_carries_its_own_finite_denominator(trained_task):
    """A fraction without its denominator is not a measurement. These are the denominators every
    later analysis divides by, counted after the zero-anchor blanking rather than before it."""
    batch = _labelled_batch(["a", "b"])
    batch.weight[0] = 0.0

    denominators = _collect(trained_task, [batch]).record["denominators"]

    assert denominators["nll_full_block"] == 1
    assert denominators["source_conditioned_kl_raw"] == 1
    assert set(denominators) >= {"pred_gap", "mc_pred_gap", "delta_mu_rms", "kld_source_null"}


def test_the_vector_readouts_are_blanked_on_the_same_rule_as_the_scalars(trained_task):
    """The per-dimension KL of a segment that scored no anchors is a row of clamped zeros, and it
    decomposes a headline the segment did not contribute to. The per-channel gap vector is on the
    same rule for the same reason: a row of exact zeros would land in a band mean as a
    measurement."""
    batch = _labelled_batch(["a", "b"])
    batch.weight[0] = 0.0

    vectors = _collect(trained_task, [batch]).vectors

    assert set(vectors) == set(VECTOR_READOUTS)
    for name in ("kld_per_dim", "gap_per_channel", "sq_error_per_channel_full"):
        assert np.isnan(vectors[name][0]).all(), name
        assert np.isfinite(vectors[name][1]).all(), name


def test_the_channel_vectors_are_the_kept_axis_and_are_row_aligned(trained_task):
    """The width every band-resolved statement is indexed on. Wrong, and a band mean would be
    taken over channels the decoder never emitted."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])
    kept = int(trained_task.orig_model.target_gate.out_channels)

    for name in (
        "gap_per_channel", "sq_error_per_channel_base", "sq_error_per_channel_full"
    ):
        assert collection.vectors[name].shape == (len(collection.per_sample), kept), name


# =================================================================================================
# The per-anchor table
# =================================================================================================
def test_the_per_anchor_row_count_is_the_summed_contributing_anchors(trained_task):
    """The two tables describe one pass, so the anchor table's length is a number the sample
    table already carries -- and a disagreement means one of them is not that pass."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])
    expected = int(collection.per_sample["n_anchors"].sum())

    assert len(collection.per_anchor) == expected
    assert expected > 0


def test_the_per_anchor_table_carries_the_anchor_level_columns(trained_task):
    """Including the three warm-up tertile gaps, which are a decomposition of ``pred_gap`` over
    the kept channel axis and must recombine per anchor as well as sum per sample."""
    frame = _collect(trained_task, [_labelled_batch(["a", "b"])]).per_anchor

    for name in (
        "kld_per_t", "nll_base_block", "nll_full_block", "pred_gap",
        "mc_nll_base_block", "mc_nll_full_block", "mc_pred_gap", "argmax_lag", "coverage",
        "pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi",
        "seconds_since_contraction",
    ):
        assert name in frame.columns, name
    assert frame["coverage"].max() <= 1.0


def test_the_anchor_column_is_the_decimated_step_not_the_row_position(trained_task):
    r"""The correctness risk this cell's gathered anchor set creates.

    The decoded set starts at the anchor floor $F$, so a table keyed on a row's *position* would
    be uniformly $F$ steps early -- a plausible time axis rather than an obviously broken one, and
    every join against the trajectory axis or the event table would be silently wrong.
    """
    model = trained_task.orig_model
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])
    anchors = collection.per_anchor["anchor"]

    assert int(anchors.min()) >= int(model.warmup_period)
    assert int(anchors.max()) <= int(model.geometry.t_valid) - 1
    # Non-vacuous only because the floor is not zero: a position-keyed table would start at 0.
    assert int(model.warmup_period) > 0
    assert int(anchors.min()) == int(model.warmup_period)
    # And it is the set the forward decoded, read back through the same public entry point the
    # geometry guard uses rather than re-derived here.
    assert (
        collection.per_anchor.groupby("guid")["anchor"].nunique().max()
        <= expected_anchors_per_sample(model)
    )


def test_the_anchor_column_agrees_with_the_forwards_own_anchor_index(trained_task):
    """Two derivations of the same set: the table's column, and the tensor the forward returned.
    A disagreement is a wrong number rather than an exception, which is why it is measured."""
    batch = _labelled_batch(["a", "b"])
    model = trained_task.orig_model
    y_st, y_ph, u_stream, _target_features, _weight = model_inputs(trained_task, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    decoded = set(int(value) for value in outputs["anchor_index"][0].tolist())

    collection = _collect(trained_task, [batch])

    assert set(int(value) for value in collection.per_anchor["anchor"]) <= decoded


def test_the_per_anchor_key_is_unique(trained_task):
    """Non-vacuous only with two segments of one recording on the table: those are the rows that
    collide when a batch reaches it without an epoch, which is the failure the key check exists
    for."""
    collection = _collect(trained_task, [_labelled_batch(["a", "a", "b", "b"])])

    assert collection.per_sample["guid"].value_counts().max() == 2
    assert not collection.per_anchor.duplicated(subset=list(PER_ANCHOR_KEY)).any()


def test_a_duplicated_key_is_refused_rather_than_silently_double_counted():
    """Two segments of one recording sharing an epoch -- or a batch with no epoch at all -- makes
    every join and every ``groupby`` built on the key double-count."""
    frame = pd.DataFrame(
        {"guid": ["a", "a"], "epoch": [-100.0, -100.0], "anchor": [3, 3], "kld_per_t": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="not unique"):
        check_per_anchor_key(frame)


def test_the_per_anchor_recombines_into_the_per_sample_row(trained_task):
    r"""The identity that makes the two tables one pass rather than two: a sample's per-anchor
    values averaged over its own anchors *are* its per-sample column."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])

    per_anchor = collection.per_anchor.groupby("guid")["nll_full_block"].mean()
    per_sample = collection.per_sample.set_index("guid")["nll_full_block"]

    for guid, value in per_anchor.items():
        assert value == pytest.approx(float(per_sample[guid]), rel=1e-6)


def test_the_per_anchor_table_round_trips_through_parquet_including_nan(tmp_path, trained_task):
    """Parquet only, so there is one format and no fallback branch; NaN has to survive it,
    because a coverage or a gap column that reads back as zero is a different measurement."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])
    collection.per_anchor.loc[0, "coverage"] = np.nan

    write_collection(collection, tmp_path)
    reloaded = pd.read_parquet(tmp_path / PER_ANCHOR_FILENAME)

    assert np.isnan(reloaded["coverage"][0])
    pd.testing.assert_frame_equal(reloaded, collection.per_anchor)


# =================================================================================================
# Heavy-artifact retention
# =================================================================================================
def test_nothing_heavy_is_retained_unless_a_cap_asks_for_it(trained_task):
    r"""Retention is opt-in. At the shipped geometry the alternative default is $3.8$ MiB per
    sample held for the whole run, for figures nobody asked for.
    """
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])

    assert collection.retained == {}
    quantities = collection.record["retention"]["quantities"]
    assert quantities["waveforms"]["cap"] == "absent"
    assert quantities["waveforms"]["n_bytes"] == 0


def test_a_cap_retains_that_many_samples_and_the_size_is_recorded(trained_task):
    """A cap that did not reduce anything is a stated intention; the measured size is the fact."""
    batches = [_labelled_batch(["a", "b"]), _labelled_batch(["c", "d"], seed=1)]

    capped = _collect(trained_task, batches, caps={"waveforms": 1}, n_total=4)
    everything = _collect(trained_task, batches, caps={"waveforms": None}, n_total=4)

    assert capped.retained["mu_full"].shape[0] == 1
    assert everything.retained["mu_full"].shape[0] == 4
    capped_bytes = capped.record["retention"]["quantities"]["waveforms"]["n_bytes"]
    full_bytes = everything.record["retention"]["quantities"]["waveforms"]["n_bytes"]
    assert 0 < capped_bytes < full_bytes
    assert capped.record["retention"]["quantities"]["waveforms"]["n_planned"] == 1
    # Every retained row can be traced back to the segment it came from.
    assert capped.retained["waveforms_sample_index"].shape == (1,)


def test_the_retained_forecast_blocks_carry_the_anchor_and_channel_axes(trained_task):
    r"""$(A_{\max}, H, C_{\mathrm{keep}})$, not a raw window: what is retained has to be the
    tensor that was scored, or a figure drawn from it describes a different forecast."""
    model = trained_task.orig_model
    collection = _collect(
        trained_task, [_labelled_batch(["a", "b"])], caps={"waveforms": 1}, n_total=2
    )
    expected = (
        1,
        expected_anchors_per_sample(model),
        int(model.horizon),
        int(model.decoder_out_channels),
    )

    for name in ("target", "mu_base", "mu_full", "logvar_full"):
        assert collection.retained[name].shape == expected, name
    # The two riders, which cost about 1% of what they travel with and are what locates a
    # contraction on the retained page at all.
    assert collection.retained["up_raw"].shape == (1, int(model.geometry.raw_len))
    assert collection.retained["weight"].shape == (1, int(model.geometry.t))


def test_a_cap_without_a_sample_count_is_refused_rather_than_becoming_a_prefix():
    """A prefix draw over eight concatenated per-subgroup shards yields one subgroup and one
    class -- the predecessor's documented "only 1 class found" failure."""
    with pytest.raises(ValueError, match="n_total"):
        RetentionPlan.build({"waveforms": 4}, n_total=None, seed=0)


def test_the_retained_arrays_survive_a_write_and_a_read(tmp_path, trained_task):
    collection = _collect(
        trained_task, [_labelled_batch(["a", "b"])], caps={"attention": 1}, n_total=2
    )

    write_collection(collection, tmp_path)
    reloaded = load_collection(tmp_path)

    assert reloaded.retained["attn_weights"].shape == collection.retained["attn_weights"].shape


def test_the_horizon_accumulator_is_exact_against_full_retention(trained_task):
    r"""The residuals and log-variances are $A_{\max} \times H \times C_{\mathrm{keep}}$ per
    sample and are streamed rather than kept. This is the check that streaming them loses nothing:
    the same sums, computed once over every retained tensor at the end, must agree to float64
    precision -- the only difference between the two is the order of the additions.
    """
    batches = [_labelled_batch(["a", "b"]), _labelled_batch(["c", "d"], seed=1)]
    model = trained_task.orig_model
    streamed = {}
    kept = {"target": [], "mu_full": [], "logvar_full": [], "mask": []}

    def _sink(batch, readout):
        for name, value in readout.horizon_sums.items():
            streamed[name] = streamed.get(name, 0.0) + value.to(torch.float64)
        for name in ("target", "mu_full", "logvar_full"):
            kept[name].append(readout.retained[name])
        y_st, y_ph, u_stream, _target_features, weight = model_inputs(trained_task, batch)
        phase, stride = DENSE_ANCHOR_GEOMETRY
        with torch.no_grad():
            outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
        mask, _coverage = forecast_mask(
            weight, model.geometry, coverage_floor=model.coverage_floor,
            anchors=outputs["anchor_index"], anchor_valid=outputs["anchor_valid"],
        )
        kept["mask"].append(mask)

    torch.manual_seed(_SEED)
    evaluate(
        trained_task,
        _StubLoader(batches),
        num_samples=1,
        perm_generator=torch.Generator().manual_seed(_SEED),
        mc_generator=torch.Generator().manual_seed(_SEED),
        retain=("target", "mu_full", "logvar_full"),
        on_batch=_sink,
    )

    reference = horizon_residual_sums(
        torch.cat(kept["mu_full"]),
        torch.cat(kept["logvar_full"]),
        torch.cat(kept["target"]),
        torch.cat(kept["mask"]),
    )
    for name, value in reference.items():
        assert streamed[f"full_{name}"].numpy() == pytest.approx(value.numpy(), rel=1e-9)
    assert float(reference["count"].sum()) > 0.0, "an all-masked fixture would agree vacuously"


def test_the_horizon_accumulator_resolves_the_axis_neither_table_carries(trained_task):
    r"""$\tau$ lives *inside* an anchor, so it survives on neither table -- which is the whole
    reason it is accumulated rather than re-derived."""
    horizon = _collect(trained_task, [_labelled_batch(["a", "b"])]).record["horizon"]
    expected = int(trained_task.orig_model.horizon)

    # Driven from the module's own constant rather than a literal list: the block grows as later
    # readouts need another per-tau accumulator, and a literal here would have to be edited every
    # time without asserting anything the constant does not.
    assert set(horizon) == {
        f"{branch}_{statistic}"
        for branch in ("base", "full")
        for statistic in HORIZON_STATISTICS
    }
    assert all(len(values) == expected for values in horizon.values())
    # The denominator is the per-tau masked count, not the per-anchor contributing indicator --
    # which is an amax over tau and would count a masked forecast step as a scored zero.
    assert all(value > 0.0 for value in horizon["full_count"])
    assert all(value > 0.0 for value in horizon["full_n_anchors"])


# =================================================================================================
# The record: geometry, channel axis, units and cost
# =================================================================================================
def test_the_record_carries_the_anchor_geometry_the_pass_ran_at(trained_task):
    """An offline analysis holds no model, so the anchor set has to be written down by the one
    pass that does -- including the training stride, without which a table read against the
    training CSV is unreadable: the anchor count differs by a factor of it."""
    model = trained_task.orig_model
    record = _collect(trained_task, [_labelled_batch(["a", "b"])]).record["geometry"]

    assert (record["anchor_phase"], record["anchor_stride"]) == DENSE_ANCHOR_GEOMETRY
    assert record["training_anchor_stride"] == int(model.anchor_stride)
    assert record["training_anchor_stride"] != record["anchor_stride"], (
        "a fixture trained at the dense stride would make the distinction vacuous"
    )
    assert record["anchor_floor"] == int(model.warmup_period)
    assert record["anchors_per_sample"] == expected_anchors_per_sample(model)
    assert record["anchor_first"] == int(model.warmup_period)
    assert record["anchor_last"] == int(model.geometry.t_valid) - 1
    assert (record["t"], record["t_valid"]) == (
        int(model.geometry.t), int(model.geometry.t_valid)
    )


def test_the_record_persists_the_surviving_channel_index_not_only_its_width(trained_task):
    """The join the band-resolved skill readout rests on. The per-channel vectors are over the
    **kept** channels while a channel-to-band map is over the **declared** ones, and the analyses
    layer may not ask the model which is which -- an offline re-run holds none. A width alone
    cannot say *which* declared channels survived."""
    model = trained_task.orig_model
    record = _collect(trained_task, [_labelled_batch(["a", "b"])]).record["geometry"]

    assert record["target_declared_width"] == int(model.c_y)
    assert record["target_kept_width"] == int(model.decoder_out_channels)
    assert record["target_keep_index"] == [
        int(value) for value in model.target_gate.keep_index.tolist()
    ]
    assert len(record["target_keep_index"]) == record["target_kept_width"]
    assert record["target_kept_width"] < record["target_declared_width"], (
        "an ungated fixture would make the distinction vacuous"
    )
    assert record["block_width"] == record["horizon"] * record["target_kept_width"]


def test_the_record_carries_the_models_own_bound_conventions(trained_task):
    """A clamp is a property of the checkpoint: nothing reconciles it against a config file, so a
    margin drawn from the merged configuration would quote whatever that file currently says."""
    model = trained_task.orig_model
    bounds = _collect(trained_task, [_labelled_batch(["a", "b"])]).record["bounds"]

    assert bounds["logvar_clamp"] == [
        float(model.logvar_clamp[0]), float(model.logvar_clamp[1])
    ]
    assert bounds["mu_scale"] == pytest.approx(float(model.mu_scale))
    assert bounds["logvar_margin"] == pytest.approx(
        bounds["logvar_margin_frac"] * (bounds["logvar_clamp"][1] - bounds["logvar_clamp"][0])
    )


def test_the_record_says_it_has_no_block_statistics_rather_than_omitting_them(trained_task):
    """The stub loader has no dataset, so this is the honest-empty case: a run without statistics
    still produces every number, in the loader's own units. The populated case needs a real
    loader and is covered where there is one."""
    record = _collect(trained_task, [_labelled_batch(["a", "b"])]).record

    assert record["normalization"] == {}
    assert record["likelihood"] == trained_task.hparams["likelihood"]
    assert NORMALIZED_BLOCKS == ("fhr_st", "fhr_ph", "up_st", "up_ph"), (
        "the four stored blocks are the only scales any number in a run is on"
    )


def test_no_coherence_sidecar_is_written_and_the_record_says_why(tmp_path, trained_task):
    """An absent file is indistinguishable from a pass that failed to write one. A stored
    coefficient is a modulus, so the estimator the raw pipeline streams sums for cannot exist
    here at any window length, and the record states that rather than leaving it to be inferred."""
    collection = _collect(trained_task, [_labelled_batch(["a", "b"])])
    write_collection(collection, tmp_path)

    assert not list(tmp_path.glob("*coherence*"))
    assert not list(tmp_path.glob("*spectra*"))
    record = collection.record["coherence"]
    assert record["ported"] is False
    assert "modulus" in record["reason"]
    # And no cross-spectral family leaked into the row-aligned sidecar either.
    with np.load(tmp_path / VECTORS_FILENAME) as handle:
        assert set(handle.files) == set(VECTOR_READOUTS)


def test_the_pass_records_what_it_cost_and_the_rate_a_longer_one_extrapolates_from(trained_task):
    """A recorded measurement, not a threshold: a CI box's timing is not a production box's, so
    a bound here would either be met by accident or fail for the machine. What an operator needs
    before starting a multi-hour pass is the rate this one ran at."""
    cost = _collect(trained_task, [_labelled_batch(["a", "b"])]).record["cost"]

    assert cost["num_mc_samples"] == 1
    assert cost["n_samples"] == 2 and cost["n_batches"] == 1
    assert cost["mean_batch_size"] == pytest.approx(2.0)
    for name in ("elapsed_s", "seconds_per_batch", "samples_per_second", "hours_per_1000_samples"):
        assert math.isfinite(cost[name]) and cost[name] > 0.0, name
    assert cost["hours_per_1000_samples"] == pytest.approx(
        1000.0 / cost["samples_per_second"] / 3600.0
    )
    # Absent rather than zero off CUDA: the allocator that reports it does not exist there, and a
    # 0 would read as "measured, and the pass used nothing".
    assert cost["device"] == "cpu" and cost["peak_allocated_bytes"] is None
    assert "hours_per_1000_samples" in cost["note"]


def test_the_per_sample_table_round_trips_through_disk_bit_for_bit(trained_task, tmp_path):
    """A re-run reads this file where the pass that wrote it held a frame, so any loss here makes
    one run report two sets of numbers.

    ``pandas`` writes a float in its shortest exactly-round-tripping form but reads it back with a
    fast parser that is not exact, so the *default* round trip silently drops the last bits -- and
    a per-recording mean amplifies that through the cancellation in a quantity like the mean signed
    error until it reaches the sixth digit of a reported statistic.
    """
    collection = _collect(
        trained_task, [_labelled_batch(["a", "b"]), _labelled_batch(["c", "d"], seed=1)]
    )
    write_collection(collection, tmp_path)

    reloaded = load_collection(tmp_path).per_sample
    numeric = collection.per_sample.select_dtypes("number")

    assert list(numeric.columns), "a frame with no numeric column would agree vacuously"
    for name in numeric.columns:
        original = np.asarray(numeric[name], dtype=np.float64)
        recovered = np.asarray(reloaded[name], dtype=np.float64)
        assert np.array_equal(original, recovered, equal_nan=True), name


def test_the_vectors_sidecar_is_row_aligned_with_the_per_sample_table(trained_task, tmp_path):
    """Positional, and asserted by a round trip: every consumer indexes the sidecar by the
    per-sample table's row number, so a family one row short would put one segment's spectrum on
    another segment's row from the gap onward."""
    collection = _collect(
        trained_task, [_labelled_batch(["a", "b"]), _labelled_batch(["c", "d"], seed=1)]
    )
    write_collection(collection, tmp_path)

    reloaded = load_collection(tmp_path)

    assert len(reloaded.per_sample) == 4
    for name, rows in reloaded.vectors.items():
        assert rows.shape[0] == len(reloaded.per_sample), name
        assert np.array_equal(rows, collection.vectors[name], equal_nan=True), name


# =================================================================================================
# Discovery and provenance
# =================================================================================================
def _provenance_inputs(tmp_path):
    """A checkpoint-shaped file and the eval_config a collection is keyed on."""
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"a checkpoint")
    return checkpoint, {"seed": 3, "caps": {}}


def _write_tables(trained_task, results_dir, checkpoint, eval_config) -> Collection:
    """Collect once into ``results_dir``, the way a first run does."""
    return load_or_collect(
        results_dir,
        lambda: _collect(trained_task, [_labelled_batch(["a", "b"])]),
        checkpoint_path=checkpoint,
        eval_config=eval_config,
        num_samples=1,
    )


def test_a_second_invocation_reads_the_tables_and_runs_no_pass(tmp_path, trained_task):
    """The offline promise, made real: the forward pass is the whole cost of a run, and an
    analysis re-run against a finished directory must not pay it twice."""
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    first = _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)

    def _must_not_run() -> Collection:
        raise AssertionError("the forward pass ran against a finished directory")

    second = load_or_collect(
        tmp_path / "run",
        _must_not_run,
        checkpoint_path=checkpoint,
        eval_config=eval_config,
        num_samples=1,
    )

    assert second.from_cache is True
    assert len(second.per_sample) == len(first.per_sample)
    # The readouts travel with the tables, so a cached directory answers what the pass answered.
    assert second.results["readouts"] == first.results["readouts"]


def test_tables_from_another_checkpoint_are_refused_and_both_are_named(tmp_path, trained_task):
    """Not re-collected on top: a directory holding two checkpoints' rows under one summary is
    not something a reader can unpick afterwards."""
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"a different checkpoint")

    with pytest.raises(TablesProvenanceMismatch) as excinfo:
        load_or_collect(
            tmp_path / "run",
            lambda: pytest.fail("should not collect"),
            checkpoint_path=other,
            eval_config=eval_config,
            num_samples=1,
        )

    message = str(excinfo.value)
    assert "fake.ckpt" in message and "other.ckpt" in message


def test_tables_collected_under_another_seed_are_refused(tmp_path, trained_task):
    """Every number in them depends on it: the Monte Carlo draw, the derangement and the loader
    order all follow from the seed."""
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)

    with pytest.raises(TablesProvenanceMismatch, match="eval_config.seed"):
        load_or_collect(
            tmp_path / "run",
            lambda: pytest.fail("should not collect"),
            checkpoint_path=checkpoint,
            eval_config=dict(eval_config, seed=99),
            num_samples=1,
        )


def test_tables_collected_at_another_draw_count_are_refused(tmp_path, trained_task):
    """$K$ decides the marginalised estimator's own value, so two draw counts are two different
    headline numbers reported under one name."""
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)

    with pytest.raises(TablesProvenanceMismatch, match="Monte Carlo"):
        load_or_collect(
            tmp_path / "run",
            lambda: pytest.fail("should not collect"),
            checkpoint_path=checkpoint,
            eval_config=eval_config,
            num_samples=4,
        )


def test_tables_collected_under_another_eval_config_are_refused(tmp_path, trained_task):
    """The digest covers the whole block, so a moved threshold -- the availability-clock margin
    among them -- is caught even though no other key changed."""
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)

    with pytest.raises(TablesProvenanceMismatch, match="eval_config block"):
        load_or_collect(
            tmp_path / "run",
            lambda: pytest.fail("should not collect"),
            checkpoint_path=checkpoint,
            eval_config=dict(eval_config, clock_margin_min_nats=0.5),
            num_samples=1,
        )


def test_a_truncated_table_is_refused(tmp_path, trained_task):
    """A run killed mid-write leaves exactly this, and it reads as a smaller population rather
    than as a broken file."""
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)
    table = tmp_path / "run" / PER_SAMPLE_FILENAME
    lines = table.read_text(encoding="utf-8").splitlines()
    table.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(TablesProvenanceMismatch, match="truncated"):
        load_collection(tmp_path / "run")


def test_a_directory_whose_verdict_registry_has_moved_is_refused_by_name(tmp_path, trained_task):
    """The reuse path's second refusal, and it cannot be left to the ordering guard: that one runs
    over the list a *fresh* pass builds and is never reached here, so a directory collected under
    the earlier criteria would be re-reported verbatim -- a summary silently missing a criterion,
    which reads exactly like one where the criterion passed."""
    from teb_vae.lag_attn_cfs.eval.metrics import StaleCachedVerdicts

    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)
    record_path = tmp_path / "run" / COLLECTION_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["results"]["verdicts"] = [
        entry for entry in record["results"]["verdicts"]
        if entry.get("name") != "coupling_exceeds_availability_clock"
    ]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(StaleCachedVerdicts, match="coupling_exceeds_availability_clock"):
        load_or_collect(
            tmp_path / "run",
            lambda: pytest.fail("should not collect"),
            checkpoint_path=checkpoint,
            eval_config=eval_config,
            num_samples=1,
        )


def test_the_sidecar_records_what_the_tables_were_collected_from(tmp_path, trained_task):
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)

    record = json.loads((tmp_path / "run" / COLLECTION_FILENAME).read_text(encoding="utf-8"))

    assert record["provenance"]["checkpoint"]["sha256"]
    assert record["provenance"]["seed"] == 3
    assert record["provenance"]["eval_config_digest"]
    assert record["n_per_sample_rows"] == 2


# =================================================================================================
# Observability
#
# The collection pass is the multi-hour step of a production run and every other step takes
# seconds, so silence here is silence for the whole run -- and an operator who cannot tell a slow
# pass from a hung one restarts a healthy one.
# =================================================================================================
def _captured_logs(function, level: str = "INFO"):
    """Run ``function`` with a loguru sink attached and return the messages it emitted."""
    from loguru import logger

    messages: List[str] = []
    sink_id = logger.add(messages.append, level=level)
    try:
        function()
    finally:
        logger.remove(sink_id)
    return messages


def test_the_pass_reports_its_throughput_and_a_remaining_estimate(
    trained_task, monkeypatch
) -> None:
    monkeypatch.setattr("teb_vae.lag_attn_cfs.eval.collect.PROGRESS_EVERY_BATCHES", 1)
    # Two recordings per batch: a batch holding one has no stranger in it to borrow a source
    # from, so the permutation control excludes it whole and it never reaches the sink.
    batches = [
        _labelled_batch(["A", "B"], seed=1),
        _labelled_batch(["C", "D"], seed=2, epoch_offset=-30000.0),
    ]

    messages = _captured_logs(lambda: _collect(trained_task, batches, n_total=4))

    progress = [line for line in messages if "collection:" in line]
    assert len(progress) == 2, progress
    assert "samples/s" in progress[0]
    assert "min remaining" in progress[0]
    assert "2/4 sample(s)" in progress[0]


def test_a_loader_that_cannot_say_how_long_it_is_gets_throughput_without_an_estimate(
    trained_task, monkeypatch
) -> None:
    """An estimate against an unknown total would be a number with no meaning; the throughput is
    still worth logging."""
    monkeypatch.setattr("teb_vae.lag_attn_cfs.eval.collect.PROGRESS_EVERY_BATCHES", 1)

    messages = _captured_logs(
        lambda: _collect(trained_task, [_labelled_batch(["A", "B"])], n_total=0)
    )

    progress = [line for line in messages if "collection:" in line]
    assert progress and "no total" in progress[0]
    assert "remaining" not in progress[0]


# =================================================================================================
# Against the real run
# =================================================================================================
@pytest.mark.slow
def test_the_run_leaves_both_tables_beside_its_summary(collected_run):
    """The demo: both tables open in pandas, with the class, subgroup and anchor columns on
    them."""
    results_dir = Path(collected_run["results_dir"])
    collection = load_collection(results_dir)

    assert (results_dir / PER_SAMPLE_FILENAME).is_file()
    assert (results_dir / PER_ANCHOR_FILENAME).is_file()
    assert collected_run["summary"]["collection"]["n_per_sample_rows"] == len(
        collection.per_sample
    )
    assert "results" not in collected_run["summary"]["collection"], (
        "the summary carries the readouts once"
    )
    assert set(collection.per_sample["subgroup"]), "the generated cohort shards are labelled"


@pytest.mark.slow
def test_a_real_run_records_the_per_block_statistics_it_normalised_with(
    collected_run, cohort_loader
):
    """Without these a reader cannot say what scale a reported coefficient is on -- and nothing
    in this pipeline converts one, so the record is the only statement of it."""
    stats = cohort_loader.dataset.get_normalization_stats()
    record = load_collection(collected_run["results_dir"]).record["normalization"]

    assert set(record) == set(NORMALIZED_BLOCKS)
    for name in NORMALIZED_BLOCKS:
        assert record[name]["n_channels"] == len(np.asarray(stats[name]["mean"]).reshape(-1))
        assert record[name]["mean"] == pytest.approx(
            [float(value) for value in np.asarray(stats[name]["mean"]).reshape(-1)]
        )


@pytest.mark.slow
def test_a_real_run_retains_what_the_shipped_caps_ask_for(collected_run):
    """The committed delta sets three caps, and a run that silently retained nothing would leave
    every figure built on them mysteriously absent rather than reported as skipped."""
    retention = load_collection(collected_run["results_dir"]).record["retention"]

    for quantity in RETAINED_QUANTITIES:
        entry = retention["quantities"][quantity]
        assert entry["cap"] != "absent", quantity
        assert entry["n_kept"] > 0, quantity
        assert entry["n_bytes"] > 0, quantity
