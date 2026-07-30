r"""The shared collection pass and the two tables every later analysis reads.

Three properties carry this suite, and each of them is a way the pipeline could report a complete
set of plausible numbers while measuring something else.

**The tables must be a census of what was scored.** One row per scored segment, one row per
contributing anchor, and the two must agree with the readouts computed in the same pass. A table
one row short is not a visible failure -- it is a slightly different population, reported with the
same confidence.

**A segment that measured nothing must not read as a segment that measured zero.** The per-sample
mean clamps its denominator to $1$, so an unscored segment's columns come out as exactly ``0.0``.
Averaged into a summed-$480$-sample block score of hundreds of nats, that pulls the headline
toward zero and shrinks ``pred_gap``, and nothing else in the output moves.

**Reuse must be safe or refused.** Skipping the forward pass is what makes an offline re-run
possible; reusing another checkpoint's rows under this run's summary is what makes it dangerous,
and only one of the two may be silent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval.collect import (
    COLLECTION_FILENAME,
    HORIZON_STATISTICS,
    PER_ANCHOR_FILENAME,
    PER_ANCHOR_KEY,
    PER_SAMPLE_FILENAME,
    Collection,
    RetentionPlan,
    TablesProvenanceMismatch,
    check_per_anchor_key,
    collect_tables,
    load_collection,
    load_or_collect,
    write_collection,
)
from teb_vae.lag_attn_rws.eval.metrics import evaluate, horizon_residual_sums
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

from .conftest import make_stub_batch

#: Any seed; every pass below fixes the global stream too, because the model's own
#: reparameterisation draws from it and a reference pass must see the same latent.
_SEED = 7

#: The analyses that hold a stated exemption from "no analysis touches the model", as ``--skip``
#: spells them. Named here rather than written twice, and pinned against the protocol test that
#: decides the list -- a third exemption must fail *that* test, not silently widen this one.
MODEL_TOUCHING_ANALYSES = "samples,sufficiency"


class _StubLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)
        # Present so the retention plan can draw over the whole index space; a bare list is
        # enough, since nothing indexes it.
        self.dataset = list(range(sum(len(b.guid) for b in self._batches)))

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
    batch = make_stub_batch(batch_size=len(guids), seed=seed)
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
        },
        num_samples=1,
        n_total=n_total,
        perm_generator=torch.Generator().manual_seed(_SEED),
        mc_generator=torch.Generator().manual_seed(_SEED),
    )


# =============================================================================
# The per-sample table
# =============================================================================
@pytest.fixture(scope="session")
def collected(evaluated) -> Collection:
    """The tables the real end-to-end run left behind, read back off disk."""
    return load_collection(evaluated["results_dir"])


def test_the_per_sample_table_holds_one_row_per_scored_sample(collected, evaluated):
    """The row count is the promise every table-driven analysis rests on: a table one row short
    is a different population reported with the same confidence."""
    results = evaluated["summary"]["results"]

    assert len(collected.per_sample) == results["n_samples"]
    assert collected.per_sample["guid"].nunique() == results["n_recordings"]


def test_every_row_carries_the_identity_a_clinical_question_is_asked_in(collected):
    """Attached once here rather than per analysis: the class and the subgroup are properties of
    the sample, so a by-class number is a ``groupby`` on an existing column."""
    frame = collected.per_sample

    for name in (
        "guid", "epoch", "clinical_class", "subgroup", "cs_label", "bg_label",
        "time_from_labor_onset", "source_file_basename", "n_anchors", "n_segments_in_guid",
    ):
        assert name in frame.columns, name
    assert set(frame["clinical_class"]) == {"healthy", "acidosis", "hie"}
    assert frame["subgroup"].notna().all(), "every generated shard is a canonical subgroup"
    assert (frame["n_segments_in_guid"] > 1).any(), "a one-segment GUID aggregates to itself"


def test_the_readout_columns_reach_the_table_beside_the_identity(collected, evaluated):
    """The table is what a later analysis re-derives the headline from, so every scalar readout
    the pass computed has to be on it."""
    frame = collected.per_sample

    for name in evaluated["summary"]["results"]["readouts"]:
        assert name in frame.columns, name
        assert frame[name].notna().any(), name


def test_a_nan_labour_onset_is_preserved_rather_than_dropped(collected):
    """NaN means the recording is absent from the labour-onset table. Dropping the row would
    silently narrow every other readout to the recordings that happen to be in that CSV."""
    frame = collected.per_sample

    assert frame["time_from_labor_onset"].isna().any(), "the fixture writes NaN for one GUID"
    assert frame["time_from_labor_onset"].notna().any(), "and finite values for the others"


def test_a_short_per_sample_column_raises_rather_than_misaligning(trained_task):
    """The batch is the authority on how many samples there are. A column that redefines it puts
    one sample's guid beside another sample's numbers, and the output stays plausible."""
    batch = _labelled_batch(["a", "b"])
    batch.epoch = torch.tensor([-1000.0])

    with pytest.raises(ValueError, match="holds 1 value"):
        _collect(trained_task, [batch])


# =============================================================================
# Zero-anchor accounting
# =============================================================================
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

    assert list(frame["n_anchors"]) == [0, pytest.approx(frame["n_anchors"][1])]
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
    assert set(denominators) >= {"pred_gap", "mc_pred_gap", "delta_mu_rms"}


def test_the_vector_readouts_are_blanked_on_the_same_rule_as_the_scalars(trained_task):
    """The per-dimension KL of a segment that scored no anchors is a row of clamped zeros, and it
    decomposes a headline the segment did not contribute to."""
    batch = _labelled_batch(["a", "b"])
    batch.weight[0] = 0.0

    vectors = _collect(trained_task, [batch]).vectors

    assert np.isnan(vectors["kld_per_dim"][0]).all()
    assert np.isfinite(vectors["kld_per_dim"][1]).all()
    assert vectors["lag_profile"].shape[0] == 2


# =============================================================================
# The per-anchor table
# =============================================================================
def test_the_per_anchor_row_count_is_the_summed_contributing_anchors(collected):
    """The two tables describe one pass, so the anchor table's length is a number the sample
    table already carries -- and a disagreement means one of them is not that pass."""
    expected = int(collected.per_sample["n_anchors"].sum())

    assert len(collected.per_anchor) == expected
    assert expected > 0


def test_the_per_anchor_table_carries_the_anchor_level_columns(collected):
    frame = collected.per_anchor

    for name in (
        "kld_per_t", "nll_base_block", "nll_full_block", "pred_gap",
        "mc_nll_base_block", "mc_nll_full_block", "mc_pred_gap", "argmax_lag", "coverage",
    ):
        assert name in frame.columns, name
    assert frame["anchor"].min() >= 0
    assert frame["coverage"].max() <= 1.0


def test_the_per_anchor_key_is_unique(collected):
    assert not collected.per_anchor.duplicated(subset=list(PER_ANCHOR_KEY)).any()


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


# =============================================================================
# Heavy-artifact retention
# =============================================================================
def test_nothing_heavy_is_retained_unless_a_cap_asks_for_it(trained_task):
    r"""Retention is opt-in. At the shipped geometry the alternative default is $2.4$ MiB per
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
    r"""The residuals and log-variances are $T_{\mathrm{valid}} \times H \times R$ per sample and
    are streamed rather than kept. This is the check that streaming them loses nothing: the same
    sums, computed once over every retained tensor at the end, must agree to float64 precision --
    the only difference between the two is the order of the additions.
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
        mask, _coverage = forecast_mask(
            batch.weight, model.geometry, coverage_floor=model.coverage_floor
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
    expected = int(trained_task.orig_model.geometry.horizon)

    # Driven from the module's own constant rather than a literal list: the block grows as later
    # readouts need another per-tau accumulator -- it grew once already, when the horizon-resolved
    # forecast score landed -- and a literal here would have to be edited every time without
    # asserting anything the constant does not.
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


def test_the_record_carries_the_geometry_the_pass_ran_at(trained_task):
    """An offline analysis holds no model, so the anchor-to-raw geometry has to be written down
    by the one pass that does. Every derived count is checked, not only the four fields, because
    a consumer shading a warm-up prefix reads ``t`` and ``t_valid`` rather than ``raw_len``."""
    geometry = trained_task.orig_model.geometry
    record = _collect(trained_task, [_labelled_batch(["a", "b"])]).record["geometry"]

    assert record == {
        "raw_len": geometry.raw_len,
        "decimation": geometry.decimation,
        "horizon": geometry.horizon,
        "warmup": geometry.warmup,
        "t": geometry.t,
        "t_valid": geometry.t_valid,
        "raw_per_step": geometry.r,
    }


def test_the_record_carries_the_loaders_fhr_normalisation_or_says_it_has_none(trained_task):
    """The stub loader has no dataset, so this is the honest-empty case: a run without statistics
    still produces every number, in the loader's own units. The populated case is covered where
    there is a real loader to read it from."""
    record = _collect(trained_task, [_labelled_batch(["a", "b"])]).record

    assert record["normalization"] == {}
    assert record["likelihood"] == trained_task.hparams["likelihood"]


def test_the_per_sample_table_round_trips_through_disk_bit_for_bit(trained_task, tmp_path):
    """A re-run reads this file where the pass that wrote it held a frame, so any loss here makes
    one run report two sets of numbers.

    ``pandas`` writes a float in its shortest exactly-round-tripping form but reads it back with a
    fast parser that is not exact, so the *default* round trip silently drops the last bits -- and
    a per-recording mean amplifies that through the cancellation in a quantity like the mean signed
    error until it reaches the sixth digit of a reported statistic.
    """
    collection = _collect(trained_task, [_labelled_batch(["a", "b"]), _labelled_batch(["c", "d"])])
    write_collection(collection, tmp_path)

    reloaded = load_collection(tmp_path).per_sample
    numeric = collection.per_sample.select_dtypes("number")

    assert list(numeric.columns), "a frame with no numeric column would agree vacuously"
    for name in numeric.columns:
        original = np.asarray(numeric[name], dtype=np.float64)
        recovered = np.asarray(reloaded[name], dtype=np.float64)
        assert np.array_equal(original, recovered, equal_nan=True), name


def test_a_real_run_records_the_fhr_statistics_it_normalised_with(collected, multi_class_loader):
    """Without these two scalars nothing downstream can say a number in bpm."""
    stats = multi_class_loader.dataset.get_normalization_stats()

    assert collected.record["normalization"]["fhr"]["mean"] == pytest.approx(
        float(stats["fhr"]["mean"])
    )
    assert collected.record["normalization"]["fhr"]["std"] == pytest.approx(
        float(stats["fhr"]["std"])
    )


# =============================================================================
# Discovery and provenance
# =============================================================================
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


def test_the_sidecar_records_what_the_tables_were_collected_from(tmp_path, trained_task):
    checkpoint, eval_config = _provenance_inputs(tmp_path)
    _write_tables(trained_task, tmp_path / "run", checkpoint, eval_config)

    record = json.loads((tmp_path / "run" / COLLECTION_FILENAME).read_text(encoding="utf-8"))

    assert record["provenance"]["checkpoint"]["sha256"]
    assert record["provenance"]["seed"] == 3
    assert record["provenance"]["eval_config_digest"]
    assert record["n_per_sample_rows"] == 2


# =============================================================================
# The run writes them
# =============================================================================
def test_the_run_leaves_both_tables_beside_its_summary(evaluated, collected):
    """The demo: both tables open in pandas, with the class, subgroup and anchor columns on
    them."""
    results_dir = Path(evaluated["results_dir"])

    assert (results_dir / PER_SAMPLE_FILENAME).is_file()
    assert (results_dir / PER_ANCHOR_FILENAME).is_file()
    assert evaluated["summary"]["collection"]["n_per_sample_rows"] == len(collected.per_sample)
    assert "results" not in evaluated["summary"]["collection"], "the summary carries it once"


def test_a_rerun_into_a_finished_directory_touches_no_forward(
    trained_run, repointed_overrides, tmp_path, monkeypatch
):
    """The end-to-end form of the cache: the same command, into the same directory, with the
    model's ``forward`` rigged to explode. Preflight reads weights rather than running anything,
    so a single call would be the pass.

    The two analyses that hold a stated model exemption are skipped on the second run, and the
    skip is the honest statement of what this cache promises. The guarantee is that the
    **collection pass** does not happen twice -- four latent branches over every anchor of the
    split, which is the whole cost of an evaluation. The per-sample pages are a bounded,
    deliberate forward over a handful of chosen segments, re-rendered rather than read off a table
    because a page is the entire forward output of a segment; the sufficiency probe runs the
    encoder once more because the state it fits on is on neither table. Leaving either in would
    make this test assert "nothing forwards", which is not the property the cache has; taking them
    out leaves exactly the property it does have, and the assertion below still fails the moment
    any *other* analysis reaches for the model.
    """
    from teb_vae.lag_attn_rws.eval import run as run_module
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    run_dir = tmp_path / "run"
    summary_path = (
        run_dir / run_module.RESULTS_DIRNAME / run_module.SUMMARY_FILENAME
    )
    run_module.main(
        trained_run, run_dir, overrides=repointed_overrides, device="cpu", num_samples=2,
        skip=MODEL_TOUCHING_ANALYSES,
    )
    first = json.loads(summary_path.read_text(encoding="utf-8"))

    def _explode(*args, **kwargs):
        raise AssertionError("the model was forwarded against a finished run directory")

    monkeypatch.setattr(SeqVaeLagAttnRws, "forward", _explode)
    run_module.main(
        trained_run, run_dir, overrides=repointed_overrides, device="cpu", num_samples=2,
        skip=MODEL_TOUCHING_ANALYSES,
    )
    second = json.loads(summary_path.read_text(encoding="utf-8"))

    # Every finding, identically. Not the artifact manifest or the step timings, which describe
    # the pass rather than the checkpoint and legitimately differ -- which is why they sit beside
    # the results rather than inside them.
    assert second["results"] == first["results"]


def test_only_the_two_exempt_analyses_forward_the_model_after_the_collection_pass(
    trained_run, repointed_overrides, tmp_path, monkeypatch
):
    """The other half of the property above: the exemption is exactly two analyses wide.

    Run everything against a finished directory with ``forward`` rigged to explode. Both exempt
    analyses fail and every other one completes, and the two fail *differently*, which is the
    part worth pinning: ``samples`` catches per page and records the failures by index, so the
    step itself succeeds; ``sufficiency`` cannot fit a probe without an encoder pass, so the whole
    step fails and the run's exit code says so. A third analysis quietly acquiring a forward would
    show up here as a third failure rather than as a slower run.
    """
    from teb_vae.lag_attn_rws.eval import run as run_module
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    run_dir = tmp_path / "run"
    run_module.main(
        trained_run, run_dir, overrides=repointed_overrides, device="cpu", num_samples=2
    )

    def _explode(*args, **kwargs):
        raise AssertionError("forwarded against a finished run directory")

    monkeypatch.setattr(SeqVaeLagAttnRws, "forward", _explode)
    run_module.main(
        trained_run, run_dir, overrides=repointed_overrides, device="cpu", num_samples=2
    )
    summary = json.loads(
        (run_dir / run_module.RESULTS_DIRNAME / run_module.SUMMARY_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    # The pages themselves fail one by one and are recorded by index rather than taking the step
    # down, so `samples` is not on the failed list -- but every page it tried is on its own.
    assert summary["failed"] == ["sufficiency"]
    failures = summary["results"]["samples"]["failures"]
    assert failures and all("forwarded against" in entry["error"] for entry in failures)


# =============================================================================
# Observability
#
# The collection pass is the multi-hour step of a production run and every other step takes
# seconds, so silence here is silence for the whole run -- and an operator who cannot tell a slow
# pass from a hung one restarts a healthy one.
# =============================================================================
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
    monkeypatch.setattr("teb_vae.lag_attn_rws.eval.collect.PROGRESS_EVERY_BATCHES", 1)
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
    monkeypatch.setattr("teb_vae.lag_attn_rws.eval.collect.PROGRESS_EVERY_BATCHES", 1)

    messages = _captured_logs(
        lambda: _collect(trained_task, [_labelled_batch(["A", "B"])], n_total=0)
    )

    progress = [line for line in messages if "collection:" in line]
    assert progress and "no total" in progress[0]
    assert "remaining" not in progress[0]
