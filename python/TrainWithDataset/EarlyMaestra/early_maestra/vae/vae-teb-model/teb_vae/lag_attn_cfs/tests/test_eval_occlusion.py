r"""The interventional readout: what a band of the source's own values was worth, in nats.

Every other lag readout in this pipeline is **observational** -- it reads a KL attribution or an
attention weight and asks where the model looked. A weight says where it looked, not what looking
there was worth, and on this family's geometry the observational answer is additionally pinned:
the pooled argmax sits at the window's near censoring edge on every arm measured, because a
physiological delay shorter than what lag $0$ encodes is reported *at* lag $0$. This analysis asks
a different question, and one the near edge cannot pin: remove the source's values in a lag band,
re-encode, and measure what the forecast loses, resolved by horizon step.

Four properties make the number readable, and each is a way it could be silently wrong:

**The announcement must not move.** Zeroing the source's values in a band and leaving its
availability pattern alone is what separates "the content at those lags was worth this much" from
"the source's arrival time was worth this much" -- and the second is the confound this whole
revision exists to control. The invariance is *measured* here rather than argued.

**The band is occluded after the channel gate.** The gate shifts each channel onto the run's common
clock, so a band of gated steps is one lag range for every kept channel at once; the same band on
the stored stream would land at $\ell + d_c$ for channel $c$ and re-smear precisely the axis the
alignment exists to un-smear.

**One anchor per segment, held fixed across bands.** The source pathway has memory, so a band
occluded relative to anchor $a$ contaminates every later anchor's state -- a second anchor scored
in the same forward would attribute one anchor's loss to another's band. The anchor is drawn from a
seeded generator over the anchors the forward marked valid, and the reference arm and every band
score at the *same* draw, which is what makes the difference paired.

**Common random numbers.** Every arm reseeds one generator for its latent draw, so two arms differ
in the band and in nothing else. Without it the deltas would carry the sampling noise of two
independent draws of a $d_z$-dimensional Gaussian, which at these magnitudes is most of them.
"""
from __future__ import annotations

import types
from typing import Any, Dict

import numpy as np
import torch

from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import occlusion as occlusion_analysis

from .conftest import TINY_SEQ_LEN, make_stub_batch

#: A four-band partition of the tiny model's lag window, in the shipped band names. Contiguous and
#: covering $[0, 8]$ exactly once, which is the shape the production delta has at the production
#: window -- the widths are a reading convenience and nothing here asserts them.
TINY_BANDS: Dict[str, tuple] = {
    "anchor": (0, 2),
    "near": (3, 4),
    "mid": (5, 6),
    "far": (7, 8),
}

#: The analysis's own block, instant and seeded.
EVAL_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "occlusion_bands": {name: list(span) for name, span in TINY_BANDS.items()},
    "caps": {"occlusion": 8},
}


def _context(task=None, loader=None) -> AnalysisContext:
    """An analysis context carrying only what this analysis reads."""
    return AnalysisContext(collection=types.SimpleNamespace(), config={}, task=task, loader=loader)


class _OneBatchLoader:
    """A loader yielding one stub batch, which is all the analysis iterates for."""

    def __init__(self, batch) -> None:
        self._batch = batch

    def __iter__(self):
        yield self._batch


def _running_task(task_factory) -> Any:
    """A task whose model carries the shipped switches, in eval mode with the device set.

    The switches are on because the interaction is what this analysis has to survive: the prior's
    clock is an encode of the source pathway the intervention edits, and the persistence residual
    is a per-anchor tensor the occluded arm has to narrow to the same scored anchor the reference
    used. Both were defects in the first draft and neither is visible at the shipped defaults off.
    """
    from .conftest import TINY_STRIDE, tiny_warmup_kwargs

    module = task_factory(
        model_kwargs=tiny_warmup_kwargs(
            anchor_stride=TINY_STRIDE,
            prior_availability_input=True,
            persistence_residual=True,
        )
    )
    module.eval()
    return module


# =================================================================================================
# The band, as a set of source steps
# =================================================================================================
def test_a_band_is_the_same_lag_range_whatever_anchor_a_sample_drew() -> None:
    r"""The band is anchored **per sample**, not fixed in absolute time.

    Two samples drawing different anchors must have the same *lags* removed, which means different
    absolute steps. The alternative -- one absolute band, every anchor scored -- gives each anchor a
    different relative band and smears exactly the axis this readout exists to resolve.
    """
    anchors = torch.tensor([12, 20])
    mask = occlusion_analysis.band_mask(anchors, (3, 5), sequence_length=TINY_SEQ_LEN)

    assert mask.shape == (2, TINY_SEQ_LEN)
    assert torch.nonzero(mask[0]).flatten().tolist() == [7, 8, 9]
    assert torch.nonzero(mask[1]).flatten().tolist() == [15, 16, 17]


def test_a_band_reaching_before_the_recording_starts_is_simply_shorter() -> None:
    """The honest state rather than an error: the recording does not reach that far back, and the
    live fraction is what reports it. A mask that wrapped, clamped or raised would each turn a
    short band into a claim about steps that do not exist."""
    mask = occlusion_analysis.band_mask(
        torch.tensor([2]), (0, 8), sequence_length=TINY_SEQ_LEN
    )

    assert torch.nonzero(mask[0]).flatten().tolist() == [0, 1, 2]


def test_the_bands_partition_the_window_without_overlapping() -> None:
    """A property of the configured partition rather than of the code, asserted because an overlap
    is invisible in the output: two bands sharing a lag would each be charged for it, and their
    deltas would sum to more than removing the union costs."""
    anchors = torch.tensor([12])
    covered: list = []
    for band in TINY_BANDS.values():
        covered.extend(
            torch.nonzero(
                occlusion_analysis.band_mask(anchors, band, sequence_length=TINY_SEQ_LEN)[0]
            )
            .flatten()
            .tolist()
        )

    assert len(covered) == len(set(covered)), "two bands occlude the same source step"
    assert sorted(covered) == list(range(12 - 8, 12 + 1))


# =================================================================================================
# The anchor draw
# =================================================================================================
def test_the_anchor_is_drawn_only_from_the_columns_the_forward_marked_valid() -> None:
    """A padded anchor slot scores nothing, so a draw that could land on one would silently reduce
    the sample count on some runs and not others. Repeated because the draw is random: a single
    trial would pass on a broken sampler most of the time."""
    valid = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    generator = torch.Generator().manual_seed(0)

    for _ in range(32):
        columns = occlusion_analysis.choose_anchors(valid, generator)
        assert columns[0].item() in (0, 2)
        assert columns[1].item() == 3


def test_a_sample_with_no_valid_anchor_yields_a_column_rather_than_raising() -> None:
    """A real state on a short recording, and one ``multinomial`` refuses outright on a row of
    zeros. The column is arbitrary because the forecast mask drops the sample anyway, so what
    matters is that one sample's absent anchors cannot take the whole batch down."""
    valid = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    columns = occlusion_analysis.choose_anchors(valid, torch.Generator().manual_seed(0))

    assert columns.shape == (2,)
    assert columns[1].item() == 0


def test_the_anchor_draw_is_reproducible_from_the_runs_own_seed() -> None:
    """The whole comparison is paired at one anchor per segment, so a draw that moved between the
    reference arm and a band would make the delta a comparison of two populations."""
    valid = torch.ones(6, 5)

    first = occlusion_analysis.choose_anchors(valid, torch.Generator().manual_seed(3))
    second = occlusion_analysis.choose_anchors(valid, torch.Generator().manual_seed(3))

    assert torch.equal(first, second)


# =================================================================================================
# The intervention, on a real model
# =================================================================================================
def test_the_announcement_is_invariant_under_every_occluded_encode(task) -> None:
    """The confound this analysis exists to avoid, **measured** rather than argued.

    If the intervention moved the availability announcement as well as the values, the delta would
    be the cost of removing the source's arrival *time* -- which is the availability clock, the
    exact quantity the family's coupling readouts already have to be defended against. Zero rather
    than small: the announcement is built from registered buffers, so anything but exact equality
    would mean something wrote to them.
    """
    module = _running_task(task)
    record = occlusion_analysis.collect_batch(
        module, make_stub_batch(seed=1), bands=TINY_BANDS, seed=0
    )

    assert record["announcement_max_abs_change"] == 0.0


def test_every_band_scores_the_same_anchor_as_the_reference_arm(task) -> None:
    """Paired, which is what makes a difference of two block scores a delta rather than a
    comparison of two populations. The anchor is drawn once per batch and reused by the reference
    and every band, so the per-band delta arrays all have the reference's shape."""
    module = _running_task(task)
    record = occlusion_analysis.collect_batch(
        module, make_stub_batch(seed=1), bands=TINY_BANDS, seed=0
    )

    assert record["reference"].ndim == 2
    assert record["reference"].shape[1] == module.orig_model.horizon
    assert set(record["deltas"]) == set(TINY_BANDS)
    for name, delta in record["deltas"].items():
        assert delta.shape == record["reference"].shape, name


def test_the_whole_collection_is_bit_identical_when_repeated(task) -> None:
    """Common random numbers, end to end. Every arm reseeds one generator for its latent draw, so
    two arms of one collection differ in the band and in nothing else -- and two collections at one
    seed differ in nothing at all.

    Without this the deltas would carry the sampling noise of two independent draws of a
    $d_z$-dimensional Gaussian, which at these magnitudes is most of them: the readout would look
    like a measurement and be a random number.
    """
    module = _running_task(task)
    batch = make_stub_batch(seed=1)

    first = occlusion_analysis.collect_batch(module, batch, bands=TINY_BANDS, seed=0)
    second = occlusion_analysis.collect_batch(module, batch, bands=TINY_BANDS, seed=0)

    assert np.array_equal(first["anchors"], second["anchors"])
    assert np.array_equal(first["reference"], second["reference"])
    for name in TINY_BANDS:
        assert np.array_equal(first["deltas"][name], second["deltas"][name]), name


def test_an_empty_occlusion_is_exactly_the_reference_arm(task) -> None:
    """The mechanism's own fixed point, and the strongest available check that the reference and the
    bands are one computation.

    A band placed entirely before the recording starts removes nothing, so its delta must be
    identically zero -- not small. Anything else means the two arms differ for a reason other than
    the occlusion: a second latent draw, a different anchor, a decode the reference did not do.
    """
    module = _running_task(task)
    # Lags far beyond the sequence: every masked step falls off the front for every anchor.
    beyond = {"beyond": (TINY_SEQ_LEN + 10, TINY_SEQ_LEN + 20)}

    record = occlusion_analysis.collect_batch(
        module, make_stub_batch(seed=1), bands=beyond, seed=0
    )

    assert np.array_equal(record["deltas"]["beyond"], np.zeros_like(record["reference"]))
    assert record["live_fraction"]["beyond"] == 0.0 or np.isnan(
        record["live_fraction"]["beyond"]
    )


def test_the_live_fraction_reports_how_much_of_a_band_held_a_value(task) -> None:
    """The column that separates "the source did not matter at those lags" from "there was less
    source there to remove".

    A band reaching into the warm-up region finds a stream the availability mechanism has already
    zeroed, so its delta is near zero for a reason that says nothing about the model. Measured on
    the *gated* stream the intervention actually edits, and bounded to $[0, 1]$ because it is a
    fraction of positions rather than a magnitude.
    """
    module = _running_task(task)
    record = occlusion_analysis.collect_batch(
        module, make_stub_batch(seed=1), bands=TINY_BANDS, seed=0
    )

    assert set(record["live_fraction"]) == set(TINY_BANDS)
    for name, fraction in record["live_fraction"].items():
        assert 0.0 <= float(fraction) <= 1.0, (name, fraction)


# =================================================================================================
# The analysis's own contract
# =================================================================================================
def test_a_pass_with_no_model_records_a_skip_rather_than_assuming_one(tmp_path) -> None:
    """The offline re-run, which is what the whole analysis protocol exists for. Every other
    analysis reads the durable tables and works with no checkpoint; this one cannot, because the
    forward it needs is one that never happened -- so it says so instead of failing the run."""
    record = occlusion_analysis.run_occlusion_analysis(
        _context(), eval_config=EVAL_CONFIG, output_dir=tmp_path
    )

    assert set(REQUIRED_RESULT_KEYS) <= set(record)
    assert record["skipped"] is True
    assert "loader" in record["reason"] and "model" in record["reason"]


def test_a_configuration_naming_no_band_is_a_skip_rather_than_an_empty_table(tmp_path) -> None:
    """A configuration state rather than a failure: the analysis measures what a named band was
    worth and has not been told which bands to remove. An empty table would read as four bands that
    all cost nothing."""
    record = occlusion_analysis.run_occlusion_analysis(
        _context(), eval_config=dict(EVAL_CONFIG, occlusion_bands={}), output_dir=tmp_path
    )

    assert record["skipped"] is True
    assert "band" in record["reason"]


def test_the_analysis_writes_its_tables_and_its_headline(task, tmp_path) -> None:
    """End to end on one batch, through the analysis's own entry point rather than through
    ``collect_batch``: what a run reads is the summary and the two CSVs, and a reduction that lost
    a band between the collection and the frames would be invisible to every test above."""
    module = _running_task(task)
    record = occlusion_analysis.run_occlusion_analysis(
        _context(task=module, loader=_OneBatchLoader(make_stub_batch(seed=1))),
        eval_config=EVAL_CONFIG,
        output_dir=tmp_path,
    )

    assert set(REQUIRED_RESULT_KEYS) <= set(record)
    assert "skipped" not in record, record.get("reason")
    for name in record["files"]:
        assert (tmp_path / (occlusion_analysis.ANALYSIS_DIRNAME + "/" + name)).is_file(), name

    # Every band survives the reduction into the summary, which is the step a lost band would go
    # missing in: the collection returns a delta per band, the frames stack them, and a name
    # dropped between the two would leave a table that reads as three bands rather than four.
    assert {row["band"] for row in record["bands"]} == set(TINY_BANDS)
    assert record["plan"]["anchors_per_segment"] == 1
    assert record["unit"] == occlusion_analysis.NATS_PER_ANCHOR_STEP

    headline = record["headline"]
    assert headline["band"] in TINY_BANDS
    assert headline["n_bands"] == len(TINY_BANDS)
    assert np.isfinite(headline["delta_total_nats"])

    # And the confound measurement reaches the record rather than only the collection: this is
    # where a reader of `summary.json` sees that the intervention moved values and not the clock.
    invariance = record["announcement_invariance"]
    assert invariance["n_batches_checked"] == 1
    assert invariance["max_abs_change"] == 0.0


def test_the_caveat_travels_with_the_number(task, tmp_path) -> None:
    """What the delta can be read as is not recoverable from the delta, so the statement bounding
    it is part of the record rather than of this file's docstring. Three things a reader has to
    carry: the announcement was held fixed, one anchor per segment was scored, and a band with a
    small live fraction says nothing about the source."""
    caveat = occlusion_analysis.OCCLUSION_CAVEAT

    assert "announcement" in caveat
    assert "one anchor per segment" in caveat
    assert "live fraction" in caveat


def test_the_delta_is_stated_in_the_unit_every_other_forecast_number_is_in() -> None:
    """Nats of the block score per anchor per horizon step, which is the same scale as ``pred_gap``
    and every ``nll_*`` -- so the interventional readout can be put beside them without a
    conversion nobody would remember to apply."""
    assert "nats" in occlusion_analysis.NATS_PER_ANCHOR_STEP
    assert "horizon step" in occlusion_analysis.NATS_PER_ANCHOR_STEP
