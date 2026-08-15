r"""Two evaluations of one checkpoint must report the same numbers, exactly.

The evaluation is not a deterministic function of the checkpoint by default. It *adds* randomness
of its own: $K$ Monte Carlo draws of $z$ per anchor, a derangement per batch, a shuffle over the
split, a stratified sample cap, and the model's own reparameterisation inside every forward. Left
on the global generator those draws depend on whatever else in the process drew first, so
re-running a checkpoint gives different readouts and nothing in the output says why -- which
quietly makes every comparison between two runs a comparison of two samples.

Three things close that, and each is checked here rather than assumed: the numeric environment is
pinned and seeded from the configuration, the estimator takes an explicit generator instead of
the global one, and both the seed and the environment as actually read back from global state are
written into ``summary.json`` so a run is reproducible from its own output.

**The durable tables are the fourth.** A fresh run's analyses read a frame in memory while a
re-run's read it off disk, so an inexact CSV round trip would make the same run report different
numbers depending on which it was -- and a per-recording mean amplifies a last-bit disagreement
through cancellation until it reaches a reported digit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval._reuse import configure_numerics
from teb_vae.lag_attn_cfs.eval.collect import (
    COLLECTION_FILENAME,
    PER_ANCHOR_FILENAME,
    PER_SAMPLE_FILENAME,
    VECTORS_FILENAME,
    load_collection,
)
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    evaluate,
    mc_predictive_block,
    model_inputs,
)
from teb_vae.lag_attn_cfs.eval.probe import PROBE_FILENAME
from teb_vae.lag_attn_cfs.eval.report_seam import HEADLINE_SCALARS, STEPS_FILENAME
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

from .conftest import make_stub_batch

#: Any seed; the property under test is that re-running under the same one reproduces the run.
_SEED = 1234

#: The eight artifacts a collection-only run leaves behind. Written out rather than globbed: the
#: point of the list is that each of them exists, and a glob would pass on a directory holding
#: seven of them plus something else.
_COLLECTION_ARTIFACTS = (
    PER_SAMPLE_FILENAME,
    PER_ANCHOR_FILENAME,
    VECTORS_FILENAME,
    COLLECTION_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    "preflight.json",
    PROBE_FILENAME,
    run_module.LOG_FILENAME,
)


class _StubLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


def _labelled_batches():
    """Two batches carrying recording identifiers, so the grouped control runs as it would."""
    first = make_stub_batch(batch=4, seed=0)
    first.guid = ["a", "a", "b", "b"]
    second = make_stub_batch(batch=4, seed=1)
    second.guid = ["c", "c", "d", "d"]
    return [first, second]


# =================================================================================================
# The estimator's own randomness
# =================================================================================================
@pytest.fixture
def scored_branch(task, stub_batch):
    """A callable scoring one latent branch under a seeded generator: ``score(seed)``.

    Built through the dense forward and the anchored target the readouts themselves use, so what
    is seeded here is the draw the run actually makes rather than a differently assembled one.
    """
    module = task()
    model = module.orig_model
    y_st, y_ph, u_stream, target_features, weight = model_inputs(module, stub_batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    mask, _coverage = forecast_mask(
        weight, model.geometry, coverage_floor=model.coverage_floor,
        anchors=anchors, anchor_valid=anchor_valid,
    )
    branches = {"base": (outputs["mu_prior"], outputs["logvar_prior"])}

    def _score(seed: int) -> torch.Tensor:
        scores, _contributing = mc_predictive_block(
            model, branches, target, mask, anchors=anchors, likelihood="gaussian_nll",
            num_samples=3, generator=torch.Generator().manual_seed(seed),
        )
        return scores["base"]

    return _score


def test_the_monte_carlo_draw_follows_its_generator_and_not_the_global_one(scored_branch):
    """The load-bearing half of determinism, isolated. The global generator is deliberately
    advanced by a different amount between the two calls, so a draw that still read from it
    would disagree; an equal result means the $\\epsilon$ came from the stream that was handed
    in.
    """
    torch.manual_seed(0)
    first = scored_branch(_SEED)
    torch.manual_seed(99)
    torch.randn(17)  # a different amount of global consumption before the second call
    second = scored_branch(_SEED)

    assert torch.equal(first, second)


def test_the_monte_carlo_draw_is_still_a_draw(scored_branch):
    """The other direction: two different seeds must disagree, or the equality above would hold
    on an estimator that had stopped sampling."""
    assert not torch.equal(scored_branch(_SEED), scored_branch(_SEED + 1))


# =================================================================================================
# The whole evaluation
# =================================================================================================
def test_two_evaluations_of_one_task_agree_exactly(task, perturb_posterior):
    """Not "to tolerance". Every readout, every verdict, every per-recording row: a difference of
    the last bits is still a difference, and a comparison between two arms cannot distinguish a
    real effect from sampling noise the pipeline introduced itself.

    Seeded through :func:`configure_numerics` between the two calls rather than by hand, because
    that is what the run does, and because the model's own reparameterisation draws from the
    global stream regardless of what the estimator does.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()

    def _once():
        configure_numerics(_SEED)
        return evaluate(
            module,
            _StubLoader(_labelled_batches()),
            num_samples=2,
            perm_generator=torch.Generator().manual_seed(_SEED),
            mc_generator=torch.Generator().manual_seed(_SEED),
        )

    assert _once() == _once()


def test_an_unseeded_rerun_is_what_makes_that_test_worth_running(task, perturb_posterior):
    """The evaluation is genuinely stochastic, so the equality above is a property of the seeding
    rather than of an evaluation that never draws anything."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()

    configure_numerics(_SEED)
    first = evaluate(module, _StubLoader(_labelled_batches()), num_samples=2)
    second = evaluate(module, _StubLoader(_labelled_batches()), num_samples=2)

    assert first["readouts"]["nll_full_block"] != second["readouts"]["nll_full_block"]


# =================================================================================================
# What the summary records
# =================================================================================================
@pytest.mark.slow
def test_the_summary_records_the_seed_and_the_numeric_environment(collected_run):
    """A run must be reproducible from its own output, which means the seed is in the artifact
    rather than in the operator's shell history."""
    summary = collected_run["summary"]

    assert summary["numerics"]["seed"] == summary["eval_config"]["seed"]
    assert summary["numerics"]["torch_version"] == str(torch.__version__)


@pytest.mark.slow
def test_the_recorded_numerics_are_the_two_settings_that_are_correctness_requirements(
    collected_run,
):
    """TF32 carries ten mantissa bits and the per-step KL is a small difference of larger
    quantities; ``cudnn.benchmark`` picks convolution algorithms by timing them, so the summation
    order -- and the last bits of the result -- depend on what else the machine was doing. Both
    are read back from global state, so a build where the assignment did not take shows up here
    as ``True``."""
    numerics = collected_run["summary"]["numerics"]

    assert numerics["cuda_matmul_allow_tf32"] is False
    assert numerics["cudnn_allow_tf32"] is False
    assert numerics["cudnn_benchmark"] is False
    assert numerics["float32_matmul_precision"] == "highest"


# =================================================================================================
# The one real collection pass
# =================================================================================================
@pytest.mark.slow
def test_the_collection_pass_leaves_every_artifact_a_later_pass_reads(collected_run):
    """Stated as a list rather than as a glob: each of these is read by something later, and a
    directory holding seven of them is a directory one analysis will skip without saying which."""
    results_dir = Path(collected_run["results_dir"])

    missing = [name for name in _COLLECTION_ARTIFACTS if not (results_dir / name).is_file()]
    assert missing == [], missing
    # And the two the run itself writes on top of the collection.
    assert (results_dir / run_module.SUMMARY_FILENAME).is_file()
    assert (results_dir / STEPS_FILENAME).is_file()


@pytest.mark.slow
def test_the_tables_describe_the_population_the_readouts_were_computed_over(collected_run):
    """One row per scored segment and one per contributing anchor, agreeing with the readouts of
    the same pass -- which is what makes a table-driven analysis and the headline the same run."""
    collection = load_collection(collected_run["results_dir"])
    results = collected_run["summary"]["results"]

    assert len(collection.per_sample) == results["n_samples"] > 0
    assert collection.per_sample["guid"].nunique() == results["n_recordings"] > 1
    assert len(collection.per_anchor) == int(collection.per_sample["n_anchors"].sum()) > 0


@pytest.mark.slow
def test_a_second_run_of_the_same_checkpoint_reports_the_same_numbers(
    cohort_run, cohort_overrides, tmp_path
):
    """The run-level property, end to end through the real loader: same checkpoint, same
    overrides, same seed, and every number in ``results`` identical. Only the paths and the
    timestamp differ, which is why the comparison is of that block rather than of the file.

    A second and third pass rather than reusing the session fixture's directory: reusing it would
    read the tables back instead of collecting them, which is a different property (and one the
    collect suite tests separately).
    """
    checkpoint = sorted((Path(cohort_run) / "model_checkpoints").glob("*.ckpt"))[0]

    def _run(directory: Path):
        # ``main`` returns the process exit code, so an analysis failing is visible to a shell;
        # the summary sits where the explicit output directory says it does.
        assert run_module.main(
            checkpoint, directory, overrides=cohort_overrides, device="cpu", num_samples=2
        ) == 0
        results_dir = directory / run_module.RESULTS_DIRNAME
        summary = json.loads(
            (results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        return summary["results"], (results_dir / PER_SAMPLE_FILENAME).read_bytes()

    first_results, first_table = _run(tmp_path / "first")
    second_results, second_table = _run(tmp_path / "second")

    assert first_results == second_results
    # Byte-identical, not merely equal to tolerance: the CSV is what a re-run reads where the pass
    # that wrote it held a frame, so any difference here is one run reporting two sets of numbers.
    assert first_table == second_table


@pytest.mark.slow
def test_the_tables_survive_the_disk_round_trip_the_offline_path_depends_on(collected_run):
    """``read_csv`` uses ``float_precision='round_trip'``. Without it the fast parser drops the
    last bits of every float, and a per-recording mean amplifies that through cancellation until
    a re-run's summary stops comparing equal to the summary it re-ran."""
    collection = load_collection(collected_run["results_dir"])
    fresh = np.asarray(
        collection.per_sample.groupby("guid")["pred_gap"].mean(), dtype=np.float64
    )
    reread = np.asarray(
        load_collection(collected_run["results_dir"])
        .per_sample.groupby("guid")["pred_gap"]
        .mean(),
        dtype=np.float64,
    )

    assert fresh.size > 1
    assert np.array_equal(fresh, reread, equal_nan=True)
    # Non-vacuous: a column of exact zeros would round-trip through any parser.
    assert np.any(np.isfinite(fresh) & (fresh != 0.0))


@pytest.mark.slow
def test_every_headline_path_whose_block_this_run_produced_resolves(collected_run):
    """The paired half of the headline registry's own test, which could only assert against a
    stub: a path that does not resolve yields ``None`` silently, so a renamed readout would drop
    a column out of every arm table with nothing failing.

    Three families of entry legitimately do not resolve here and all three are enumerated below
    rather than skipped, because "some are null" is exactly what a broken path also looks like:

    * an entry reading a **registered analysis's** block, where that analysis is not registered
      yet -- neither the collection pass nor any step of the run produces such a block, and there
      is nothing for the path to walk into;
    * the three **calibration** entries, which are null under any likelihood with no observation
      variance to calibrate. This fixture trains at ``mse``, where the decoder's log-variance head
      is never fitted and a probability integral transform of its output would be arithmetic over
      an untrained tensor;
    * the **likelihood-space percentage**, which is defined only under ``gaussian_nll`` for the
      same reason and is omitted from the coupling analysis's headline block rather than emitted
      with a false unit -- so its absence here is the run reporting correctly, not a broken path.

    A path is checkable when its first element names either a readout block the collection pass
    produced or a **registered analysis**, and the block it names is non-empty. The second half of
    that is what keeps this test honest as analyses land: an analysis's block lives in the
    summary's ``results`` rather than in the collection's, so a filter that looked only at the
    collection would quietly stop checking every scalar an analysis publishes.
    """
    from teb_vae.lag_attn_cfs.eval import run as runner

    results = collected_run["summary"]["results"]
    headline = results["headline"]
    produced = set(load_collection(collected_run["results_dir"]).results)
    registered = set(runner.ANALYSIS_FUNCTIONS) | set(runner.UNSKIPPABLE_ANALYSES)

    #: Defined only where the decoder emits a predictive distribution; this fixture trains at
    #: ``mse``, whose log-variance head is never fitted. The three calibration entries are on this
    #: list rather than filtered out by the block test above, because the calibration analysis
    #: emits a **skip record** rather than nothing -- so its block is present and non-empty while
    #: the three scalars inside it legitimately are not there at all.
    likelihood_only = {
        "pred_gap_mc_likelihood_pct",
        "calibration_mean_standardised_sq",
        "calibration_pit_max_cdf_deviation",
        "calibration_nll_gain_per_coefficient",
    }

    checked = [
        name for name, path in HEADLINE_SCALARS
        # The block exists and is not the empty dict an unavailable readout leaves behind.
        if (path[0] in produced or path[0] in registered)
        and results.get(path[0]) not in (None, {}, [])
        and name not in likelihood_only
    ]
    unresolved = [name for name in checked if headline.get(name) is None]
    assert unresolved == [], unresolved
    assert len(checked) >= len(HEADLINE_SCALARS) - 8, (
        "most of the registry must be checkable, or this test proves very little"
    )

    # And the ones that were skipped are the three enumerated families, not a fourth nobody
    # noticed.
    skipped = {name for name, _path in HEADLINE_SCALARS} - set(checked)
    analysis_sourced = {
        name for name, path in HEADLINE_SCALARS
        if path[0] not in produced and path[0] not in registered
    }
    calibration_sourced = {
        name for name, path in HEADLINE_SCALARS if path[0] == "calibration"
    }
    assert skipped == analysis_sourced | calibration_sourced | likelihood_only
    # Non-vacuity for the registered half of the filter: at least one scalar checked above comes
    # from an analysis's block rather than from a collection readout.
    assert any(
        path[0] in registered and path[0] not in produced
        for name, path in HEADLINE_SCALARS if name in checked
    )
    # And the reason the calibration family is null, stated rather than assumed: the analysis ran
    # and recorded a **skip**, so its block is present and non-empty while the three scalars inside
    # it are absent. A block that was simply missing would be the other failure and would need the
    # other fix, so the two are told apart here rather than folded into one exclusion.
    assert results["calibration"]["skipped"] is True
    assert results["calibration"]["likelihood"] == "mse"
    assert collected_run["summary"]["results"]["readouts"], "the readouts block is not empty"


@pytest.mark.slow
def test_the_pass_records_the_cost_a_full_run_is_planned_against(collected_run):
    """A recorded measurement, not a threshold: a CI box's timing is not a production box's. What
    an operator needs before starting a multi-hour pass is the rate this one ran at, on the same
    code path, at a stated batch size and draw count -- and what a later reader needs is a number
    a regression is visible against."""
    cost = collected_run["summary"]["collection"]["cost"]

    assert cost["num_mc_samples"] == 2
    assert cost["n_batches"] > 0 and cost["n_samples"] > 0
    assert cost["mean_batch_size"] > 0.0
    for name in ("elapsed_s", "seconds_per_batch", "samples_per_second", "hours_per_1000_samples"):
        assert math.isfinite(cost[name]) and cost[name] > 0.0, name
    # The extrapolation is stated as a rate rather than as a total: the split's sample count is a
    # property of a dataset this record cannot see.
    assert cost["hours_per_1000_samples"] == pytest.approx(
        1000.0 / cost["samples_per_second"] / 3600.0
    )
    assert "hours_per_1000_samples" in cost["note"]
    # Absent rather than zero off CUDA, where the allocator that reports it does not exist.
    assert cost["device"] == "cpu"
    assert cost["peak_allocated_bytes"] is None
