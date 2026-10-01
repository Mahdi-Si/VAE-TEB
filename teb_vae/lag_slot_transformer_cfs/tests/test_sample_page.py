r"""The diagnostic page: what it draws, what it refuses to claim, and what it must never break.

Four properties carry most of the weight here, and each guards a failure that a rendered figure
would not announce.

**The block scores drawn on the gap row are the objective's own.** This cell weights the score by
channel and by horizon step; the causal sibling's row applies neither, so inheriting it would draw
curves in different units from the ``nll_base_block``, ``nll_full_block`` and ``pred_gap`` printed
in the same figure's title. Both directions are checked: the reduction reproduces the logged
metrics exactly, and it differs from the unweighted one whenever a weighting is in force.

**The lag rows are drawn at the decoded anchors.** Every latent tensor here carries an anchor axis,
and a page that indexed it as a stored step would draw real numbers at the wrong columns with no
shape error anywhere in it.

**The suppression row is the controls' own per-lag divergence drop**, on a woken pathway and on
the anchor axis the rest of the page uses; the map's exact ends are the controls' own tests.

**A failure never reaches the fit.** The callback swallows exceptions to protect a multi-day run,
so a page that stopped being drawn would otherwise be visible only as one log line per epoch.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest
import torch
import torch.distributed as dist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn_cfs.sample_page import (  # noqa: E402
    CAUSAL_EXTRA_ROWS,
    LAG_TIME_CAVEAT,
    _Stitched,
    _window_block_scores,
)
from teb_vae.lag_attn_rws.nets.raw_masks import (  # noqa: E402
    contributing_anchors,
    forecast_mask,
)
from teb_vae.lag_attn_rws.sample_page import ForecastRowInputs  # noqa: E402
from teb_vae.lag_slot_transformer_cfs import plotting, sample_page  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.nets.controls import suppressed_parameters  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.plotting import (  # noqa: E402
    LagResidualTrfCfsPlotCallback,
)
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (  # noqa: E402
    TINY_BATCH,
    build_tiny_model,
    tiny_streams,
)
from teb_vae.lag_slot_transformer_cfs.tests.test_task import (  # noqa: E402
    StubBatch,
    build_task,
)
from train.test_utils import FakeMLflowLogger, FakeTrainer  # noqa: E402
from utils.style import SAVE_DPI  # noqa: E402

#: Rows the page reserves on the fully populated arm, as arithmetic over the constants that name
#: them rather than as a literal. A row added to the drawing and not to a constant, or the reverse,
#: fails here by count instead of appearing as a blank strip on every page of a run.
_INPUT_ROWS = 2
_PAGE_ROWS = (
    2  # the raw context row and the forecast lane row
    + len(CAUSAL_EXTRA_ROWS)
    + _INPUT_ROWS
    + len(sample_page.PROPOSAL_ROWS)
    + len(sample_page.LAG_ROWS)
)

#: The page's own file stem. Globbed rather than ``*`` because the run-level budget figure shares
#: the directory, and a bare glob would count it as a page.
_PAGE_GLOB = "lag_residual_trf_cfs_epoch*"

#: A weighting in the shape the shipped configuration ships one: a ratio between the two stored
#: target blocks, and a decaying horizon weight. The tiny fixture's own weights are uniform and its
#: horizon weight absent, which makes the weighting **inert** there -- so a score test built on the
#: bare fixture would pass whether or not the weights were applied at all.
_SHIPPED_WEIGHTING = {"target_weight_ph": 0.1, "horizon_weight_halflife_steps": 5.0}


def _forward(task: Any, batch: Any) -> Tuple[Any, Any, Any, Any]:
    """Run one forward with the proposals retained, as the callback runs it.

    Args:
        task: The Lightning task.
        batch: The batch to run on.

    Returns:
        ``(outs, target_features, weight, inputs)``.
    """
    inputs = task._build_forward_inputs(batch)
    target_features, weight = task._build_raw_target(batch)
    with torch.no_grad():
        outs = task.orig_model(*inputs, return_proposals=True)
    return outs, target_features, weight, inputs


def _row_inputs(task: Any, batch: Any, outs: Any, target: Any, index: int) -> ForecastRowInputs:
    """A minimal row-input carrier for the score helper, which draws nothing.

    The layout hooks are stubs: :func:`_weighted_window_scores` reads the batch, the geometry, the
    target stream and the forward dict, and touches no axes at all.

    Args:
        task: The Lightning task, for the model geometry.
        batch: The batch the score's mask is built from.
        outs: The forward dict.
        target: The declared-width target stream.
        index: Which sample the scores are for.

    Returns:
        The carrier.
    """
    return ForecastRowInputs(
        outs=outs,
        target=target,
        batch=batch,
        geometry=task.orig_model.geometry,
        sample_index=index,
        normalization_stats=None,
        up_raw=None,
        time_raw=np.zeros(1),
        t_max=1.0,
        row_axes=lambda name: (None, None),
        finalise_time_axis=lambda ax, **kwargs: None,
        attach_cbar=lambda cax, image, label: None,
        heatmap_spines=lambda ax: None,
        figure=None,
        included_rows=frozenset(),
    )


def _stitched_over(task: Any, outs: Any, index: int) -> _Stitched:
    """A tiling description over every decoded anchor, carrying what the score helpers read.

    Three fields are load-bearing here and the drawing fields are not: the anchor set, which
    positions of it to score, and **which declared channel each decoder lane is**. The last is
    what the sibling's scorer gathers its target with, so a placeholder there would have it score
    a one-channel block against a full one.

    Args:
        task: The Lightning task, for the model's kept channel index.
        outs: The forward dict, for the anchor set.
        index: Which sample the tiling is for.

    Returns:
        A ``_Stitched`` whose drawing fields are empty.
    """
    gate = task.orig_model.target_gate
    keep = (
        np.arange(int(outs["mu_full"].shape[-1]))
        if gate is None
        else gate.keep_index.cpu().numpy().astype(int)
    )
    empty = np.zeros((1, 1))
    return _Stitched(
        truth=empty, base_mean=empty, base_sigma=empty, full_mean=empty, full_sigma=empty,
        keep=keep, block_split=0,
        anchors=outs["anchor_index"][index].cpu().numpy().astype(int),
        positions=list(range(int(outs["anchor_index"].shape[1]))),
    )


def _trainer_with_batch(batch: Any, **kwargs: Any) -> FakeTrainer:
    """A rank-zero trainer whose validation loader yields one batch.

    Args:
        batch: The batch to yield.
        **kwargs: Overrides for the fake trainer.

    Returns:
        The trainer.
    """
    kwargs.setdefault("is_global_zero", True)
    kwargs.setdefault("current_epoch", 0)
    trainer = FakeTrainer(**kwargs)
    trainer.val_dataloaders = [[batch]]  # type: ignore[attr-defined]
    return trainer


def _pages(callback: LagResidualTrfCfsPlotCallback) -> List[Any]:
    """The page files this callback has written, by its own stem.

    Args:
        callback: The callback.

    Returns:
        The paths, sorted.
    """
    return sorted(callback.output_dir.glob(f"{_PAGE_GLOB}.{callback.file_format}"))


# =================================================================================================
# The callback
# =================================================================================================
def test_it_writes_one_page_per_drawn_sample_and_logs_each(tmp_path) -> None:
    """The figure a run is judged by, and the artifact seam that makes it reachable."""
    task, batch = build_task(), StubBatch()
    logger = FakeMLflowLogger()
    callback = LagResidualTrfCfsPlotCallback(
        tmp_path, num_examples=2, file_format="png", mlflow_logger=logger
    )
    callback._generate_plots(_trainer_with_batch(batch), batch, task, epoch=0)

    assert len(_pages(callback)) == 2
    assert all(path.stat().st_size > 0 for path in _pages(callback))
    assert len(logger.experiment.calls) >= 2


def test_a_failure_inside_the_page_never_reaches_the_training_loop(tmp_path, monkeypatch) -> None:
    """The handler exists to protect a multi-day fit, and is why the tests above matter."""

    def explode(**_kwargs: Any) -> Any:
        """Stand in for a page builder that fails at the worst moment.

        Args:
            **_kwargs: The builder's keywords, ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("the page could not be drawn")

    monkeypatch.setattr(plotting, "build_residual_page", explode)
    task, batch = build_task(), StubBatch()
    callback = LagResidualTrfCfsPlotCallback(tmp_path, num_examples=1, file_format="png")
    trainer = _trainer_with_batch(batch)

    # Through on_validation_epoch_end, which is where the handler lives.
    callback.on_validation_epoch_end(trainer, task)
    assert _pages(callback) == []


def test_the_forward_runs_over_the_drawn_samples_alone(tmp_path, monkeypatch) -> None:
    """The proposals are the largest tensor this architecture holds.

    A page drawing one sample of a production batch would otherwise materialise the whole batch's
    per-lag proposal array to discard all but one row of it.
    """
    seen: List[int] = []
    original = plotting.LagResidualTrfCfsPlotCallback._generate_plots

    task, batch = build_task(), StubBatch()
    model = task.orig_model
    forward = model.forward

    def record(*args: Any, **kwargs: Any) -> Any:
        """Note the batch size the forward was called with, then run it.

        Args:
            *args: The forward's positional arguments; the first carries the batch axis.
            **kwargs: The forward's keyword arguments.

        Returns:
            Whatever the real forward returns.
        """
        seen.append(int(args[0].shape[0]))
        return forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record)
    callback = LagResidualTrfCfsPlotCallback(tmp_path, num_examples=1, file_format="png")
    original(callback, _trainer_with_batch(batch), batch, task, 0)

    assert seen == [1]
    assert TINY_BATCH > 1, "the fixture must have more samples than the page draws"


# =================================================================================================
# The layout
# =================================================================================================
def _render(task: Any, batch: Any, **overrides: Any) -> Any:
    """Build one page directly, exactly as the callback builds it.

    Args:
        task: The Lightning task.
        batch: The batch to draw from.
        **overrides: Page keywords to replace.

    Returns:
        The figure; the caller closes it.
    """
    outs, target, _weight, inputs = _forward(task, batch)
    model = task.orig_model
    kwargs = dict(
        outs=outs,
        target_features=target,
        geometry=model.geometry,
        sample_index=0,
        epoch=0,
        guid="guid-0",
        beta=1.0,
        scalars={},
        batch=batch,
        forecast_rows=task.forecast_rows,
        forecast_extra_rows=task.forecast_extra_rows,
        input_streams=task.input_stream_panels(model, inputs, sample_index=0),
        lag_panels=sample_page.residual_lag_panels(model, outs, sample_index=0),
    )
    kwargs.update(overrides)
    return sample_page.build_residual_page(**kwargs)


def test_the_page_reserves_exactly_the_rows_its_constants_name() -> None:
    """A row drawn and not reserved raises inside a handler that swallows it; a row reserved and
    not drawn is a blank strip on every page of the run."""
    figure = _render(build_task(), StubBatch())
    try:
        assert figure.axes, "the page drew no axes at all"
        assert figure.axes[0].get_gridspec().nrows == _PAGE_ROWS
    finally:
        plt.close(figure)


def test_every_row_spans_the_whole_recording_on_one_axis() -> None:
    """A column of the page is one instant on all of its rows, or it is nothing."""
    task = build_task()
    figure = _render(task, StubBatch())
    try:
        geometry = task.orig_model.geometry
        t_max = float(geometry.raw_len) / 4.0
        spans = {
            tuple(round(value, 6) for value in ax.get_xlim())
            for ax in figure.axes
            # The colorbars and the two profile insets carry their own axes and their own units.
            if ax.get_xlabel() and "Time" in ax.get_xlabel() or "Anchor step" in ax.get_xlabel()
        }
        assert spans == {(0.0, round(t_max, 6))}
    finally:
        plt.close(figure)


def test_the_lag_rows_columns_land_on_the_decoded_anchors() -> None:
    """Every latent tensor here is anchor-indexed, and the lag axis is the net's own bank.

    A page that read the anchor axis as a stored step would draw real numbers at the wrong
    columns, and nothing about the array's shape would say so. The lag axis is pinned for the
    converse reason: the rows must span exactly the candidate lags the model was built with,
    $L = \\texttt{max\\_lag} + 1$, read off the forward rather than off any constant, so a
    shortened bank draws a shortened axis.
    """
    task = build_task()
    batch = StubBatch()
    outs, _target, _weight, _inputs = _forward(task, batch)
    geometry = task.orig_model.geometry
    seconds_per_step = (float(geometry.raw_len) / 4.0) / float(geometry.t)
    n_lags = int(task.orig_model.n_lags)
    assert n_lags == int(outs["mean_proposals"].shape[2])

    anchors = outs["anchor_index"][0][outs["anchor_valid"][0]].cpu().numpy()
    figure = _render(task, batch)
    try:
        images = [
            ax.get_images()[0]
            for ax in figure.axes
            if ax.get_ylabel().startswith("Lag ") and ax.get_images()
        ]
        assert len(images) == len(sample_page.LAG_ROWS) - 1, "one image per lag heatmap"
        for image in images:
            left, right, bottom, top = image.get_extent()
            # Sample-centred: the first anchor sits half a cell in from the left edge.
            spacing = float(np.median(np.diff(anchors))) * seconds_per_step
            assert left == pytest.approx(anchors[0] * seconds_per_step - spacing / 2.0)
            assert right == pytest.approx(
                anchors[0] * seconds_per_step + (len(anchors) - 0.5) * spacing
            )
            # One cell per candidate lag, lag 0 at the bottom: the y-extent is the bank length.
            assert (bottom, top) == (-0.5, n_lags - 0.5)
            assert image.get_array().shape[0] == n_lags
    finally:
        plt.close(figure)


def test_the_page_draws_both_input_rows() -> None:
    """The source stream has no per-step availability adapter, and the family's builder reads one.

    Declared as ``None`` rather than left missing, so the builder skips the multiply instead of
    raising -- which would cost the page both of its input rows and one log line.
    """
    task = build_task()
    model = task.orig_model
    _outs, _target, _weight, inputs = _forward(task, StubBatch())
    panels = task.input_stream_panels(model, inputs, sample_index=0)
    assert [panel.name for panel in panels] == ["target", "source"]


def test_the_lag_rows_are_absent_where_no_per_lag_update_exists() -> None:
    """The comparator's fusion normalises over lags and has nothing to remove.

    Four empty rows would read as four measured zeros, which is the one reading that cannot be
    told from an absent effect afterwards.
    """
    task = build_task(lag_fusion="attention")
    batch = StubBatch()
    outs, _target, _weight, _inputs = _forward(task, batch)
    assert sample_page.residual_lag_panels(task.orig_model, outs, sample_index=0) is None

    figure = _render(task, batch)
    try:
        assert figure.axes[0].get_gridspec().nrows == _PAGE_ROWS - len(sample_page.LAG_ROWS)
    finally:
        plt.close(figure)


# =================================================================================================
# The window score
# =================================================================================================
def _page_scores(
    task: Any, batch: Any, outs: Any, target: Any, weight: Any
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Every anchor's drawn block score, for every sample, with the contributing count.

    Takes the forward dict rather than running one: both branches are decoded from a **sampled**
    latent, so a second forward would score a different draw than the metrics it is compared with
    and the two would disagree by the noise alone.

    Args:
        task: The Lightning task.
        batch: The batch, for the objective's validity signal.
        outs: The forward dict the metrics were computed from.
        target: The declared-width target stream.
        weight: The batch's validity signal.

    Returns:
        ``(base, full, contributing)`` -- the two score stacks and the objective's own anchor
        count, which is the denominator the logged block metrics use.
    """
    model = task.orig_model

    base, full = [], []
    for index in range(int(outs["anchor_index"].shape[0])):
        scores = sample_page._weighted_window_scores(
            _row_inputs(task, batch, outs, target, index),
            _stitched_over(task, outs, index),
            likelihood="gaussian_nll",
            coverage_floor=float(model.coverage_floor),
            forecast_target=model._build_forecast_target,
            scored_weight=model.scored_weight,
            channel_weight=getattr(model, "target_channel_weight", None),
            horizon_weight=getattr(model, "horizon_weight", None),
        )
        assert scores is not None
        base.append(scores["base"])
        full.append(scores["full"])

    with torch.no_grad():
        mask, _coverage = forecast_mask(
            model.scored_weight(weight), model.geometry,
            coverage_floor=float(model.coverage_floor),
            anchors=outs["anchor_index"], anchor_valid=outs["anchor_valid"],
        )
        contributing = float(contributing_anchors(mask).to(torch.float64).sum())
    return np.asarray(base), np.asarray(full), contributing


def test_the_drawn_block_scores_are_the_ones_the_run_logs() -> None:
    """The row and the title must be the same number, or the page argues with its own run.

    Built under the shipped weighting rather than the fixture's uniform one, so the equality is a
    statement about a weighted reduction rather than about a case where the weights do nothing.
    """
    task, batch = build_task(**_SHIPPED_WEIGHTING), StubBatch()
    outs, target, weight, _inputs = _forward(task, batch)
    with torch.no_grad():
        metrics = task.orig_model.compute_loss(
            outs, target, weight=weight, beta=1.0, beta_prior=0.1,
            lambda_full=1.0, lambda_base=1.0, likelihood="gaussian_nll", free_bits=0.0,
        )["metrics"]

    base, full, contributing = _page_scores(task, batch, outs, target, weight)
    assert float(base.sum()) / contributing == pytest.approx(
        float(metrics["nll_base_block"]), rel=1e-5
    )
    assert float(full.sum()) / contributing == pytest.approx(
        float(metrics["nll_full_block"]), rel=1e-5
    )
    assert float((base - full).sum()) / contributing == pytest.approx(
        float(metrics["pred_gap"]), rel=1e-4
    )


def test_the_unweighted_reduction_is_the_siblings_own() -> None:
    """With no weighting configured the two must agree exactly.

    This is what makes the test above a statement about the weighting rather than about a second
    implementation of the reduction: strip the weights and this row is the causal page's row.
    """
    task, batch = build_task(), StubBatch()
    outs, target, _weight, _inputs = _forward(task, batch)
    model = task.orig_model
    stitched = _stitched_over(task, outs, 0)
    rows = _row_inputs(task, batch, outs, target, 0)

    mine = sample_page._weighted_window_scores(
        rows, stitched, likelihood="gaussian_nll",
        coverage_floor=float(model.coverage_floor),
        forecast_target=model._build_forecast_target,
        scored_weight=model.scored_weight,
        channel_weight=None,
        horizon_weight=None,
    )
    theirs = _window_block_scores(
        rows, stitched, likelihood="gaussian_nll",
        coverage_floor=float(model.coverage_floor),
    )
    assert mine is not None and theirs is not None
    for branch in ("base", "full"):
        assert mine[branch] == pytest.approx(theirs[branch], rel=1e-5)


def test_the_weighted_score_differs_from_the_row_this_cell_would_otherwise_inherit() -> None:
    """The regression the local builder exists to prevent.

    The shipped configuration weights the block by channel, so the inherited row would draw curves
    whose units are not the ones printed above them.
    """
    task, batch = build_task(**_SHIPPED_WEIGHTING), StubBatch()
    outs, target, _weight, _inputs = _forward(task, batch)
    model = task.orig_model
    assert getattr(model, "target_channel_weight", None) is not None
    assert getattr(model, "horizon_weight", None) is not None
    stitched = _stitched_over(task, outs, 0)
    rows = _row_inputs(task, batch, outs, target, 0)

    weighted = sample_page._weighted_window_scores(
        rows, stitched, likelihood="gaussian_nll",
        coverage_floor=float(model.coverage_floor),
        forecast_target=model._build_forecast_target,
        scored_weight=model.scored_weight,
        channel_weight=model.target_channel_weight,
        horizon_weight=getattr(model, "horizon_weight", None),
    )
    unweighted = _window_block_scores(
        rows, stitched, likelihood="gaussian_nll",
        coverage_floor=float(model.coverage_floor),
    )
    assert weighted is not None and unweighted is not None
    assert not np.allclose(weighted["full"], unweighted["full"])


# =================================================================================================
# The suppression map
# =================================================================================================
def _woken_forward() -> Tuple[Any, Any]:
    """The tiny model with its proposal head moved off zero, and one dense forward of it.

    At the zero start every proposal, update and divergence is exactly zero, so every per-lag row
    would compare zeros with zeros.

    Returns:
        ``(model, outs)``, the forward carrying its per-lag proposals.
    """
    model = build_tiny_model().eval()
    torch.manual_seed(20260909)
    torch.nn.init.normal_(model.proposal_head.output_proj.weight, std=0.3)
    torch.nn.init.normal_(model.proposal_head.output_proj.bias, std=0.3)
    y_st, y_ph, source = tiny_streams()
    with torch.no_grad():
        outs = model(y_st, y_ph, source, 0, 1, return_proposals=True)
    assert float(outs["kld_per_anchor"].abs().max()) > 0.0, "the pathway did not wake"
    return model, outs


def test_the_suppression_row_is_the_divergence_each_lag_alone_accounts_for() -> None:
    """One lag at a time, against the matched arm, on the anchor axis the rest of the page uses."""
    model, outs = _woken_forward()
    panels = sample_page.residual_lag_panels(model, outs, sample_index=0)
    assert panels is not None

    n_lags = int(outs["mean_proposals"].shape[2])
    assert panels.suppression.shape == (int(outs["anchor_index"].shape[1]), n_lags)
    for lag in range(n_lags):
        removed = torch.zeros(n_lags, dtype=torch.bool)
        removed[lag] = True
        with torch.no_grad():
            expected = (
                outs["kld_per_anchor"][0]
                - suppressed_parameters(model, outs, removed)["kld_per_anchor"][0]
            )
        assert panels.suppression[:, lag] == pytest.approx(
            expected.cpu().numpy(), rel=1e-5, abs=1e-7
        )


def test_the_cancellation_row_carries_both_parts_of_the_ratio() -> None:
    """A near-zero ratio means proposals that cancel or proposals that are near zero.

    The denominator is what separates them, and a row drawing the ratio alone could not.
    """
    model, outs = _woken_forward()
    panels = sample_page.residual_lag_panels(model, outs, sample_index=0)
    assert panels is not None
    assert panels.cancellation_numerator == pytest.approx(
        outs["cancellation_numerator_mean"][0].cpu().numpy(), rel=1e-6
    )
    assert panels.cancellation_denominator == pytest.approx(
        outs["cancellation_denominator_mean"][0].cpu().numpy(), rel=1e-6
    )


# =================================================================================================
# What the page costs the fit
# =================================================================================================
def test_the_page_carries_one_note_line_inside_the_page() -> None:
    """One short note under the lag rows, and the suptitle names the epoch, the recording and
    its subgroup only. The rendered width is asserted rather than assumed: a note that runs off
    the page raises nothing."""
    figure = _render(build_task(), StubBatch(), cohort="healthy_bg_no_cs")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        # Every figure-level text except the suptitle, which is placed by its own rule.
        notes = [text for text in figure.texts if text is not figure._suptitle]
        assert [text.get_text() for text in notes] == [LAG_TIME_CAVEAT]
        assert figure._suptitle.get_text() == "epoch 0, guid guid-0, subgroup healthy_bg_no_cs"
        width = figure.get_size_inches()[0] * figure.dpi
        for text in [*notes, figure._suptitle]:
            assert text.get_window_extent(renderer).width <= width
    finally:
        plt.close(figure)


def test_the_page_is_saved_directly_below_the_publication_resolution(tmp_path, monkeypatch) -> None:
    """The single most expensive thing this callback could do, and it is not worth anything.

    Every heatmap here is a small array -- one column per decoded anchor -- and matplotlib
    resamples each to the axes' size in device pixels, so the publication resolution multiplies
    the draw cost for a file of about the same size. A tight bounding box costs a second full
    draw on top, and this page sets its margins explicitly. MEASURED at the production geometry: a
    plotted epoch cost $49.6$ s through the shared helper and $12.7$ s through this one. On rank
    zero inside a distributed fit, that is time every other rank spends waiting at the next
    collective.
    """
    callback = LagResidualTrfCfsPlotCallback(tmp_path, num_examples=1, file_format="png")
    figure = plt.figure()
    seen: dict = {}
    monkeypatch.setattr(figure, "savefig", lambda *_args, **kwargs: seen.update(kwargs))

    callback._save(figure, tmp_path / "page.png", None)

    assert seen["dpi"] < SAVE_DPI
    assert "bbox_inches" not in seen
    assert figure.number not in plt.get_fignums(), "the figure must be closed after the save"


def test_a_save_that_raises_still_closes_the_figure(tmp_path, monkeypatch) -> None:
    """pyplot holds every unclosed figure in a global registry, and the caller has let go of it.

    A page-per-epoch leak would grow the process for the length of the run.
    """
    callback = LagResidualTrfCfsPlotCallback(tmp_path, num_examples=1, file_format="png")
    figure = plt.figure()

    def explode(*_args: Any, **_kwargs: Any) -> None:
        """Fail the way a full disk would.

        Raises:
            OSError: Always.
        """
        raise OSError("no space left on device")

    monkeypatch.setattr(figure, "savefig", explode)
    before = len(plt.get_fignums())
    with pytest.raises(OSError):
        callback._save(figure, tmp_path / "page.png", None)
    assert len(plt.get_fignums()) == before - 1


# =================================================================================================
# What the page must never do to a distributed fit
# =================================================================================================
def _pretend_distributed(monkeypatch, *, world: int = 7) -> List[str]:
    """Make this one process look like rank zero of a live process group.

    Every collective is replaced by one that refuses. That is the whole instrument: a collective
    entered from here would, in a real run, block until the watchdog killed the job, and a test
    that waited for that would hang the suite instead.

    Args:
        monkeypatch: pytest's patcher.
        world: The process count to report.

    Returns:
        A list that records the name of any collective that was reached.
    """
    reached: List[str] = []

    def refuse(name: str):
        """Build a stand-in for one collective."""

        def call(*_args: Any, **_kwargs: Any) -> None:
            """Record and refuse.

            Raises:
                AssertionError: Always.
            """
            reached.append(name)
            raise AssertionError(f"the diagnostic page reached torch.distributed.{name}")

        return call

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: world)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 0)
    for name in ("all_reduce", "barrier", "broadcast", "all_gather", "reduce"):
        if hasattr(dist, name):
            monkeypatch.setattr(dist, name, refuse(name))
    return reached


def test_the_instrument_catches_a_collective() -> None:
    """The guard below is only worth anything if this model really does synchronise.

    Written first and separately, because a test that patched nothing reachable would pass for
    the wrong reason for as long as it existed.
    """
    with pytest.MonkeyPatch.context() as patch:
        _pretend_distributed(patch)
        task, batch = build_task(), StubBatch()
        outs, target, weight, _inputs = _forward(task, batch)
        with pytest.raises(AssertionError, match="all_reduce"):
            task.orig_model.compute_loss(
                outs, target, weight=weight, beta=1.0, beta_prior=0.1,
                lambda_full=1.0, lambda_base=1.0, likelihood="gaussian_nll", free_bits=0.0,
            )


def test_the_page_reaches_no_collective_on_rank_zero(tmp_path, monkeypatch) -> None:
    """The defect that stopped a seven-rank fit dead at its first plotted epoch.

    The inherited handler returns early on every rank but zero, so everything this callback does
    is done by one process. This architecture's objective ends in an ``all_reduce``; the shared
    callback calls it to put the loss on the page, and here that call blocked rank zero on six
    peers that had already moved on. Nothing was written, nothing raised, and the run stopped.

    The page's title takes what the epoch logged instead, so this path must now be collective-free
    end to end -- forward, lag panels, window scores and all.
    """
    reached = _pretend_distributed(monkeypatch)
    task, batch = build_task(), StubBatch()
    trainer = _trainer_with_batch(batch)
    trainer.callback_metrics = {  # type: ignore[attr-defined]
        "val/nll_full_block": torch.tensor(537.9),
        "val/nll_base_block": torch.tensor(538.4),
        "val/pred_gap": torch.tensor(0.427),
    }
    callback = LagResidualTrfCfsPlotCallback(tmp_path, num_examples=1, file_format="png")

    callback._generate_plots(trainer, batch, task, epoch=0)

    assert reached == [], f"the page synchronised through {reached}"
    assert len(_pages(callback)) == 1


def test_the_title_carries_the_epochs_own_readouts() -> None:
    """Taken from the run rather than recomputed, so the figure and the curve cannot disagree:
    only the finite validation metrics, with the stage prefix stripped."""
    trainer = _trainer_with_batch(StubBatch())
    trainer.callback_metrics = {  # type: ignore[attr-defined]
        "val/pred_gap": torch.tensor(0.25),
        "train/pred_gap": torch.tensor(9.0),
        "val/not_a_number": "text",
        "lr": torch.tensor(1e-3),
    }
    readouts = plotting._epoch_readouts(trainer)
    assert readouts == {"pred_gap": 0.25}, "only finite validation metrics, unprefixed"
    assert plotting._epoch_readouts(FakeTrainer(is_global_zero=True, current_epoch=0)) == {}
