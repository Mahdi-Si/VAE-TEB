r"""The diagnostic page is reached through **two** levels of inheritance, and writes nothing here.

Five of the page's seven rows are the shared raw-signal builder's. Two are replaced, and the
replacements live in the conv-LSTM causal cell rather than here, for the reason every other member
of this composition lives there: they name no encoder. The forecast rows are re-pointed because the
anchor axis is sparse; the input rows and the run-level budget figure because the shipped builders
consult the production two-sided Morlet bank, which did not produce these coefficients and refuses
these channel widths.

**All three of those failures are silent.** ``describe_streams`` raises inside a handler that warns
and continues, and the forecast-row builder would index an anchor into a dense
$(T_{\mathrm{valid}}, H, C)$ block and draw a real forecast at the wrong time with no shape error in
it. So this package's contribution to the page is a *resolution*, and what is tested is that the
resolution lands on the causal cell's builders by object identity -- not that a figure appeared.

The other half is an absence: this package must write no ``plotting.py`` and no ``sample_page.py``.
A module here would be a second copy of two builders that must not drift from the ones the
comparison model's page is drawn with, and it would drift silently, because a page is never worth
failing a multi-day fit for.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from teb_vae.lag_attn_cfs import sample_page as causal_page
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask

from .conftest import TINY_STRIDE

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


# =================================================================================================
# What this package does not write
# =================================================================================================
@pytest.mark.parametrize("name", ["plotting.py", "sample_page.py", "warmup_budget.py"])
def test_this_package_writes_no_figure_module(name):
    """A directory check, because the failure it guards is a module that exists and is never
    reached: the callback resolves its builders off the *task*, so a local copy would sit unused
    until someone imported it by hand and then diverge from the drawn page."""
    assert not (_PACKAGE_DIR / name).exists(), name


@pytest.mark.parametrize(
    "module",
    [
        "teb_vae.lag_attn_transformer_cfs.plotting",
        "teb_vae.lag_attn_transformer_cfs.sample_page",
        "teb_vae.lag_attn_transformer_cfs.warmup_budget",
    ],
)
def test_the_figure_modules_are_not_importable_from_this_package(module):
    """The other half of the directory check, and the one that would catch a module smuggled in
    through a namespace package or an ``__init__`` re-export."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


# =================================================================================================
# Where the three seams resolve
# =================================================================================================
def test_the_forecast_rows_resolve_to_the_causal_cells_builder(task):
    """By object identity on the *underlying function*, not merely by shape: the rows are bound with
    five values the page cannot recover from the arrays it is handed -- which declared channel each
    decoder output is, where the two stored blocks meet on that axis, the stride a training step
    tiles at, and the likelihood and coverage floor the per-window score row scores under."""
    module = task()
    rows = module.forecast_rows

    assert rows.func is causal_page.causal_forecast_rows
    assert set(rows.keywords) == {
        "keep_index", "block_split", "training_stride", "likelihood", "coverage_floor",
    }
    assert rows.keywords["block_split"] == 36
    assert rows.keywords["training_stride"] == module.orig_model.anchor_stride == TINY_STRIDE
    # The score row's two: the objective's own likelihood, and the net's own floor. Bound from
    # where the objective takes them, so a window's height is this run's block score.
    assert rows.keywords["likelihood"] == module.hparams["likelihood"]
    assert rows.keywords["coverage_floor"] == module.orig_model.coverage_floor


def test_the_extra_page_rows_resolve_to_the_causal_cells_constant(task):
    """The six rows the seam draws below the two the layout always reserves. Returned from the
    drawing module's own constant so the names reserved and the names drawn are one object: a name
    reserved and not drawn is a blank row, and a name drawn and not reserved is a ``KeyError``
    raised inside a handler that swallows it -- a page silently missing from the whole run."""
    module = task()

    assert module.forecast_extra_rows is causal_page.CAUSAL_EXTRA_ROWS
    assert [name for name, _height in module.forecast_extra_rows] == [
        "pred_truth", "pred_base", "pred_full", "pred_skill", "pred_sigma", "pred_gap",
    ]


def test_the_input_panel_builder_resolves_to_the_causal_cells(task):
    """The seam whose absence costs two page rows, one log line and nothing else. The replacement
    reads the gates, the adapters' own availability buffers and the warm-up vectors off the net it
    is handed, which is what keeps the drawn stream the encoder's input by construction."""
    module = task()

    assert module.input_stream_panels is causal_page.causal_stream_panels


def test_the_run_level_budget_figure_resolves_to_the_causal_cells(task):
    """A method rather than a property, deliberately: the callback resolves this seam with
    ``getattr(pl_module, ..., None)``, which does not swallow an exception raised *inside* a
    property -- so a task with no budget would take down the whole page rather than cost the one
    figure it can no longer draw."""
    assert (
        SeqVaeLagAttnTrfCfsTask.input_budget_figure
        is SeqVaeLagAttnCfsTask.input_budget_figure
    )
    assert not isinstance(
        vars(SeqVaeLagAttnCfsTask)["input_budget_figure"], property
    )

    module = task()
    with pytest.raises(ValueError, match="no resolved warm-up budget"):
        module.input_budget_figure(_PACKAGE_DIR)


def test_the_page_seams_are_the_same_objects_both_causal_cells_draw_with(task):
    """The encoder edge on the diagnostic page: two runs read side by side must be drawn by one
    builder, or a difference in a figure is a difference between two plotting routines."""
    for name in ("forecast_rows", "input_stream_panels", "forecast_extra_rows"):
        assert (
            vars(SeqVaeLagAttnCfsTask)[name] is not None
        ), f"{name} moved off the causal parent"
        assert name not in vars(SeqVaeLagAttnTrfCfsTask)
