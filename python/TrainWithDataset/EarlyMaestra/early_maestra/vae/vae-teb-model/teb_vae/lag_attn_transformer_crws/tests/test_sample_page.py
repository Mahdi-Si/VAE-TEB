r"""The diagnostic page is reached through **two** levels of inheritance, and writes nothing here.

Five of the page's seven rows are the shared raw-signal builder's. Two are replaced, and the
replacements live in the conv-LSTM cell of this row rather than here, for the reason every other
member of this composition lives there: they name no encoder. The forecast rows are re-pointed
because the anchor axis is sparse -- the shipped rows walk a dense $(T_{\mathrm{valid}}, H, R)$ block
and index an *anchor* into it, while this model's forecast is $(A_{\max}, H, R)$ indexed by position
in the decoded set. The input rows and the run-level budget figure are re-pointed because the shipped
builders consult the production two-sided Morlet bank, which did not produce these coefficients and
refuses these channel widths.

**All three of those failures are silent.** ``describe_streams`` raises inside a handler that warns
and continues, and the forecast-row builder would draw a real forecast at the wrong time with no
shape error in it. So this package's contribution to the page is a *resolution*, and what is tested
is that the resolution lands on the conv-LSTM cell's builders by object identity -- not that a figure
appeared. That figure is drawn and asserted where the builders live.

The other half is an absence: this package must write no ``plotting.py`` and no ``sample_page.py``.
A module here would be a second copy of two builders that must not drift from the ones the comparison
model's page is drawn with, and it would drift silently, because a page is never worth failing a
multi-day fit for.
"""
from __future__ import annotations

import importlib
from functools import partial
from pathlib import Path

import pytest

from teb_vae.lag_attn_cfs import sample_page as causal_feature_page
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_crws import sample_page as causal_raw_page
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask

from .conftest import TINY_STRIDE

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


# =================================================================================================
# What this package does not write
# =================================================================================================
@pytest.mark.parametrize("name", ["plotting.py", "sample_page.py", "warmup_budget.py"])
def test_this_package_writes_no_figure_module(name) -> None:
    """A directory check, because the failure it guards is a module that exists and is never
    reached: the callback resolves its builders off the *task*, so a local copy would sit unused
    until someone imported it by hand and then diverge from the drawn page."""
    assert not (_PACKAGE_DIR / name).exists(), name


@pytest.mark.parametrize(
    "module",
    [
        "teb_vae.lag_attn_transformer_crws.plotting",
        "teb_vae.lag_attn_transformer_crws.sample_page",
        "teb_vae.lag_attn_transformer_crws.warmup_budget",
    ],
)
def test_the_figure_modules_are_not_importable_from_this_package(module) -> None:
    """The other half of the directory check, and the one that would catch a module smuggled in
    through a namespace package or an ``__init__`` re-export."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


# =================================================================================================
# Where the three seams resolve
# =================================================================================================
def test_the_forecast_rows_resolve_to_the_conv_lstm_cells_builder(task) -> None:
    """By object identity on the *underlying function*, not merely by shape: the rows are bound with
    the one value the page cannot recover from the arrays it is handed -- the stride a training step
    tiles at, which the overlay draws beside the dense set this page is itself produced at."""
    module = task()
    rows = module.forecast_rows

    assert isinstance(rows, partial)
    assert rows.func is causal_raw_page.causal_raw_forecast_rows
    assert set(rows.keywords) == {"training_stride"}
    assert rows.keywords["training_stride"] == module.orig_model.anchor_stride == TINY_STRIDE


def test_the_forecast_rows_property_itself_is_the_conv_lstm_cells(task) -> None:
    """One level up from the bound value: the *property* is the causal parent's own descriptor, so
    the two cells of this row cannot come to disagree about what a forecast row draws."""
    assert "forecast_rows" not in vars(SeqVaeLagAttnTrfCrwsTask)
    assert (
        SeqVaeLagAttnTrfCrwsTask.forecast_rows is SeqVaeLagAttnCrwsTask.__dict__["forecast_rows"]
    )


def test_this_cell_reserves_no_extra_forecast_rows(task) -> None:
    r"""The absence the causal-*feature* cells do not share, and it is a design decision rather than
    an omission: that page reserves six stitched field rows because three of $98$ target channels
    drawn as lanes is not a picture of that forecast. This decoder emits $R = 16$ raw samples of
    **one** signal, so the shipped two-row layout already is the picture.

    A reserved name that is never drawn is a blank row; a drawn name that is not reserved is a
    ``KeyError`` inside a handler that swallows it. Neither can happen while there are none.
    """
    module = task()

    assert getattr(module, "forecast_extra_rows", None) in (None, (), [])
    assert not hasattr(SeqVaeLagAttnCrwsTask, "forecast_extra_rows")


def test_the_input_panel_builder_resolves_to_the_causal_feature_cells(task) -> None:
    """The seam whose absence costs two page rows, one log line and nothing else. The replacement
    reads the gates, the adapters' own availability buffers and the warm-up vectors off the net it
    is handed, which is what keeps the drawn stream the encoder's input by construction.

    Two hops rather than one, and both are asserted: this task binds the conv-LSTM cell's descriptor,
    which is itself the causal-feature cell's -- so all four cells that read these three input tensors
    draw them with one builder.
    """
    module = task()

    assert "input_stream_panels" not in vars(SeqVaeLagAttnTrfCrwsTask)
    assert (
        SeqVaeLagAttnCrwsTask.__dict__["input_stream_panels"]
        is SeqVaeLagAttnCfsTask.__dict__["input_stream_panels"]
    )
    assert module.input_stream_panels is causal_feature_page.causal_stream_panels


def test_the_run_level_budget_figure_resolves_to_the_causal_feature_cells(task) -> None:
    """A method rather than a property, deliberately: the callback resolves this seam with
    ``getattr(pl_module, ..., None)``, which does not swallow an exception raised *inside* a
    property -- so a task with no budget would take down the whole page rather than cost the one
    figure it can no longer draw."""
    assert "input_budget_figure" not in vars(SeqVaeLagAttnTrfCrwsTask)
    assert (
        SeqVaeLagAttnCrwsTask.__dict__["input_budget_figure"]
        is SeqVaeLagAttnCfsTask.__dict__["input_budget_figure"]
    )
    assert not isinstance(vars(SeqVaeLagAttnCfsTask)["input_budget_figure"], property)

    module = task()
    with pytest.raises(ValueError, match="no resolved warm-up budget"):
        module.input_budget_figure(_PACKAGE_DIR)


def test_the_page_seams_are_the_same_objects_both_cells_of_this_row_draw_with(task) -> None:
    """The encoder edge on the diagnostic page: two runs read side by side must be drawn by one
    builder, or a difference in a figure is a difference between two plotting routines."""
    for name in ("forecast_rows", "input_stream_panels", "input_budget_figure"):
        assert name in vars(SeqVaeLagAttnCrwsTask), f"{name} moved off the causal parent"
        assert name not in vars(SeqVaeLagAttnTrfCrwsTask)
        assert (
            getattr(SeqVaeLagAttnTrfCrwsTask, name)
            is SeqVaeLagAttnCrwsTask.__dict__[name]
        ), name


def test_the_lag_caveat_the_drawn_page_carries_is_this_rows_one_sided_one() -> None:
    """Read off the constant the rows draw rather than re-stated here, so the two cells of this row
    cannot disagree about what a lag axis on this page means.

    The caveat is **one-sided**: the target is a raw sample with no group delay at all, so only the
    inputs carry one and the anchor itself is exact. Stating the causal-feature page's two-sided
    correction here would be wrong in the direction that reads as more careful, which is why the
    string is asserted rather than merely assumed present.
    """
    caveat = causal_raw_page.LAG_TIME_CAVEAT

    assert "the raw target carries no group delay" in caveat
    assert "input side" in caveat
    assert "13-791 s" in caveat
