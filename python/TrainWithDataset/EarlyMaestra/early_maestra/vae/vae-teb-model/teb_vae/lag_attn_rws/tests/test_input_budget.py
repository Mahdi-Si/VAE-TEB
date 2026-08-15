r"""The input rows and the causal-input-budget figure.

Two claims are worth a test here, and they are the two a reader of these figures will act on.

**The rows are the encoder's input, not a reconstruction of it.** They are built from the tensors
the forward pass consumed, put through the model's own gates, so the assertion below is an exact
equality against ``gate(cat([y_st, y_ph]))`` rather than a tolerance against a re-application of
the budget. A figure that merely *agreed* with the model most of the time would be worse than no
figure, because its disagreements would be invisible.

**A kept channel cannot see past the anchor and a dropped one could have.** That is the whole
content of the run-level figure, and it is asserted off the drawn rectangles rather than off the
arithmetic that positions them: the arithmetic is already covered by ``test_channel_reach.py``,
and what is new here is whether the picture says what the arithmetic means.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.channel_reach import (  # noqa: E402
    block_center_hz,
    block_reach_seconds,
    resolve_stream_budgets,
)
from teb_vae.lag_attn.nets.delays import ChannelGate  # noqa: E402
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_rws import input_budget  # noqa: E402
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry  # noqa: E402
from teb_vae.lag_attn_rws.sample_page import build_diagnostic_figure  # noqa: E402

#: The shipped guard, resolved once: the figures under test are drawn against it, and it is the
#: configuration whose tradeoff a reader is checking.
SHIPPED_BUDGET_S = 120.0


def _shipped_model() -> Any:
    """A stand-in carrying the shipped guard, and nothing else the figures read.

    A namespace rather than the net: :func:`~teb_vae.lag_attn_rws.input_budget.describe_streams`
    reads four attributes, and building a $10^7$-parameter model to supply them would make this
    test about the constructor.

    Returns:
        An object with ``c_y``, ``c_u``, ``use_up_st``, both gates and a geometry.
    """
    budget = resolve_stream_budgets(
        {"causal_reach_budget_s": SHIPPED_BUDGET_S, "use_up_st": True, "warmup_period": 30}
    )
    assert budget is not None
    return types.SimpleNamespace(
        c_y=109,
        c_u=58,
        use_up_st=True,
        target_gate=ChannelGate(
            declared_width=109,
            keep_index=budget.target_keep_index,
            delays=budget.target_delays,
        ),
        source_gate=ChannelGate(
            declared_width=58,
            keep_index=budget.source_keep_index,
            delays=budget.source_delays,
        ),
        geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30),
    )


# =============================================================================
# The channel description
# =============================================================================
def test_the_frequency_map_indexes_the_same_channels_as_the_reach_map():
    """The two vectors are drawn against one channel axis, so a width difference would label
    every channel of the offending block with another channel's frequency."""
    reaches, frequencies = block_reach_seconds(), block_center_hz()

    assert set(reaches) == set(frequencies)
    for block, values in reaches.items():
        assert len(frequencies[block]) == len(values), block

    scattering = np.array(frequencies["fhr_st"])
    # The order-0 low-pass has no centre frequency, and is reported as absent rather than as
    # 0 Hz: the second would band it as the slowest wavelet of the bank.
    assert np.isnan(scattering[0])
    # kymatio's bank descends with index, and the whole y-axis reading of both figures depends
    # on it.
    assert np.all(np.diff(scattering[1:]) < 0)


def test_describe_streams_reports_the_gate_the_model_holds():
    """Read off the model's gates, so a figure cannot describe a guard the loaded model has
    not got -- which is exactly the case its reader is checking."""
    target, source = input_budget.describe_streams(_shipped_model())

    assert (target.name, source.name) == ("target", "source")
    assert (target.declared_width, source.declared_width) == (109, 58)
    assert (target.kept_width, source.kept_width) == (78, 29)
    assert target.max_delay == source.max_delay == 30

    # Re-based into surviving-channel coordinates: under a budget the two blocks lose different
    # numbers of channels, so the boundary is not where the declared widths put it.
    assert target.kept_block_spans() == (("fhr_st", 0, 27), ("fhr_ph", 27, 78))
    assert source.kept_block_spans() == (("up_st", 0, 27), ("up_ph", 27, 29))


def test_describe_streams_refuses_a_width_the_filter_bank_cannot_describe():
    """The reach and frequency vectors are positional into the declared width, so a mismatch
    would mislabel every channel rather than fail."""
    model = types.SimpleNamespace(c_y=7, c_u=58, use_up_st=True, target_gate=None,
                                  source_gate=None)

    with pytest.raises(ValueError, match=r"c_y=7 disagrees"):
        input_budget.describe_streams(model)


def test_an_unguarded_model_keeps_every_channel_at_no_delay():
    """The unguarded configuration is represented by having no gate at all, and must describe
    as the model that has no delay rather than as a guard with zero delays."""
    model = types.SimpleNamespace(c_y=109, c_u=58, use_up_st=True, target_gate=None,
                                  source_gate=None)

    target, source = input_budget.describe_streams(model)

    assert (target.kept_width, source.kept_width) == (109, 58)
    assert target.max_delay == source.max_delay == 0


# =============================================================================
# The per-sample rows
# =============================================================================
def test_the_rows_draw_exactly_what_the_encoder_receives(task, stub_batch):
    """Equality, not a tolerance: the rows go through the model's own gates on the tensors the
    forward pass consumed, so anything short of an exact match is a defect in the seam."""
    module = task()
    model = module.orig_model
    inputs = module._build_forward_inputs(stub_batch)

    target, source = input_budget.stream_panels(model, inputs, sample_index=1)

    expected_target = torch.cat([inputs[0], inputs[1]], dim=-1)[1].numpy()
    assert np.array_equal(target.values, expected_target)
    assert np.array_equal(source.values, inputs[2][1].numpy())
    assert target.center_hz.size == target.values.shape[1]


def test_a_delayed_channel_starts_at_its_own_delay(stub_batch):
    """The guard emits zero for the first $\\delta_c$ steps of a channel, and that span is what
    the row's staircase marks. Drawn from the gated tensor, so the mark cannot drift from it."""
    delays = (0, 3, 5)
    model = types.SimpleNamespace(
        c_y=109,
        c_u=58,
        use_up_st=True,
        target_gate=ChannelGate(declared_width=109, keep_index=(0, 1, 2), delays=delays),
        source_gate=None,
    )
    inputs = (
        stub_batch.fhr_st,
        stub_batch.fhr_ph,
        torch.cat([stub_batch.up_st, stub_batch.up_ph], dim=-1),
    )

    target, _source = input_budget.stream_panels(model, inputs, sample_index=0)

    assert tuple(target.delays) == delays
    for channel, delay in enumerate(delays):
        assert np.all(target.values[:delay, channel] == 0.0)
        assert np.any(target.values[delay:, channel] != 0.0)


def test_the_page_gains_one_row_per_stream_and_is_unchanged_without_them(task, stub_batch):
    """Additive: a caller that passes no panels gets exactly the page it got before."""
    module = task()
    model = module.orig_model
    inputs = module._build_forward_inputs(stub_batch)
    with torch.no_grad():
        outs = model(*inputs)
        kld = model.kld_tensor(
            mu_prior=outs["mu_prior"], logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"], logvar_post=outs["logvar_post"],
        )
    common: dict[str, Any] = dict(
        outs=outs, kld_per_dim=kld, fhr_raw=stub_batch.fhr, geometry=model.geometry,
        sample_index=0, epoch=1, guid="rec-0001", beta=0.5, scalars={},
        up_raw=stub_batch.up, batch=stub_batch,
    )

    without = build_diagnostic_figure(**common)
    # The second build is inside the guard: raising there would otherwise orphan the first page,
    # which is a ~20-inch figure pyplot would hold for the rest of the session.
    with_rows = None
    try:
        with_rows = build_diagnostic_figure(
            **common, input_streams=input_budget.stream_panels(model, inputs)
        )
        added = [ax.get_title() for ax in with_rows.axes if ax.get_title()]
        assert sum(title.startswith("Model input — target") for title in added) == 1
        assert sum(title.startswith("Model input — source") for title in added) == 1
        # Every other row is the same panel with the same title, in the same order: the two rows
        # are inserted into the page, not a re-layout of it.
        before = [ax.get_title() for ax in without.axes if ax.get_title()]
        assert [title for title in added if not title.startswith("Model input")] == before
    finally:
        plt.close(without)
        if with_rows is not None:
            plt.close(with_rows)


# =============================================================================
# The run-level figure
# =============================================================================
def test_a_kept_channel_stops_at_the_anchor_and_a_dropped_one_would_not():
    """The figure's whole claim, asserted off the drawn rectangles: a kept bar ends at or before
    the anchor's causal endpoint, and a dropped bar -- drawn where it would sit had it been read
    at the anchor -- runs past it into the window being forecast."""
    figure = input_budget.build_input_budget_figure(_shipped_model())
    try:
        drawn = 0
        for axes in figure.axes:
            for container in axes.containers:
                label = container.get_label()
                for patch in container.patches:
                    right = patch.get_x() + patch.get_width()
                    drawn += 1
                    if label.startswith("kept"):
                        assert right <= 1e-9, (
                            f"a kept channel reaches {right:.3g} s past the anchor"
                        )
                    else:
                        assert right > 0.0, "a dropped channel would not have crossed the anchor"
        assert drawn == 109 + 58
    finally:
        plt.close(figure)


def test_the_forecast_window_is_the_models_own_horizon():
    """Drawn from the geometry rather than from a constant, so a horizon change moves the shaded
    span with it instead of leaving the figure quietly describing the previous one."""
    model = _shipped_model()
    figure = input_budget.build_input_budget_figure(model)
    try:
        expected = model.geometry.horizon * model.geometry.r / 4.0
        assert f"{expected:.0f} s forecast window" in figure._suptitle.get_text()
        axes = figure.axes[0]
        bars = {id(patch) for container in axes.containers for patch in container.patches}
        spans = [patch for patch in axes.patches if id(patch) not in bars]
        assert len(spans) == 1, "the forecast window is the only patch that is not a channel bar"
        left = float(spans[0].get_x())
        assert (left, left + float(spans[0].get_width())) == (0.0, expected)
    finally:
        plt.close(figure)


def test_the_figure_is_written_under_the_directory_it_is_given(tmp_path):
    """The callback hands it the diagnostics directory; nothing else decides where it lands."""
    path = input_budget.write_input_budget_figure(
        _shipped_model(), tmp_path / "diagnostics", file_format="png"
    )

    assert path == tmp_path / "diagnostics" / f"{input_budget.BUDGET_FIGURE_STEM}.png"
    assert path.exists() and path.stat().st_size > 0


def test_the_delay_is_reported_in_the_units_the_guard_resolves_it_in():
    """Steps in the arithmetic, seconds on the page: the summary states both, because a delay of
    $30$ means nothing without $\\Delta$ and the two are resolved in different modules."""
    target, _source = input_budget.describe_streams(_shipped_model())

    assert f"delay 0–{target.max_delay} steps" in target.summary()
    assert f"({target.max_delay * SECONDS_PER_STEP:g} s)" in target.summary()
