r"""The diagnostic page, rendered for this model at the shipped geometry and reach budget.

No figure code is written here. The page is the shared seven-row builder; two of its rows are the
feature-domain sibling's ``feature_forecast_rows``, whose layout, lanes and labels that sibling's own
suite pins. The binding of this net's channel facts into those rows is asserted on the
``functools.partial`` in ``test_task.py``; this file renders the page once at production size.

**The lag axes are read after a draw.** Matplotlib defers a secondary axis's limits to draw time, so an
assertion made before one passes against the default $(0, 1)$ whatever the transform is -- which is how
the historical failure this pairing could repeat went unnoticed: two consumers each reached into the
model for the causal input delay under a name of their own guessing, one of those names did not exist,
and the figure's lag axis was silently short by two minutes. The delay accessor exists on both bases,
so what is new here is the *value* pairing rather than the axis arithmetic: one test, not the sibling's
five.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    COMPENSATED_LAG_AXIS_LABEL,
    lag_compensated_seconds,
)
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.plotting import _source_delay_steps  # noqa: E402

from .conftest import SHIPPED_KWARGS, make_patterned_batch  # noqa: E402

#: Plausible raw-signal scales, so the raw context row has something to invert. Only the two raw
#: signals carry statistics: the forecast target is the loader's ``normalize_fields`` output used as
#: delivered, and no model in this family adds a second normalisation for it.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: The two lag panels' title prefixes, in the order the page lays them out.
_LAG_PANELS = ("Lag attention", r"$\widetilde K_{t,\ell}$")


def _forward(module: Any, data: Any) -> dict:
    """Run the net once and return everything the page builder needs.

    Separate from :func:`_render` because the forward is **stochastic** -- the full branch decodes a
    reparameterised draw -- so a test comparing a drawn artist against the arrays behind it has to
    compare against *this* forward's, not a second one's.

    Args:
        module: A ``SeqVaeLagAttnTrfFsTask``.
        data: The batch to run on.

    Returns:
        ``{'outs', 'kld_per_dim', 'target'}``.
    """
    model = module.orig_model
    with torch.no_grad():
        outs = model(*module._build_forward_inputs(data))
        target, _weight = module._build_raw_target(data)
        kld_per_dim = model.kld_tensor(
            mu_prior=outs["mu_prior"],
            logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"],
            logvar_post=outs["logvar_post"],
        )
    return {"outs": outs, "kld_per_dim": kld_per_dim, "target": target}


def _render(module: Any, data: Any, pieces: Any = None, *, draw: bool = False, **overrides) -> Any:
    """Build the whole page for sample 0, through the seam the callback resolves.

    Args:
        module: A ``SeqVaeLagAttnTrfFsTask``.
        data: The batch to draw from.
        pieces: A :func:`_forward` result to draw, or ``None`` to run one.
        draw: Force a canvas draw before returning, so deferred axis limits are real.
        **overrides: Passed to the builder, e.g. ``normalization_stats=None``.

    Returns:
        The matplotlib ``Figure``. The caller closes it.
    """
    pieces = _forward(module, data) if pieces is None else pieces
    kwargs = dict(
        outs=pieces["outs"],
        kld_per_dim=pieces["kld_per_dim"],
        # The builder's parameter is still named for the raw models' target; what this model passes
        # through it is the concatenated feature stream its loss was computed against.
        fhr_raw=pieces["target"],
        geometry=module.orig_model.geometry,
        sample_index=0,
        epoch=3,
        guid="rec-0001",
        beta=0.25,
        scalars={"pred_gap": 0.5},
        up_raw=data.up,
        normalization_stats=_STATS,
        # The callback's own probe, not a constant: this is the value under test.
        delay_steps=_source_delay_steps(module.orig_model),
        # Exactly what `LagAttnRwsPlotCallback._generate_plots` passes.
        forecast_rows=module.forecast_rows,
        batch=data,
    )
    kwargs.update(overrides)
    figure = plotting.build_diagnostic_figure(**kwargs)
    if draw:
        figure.canvas.draw()
    return figure


def _axes_titled(figure: Any, prefix: str) -> Any:
    """Return the single axes whose title starts with ``prefix``."""
    matches = [ax for ax in figure.axes if ax.get_title().startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} panel, found {len(matches)}"
    return matches[0]


def _error_map(figure: Any) -> Any:
    """Return the forecast row's inset error map."""
    insets = _axes_titled(figure, "Forecast").child_axes
    assert len(insets) == 1, f"expected one inset on the forecast row, found {len(insets)}"
    return insets[0]


def _lag_axes(figure: Any) -> List[Tuple[str, Any, Any]]:
    """Return ``(title prefix, panel, secondary axis)`` for each of the two lag panels."""
    found = []
    for prefix in _LAG_PANELS:
        matches = [ax for ax in figure.axes if ax.get_title().startswith(prefix)]
        assert len(matches) == 1, f"expected one {prefix!r} panel, found {len(matches)}"
        panel = matches[0]
        assert len(panel.child_axes) == 1, f"{prefix}: {len(panel.child_axes)} secondary axes"
        found.append((prefix, panel, panel.child_axes[0]))
    return found


def test_the_shipped_page_renders_the_budgets_block_and_the_compensated_lag_axis(
    task, shipped_gated
):
    r"""One render at the production geometry and the shipped reach budget, read after a draw.

    The page must carry seven titled rows, a $C_{\mathrm{keep}} \times H$ error map with the stored
    block boundary drawn where the kept channels cross it, and -- the pairing this file exists for --
    lag panels whose secondary axis is $4(\ell + \delta)$ seconds for the model's *own* input delay
    $\delta$, identical on both panels and labelled *compensated*. A zero-offset axis is what an
    unresolved delay silently produces.
    """
    module = task(model_kwargs=shipped_gated)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    figure = _render(module, batch, draw=True)
    try:
        model = module.orig_model
        assert len([ax for ax in figure.axes if ax.get_title()]) == 7
        assert _error_map(figure).images[0].get_array().shape == (
            model.decoder_out_channels,
            model.horizon,
        )
        keep = np.asarray([int(value) for value in shipped_gated["target_keep_index"]])
        expected = int(np.count_nonzero(keep < model.TARGET_BLOCK_SPLIT))
        boundaries = [line.get_ydata()[0] for line in _error_map(figure).lines]
        assert boundaries == pytest.approx([expected - 0.5])

        delay = int(model.source_delay_steps)
        assert delay == _source_delay_steps(model) > 0, (
            "an unguarded model makes every equality below trivial"
        )
        seen = []
        for prefix, panel, secondary in _lag_axes(figure):
            low, high = panel.get_ylim()
            expected_limits = (
                float(lag_compensated_seconds(low, delay_steps=delay)),
                float(lag_compensated_seconds(high, delay_steps=delay)),
            )
            assert secondary.get_ylim() == pytest.approx(expected_limits), prefix
            assert secondary.get_ylabel() == COMPENSATED_LAG_AXIS_LABEL, prefix
            seen.append(secondary.get_ylim())
        assert seen[0] == pytest.approx(seen[1])
    finally:
        plt.close(figure)
