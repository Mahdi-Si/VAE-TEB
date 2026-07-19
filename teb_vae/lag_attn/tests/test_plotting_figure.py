r"""The diagnostic figure builder, exercised with no Trainer and no callback.

A 500-line figure builder fails in ways that "the file exists and is non-trivial in size" cannot
see: a dropped row, a title bound to the wrong panel, an axes left empty because its data key was
renamed. So these tests forward a real model, hand the output dict straight to
:func:`_build_diagnostic_figure`, and check the structure of the returned figure -- the row count,
that each row's title is the intended one in the intended order, and that no row is blank.
"""
from __future__ import annotations

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn  # noqa: E402
from teb_vae.lag_attn.plotting import (  # noqa: E402
    _build_companion_figure,
    _build_diagnostic_figure,
)

# The fixed prefix of every row title, in top-to-bottom order, for a batch that carries raw signals
# and up_st. Dynamic tails (d_z, L, H_d, channel counts, vmax) are excluded so the check pins the
# panel identity without pinning a runtime number.
_TITLE_PREFIXES_WITH_RAW = (
    "Raw FHR / UP signals",
    "FHR features — scattering (rows 0-",
    "UP features — scattering (rows 0-",
    "Latent z (d_z=",
    "Posterior vs Prior means (TEB residual = posterior − prior)",
    "KLD per latent dim (d_z=",
    "Total KL per timestep vs mean attention entropy",
    "Lag attention — mean over ",
    "TE lag attribution (KL × mean-α) — p99-clipped ",
    "TE lag attribution — column-normalised ",
    "Average forecast μ_full — overlap-averaged per-anchor ",
    "Single-horizon forecast μ_full — non-overlapping ",
)


def _forward(prod_kwargs):
    """Build a tiny model and return ``(model, outs, tensors)`` under a pinned seed.

    The VAE samples inside ``forward``, so the seed is set immediately before it: without that the
    output -- and therefore the figure -- would differ run to run.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    b, t = 2, model.sequence_length
    generator = torch.Generator().manual_seed(1)
    y_st = torch.randn(b, t, 43, generator=generator)
    y_ph = torch.randn(b, t, 66, generator=generator)
    up_st = torch.randn(b, t, 43, generator=generator)
    up_ph = torch.randn(b, t, 15, generator=generator)
    u_stream = torch.cat([up_st, up_ph], dim=-1)
    fhr_raw = torch.randn(b, t * 16, generator=generator)
    up_raw = torch.randn(b, t * 16, generator=generator)
    torch.manual_seed(0)
    with torch.no_grad():
        outs = model(y_st, y_ph, u_stream)
    return model, outs, (y_st, y_ph, up_st, up_ph, fhr_raw, up_raw)


def _titled_axes(fig):
    """The figure's main row axes: the ones that carry a title (twins/caxes/lag axes do not)."""
    return [ax for ax in fig.axes if ax.get_title()]


def _build(model, outs, tensors, *, up_st, fhr_raw, up_raw):
    y_st, y_ph, up_st_t, up_ph, fhr_raw_t, up_raw_t = tensors
    fig = _build_diagnostic_figure(
        outs=outs,
        y_st=y_st,
        y_ph=y_ph,
        up_st=up_st_t if up_st else None,
        up_ph=up_ph,
        fhr_raw=fhr_raw_t if fhr_raw else None,
        up_raw=up_raw_t if up_raw else None,
        sample_idx=0,
        epoch=0,
        guid="test-guid",
        warmup=model.warmup_period,
        horizon=model.horizon,
        beta=0.01,
        feat_loss=0.0,
        base_loss=0.0,
        kld_loss=0.0,
    )
    return fig


def test_the_builder_returns_a_figure_without_a_trainer(prod_kwargs):
    model, outs, tensors = _forward(prod_kwargs)
    fig = _build(model, outs, tensors, up_st=True, fhr_raw=True, up_raw=True)
    try:
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)


def test_the_feature_row_ranges_are_derived_from_the_tensors(prod_kwargs):
    """The row ranges in the two heatmap titles must come from the data, not from a literal.

    They were literals once ("rows 43-86", "rows 43-100") and went stale when the dataset's
    phase-harmonic selection changed the widths -- a heatmap mislabelling 15 rows as "43-100" is
    invisible in a loss curve. ``_TITLE_PREFIXES_WITH_RAW`` deliberately pins only the static
    part of these titles, so this is the test that would catch a re-hardcoded range.
    """
    model, outs, tensors = _forward(prod_kwargs)
    y_st, y_ph, up_st, up_ph = tensors[0], tensors[1], tensors[2], tensors[3]
    fig = _build(model, outs, tensors, up_st=True, fhr_raw=True, up_raw=True)
    try:
        titles = [ax.get_title() for ax in _titled_axes(fig)]
        st, c_y = y_st.shape[-1], y_st.shape[-1] + y_ph.shape[-1]
        up_st_ch, c_u = up_st.shape[-1], up_st.shape[-1] + up_ph.shape[-1]
        fhr_expected = f"scattering (rows 0-{st - 1})  |  phase (rows {st}-{c_y - 1})"
        up_expected = (
            f"scattering (rows 0-{up_st_ch - 1})  |  self-phase (rows {up_st_ch}-{c_u - 1})"
        )
        assert any(fhr_expected in t for t in titles), f"{fhr_expected!r} not in {titles}"
        assert any(up_expected in t for t in titles), f"{up_expected!r} not in {titles}"
    finally:
        plt.close(fig)


def test_every_row_is_present_titled_and_non_empty(prod_kwargs):
    """Twelve rows with raw, each carrying its intended title in order, each with data drawn."""
    model, outs, tensors = _forward(prod_kwargs)
    fig = _build(model, outs, tensors, up_st=True, fhr_raw=True, up_raw=True)
    try:
        rows = _titled_axes(fig)
        assert len(rows) == len(_TITLE_PREFIXES_WITH_RAW)
        for ax, prefix in zip(rows, _TITLE_PREFIXES_WITH_RAW):
            assert ax.get_title().startswith(prefix), (
                f"row title {ax.get_title()!r} does not start with {prefix!r}"
            )
            assert ax.has_data(), f"row {prefix!r} drew nothing"
    finally:
        plt.close(fig)


def test_the_raw_row_drops_out_when_raw_signals_are_absent(prod_kwargs):
    """Eleven rows without raw; the first row becomes the FHR-features heatmap."""
    model, outs, tensors = _forward(prod_kwargs)
    fig = _build(model, outs, tensors, up_st=True, fhr_raw=False, up_raw=False)
    try:
        rows = _titled_axes(fig)
        assert len(rows) == len(_TITLE_PREFIXES_WITH_RAW) - 1
        assert rows[0].get_title().startswith("FHR features")
    finally:
        plt.close(fig)


def test_the_up_feature_title_collapses_without_up_st(prod_kwargs):
    """With up_st absent the UP row shows self-phase only and says so."""
    model, outs, tensors = _forward(prod_kwargs)
    fig = _build(model, outs, tensors, up_st=False, fhr_raw=True, up_raw=True)
    try:
        up_titles = [ax.get_title() for ax in _titled_axes(fig) if ax.get_title().startswith("UP features")]
        assert up_titles == ["UP features — self-phase only (up_st absent)"]
    finally:
        plt.close(fig)


def test_the_companion_covers_every_channel_and_finds_the_worst(prod_kwargs):
    """Calibration is reported for all $c_y$ channels, and the drawn band is data-selected.

    This replaced three hand-picked ``forecast_channels``. Three spot-checks out of $c_y$ cannot
    show a subset of badly-calibrated channels, and choosing them presupposed knowing which
    channels were worth looking at -- which the model cannot tell you, since its TE readout is a
    per-step scalar with no channel axis at all.

    Sabotaging one channel's predictive variance is what makes this test non-vacuous: a panel that
    merely drew a fixed channel would not follow the sabotage.
    """
    model, outs, tensors = _forward(prod_kwargs)
    y_st, y_ph = tensors[0], tensors[1]
    c_y = int(y_st.shape[-1] + y_ph.shape[-1])

    # A near-zero variance makes this channel's +-2 sigma band far too narrow to contain the
    # truth, collapsing its coverage while leaving every other channel untouched.
    sabotaged = 57
    outs = dict(outs)
    logvar = outs["logvar_full"].clone()
    logvar[..., sabotaged] = -12.0
    outs["logvar_full"] = logvar

    fig = _build_companion_figure(
        outs=outs,
        y_st=y_st,
        y_ph=y_ph,
        kld_shuffled_per_t=outs["kld_per_t"],
        sample_idx=0,
        epoch=0,
        guid="TEST",
        warmup=model.warmup_period,
        horizon=model.horizon,
        forecast_anchor_frac=0.6,
        kld_active_frac=0.5,
        kld_shuffled_scalar=0.01,
    )
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert any(f"across all {c_y} channels" in t for t in titles), titles
        assert any(f"worst: ch {sabotaged}" in t for t in titles), titles
        assert any(f"worst channel {sabotaged}" in t for t in titles), titles
    finally:
        plt.close(fig)
