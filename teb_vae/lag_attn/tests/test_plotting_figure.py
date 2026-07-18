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
from teb_vae.lag_attn.plotting import _build_diagnostic_figure  # noqa: E402

# The fixed prefix of every row title, in top-to-bottom order, for a batch that carries raw signals
# and up_st. Dynamic tails (d_z, L, H_d, channel counts, vmax) are excluded so the check pins the
# panel identity without pinning a runtime number.
_TITLE_PREFIXES_WITH_RAW = (
    "Raw FHR / UP signals",
    "FHR features — scattering (rows 0-42)  |  phase (rows 43-86)",
    "UP features — scattering (rows 0-42)  |  self-phase (rows 43-100)",
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
    y_ph = torch.randn(b, t, 44, generator=generator)
    up_st = torch.randn(b, t, 43, generator=generator)
    up_ph = torch.randn(b, t, 58, generator=generator)
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
