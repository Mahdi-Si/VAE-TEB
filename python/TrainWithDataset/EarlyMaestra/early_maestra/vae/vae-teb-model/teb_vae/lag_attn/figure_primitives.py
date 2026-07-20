r"""Figure helpers shared by the training callback and the evaluation pipeline.

These were originally private to :mod:`teb_vae.lag_attn.plotting`, which is a Lightning
``Callback`` module. The evaluation pipeline needs the same conversions -- the same overlap
averaging, the same symmetric colour limit, the same lag-seconds axis -- and had exactly two
options: import a Lightning callback module into an offline analysis script, or keep a second
copy that a test would have to keep proving identical to the first.

Neither is good, and neither is necessary: every helper here is pure ``numpy`` / ``torch`` /
duck-typed matplotlib. Lifting them into a module of their own leaves **one** copy in the tree,
importable from both sides, and drags no framework into the eval path.

The colour literals come with them. They are deliberately *not* re-exported from
``utils.style``: two of the eight genuinely differ (``COLOR_PURPLE`` and ``COLOR_BLACK``), the
figures depend on these exact hues, and ``utils.style``'s palette is a separate evolving set.
They live here rather than in ``plotting.py`` because :func:`shade_warmup` takes one as a
default argument -- had they stayed behind, this module would have to import from ``plotting``
while ``plotting`` imports from it.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

# The figures depend on these exact hues. Only the colours the builders actually draw with are
# kept; see the module docstring for why they are not sourced from ``utils.style``.
COLOR_BLUE = "#3F72AF"
COLOR_ORANGE = "#FFB200"
COLOR_GREEN = "#46D855"
COLOR_PURPLE = "#5642EB"
COLOR_VERMILLION = "#F23F04"
COLOR_GRAY = "#393E46"
COLOR_BLACK = "#000000"
COLOR_LIGHT_GRAY = "#EEEEEE"


# =============================================================================
# Conversions
# =============================================================================
def to_numpy(tensor: Any) -> np.ndarray:
    """Detach a tensor to a CPU float32 array, passing anything else to ``numpy.asarray``.

    Args:
        tensor: A tensor, an array, or anything ``numpy.asarray`` accepts.

    Returns:
        A numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor)


def future_target(y_st: torch.Tensor, y_ph: torch.Tensor, horizon: int) -> torch.Tensor:
    r"""Unfold $Y^{+}$: at anchor $t$, the window ``Y[t+1 : t+1+H_d]``.

    The same construction ``compute_loss`` performs internally, so a forecast can be scored
    against the identical target the training objective used.

    Args:
        y_st: FHR scattering features ``(B, T, 43)``.
        y_ph: FHR phase-harmonic features ``(B, T, 66)``.
        horizon: Forecast horizon $H_d$.

    Returns:
        ``(B, T - H_d, H_d, c_y)``.
    """
    target = torch.cat([y_st, y_ph], dim=-1)
    return target[:, 1:, :].unfold(dimension=1, size=horizon, step=1).permute(0, 1, 3, 2)


def kld_per_dim_np(
    mu_prior: np.ndarray,
    logvar_prior: np.ndarray,
    mu_post: np.ndarray,
    logvar_post: np.ndarray,
) -> np.ndarray:
    r"""Closed-form diagonal-Gaussian KL, per timestep per latent dimension.

    The numpy mirror of ``SeqVaeLagAttn.kld_tensor``, for figure code that already holds arrays.
    A consumer that still has tensors should call the model's method instead, so the reported
    number and the trained number come from one formula.

    Args:
        mu_prior: ``(T, d_z)`` prior mean.
        logvar_prior: ``(T, d_z)`` prior log-variance.
        mu_post: ``(T, d_z)`` posterior mean.
        logvar_post: ``(T, d_z)`` posterior log-variance.

    Returns:
        ``(T, d_z)`` per-step per-dimension KL in nats.
    """
    return 0.5 * (
        logvar_prior
        - logvar_post
        + (np.exp(logvar_post) + (mu_post - mu_prior) ** 2) / np.exp(logvar_prior)
        - 1.0
    )


def time_axes(T: int, R: int, fs_raw: float = 4.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(time_raw_sec, time_dec_sec, t_max_sec)`` for unified alignment.

    All diagnostic rows share a single physical-time axis. The raw FHR/UP trace lives on
    ``time_raw`` (step ``1/fs_raw``); decimated features and latents live on ``time_dec`` (step
    ``t_max / T``). The imshow ``extent`` and every ``ax.set_xlim`` call use ``(0.0, t_max_sec)``.

    Args:
        T: Number of decimated steps (e.g. 300).
        R: Number of raw samples (e.g. 4800).
        fs_raw: Raw sampling rate in Hz.

    Returns:
        ``(time_raw, time_dec, t_max_sec)``: the ``(R,)`` raw seconds axis, the ``(T,)`` decimated
        seconds axis with step centres at $0, \\Delta, 2\\Delta, \\ldots$, and the window length
        in seconds.
    """
    t_max = float(R) / float(fs_raw)
    time_raw = np.arange(R, dtype=np.float64) / float(fs_raw)
    time_dec = np.arange(T, dtype=np.float64) * (t_max / float(T))
    return time_raw, time_dec, t_max


# =============================================================================
# Axes decoration
# =============================================================================
def attach_lag_seconds_axis(ax: Any, step_seconds: float, delta_up_seconds: float) -> Any:
    r"""Add a right-hand secondary y-axis in physical seconds.

    Maps a decimated lag index $\ell$ to $\mathrm{lag}_{\mathrm{phys}}(\ell) = s\,\ell +
    \Delta_{UP}$, so the lag panels read in both model-lag and physical-second coordinates.

    Non-fatal by design: any matplotlib error is swallowed, because this is called from a
    training callback where a failed axis decoration must not take down a run.

    Args:
        ax: The lag-panel axes, whose primary y is the decimated lag $\ell$.
        step_seconds: Decimated step duration $s$ in seconds.
        delta_up_seconds: Fixed preprocessing UP shift $\Delta_{UP}$ in seconds.

    Returns:
        The created secondary axis, or ``None`` if it could not be attached.
    """
    s = float(step_seconds)
    d = float(delta_up_seconds)
    if s <= 0.0:
        return None
    try:
        sec = ax.secondary_yaxis(
            "right",
            functions=(lambda l: s * l + d, lambda v: (v - d) / s),
        )
        sec.set_ylabel("Lag (s)", fontsize=8)
        return sec
    except Exception:  # noqa: BLE001 — plotting must never crash training
        return None


def shade_warmup(
    ax: Any,
    warmup: int,
    t_max: float,
    T: int,
    *,
    color: str = COLOR_LIGHT_GRAY,
) -> None:
    """Shade the first ``warmup`` decimated steps, in seconds, on an axes.

    Args:
        ax: Target axes.
        warmup: Warm-up length in decimated steps.
        t_max: Full x-axis extent in seconds (``R / fs_raw``).
        T: Total number of decimated steps, for the step-to-seconds conversion.
        color: Shading colour.
    """
    if warmup and warmup > 0 and T > 0:
        warmup_sec = float(warmup) * (t_max / float(T))
        ax.axvspan(0.0, warmup_sec, color=color, alpha=0.35, zorder=0)


# =============================================================================
# Forecast rendering
# =============================================================================
def average_forecast_per_channel(
    mu_pred: np.ndarray,
    T: int,
    H_d: int,
    warmup: int,
) -> np.ndarray:
    """Average overlapping per-anchor horizon forecasts onto the decimated axis.

    Anchor $t \\in [\\mathrm{warmup},\\ T - H_d)$ contributes its per-horizon prediction
    ``mu_pred[t, h, :]`` to the target decimated index $\\tau = t + 1 + h$. The result averages
    every anchor's contribution to each $\\tau$. Positions with no contributing anchor are
    ``NaN``, so they render as gaps rather than as a fabricated zero.

    Args:
        mu_pred: ``(T, H_d, C)`` per-anchor horizon prediction.
        T: Number of decimated steps.
        H_d: Forecast horizon in decimated steps.
        warmup: Warm-up length in decimated steps.

    Returns:
        ``(T, C)`` averaged forecast, float32, ``NaN`` where uncovered.
    """
    C = mu_pred.shape[-1]
    acc = np.zeros((T, C), dtype=np.float64)
    cnt = np.zeros((T,), dtype=np.float64)
    t_start = max(int(warmup), 0)
    t_end = max(t_start, T - H_d)
    for t in range(t_start, t_end):
        tau_end = min(t + 1 + H_d, T)
        tau = np.arange(t + 1, tau_end)
        h = tau - (t + 1)
        acc[tau] += mu_pred[t, h, :]
        cnt[tau] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = acc / np.where(cnt > 0.0, cnt, 1.0)[:, None]
    avg[cnt == 0.0] = np.nan
    return avg.astype(np.float32)


def concat_single_forecasts(
    mu_pred: np.ndarray,
    T: int,
    H_d: int,
    warmup: int,
) -> np.ndarray:
    """Non-overlapping, stride-``H_d`` concatenation of per-anchor horizons.

    Starting at ``t = warmup``, walk forward in strides of ``H_d`` anchors; each anchor
    contributes its full horizon slice ``[t+1, t+1+H_d)``. Uncovered positions stay ``NaN``.

    Args:
        mu_pred: ``(T, H_d, C)`` per-anchor horizon prediction.
        T: Number of decimated steps.
        H_d: Forecast horizon in decimated steps.
        warmup: Warm-up length in decimated steps.

    Returns:
        ``(T, C)`` concatenated forecast, float32, ``NaN`` where uncovered.
    """
    C = mu_pred.shape[-1]
    out = np.full((T, C), np.nan, dtype=np.float32)
    t = max(int(warmup), 0)
    while t + 1 + H_d <= T and t < T:
        out[t + 1 : t + 1 + H_d, :] = mu_pred[t, :, :].astype(np.float32)
        t += H_d
    return out


def stack_feature_blocks(
    top: np.ndarray, bottom: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[int]]:
    """Vertically stack two feature blocks and return the index of the last row of ``top``.

    Args:
        top: ``(C_top, T)`` upper block.
        bottom: ``(C_bot, T)`` lower block, or ``None``.

    Returns:
        ``(stacked, separator_row)``. ``separator_row`` is ``C_top - 1``, the *row index* of the
        top block's last channel -- a caller drawing the boundary line wants
        ``separator_row + 0.5``, which is where the two blocks actually meet. ``None`` when there
        is no bottom block.
    """
    if bottom is None:
        return top, None
    stacked = np.concatenate([top, bottom], axis=0)
    return stacked, top.shape[0] - 1


def safe_vabs(arr: np.ndarray) -> float:
    """Return a strictly positive symmetric colour limit for a diverging imshow.

    Ignores ``NaN`` and ``Inf``. Falls back to $1.0$ when the array has no finite values or its
    finite maximum is zero -- a ``vmax`` of $0$ would make every cell the same colour and read as
    a uniformly zero field rather than as an empty one.

    Args:
        arr: The array to be rendered.

    Returns:
        A positive symmetric limit.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    vabs = float(np.abs(finite).max())
    return vabs if vabs > 0.0 else 1.0
