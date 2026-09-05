r"""Pure functions on tensors: forecast error, uplift, residual activity, KL aggregates.

Nothing here does I/O, holds a model, or reads a config. Every function takes the tensors and
the mask it should honour, so a caller can be certain which window a number was computed over
and a test can hand-build an input whose answer is known by hand.

**Two reductions, and they are different numbers.** The *pooled* form matches
``compute_loss`` exactly -- one global denominator over the whole batch -- and is what the
parity test compares against ``feat_loss``. The *per-sample* form divides each sample by its own
mask sum, which is what a per-sample CSV needs. They agree only when every sample has the same
mask density, which real data never has: a batch where one recording is half gaps has a pooled
mean dominated by the intact recordings and a per-sample mean that treats both equally. Neither
is wrong; reporting one under the other's name is.

**A fully masked sample yields ``NaN``, not $0$.** Zero is a legitimate value for every metric
here -- a perfect forecast has zero error -- so a zero returned for "no data" is
indistinguishable from a spectacular result, and it drags every downstream mean toward it. The
pooled form, which must match ``compute_loss``, keeps that function's ``clamp_min(1.0)``
instead; that is a deliberate asymmetry between the two, not an oversight.

**$R^2$ is computed against the *masked per-channel* mean.** Against a single scalar mean over
all channels, $SS_{\mathrm{tot}}$ would be dominated by the offsets *between* the 109 feature
channels rather than by the variance within each, and every $R^2$ would read high for a model
that had learned nothing but the channel means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from teb_vae.lag_attn.nets.model import _KLD_ACTIVE_EPS

#: Duration of one decimated step in seconds. The features are produced at 4 Hz and decimated
#: 16x, so one model step spans $16/4 = 4$ s.
STEP_SECONDS = 4.0

#: Imported rather than mirrored: a second literal is how the eval's notion of "active" and the
#: model's drift apart, and unlike ``trainer.py`` the nets package is framework-free -- reaching
#: into it costs nothing. Private by name because nothing outside the model was expected to need
#: it; this pipeline needs it precisely so that it does not invent a rival threshold.
KLD_ACTIVE_EPS = _KLD_ACTIVE_EPS


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
def masked_pooled_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Reduce with ``compute_loss``'s single global denominator.

    $$\bar{v} = \frac{\sum v \odot m}{\max\left(\sum m \cdot C,\ 1\right)}$$

    The channel factor is what makes the denominator count *entries* rather than mask cells:
    ``mask`` carries a trailing singleton channel axis, so its sum is short by $C$. Omitting the
    factor would multiply every reported loss by $C$ -- $109$ for this model -- which is large
    enough to be obviously wrong and small enough to be mistaken for a scale convention.

    Args:
        values: Per-element quantity, $(B, A, H_d, C)$.
        mask: Feature mask from :mod:`~teb_vae.lag_attn.eval.masks`, $(B, A, H_d, 1)$.

    Returns:
        A scalar tensor.
    """
    channels = float(values.shape[-1])
    denom = (mask.sum() * channels).clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_per_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Reduce per sample, yielding ``NaN`` where a sample has no unmasked entry.

    Args:
        values: Per-element quantity, $(B, A, H_d, C)$.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        $(B,)$, ``NaN`` for a fully masked sample.
    """
    channels = float(values.shape[-1])
    total = (values * mask).sum(dim=(1, 2, 3))
    count = mask.sum(dim=(1, 2, 3)) * channels
    return torch.where(count > 0, total / count.clamp_min(1.0), torch.full_like(total, float("nan")))


def _channel_slice(tensor: torch.Tensor, start: Optional[int], stop: Optional[int]) -> torch.Tensor:
    """Return a channel-axis slice, or the tensor itself when both bounds are ``None``."""
    if start is None and stop is None:
        return tensor
    return tensor[..., slice(start, stop)]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def per_element_loss(
    squared_error: torch.Tensor,
    logvar: torch.Tensor,
    *,
    likelihood: str,
    sigma_obs: Any,
) -> torch.Tensor:
    r"""Per-element loss under the checkpoint's objective, mirroring ``compute_loss``.

    Under ``'mse'`` this is the squared error itself. Under ``'gaussian_nll'`` it is the Gaussian
    negative log-likelihood in nats with the constant dropped:

    $$\ell = \tfrac{1}{2}\,\frac{(\mu - y)^2}{\sigma^2} + \tfrac{1}{2}\log\sigma^2$$

    where $\log\sigma^2$ is the decoder's own head under ``sigma_obs='learned'`` and a constant
    otherwise. The constant term $\tfrac{1}{2}\log 2\pi$ is dropped in both, matching training,
    so an eval NLL is comparable to a training NLL but is *not* a calibrated log density -- the
    calibration analysis adds the constant back before reporting one.

    Args:
        squared_error: $(\mu - y)^2$, any shape.
        logvar: The matching predictive log-variance, same shape. Ignored under ``'mse'``.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        sigma_obs: ``'learned'`` to use ``logvar``, or a positive scalar observation noise.

    Returns:
        The per-element loss, same shape as ``squared_error``.

    Raises:
        ValueError: If ``likelihood`` is unknown, or ``sigma_obs`` is an unusable string or a
            non-positive scalar.
    """
    if likelihood == "mse":
        return squared_error
    if likelihood != "gaussian_nll":
        raise ValueError(
            f"unknown likelihood {likelihood!r}; expected 'mse' or 'gaussian_nll'."
        )
    if isinstance(sigma_obs, str):
        if sigma_obs != "learned":
            raise ValueError(f"sigma_obs string must be 'learned', got {sigma_obs!r}")
        logvar_obs = logvar
    else:
        sigma_value = float(sigma_obs)
        if sigma_value <= 0.0:
            raise ValueError(f"sigma_obs scalar must be positive, got {sigma_value}")
        logvar_obs = torch.full_like(squared_error, math.log(sigma_value**2))
    return 0.5 * squared_error * torch.exp(-logvar_obs) + 0.5 * logvar_obs


def feature_loss(
    mu: torch.Tensor,
    y_plus: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    sigma_obs: Any,
) -> torch.Tensor:
    r"""The pooled masked feature loss -- the quantity ``compute_loss`` calls ``feat_loss``.

    This is the function the parity test compares against the model's own, so it is deliberately
    a composition of :func:`per_element_loss` and :func:`masked_pooled_mean` with nothing else
    in between.

    Args:
        mu: Forecast mean, $(B, A, H_d, C)$, already sliced to the valid anchors.
        y_plus: The future target $Y^{+}$, same shape.
        logvar: Predictive log-variance, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.
        likelihood: The checkpoint's likelihood.
        sigma_obs: The checkpoint's observation noise setting.

    Returns:
        A scalar tensor.
    """
    squared_error = (mu - y_plus) ** 2
    return masked_pooled_mean(
        per_element_loss(squared_error, logvar, likelihood=likelihood, sigma_obs=sigma_obs),
        mask,
    )


# ---------------------------------------------------------------------------
# Forecast quality
# ---------------------------------------------------------------------------
def _masked_r2(
    mu: torch.Tensor, y_plus: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    r"""Per-sample $R^2$ against the masked per-channel mean of the target.

    $$R^2 = 1 - \frac{\sum m (\mu - y)^2}{\sum m (y - \bar{y}_c)^2}$$

    with $\bar{y}_c$ the mask-weighted mean of channel $c$ *within that sample*. See the module
    docstring for why the mean is per channel.

    Args:
        mu: Forecast mean, $(B, A, H_d, C)$.
        y_plus: Target, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        $(B,)$, ``NaN`` where the sample is fully masked or the target has no variance to
        explain -- the latter because $R^2$ is undefined there, not zero.
    """
    per_sample_count = mask.sum(dim=(1, 2), keepdim=True)
    channel_mean = (y_plus * mask).sum(dim=(1, 2), keepdim=True) / per_sample_count.clamp_min(1.0)

    ss_res = (((mu - y_plus) ** 2) * mask).sum(dim=(1, 2, 3))
    ss_tot = (((y_plus - channel_mean) ** 2) * mask).sum(dim=(1, 2, 3))

    usable = (mask.sum(dim=(1, 2, 3)) > 0) & (ss_tot > 0)
    return torch.where(usable, 1.0 - ss_res / ss_tot.clamp_min(torch.finfo(ss_tot.dtype).tiny),
                       torch.full_like(ss_res, float("nan")))


def forecast_metrics(
    mu: torch.Tensor,
    y_plus: torch.Tensor,
    mask: torch.Tensor,
    n_scattering: int,
) -> Dict[str, torch.Tensor]:
    r"""Per-sample forecast error and $R^2$, whole-feature and per feature block.

    Every value is a **per-sample** mean: each sample divided by its own mask sum. That is a
    different quantity from ``compute_loss``'s pooled mask-weighted mean over the whole batch,
    and the two do not agree unless every sample has the same mask density.

    Args:
        mu: Forecast mean at the valid anchors, $(B, A, H_d, C)$.
        y_plus: The future target, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.
        n_scattering: Width of the scattering block, i.e. ``batch.fhr_st.shape[-1]``. Passed in
            rather than assumed: the model stores only the combined $c_y = 109$ and cannot
            supply the split, and hardcoding $43$ would silently mis-slice the day the feature
            layout changes -- which it already has once.

    Returns:
        Per-sample tensors $(B,)$ keyed ``feat_mse_total`` / ``_scattering`` / ``_phase`` and
        ``feat_r2_total`` / ``_scattering`` / ``_phase``, plus ``n_valid_entries``.
    """
    split = int(n_scattering)
    blocks = {
        "total": (None, None),
        "scattering": (None, split),
        "phase": (split, None),
    }

    metrics: Dict[str, torch.Tensor] = {}
    for name, (start, stop) in blocks.items():
        mu_block = _channel_slice(mu, start, stop)
        y_block = _channel_slice(y_plus, start, stop)
        if mu_block.shape[-1] == 0:
            # A degenerate split -- an all-scattering or all-phase feature vector -- is a real
            # configuration, not an error; report NaN rather than a division by zero.
            nan = torch.full((mu.shape[0],), float("nan"), device=mu.device, dtype=mu.dtype)
            metrics[f"feat_mse_{name}"] = nan
            metrics[f"feat_r2_{name}"] = nan.clone()
            continue
        metrics[f"feat_mse_{name}"] = masked_per_sample_mean((mu_block - y_block) ** 2, mask)
        metrics[f"feat_r2_{name}"] = _masked_r2(mu_block, y_block, mask)

    metrics["n_valid_entries"] = mask.sum(dim=(1, 2, 3)) * float(mu.shape[-1])
    return metrics


def horizon_error_profile(
    mu: torch.Tensor, y_plus: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    r"""Per-sample squared error as a function of horizon step $h$.

    The profile the forecast is *supposed* to have: error rising with $h$, because a step
    further into the future is harder. A flat profile means the model is predicting a constant.

    Args:
        mu: Forecast mean, $(B, A, H_d, C)$.
        y_plus: Target, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        $(B, H_d)$, ``NaN`` where a (sample, horizon) cell has no unmasked entry.
    """
    channels = float(mu.shape[-1])
    squared_error = (mu - y_plus) ** 2
    total = (squared_error * mask).sum(dim=(1, 3))
    count = mask.sum(dim=(1, 3)) * channels
    return torch.where(count > 0, total / count.clamp_min(1.0), torch.full_like(total, float("nan")))


def anchor_error_profile(
    mu: torch.Tensor, y_plus: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    r"""Per-sample squared error as a function of anchor position $t$.

    Reads as a time course: the warm-up anchors are masked to ``NaN``, and a rise or fall across
    the recording localises where a forecast degrades rather than only reporting that it does.

    Args:
        mu: Forecast mean, $(B, A, H_d, C)$.
        y_plus: Target, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        $(B, A)$, ``NaN`` where an anchor has no unmasked entry.
    """
    channels = float(mu.shape[-1])
    squared_error = (mu - y_plus) ** 2
    total = (squared_error * mask).sum(dim=(2, 3))
    count = mask.sum(dim=(2, 3)) * channels
    return torch.where(count > 0, total / count.clamp_min(1.0), torch.full_like(total, float("nan")))


# ---------------------------------------------------------------------------
# Frequency-band and per-channel resolution
# ---------------------------------------------------------------------------
def band_forecast_metrics(
    mu: torch.Tensor,
    y_plus: torch.Tensor,
    mask: torch.Tensor,
    partition_idx: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, Any]]:
    r"""Resolve the forecast error over an arbitrary label-to-channel-index mapping.

    Generalised over the mapping rather than written against the clinical bands: the same
    function serves the ``clinical`` partition, the ``by_kind`` one, and any grouping a later
    question needs, because none of the arithmetic depends on what a label *means*.

    **The channel-count-weighted mean over the labels reproduces the overall MSE**, per sample:

    $$\frac{\sum_g C_g\,\mathrm{mse}_g}{\sum_g C_g}
    = \frac{\sum_g \sum_{c \in g} S_c}{m\,C} = \mathrm{mse}_{\mathrm{total}}$$

    since each label's mean divides by its own channel count $C_g$ against the *same* mask sum
    $m$. That identity is the reason it is worth asserting: it holds exactly when the labels tile
    the channel space, and fails the moment a partition carries a gap or an overlap -- which is
    the one defect a per-band number cannot show on its own face.

    An **empty label is dropped, not reported as zero**. Zero is a legitimate MSE, so an empty
    band reported as $0$ reads as the best-forecast band in the run; and it would additionally
    break the weighted-mean identity above by contributing a $C_g = 0$ term.

    Args:
        mu: Forecast mean at the valid anchors, $(B, A, H_d, C)$.
        y_plus: The future target, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.
        partition_idx: Label to the channel indices it covers, from
            :meth:`~teb_vae.lag_attn.eval.band_partition.BandPartition.partition`.

    Returns:
        Label to a record carrying per-sample ``feat_mse`` and ``feat_r2`` $(B,)$, the
        ``horizon`` $(B, H_d)$ and ``anchor`` $(B, A)$ profiles, and the ``n_channels`` the
        label covers. Labels with no channel are absent from the result.

    Raises:
        IndexError: If a label names a channel outside $[0, C)$. Silently clipping would place
            real error in the wrong band and nothing downstream would notice.
    """
    n_channels = int(mu.shape[-1])
    results: Dict[str, Dict[str, Any]] = {}

    for label, channels in partition_idx.items():
        indices = [int(value) for value in channels]
        if not indices:
            continue
        out_of_range = [value for value in indices if not 0 <= value < n_channels]
        if out_of_range:
            raise IndexError(
                f"partition label {label!r} names channel(s) {out_of_range} outside the "
                f"forecast's [0, {n_channels}) channel axis. The partition describes a "
                f"different feature layout from the one this checkpoint forecasts."
            )
        selector = torch.as_tensor(indices, dtype=torch.long, device=mu.device)
        mu_band = mu.index_select(-1, selector)
        y_band = y_plus.index_select(-1, selector)

        results[str(label)] = {
            "feat_mse": masked_per_sample_mean((mu_band - y_band) ** 2, mask),
            "feat_r2": _masked_r2(mu_band, y_band, mask),
            "horizon": horizon_error_profile(mu_band, y_band, mask),
            "anchor": anchor_error_profile(mu_band, y_band, mask),
            "n_channels": len(indices),
        }
    return results


@dataclass
class ChannelErrorAccumulator:
    r"""Streaming per-channel and per-(channel, horizon) squared error over a whole split.

    Bounded by $O(C \cdot H_d)$ regardless of how many samples pass through it -- $109 \times 15$
    float64 accumulators, a few kilobytes -- so the per-channel pass costs no extra inference and
    no retention. Retaining the per-sample $(N, C, H_d)$ field instead would be roughly $6.5$ KB
    per sample and would force either a cap or a second loader pass, and the whole point of this
    accumulator is that the band pass and the channel pass share one loop.

    The reduction is the **pooled** one -- a single denominator over every sample the accumulator
    saw, matching ``compute_loss`` rather than the per-sample form the CSVs carry. That is what
    makes the identity worth asserting: the mean over channels of :meth:`per_channel_mse`
    reproduces :meth:`total_mse` exactly, which fails if the channel axis is mis-sliced or a
    horizon is double-counted.
    """

    n_channels: int
    horizon: int
    #: $(H_d, C)$ sum of masked squared error. Stored horizon-major because that is the axis
    #: order the tensors arrive in; exposed channel-major, which is how a reader wants it.
    squared_error: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    #: $(H_d,)$ count of unmasked $(sample, anchor)$ cells per horizon step.
    cells: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    n_samples: int = 0

    def __post_init__(self) -> None:
        """Allocate the accumulators in ``float64``.

        Not the tensors' ``float32``: this sums over hundreds of thousands of cells, and a
        ``float32`` running total loses low-order bits to a large accumulated value long before
        the split is exhausted -- which would show up as a per-channel mean that no longer
        reproduces the total.
        """
        if self.squared_error is None:
            self.squared_error = np.zeros((int(self.horizon), int(self.n_channels)), dtype=np.float64)
        if self.cells is None:
            self.cells = np.zeros((int(self.horizon),), dtype=np.float64)

    def update(self, mu: torch.Tensor, y_plus: torch.Tensor, mask: torch.Tensor) -> None:
        r"""Fold one batch in.

        Args:
            mu: Forecast mean at the valid anchors, $(B, A, H_d, C)$.
            y_plus: Target, same shape.
            mask: Feature mask, $(B, A, H_d, 1)$.

        Raises:
            ValueError: If the batch's geometry disagrees with the accumulator's. Continuing
                would broadcast one horizon's error across another's.
        """
        if int(mu.shape[-1]) != int(self.n_channels) or int(mu.shape[2]) != int(self.horizon):
            raise ValueError(
                f"batch is (H_d={int(mu.shape[2])}, C={int(mu.shape[-1])}) but this accumulator "
                f"was built for (H_d={self.horizon}, C={self.n_channels})."
            )
        squared_error = ((mu - y_plus) ** 2) * mask
        # Reduced on the compute device and transferred once per batch as a (H_d, C) block,
        # rather than moving the (B, A, H_d, C) tensor to the host and reducing there.
        self.squared_error += squared_error.sum(dim=(0, 1)).double().cpu().numpy()
        self.cells += mask.squeeze(-1).sum(dim=(0, 1)).double().cpu().numpy()
        self.n_samples += int(mu.shape[0])

    @property
    def _total_cells(self) -> float:
        """Total unmasked cells across every horizon step."""
        return float(self.cells.sum())

    def per_channel_mse(self) -> np.ndarray:
        r"""Pooled MSE per channel, $(C,)$, ``NaN`` when nothing was ever unmasked."""
        total = self._total_cells
        if total <= 0.0:
            return np.full((self.n_channels,), np.nan)
        return self.squared_error.sum(axis=0) / total

    def per_channel_horizon_mse(self) -> np.ndarray:
        r"""Pooled MSE per (channel, horizon), $(C, H_d)$, ``NaN`` in an unpopulated column."""
        with np.errstate(invalid="ignore", divide="ignore"):
            denominator = np.where(self.cells > 0.0, self.cells, np.nan)[:, None]
            return (self.squared_error / denominator).T

    def total_mse(self) -> float:
        r"""The pooled MSE over every channel -- the quantity the channel mean must reproduce."""
        total = self._total_cells
        if total <= 0.0 or self.n_channels <= 0:
            return float("nan")
        return float(self.squared_error.sum() / (total * float(self.n_channels)))


# ---------------------------------------------------------------------------
# Uplift and residual activity
# ---------------------------------------------------------------------------
def uplift_metrics(
    mu_full: torch.Tensor,
    mu_base: torch.Tensor,
    y_plus: torch.Tensor,
    logvar_full: torch.Tensor,
    logvar_base: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    sigma_obs: Any,
) -> Dict[str, torch.Tensor]:
    r"""Per-sample full-versus-baseline loss and the uplift between them.

    $$u_{\mathrm{abs}} = L_{\mathrm{base}} - L_{\mathrm{full}}, \qquad
    u_{\mathrm{rel}} = \frac{u_{\mathrm{abs}}}{|L_{\mathrm{base}}|}$$

    Positive uplift means the source pathway helped. The relative form divides by the
    *magnitude* of the baseline loss, because under ``gaussian_nll`` a well-calibrated baseline
    loss is routinely negative and dividing by the signed value would flip the sign of the
    uplift on exactly the healthy runs.

    Under ``sigma_obs='learned'`` the two losses read *different variance heads*, so they differ
    even when the mean correction is identically zero. A near-zero uplift is therefore evidence
    about the mean pathway only under ``'mse'``; :func:`residual_usage` is the readout that
    isolates it.

    Args:
        mu_full: Full forecast mean, $(B, A, H_d, C)$.
        mu_base: Baseline forecast mean, same shape.
        y_plus: Target, same shape.
        logvar_full: Full predictive log-variance, same shape.
        logvar_base: Baseline predictive log-variance, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.
        likelihood: The checkpoint's likelihood.
        sigma_obs: The checkpoint's observation noise setting.

    Returns:
        Per-sample tensors $(B,)$: ``l_full``, ``l_base``, ``uplift_abs``, ``uplift_rel``.
    """
    l_full = masked_per_sample_mean(
        per_element_loss((mu_full - y_plus) ** 2, logvar_full, likelihood=likelihood, sigma_obs=sigma_obs),
        mask,
    )
    l_base = masked_per_sample_mean(
        per_element_loss((mu_base - y_plus) ** 2, logvar_base, likelihood=likelihood, sigma_obs=sigma_obs),
        mask,
    )
    uplift_abs = l_base - l_full
    denom = l_base.abs()
    uplift_rel = torch.where(
        denom > 0, uplift_abs / denom.clamp_min(torch.finfo(denom.dtype).tiny),
        torch.full_like(uplift_abs, float("nan")),
    )
    return {
        "l_full": l_full,
        "l_base": l_base,
        "uplift_abs": uplift_abs,
        "uplift_rel": uplift_rel,
    }


def residual_usage(
    delta_mu_src: torch.Tensor, mu_full: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Per-sample activity of the source-driven mean correction.

    $$\mathrm{rms}(\delta) = \sqrt{\frac{\sum m \sum_c \delta_c^2}{\sum m}}, \qquad
    \mathrm{ratio} = \frac{\mathrm{rms}(\delta)}{\mathrm{rms}(\mu_{\mathrm{full}})}$$

    The channel sum inside the root matches ``task.py``'s ``delta_mu_rms``, so the number is
    directly comparable with what training logged. ``residual_ratio`` is the scale-free form and
    is the primary collapse signal: an absolute RMS is only interpretable next to the magnitude
    of the forecast it corrects.

    Args:
        delta_mu_src: Source-driven mean correction at the valid anchors, $(B, A, H_d, C)$.
        mu_full: Full forecast mean, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        Per-sample tensors $(B,)$: ``residual_rms``, ``forecast_rms``, ``residual_ratio``.
    """
    cell_mask = mask.squeeze(-1)
    count = cell_mask.sum(dim=(1, 2))

    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        energy = (tensor**2).sum(dim=-1)
        total = (energy * cell_mask).sum(dim=(1, 2))
        return torch.where(
            count > 0,
            torch.sqrt(total / count.clamp_min(1.0)),
            torch.full_like(total, float("nan")),
        )

    residual_rms = _rms(delta_mu_src)
    forecast_rms = _rms(mu_full)
    ratio = torch.where(
        forecast_rms > 0,
        residual_rms / forecast_rms.clamp_min(torch.finfo(forecast_rms.dtype).tiny),
        torch.full_like(residual_rms, float("nan")),
    )
    return {
        "residual_rms": residual_rms,
        "forecast_rms": forecast_rms,
        "residual_ratio": ratio,
    }


def residual_per_anchor(
    delta_mu_src: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    r"""Per-anchor RMS of the source-driven correction, $(B, A)$.

    Localises a dead pathway in time: a residual that is active early and flat later is a
    different finding from one that never activates at all.

    Args:
        delta_mu_src: Source-driven mean correction, $(B, A, H_d, C)$.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        $(B, A)$, ``NaN`` at anchors with no unmasked entry.
    """
    cell_mask = mask.squeeze(-1)
    energy = (delta_mu_src**2).sum(dim=-1)
    total = (energy * cell_mask).sum(dim=2)
    count = cell_mask.sum(dim=2)
    return torch.where(
        count > 0,
        torch.sqrt(total / count.clamp_min(1.0)),
        torch.full_like(total, float("nan")),
    )


# ---------------------------------------------------------------------------
# Latent
# ---------------------------------------------------------------------------
def kld_per_dim(outputs: Dict[str, torch.Tensor], model: Any) -> torch.Tensor:
    r"""Per-step per-dimension KL, $(B, T, d_z)$, from the model's own closed form.

    Delegates to ``model.kld_tensor`` rather than restating the formula. A local copy is how the
    reported KL and the trained KL drift apart, and the drift is invisible: both are plausible
    numbers of the same magnitude.

    Args:
        outputs: A forward dict.
        model: The rebuilt ``SeqVaeLagAttn``.

    Returns:
        The unmasked KL, $(B, T, d_z)$.
    """
    return model.kld_tensor(
        mu_prior=outputs["mu_prior"],
        logvar_prior=outputs["logvar_prior"],
        mu_post=outputs["mu_post"],
        logvar_post=outputs["logvar_post"],
    )


def kld_pooled(
    kld_btd: torch.Tensor, mask_bt: torch.Tensor, *, free_bits: float = 0.0
) -> torch.Tensor:
    r"""Reduce the per-dimension KL the way ``_kld_loss`` does -- the ``kld_raw`` quantity.

    ``free_bits`` clamps each per-dimension per-step KL upward **before** masking, matching the
    model. That ordering is what makes ``kld_train >= kld_raw`` hold; clamping after would floor
    the aggregate instead of each term and give a smaller number.

    Args:
        kld_btd: Per-step per-dimension KL, $(B, T, d_z)$.
        mask_bt: The KL mask, $(B, T)$, from :func:`~teb_vae.lag_attn.eval.masks.kld_mask`.
        free_bits: Per-dimension per-step floor. $0$ is a no-op, the closed-form KL already
            being non-negative.

    Returns:
        A scalar tensor. Zero when the support is empty, matching the model.
    """
    if free_bits > 0.0:
        kld_btd = kld_btd.clamp(min=float(free_bits))
    mask_btd = mask_bt.unsqueeze(-1)
    denom = mask_btd.sum() * float(kld_btd.shape[-1])
    if float(denom) <= 0.0:
        return torch.zeros((), device=kld_btd.device, dtype=kld_btd.dtype)
    return (kld_btd * mask_btd).sum() / denom


def kld_aggregates(kld_btd: torch.Tensor, mask_bt: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""Per-sample KL aggregates over the support: mean, sum, and per-dimension $L^2$.

    The three answer different questions. The mean is comparable across recordings of different
    lengths; the sum is the total information the latent carried; the $L^2$ over the
    per-dimension means is large when the KL is concentrated in a few dimensions and small when
    it is spread, which is the shape ``kld_active_frac`` reports as a count.

    Args:
        kld_btd: Per-step per-dimension KL, $(B, T, d_z)$.
        mask_bt: The KL mask, $(B, T)$.

    Returns:
        Per-sample tensors $(B,)$: ``kld_mean``, ``kld_sum``, ``kld_dim_l2``; and
        ``kld_per_dim_mean`` $(B, d_z)$. All four are ``NaN`` where the sample's support is
        empty, per this module's no-data contract.
    """
    mask_btd = mask_bt.unsqueeze(-1)
    d_z = float(kld_btd.shape[-1])
    steps = mask_bt.sum(dim=1)

    weighted = (kld_btd * mask_btd).sum(dim=1)
    per_dim_mean = torch.where(
        steps.unsqueeze(-1) > 0,
        weighted / steps.unsqueeze(-1).clamp_min(1.0),
        torch.full_like(weighted, float("nan")),
    )
    kld_sum = weighted.sum(dim=1)
    kld_mean = torch.where(
        steps > 0, kld_sum / (steps * d_z).clamp_min(1.0), torch.full_like(kld_sum, float("nan"))
    )
    return {
        "kld_mean": kld_mean,
        "kld_sum": torch.where(steps > 0, kld_sum, torch.full_like(kld_sum, float("nan"))),
        # The ``nan_to_num`` is a shape guard, not a repair: ``per_dim_mean``'s ``where`` is
        # per *sample*, so a row is either wholly finite or wholly ``NaN`` and there is no
        # partial-NaN case to fix. Left alone it would turn an all-NaN row into an all-zero one
        # whose norm is exactly $0$ -- the module's no-data value silently replaced by a
        # legitimate one, and the smallest one at that. ``latent.py`` filters with
        # ``np.isfinite``, which drops the NaN but keeps that zero, so an empty support would
        # drag ``mean_kld_dim_l2`` down and put a spike at $0$ in the by-subgroup violins that
        # reads as latent collapse. The outer ``where`` restores the contract.
        "kld_dim_l2": torch.where(
            steps > 0,
            torch.linalg.vector_norm(torch.nan_to_num(per_dim_mean), dim=1),
            torch.full_like(kld_sum, float("nan")),
        ),
        "kld_per_dim_mean": per_dim_mean,
    }


def posterior_drift(outputs: Dict[str, torch.Tensor], mask_bt: torch.Tensor) -> torch.Tensor:
    r"""Per-sample RMS of the posterior-prior mean gap over the KL support.

    $$\mathrm{rms} = \sqrt{\frac{\sum_t m_t \sum_d (\mu^q_{td} - \mu^p_{td})^2}{\sum_t m_t}}$$

    The mean-space companion to the KL: the same masking rules as ``task.py``'s
    ``mu_post_prior_gap_rms``, so the two are directly comparable run to run.

    Args:
        outputs: A forward dict.
        mask_bt: The KL mask, $(B, T)$.

    Returns:
        $(B,)$, ``NaN`` where the support is empty.
    """
    gap = ((outputs["mu_post"] - outputs["mu_prior"]) ** 2).sum(dim=-1)
    total = (gap * mask_bt).sum(dim=1)
    count = mask_bt.sum(dim=1)
    return torch.where(
        count > 0,
        torch.sqrt(total / count.clamp_min(1.0)),
        torch.full_like(total, float("nan")),
    )


def latent_health(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Pass through the model's own collapse diagnostics as plain floats.

    A passthrough rather than a recomputation. These three are computed inside the forward under
    the model's own support and thresholds, and recomputing them here would introduce a second
    definition of "active" that could disagree with the one training logged.

    Args:
        outputs: A forward dict.

    Returns:
        ``kld_active_frac``, ``mu_prior_sat_frac`` and ``delta_mu_sat_frac``. A key the forward
        did not carry is reported as ``NaN``.
    """
    names = ("kld_active_frac", "mu_prior_sat_frac", "delta_mu_sat_frac")
    return {
        name: float(outputs[name]) if name in outputs else float("nan") for name in names
    }


def masked_latent_diagnostics(
    outputs: Dict[str, torch.Tensor], model: Any, mask_bt: torch.Tensor
) -> Dict[str, float]:
    r"""Recompute the three headline latent diagnostics under this pipeline's masking rules.

    The model computes all three differently from everything else it reports, and the
    differences run in opposite directions:

    * ``kld_active_frac`` **is** restricted to the KL support, but ignores the per-step validity
      ``weight`` entirely -- so a recording that is half gaps contributes its gap steps to the
      per-dimension mean that decides whether a dimension is active;
    * the two saturation fractions apply **no masking at all** -- a flat ``.mean()`` over every
      $(B, T, d_z)$ element, warm-up prefix and untrained tail included.

    Neither is a bug in the model: they are cheap in-forward diagnostics logged every step, and
    the mask would cost a broadcast on the hot path. But they contradict this pipeline's rule
    that every reported metric is masked exactly as the loss masks, so both readings are emitted
    -- the model's own under its original name, and this one beside it. A large gap between them
    is itself informative: it means the diagnostic is dominated by steps the loss never scored.

    Args:
        outputs: A forward dict.
        model: The rebuilt ``SeqVaeLagAttn``, for ``kld_tensor`` and the two scale bounds.
        mask_bt: The KL mask, $(B, T)$, from
            :func:`~teb_vae.lag_attn.eval.masks.kld_mask` -- support intersected with ``weight``.

    Returns:
        ``kld_active_frac``, ``mu_prior_sat_frac`` and ``delta_mu_sat_frac``, each masked. All
        ``NaN`` when the mask is empty, since a fraction over no elements is undefined.
    """
    total = float(mask_bt.sum())
    if total <= 0.0:
        return {
            "kld_active_frac": float("nan"),
            "mu_prior_sat_frac": float("nan"),
            "delta_mu_sat_frac": float("nan"),
        }

    mask_btd = mask_bt.unsqueeze(-1)
    kld_btd = kld_per_dim(outputs, model)
    # Mean per dimension first, then threshold, then mean over dimensions -- the model's own
    # order. Thresholding per step first would count a dimension active on the strength of a
    # single spike rather than of a sustained coupling.
    per_dim_mean = (kld_btd * mask_btd).sum(dim=(0, 1)) / total
    kld_active_frac = float((per_dim_mean > KLD_ACTIVE_EPS).to(kld_btd.dtype).mean())

    def _saturated_fraction(values: torch.Tensor, bound: float) -> float:
        indicator = (values.abs() >= 0.99 * float(bound)).to(values.dtype)
        return float((indicator * mask_btd).sum() / (total * float(values.shape[-1])))

    return {
        "kld_active_frac": kld_active_frac,
        "mu_prior_sat_frac": _saturated_fraction(outputs["mu_prior"], model.mu_scale),
        "delta_mu_sat_frac": _saturated_fraction(
            outputs["mu_post"] - outputs["mu_prior"], model.delta_mu_scale
        ),
    }


# ---------------------------------------------------------------------------
# Predictive calibration
# ---------------------------------------------------------------------------
def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    r"""$\Phi(z)$, the standard normal CDF.

    Computed with ``torch.erf`` rather than ``scipy.stats.norm.cdf``: the inputs are
    $(B, A, H_d, c_y)$ tensors that may be on a GPU, and routing them through SciPy would force a
    host transfer per batch for a function ``torch`` already provides exactly.

    Args:
        z: Standardised residuals, any shape.

    Returns:
        $\Phi(z)$, same shape.
    """
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def nominal_central_coverage(k_sigma: float) -> float:
    r"""The exact two-sided coverage of a $\pm k\sigma$ band, $\mathrm{erf}(k/\sqrt{2})$.

    Computed rather than tabulated, because the tabulated value is exactly where this goes
    wrong: a $\pm 2\sigma$ band covers $0.9545$, not $0.95$ -- $0.95$ is $\pm 1.96\sigma$ -- and
    scoring a $2\sigma$ band against $0.95$ reports a well-calibrated model as over-confident by
    half a percentage point on every horizon.

    Args:
        k_sigma: Band half-width in standard deviations.

    Returns:
        The nominal coverage in $[0, 1]$.
    """
    return math.erf(float(k_sigma) / math.sqrt(2.0))


def pit_values(
    mu: torch.Tensor, y_plus: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    r"""Probability integral transform, $\Phi\!\left((y - \mu)/\sigma\right)$.

    Uniform on $[0, 1]$ exactly when the predictive Gaussian is calibrated. The *shape* of the
    departure is the diagnostic: a $\cup$ shape means the variance is too small, a $\cap$ shape
    too large, and a tilt means the mean is biased.

    Args:
        mu: Forecast mean, any shape.
        y_plus: Target, same shape.
        logvar: Predictive log-variance, same shape.

    Returns:
        PIT values in $[0, 1]$, same shape.
    """
    sigma = torch.exp(0.5 * logvar)
    return normal_cdf((y_plus - mu) / sigma.clamp_min(torch.finfo(sigma.dtype).tiny))


def gaussian_log_density(
    mu: torch.Tensor, y_plus: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    r"""Per-element Gaussian negative log density **including** the $\tfrac{1}{2}\log 2\pi$ term.

    $$-\log p(y) = \tfrac{1}{2}\log 2\pi + \tfrac{1}{2}\log\sigma^2
    + \tfrac{1}{2}\frac{(y-\mu)^2}{\sigma^2}$$

    The constant is dropped everywhere else in this pipeline, matching training -- which makes
    those numbers comparable with a training loss but *not* a log density. Calibration is the one
    place the absolute value has to mean something: an NLL compared against a homoscedastic
    reference is a likelihood-ratio statement, and it is only a valid one if both sides are
    genuine densities.

    Args:
        mu: Forecast mean, any shape.
        y_plus: Target, same shape.
        logvar: Predictive log-variance, same shape.

    Returns:
        The per-element negative log density in nats, same shape.
    """
    return 0.5 * (
        math.log(2.0 * math.pi) + logvar + (y_plus - mu) ** 2 * torch.exp(-logvar)
    )


def crps_gaussian(
    mu: torch.Tensor, y_plus: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    r"""Per-element continuous ranked probability score, in closed form for a Gaussian.

    $$\mathrm{CRPS} = \sigma\left[z\left(2\Phi(z) - 1\right) + 2\varphi(z)
    - \tfrac{1}{\sqrt{\pi}}\right], \qquad z = \frac{y - \mu}{\sigma}$$

    Reported beside the NLL because the two fail differently and CRPS is the more robust of the
    pair. NLL is unbounded: one over-confident outlier -- a tiny $\sigma$ against a large
    residual -- can dominate the average over an entire split. CRPS is in the units of $y$ and
    grows linearly in the residual, so it ranks two models the way a reader expects while still
    rewarding a sharp distribution.

    Args:
        mu: Forecast mean, any shape.
        y_plus: Target, same shape.
        logvar: Predictive log-variance, same shape.

    Returns:
        The per-element CRPS, same shape. Non-negative, $0$ for a point mass at the truth.
    """
    sigma = torch.exp(0.5 * logvar).clamp_min(torch.finfo(logvar.dtype).tiny)
    z = (y_plus - mu) / sigma
    density = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi))


def coverage_indicator(
    mu: torch.Tensor, y_plus: torch.Tensor, logvar: torch.Tensor, k_sigma: float
) -> torch.Tensor:
    r"""$\mathbb{1}\left[|y - \mu| \le k\sigma\right]$, per element.

    Args:
        mu: Forecast mean, any shape.
        y_plus: Target, same shape.
        logvar: Predictive log-variance, same shape.
        k_sigma: Band half-width in standard deviations.

    Returns:
        A $0/1$ indicator, same shape and dtype as ``mu``.
    """
    sigma = torch.exp(0.5 * logvar)
    return ((y_plus - mu).abs() <= float(k_sigma) * sigma).to(mu.dtype)


def homoscedastic_logvar(
    mu: torch.Tensor, y_plus: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    r"""Fit the single variance that best explains the residuals, and return its log.

    $$\hat{\sigma}^2 = \frac{\sum m\,(y - \mu)^2}{\sum m \cdot C}$$

    The maximum-likelihood constant variance for these residuals, and therefore the *strongest*
    homoscedastic reference available -- not a straw man. A learned variance head that fails to
    beat it has learned nothing about where the forecast is uncertain, only about how uncertain
    it is on average, which the residuals already say.

    Args:
        mu: Forecast mean, $(B, A, H_d, C)$.
        y_plus: Target, same shape.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        A scalar tensor, $\log\hat{\sigma}^2$.
    """
    variance = masked_pooled_mean((y_plus - mu) ** 2, mask)
    return torch.log(variance.clamp_min(torch.finfo(variance.dtype).tiny))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def attention_diagnostics(
    attn_weights: torch.Tensor, support: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Summarise one batch's lag attention over a given anchor support.

    Every quantity is computed **only** over ``support``. That is not a presentation choice: the
    warm-up anchors attend over a window that is mostly not yet causally available, and a dead
    anchor carries an all-zero row that is not a distribution at all. Averaging either in shifts
    the reported lag profile toward the short lags and lowers every entropy.

    $$\bar{\alpha}^{(m)}_\ell = \frac{\sum_t s_t\,\alpha^{(m)}_{t,\ell}}{\sum_t s_t},
    \qquad H^{(m)}_t = -\sum_\ell \alpha^{(m)}_{t,\ell}\log\alpha^{(m)}_{t,\ell}$$

    Entropy is in nats and is the readout that separates "this head has found a lag" from "this
    head is averaging over the whole window": ``argmax_lag`` names a peak whether or not one
    exists, and on a near-uniform row it names noise.

    **Its ceiling is not $\log L$.** ``build_lag_mask`` gives anchor $t$ only $\min(t+1, L)$
    causally valid lags, so an anchor inside the first $L - 1$ steps cannot reach $\log L$
    however flat its row is. ``attainable_entropy`` is that true bound, averaged over exactly the
    support the entropy is averaged over:

    $$H^{\max} = \frac{\sum_t s_t \log\min(t + 1, L)}{\sum_t s_t}$$

    At the production geometry ($L = 91$, warm-up $30$, $240$ supported anchors) $60$ of those
    anchors top out between $\log 31 = 3.43$ and $\log 90 = 4.50$, so attention that is exactly
    uniform over every available lag -- the degenerate case, with no lag structure at all --
    reports $4.398$ against $\log 91 = 4.511$. Read against $\log L$ that $2.5\%$ shortfall looks
    like mild lag concentration, and a uniformity check with a $1\%$ margin can never fire.
    Divide by ``attainable_entropy``, not by $\log L$.

    ``head_diversity`` is the mean pairwise total-variation distance between the heads'
    support-mean lag distributions, in $[0, 1]$. It answers whether head structure bought
    anything -- four heads that all settled on the same lag are one head with four times the
    parameters, and the per-head KL decomposition attributes across them regardless.

    Args:
        attn_weights: The forward's ``attn_weights``, $(B, T, M, L)$, in lag order with index
            $0$ the current step. Already in lag order as returned -- do not flip it.
        support: Anchors to include, $(B, T)$ or $(T,)$, bool or float. Intersect
            :func:`~teb_vae.lag_attn.eval.masks.live_anchor_mask` into this before calling, or
            dead anchors are averaged in as zero rows.

    Returns:
        ``alpha_bar`` $(B, M, L)$, ``mass_by_lag`` $(B, L)$, ``argmax_lag`` $(B,)$,
        ``entropy`` $(B, T, M)$ with ``NaN`` outside the support, ``entropy_mean`` $(B, M)$,
        ``attainable_entropy`` $(B,)$, ``head_diversity`` $(B,)$ and ``n_support_anchors``
        $(B,)$. Per-sample values are ``NaN`` where the sample has no supported anchor;
        ``argmax_lag`` is $-1$ there, being an integer index with no NaN to fall back on.
    """
    batch, seq_len, n_heads, num_lags = attn_weights.shape
    if support.dim() == 1:
        support = support.unsqueeze(0).expand(batch, seq_len)
    support = support.to(device=attn_weights.device, dtype=attn_weights.dtype)

    weights = support[:, :, None, None]
    counts = support.sum(dim=1)
    usable = counts > 0

    alpha_bar = (attn_weights * weights).sum(dim=1) / counts[:, None, None].clamp_min(1.0)
    alpha_bar = torch.where(
        usable[:, None, None], alpha_bar, torch.full_like(alpha_bar, float("nan"))
    )
    mass_by_lag = alpha_bar.mean(dim=1)

    # -1 rather than 0 for an empty support: 0 is a real lag, and the most commonly reported one,
    # so it cannot double as "no answer".
    argmax_lag = torch.where(
        usable,
        torch.nan_to_num(mass_by_lag, nan=-1.0).argmax(dim=-1),
        torch.full_like(counts, -1, dtype=torch.long),
    )

    # xlogy rather than a masked log: it defines 0*log(0) = 0, which is the convention Shannon
    # entropy needs and which entmax15 makes routine -- it produces exact zeros, so a plain
    # log would give -inf on most of the window rather than on none of it.
    entropy = -torch.xlogy(attn_weights, attn_weights).sum(dim=-1)
    entropy = torch.where(
        support[:, :, None] > 0, entropy, torch.full_like(entropy, float("nan"))
    )
    entropy_mean = torch.where(
        usable[:, None],
        (torch.nan_to_num(entropy) * support[:, :, None]).sum(dim=1)
        / counts[:, None].clamp_min(1.0),
        torch.full((batch, n_heads), float("nan"), device=entropy.device, dtype=entropy.dtype),
    )

    # Accumulated with the *same* ``support`` weights and the *same* ``counts`` denominator as
    # ``entropy_mean`` above. A ceiling averaged over a different anchor set than the entropy is
    # the same bug one level down: the ratio would then compare two different windows and no
    # value of it would mean anything in particular. Head-independent, because lag availability
    # is a property of the anchor, not of the head -- so it needs no head axis to be compared
    # against ``entropy_mean`` collapsed over heads.
    #
    # lean-limit: counts causal validity only, matching the one caller, which forwards without a
    # band mask; replace $\min(t+1, L)$ with the band's own per-anchor kept-lag count when a
    # caller starts passing ``lag_band_mask`` through to this function.
    per_anchor_ceiling = torch.log(
        torch.arange(
            1, seq_len + 1, device=attn_weights.device, dtype=attn_weights.dtype
        ).clamp(max=float(num_lags))
    )
    attainable_entropy = torch.where(
        usable,
        (support * per_anchor_ceiling[None, :]).sum(dim=1) / counts.clamp_min(1.0),
        torch.full_like(counts, float("nan")),
    )

    return {
        "alpha_bar": alpha_bar,
        "mass_by_lag": mass_by_lag,
        "argmax_lag": argmax_lag,
        "entropy": entropy,
        "entropy_mean": entropy_mean,
        "attainable_entropy": attainable_entropy,
        "head_diversity": _head_diversity(alpha_bar),
        "n_support_anchors": counts,
    }


def _head_diversity(alpha_bar: torch.Tensor) -> torch.Tensor:
    r"""Mean pairwise total-variation distance between heads' lag distributions, $(B,)$.

    $$d = \binom{M}{2}^{-1}\sum_{m < m'} \tfrac{1}{2}\sum_\ell
    \left|\bar{\alpha}^{(m)}_\ell - \bar{\alpha}^{(m')}_\ell\right|$$

    $0$ when every head attends identically, $1$ when no two heads share any lag. Total variation
    rather than a KL: it is symmetric, bounded, and finite on the disjoint supports ``entmax15``
    routinely produces, where a KL would be infinite.

    Args:
        alpha_bar: Support-mean per-head lag distributions, $(B, M, L)$.

    Returns:
        $(B,)$, ``NaN`` for a single-head model, where the quantity is undefined rather than $0$.
    """
    n_heads = int(alpha_bar.shape[1])
    if n_heads < 2:
        return torch.full(
            (int(alpha_bar.shape[0]),), float("nan"),
            device=alpha_bar.device, dtype=alpha_bar.dtype,
        )
    distances = 0.5 * (alpha_bar[:, :, None, :] - alpha_bar[:, None, :, :]).abs().sum(dim=-1)
    upper = torch.triu(torch.ones_like(distances), diagonal=1)
    return (distances * upper).sum(dim=(1, 2)) / float(n_heads * (n_heads - 1) // 2)


# ---------------------------------------------------------------------------
# Lag conversion
# ---------------------------------------------------------------------------
def lag_to_seconds(lag: Any, *, step_seconds: float = STEP_SECONDS) -> Any:
    r"""Convert a model lag index to seconds on the stored timeline.

    $$\mathrm{seconds}(\ell) = s\,\ell$$

    **The stored timeline is canonical.** The dataset builder shifts the UP channel when it writes
    the shards; that shift is part of how the stored signals are, and no downstream quantity adds
    it back or subtracts it. An earlier revision carried an ``up_shift_secs`` term here that undid
    the builder's shift; it was removed on purpose and must not return under another name.

    Args:
        lag: A lag index, or an array or tensor of them.
        step_seconds: Duration of one decimated step, $s$.

    Returns:
        The lag in seconds, in the same shape as ``lag``.
    """
    return float(step_seconds) * lag


def lag_seconds_physical(lags: Any, *, step_seconds: float = STEP_SECONDS) -> "np.ndarray":
    r"""The ``lag_seconds_physical`` column: lag indices as seconds, as a float array.

    The one name under which the converted lag appears in every CSV and every figure axis in the
    pipeline, so a reader never has to work out which of two lag columns is which. It accepts a
    tensor, an array, a list or a scalar and always returns ``float64``, because its consumers
    are a ``DataFrame`` column and a matplotlib axis and neither wants a torch tensor.

    The arithmetic is :func:`lag_to_seconds`'s, unchanged: $s\ell$ on the canonical stored
    timeline, with no dataset-shift term.

    Args:
        lags: Lag indices, any shape.
        step_seconds: Duration of one decimated step, $s$.

    Returns:
        The lags in seconds, ``float64``, same shape as ``lags``.
    """
    indices = np.asarray(
        lags.detach().cpu().numpy() if isinstance(lags, torch.Tensor) else lags,
        dtype=np.float64,
    )
    return lag_to_seconds(indices, step_seconds=step_seconds)
