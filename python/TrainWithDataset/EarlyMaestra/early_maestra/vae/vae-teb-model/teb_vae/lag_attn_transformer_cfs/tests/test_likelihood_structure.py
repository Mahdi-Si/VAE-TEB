r"""The forecast density's two structural terms: the per-channel scored horizon and AR(1) residual.

Both change *which* density the block is scored under rather than how it is weighted, so the
claims pinned here are claims about a log-density:

* **The score function.** ``cell_mask`` removes cells; ``ar_coef`` scores every cell on its
  innovation $e_\tau = r_\tau - \phi_c r_{\tau-1}$ with $r_{-1} = 0$. Both reduce bitwise to the
  factorised score at their neutral values, and the innovation sum is the exact joint negative
  log-density of a Gaussian AR(1) residual -- the map $r \mapsto e$ is unit lower-triangular, so
  its Jacobian is $1$.
* **The model.** Off by default and bitwise the historical model at initialisation
  ($\phi_c = \tanh(0) = 0$); on, both branches are scored under one $\phi$, so ``pred_gap`` and
  its in-training splits stay differences of one family's log-densities, and the coefficient is
  reachable by the gradient.
* **The resolver and its plumbing.** $H_c$ is read off the shard's slow-leg frequencies and the
  two configuration keys, and the driver hands it to the constructor.
"""
from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.scored_horizon import resolve_target_scored_horizon
from teb_vae.lag_attn_cfs.tests.conftest import INT_C_Y, INT_CAUSAL_SHARD
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor, raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer

from .conftest import (
    CAUSAL_C_Y,
    CAUSAL_ST_WIDTH,
    TINY_HORIZON,
    TINY_STRIDE,
    absolutize_dataset_paths,
    build,
    causal_config,
    make_streams,
    tiny_warmup_kwargs,
)

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: The metrics every "the loss did not move" assertion compares.
_LOSS_METRICS = ("nll_full_block", "nll_base_block", "total_loss", "pred_gap")


def _scored_horizon(horizon: int = TINY_HORIZON) -> list:
    r"""A declared-width $H_c$ vector: every ``fhr_st`` channel at $H$, the phase block cycling
    through $[1, H - 1]$, so the kept phase channels carry masks of several lengths."""
    return [
        horizon if index < CAUSAL_ST_WIDTH else 1 + index % (horizon - 1)
        for index in range(CAUSAL_C_Y)
    ]


def _kwargs(**overrides):
    """The guarded tiny set at the tiling stride, with the named leaves moved."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, **overrides)


def _run(model, kwargs, *, grad: bool = False):
    """One seeded forward and the concatenated target stream and all-valid weight."""
    y_st, y_ph, u_stream = make_streams(kwargs)
    torch.manual_seed(0)
    with torch.set_grad_enabled(grad):
        out = model(y_st, y_ph, u_stream, anchor_phase=0)
    features = torch.cat([y_st, y_ph], dim=-1)
    weight = torch.ones(y_st.shape[0], model.geometry.t)
    return out, features, weight


def _set_ar_logit(model) -> None:
    r"""Move every $a_c$ off zero, to distinct values of both signs."""
    with torch.no_grad():
        model.target_ar_logit.copy_(torch.linspace(-1.2, 1.5, model.target_ar_logit.numel()))


# =================================================================================================
# The score function
# =================================================================================================
def test_neutral_ar_coef_and_cell_mask_are_bitwise_the_factorised_score() -> None:
    r"""$\phi = 0$ and an all-ones mask each reproduce the ``None`` score bitwise, under both
    likelihoods: the neutral values are the historical model, not an approximation of it."""
    generator = torch.Generator().manual_seed(0)
    mu, target, logvar = (torch.randn(2, 3, 5, 4, generator=generator) for _ in range(3))
    for likelihood in ("mse", "gaussian_nll"):
        base = raw_sample_score(mu, target, likelihood=likelihood, logvar=logvar)
        zero_ar = raw_sample_score(
            mu, target, likelihood=likelihood, logvar=logvar, ar_coef=torch.zeros(4)
        )
        ones_mask = raw_sample_score(
            mu, target, likelihood=likelihood, logvar=logvar, cell_mask=torch.ones(5, 4)
        )
        assert torch.equal(base, zero_ar), likelihood
        assert torch.equal(base, ones_mask), likelihood


def test_the_innovation_matches_a_hand_computation() -> None:
    r"""$e_{\tau,c} = r_{\tau,c} - \phi_c r_{\tau-1,c}$ along the horizon axis, per channel, with
    $\tau = 0$ scored on $r_0$ unchanged; under ``'mse'`` the score is $e^2$."""
    generator = torch.Generator().manual_seed(1)
    mu = torch.randn(2, 3, 5, 4, generator=generator)
    target = torch.randn(2, 3, 5, 4, generator=generator)
    phi = torch.tensor([0.0, 0.5, -0.7, 0.95])

    score = raw_sample_score(mu, target, likelihood="mse", ar_coef=phi)

    residual = target - mu
    expected = torch.empty_like(residual)
    expected[..., 0, :] = residual[..., 0, :]
    for tau in range(1, residual.shape[-2]):
        for c in range(residual.shape[-1]):
            expected[..., tau, c] = residual[..., tau, c] - phi[c] * residual[..., tau - 1, c]
    assert torch.equal(score[..., 0, :], residual[..., 0, :] ** 2)
    torch.testing.assert_close(score, expected**2, rtol=1e-6, atol=1e-7)


def test_the_innovation_sum_is_the_exact_joint_ar1_log_density() -> None:
    r"""The "joint log-density, Jacobian $1$" claim, against an independent construction.

    For $r_\tau = \phi r_{\tau-1} + e_\tau$, $r_{-1} = 0$, $e_\tau \sim \mathcal N(0, s_\tau^2)$,
    the horizon is $r = L e$ with $L_{ij} = \phi^{i-j}$ for $i \ge j$, so
    $r \sim \mathcal N(0, \Sigma)$, $\Sigma = L\, \mathrm{diag}(s^2) L^\top$. The summed Gaussian
    innovation score with $\log s_\tau^2$ as the log-variance must equal
    $-\log \mathcal N(r; 0, \Sigma)$, per channel, in float64.
    """
    dtype = torch.float64
    horizon = 7
    phis = torch.tensor([0.0, 0.6, -0.8], dtype=dtype)
    generator = torch.Generator().manual_seed(2)
    variances = 0.2 + torch.rand(horizon, phis.numel(), generator=generator, dtype=dtype)
    residual = torch.randn(1, 1, horizon, phis.numel(), generator=generator, dtype=dtype)

    score = raw_sample_score(
        torch.zeros_like(residual),
        residual,
        likelihood="gaussian_nll",
        logvar=variances.log()[None, None],
        ar_coef=phis,
    )

    for c, phi in enumerate(phis.tolist()):
        steps = torch.arange(horizon, dtype=dtype)
        lower = torch.tril(phi ** (steps[:, None] - steps[None, :]).clamp_min(0))
        sigma = lower @ torch.diag(variances[:, c]) @ lower.T
        density = torch.distributions.MultivariateNormal(torch.zeros(horizon, dtype=dtype), sigma)
        expected = -density.log_prob(residual[0, 0, :, c])
        torch.testing.assert_close(score[0, 0, :, c].sum(), expected, rtol=1e-12, atol=1e-12)


def test_the_cell_mask_zeroes_exactly_the_masked_cells_and_shapes_are_checked() -> None:
    r"""$m_{\tau,c} = 0$ cells score exactly $0$ and the rest are untouched; a mask that is not
    $(H, C)$ or an ``ar_coef`` that is not $(C,)$ is refused rather than broadcast."""
    generator = torch.Generator().manual_seed(3)
    mu, target, logvar = (torch.randn(2, 3, 5, 4, generator=generator) for _ in range(3))
    mask = (torch.arange(5)[:, None] < torch.tensor([5, 3, 1, 4])[None, :]).float()

    plain = raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar)
    masked = raw_sample_score(
        mu, target, likelihood="gaussian_nll", logvar=logvar, cell_mask=mask
    )

    kept = mask.bool().expand_as(plain)
    assert torch.all(masked[~kept] == 0.0)
    assert torch.equal(masked[kept], plain[kept])

    for bad_mask in (torch.ones(5, 5), torch.ones(4, 5), torch.ones(5)):
        with pytest.raises(ValueError, match="cell_mask"):
            raw_sample_score(mu, target, likelihood="mse", cell_mask=bad_mask)
    for bad_coef in (torch.zeros(5), torch.zeros(1, 4), torch.zeros(())):
        with pytest.raises(ValueError, match="ar_coef"):
            raw_sample_score(mu, target, likelihood="mse", ar_coef=bad_coef)


# =================================================================================================
# The model
# =================================================================================================
def test_both_terms_are_off_by_default() -> None:
    """No keyword, no buffer, no parameter -- and the likelihood kwargs are both ``None``, which is
    the factorised all-cells score every site passes."""
    model = build(_kwargs())

    assert "target_cell_mask" not in dict(model.named_buffers())
    assert "target_ar_logit" not in dict(model.named_parameters())
    assert model.forecast_likelihood_kwargs() == {"cell_mask": None, "ar_coef": None}


def test_on_the_mask_and_the_coefficient_have_the_kept_geometry() -> None:
    r"""$(H, C_{\mathrm{keep}})$ with $m_{\tau,k} = \mathbb 1[\tau < H_{\mathrm{keep}[k]}]$, and a
    zero $(C_{\mathrm{keep}},)$ logit. The mask is non-persistent -- the vector reaches the
    checkpoint through ``model_kwargs`` -- and the logit, a learned parameter, is persistent."""
    steps = _scored_horizon()
    model = build(_kwargs(target_scored_horizon=steps, forecast_ar_residual=True))
    keep = model.target_gate.keep_index.tolist()
    horizon, kept = model.horizon, len(keep)

    mask = model.target_cell_mask
    assert mask.shape == (horizon, kept)
    expected = torch.tensor(
        [[1.0 if tau < steps[index] else 0.0 for index in keep] for tau in range(horizon)]
    )
    assert torch.equal(mask, expected)
    # Not vacuous: some kept channel is actually cut short.
    assert bool((mask == 0).any())

    assert model.target_ar_logit.shape == (kept,)
    assert torch.equal(model.target_ar_logit.detach(), torch.zeros(kept))
    state = model.state_dict()
    assert "target_cell_mask" not in state
    assert "target_ar_logit" in state

    likelihood_kwargs = model.forecast_likelihood_kwargs()
    assert likelihood_kwargs["cell_mask"] is model.target_cell_mask
    assert torch.equal(likelihood_kwargs["ar_coef"], torch.zeros(kept))


def test_at_init_the_ar_residual_is_bitwise_the_factorised_model() -> None:
    r"""$\phi_c = \tanh(0) = 0$: the same-seed model with and without the AR residual reports the
    same loss, and ``pred_gap`` is exactly $0$ because the posterior deltas start at zero."""
    plain_kwargs = _kwargs()
    ar_kwargs = _kwargs(forecast_ar_residual=True)
    plain, ar = build(plain_kwargs), build(ar_kwargs)

    out_plain, features, weight = _run(plain, plain_kwargs)
    out_ar, _features, _weight = _run(ar, ar_kwargs)
    metrics_plain = plain.compute_loss(out_plain, features, weight=weight)["metrics"]
    metrics_ar = ar.compute_loss(out_ar, features, weight=weight)["metrics"]

    for name in _LOSS_METRICS:
        assert torch.equal(metrics_plain[name], metrics_ar[name]), name
    assert float(metrics_ar["pred_gap"]) == 0.0


def test_both_branches_are_scored_under_one_phi_and_phi_is_reachable(perturb_posterior) -> None:
    r"""With $\phi \ne 0$ and the mask on, each branch's block recomputed through
    ``masked_raw_block_per_anchor`` with the model's own likelihood kwargs reproduces the reported
    ``nll_*_block``, and ``pred_gap`` is their difference; scoring either branch factorised
    instead would not. The logit then receives a nonzero gradient from ``total_loss`` -- which is
    what ``find_unused_parameters=False`` needs of it."""
    kwargs = _kwargs(target_scored_horizon=_scored_horizon(), forecast_ar_residual=True)
    model = build(kwargs)
    perturb_posterior(model)
    _set_ar_logit(model)

    out, features, weight = _run(model, kwargs, grad=True)
    metrics = model.compute_loss(out, features, weight=weight)["metrics"]

    mask, _coverage = forecast_mask(
        model.scored_weight(weight),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=out["anchor_index"],
        anchor_valid=out["anchor_valid"],
    )
    target = model._build_forecast_target(features, out["anchor_index"])

    def _block(branch, **likelihood_kwargs):
        per_anchor, contributing = masked_raw_block_per_anchor(
            out[f"mu_{branch}"].detach(),
            target,
            mask,
            likelihood="gaussian_nll",
            logvar=out[f"logvar_{branch}"].detach(),
            channel_weight=model.target_channel_weight,
            horizon_weight=getattr(model, "horizon_weight", None),
            **likelihood_kwargs,
        )
        return per_anchor.sum() / contributing.sum().clamp_min(1.0)

    shared = {
        name: None if value is None else value.detach()
        for name, value in model.forecast_likelihood_kwargs().items()
    }
    full, base = _block("full", **shared), _block("base", **shared)
    torch.testing.assert_close(metrics["nll_full_block"].detach(), full, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(metrics["nll_base_block"].detach(), base, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(metrics["pred_gap"], base - full, rtol=1e-5, atol=1e-6)
    factorised = _block("full", cell_mask=shared["cell_mask"])
    assert not torch.allclose(factorised, full), "phi reached no score; the test is vacuous"

    model.zero_grad(set_to_none=True)
    metrics["total_loss"].backward()
    grad = model.target_ar_logit.grad
    assert grad is not None and bool(torch.isfinite(grad).all())
    assert float(grad.abs().sum()) > 0.0


def test_perturbing_masked_cells_moves_the_loss_by_exactly_zero(perturb_posterior) -> None:
    r"""The mask is a per-channel suffix $\tau \ge H_c$, so a masked cell's residual only ever
    conditions the next cell of its channel, which is masked too. Perturbing ``mu`` and
    ``logvar`` of both branches at every masked $(\tau, k)$ therefore leaves every reported loss
    bitwise unchanged under a nonzero $\phi$; the same perturbation at one scored cell moves it."""
    kwargs = _kwargs(target_scored_horizon=_scored_horizon(), forecast_ar_residual=True)
    model = build(kwargs)
    perturb_posterior(model)
    _set_ar_logit(model)
    out, features, weight = _run(model, kwargs)
    masked_cells = model.target_cell_mask == 0
    assert bool(masked_cells.any()), "no masked cell; the test is vacuous"

    def _loss(outputs):
        with torch.no_grad():
            return model.compute_loss(outputs, features, weight=weight)["metrics"]

    def _perturbed(cells):
        moved = dict(out)
        for name in ("mu_full", "mu_base", "logvar_full", "logvar_base"):
            moved[name] = out[name] + 3.0 * cells.to(out[name].dtype)
        return moved

    reference = _loss(out)
    shifted = _loss(_perturbed(masked_cells))
    for name in _LOSS_METRICS + ("pred_gap_st", "pred_gap_ph"):
        assert torch.equal(reference[name], shifted[name]), name

    scored_cell = torch.zeros_like(masked_cells)
    scored_cell[0, 0] = True
    assert not bool(masked_cells[0, 0])
    moved = _loss(_perturbed(scored_cell))
    assert not torch.equal(reference["nll_full_block"], moved["nll_full_block"])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda steps: steps[:-1], id="short"),
        pytest.param(lambda steps: steps + [TINY_HORIZON], id="long"),
        pytest.param(lambda steps: [0] + steps[1:], id="zero"),
        pytest.param(lambda steps: [TINY_HORIZON + 1] + steps[1:], id="beyond_horizon"),
    ],
)
def test_a_malformed_scored_horizon_is_refused_at_construction(bad) -> None:
    r"""$H_c$ is positional over the $c_y$ declared channels and lies in $[1, H]$; anything else is
    a construction-time ``ValueError``, not a silently broadcast mask."""
    with pytest.raises(ValueError, match="target_scored_horizon"):
        build(_kwargs(target_scored_horizon=bad(_scored_horizon())))


def test_the_in_training_gap_splits_still_sum_to_pred_gap(perturb_posterior) -> None:
    r"""$\mathrm{pred\_gap\_st} + \mathrm{pred\_gap\_ph} = \mathrm{pred\_gap}$ with the mask on and
    $\phi \ne 0$: the splits score under the same likelihood kwargs as the objective."""
    kwargs = _kwargs(target_scored_horizon=_scored_horizon(), forecast_ar_residual=True)
    model = build(kwargs)
    perturb_posterior(model)
    _set_ar_logit(model)
    out, features, weight = _run(model, kwargs)

    metrics = model.compute_loss(out, features, weight=weight)["metrics"]

    assert float(metrics["pred_gap"].abs()) > 0.0, "pred_gap is zero; the test is vacuous"
    torch.testing.assert_close(
        metrics["pred_gap_st"] + metrics["pred_gap_ph"], metrics["pred_gap"], rtol=1e-5, atol=1e-6
    )


# =================================================================================================
# The resolver and the driver
# =================================================================================================
def _int_config(**vae_overrides):
    """A config on the committed integer-operator shard, at its declared target width."""
    return causal_config(paths=[INT_CAUSAL_SHARD], **dict({"c_y": INT_C_Y}, **vae_overrides))


def test_the_resolver_reads_h_c_off_the_shards_slow_leg_frequencies() -> None:
    r"""Both keys null resolves to ``None``; both set gives $H$ on every ``fhr_st`` channel and on
    every ``fhr_ph`` channel with $\xi_{i(c)} \le f_{\mathrm{cut}}$, and $H_{\mathrm{fast}}$
    exactly where $\xi_{i(c)} > f_{\mathrm{cut}}$."""
    assert resolve_target_scored_horizon(_int_config()) is None
    assert (
        resolve_target_scored_horizon(
            _int_config(target_phase_fast_cutoff_hz=None, target_phase_fast_horizon=None)
        )
        is None
    )

    with h5py.File(INT_CAUSAL_SHARD, "r") as handle:
        width_st = int(handle["fhr_st"].shape[1])
        xi_slow = np.asarray(handle["fhr_ph"].attrs["sel_xi_i_hz"], dtype=float)
    cutoff = float(np.median(xi_slow))
    config = _int_config(target_phase_fast_cutoff_hz=cutoff, target_phase_fast_horizon=2)
    horizon = int(config["model_config"]["VAE_model"]["horizon"])

    resolved = resolve_target_scored_horizon(config)

    assert len(resolved) == INT_C_Y == width_st + xi_slow.size
    assert resolved[:width_st] == (horizon,) * width_st
    expected_ph = tuple(2 if xi > cutoff else horizon for xi in xi_slow)
    assert resolved[width_st:] == expected_ph
    # Not vacuous: both branches of the rule occur on the fixture.
    assert 2 in expected_ph and horizon in expected_ph


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(dict(target_phase_fast_cutoff_hz=0.1), id="cutoff_only"),
        pytest.param(dict(target_phase_fast_horizon=2), id="fast_only"),
        pytest.param(
            dict(target_phase_fast_cutoff_hz=0.1, target_phase_fast_horizon=0), id="fast_zero"
        ),
        pytest.param(
            dict(target_phase_fast_cutoff_hz=0.1, target_phase_fast_horizon=10**6),
            id="fast_beyond_horizon",
        ),
        pytest.param(
            dict(
                target_phase_fast_cutoff_hz=0.1, target_phase_fast_horizon=2, c_y=INT_C_Y + 1
            ),
            id="c_y_disagrees",
        ),
    ],
)
def test_the_resolver_refuses_a_half_rule_a_bad_horizon_and_a_width_mismatch(overrides) -> None:
    """The two keys are a pair, the fast horizon lies in $[1, H]$, and the shard's two target
    blocks must add up to the declared $c_y$."""
    with pytest.raises(ValueError):
        resolve_target_scored_horizon(_int_config(**overrides))


def test_the_driver_hands_the_resolved_h_c_to_the_constructor(tmp_path) -> None:
    r"""With the two rule keys and ``forecast_ar_residual`` set on the tiny config, the driver's
    kwarg sweep carries the resolver's $H_c$ vector and the flag, and the constructor accepts
    them. The keys are set here rather than read off ``tiny.yaml``, so the test does not depend on
    what that file currently ships."""
    config = copy.deepcopy(absolutize_dataset_paths(load_config(str(_TINY))))
    vae = config["model_config"]["VAE_model"]
    vae.update(
        target_phase_fast_cutoff_hz=0.1,
        target_phase_fast_horizon=1,
        forecast_ar_residual=True,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    driver = LagAttnTrfCfsTrainer(config_file_path=str(path))
    driver.output_base_dir = str(tmp_path)

    kwargs = driver._build_model_kwargs()

    assert tuple(kwargs["target_scored_horizon"]) == resolve_target_scored_horizon(driver.config)
    assert kwargs["forecast_ar_residual"] is True
    for key in ("target_phase_fast_cutoff_hz", "target_phase_fast_horizon"):
        assert key not in kwargs
    model = driver.MODEL_CLS(**kwargs)
    assert model.target_cell_mask.shape == (model.horizon, model.decoder_out_channels)


def test_a_masked_step_restarts_the_ar_recursion() -> None:
    r"""A gap inside the window is neither read nor trained through.

    With the step at $\tau = 2$ masked, the innovation at $\tau = 3$ must not lag the gap's residual:
    the recursion restarts there as it starts at $\tau = 0$. So changing the gap's target moves no
    block score, and the mean at the gap receives exactly zero gradient.
    """
    torch.manual_seed(0)
    horizon, channels = 6, 3
    target = torch.randn(1, 1, horizon, channels)
    mu = torch.randn(1, 1, horizon, channels, requires_grad=True)
    logvar = torch.zeros(1, 1, horizon, channels)
    mask = torch.ones(1, 1, horizon)
    mask[..., 2] = 0.0
    ar_coef = torch.full((channels,), 0.8)

    block, _ = masked_raw_block_per_anchor(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar, ar_coef=ar_coef
    )
    block.sum().backward()
    assert mu.grad is not None
    assert torch.count_nonzero(mu.grad[..., 2, :]) == 0

    perturbed = target.clone()
    perturbed[..., 2, :] += 100.0
    moved, _ = masked_raw_block_per_anchor(
        mu.detach(), perturbed, mask, likelihood="gaussian_nll", logvar=logvar, ar_coef=ar_coef
    )
    assert torch.equal(moved, block.detach())
