r"""The evaluation scores every forecast under the density the model trained under.

Two terms of the causal cells' forecast likelihood are part of the density rather than weights on
it: the per-channel scored horizon $m_{\tau,c} = \mathbb 1[\tau < H_c]$, which removes cells from
the block, and the AR(1) residual $e_\tau = r_\tau - \phi_c\, r_{\tau-1}$, which scores each cell
on its innovation. The objective passes both; so must every density readout here, or ``nll_*``,
``pred_gap``, the Monte Carlo marginal and the baselines describe a likelihood nobody fitted.

Every test builds the tiny model with **both** terms on and a non-zero $\phi$ -- at the zero
initialisation $\phi_c = \tanh 0 = 0$ and the AR term would be inert, so an assertion about it
would pass on an evaluation that ignored it. The objective weights stay at their unweighted
defaults (equal block weights, no horizon half-life), which is the only setting in which the
evaluation's unweighted score and the objective's are the same number.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    anchor_support,
    branch_channel_scores,
    calibration_sums,
    describe_likelihood_structure,
    evaluate_batch,
    forecast_likelihood_terms,
    horizon_block_sums,
    horizon_residual_sums,
    likelihood_structure_record,
    masked_raw_error_sums,
    mc_predictive_block,
    mean_decoded_block,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor

from .conftest import (
    CAUSAL_C_Y,
    CAUSAL_ST_WIDTH,
    TINY_HORIZON,
    TINY_STRIDE,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)

#: $H_c$ per declared channel: the full horizon on the envelope block, half of it on the phase
#: block -- so the cell mask removes a real share of the block rather than nothing.
SCORED_HORIZON = (TINY_HORIZON,) * CAUSAL_ST_WIDTH + (max(1, TINY_HORIZON // 2),) * (
    CAUSAL_C_Y - CAUSAL_ST_WIDTH
)


@pytest.fixture
def structured_task(perturb_posterior):
    """The tiny task with a scored horizon, an AR(1) residual at a non-zero $\\phi$, and a
    posterior moved off the prior; ``base_decode='mean'`` so the base branch is deterministic."""
    module = make_task(
        model_kwargs=tiny_warmup_kwargs(
            anchor_stride=TINY_STRIDE,
            base_decode="mean",
            target_scored_horizon=SCORED_HORIZON,
            forecast_ar_residual=True,
        )
    )
    model = module.orig_model
    perturb_posterior(model)
    with torch.no_grad():
        width = int(model.target_ar_logit.shape[0])
        model.target_ar_logit.copy_(torch.linspace(-1.2, 1.2, width))
    module.eval()
    return module


def _dense(module, batch, seed: int = 11):
    """The dense forward and everything the collection pass builds from it, under a fixed seed."""
    model = module.orig_model
    y_st, y_ph, u_stream, target_features, weight = model_inputs(module, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    target = model._build_forecast_target(target_features, outputs["anchor_index"])
    mask, _coverage, _support = anchor_support(model, weight, outputs)
    return outputs, target_features, weight, target, mask


def test_the_structure_is_on_and_not_inert(structured_task) -> None:
    """Non-vacuity: some cells are unscored, and $\\phi$ is away from zero on both blocks."""
    model = structured_task.orig_model
    terms = forecast_likelihood_terms(model)
    assert terms["cell_mask"] is not None and terms["ar_coef"] is not None
    assert not terms["ar_coef"].requires_grad, "the evaluation must never reach phi's gradient"

    record = likelihood_structure_record(model)
    assert record["forecast_ar_residual"] is True
    assert record["block_cells"] == TINY_HORIZON * int(model.decoder_out_channels)
    assert record["scored_cells"] == int(terms["cell_mask"].sum().item())
    assert 0 < record["scored_cells"] < record["block_cells"]
    assert record["ar_coef_mean_st"] is not None and record["ar_coef_mean_ph"] is not None
    assert "AR(1) residual on" in describe_likelihood_structure(record)


def test_the_eval_block_scores_are_the_training_objectives(structured_task) -> None:
    """(a) Both branches' training-path block scores, pooled over the contributing anchors, are the
    objective's own ``nll_*_block`` on the same forward -- which they are only if the evaluation
    applies the cell mask and the AR(1) term exactly as ``compute_loss`` does."""
    module = structured_task
    model = module.orig_model
    batch = make_stub_batch(seed=5)

    torch.manual_seed(11)
    readout = evaluate_batch(module, batch, num_samples=1)
    outputs, target_features, weight, target, mask = _dense(module, batch, seed=11)
    with torch.no_grad():
        objective = model.compute_loss(
            outputs, target_features, weight=weight, likelihood="gaussian_nll"
        )["metrics"]

    contributing = readout.per_anchor["contributing"]
    assert float(contributing.sum()) > 0.0
    for branch in ("base", "full"):
        pooled = (readout.per_anchor[f"nll_{branch}_block"] * contributing).sum() / contributing.sum()
        torch.testing.assert_close(
            pooled, objective[f"nll_{branch}_block"].to(pooled.dtype), rtol=1e-5, atol=1e-4
        )

    # Under base_decode='mean' the mean-decoded base score IS the training path's, anchor by anchor.
    torch.testing.assert_close(
        readout.per_anchor["mean_nll_base_block"] * contributing,
        readout.per_anchor["nll_base_block"] * contributing,
    )
    # The per-channel gap vector still decomposes the gap the same readout reports.
    torch.testing.assert_close(
        readout.gap_per_channel.sum(dim=1), readout.columns["pred_gap"], rtol=1e-4, atol=1e-3
    )

    # Non-vacuity: the factorised all-cells score is a different number on this model.
    factorised, _ = masked_raw_block_per_anchor(
        outputs["mu_base"], target, mask, likelihood="gaussian_nll", logvar=outputs["logvar_base"]
    )
    factorised_pooled = (factorised * contributing).sum() / contributing.sum()
    assert not torch.allclose(factorised_pooled, objective["nll_base_block"], rtol=1e-3)


def test_the_monte_carlo_marginal_at_one_draw_is_that_draws_block(structured_task) -> None:
    """(b) At $K = 1$ the log-mean-exp is the identity, so the marginal is the single draw's joint
    block -- scored under the same two terms -- and a draw pinned at the mean is the mean-decoded
    block."""
    module = structured_task
    model = module.orig_model
    outputs, _tf, _weight, target, mask = _dense(module, make_stub_batch(seed=6))
    anchors = outputs["anchor_index"]
    persistence = outputs.get("persistence")
    posterior = (outputs["mu_post"], outputs["logvar_post"])

    with torch.no_grad():
        scores, _ = mc_predictive_block(
            model, {"full": posterior}, target, mask, anchors=anchors, likelihood="gaussian_nll",
            num_samples=1, generator=torch.Generator().manual_seed(0), persistence=persistence,
        )
        epsilon = torch.empty_like(posterior[0]).normal_(generator=torch.Generator().manual_seed(0))
        latent = posterior[0] + epsilon * torch.exp(0.5 * posterior[1])
        index = anchors.to(torch.long)[:, :, None].expand(-1, -1, latent.shape[-1])
        forecast_mu, forecast_logvar = model.decoder(latent.gather(1, index), persistence=persistence)
        single, _ = masked_raw_block_per_anchor(
            forecast_mu, target, mask, likelihood="gaussian_nll", logvar=forecast_logvar,
            **forecast_likelihood_terms(model),
        )
        torch.testing.assert_close(scores["full"], single)

        pinned = (posterior[0], torch.full_like(posterior[1], -1.0e30))
        at_mean, _ = mc_predictive_block(
            model, {"full": pinned}, target, mask, anchors=anchors, likelihood="gaussian_nll",
            num_samples=1, persistence=persistence,
        )
        mean_scores, _ = mean_decoded_block(
            model, {"full": posterior}, target, mask, anchors=anchors,
            likelihood="gaussian_nll", persistence=persistence,
        )
        torch.testing.assert_close(at_mean["full"], mean_scores["full"])


def test_an_unscored_cell_moves_no_density_readout(structured_task) -> None:
    """(c) A cell beyond its channel's $H_c$ is not part of the density: perturbing the target there
    changes no block score, no marginal, no split, no calibration sum and no point error. The AR(1)
    innovation cannot carry it back either -- unscored cells are a trailing run of each channel's
    horizon, so no scored innovation reads one as its predecessor."""
    module = structured_task
    model = module.orig_model
    outputs, _tf, _weight, target, mask = _dense(module, make_stub_batch(seed=7))
    cell_mask = forecast_likelihood_terms(model)["cell_mask"]
    moved = target + 100.0 * (1.0 - cell_mask)
    assert not torch.equal(moved, target)

    anchors = outputs["anchor_index"]
    persistence = outputs.get("persistence")
    branches = {
        "base": (outputs["mu_prior"], outputs["logvar_prior"]),
        "full": (outputs["mu_post"], outputs["logvar_post"]),
    }
    density = forecast_likelihood_terms(model)
    mu, logvar = outputs["mu_full"], outputs["logvar_full"]

    def _readouts(block_target):
        with torch.no_grad():
            marginal, _ = mc_predictive_block(
                model, branches, block_target, mask, anchors=anchors,
                likelihood="gaussian_nll", num_samples=2,
                generator=torch.Generator().manual_seed(3), persistence=persistence,
            )
            mean_scores, _ = mean_decoded_block(
                model, branches, block_target, mask, anchors=anchors,
                likelihood="gaussian_nll", persistence=persistence,
            )
            training, _ = masked_raw_block_per_anchor(
                mu, block_target, mask, likelihood="gaussian_nll", logvar=logvar, **density
            )
            values = {
                **{f"mc_{name}": value for name, value in marginal.items()},
                **{f"mean_{name}": value for name, value in mean_scores.items()},
                "training": training,
                "by_channel": branch_channel_scores(
                    mu, logvar, block_target, mask, likelihood="gaussian_nll", **density
                ),
                "baseline": masked_raw_block_per_anchor(
                    torch.zeros(()), block_target, mask, likelihood="gaussian_nll",
                    logvar=torch.zeros(()), **density,
                )[0],
            }
            values.update({
                f"horizon_{name}": value
                for name, value in horizon_block_sums(
                    mu, logvar, block_target, mask, likelihood="gaussian_nll", **density
                ).items()
            })
            values.update({
                f"residual_{name}": value
                for name, value in horizon_residual_sums(
                    mu, logvar, block_target, mask, **density
                ).items()
            })
            values.update({
                f"calibration_{name}": value
                for name, value in calibration_sums(
                    mu, logvar, block_target, mask, logvar_clamp=model.logvar_clamp, **density
                ).items()
            })
            values.update({
                f"error_{name}": value
                for name, value in masked_raw_error_sums(
                    mu, block_target, mask, cell_mask=density["cell_mask"]
                ).items()
            })
        return values

    reference, perturbed = _readouts(target), _readouts(moved)
    for name, value in reference.items():
        assert torch.equal(value, perturbed[name]), name

    # And the counts are the scored cells, not H * C_keep.
    scored = float((mask[..., None] * cell_mask).sum())
    assert float(reference["calibration_count"]) == pytest.approx(scored)
    assert float(reference["residual_count"].sum()) == pytest.approx(scored)
    assert float(reference["error_n_coefficients"].sum()) == pytest.approx(scored)
    assert scored < float(mask.sum()) * cell_mask.shape[-1]


def test_a_scored_horizon_the_config_does_not_resolve_to_is_refused() -> None:
    """``target_scored_horizon`` has no config key, so the reconciliation re-resolves the rule and
    compares vectors; a checkpoint built with one against a config that sets no rule is refused."""
    stamped = {"target_scored_horizon": SCORED_HORIZON}
    no_rule = {"model_config": {"VAE_model": {}}, "dataset_config": {}}

    with pytest.raises(preflight.EvalPreconditionUnmet, match="target_scored_horizon"):
        preflight.reconcile_with_checkpoint(
            no_rule, model_kwargs=stamped, hyper_parameters={}, geometry_keys=()
        )
    record = preflight.reconcile_with_checkpoint(
        no_rule, model_kwargs={}, hyper_parameters={}, geometry_keys=()
    )
    assert "target_scored_horizon" not in record["compared"]


def test_the_ar_flag_is_reconciled_against_the_checkpoint() -> None:
    """A config and a checkpoint disagreeing about the AR(1) residual would report one likelihood's
    numbers under the other's name. Reconciled in the transformer cell's tuple, the one cell whose
    configuration names the key; the conv-LSTM cell never sets it."""
    from teb_vae.lag_attn_cfs.eval.binding import GEOMETRY_KEYS as CFS_GEOMETRY_KEYS
    from teb_vae.lag_attn_transformer_cfs.eval.binding import GEOMETRY_KEYS

    assert "forecast_ar_residual" in GEOMETRY_KEYS
    assert "forecast_ar_residual" not in CFS_GEOMETRY_KEYS
    config = {"model_config": {"VAE_model": {"forecast_ar_residual": False}}, "dataset_config": {}}
    with pytest.raises(preflight.EvalPreconditionUnmet, match="forecast_ar_residual"):
        preflight.reconcile_with_checkpoint(
            config,
            model_kwargs={"forecast_ar_residual": True},
            hyper_parameters={},
            geometry_keys=GEOMETRY_KEYS,
        )
