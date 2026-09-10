r"""What this package's binding declares, and how its predictive score is actually computed.

Two halves, and they are here together because they are the two facts a run's numbers depend on
that nothing in the pipeline can derive: which model is rebuilt and which keys are reconciled
against its checkpoint, and what "the predictive score" means once it is.

**The interesting binding failure is a key that names nothing.** ``preflight.reconcile`` compares
``model_config.VAE_model[key]`` against ``model_kwargs[key]`` and silently skips any key absent from
either side, so a ``geometry_keys`` entry that is not both a constructor parameter *and* a config
key is a reconciliation that never happens and never says so. Both halves are asserted against the
class and against the shipped ``configs/default.yaml`` rather than against a second hand-kept list.

**The interesting scoring failures are quieter still.** A marginalised predictive density that is
secretly a mean of per-draw negative log likelihoods improves as the latent becomes less
informative; a mixture calibration built from a mean of conditional standard deviations is a
different distribution from the one the model actually predicts, and both produce plausible numbers.
The checks below pin each against a value computed a second way.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch
import yaml

from teb_vae.lag_attn_cfs.eval import run as shared_run
from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides
from teb_vae.lag_slot_transformer_cfs.eval import predictive
from teb_vae.lag_slot_transformer_cfs.eval.binding import (
    ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE,
    EXCLUDED_ANALYSES,
    GEOMETRY_KEYS,
    LAG_RESIDUAL_BINDING,
    UNREGISTERED_ANALYSES,
    residual_encoder_disclosure,
)
from teb_vae.lag_slot_transformer_cfs.nets.core import REFUSED_KEYWORDS
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.task import SeqVaeLagResidualTrfCfsTask

from .conftest import build_tiny_model

#: The shipped production configuration, which the geometry keys are checked against.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: Every analysis this architecture cannot produce, written out rather than imported: importing
#: the mapping and comparing it with itself would pass on any edit.
EXPECTED_ABSENT = (
    "attention",
    "lag_kl",
    "source_null",
    "occlusion",
    "lag_clocks",
    "lag_kld_scaled",
    "lag_high_kl",
)

#: The two of them the SHARED registry actually holds, which are therefore the only two a binding
#: can remove. The other five are the lag-attentive cell's own extras and are absent here because
#: nothing registers them, which is a different mechanism with the same cause.
EXPECTED_REMOVALS = ("attention", "lag_kl")


@pytest.fixture(scope="module")
def shipped_vae_config():
    """The ``model_config.VAE_model`` block of the shipped production configuration.

    Returns:
        The block as a dict.
    """
    from teb_vae.lag_attn.config import load_config

    return load_config(str(DEFAULT_CONFIG))["model_config"]["VAE_model"]


@pytest.fixture(scope="module")
def constructor_parameters():
    """Every keyword the model's constructor accepts.

    Returns:
        A frozenset of names.
    """
    return frozenset(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters) - {"self"}


# =============================================================================
# The binding
# =============================================================================
def test_the_binding_names_this_packages_model_and_task() -> None:
    """A wrong class either refuses by name or evaluates one architecture under another's."""
    assert LAG_RESIDUAL_BINDING.model_cls is SeqVaeLagResidualTrfCfs
    assert LAG_RESIDUAL_BINDING.task_cls is SeqVaeLagResidualTrfCfsTask
    assert LAG_RESIDUAL_BINDING.tag == "lag_slot_transformer_cfs"


def test_every_geometry_key_is_a_parameter_of_this_constructor(constructor_parameters) -> None:
    """A key the constructor does not accept can never match, and refuses every run."""
    unknown = sorted(set(GEOMETRY_KEYS) - constructor_parameters)
    assert not unknown, unknown


def test_every_geometry_key_is_also_a_shipped_config_key(shipped_vae_config) -> None:
    """A key absent from the config is skipped by the reconciler with nothing said.

    This is the half that fails silently: the run passes, the config and the checkpoint are free to
    disagree about that key, and the symptom appears later as numbers computed at a geometry nobody
    chose.
    """
    unknown = sorted(set(GEOMETRY_KEYS) - set(shipped_vae_config))
    assert not unknown, unknown


def test_no_refused_keyword_is_reconciled() -> None:
    """A key naming a mechanism the constructor rejects would refuse every run outright."""
    overlap = sorted(set(GEOMETRY_KEYS) & set(REFUSED_KEYWORDS))
    assert not overlap, overlap


def test_the_geometry_keys_are_unique_and_carry_this_architectures_own() -> None:
    """The five that decide what a reported number means rather than how well the model fits."""
    assert len(set(GEOMETRY_KEYS)) == len(GEOMETRY_KEYS)
    for key in (
        "residual_mu_scale",
        "residual_logsigma_scale",
        "lag_scale",
        "mean_only_residual",
        "source_scalar_lift",
    ):
        assert key in GEOMETRY_KEYS, key


def test_the_capacity_and_chunk_keys_are_deliberately_absent() -> None:
    """Reconciling them would refuse a correct run.

    The chunk sizes change floating-point summation order and nothing else, and an evaluation
    legitimately runs at a different tiling from the fit that produced the checkpoint.
    """
    for key in ("anchor_chunk", "lag_chunk", "proposal_hidden", "lag_embed_dim"):
        assert key not in GEOMETRY_KEYS, key


def test_the_binding_removes_the_two_shared_analyses_it_cannot_produce() -> None:
    """Removed by name, never handed a substitute.

    The alternative to removing them is feeding a lag readout a proposal norm under an attention
    name, which every reader and every downstream table would take for an attention allocation.
    """
    assert EXCLUDED_ANALYSES == EXPECTED_REMOVALS

    shared = list(shared_run.ANALYSIS_FUNCTIONS)
    reduced = list(shared_run.merged_analysis_functions(LAG_RESIDUAL_BINDING))
    assert set(shared) - set(reduced) == set(EXPECTED_REMOVALS)
    # Nothing else moved, so a reader comparing this cell's output against the lag-attentive one
    # finds fewer columns in the same order rather than a reordering.
    assert reduced == [name for name in shared if name not in EXPECTED_REMOVALS]


def test_only_analyses_the_shared_registry_holds_may_be_removed() -> None:
    """The five that are absent rather than removed, and why the distinction has to be kept.

    Naming one of them on the binding would refuse every run: an exclusion the registry does not
    hold is rejected rather than ignored, which is the guard that catches a misspelt one.
    """
    assert set(UNREGISTERED_ANALYSES) == set(EXPECTED_ABSENT) - set(EXPECTED_REMOVALS)
    assert not set(UNREGISTERED_ANALYSES) & set(shared_run.ANALYSIS_FUNCTIONS)


def test_every_absent_analysis_says_which_tensor_it_would_have_needed() -> None:
    """A reader finding fewer columns than a sibling has to be able to read why."""
    assert set(ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE) == set(EXPECTED_ABSENT)
    for name, reason in ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE.items():
        assert len(reason.strip()) > 40, name


def test_the_overrides_path_is_a_committed_delta_and_not_a_config() -> None:
    """It has no ``base:`` chain, which is refused: a chain would evaluate against what a config
    file says today rather than against what produced the checkpoint."""
    path = LAG_RESIDUAL_BINDING.overrides_path
    assert path.is_file(), path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "base" not in raw
    # And it loads through the shared loader, which is what a run actually uses.
    assert load_eval_overrides(path)["eval_config"]["occlusion_bands"]


def test_the_disclosure_reads_the_model_rather_than_a_literal() -> None:
    """A run at another geometry has to disclose its own, and the searched window is one of the
    two numbers a lag readout is unreadable without."""
    model = build_tiny_model()
    record = residual_encoder_disclosure(model)

    assert record["source_receptive_field_steps"] == 1
    assert record["searched_lag_steps"] == model.n_lags
    assert record["furthest_searched_lag"] == model.n_lags - 1
    # The recommended arm's central structural claim, on the model a run would actually score.
    assert record["source_encoder_parameters"] == 0
    assert "NEURAL" in record["qualification"]
    assert "feature extraction" in record["qualification"]


def test_the_disclosure_reports_the_lift_arm_as_having_parameters() -> None:
    """The claim is about the shipped arm and must not be reported for one that does not hold."""
    lifted = build_tiny_model(source_scalar_lift=True)
    assert residual_encoder_disclosure(lifted)["source_encoder_parameters"] > 0


# =============================================================================
# The predictive scoring path
# =============================================================================
class _TwoStepDecoder(torch.nn.Module):
    """A decoder whose forecast is an affine function of the latent, with a fixed log-variance.

    Deterministic and invertible by hand, so a score computed through the estimator can be checked
    against one computed directly rather than against a second run of the same code.
    """

    def __init__(self, horizon: int, channels: int, logvar: float) -> None:
        """Initialize the stub.

        Args:
            horizon: Forecast steps $H$.
            channels: Forecast channels $C$.
            logvar: The constant observation log-variance it reports.
        """
        super().__init__()
        self.horizon, self.channels, self.logvar = horizon, channels, logvar

    def forward(self, latent, persistence=None):
        """Return the broadcast latent mean and a constant log-variance.

        Args:
            latent: The latent $(B, A, d_z)$.
            persistence: Ignored; present so the call signature matches the real decoder's.

        Returns:
            ``(mu, logvar)``, each $(B, A, H, C)$.
        """
        del persistence
        mu = latent.mean(dim=-1)[..., None, None].expand(-1, -1, self.horizon, self.channels)
        return mu.contiguous(), torch.full_like(mu, self.logvar)


class _StubModel:
    """The minimum a predictive score needs: something with a ``decoder``."""

    def __init__(self, decoder) -> None:
        """Bind the decoder.

        Args:
            decoder: The module invoked once per branch per draw.
        """
        self.decoder = decoder


def _scoring_fixture(*, batch=2, anchors=3, horizon=2, channels=2, d_z=4):
    """Build a stub model, a target and a mask at a small declared geometry.

    Args:
        batch: Samples.
        anchors: Decoded anchors.
        horizon: Forecast steps.
        channels: Forecast channels.
        d_z: Latent width.

    Returns:
        ``(model, target, mask, shape)``.
    """
    torch.manual_seed(20260909)
    model = _StubModel(_TwoStepDecoder(horizon, channels, logvar=0.0))
    target = torch.randn(batch, anchors, horizon, channels)
    mask = torch.ones(batch, anchors, horizon)
    return model, target, mask, (batch, anchors, d_z)


def test_two_branches_with_identical_parameters_score_bitwise_identically() -> None:
    """The common-random-numbers property, and the reason every arm is scored in one loop.

    Without it the selectors-off arm's margin would be a small nonzero number that has to be argued
    away rather than a zero that proves the two paths are one computation.
    """
    model, target, mask, shape = _scoring_fixture()
    mu, logvar = torch.randn(*shape), torch.zeros(*shape)
    generator = torch.Generator().manual_seed(11)

    scored = predictive.matched_predictive_scores(
        model,
        {"left": (mu, logvar), "right": (mu.clone(), logvar.clone())},
        target,
        mask,
        likelihood="gaussian_nll",
        num_samples=4,
        generator=generator,
    )

    assert torch.equal(scored["left"].marginal, scored["right"].marginal)
    assert torch.equal(scored["left"].per_draw, scored["right"].per_draw)


def test_the_marginal_is_the_log_mean_likelihood_and_not_the_mean_log() -> None:
    r"""By Jensen the marginal is at most the mean of the per-draw scores, strictly below it
    whenever the draws disagree.

    The two are easy to confuse and behave in opposite directions: the mean of the per-draw
    negative log likelihoods *improves* as the latent becomes less informative, so a model that
    collapsed its latent would report a better score under it.
    """
    model, target, mask, shape = _scoring_fixture()
    # A wide latent, so the draws genuinely disagree and the gap is not a rounding artifact.
    mu, logvar = torch.randn(*shape), torch.full(shape, 1.0)
    generator = torch.Generator().manual_seed(12)

    scored = predictive.matched_predictive_scores(
        model,
        {"only": (mu, logvar)},
        target,
        mask,
        likelihood="gaussian_nll",
        num_samples=16,
        generator=generator,
    )
    branch = scored["only"]
    mean_of_logs = branch.per_draw.mean(dim=0)

    assert bool((branch.marginal <= mean_of_logs + 1e-9).all())
    assert float((mean_of_logs - branch.marginal).max()) > 0.0
    # And it is exactly the log-mean-likelihood, recomputed here rather than trusted.
    expected = -(
        torch.logsumexp(-branch.per_draw, dim=0) - math.log(float(branch.per_draw.shape[0]))
    )
    assert torch.allclose(branch.marginal, expected, atol=1e-6)


def test_a_stored_clock_latent_is_refused_rather_than_broadcast() -> None:
    """Every latent tensor of this architecture is indexed by anchor, not by stored step.

    Feeding one indexed the other way is the mistake whose symptom is a plausible number: the two
    axes are different lengths and different orders, and a score taken over the wrong rows is still
    a score.
    """
    model, target, mask, shape = _scoring_fixture()
    batch, anchors, d_z = shape
    stored = torch.randn(batch, anchors + 5, d_z)

    with pytest.raises(ValueError, match="indexed by decoded anchor"):
        predictive.matched_predictive_scores(
            model,
            {"only": (stored, torch.zeros_like(stored))},
            target,
            mask,
            likelihood="gaussian_nll",
            num_samples=2,
        )


def test_the_concentration_lies_between_one_and_the_draw_count() -> None:
    """It says how many of the $K$ draws a score effectively rests on, which the marginal cannot."""
    equal = torch.zeros(8, 2, 3)
    assert torch.allclose(
        predictive.draw_concentration(equal), torch.full((2, 3), 8.0, dtype=torch.float64)
    )

    # One draw carrying the whole average: every other draw is worse by a wide margin.
    dominated = torch.full((8, 2, 3), 60.0)
    dominated[0] = 0.0
    assert torch.allclose(
        predictive.draw_concentration(dominated),
        torch.ones(2, 3, dtype=torch.float64),
        atol=1e-6,
    )


def test_the_mixture_cdf_averages_probabilities_and_not_parameters() -> None:
    r"""The decisive check, and the one a plausible wrong implementation fails.

    With two components at $\pm 3$ and unit scale, the mixture's cumulative probability at $y = 3$
    is $\tfrac12\Phi(6) + \tfrac12\Phi(0) = 0.75$. A single Gaussian built from the *average* of the
    components' parameters -- mean $0$, the mean of the conditional standard deviations $1$ -- gives
    $\Phi(3) \approx 0.9987$. The two are not close, and only the first is the law the model
    predicts.
    """
    value = torch.tensor(3.0)
    components = predictive.gaussian_cdf(
        value, torch.tensor([-3.0, 3.0]), torch.zeros(2)
    )
    mixture = float(components.mean())

    assert mixture == pytest.approx(0.75, abs=1e-6)
    averaged_parameters = float(predictive.gaussian_cdf(value, torch.tensor(0.0), torch.tensor(0.0)))
    assert averaged_parameters == pytest.approx(0.99865, abs=1e-4)
    assert abs(mixture - averaged_parameters) > 0.2


def test_central_coverage_follows_from_the_transform_with_no_quantile_solve() -> None:
    r"""An observation lies inside the central-$q$ interval exactly when its own cumulative
    probability lies in $[(1-q)/2, (1+q)/2]$, whatever shape the distribution has.

    Built from a transform that is uniform by construction, so the coverage must equal its nominal
    level and the mean and variance must be the uniform ones.
    """
    uniform = torch.linspace(0.0005, 0.9995, 1000).reshape(1, 1, 1, 1000)
    mask = torch.ones(1, 1, 1)
    totals = predictive.calibration_census(uniform, mask, levels=(0.5, 0.9))
    finished = predictive.finish_calibration(totals)

    assert finished["n_coefficients"] == 1000.0
    assert finished["pit_mean"] == pytest.approx(0.5, abs=1e-3)
    assert finished["pit_var"] == pytest.approx(1.0 / 12.0, abs=1e-3)
    assert finished["coverage"]["0.5"] == pytest.approx(0.5, abs=2e-3)
    assert finished["coverage"]["0.9"] == pytest.approx(0.9, abs=2e-3)


def test_the_calibration_census_accumulates_exactly_across_batches() -> None:
    """Two halves merged must equal the whole, or a long run reports a mean of per-batch means.

    That is not a rounding difference: it weights a short final batch equally with a full one, and
    the variance is not recoverable from means at all, which is why the squared sum travels.
    """
    values = torch.rand(1, 1, 1, 40)
    mask = torch.ones(1, 1, 1)
    whole = predictive.finish_calibration(
        predictive.calibration_census(values, mask, levels=(0.9,))
    )

    merged = None
    for start in (0, 17):
        stop = 17 if start == 0 else 40
        merged = predictive.merge_calibration(
            merged, predictive.calibration_census(values[..., start:stop], mask, levels=(0.9,))
        )
    halves = predictive.finish_calibration(merged)

    assert halves["n_coefficients"] == whole["n_coefficients"]
    assert halves["pit_mean"] == pytest.approx(whole["pit_mean"], abs=1e-12)
    assert halves["pit_var"] == pytest.approx(whole["pit_var"], abs=1e-12)
    assert halves["coverage"]["0.9"] == pytest.approx(whole["coverage"]["0.9"], abs=1e-12)


def test_a_pass_that_scored_nothing_reports_absence_rather_than_zero() -> None:
    """A zero probability-integral transform mean is a measurement; an empty pass is not one."""
    finished = predictive.finish_calibration(None)
    assert finished["n_coefficients"] == 0.0
    assert finished["pit_mean"] is None
    assert finished["coverage"] == {}
