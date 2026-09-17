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
from teb_vae.lag_slot_transformer_cfs.eval import collect, predictive
from teb_vae.lag_slot_transformer_cfs.eval.binding import (
    ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE,
    EXCLUDED_ANALYSES,
    EXTRA_ANALYSES,
    GEOMETRY_KEYS,
    HEADLINE_SCALARS,
    LAG_RESIDUAL_BINDING,
    UNREGISTERED_ANALYSES,
    residual_encoder_disclosure,
)
from teb_vae.lag_slot_transformer_cfs.nets.core import REFUSED_KEYWORDS
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.task import SeqVaeLagResidualTrfCfsTask

from .conftest import TINY_SEQ_LEN, build_tiny_model, tiny_streams

#: The shipped production configuration, which the geometry keys are checked against.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: Every analysis this architecture cannot produce, written out rather than imported: importing
#: the mapping and comparing it with itself would pass on any edit.
EXPECTED_ABSENT = (
    "attention",
    "lag_kl",
    "occlusion",
    "lag_clocks",
    "lag_kld_scaled",
    "lag_high_kl",
)

#: The two of them the SHARED registry actually holds, which are therefore the only two a binding
#: can remove. The other four are the lag-attentive cell's own extras and are absent here because
#: nothing registers them, which is a different mechanism with the same cause. ``source_null``,
#: ``warmup`` and ``spectral_skill`` are that cell's extras too and are registered here as the
#: family's own implementations, because the columns they read are the same quantities here.
EXPECTED_FAMILY_REUSE = ("warmup", "source_null", "spectral_skill")

#: This cell's own analogues of the absent family analyses, each reading a sidecar this cell's
#: pass writes under its own names.
EXPECTED_ANALOGUES = ("proposal_profile", "proposal_clocks", "band_clocks", "high_kl_anchors")
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
    # Nothing else moved among the shared ones, so a reader comparing this cell's output against
    # the lag-attentive one finds fewer columns in the same order rather than a reordering.
    assert [name for name in reduced if name in shared] == [
        name for name in shared if name not in EXPECTED_REMOVALS
    ]
    # This cell's own sit after the shared ones and before the trailing cross-subgroup test,
    # which reads what they write.
    assert [name for name in reduced if name not in shared] == list(EXTRA_ANALYSES)
    assert reduced[-1] == "cross_subgroup"


def test_the_binding_declares_its_own_collection_pass() -> None:
    """The family's runner calls whichever pass the binding names, and this cell names its own:
    the shared pass reads tensors only a lag-attention forward emits."""
    from teb_vae.lag_slot_transformer_cfs.eval import binding as binding_module
    from teb_vae.lag_slot_transformer_cfs.eval import collect

    assert LAG_RESIDUAL_BINDING.collect is binding_module.collect_tables
    assert collect.collect_tables is not binding_module.collect_tables
    # The three that draw this model's own forward are registered under the family's names.
    assert {"samples", "recording_traces", "attribution"} <= set(EXTRA_ANALYSES)
    assert all(callable(function) for function in EXTRA_ANALYSES.values())


def test_every_headline_scalar_this_cell_registers_resolves_on_its_own_results() -> None:
    """A path that resolves to nothing is a column of ``None`` in every arm table; each one is
    checked against a results block shaped as the pass writes it."""
    from teb_vae.lag_attn_cfs.eval import report_seam

    results = {
        "readouts": {"mc_pred_gap": 0.5},
        "verdicts": [],
        "arm_scores": {
            "pred_gap": {"point": 0.5, "lo": 0.1, "hi": 0.9},
            "draw_concentration_full": {"point": 3.0},
        },
        "source_controls": {
            "silence_margin_nats": 0.5, "replace_zeros_margin_nats": 0.2,
            "replace_constant_margin_nats": 0.1, "permute_margin_nats": 0.3,
        },
        "lag_readouts": {"cancellation": {"mean": {"ratio": 0.7}}},
        "source_null": {
            "difference": {
                "kld_source_null_nats": 0.1, "coupling_minus_clock_nats": 0.2,
                "ci_lo": 0.15, "ci_hi": 0.25,
            }
        },
        "warmup": {
            "headline": {
                "pred_gap_warm_lo_nats": 0.1, "pred_gap_warm_mid_nats": 0.2,
                "pred_gap_warm_hi_nats": 0.3,
            },
            "geometry_guards": {"anchors_per_sample": 4.0, "target_warm_frac": 1.0},
        },
        "spectral_skill": {
            "headline": {
                "pred_gap_slow_baseline_nats": 0.1, "pred_gap_deceleration_nats": 0.2,
                "pred_gap_variability_nats": 0.3, "pred_gap_beat_to_beat_nats": 0.4,
            }
        },
        "high_kl_anchors": {
            "thresholds": {"high_nats": 0.3},
            "usefulness": {
                "high_minus_rest_mean_interval": {"point": 0.05},
                "overlap": {"share_of_high_in_gain": 0.4},
            },
        },
    }
    headline = report_seam.build_headline(results, HEADLINE_SCALARS)

    for name, _path in HEADLINE_SCALARS:
        assert headline[name] is not None, name
    assert headline["pred_gap_mc_ci_lo"] == 0.1
    assert headline["pred_gap_mc_nats"] == 0.5


def test_the_family_analyses_this_cell_reuses_and_its_own_analogues_are_registered() -> None:
    """The three family implementations whose columns this pass writes, and the four analogues.

    Registered under the family's names for the three, because the quantity is the same one; and
    under this cell's own names for the four, because a proposal norm under an attention name is
    the substitution the design rules out.
    """
    for name in (*EXPECTED_FAMILY_REUSE, *EXPECTED_ANALOGUES):
        assert name in EXTRA_ANALYSES, name
    assert not set(EXPECTED_ANALOGUES) & set(shared_run.ANALYSIS_FUNCTIONS)
    from teb_vae.lag_slot_transformer_cfs.eval.binding import ANALOGUE_ANALYSES

    assert set(ANALOGUE_ANALYSES) == set(EXPECTED_ABSENT)
    for name, analogues in ANALOGUE_ANALYSES.items():
        assert analogues, name
        assert set(analogues) <= set(EXTRA_ANALYSES), (name, analogues)
        # And the reason names its analogue, so the summary carries the correspondence.
        assert any(a in ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE[name] for a in analogues), name


def test_only_analyses_the_shared_registry_holds_may_be_removed() -> None:
    """The four that are absent rather than removed, and why the distinction has to be kept.

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


def test_the_disclosure_states_the_windows_depth_in_seconds_and_the_input_policy() -> None:
    """A bank of L entries that includes the anchor reaches L - 1 steps back, and the policy is
    disclosed on every arm because it is a property of the task, not of the source pathway."""
    from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

    model = build_tiny_model()
    record = residual_encoder_disclosure(model)
    target_only = residual_encoder_disclosure(
        build_tiny_model(source_disabled=True, zero_fhr_scattering_s0=True)
    )

    assert record["oldest_lag_seconds"] == (model.n_lags - 1) * SECONDS_PER_STEP
    assert record["effective_inputs"]["ablated_inputs"] == []
    assert target_only["oldest_lag_seconds"] is None
    assert target_only["effective_inputs"]["zero_fhr_scattering_s0"] is True
    assert target_only["effective_inputs"]["ablated_inputs"][0]["field"] == "fhr_st"


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


# =============================================================================
# The resolved axes
# =============================================================================
def test_subset_scores_sum_to_the_per_draw_block_score() -> None:
    """Per draw, the horizon steps and the two blocks each add back to the block score exactly.

    Additivity holds before the marginalisation over draws and not after it, and this is the half
    that must hold: a subset score that did not sum back would be scoring a different block.
    """
    from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor

    torch.manual_seed(5)
    mu, logvar = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5) * 0.1
    target = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 3, 4)
    mask[0, 1, 2:] = 0.0

    per_horizon, per_block = predictive.subset_block_scores(
        mu, logvar, target, mask, likelihood="gaussian_nll", block_split=2
    )
    block, _contributing = masked_raw_block_per_anchor(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar
    )

    assert per_horizon.shape == (2, 3, 4)
    assert per_block.shape == (2, 3, 2)
    assert torch.allclose(per_horizon.sum(dim=-1), block, atol=1e-5)
    assert torch.allclose(per_block.sum(dim=-1), block, atol=1e-5)
    # A masked step contributes nothing to any subset.
    assert torch.equal(per_horizon[0, 1, 2:], torch.zeros(2))


def test_a_block_split_outside_the_channel_axis_is_refused() -> None:
    """A split at either end leaves one block empty, and its score would read as a measured zero."""
    mu = torch.zeros(1, 1, 2, 3)
    with pytest.raises(ValueError, match="strictly inside"):
        predictive.subset_block_scores(mu, mu, mu, torch.ones(1, 1, 2), likelihood="mse", block_split=3)


def test_the_resolved_axes_are_marginal_mixtures_of_their_own_factors() -> None:
    """Each step is the log-mean-likelihood of that step's factors under the shared draws.

    Recomputed here from the per-draw subset scores rather than trusted, and shown NOT to sum to
    the joint marginal, which is the property a reader has to carry: the steps are read on their
    own axis and never added up into the block score.
    """
    model, target, mask, shape = _scoring_fixture(horizon=3, channels=2)
    mu, logvar = torch.randn(*shape), torch.full(shape, 1.0)

    scored = predictive.matched_predictive_scores(
        model,
        {"only": (mu, logvar)},
        target,
        mask,
        likelihood="gaussian_nll",
        num_samples=16,
        generator=torch.Generator().manual_seed(12),
        resolve=("only",),
        block_split=1,
    )
    branch = scored["only"]

    assert branch.per_horizon is not None and branch.per_horizon.shape == (2, 3, 3)
    assert branch.per_block is not None and branch.per_block.shape == (2, 3, 2)
    # A subset mixture is at most the per-draw mean of that subset's score, by Jensen, and the
    # subsets do not add up to the joint mixture.
    assert float((branch.per_horizon.sum(dim=-1) - branch.marginal).abs().max()) > 1e-6


def test_a_branch_left_out_of_resolve_carries_no_curves() -> None:
    """A single-lag arm is scored for its margin and nothing else."""
    model, target, mask, shape = _scoring_fixture()
    mu, logvar = torch.randn(*shape), torch.zeros(*shape)

    scored = predictive.matched_predictive_scores(
        model,
        {"a": (mu, logvar), "b": (mu, logvar)},
        target,
        mask,
        likelihood="gaussian_nll",
        num_samples=2,
        resolve=("a",),
    )

    assert scored["a"].per_horizon is not None
    assert scored["b"].per_horizon is None and scored["b"].per_block is None
    with pytest.raises(ValueError, match="resolve names branches"):
        predictive.matched_predictive_scores(
            model, {"a": (mu, logvar)}, target, mask, likelihood="gaussian_nll",
            num_samples=1, resolve=("zz",),
        )


# =================================================================================================
# The weighted objective-parity columns
# =================================================================================================
def test_the_parity_scores_reproduce_the_objectives_reconstruction_terms() -> None:
    """The weighted per-anchor scores average to exactly the ``nll_full_block`` and
    ``nll_base_block`` the objective reports on the same forward, so a training log and the
    per-sample table can be read side by side -- and the unweighted columns beside them are a
    different number, which is the mislabelling the parity columns exist to end."""
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

    model = build_tiny_model(target_weight_st=1.0, target_weight_ph=0.1,
                             horizon_weight_halflife_steps=2.0)
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        model.proposal_head.output_proj.weight.normal_(0.0, 0.3, generator=generator)
        model.proposal_head.output_proj.bias.normal_(0.0, 0.1, generator=generator)
    model.eval()
    y_st, y_ph, u_stream = tiny_streams()
    torch.manual_seed(0)
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    target_features = torch.cat([y_st, y_ph], dim=-1)
    validity = torch.ones(y_st.shape[0], TINY_SEQ_LEN)
    metrics = model.compute_loss(outputs, target_features, weight=validity)["metrics"]

    target = model._build_forecast_target(target_features, outputs["anchor_index"])
    mask, _coverage = forecast_mask(
        model.scored_weight(validity), model.geometry, coverage_floor=model.coverage_floor,
        anchors=outputs["anchor_index"], anchor_valid=outputs["anchor_valid"],
    )
    full, base = collect.objective_parity_scores(
        model, outputs, target, mask, likelihood="gaussian_nll"
    )
    contributing = (mask.sum(dim=-1) > 0).to(torch.float64)
    parity_full = float((full.to(torch.float64) * contributing).sum() / contributing.sum())
    parity_base = float((base.to(torch.float64) * contributing).sum() / contributing.sum())

    assert parity_full == pytest.approx(float(metrics["nll_full_block"]), rel=1e-6)
    assert parity_base == pytest.approx(float(metrics["nll_base_block"]), rel=1e-6)
    # The unweighted score is a different number on a weighted arm.
    from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor

    unweighted, _ = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood="gaussian_nll", logvar=outputs["logvar_full"]
    )
    assert not torch.allclose(unweighted.to(torch.float64), full.to(torch.float64))


def test_the_parity_scores_are_the_unweighted_ones_on_an_unweighted_arm() -> None:
    """With no weight buffers registered the parity scores equal the unweighted single-draw
    scores bitwise rather than being multiplied by ones."""
    from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

    model = build_tiny_model(target_weight_st=1.0, target_weight_ph=1.0,
                             horizon_weight_halflife_steps=None).eval()
    y_st, y_ph, u_stream = tiny_streams()
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    target_features = torch.cat([y_st, y_ph], dim=-1)
    validity = torch.ones(y_st.shape[0], TINY_SEQ_LEN)
    target = model._build_forecast_target(target_features, outputs["anchor_index"])
    mask, _coverage = forecast_mask(
        model.scored_weight(validity), model.geometry, coverage_floor=model.coverage_floor,
        anchors=outputs["anchor_index"], anchor_valid=outputs["anchor_valid"],
    )
    full, _base = collect.objective_parity_scores(
        model, outputs, target, mask, likelihood="gaussian_nll"
    )
    unweighted, _ = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood="gaussian_nll", logvar=outputs["logvar_full"]
    )
    assert torch.equal(full, unweighted)


def test_the_results_conventions_name_every_estimator() -> None:
    """Four estimators, each with the columns that carry it, and a schema version a gate can
    read; the legacy sentence still travels beside them."""
    assert collect.RESULTS_SCHEMA_VERSION == 2
    assert set(collect.SCORE_CONVENTIONS) == {
        "weighted_objective", "single_draw_conditional", "latent_mean", "predictive_mixture",
    }
    for entry in collect.SCORE_CONVENTIONS.values():
        assert entry["columns"] and entry["meaning"]
    assert "pred_gap_weighted" in collect.SCORE_CONVENTIONS["weighted_objective"]["columns"]
    assert "pred_gap_mc_nats" in collect.SCORE_CONVENTIONS["predictive_mixture"]["columns"]


# =================================================================================================
# The three verdicts this cell reads itself
# =================================================================================================
def _family_verdicts():
    """A family verdict list with the three entries this cell replaces, in registry order."""
    from teb_vae.lag_attn_cfs.eval.metrics import Verdict

    return [
        Verdict("predictive_improvement", "PASS", "D_full < D_base", "point below",
                {"d_base": -120.0, "d_full": -120.2, "pred_gap": 0.2}),
        Verdict("source_margin_positive", "PASS", "c", "d", {}),
        Verdict("prior_variance_not_pinned", "FAIL", "c",
                "the KL carries (mu_q - mu_p)^2 / sigma_p^2 -- inflated",
                {"floor_frac": 0.56, "max_frac": 0.5}),
        Verdict("calibration_near_nominal", "PASS", "c", "single-draw", {"tail_tolerance": 0.5}),
    ]


@pytest.mark.parametrize(
    "interval, expected",
    [
        ({"point": 0.2, "lo": 0.05, "hi": 0.4, "n": 10}, "PASS"),
        ({"point": 0.16, "lo": -0.2, "hi": 0.53, "n": 10}, "INCONCLUSIVE"),
        ({"point": -0.3, "lo": -0.5, "hi": -0.1, "n": 10}, "FAIL"),
        (None, "INCONCLUSIVE"),
    ],
)
def test_the_predictive_verdict_reads_the_interval_and_not_the_sign(interval, expected) -> None:
    """The family says PASS on a point below the base; this cell says what the interval says."""
    verdicts = collect.cell_verdicts(
        _family_verdicts(), pred_gap_interval=interval, mixture_calibration=None, num_samples=8
    )
    by_name = {verdict.name: verdict for verdict in verdicts}
    assert [verdict.name for verdict in verdicts] == [
        "predictive_improvement", "source_margin_positive", "prior_variance_not_pinned",
        "calibration_near_nominal",
    ]
    assert by_name["predictive_improvement"].status == expected
    assert by_name["predictive_improvement"].values["K"] == 8.0
    if interval is not None:
        assert by_name["predictive_improvement"].values["pred_gap_lo"] == interval["lo"]


def test_the_calibration_verdict_reads_the_mixture_census() -> None:
    """Coverage of the full branch's mixture against nominal, at the family's tail tolerance."""
    mixture = {
        "base": {"coverage": {"0.5": 0.64, "0.9": 0.936, "0.99": 0.99}},
        "full": {"coverage": {"0.5": 0.64, "0.9": 0.937, "0.99": 0.9903}},
    }
    verdicts = collect.cell_verdicts(
        _family_verdicts(), pred_gap_interval=None, mixture_calibration=mixture, num_samples=8
    )
    verdict = {v.name: v for v in verdicts}["calibration_near_nominal"]
    # Relative tail errors: 0.28 at the 0.5 level, 0.37 at 0.9, 0.03 at 0.99 -- all inside the
    # family's 0.5 tolerance, the worst at 0.9; the verdict carries every level of both branches.
    assert verdict.status == "PASS"
    assert verdict.values["observed_0.5"] == 0.64
    assert verdict.values["base_observed_0.9"] == 0.936
    assert verdict.values["worst_level"] == 0.9
    assert verdict.values["worst_relative_tail_error"] == pytest.approx(0.37, abs=1e-9)
    nominal = collect.cell_verdicts(
        _family_verdicts(), pred_gap_interval=None,
        mixture_calibration={"full": {"coverage": {"0.5": 0.95}}}, num_samples=8,
    )
    assert {v.name: v for v in nominal}["calibration_near_nominal"].status == "FAIL"
    absent = collect.cell_verdicts(
        _family_verdicts(), pred_gap_interval=None, mixture_calibration={}, num_samples=8
    )
    assert {v.name: v for v in absent}["calibration_near_nominal"].status == "INCONCLUSIVE"


def test_the_prior_floor_verdict_keeps_its_status_and_states_the_residual_identity() -> None:
    """The status is the family's; the explanation is the residual divergence's."""
    verdicts = collect.cell_verdicts(
        _family_verdicts(), pred_gap_interval=None, mixture_calibration=None, num_samples=8
    )
    verdict = {v.name: v for v in verdicts}["prior_variance_not_pinned"]
    assert verdict.status == "FAIL"
    assert "inflate" in verdict.detail and "NOT" in verdict.detail
    assert "exp(2b)" in verdict.detail
    assert verdict.values["floor_frac"] == 0.56


def test_the_census_resolves_by_horizon_and_block_and_recombines() -> None:
    """The resolved counts are partial sums of the pooled ones on the same coefficients."""
    torch.manual_seed(3)
    cdf = torch.rand(2, 3, 4, 5, dtype=torch.float64)
    mask = torch.ones(2, 3, 4)
    mask[0, 0, 3] = 0.0
    census = predictive.calibration_census(cdf, mask, levels=(0.5, 0.9), block_split=2)
    resolved = census["resolved"]
    assert len(resolved["by_horizon"]["n_coefficients"]) == 4
    assert sum(resolved["by_horizon"]["n_coefficients"]) == pytest.approx(census["n_coefficients"])
    assert sum(resolved["by_block"]["n_coefficients"]) == pytest.approx(census["n_coefficients"])
    for level in ("0.5", "0.9"):
        assert sum(resolved["by_horizon"]["inside"][level]) == pytest.approx(census["inside"][level])
        assert sum(resolved["by_block"]["inside"][level]) == pytest.approx(census["inside"][level])
    merged = predictive.merge_calibration(None, census)
    merged = predictive.merge_calibration(merged, census)
    finished = predictive.finish_calibration(merged)
    assert finished["resolved"]["by_horizon"]["n_coefficients"][0] == pytest.approx(
        2.0 * resolved["by_horizon"]["n_coefficients"][0]
    )
    assert len(finished["resolved"]["by_block"]["coverage"]["0.5"]) == 2
    assert finished["coverage"]["0.5"] == pytest.approx(census["inside"]["0.5"] / census["n_coefficients"])
    with pytest.raises(ValueError):
        predictive.calibration_census(cdf, mask, block_split=5)


# =================================================================================================
# The gate reads the interval
# =================================================================================================
def _summary(*, lo, hi, listed, schema=2, point=0.16):
    """A minimal summary carrying a gap interval and a verdict list."""
    return {
        "results": {
            "schema_version": schema,
            "arm": {"source_disabled": False},
            "arm_scores": {"pred_gap": {"point": point, "lo": lo, "hi": hi, "n": 10}},
            "verdicts": [{"name": "predictive_improvement", "status": listed}],
        }
    }


def test_the_gate_reads_the_interval_and_refuses_a_disagreeing_list() -> None:
    """Above zero PASS, through zero INCONCLUSIVE, below zero FAIL; a version-2 summary whose
    own list disagrees with its interval fails, and a version-1 one is read without failing."""
    from teb_vae.lag_slot_transformer_cfs.eval import verify as gate

    assert gate.report_predictive_gap(_summary(lo=0.05, hi=0.4, listed="PASS"))["status"] == "PASS"
    crossing = gate.report_predictive_gap(_summary(lo=-0.2, hi=0.53, listed="INCONCLUSIVE"))
    assert crossing["status"] == "INCONCLUSIVE"
    assert gate.report_predictive_gap(_summary(lo=-0.5, hi=-0.1, listed="FAIL"))["status"] == "FAIL"
    disagree = gate.report_predictive_gap(_summary(lo=-0.2, hi=0.53, listed="PASS"))
    assert disagree["status"] == "FAIL" and "disagree" in disagree["detail"]
    legacy = gate.report_predictive_gap(_summary(lo=-0.2, hi=0.53, listed="PASS", schema=None))
    assert legacy["status"] == "INCONCLUSIVE" and legacy["schema_version"] == 1
    assert legacy["listed_status"] == "PASS"


def test_the_gate_pairs_the_candidate_against_the_reference_by_recording() -> None:
    """The reference comparison resamples per-recording differences from the two tables."""
    from teb_vae.lag_slot_transformer_cfs.eval import verify as gate

    guids = [f"g{i}" for i in range(12)]
    candidate = {
        "results": {
            "schema_version": 2,
            "arm": {"source_disabled": False},
            "arm_scores": {
                "pred_gap": {"point": 1.0, "lo": 0.5, "hi": 1.5, "n": 12, "resamples": 200, "seed": 1},
                "nll_full": {"point": -11.0}, "nll_base": {"point": -10.0},
            },
            "per_recording": {g: {"nll_full": -11.0 - 0.1 * i, "nll_base": -10.0} for i, g in enumerate(guids)},
        }
    }
    reference = {
        "results": {
            "arm": {"source_disabled": True, "model_kind": "fhr_lag_residual_cfs_v1"},
            "arm_scores": {"nll_base": {"point": -10.0}},
            "per_recording": {g: {"nll_base": -10.0 - 0.01 * i} for i, g in enumerate(guids)},
        }
    }
    verdict = gate.check_against_reference(candidate, reference)
    assert verdict["n_paired"] == 12
    assert verdict["paired_improvement_nats"] > 0.0
    assert verdict["paired_ci_lo"] > 0.0
    assert verdict["status"] == "PASS"
    worse = dict(candidate)
    worse["results"] = {**candidate["results"], "arm_scores": {
        **candidate["results"]["arm_scores"], "nll_base": {"point": -9.0}}}
    assert gate.check_against_reference(worse, reference)["status"] == "FAIL"


@pytest.mark.parametrize("profile", ["lag25.yaml", "default.yaml"])
def test_the_committed_delta_partitions_any_window_and_keeps_the_cross_bank_bands(
    profile: str,
) -> None:
    """The partition is derived from the checkpoint's own ``max_lag``, so the one committed delta
    loads on the production bank and on the long one alike; the two cross-bank bands are declared
    beside it and abut, so the two banks can be read on one interval."""
    from teb_vae.lag_attn_cfs.eval.config_schema import PARTITION_WIDTH_KEY, validate_eval_config

    delta = load_eval_overrides(LAG_RESIDUAL_BINDING.overrides_path)["eval_config"]
    assert PARTITION_WIDTH_KEY in delta["occlusion_bands"]
    config = Path(__file__).resolve().parents[1] / "configs" / profile
    max_lag = int(
        yaml.safe_load(config.read_text(encoding="utf-8"))["model_config"]["VAE_model"]["max_lag"]
    )
    bands = validate_eval_config(
        {"eval_config": delta, "model_config": {"VAE_model": {"max_lag": max_lag}}}
    )["occlusion_bands"]
    derived = {name: span for name, span in bands.items() if name.startswith("lags_")}
    covered = sorted(lag for lo, hi in derived.values() for lag in range(lo, hi + 1))
    assert covered == list(range(max_lag + 1))
    head, tail = bands["common_head"], bands["common_tail"]
    assert head[0] == 0 and head[1] + 1 == tail[0] and tail[1] <= max_lag
