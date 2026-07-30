r"""Two evaluations of one checkpoint must report the same numbers, exactly.

The evaluation is not a deterministic function of the checkpoint by default. It *adds* randomness
of its own: $K$ Monte Carlo draws of $z$ per anchor, a derangement per batch, a shuffle over the
split, and the model's own reparameterisation inside every forward. Left on the global generator
those draws depend on whatever else in the process drew first, so re-running a checkpoint gives
different readouts and nothing in the output says why -- which quietly makes every comparison
between two runs a comparison of two samples.

Three things close that, and each is checked here rather than assumed: the numeric environment is
pinned and seeded from the configuration, the estimator takes an explicit generator instead of
the global one, and both the seed and the environment as actually read back from global state are
written into ``summary.json`` so a run is reproducible from its own output.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.eval._reuse import configure_numerics
from teb_vae.lag_attn_rws.eval.metrics import evaluate, mc_predictive_block
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

from .conftest import make_stub_batch

#: Any seed; the property under test is that re-running under the same one reproduces the run.
_SEED = 1234


class _StubLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


def _labelled_batches():
    """Two batches carrying recording identifiers, so the grouped control runs as it would."""
    first = make_stub_batch(batch_size=4, seed=0)
    first.guid = ["a", "a", "b", "b"]
    second = make_stub_batch(batch_size=4, seed=1)
    second.guid = ["c", "c", "d", "d"]
    return [first, second]


# =============================================================================
# The estimator's own randomness
# =============================================================================
@pytest.fixture
def scored_branch(task, inputs):
    """A callable scoring one latent branch under a seeded generator: ``score(seed)``."""
    module = task()
    model = module.orig_model
    y_st, y_ph, u_stream = inputs
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream)
    batch = make_stub_batch(batch_size=int(y_st.shape[0]), seed=0)
    fhr_raw, weight = module._build_raw_target(batch)
    mask, _coverage = forecast_mask(weight, model.geometry, coverage_floor=model.coverage_floor)
    target = build_future_target(fhr_raw, model.geometry, future_index=model.future_index)
    branches = {"base": (outputs["mu_prior"], outputs["logvar_prior"])}

    def _score(seed: int) -> torch.Tensor:
        scores, _contributing = mc_predictive_block(
            model, branches, target, mask, likelihood="gaussian_nll", num_samples=3,
            generator=torch.Generator().manual_seed(seed),
        )
        return scores["base"]

    return _score


def test_the_monte_carlo_draw_follows_its_generator_and_not_the_global_one(scored_branch):
    """The load-bearing half of determinism, isolated. The global generator is deliberately
    advanced by a different amount between the two calls, so a draw that still read from it
    would disagree; an equal result means the $\\epsilon$ came from the stream that was handed
    in.
    """
    torch.manual_seed(0)
    first = scored_branch(_SEED)
    torch.manual_seed(99)
    torch.randn(17)  # a different amount of global consumption before the second call
    second = scored_branch(_SEED)

    assert torch.equal(first, second)


def test_the_monte_carlo_draw_is_still_a_draw(scored_branch):
    """The other direction: two different seeds must disagree, or the equality above would hold
    on an estimator that had stopped sampling."""
    assert not torch.equal(scored_branch(_SEED), scored_branch(_SEED + 1))


# =============================================================================
# The whole evaluation
# =============================================================================
def test_two_evaluations_of_one_task_agree_exactly(task, perturb_posterior):
    """Not "to tolerance". Every readout, every verdict, every per-recording row: a difference of
    the last bits is still a difference, and a comparison between two arms cannot distinguish a
    real effect from sampling noise the pipeline introduced itself.

    Seeded through :func:`configure_numerics` between the two calls rather than by hand, because
    that is what the run does, and because the model's own reparameterisation draws from the
    global stream regardless of what the estimator does.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()

    def _once():
        configure_numerics(_SEED)
        return evaluate(
            module,
            _StubLoader(_labelled_batches()),
            num_samples=2,
            perm_generator=torch.Generator().manual_seed(_SEED),
            mc_generator=torch.Generator().manual_seed(_SEED),
        )

    assert _once() == _once()


def test_an_unseeded_rerun_is_what_makes_that_test_worth_running(task, perturb_posterior):
    """The evaluation is genuinely stochastic, so the equality above is a property of the seeding
    rather than of an evaluation that never draws anything."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()

    configure_numerics(_SEED)
    first = evaluate(module, _StubLoader(_labelled_batches()), num_samples=2)
    second = evaluate(module, _StubLoader(_labelled_batches()), num_samples=2)

    assert first["readouts"]["nll_full_block"] != second["readouts"]["nll_full_block"]


# =============================================================================
# What the summary records
# =============================================================================
def test_the_summary_records_the_seed_and_the_numeric_environment(evaluated):
    """A run must be reproducible from its own output, which means the seed is in the artifact
    rather than in the operator's shell history."""
    summary = evaluated["summary"]

    assert summary["numerics"]["seed"] == summary["eval_config"]["seed"]
    assert summary["numerics"]["torch_version"] == str(torch.__version__)


def test_the_recorded_numerics_are_the_two_settings_that_are_correctness_requirements(evaluated):
    """TF32 carries ten mantissa bits and the per-step KL is a small difference of larger
    quantities; ``cudnn.benchmark`` picks convolution algorithms by timing them, so the summation
    order -- and the last bits of the result -- depend on what else the machine was doing. Both
    are read back from global state, so a build where the assignment did not take shows up here
    as ``True``."""
    numerics = evaluated["summary"]["numerics"]

    assert numerics["cuda_matmul_allow_tf32"] is False
    assert numerics["cudnn_allow_tf32"] is False
    assert numerics["cudnn_benchmark"] is False
    assert numerics["float32_matmul_precision"] == "highest"


def test_a_second_run_of_the_same_checkpoint_reports_the_same_numbers(
    trained_run, repointed_overrides, tmp_path
):
    """The run-level property, end to end through the real loader: same checkpoint, same
    overrides, same seed, and every number in ``results`` identical. Only the paths and the
    timestamp differ, which is why the comparison is of that block rather than of the file.
    """
    import json

    from teb_vae.lag_attn_rws.eval import run as run_module

    def _results(directory):
        # ``main`` returns the process exit code, so an analysis failing is visible to a shell;
        # the summary sits where the explicit output directory says it does.
        run_module.main(
            trained_run, directory, overrides=repointed_overrides, device="cpu", num_samples=2
        )
        summary_path = (
            directory / run_module.RESULTS_DIRNAME / run_module.SUMMARY_FILENAME
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))["results"]

    assert _results(tmp_path / "first") == _results(tmp_path / "second")
