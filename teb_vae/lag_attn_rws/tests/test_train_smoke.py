r"""One real fit, through the real framework, against the committed shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
data module -> model -> ``build_trainer`` -> ``fit`` -> checkpoint, on a CPU, in seconds. It is
the only test that can catch the failures that live *between* the pieces -- a metric key no
callback collects, a ``None`` in a metrics dict, a checkpoint that will not reload, a callback
constructed with the wrong keyword.

The invariant worth naming: at step 0 the KL is exactly zero, because the posterior's delta
heads are zero-initialised and the posterior therefore *is* the prior. Asserting it here rather
than on a bare model is the point -- it is the one place the property is checked after config
resolution, kwarg sweeping, data normalization and the framework's own seeding have all had a
chance to break it.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"


def _run_fit(tmp_path, *, causal_reach_budget_s=None):
    """Run one real fit against the committed shard and return the driver and its trainer.

    Args:
        tmp_path: Directory to run in.
        causal_reach_budget_s: The reach budget, in seconds, or ``None`` for the shipped
            unguarded default.

    Returns:
        ``(driver, trainer)``.
    """
    config = load_config(str(_TINY))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = 2
    config["model_config"]["VAE_model"]["causal_reach_budget_s"] = causal_reach_budget_s
    absolutize_dataset_paths(config)
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from train.data_module import GraphDataModule

    driver = LagAttnRwsTrainer(config_file_path=str(config_path))
    driver.setup_config()
    data_module = GraphDataModule(driver.config)
    driver.create_model()
    trainer = driver.train_model(data_module.train_dataloader(), data_module.val_dataloader())
    return driver, trainer


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """Run one real fit at the shipped (unguarded) configuration.

    Module-scoped: this is the expensive test in the suite, and every assertion below is a
    different question about the same run. Two epochs, not the config's one: ``lr`` is logged
    at train-epoch *start* with ``on_epoch=True``, so its first CSV cell is always NaN, and the
    second epoch is also what exercises the LR scheduler stepping at all.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke"))


@pytest.fixture(scope="module")
def guarded_fit(tmp_path_factory):
    """The same fit with the causal input guard on, at the $120$ s reach budget.

    A second full fit rather than an assertion on the first, because what it exercises is only
    reachable end to end: the budget resolving from config, the four channel tuples surviving
    the kwarg sweep into narrower input adapters, the delayed stream reaching the encoders, and
    the resolved delay vector being written into the run's own configuration.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke_guarded"), causal_reach_budget_s=120.0)


def test_the_fit_completes(fit):
    driver, trainer = fit

    assert trainer.current_epoch == 2
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    driver, trainer = fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch
    the KL is legitimately nonzero, and the question is whether the model the config builds
    starts at zero.

    The KL half holds in every configuration, because it is a function of the two *distributions*.
    The forecast half depends on ``base_decode``, and the assertion is split accordingly rather
    than relaxed to a tolerance for both:

    * ``'sample'`` -- the shared $\epsilon$ makes $z^p = z^q$ elementwise, so the two forecasts
      are bitwise identical.
    * ``'mean'`` (shipped) -- the base branch decodes $\mu^p$ while the full branch still samples,
      so the two differ by the posterior's own noise. What must hold instead is that the base
      forecast **is** the decode of $\mu^p$, which is the property that makes $D_0$ noise-free.
    """
    driver, _ = fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    kwargs = driver._build_model_kwargs()
    model = SeqVaeLagAttnRws(**kwargs).eval()
    batch_size, seq_len = 2, model.sequence_length
    generator = torch.Generator().manual_seed(0)
    outputs = model(
        torch.randn(batch_size, seq_len, 43, generator=generator),
        torch.randn(batch_size, seq_len, 66, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
    )

    assert float(outputs["kld_per_t"].abs().max()) < 1e-6

    if model.base_decode == "sample":
        assert torch.equal(outputs["mu_base"], outputs["mu_full"])
    else:
        assert torch.equal(outputs["z_prior"], outputs["mu_prior"])
        expected_base, _ = model.decoder(outputs["mu_prior"][:, : model.geometry.t_valid])
        assert torch.equal(outputs["mu_base"], expected_base)
        # Non-vacuous: the full branch must still be somewhere else, or this would pass on a
        # model whose posterior had stopped sampling.
        assert not torch.equal(outputs["mu_base"], outputs["mu_full"])


def test_the_metrics_history_csv_has_no_all_nan_column(fit):
    """The check that catches a tracked name the framework never emits: the column appears in
    every run's CSV, NaN in every row, and nothing anywhere reports it."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_configured_objective_weights_reach_the_csv_and_their_terms_are_live(fit):
    """Every objective weight the config sets, echoed back through a real fit, and every weighted
    term nonzero -- under the tiny config's ``mse`` likelihood, where a term emitted only under
    ``gaussian_nll`` would silently produce an all-NaN column.

    The echo is compared against the config rather than a constant, so a weight dropped at the
    kwarg sweep or lost in the task's forwarding shows up as the driver's $0.0$ fallback. Nonzero
    is not implied by the column existing: a term whose weight resolves to $0.0$ is deliberately
    **not computed** and reports an exact $0.0$, a well-formed all-zero column and no error
    anywhere. ``mse`` is the right likelihood to check this under, not a limitation: the shape
    terms read the forecast *means* only, so they are identical under either likelihood.
    """
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")
    vae = driver.config["model_config"]["VAE_model"]

    for stage in ("train", "val"):
        for term, weight in (
            ("prior_rate", "beta_prior"),
            ("aux_multiscale", "lambda_ms"),
            ("aux_derivative", "lambda_deriv"),
            ("aux_boundary", "lambda_boundary"),
        ):
            assert float(vae[weight]) > 0.0, f"the tiny config does not weight {term}"
            echoed = frame[f"{stage}/{weight}"].dropna()
            assert float(echoed.iloc[0]) == pytest.approx(float(vae[weight])), echoed.name
            values = frame[f"{stage}/{term}"].dropna()
            assert not values.empty and bool(values.abs().lt(float("inf")).all()), values.name
            assert float(values.abs().max()) > 0.0, (
                f"{values.name} is zero in every epoch: its weight never reached the objective"
            )


def test_the_checkpoint_carries_its_contract_and_reloads(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file,
    through the repository's own loading helpers."""
    driver, _ = fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnRws"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttnRws")
    rebuilt = SeqVaeLagAttnRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )


def test_the_validation_figure_is_written_by_a_real_fit(fit):
    """The plotting callback driven by a real trainer and a real loader rather than by a fake.

    Everything the unit tests cannot reach meets here: the batch actually coming off the HDF5
    loader, the normalization statistics actually being reachable through it, and the callback
    surviving a Lightning validation epoch. The callback swallows its own exceptions by design,
    so a broken figure is silent everywhere except in this file count.
    """
    driver, _ = fit

    figures = list(
        (Path(driver.train_results_dir) / "lag_attn_rws_diagnostics").glob("*.pdf")
    )

    assert figures, "the enabled plotting callback wrote no figure"


def test_a_fit_completes_under_the_causal_reach_budget(guarded_fit):
    """The whole guard, end to end: config to filter bank to channel tuples to a trained model.

    A unit test can check each link; only a fit can check that they are connected -- most
    concretely that the narrowed adapters and the full declared ``c_y``/``c_u`` coexist, since
    the data boundary validates the batch against the declared widths while the model reads only
    the survivors.
    """
    driver, trainer = guarded_fit
    model = driver.pytorch_model
    vae = driver.config["model_config"]["VAE_model"]
    budget = resolve_stream_budgets(vae)

    assert trainer.current_epoch == 2
    assert trainer.state.finished
    assert budget is not None
    assert model.target_adapter.linear.in_features == len(budget.target_keep_index)
    assert model.source_adapter.linear.in_features == len(budget.source_keep_index)
    assert model.target_gate.max_delay == max(budget.target_delays)
    assert model.source_delay_steps == max(budget.source_delays) > 0
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (model.c_y, model.c_u) == (vae["c_y"], vae["c_u"])


def test_the_guarded_runs_losses_stay_finite(guarded_fit):
    """A delayed stream is zero for its first max(delta) steps; those steps must fall inside the
    warm-up rather than reaching the loss as a block of zeros.

    ``train/grad_norm`` is excluded and has its own test below: under the guard the *losses* are
    finite while the gradient is not, and collapsing the two would report the gradient defect
    under a name that says "losses" -- or, worse, invite someone to make it pass by relaxing the
    loss check.
    """
    _, trainer = guarded_fit

    for name, value in trainer.callback_metrics.items():
        if name.startswith("train/grad_norm"):
            continue
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


#: What the guarded gradient norm must stay under. Not a tolerance -- a *decision boundary*
#: between the two regimes below, set orders of magnitude above the measured value and orders of
#: magnitude below the defect it replaces, so it can only be crossed by the defect returning.
_GUARDED_GRAD_NORM_CEILING = 1e6


def test_the_guarded_runs_gradient_stays_finite_and_small(guarded_fit):
    r"""The gradient the guarded arms actually optimise with.

    This was a ``strict`` xfail recording a real defect, and it is now a positive assertion
    because the defect is fixed. What it was:

    Every finite ``causal_reach_budget_s`` zero-fills the delayed prefix, and at step $0$ *every*
    surviving channel is zero -- the fastest survivor is already one step stale. That all-zero
    vector is zero-variance input to the adapter norm and to each causal conv pre-norm, and the
    $1/\sqrt{\epsilon} = 316\times$ backward amplification of ~$10$ stacked norms compounds:
    measured in float64 on this fixture, the global gradient norm reached ~$10^{26}$ at *every*
    budget ($32/60/120/240$ s) against ~$98$ unguarded, and overflowed fp32 to ``inf``. It did not
    scale with ``max_delay`` -- one all-zero step was enough -- so raising ``warmup_period`` did
    not help, and at ``gradient_clip_val=250`` the clip coefficient was ~$10^{-24}$: every
    reach-arm step was scaled to nothing and the arm trained AdamW's weight decay while completing
    normally.

    What fixed it: the adapters are now ``AvailabilityInputAdapter``, which adds
    $W_m(m_t - \mathbf 1)$ and a start embedding, so "this position is empty" is a representation
    the model is told about rather than a zero-variance accident. Measured through the real driver
    on the committed shard at the $120$ s budget, the guarded gradient norm is ~$135$ against
    ~$125$ for the transformer sibling -- the same order of magnitude as an unguarded run.

    Finiteness alone would not catch a regression: the defect's $10^{26}$ is finite in float64.
    The ceiling is what separates the two regimes.
    """
    _, trainer = guarded_fit

    grad_norm = float(trainer.callback_metrics["train/grad_norm"])

    assert math.isfinite(grad_norm), f"guarded train/grad_norm is {grad_norm}"
    assert grad_norm < _GUARDED_GRAD_NORM_CEILING, (
        f"guarded train/grad_norm is {grad_norm:.3g}, past the {_GUARDED_GRAD_NORM_CEILING:g} "
        f"ceiling -- the zero-prefix amplification this adapter exists to remove is back"
    )


def test_the_guarded_checkpoint_rebuilds_at_its_own_channel_widths(guarded_fit):
    """The adapters' widths depend on the resolved budget, so a checkpoint that recorded only
    the budget in seconds could not be rebuilt without re-running the resolution. The four
    channel tuples are therefore in ``model_kwargs``."""
    driver, _ = guarded_fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_kwargs"] == driver._build_model_kwargs()
    assert blob["model_kwargs"]["target_keep_index"] is not None
    rebuilt = SeqVaeLagAttnRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None
