r"""Two real fits, through the real entry point, against the committed shard.

Everything else in this suite tests a piece in isolation. These run the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a schedule
attached at the wrong interval, a callback that raises on the first validation epoch, a diagnostic
figure that fails to draw.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the four
pre-flight guards, the temporary resolved-config file and the resolved-config write beside the
checkpoints hang off the entry point and are reached no other way.

The second fit is the one that decides whether the availability representation worked. The model
this one is compared against records, as a strict expected failure, that *every* finite reach
budget there drives the global gradient norm to about $10^{26}$: the delayed prefix is zero-filled,
at step $0$ every surviving channel is zero, and that zero-variance vector enters a stack of
normalisers whose backward amplifies by $1/\sqrt{\epsilon}$ each. The availability projection and
the start embedding exist to make that prefix representable instead. Whether they do is not
arguable from the architecture; it is measured here.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS
from teb_vae.lag_attn_transformer_rws import trainer as trainer_module
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import absolutize_dataset_paths

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"
#: Epochs each fit runs. Three rather than the config's one, for two reasons that are both about
#: what only a multi-epoch run can show: ``lr`` is logged at train-epoch *start* with
#: ``on_epoch=True``, so its first CSV cell is always NaN, and the step warm-up needs more than one
#: epoch's worth of steps to be visibly non-constant at epoch granularity.
SMOKE_EPOCHS = 3

#: The reach budget the guarded fit runs at. Its resolved delay is exactly $30$ steps, which is the
#: shipped ``warmup_period`` -- the resolution refuses anything deeper, so this is the largest
#: admissible budget and therefore the hardest case for the availability representation.
GUARDED_BUDGET_S = 120.0


def _run_fit(tmp_path, *, causal_reach_budget_s=None):
    """Run one real fit through the entry point and return what it built.

    Args:
        tmp_path: Directory the run writes into.
        causal_reach_budget_s: The reach budget in seconds, or ``None`` for the shipped unguarded
            default.

    Returns:
        ``(driver, trainer)`` -- the driver and its fitted Lightning ``Trainer``.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    config["model_config"]["VAE_model"]["causal_reach_budget_s"] = causal_reach_budget_s
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original_train_model = LagAttnTrfRwsTrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnTrfRwsTrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnTrfRwsTrainer.train_model

    return captured["driver"], captured["trainer"]


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """One real fit with the causal input guard off.

    Module-scoped: this is the expensive test in the suite, and every assertion below is a different
    question about the same run.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke"))


@pytest.fixture(scope="module")
def guarded_fit(tmp_path_factory):
    """The same fit with the causal input guard on, at the $120$ s reach budget.

    A second full fit rather than an assertion on the first, because what it exercises is only
    reachable end to end: the budget resolving from config, the four channel tuples surviving the
    kwarg sweep into narrower input adapters, the availability terms being constructed from the
    resolved delays, and the zero-filled prefix reaching the encoders through them.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke_guarded"), causal_reach_budget_s=GUARDED_BUDGET_S)


# --------------------------------------------------------------------------------------
# The unguarded fit
# --------------------------------------------------------------------------------------
def test_the_fit_completes(fit):
    _, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    _, trainer = fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it.
    """
    driver, _ = fit
    model = SeqVaeLagAttnTrfRws(**driver._build_model_kwargs()).eval()
    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, model.sequence_length
    outputs = model(
        torch.randn(batch_size, seq_len, 43, generator=generator),
        torch.randn(batch_size, seq_len, 66, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
    )

    # The KL half holds in every configuration, because it is a function of the two
    # *distributions* -- including under the shipped independent posterior log-variance head,
    # which head_init_calibration pins to the prior's own constant.
    assert float(outputs["kld_per_t"].abs().max()) == 0.0

    # The forecast half depends on base_decode, so it is split rather than relaxed for both.
    if model.base_decode == "sample":
        assert torch.equal(outputs["mu_base"], outputs["mu_full"])
    else:
        # Shipped: the base branch decodes mu^p while the full branch still samples, so the two
        # differ by the posterior's own noise. What must hold instead is that the base forecast
        # IS the decode of mu^p -- the property that makes D_0 noise-free.
        assert torch.equal(outputs["z_prior"], outputs["mu_prior"])
        expected_base, _ = model.decoder(outputs["mu_prior"][:, : model.geometry.t_valid])
        assert torch.equal(outputs["mu_base"], expected_base)
        assert not torch.equal(outputs["mu_base"], outputs["mu_full"])


def test_the_metrics_csv_carries_every_tracked_key_and_no_all_nan_column(fit):
    """Both halves of the tracked list's contract, on a real run: a name the framework never emits
    is a column that is NaN in every row of every run, and a tracked name that produced no column
    at all is a readout nothing ever recorded."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    missing = [name for name in _TRACKED_METRICS if name not in frame.columns]
    assert missing == [], f"tracked but never written to the CSV: {missing}"
    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_logged_learning_rate_is_non_constant(fit):
    """Evidence that the step warm-up configured in ``tiny.yaml`` actually ran. A ramp that was
    silently attached at epoch granularity, or never attached at all, produces a flat column here
    while every other assertion in this file still passes."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    observed = frame["lr"].dropna().tolist()

    assert len(set(observed)) > 1, f"the learning rate never moved: {observed}"
    # And it moved *upwards*: the ramp is a warm-up, not the milestone decay, which cannot have
    # fired in three epochs against milestones at 400 and 800.
    assert observed[-1] > observed[0]


def test_the_checkpoint_carries_its_contract_and_reloads(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file,
    through the repository's own loading helpers."""
    driver, _ = fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnTrfRws"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttnTrfRws")
    rebuilt = SeqVaeLagAttnTrfRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )


def test_the_validation_figures_are_written_by_a_real_fit(fit):
    """The plotting callback driven by a real trainer and a real loader rather than by a fake.

    Everything the unit tests cannot reach meets here: the batch actually coming off the HDF5
    loader, the normalization statistics actually being reachable through it, and the callback
    surviving a Lightning validation epoch. The callback swallows its own exceptions by design, so a
    broken figure is silent everywhere except in this file count.
    """
    driver, _ = fit

    directory = Path(driver.train_results_dir) / "lag_attn_rws_diagnostics"
    figures = list(directory.glob("lag_attn_rws_epoch*.pdf"))

    # One per requested example per plotted epoch, at plot_frequency 1.
    assert len(figures) == 2 * SMOKE_EPOCHS, [path.name for path in figures]
    # And the run-level causal-input-budget figure, exactly once however many epochs ran: it
    # describes the guard the model was built with, which no epoch changes.
    assert [path.name for path in directory.glob("causal_input_budget.*")] == [
        "causal_input_budget.pdf"
    ]


# --------------------------------------------------------------------------------------
# The guarded fit
# --------------------------------------------------------------------------------------
def test_a_fit_completes_under_the_causal_reach_budget(guarded_fit):
    """The whole guard, end to end: config to filter bank to channel tuples to a trained model.

    A unit test can check each link; only a fit can check that they are connected -- most concretely
    that the narrowed adapters and the full declared ``c_y``/``c_u`` coexist, since the data boundary
    validates the batch against the declared widths while the model reads only the survivors.
    """
    driver, trainer = guarded_fit
    model = driver.pytorch_model

    vae = driver.config["model_config"]["VAE_model"]

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished
    assert model.target_gate is not None and model.source_gate is not None
    assert model.source_delay_steps == model.source_gate.max_delay > 0
    assert model.target_adapter.linear.in_features == model.target_gate.out_channels
    assert model.source_adapter.linear.in_features == model.source_gate.out_channels
    # Narrowed below the declared widths, which stay untouched -- they are what the data boundary
    # checks the batch against.
    assert (model.c_y, model.c_u) == (vae["c_y"], vae["c_u"])
    assert model.target_gate.out_channels < model.c_y
    assert model.source_gate.out_channels < model.c_u


def test_the_guarded_runs_losses_stay_finite(guarded_fit):
    """A delayed stream is zero for its first $\\max_c \\delta_c$ steps; those steps must fall
    inside the warm-up rather than reaching the loss as a block of zeros."""
    _, trainer = guarded_fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_guarded_runs_gradient_stays_finite(guarded_fit):
    r"""The gradient the guarded arms actually optimise with, and the reason the availability
    representation exists.

    The model this one is compared against records a strict expected failure here: at every finite
    reach budget its global gradient norm is about $10^{26}$ and overflows float32 to infinity,
    because the zero-filled prefix is a zero-variance input to a stack of normalisers whose backward
    amplifies by $1/\sqrt{\epsilon} \approx 316$ each. With a clipping threshold of $250$ that makes
    the clip coefficient about $10^{-24}$: every step is scaled to nothing and the arm trains only
    the optimiser's weight decay, while completing normally and reporting finite losses. So finite
    losses are not evidence of anything here -- this column is.
    """
    driver, trainer = guarded_fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    observed = [float(value) for value in frame["train/grad_norm"].dropna().tolist()]
    observed.append(float(trainer.callback_metrics["train/grad_norm"]))

    assert len(observed) > 1, "train/grad_norm was never logged, so this proves nothing"
    worst = max(abs(value) for value in observed)
    assert all(math.isfinite(value) for value in observed), (
        f"guarded train/grad_norm is not finite at every logged step; largest magnitude "
        f"{worst:.3e} over {len(observed)} readings"
    )


def test_the_guarded_checkpoint_rebuilds_at_its_own_channel_widths(guarded_fit):
    """The adapters' widths depend on the resolved budget, so a checkpoint that recorded only the
    budget in seconds could not be rebuilt without re-running the resolution."""
    driver, _ = guarded_fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    rebuilt = SeqVaeLagAttnTrfRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None
    trained = driver.pytorch_model
    assert rebuilt.target_adapter.linear.in_features == trained.target_adapter.linear.in_features
    assert rebuilt.source_adapter.linear.in_features == trained.source_adapter.linear.in_features
    assert rebuilt.source_delay_steps == trained.source_delay_steps
