r"""One real fit, through the real entry point, against the committed shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a schedule
attached at the wrong interval, a callback that raises on the first validation epoch, a diagnostic
figure that fails to draw, or a front end that receives no gradient at all.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the inherited
pre-flight guards, this package's own three and the temporary resolved-config file all hang off the
entry point and are reached no other way.

There is no evaluation pipeline for this architecture yet, which changes what this file is for.
``metrics_history.csv``, the tracked metric surface, ``train/grad_norm`` and the per-epoch
diagnostic figure are the **only** readout a run of this model produces, so the assertions that
those four exist and are populated are not hygiene -- they are the check that a multi-day production
run will have produced something readable at the end of it.
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
from teb_vae.lag_attn_transformer_e2e import trainer as trainer_module
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import absolutize_dataset_paths

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. Three rather than the config's one, for two reasons that are both about what
#: only a multi-epoch run can show: ``lr`` is logged at train-epoch *start* with ``on_epoch=True``,
#: so its first CSV cell is always NaN, and the step warm-up needs more than one epoch's worth of
#: steps to be visibly non-constant at epoch granularity.
SMOKE_EPOCHS = 3


def _run_fit(tmp_path):
    """Run one real fit through the entry point and return what it built.

    Args:
        tmp_path: Directory the run writes into.

    Returns:
        ``(driver, trainer)`` -- the driver and its fitted Lightning ``Trainer``.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original_train_model = LagAttnTrfE2ETrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnTrfE2ETrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnTrfE2ETrainer.train_model

    return captured["driver"], captured["trainer"]


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """One real fit at the shipped smoke configuration.

    Module-scoped: this is the expensive test in the suite, and every assertion below is a
    different question about the same run.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke"))


# --------------------------------------------------------------------------------------
# The fit itself
# --------------------------------------------------------------------------------------
def test_the_fit_completes(fit):
    _, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    _, trainer = fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_gradient_norm_is_finite_and_non_zero(fit):
    r"""The two real failures this column can show, and the reason it is not compared against the
    comparison model's smoke value.

    Non-finite means the clip coefficient is zero and the run trains nothing while completing
    normally. Exactly zero means no parameter received a gradient at all -- which for this
    architecture is the specific failure worth watching, since the two front ends are a gradient
    path neither sibling has.

    Deliberately *not* compared against the sibling's number: the gradient scale through a new
    front-end path is unknown, ``gradient_clip_val`` is inherited and marked provisional for exactly
    that reason, and it is re-derived from this metric on the first production run. A comparison
    here could fail a correct implementation and would carry no information either way.
    """
    driver, trainer = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    observed = [float(value) for value in frame["train/grad_norm"].dropna().tolist()]
    observed.append(float(trainer.callback_metrics["train/grad_norm"]))

    assert len(observed) > 1, "train/grad_norm was never logged, so this proves nothing"
    assert all(math.isfinite(value) for value in observed), observed
    assert all(value > 0.0 for value in observed), observed


def test_every_front_end_parameter_moved_during_the_fit(fit):
    """The end-to-end form of the DDP-reachability claim: not merely that a gradient existed, but
    that the optimizer actually applied it. A front end that trained nothing would leave every
    downstream number looking exactly like a run of a model with a frozen input stage."""
    driver, _ = fit
    trained = driver.pytorch_model
    fresh = SeqVaeLagAttnTrfE2E(**driver._build_model_kwargs())

    fresh_state = fresh.state_dict()
    unmoved = [
        name
        for name, tensor in trained.state_dict().items()
        if name.startswith(("target_frontend.", "source_frontend."))
        and torch.equal(tensor.detach().cpu(), fresh_state[name])
    ]

    assert unmoved == [], (
        f"front-end tensors identical to a fresh model after {SMOKE_EPOCHS} epochs: {unmoved}"
    )


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it.
    """
    driver, _ = fit
    model = SeqVaeLagAttnTrfE2E(**driver._build_model_kwargs()).eval()
    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, model.sequence_length
    raw_len = seq_len * model.raw_per_step
    outputs = model(
        torch.randn(batch_size, raw_len, generator=generator),
        torch.randn(batch_size, raw_len, generator=generator),
        torch.ones(batch_size, seq_len),
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


# --------------------------------------------------------------------------------------
# The readout: the only one this architecture has
# --------------------------------------------------------------------------------------
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
    """Evidence that the step warm-up configured in ``tiny.yaml`` actually ran -- which is what
    catches the wrong-parent inheritance error at the level of a real fit. A ramp silently attached
    at epoch granularity, or never attached at all, produces a flat column here while every other
    assertion in this file still passes."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    observed = frame["lr"].dropna().tolist()

    assert len(set(observed)) > 1, f"the learning rate never moved: {observed}"
    # And it moved *upwards*: the ramp is a warm-up, not the milestone decay, which cannot have
    # fired in three epochs against milestones at 400 and 800.
    assert observed[-1] > observed[0]


def test_the_validation_figures_are_written_by_a_real_fit(fit):
    """The plotting callback driven by a real trainer and a real loader rather than by a fake.

    Everything the unit tests cannot reach meets here: the batch actually coming off the HDF5
    loader, the normalization statistics actually being reachable through it, and the callback
    surviving a Lightning validation epoch through this model's own ``_build_forward_inputs``. The
    callback swallows its own exceptions by design, so a broken figure is silent everywhere except
    in this file count.
    """
    driver, _ = fit

    directory = Path(driver.train_results_dir) / "lag_attn_rws_diagnostics"
    figures = list(directory.glob("lag_attn_rws_epoch*.pdf"))

    # One per requested example per plotted epoch, at plot_frequency 1.
    assert len(figures) == 2 * SMOKE_EPOCHS, [path.name for path in figures]
    # And no input figures at all: those describe the stored scattering and phase-harmonic
    # channels, and this model declares no `c_y` because it is handed the raw traces instead.
    # The absence is the designed behaviour, not a failure the callback swallowed.
    assert list(directory.glob("causal_input_budget.*")) == []


# --------------------------------------------------------------------------------------
# The checkpoint
# --------------------------------------------------------------------------------------
def test_the_checkpoint_carries_its_contract_and_reloads(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file,
    through the repository's own loading helpers. The front ends' fixed anti-alias filters are
    non-persistent, so the strict load has to align without them."""
    driver, _ = fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnTrfE2E"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttnTrfE2E")
    rebuilt = SeqVaeLagAttnTrfE2E(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )
