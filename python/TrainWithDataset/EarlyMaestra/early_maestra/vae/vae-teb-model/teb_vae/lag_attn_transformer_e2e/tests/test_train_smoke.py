r"""One real fit, through the real entry point, against the committed shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a schedule
attached at the wrong interval, a callback that raises on the first validation epoch, a diagnostic
figure that fails to draw, or a front end that receives no gradient at all.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the four
inherited pre-flight guards, this package's own three, the temporary resolved-config file and the
resolved-config write beside the checkpoints all hang off the entry point and are reached no other
way.

There is no evaluation pipeline for this architecture yet, which changes what this file is for.
``metrics_history.csv``, the tracked metric surface, ``train/grad_norm`` and the per-epoch
diagnostic figure are the **only** readout a run of this model produces, so the assertions that
those four exist and are populated are not hygiene -- they are the check that a multi-day production
run will have produced something readable at the end of it.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws import plotting as plotting_module
from teb_vae.lag_attn_rws import sample_page
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, RESOLVED_CONFIG_FILENAME
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


def _figure_outs_keys() -> set:
    """The forward-dict keys the diagnostic figure builder indexes, read off its own source.

    Derived rather than listed, so a key the figure starts reading is covered here without anything
    being updated -- and a forward that stopped exporting one is caught by the fit rather than by a
    swallowed exception inside the callback.

    Returns:
        The key names.
    """
    source = Path(sample_page.__file__).read_text(encoding="utf-8")
    return {
        name or fallback
        for name, fallback in re.findall(
            r"outs\[['\"]([a-z_]+)['\"]\]|outs\.get\(['\"]([a-z_]+)['\"]", source
        )
    }


def _run_fit(tmp_path):
    """Run one real fit through the entry point and return what it built.

    Args:
        tmp_path: Directory the run writes into.

    Returns:
        ``(driver, trainer, figure_calls)`` -- the driver, its fitted Lightning ``Trainer``, and one
        recorded ``outs`` key set per diagnostic figure the run drew.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    figure_calls = []
    original_train_model = LagAttnTrfE2ETrainer.train_model
    original_builder = plotting_module.build_diagnostic_figure

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    def _capture_builder(*args, **kwargs):
        figure_calls.append(set(kwargs["outs"]))
        return original_builder(*args, **kwargs)

    LagAttnTrfE2ETrainer.train_model = _capture_train_model
    plotting_module.build_diagnostic_figure = _capture_builder
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnTrfE2ETrainer.train_model
        plotting_module.build_diagnostic_figure = original_builder

    return captured["driver"], captured["trainer"], figure_calls


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
    _, trainer, _ = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    _, trainer, _ = fit

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
    driver, trainer, _ = fit
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
    driver, _, _ = fit
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
    driver, _, _ = fit
    model = SeqVaeLagAttnTrfE2E(**driver._build_model_kwargs()).eval()
    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, model.sequence_length
    raw_len = seq_len * model.raw_per_step
    outputs = model(
        torch.randn(batch_size, raw_len, generator=generator),
        torch.randn(batch_size, raw_len, generator=generator),
        torch.ones(batch_size, seq_len),
    )

    assert float(outputs["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(outputs["mu_base"], outputs["mu_full"])


# --------------------------------------------------------------------------------------
# The readout: the only one this architecture has
# --------------------------------------------------------------------------------------
def test_every_declared_metric_reaches_the_logger(fit):
    """The gap between "the task emits it" and "a callback collected it" is silent otherwise. The
    shuffled readouts are the ones only a real validation loop can prove wired."""
    _, trainer, _ = fit

    for name in (
        "train/total_loss",
        "train/main_loss",
        "train/grad_norm",
        "train/source_conditioned_kl_raw",
        "val/total_loss",
        "val/nll_shuffled_block",
        "val/kld_shuffled",
        "val/shuffle_penalty",
    ):
        assert name in trainer.callback_metrics, f"{name} never reached callback_metrics"


def test_the_metrics_csv_carries_every_tracked_key_and_no_all_nan_column(fit):
    """Both halves of the tracked list's contract, on a real run: a name the framework never emits
    is a column that is NaN in every row of every run, and a tracked name that produced no column
    at all is a readout nothing ever recorded."""
    driver, _, _ = fit
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
    driver, _, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    observed = frame["lr"].dropna().tolist()

    assert len(set(observed)) > 1, f"the learning rate never moved: {observed}"
    # And it moved *upwards*: the ramp is a warm-up, not the milestone decay, which cannot have
    # fired in three epochs against milestones at 400 and 800.
    assert observed[-1] > observed[0]


def test_the_scheduled_beta_reaches_the_csv(fit):
    """The resolved schedule value, which starts at exactly zero -- the posterior-collapse guard
    the config documents."""
    driver, _, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    assert "train/kld_beta" in frame.columns
    assert float(frame["train/kld_beta"].iloc[0]) == pytest.approx(0.0)


def test_the_run_directory_has_the_expected_layout(fit):
    """The log sinks, the checkpoint directory and the resolved config a later offline pass needs."""
    driver, _, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()
    assert (Path(driver.train_results_dir) / "metrics_history.csv").is_file()
    assert Path(driver.model_checkpoint_dir).is_dir()


def test_the_resolved_config_is_written_beside_the_checkpoints(fit):
    """A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive."""
    driver, _, _ = fit

    written = Path(driver.model_checkpoint_dir) / RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    assert reloaded["model_config"]["VAE_model"]["target_attention_blocks"] == 4
    assert reloaded["dataset_config"]["dataloader_config"]["normalize_fields"] == ["fhr", "up"]


def test_the_validation_figures_are_written_by_a_real_fit(fit):
    """The plotting callback driven by a real trainer and a real loader rather than by a fake.

    Everything the unit tests cannot reach meets here: the batch actually coming off the HDF5
    loader, the normalization statistics actually being reachable through it, and the callback
    surviving a Lightning validation epoch through this model's own ``_build_forward_inputs``. The
    callback swallows its own exceptions by design, so a broken figure is silent everywhere except
    in this file count.
    """
    driver, _, _ = fit

    figures = list((Path(driver.train_results_dir) / "lag_attn_rws_diagnostics").glob("*.pdf"))

    # One per requested example per plotted epoch, at plot_frequency 1.
    assert len(figures) == 2 * SMOKE_EPOCHS, [path.name for path in figures]


def test_the_figure_builder_receives_every_key_it_reads(fit):
    """The figure reads several keys off the forward dict, and the callback swallows the
    ``KeyError`` a missing one would raise -- so without this the failure mode is a run that
    silently draws nothing."""
    _, _, figure_calls = fit
    needed = _figure_outs_keys()

    assert needed, "the key scan found nothing; the figure builder's source changed shape"
    assert figure_calls, "the figure builder was never called"
    for keys in figure_calls:
        assert needed <= keys, f"the forward dict is missing {sorted(needed - keys)}"


# --------------------------------------------------------------------------------------
# The checkpoint
# --------------------------------------------------------------------------------------
def test_the_checkpoint_is_written_under_this_models_stem(fit):
    """Three architectures writing ``lag-attn-rws-epoch=00.ckpt`` would be indistinguishable by
    name, and the stem is the one string this package supplies to the inherited callback
    assembly."""
    driver, _, _ = fit

    checkpoints = list(Path(driver.model_checkpoint_dir).glob("*.ckpt"))

    assert checkpoints, "no checkpoint was written; Lightning's default would have gone elsewhere"
    assert all(path.name.startswith("lag-attn-trf-e2e-epoch=") for path in checkpoints), [
        path.name for path in checkpoints
    ]


def test_the_checkpoint_carries_its_contract_and_reloads(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file,
    through the repository's own loading helpers. The front ends' fixed anti-alias filters are
    non-persistent, so the strict load has to align without them."""
    driver, _, _ = fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnTrfE2E"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttnTrfE2E")
    rebuilt = SeqVaeLagAttnTrfE2E(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )
