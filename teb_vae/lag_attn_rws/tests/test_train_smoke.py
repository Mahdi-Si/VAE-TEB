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

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"
_PACKAGE_DIR = _REPO_ROOT / "teb_vae" / "lag_attn_rws"


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
    """
    driver, _ = fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    model = SeqVaeLagAttnRws(**driver._build_model_kwargs()).eval()
    batch_size, seq_len = 2, model.sequence_length
    generator = torch.Generator().manual_seed(0)
    outputs = model(
        torch.randn(batch_size, seq_len, 43, generator=generator),
        torch.randn(batch_size, seq_len, 66, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
    )

    assert float(outputs["kld_per_t"].abs().max()) < 1e-6
    assert torch.equal(outputs["mu_base"], outputs["mu_full"])


def test_every_declared_metric_reaches_the_logger(fit):
    """The gap between "the task emits it" and "a callback collected it" is silent otherwise.
    The shuffled readouts are the ones only a real validation loop can prove wired."""
    driver, trainer = fit

    for name in (
        "train/total_loss",
        "train/main_loss",
        "train/source_conditioned_kl_raw",
        "val/total_loss",
        "val/nll_shuffled_block",
        "val/kld_shuffled",
        "val/shuffle_penalty",
    ):
        assert name in trainer.callback_metrics, f"{name} never reached callback_metrics"


def test_the_metrics_history_csv_has_no_all_nan_column(fit):
    """The check that catches a tracked name the framework never emits: the column appears in
    every run's CSV, NaN in every row, and nothing anywhere reports it."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_scheduled_beta_reaches_the_csv(fit):
    """The resolved schedule value, which starts at exactly zero -- the posterior-collapse
    guard the config documents."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    assert "train/kld_beta" in frame.columns
    assert float(frame["train/kld_beta"].iloc[0]) == pytest.approx(0.0)


def test_the_prior_anchor_and_clip_diagnostics_reach_the_csv(fit):
    """The prior scale rate, its echoed weight and the clip-exceedance fraction, through a real
    fit under the tiny config's ``mse`` likelihood -- the configuration where a term emitted only
    under ``gaussian_nll`` would silently produce an all-NaN column."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    for column in (
        "train/prior_rate", "val/prior_rate",
        "train/beta_prior", "val/beta_prior",
        "train/grad_clip_frac",
    ):
        assert column in frame.columns, f"{column} never reached the CSV"
        assert frame[column].notna().any(), f"{column} is NaN in every epoch"

    # The tiny config opts into the anchor, so the echoed weight is its constant -- proof the
    # config key reaches the objective rather than being silently dropped at the kwarg sweep.
    assert float(frame["train/beta_prior"].iloc[0]) == pytest.approx(0.1)
    # An epoch's clip fraction is the mean of a 0/1 per-step indicator.
    clip_frac = frame["train/grad_clip_frac"].dropna()
    assert bool(((clip_frac >= 0.0) & (clip_frac <= 1.0)).all())


def test_the_checkpoint_is_written_to_the_run_checkpoint_directory(fit):
    driver, _ = fit

    checkpoints = list(Path(driver.model_checkpoint_dir).glob("*.ckpt"))

    assert checkpoints, "no checkpoint was written; Lightning's default would have gone elsewhere"


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


def test_the_run_directory_holds_the_logs(fit):
    driver, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()


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

    assert trainer.current_epoch == 2
    assert trainer.state.finished
    assert driver.pytorch_model.target_adapter.linear.in_features == 78
    assert driver.pytorch_model.source_adapter.linear.in_features == 29
    assert driver.pytorch_model.target_gate.max_delay == 30
    assert driver.pytorch_model.source_delay_steps == 30
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (driver.pytorch_model.c_y, driver.pytorch_model.c_u) == (109, 58)


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, reach arms only. Every finite causal_reach_budget_s zero-fills the "
        "delayed prefix, and at step 0 EVERY surviving channel is zero (the fastest survivor is "
        "already one step stale). That all-zero vector is zero-variance input to the adapter "
        "norm and to each causal conv pre-norm, and the 1/sqrt(eps) = 316x backward "
        "amplification of ~10 stacked norms compounds: measured in float64 on this fixture, the "
        "global gradient norm is ~1e+26 at every budget (32/60/120/240 s) against ~98 unguarded, "
        "and overflows fp32 to inf. It does NOT scale with max_delay -- one all-zero step is "
        "enough -- so it is not fixed by raising warmup_period. Consequence: with "
        "gradient_clip_val=250 the clip coefficient is ~1e-24, every reach-arm step is scaled to "
        "nothing, and the arm trains only AdamW's weight decay while completing normally. "
        "The unguarded baseline (causal_reach_budget_s: null) builds no gate and is unaffected."
    ),
)
def test_the_guarded_runs_gradient_stays_finite(guarded_fit):
    """The gradient the guarded arms actually optimise with. Fix this before running them."""
    _, trainer = guarded_fit

    grad_norm = trainer.callback_metrics["train/grad_norm"]

    assert math.isfinite(float(grad_norm)), f"guarded train/grad_norm is {float(grad_norm)}"


def test_the_guarded_checkpoint_rebuilds_at_its_own_channel_widths(guarded_fit):
    """The adapters' widths depend on the resolved budget, so a checkpoint that recorded only
    the budget in seconds could not be rebuilt without re-running the resolution. The four
    channel tuples are therefore in ``model_kwargs``."""
    driver, _ = guarded_fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert len(blob["model_kwargs"]["target_keep_index"]) == 78
    assert len(blob["model_kwargs"]["source_delays"]) == 29
    rebuilt = SeqVaeLagAttnRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None


#: The one module allowed to call a global seeding function, and what makes it safe.
#:
#: The evaluation's oracle fits a probe, and a probe's *initialisation* runs on the global
#: generators -- ``nn.init`` takes no generator, so seeding it locally is not available. What the
#: ban actually protects against is a seed that **persists**: one that silently overrides
#: ``general_config.seed`` while looking like diligence. ``oracle.py`` seeds inside
#: ``torch.random.fork_rng``, which restores the state it found on exit, so nothing downstream of
#: the fit sees a stream it would not otherwise have seen.
#:
#: That is a property of behaviour rather than of source text, so it is asserted where it can be:
#: ``test_eval_oracle.py`` runs the fit and checks the global state afterwards against the state
#: the same pass leaves without one.
SEEDING_EXEMPT_MODULES = {"oracle.py"}


def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only
    seeding route; a stray global seed would silently override it while looking like
    diligence. The permutation generator's own rank-derived seed is a *local*
    ``torch.Generator``, deliberately different per rank, and leaves the global RNG untouched
    -- which is why the patterns below name the global calls specifically.

    :data:`SEEDING_EXEMPT_MODULES` is the single narrow exception, and it is asserted to be *used*
    as well as permitted: an exemption for a module that no longer seeds is a permission that
    outlived its reason, and the next reach for a global seed there would go unreported.
    """
    offenders = []
    exempt_and_seeding = set()
    for path in _PACKAGE_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue  # tests seed themselves for reproducibility, legitimately
        source = path.read_text(encoding="utf-8")
        for pattern in ("torch.manual_seed", "seed_everything", "np.random.seed"):
            if pattern not in source:
                continue
            if path.name in SEEDING_EXEMPT_MODULES:
                exempt_and_seeding.add(path.name)
                continue
            offenders.append(f"{path.name}: {pattern}")
    assert offenders == []
    assert exempt_and_seeding == SEEDING_EXEMPT_MODULES, (
        f"{sorted(SEEDING_EXEMPT_MODULES - exempt_and_seeding)} no longer seeds; drop the "
        f"exemption rather than leaving a permission nothing uses"
    )
