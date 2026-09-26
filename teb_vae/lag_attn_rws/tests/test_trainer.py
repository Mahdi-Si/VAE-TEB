r"""The driver turns config into a model, and forwards what only it can forward.

The config-to-constructor sweep is the part that fails silently: a key that fails to reach the
constructor does not raise -- the constructor has a default for everything -- so the run trains
a *different architecture* than its config describes, and only a checkpoint that will not reload
months later reveals it. The assertions below check the resolved kwargs against the flags the
shipped config sets, name by name, and against the suite's ``SHIPPED_KWARGS`` description of the
production model.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_rws.tests.conftest import SHIPPED_KWARGS
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from train.callbacks import MetricsLoggingCallback

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _redirected(trainer_cls, base_dir):
    """Build one driver of ``trainer_cls`` on the shipped config, writing under ``base_dir``."""
    driver = trainer_cls(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(base_dir)
    driver.train_results_dir = str(Path(base_dir) / "train_results")
    driver.model_checkpoint_dir = str(Path(base_dir) / "model_checkpoints")
    return driver


def _capture_callbacks(driver, monkeypatch):
    """Return the callback list ``train_model`` assembles, without building a model or fitting.

    ``build_trainer`` is intercepted, and ``pl_model`` is left unset: the assembly reads it only
    to hand it on, so the whole production net does not have to be constructed to find out which
    callbacks a config wires.
    """
    captured = {}

    def _capture(callbacks, model=None):
        captured["callbacks"] = callbacks

        class _StubTrainer:
            def fit(self, *args, **kwargs):
                captured["fit"] = True

        return _StubTrainer()

    monkeypatch.setattr(type(driver), "build_trainer", staticmethod(_capture))
    driver.pl_model = None
    driver.train_model(object(), object())
    return captured


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config, with its output directories redirected under
    ``tmp_path``. ``setup_config`` is never called -- it would seed, open log sinks and probe
    MLflow -- so the directories are assigned directly."""
    return _redirected(LagAttnRwsTrainer, tmp_path)


# --------------------------------------------------------------------------------------
# Config -> constructor
# --------------------------------------------------------------------------------------
def test_the_shipped_config_resolves_to_the_shipped_architecture(trainer):
    """Every architectural flag and every geometry value ``SHIPPED_KWARGS`` claims the config sets,
    the config must set and the sweep must carry to the constructor. That fixture is the suite's
    description of the production model; this keeps it honest against the config file itself, and
    a key that fails to reach the constructor would otherwise fall back to its default silently."""
    kwargs = trainer._build_model_kwargs()

    for name in (
        "causal_norm", "lag_bias_init", "use_entmax", "use_up_st",
        "horizon_depth", "horizon_kernel", "horizon_film",
        "sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "warmup_period",
        "c_y", "c_u", "max_lag", "coverage_floor",
    ):
        assert kwargs[name] == SHIPPED_KWARGS[name], f"{name} disagrees with the shipped kwargs"
    # YAML has no tuple; the constructor coerces, so the sweep hands the list through.
    assert tuple(kwargs["encoder_extra_dilations"]) == SHIPPED_KWARGS["encoder_extra_dilations"]
    assert tuple(kwargs["logvar_clamp"]) == SHIPPED_KWARGS["logvar_clamp"]


def test_the_resolved_kwargs_actually_build_a_model(trainer):
    """The sweep's output is only correct if the constructor accepts it."""
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    model = SeqVaeLagAttnRws(**trainer._build_model_kwargs())

    assert model.causal_norm is True
    assert model.n_causalized_norms > 0
    # The unconditional freeze the DDP strategy relies on.
    assert not any(p.requires_grad for p in model.lag_attn.W_o.parameters())


def test_an_unknown_config_key_is_ignored_rather_than_forwarded(trainer):
    """The sweep forwards by name against the real signature, so a stale key cannot crash."""
    trainer.config["model_config"]["VAE_model"]["a_key_from_an_older_model"] = 42

    assert "a_key_from_an_older_model" not in trainer._build_model_kwargs()


def test_a_null_config_value_falls_through_to_the_constructor_default(trainer):
    """``null`` in YAML means "unset", and the constructor's default is the single source."""
    trainer.config["model_config"]["VAE_model"]["dropout"] = None

    assert "dropout" not in trainer._build_model_kwargs()


def test_init_weights_is_never_a_config_decision(trainer):
    """Skipping initialisation would also skip the post-init delta-head zeroing order the
    zero-KL start depends on; the key is refused even when a config supplies it."""
    trainer.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in trainer._build_model_kwargs()


# --------------------------------------------------------------------------------------
# create_model
# --------------------------------------------------------------------------------------
def test_create_model_passes_the_spike_breaker_block_to_the_task(trainer):
    """The block is validated by the framework and read by the module -- but nothing forwards
    it. ``GraphModelBase`` never passes it on, so a driver that forgets leaves a
    fully-configured ``enabled: true`` block doing nothing at all."""
    trainer.create_model()

    configured = trainer.config["advanced_config"]["spike_breaker"]
    assert configured["enabled"] is True, "the shipped config must exercise the forwarding"
    assert dict(trainer.pl_model.hparams["spike_breaker"]) == dict(configured)


def test_create_model_passes_the_loss_hyperparameters_to_the_task(trainer):
    """Each objective key the config sets reaches the task by value. The weights are checked
    nonzero in the config first: a driver that stopped reading a key falls back to $0.0$
    silently, and only a nonzero configured value can tell the two apart."""
    trainer.create_model()

    hparams = trainer.pl_model.hparams
    vae = trainer.config["model_config"]["VAE_model"]
    for name in (
        "likelihood", "lambda_full", "lambda_base", "free_bits", "beta_schedule",
        "beta_prior", "lambda_ms", "lambda_deriv", "lambda_boundary",
    ):
        assert hparams[name] == vae[name], name
    for name in ("beta_prior", "lambda_ms", "lambda_deriv", "lambda_boundary"):
        assert float(vae[name]) != 0.0, f"the shipped config no longer weights {name}"


def test_create_model_forces_eager_execution_even_when_the_config_asks_to_compile(trainer):
    """The LSTM encoders defeat TorchInductor unconditionally, so the driver refuses compilation
    without reading ``advanced_config.trainer.compile`` at all."""
    trainer.config["advanced_config"]["trainer"]["compile"] = True

    trainer.create_model()

    assert trainer.pl_model.model is trainer.pl_model.orig_model


# --------------------------------------------------------------------------------------
# The startup causal-standing log
#
# The sentence is a claim about what this architecture's history states are a function of, and it
# is the premise every coupling number a run produces rests on.
# --------------------------------------------------------------------------------------
def test_a_configured_budget_is_stated_as_the_resolved_survivor_counts(trainer):
    """With a budget the sentence is the resolution's own summary, so a run records the guard it
    actually got rather than the one it asked for."""
    trainer.config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 120.0
    trainer._build_model_kwargs()

    assert trainer.resolved_budget is not None
    assert trainer.causal_standing_message() == trainer.resolved_budget.summary()


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(trainer):
    """So the blob rebuilds into this architecture and not the constructor's defaults."""
    trainer.create_model()

    assert trainer.pl_model._model_kwargs == trainer._build_model_kwargs()


def test_an_unalignable_core_checkpoint_raises_rather_than_training_from_scratch(
    trainer, tmp_path
):
    """``load_checkpoint_strict`` returns ``None`` when nothing lines up; it does not raise. An
    unchecked call therefore trains a randomly-initialised model that was supposed to be warm
    started, and says nothing about it."""
    unrelated = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"nothing.like.this": torch.zeros(2)}}, unrelated)
    trainer.config["model_config"]["core_model_checkpoint"] = str(unrelated)

    with pytest.raises(RuntimeError, match="could not align"):
        trainer.create_model()


def test_a_core_checkpoint_from_another_model_is_refused_before_it_is_loaded(trainer, tmp_path):
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {}, "model_class": "SeqVaeLagAttn"}, foreign)
    trainer.config["model_config"]["core_model_checkpoint"] = str(foreign)

    with pytest.raises(ValueError, match="does not match the active model class"):
        trainer.create_model()


# --------------------------------------------------------------------------------------
# The config-to-constructor seam for the zero-parameter init policies
# --------------------------------------------------------------------------------------
def _built_model(trainer):
    """The model the shipped config actually produces, read back off the assembled object."""
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    return SeqVaeLagAttnRws(**trainer._build_model_kwargs())


def test_the_shipped_config_drives_the_live_init_policies(trainer):
    """The seam that fails silently: an init-policy key that does not reach the constructor reverts
    its policy without raising, so the run trains a different starting point than its config
    describes. Read the policies off the assembled model, so a mistyped or dropped key is caught
    here rather than months later in an unreloadable checkpoint."""
    from teb_vae.lag_attn.nets.blocks import smooth_bound

    vae = trainer.config["model_config"]["VAE_model"]
    model = _built_model(trainer)

    # The posterior source gain, off its unit default.
    gain = model.posterior_head.a_head_norm.weight
    assert float(vae["a_head_gain"]) != 1.0
    assert torch.equal(gain, torch.full_like(gain, float(vae["a_head_gain"]))), (
        "a_head_gain did not reach the model"
    )

    # The horizon embedding, re-seeded at the configured scale.
    std = float(model.horizon_core.horizon_embedding.std())
    assert std == pytest.approx(float(vae["horizon_embed_std"]), rel=0.15), (
        "horizon_embed_std did not reach the model"
    )

    # Output-head calibration: the log-variance bias is the pre-image of log-variance 0.
    assert vae["head_init_calibration"] is True
    bias = model.decoder.logvar_head.bias
    assert torch.allclose(smooth_bound(bias, *model.logvar_clamp), torch.zeros_like(bias),
                          atol=1e-6), "head_init_calibration did not reach the model"


def test_a_mistyped_init_policy_key_is_caught_by_the_seam(trainer):
    """The sensitivity control: renaming a policy key so it no longer names a constructor argument
    makes the model silently revert that policy (the config-to-kwargs mapping drops unknown keys),
    and the assembled-model read above turns that into a failure. Here the horizon-embedding reverts
    to the small constructor seed instead of the configured scale."""
    vae = trainer.config["model_config"]["VAE_model"]
    vae["horizon_embed_stdd"] = vae.pop("horizon_embed_std")  # a typo the signature sweep drops

    model = _built_model(trainer)

    assert float(model.horizon_core.horizon_embedding.std()) < 0.05


# --------------------------------------------------------------------------------------
# The callback seams
# --------------------------------------------------------------------------------------
def test_the_tracked_metric_list_is_reached_through_the_class_attribute(trainer, monkeypatch):
    """A sibling adding a metric must not have to override ``train_model`` to collect it: that
    method is the whole callback assembly, and a copy of it would be free to drift from this one
    on every knob it wires -- the checkpoint monitor, the hyperparameter keys, the plot cadence."""
    class _ExtraMetricTrainer(LagAttnRwsTrainer):
        TRACKED_METRICS = _TRACKED_METRICS + ("val/a_sibling_metric",)

    driver = _redirected(_ExtraMetricTrainer, trainer.output_base_dir)
    captured = _capture_callbacks(driver, monkeypatch)

    collector = next(
        cb for cb in captured["callbacks"] if isinstance(cb, MetricsLoggingCallback)
    )
    assert collector.tracked_metrics == _TRACKED_METRICS + ("val/a_sibling_metric",)


def test_the_plotting_block_name_is_an_attribute_the_assembly_reads(trainer, monkeypatch):
    """The literal is one string in one place, and a run whose config spells the block
    differently gets no figure, no error, and nothing in the log saying why."""
    class _RenamedBlockTrainer(LagAttnRwsTrainer):
        PLOT_CONFIG_KEY = "a_block_no_config_carries"

    driver = _redirected(_RenamedBlockTrainer, trainer.output_base_dir)
    captured = _capture_callbacks(driver, monkeypatch)

    names = [type(cb).__name__ for cb in captured["callbacks"]]
    assert "LagAttnRwsPlotCallback" not in names


def test_the_plot_callback_import_happens_only_when_the_figure_is_enabled(trainer, monkeypatch):
    """The callback pulls matplotlib. Resolved eagerly -- as a class attribute would be -- every
    run would import it, including the ones on a box that has no display stack at all."""
    import sys

    monkeypatch.delitem(sys.modules, "teb_vae.lag_attn_rws.plotting", raising=False)

    disabled = _redirected(LagAttnRwsTrainer, trainer.output_base_dir)
    disabled.config["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] = False
    _capture_callbacks(disabled, monkeypatch)
    assert "teb_vae.lag_attn_rws.plotting" not in sys.modules

    enabled = _redirected(LagAttnRwsTrainer, trainer.output_base_dir)
    captured = _capture_callbacks(enabled, monkeypatch)
    assert "teb_vae.lag_attn_rws.plotting" in sys.modules
    assert "LagAttnRwsPlotCallback" in [type(cb).__name__ for cb in captured["callbacks"]]
    assert LagAttnRwsTrainer.plot_callback_cls().__name__ == "LagAttnRwsPlotCallback"
