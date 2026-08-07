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
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config, with its output directories redirected under
    ``tmp_path``. ``setup_config`` is never called -- it would seed, open log sinks and probe
    MLflow -- so the directories are assigned directly."""
    driver = LagAttnRwsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    driver.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return driver


# --------------------------------------------------------------------------------------
# Config -> constructor
# --------------------------------------------------------------------------------------
def test_the_shipped_config_resolves_to_the_shipped_architecture(trainer):
    """Every architectural flag ``SHIPPED_KWARGS`` claims the config sets, it must set. That
    fixture is the suite's description of the production model; this keeps it honest against
    the config file itself. The geometry deliberately differs (the fixture is tiny) and is
    asserted against the config's real values below."""
    kwargs = trainer._build_model_kwargs()

    for name in (
        "causal_norm", "lag_bias_init", "use_entmax", "use_up_st",
        "horizon_depth", "horizon_kernel", "horizon_film",
    ):
        assert kwargs[name] == SHIPPED_KWARGS[name], f"{name} disagrees with the shipped flag set"
    # YAML has no tuple; the constructor coerces, so the sweep hands the list through.
    assert tuple(kwargs["encoder_extra_dilations"]) == SHIPPED_KWARGS["encoder_extra_dilations"]
    assert tuple(kwargs["logvar_clamp"]) == SHIPPED_KWARGS["logvar_clamp"]


def test_the_geometry_reaches_the_constructor(trainer):
    kwargs = trainer._build_model_kwargs()

    assert kwargs["sequence_length"] == 300
    assert kwargs["d_model"] == 128
    assert kwargs["d_z"] == 48
    assert kwargs["horizon"] == 30
    assert kwargs["raw_per_step"] == 16
    assert kwargs["warmup_period"] == 30
    assert kwargs["c_y"] == 109
    assert kwargs["c_u"] == 58
    assert kwargs["max_lag"] == 90
    assert kwargs["coverage_floor"] == 0.9


def test_loss_only_keys_do_not_reach_the_constructor(trainer):
    """The net takes tensors and computes a loss on request; it owns none of these. The
    constructor is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError``
    on the production config -- a poor place to find out."""
    kwargs = trainer._build_model_kwargs()

    for name in (
        "likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
        "kld_beta", "beta_prior", "causal_reach_budget_s",
    ):
        assert name not in kwargs, f"{name} is not the net's"


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
def test_create_model_wraps_the_net_in_its_task(trainer):
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask

    trainer.create_model()

    assert isinstance(trainer.pl_model, SeqVaeLagAttnRwsTask)
    assert trainer.pl_model.orig_model is trainer.pytorch_model


def test_create_model_passes_the_spike_breaker_block_to_the_task(trainer):
    """The block is validated by the framework and read by the module -- but nothing forwards
    it. ``GraphModelBase`` never passes it on, so a driver that forgets leaves a
    fully-configured ``enabled: true`` block doing nothing at all."""
    trainer.create_model()

    breaker = trainer.pl_model.hparams["spike_breaker"]
    assert breaker["enabled"] is True
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9


def test_create_model_passes_the_loss_hyperparameters_to_the_task(trainer):
    trainer.create_model()

    hparams = trainer.pl_model.hparams
    assert hparams["likelihood"] == "gaussian_nll"
    assert hparams["lambda_full"] == 1.0
    assert hparams["lambda_base"] == 1.0
    assert hparams["free_bits"] == 0.0
    assert hparams["beta_schedule"]["kind"] == "linear_warmup"
    assert hparams["beta_schedule"]["start"] == 0.0
    # The shipped anchor weight, forwarded from default.yaml rather than the driver's 0.0
    # fallback -- a driver that stopped reading the key would fall back silently, and this is
    # the assertion that would catch it.
    assert hparams["beta_prior"] == 0.1


def test_create_model_forces_eager_execution(trainer):
    trainer.create_model()

    assert trainer.pl_model.model is trainer.pl_model.orig_model


# --------------------------------------------------------------------------------------
# The startup causal-standing log
#
# The sentence is a claim about what this architecture's history states are a function of, and it
# is the premise every coupling number a run produces rests on. It moved behind a method so a
# sibling architecture can state its own; these pin what *this* one still says, in both branches.
# --------------------------------------------------------------------------------------
@pytest.fixture
def loguru_messages():
    """Collect loguru output.

    ``caplog`` cannot see it: loguru does not route through the stdlib ``logging`` module, so a
    ``caplog.at_level`` assertion against these lines would pass on a driver that logged nothing.
    """
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    yield messages
    logger.remove(sink_id)


def test_the_shipped_causal_standing_is_the_resolved_budget(trainer, loguru_messages):
    """The shipped config now runs guarded, so what every production log must state is what the
    budget resolved TO -- the surviving channel counts and the worst delay. A run that logged the
    unguarded sentence while training guarded would misdescribe its own inputs."""
    trainer.create_model()

    assert trainer.resolved_budget is not None
    message = trainer.causal_standing_message()
    assert message.startswith("causal reach budget 120 s:")
    assert "c_y 78, c_u 29" in message
    assert "max delay 30 steps" in message
    # Substring, as the unguarded assertion below is: the logger prefixes what it emits, so an
    # equality check would be testing the log format rather than the standing.
    assert any("causal reach budget 120 s:" in logged for logged in loguru_messages)


def test_the_unguarded_causal_standing_is_stated_verbatim(trainer):
    """The other branch, still reachable through ``sweep_reach_null.yaml``. It says the inputs are
    two-sided and that the KL is therefore not a transfer entropy -- the one claim a reader could
    otherwise take too far, and the reason that arm has to keep saying it."""
    trainer.resolved_budget = None

    assert trainer.causal_standing_message() == (
        "causal reach budget: none (all channels, no delay) -- input features at step t "
        "read up to 974 s into their own future, so the source-conditioned KL is not a "
        "transfer entropy."
    )


def test_a_configured_budget_is_stated_as_the_resolved_survivor_counts(trainer):
    """The other branch: with a budget the sentence is the resolution's own summary, so a run
    records the guard it actually got rather than the one it asked for."""
    trainer.config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 120.0
    trainer._build_model_kwargs()

    assert trainer.resolved_budget is not None
    assert trainer.causal_standing_message() == trainer.resolved_budget.summary()
    assert trainer.causal_standing_message().startswith("causal reach budget 120 s:")


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
    import math

    model = _built_model(trainer)

    # Per-block FiLM generators present and re-zeroed (identity at init), single film_gen not built.
    film = model.horizon_core.refine.film
    assert model.horizon_core.film_gen is None
    assert film is not None
    assert all(
        float(gen.weight.abs().max()) == 0.0 and float(gen.bias.abs().max()) == 0.0 for gen in film
    ), "per-block FiLM generators are not zero -- the re-zero policy did not reach the model"

    # Posterior source gain = 2.0.
    gain = model.posterior_head.a_head_norm.weight
    assert torch.equal(gain, torch.full_like(gain, 2.0)), "a_head_gain did not reach the model"

    # Horizon-embedding re-seeded at ~0.8.
    assert 0.7 < float(model.horizon_core.horizon_embedding.std()) < 0.9, (
        "horizon_embed_std did not reach the model"
    )

    # Output-head calibration: log(5/3) log-variance bias (maps to log-variance 0 under the clamp).
    bias = model.decoder.logvar_head.bias
    assert torch.allclose(bias, torch.full_like(bias, math.log(5.0 / 3.0)), atol=1e-6), (
        "head_init_calibration did not reach the model"
    )


def test_a_mistyped_init_policy_key_is_caught_by_the_seam(trainer):
    """The sensitivity control: renaming a policy key so it no longer names a constructor argument
    makes the model silently revert that policy (the config-to-kwargs mapping drops unknown keys),
    and the assembled-model read above turns that into a failure. Here the horizon-embedding reverts
    to the small constructor seed instead of the shipped 0.8."""
    vae = trainer.config["model_config"]["VAE_model"]
    vae["horizon_embed_stdd"] = vae.pop("horizon_embed_std")  # a typo the signature sweep drops

    model = _built_model(trainer)

    assert float(model.horizon_core.horizon_embedding.std()) < 0.05
