r"""The task sits on the framework seams rather than around them.

Most of what a Lightning module needs is inherited. The value of that is entirely in what is
*absent* here -- no ``training_step``, no ``configure_optimizers``, no constructor bypass, no
hand-rolled spike breaker -- and absence is exactly what a normal test cannot see: a re-added
override does not fail anything, it just quietly takes back the seam. So several tests below assert
that this class does not define a method, which is unusual and deliberate.

The rest pins the two contracts the framework enforces by convention rather than by type: the
metrics dict must be numeric and unprefixed, and ``main_loss`` must be present under exactly that
name or the breaker silently watches something else.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn.task import SeqVaeLagAttnTask
from train.pl_model_base import LightningModelBase


# --------------------------------------------------------------------------------------
# What the task does not do
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers"],
)
def test_the_task_does_not_override_the_inherited_step_machinery(method):
    """Each of these is a seam that took work to make usable; overriding one takes it back.

    ``training_step`` is the one that matters most: the framework's version is what runs the
    config-gated spike breaker, and a subclass that defines its own silently disables it -- with
    ``advanced_config.spike_breaker.enabled: true`` still sitting in the config.
    """
    assert method not in vars(SeqVaeLagAttnTask), (
        f"{method} is overridden; the inherited implementation is the seam this model is meant "
        f"to use"
    )


def test_there_is_no_curriculum_hook():
    """A v2-era mechanism this model has no stage function for; it could only ever early-return."""
    assert "_apply_curriculum" not in vars(SeqVaeLagAttnTask)
    assert "_on_train_epoch_start_hook" not in vars(SeqVaeLagAttnTask)


def test_the_constructor_goes_through_the_base(task):
    """Not through a grandparent ``pl.LightningModule.__init__`` bypass.

    The bypass predates the ``compile_model`` flag and silently drops every base side effect:
    ``save_hyperparameters``, ``_orig_model`` (so ``orig_model`` and the checkpoint stamp both
    break), ``self.model``, and the spike-breaker counters. Because the base reads hparams
    defensively, the result is fallback defaults rather than an error.
    """
    module = task()

    assert isinstance(module, LightningModelBase)
    assert module.orig_model is module._orig_model
    assert hasattr(module, "_spike_ema_loss")  # a bypass would leave the counters unset
    assert module.hparams.get("lr") == 1e-3


def test_compilation_is_off_and_the_eager_module_is_what_runs(task):
    """Four independent things in this net defeat inductor, so this is permanent.

    A compiled ``self.model`` would also put an ``_orig_mod.`` prefix on every state-dict key.
    """
    module = task()

    assert module.model is module.orig_model
    assert module.hparams.get("compile_model") is False


def test_compilation_is_not_a_constructor_argument():
    """It is a property of this net, not a caller's choice, so the task does not expose the knob."""
    assert "compile_model" not in inspect.signature(SeqVaeLagAttnTask.__init__).parameters


# --------------------------------------------------------------------------------------
# The metrics contract
# --------------------------------------------------------------------------------------
def test_every_metric_is_numeric(task, stub_batch, perturb_posterior):
    """The loss dict carries ``likelihood``, a str, and the logger coerces a non-numeric value to
    a clean ``0.0`` rather than raising. A splatted loss dict would log the string as zero."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    for name, value in metrics.items():
        assert isinstance(value, (torch.Tensor, float, int)), f"{name} is a {type(value).__name__}"
        assert not isinstance(value, str)


def test_no_metric_name_carries_a_slash(task, stub_batch, perturb_posterior):
    """A name containing '/' bypasses stage prefixing entirely.

    Returning ``val/foo`` from a train-stage call logs it under ``val/foo`` and can poison a
    ``ModelCheckpoint`` monitor -- and a prefixed ``main_loss`` would make the breaker fall back to
    the returned loss without a word.
    """
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert [name for name in metrics if "/" in name] == []


def test_main_loss_is_emitted_under_exactly_that_name(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert "main_loss" in metrics


def test_the_breaker_actually_consumes_main_loss(task):
    """Emission is not consumption, and the difference is silent.

    The framework falls back to the returned loss when ``metrics['main_loss']`` is missing or
    ``None`` -- no warning. So this drives the real breaker with a ``main_loss`` far below the
    returned loss and checks which one the EMA learned. A breaker that had fallen back would seed
    its EMA from the returned value instead.
    """
    module = task(
        spike_breaker={"enabled": True, "warmup_batches": 0, "comparison_metric": "main_loss"}
    )

    returned = torch.tensor(100.0, requires_grad=True)
    metrics = {"total_loss": returned, "main_loss": torch.tensor(1.0)}
    module._apply_spike_breaker(returned, metrics, module.hparams["spike_breaker"])

    assert module._spike_ema_loss == pytest.approx(1.0), (
        "the breaker seeded its EMA from the returned loss, so it is not watching main_loss"
    )


# --------------------------------------------------------------------------------------
# Loss composition
# --------------------------------------------------------------------------------------
def test_the_loss_is_finite_and_carries_gradient(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    loss, _ = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_the_source_stream_is_the_concatenation_the_model_was_built_for(task, stub_batch):
    module = task()

    u_stream = module._build_source_stream(stub_batch)

    assert u_stream.shape[-1] == module.orig_model.c_u == 101
    assert torch.equal(u_stream[..., :43], stub_batch.up_st)
    assert torch.equal(u_stream[..., 43:], stub_batch.up_ph)


def test_the_phase_only_ablation_drops_the_scattering_block(task, prod_kwargs, stub_batch):
    module = task(model_kwargs=dict(prod_kwargs, use_up_st=False, c_u=58))

    u_stream = module._build_source_stream(stub_batch)

    assert u_stream.shape[-1] == 58
    assert torch.equal(u_stream, stub_batch.up_ph)


def test_a_missing_source_field_names_the_config_key_that_fixes_it(task, stub_batch):
    """The net would otherwise fail with a channel-count error naming neither the field nor the
    key, several frames from the actual mistake."""
    module = task()
    del stub_batch.up_st

    with pytest.raises(RuntimeError, match="load_fields"):
        module._build_source_stream(stub_batch)


def test_kld_raw_is_reported_separately_from_kld_train(task, stub_batch, perturb_posterior):
    r"""Only ``kld_raw`` may be read as a TE surrogate.

    ``kld_train`` is floored by free bits and is what enters the loss; reading it as the TE
    surrogate would report the floor as signal. With ``free_bits > 0`` they genuinely differ.
    """
    module = task()
    perturb_posterior(module.orig_model)
    assert float(module.hparams["free_bits"]) > 0.0

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["kld_train"]) >= float(metrics["kld_raw"])


def test_the_residual_diagnostics_are_finite_scalars(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    for name in ("delta_mu_rms", "mu_post_prior_gap_rms"):
        assert metrics[name].ndim == 0
        assert torch.isfinite(metrics[name])


def test_the_latent_gap_is_zero_at_init_and_positive_once_perturbed(task, stub_batch, perturb_posterior):
    """The zero-init invariant, seen through the diagnostic rather than the KL.

    It is also the reason every KL assertion in this suite perturbs first: at init the posterior
    *is* the prior, so an untouched model reports 0 for reasons that have nothing to do with being
    correct.
    """
    module = task()

    _, at_init = module.compute_loss_and_metrics(stub_batch, 1, "train")
    assert float(at_init["mu_post_prior_gap_rms"]) == pytest.approx(0.0, abs=1e-6)

    perturb_posterior(module.orig_model)
    _, perturbed = module.compute_loss_and_metrics(stub_batch, 1, "train")
    assert float(perturbed["mu_post_prior_gap_rms"]) > 0.0


def test_the_validity_mask_changes_the_loss(task, make_stub_batch_fn, perturb_posterior):
    """A weight the loss ignored would let gaps pollute the KL curve, silently."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch_fn()

    torch.manual_seed(1)
    _, all_valid = module.compute_loss_and_metrics(batch, 1, "train")
    batch.weight[:, : batch.weight.shape[1] // 2] = 0.0
    torch.manual_seed(1)
    _, half_masked = module.compute_loss_and_metrics(batch, 1, "train")

    assert float(all_valid["feat_loss"]) != pytest.approx(float(half_masked["feat_loss"]), rel=1e-6)


# --------------------------------------------------------------------------------------
# The beta schedule
# --------------------------------------------------------------------------------------
def test_a_constant_schedule_falls_back_to_kld_beta(task):
    module = task(hparams={"beta_schedule": {"kind": "constant"}, "kld_beta": 0.007})

    assert module._resolve_beta(0) == pytest.approx(0.007)
    assert module._resolve_beta(999) == pytest.approx(0.007)


def test_a_constant_schedule_prefers_its_own_value(task):
    module = task(hparams={"beta_schedule": {"kind": "constant", "value": 0.5}, "kld_beta": 0.007})

    assert module._resolve_beta(10) == pytest.approx(0.5)


def test_linear_warmup_ramps_then_holds(task):
    module = task(
        hparams={"beta_schedule": {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 10}}
    )

    assert module._resolve_beta(0) == pytest.approx(0.0)
    assert module._resolve_beta(5) == pytest.approx(0.5)
    assert module._resolve_beta(10) == pytest.approx(1.0)
    assert module._resolve_beta(1000) == pytest.approx(1.0)  # holds; does not keep climbing


def test_a_zero_warmup_is_the_end_value_rather_than_a_division_by_zero(task):
    module = task(
        hparams={"beta_schedule": {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 0}}
    )

    assert module._resolve_beta(0) == pytest.approx(1.0)


def test_no_schedule_is_the_constant_kld_beta(task):
    module = task(hparams={"beta_schedule": None, "kld_beta": 0.01})

    assert module._resolve_beta(50) == pytest.approx(0.01)


def test_an_unknown_schedule_kind_raises(task):
    """Rather than silently training a different objective than the config describes."""
    module = task(hparams={"beta_schedule": {"kind": "cosine"}})

    with pytest.raises(ValueError, match="cosine"):
        module._resolve_beta(0)


def test_the_scheduled_beta_is_what_weights_the_kl_and_what_is_reported(task, stub_batch, perturb_posterior):
    """``kld_beta`` in the metrics must be the resolved value, not the raw hparam.

    They differ the moment a schedule exists, and the plots read the reported one.
    """
    module = task(
        hparams={
            "beta_schedule": {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 10},
            "kld_beta": 0.01,
        }
    )
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["kld_beta"]) == pytest.approx(module._resolve_beta(module.current_epoch))
    assert float(metrics["kld_beta"]) != pytest.approx(0.01)  # not the raw hparam
