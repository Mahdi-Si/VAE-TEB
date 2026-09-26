r"""The task sits on the framework seams rather than around them.

Most of what a Lightning module needs is inherited. The value of that is entirely in what is
*absent* here -- no ``training_step``, no ``configure_optimizers``, no constructor bypass, no
hand-rolled spike breaker -- and absence is exactly what a normal test cannot see: a re-added
override does not fail anything, it just quietly takes back the seam. So the first test below
asserts that this class does not define a method, which is unusual and deliberate.

The rest pins the two contracts the framework enforces by convention rather than by type (the
metrics dict must be numeric and unprefixed, and ``main_loss`` must be present under exactly that
name or the breaker silently watches something else), the stream assembly and its width checks
against the actual batch, and the $\beta$ schedule.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.task import SeqVaeLagAttnTask


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


# --------------------------------------------------------------------------------------
# The metrics contract
# --------------------------------------------------------------------------------------
def test_the_metrics_are_numeric_unprefixed_and_carry_main_loss(task, stub_batch, perturb_posterior):
    """The loss dict carries ``likelihood``, a str, and the logger coerces a non-numeric value to
    a clean ``0.0`` rather than raising. A splatted loss dict would log the string as zero.

    A name containing '/' bypasses stage prefixing entirely: returning ``val/foo`` from a
    train-stage call logs it under ``val/foo`` and can poison a ``ModelCheckpoint`` monitor. And
    the framework falls back to the returned loss, without a word, when ``main_loss`` is missing.
    """
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    for name, value in metrics.items():
        assert isinstance(value, (torch.Tensor, float, int)), f"{name} is a {type(value).__name__}"
    assert [name for name in metrics if "/" in name] == []
    assert "main_loss" in metrics


# --------------------------------------------------------------------------------------
# Loss composition
# --------------------------------------------------------------------------------------
def test_the_source_stream_is_the_concatenation_the_model_was_built_for(task, stub_batch):
    module = task()

    u_stream = module._build_source_stream(stub_batch)

    assert u_stream.shape[-1] == module.orig_model.c_u == 58
    assert torch.equal(u_stream[..., :43], stub_batch.up_st)
    assert torch.equal(u_stream[..., 43:], stub_batch.up_ph)


def test_the_phase_only_ablation_drops_the_scattering_block(task, prod_kwargs, stub_batch):
    module = task(model_kwargs=dict(prod_kwargs, use_up_st=False, c_u=15))

    u_stream = module._build_source_stream(stub_batch)

    assert u_stream.shape[-1] == 15
    assert torch.equal(u_stream, stub_batch.up_ph)


def test_a_missing_source_field_names_the_config_key_that_fixes_it(task, stub_batch):
    """The net would otherwise fail with a channel-count error naming neither the field nor the
    key, several frames from the actual mistake."""
    module = task()
    del stub_batch.up_st

    with pytest.raises(RuntimeError, match="load_fields"):
        module._build_source_stream(stub_batch)


# --------------------------------------------------------------------------------------
# Channel widths are checked against the data, not against a constant
# --------------------------------------------------------------------------------------
def test_a_stale_phase_only_c_u_is_caught_against_the_actual_batch(task, prod_kwargs, stub_batch):
    r"""The trap a constant-based constructor check could not see.

    $58$ is now the *with-scattering* width and used to be the phase-only one. A config left at
    the old phase-only setting -- ``use_up_st: false`` with ``c_u: 58`` -- therefore passes every
    config-shaped check while the data is $15$ wide. Only a comparison against the batch catches
    it, and the message has to name the per-field widths or the reader cannot tell which of the
    two numbers is wrong.
    """
    module = task(model_kwargs=dict(prod_kwargs, use_up_st=False, c_u=58))

    with pytest.raises(RuntimeError) as excinfo:
        module._build_source_stream(stub_batch)

    message = str(excinfo.value)
    for fragment in ("up_ph=15", "c_u=58", "use_up_st=False", "model_config.VAE_model.c_u"):
        assert fragment in message, f"{fragment!r} missing from: {message}"


def test_a_batch_from_a_pre_migration_shard_is_caught(task, stub_batch):
    """The other direction: a correct config pointed at an old-width HDF5.

    No constant-based check could ever catch this one -- it validates the config against the
    config, and here the config is right and the data is wrong.
    """
    module = task()  # c_u=58, use_up_st=True; an old shard makes the stream 43+58=101
    stub_batch.up_ph = torch.randn(stub_batch.up_st.shape[0], stub_batch.up_st.shape[1], 58)

    with pytest.raises(RuntimeError, match="source stream is 101 channels"):
        module._build_source_stream(stub_batch)


def test_the_target_width_is_checked_too(task, stub_batch):
    """``c_y`` had no validation of any kind before the widths moved to the data boundary."""
    module = task()
    stub_batch.fhr_ph = torch.randn(stub_batch.fhr_st.shape[0], stub_batch.fhr_st.shape[1], 44)

    with pytest.raises(RuntimeError, match="target stream is 87 channels"):
        module._build_target_streams(stub_batch)


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
_ZERO_WARMUP = {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 0}


@pytest.mark.parametrize(
    "schedule, epochs, expected",
    [
        ({"kind": "constant"}, (0, 999), 0.007),
        ({"kind": "constant", "value": 0.5}, (0, 10), 0.5),
        (None, (0, 50), 0.007),
        # The end value, rather than a division by zero.
        (_ZERO_WARMUP, (0,), 1.0),
    ],
    ids=["constant-falls-back-to-kld_beta", "constant-own-value", "no-schedule", "zero-warmup"],
)
def test_a_flat_schedule_resolves_to_its_constant(task, schedule, epochs, expected):
    module = task(hparams={"beta_schedule": schedule, "kld_beta": 0.007})

    for epoch in epochs:
        assert module._resolve_beta(epoch) == pytest.approx(expected)


def test_linear_warmup_ramps_then_holds(task):
    module = task(
        hparams={"beta_schedule": {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 10}}
    )

    assert module._resolve_beta(0) == pytest.approx(0.0)
    assert module._resolve_beta(5) == pytest.approx(0.5)
    assert module._resolve_beta(10) == pytest.approx(1.0)
    assert module._resolve_beta(1000) == pytest.approx(1.0)  # holds; does not keep climbing


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
