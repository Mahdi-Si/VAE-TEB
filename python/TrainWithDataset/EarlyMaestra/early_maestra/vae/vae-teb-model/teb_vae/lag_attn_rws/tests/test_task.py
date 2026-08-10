r"""The task sits on the framework seams rather than around them.

Most of what a Lightning module needs is inherited, and the value of that is entirely in what is
*absent* here -- no ``training_step``, no ``configure_optimizers``, no constructor bypass, no
hand-rolled spike breaker. Absence is exactly what a normal test cannot see (a re-added override
does not fail anything, it just quietly takes back the seam), so several tests below assert that
this class does not define a method.

The rest pins the contracts the framework enforces by convention rather than by type: metrics
must be numeric and unprefixed, ``main_loss`` must exist under exactly that name, and the metric
set must be the documented one -- a silently added metric is a column no callback collects.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from train.pl_model_base import LightningModelBase

#: The metric set every stage emits. The three permutation-control metrics are validation-only
#: and deliberately not in this set.
_STAGE_METRICS = {
    "total_loss", "main_loss",
    "nll_full_block", "nll_base_block", "nll_full_sample", "nll_base_sample",
    "pred_gap",
    "source_conditioned_kl_raw", "source_conditioned_kl_train",
    "kld_active_frac", "kld_beta",
    "prior_rate", "beta_prior",
    # The three shape terms and their echoed weights. Present on every stage whether or not a
    # config weights them: the pair is what distinguishes a term that was off from one that was
    # on and small, and a column that appeared only in some arms could not be read across runs.
    "aux_multiscale", "aux_derivative", "aux_boundary",
    "lambda_ms", "lambda_deriv", "lambda_boundary",
    "anchor_coverage_frac",
    "mean_logvar_full", "mean_logvar_base",
    "logvar_full_floor_frac", "logvar_full_ceil_frac",
    "mean_logvar_prior", "mean_logvar_post", "logvar_prior_floor_frac",
    "delta_mu_rms", "mu_post_prior_gap_rms",
    "mu_prior_sat_frac", "delta_mu_sat_frac",
}

_VAL_ONLY_METRICS = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}


# --------------------------------------------------------------------------------------
# What the task does not do
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers"],
)
def test_the_task_does_not_override_the_inherited_step_machinery(method):
    """``training_step`` is the one that matters most: the framework's version runs the
    config-gated spike breaker, and a subclass that defines its own silently disables it."""
    assert method not in vars(SeqVaeLagAttnRwsTask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


def test_the_constructor_goes_through_the_base(task):
    """Not through a grandparent ``pl.LightningModule.__init__`` bypass, which would silently
    drop ``save_hyperparameters``, ``_orig_model``, ``self.model`` and the breaker counters."""
    module = task()

    assert isinstance(module, LightningModelBase)
    assert module.orig_model is module._orig_model
    assert hasattr(module, "_spike_ema_loss")  # a bypass would leave the counters unset
    assert module.hparams.get("lr") == 1e-3


def test_compilation_is_off_and_the_eager_module_is_what_runs(task):
    """Three independent things in this net defeat inductor, so this is permanent."""
    module = task()

    assert module.model is module.orig_model
    assert module.hparams.get("compile_model") is False


def test_compilation_defaults_off_and_no_config_can_turn_it_on_for_this_net():
    """The invariant is unchanged -- no configuration reaches compilation for *this* net, whose
    LSTM encoders defeat inductor unconditionally -- but it is now enforced where the refusal is
    true rather than by the keyword's absence.

    The keyword exists so a subclass over a net **without** an LSTM can opt in; the refusal for
    this one lives in ``LagAttnRwsTrainer.compile_model_requested``, which does not read
    ``advanced_config.trainer.compile`` at all. Both halves are asserted, because the keyword
    defaulting to ``False`` would be worth nothing if the driver passed a config value into it."""
    from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

    parameter = inspect.signature(SeqVaeLagAttnRwsTask.__init__).parameters["compile_model"]
    assert parameter.default is False

    # The driver's refusal is unconditional: a config asking for compilation is still refused.
    assert LagAttnRwsTrainer.compile_model_requested(object()) is False


def test_the_forward_goes_through_model_and_everything_else_through_orig_model():
    """``self.model`` is the (potentially compiled) forward handle; ``self.orig_model`` is the
    eager module whose helpers (``compute_loss``, geometry) must be called directly. With
    compilation permanently off the two alias one object, which is exactly why only a source
    check can catch a future regression."""
    source = Path(inspect.getfile(SeqVaeLagAttnRwsTask)).read_text(encoding="utf-8")

    assert len(re.findall(r"self\.model\(", source)) == 1  # exactly the forward call
    assert "self.orig_model.compute_loss" in source
    assert "self.model.compute_loss" not in source


# --------------------------------------------------------------------------------------
# The metrics contract
# --------------------------------------------------------------------------------------
def test_every_metric_is_numeric_and_unprefixed(task, stub_batch, perturb_posterior):
    """The net's loss dict keeps its ``likelihood`` string outside the metric dict; the task
    must preserve that, and no name may carry a '/' -- a prefixed name bypasses stage framing
    and can poison a ``ModelCheckpoint`` monitor."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    for name, value in metrics.items():
        assert isinstance(value, torch.Tensor), f"{name} is a {type(value).__name__}"
        assert "/" not in name


def test_the_train_metric_set_is_exactly_the_documented_one(task, stub_batch, perturb_posterior):
    """Exact equality in both directions: a missing metric is a lost readout, and an extra one
    is a column no callback collects -- both silent."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert set(metrics) == _STAGE_METRICS


def test_validation_additionally_emits_the_shuffled_readouts(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert set(metrics) == _STAGE_METRICS | _VAL_ONLY_METRICS


def test_the_breaker_actually_consumes_main_loss(task):
    """Emission is not consumption: the framework falls back to the returned loss when
    ``metrics['main_loss']`` is missing, silently. Drive the real breaker with a ``main_loss``
    far below the returned loss and check which one seeded the EMA."""
    module = task(
        spike_breaker={"enabled": True, "warmup_batches": 0, "comparison_metric": "main_loss"}
    )

    returned = torch.tensor(100.0, requires_grad=True)
    metrics = {"total_loss": returned, "main_loss": torch.tensor(1.0)}
    module._apply_spike_breaker(returned, metrics, module.hparams["spike_breaker"])

    assert float(module._spike_ema_loss) == pytest.approx(1.0), (
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

    assert u_stream.shape[-1] == module.orig_model.c_u == 58
    assert torch.equal(u_stream[..., :43], stub_batch.up_st)
    assert torch.equal(u_stream[..., 43:], stub_batch.up_ph)


def test_the_phase_only_ablation_drops_the_scattering_block(task, tiny_kwargs, stub_batch):
    module = task(model_kwargs=dict(tiny_kwargs, use_up_st=False, c_u=15))

    u_stream = module._build_source_stream(stub_batch)

    assert u_stream.shape[-1] == 15
    assert torch.equal(u_stream, stub_batch.up_ph)


def test_a_missing_source_field_names_the_config_key_that_fixes_it(task, stub_batch):
    module = task()
    del stub_batch.up_st

    with pytest.raises(RuntimeError, match="load_fields"):
        module._build_source_stream(stub_batch)


@pytest.mark.parametrize("field", ["fhr", "weight"])
def test_a_missing_raw_target_field_names_the_config_key(task, stub_batch, field):
    """Both are hard requirements here: the raw signal is the target and the weight is its only
    trustworthy validity signal."""
    module = task()
    delattr(stub_batch, field)

    with pytest.raises(RuntimeError, match="load_fields"):
        module._build_raw_target(stub_batch)


# --------------------------------------------------------------------------------------
# The forward-input seam
#
# ``_build_forward_inputs`` is the one place a sibling architecture over a different input
# representation is meant to differ, so it gets both halves: the default tuple is exactly what the
# builders produce, and an override actually decides what the net receives.
# --------------------------------------------------------------------------------------
def test_the_default_forward_inputs_are_the_three_tensors_the_builders_produce(task, stub_batch):
    """Same tensors, same order, same objects. A hook that rebuilt them independently could drift
    from the builders whose width checks are the only thing standing between a mismatched shard
    and a silently wrong run."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)

    y_st, y_ph = module._build_target_streams(stub_batch)
    assert len(inputs) == 3
    assert inputs[0] is y_st
    assert inputs[1] is y_ph
    assert torch.equal(inputs[2], module._build_source_stream(stub_batch))


def test_an_override_changes_what_the_net_receives_and_nothing_else(
    task, stub_batch, perturb_posterior
):
    """The seam's whole contract, asserted on the one readout that can tell the difference: the
    prior branch never sees the source, so a deranged source stream must leave ``nll_base_block``
    bitwise identical while ``nll_full_block`` moves. Both halves are needed -- movement alone
    would also be produced by an override that perturbed the target.

    Overridden on the instance rather than by subclassing, so the *weights* are provably the same
    object in both calls; the seed is re-set because one ``randn_like`` draw enters both branches
    and an unseeded second call would move ``nll_base_block`` for reasons of its own.
    """
    module = task()
    perturb_posterior(module.orig_model)
    default_inputs = module._build_forward_inputs
    torch.manual_seed(7)
    _loss, reference = module.compute_loss_and_metrics(stub_batch, 0, "train")

    def _deranged_source(batch):
        y_st, y_ph, u_stream = default_inputs(batch)
        return y_st, y_ph, u_stream.flip(0)

    module._build_forward_inputs = _deranged_source
    torch.manual_seed(7)
    _loss, overridden = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert torch.equal(overridden["nll_base_block"], reference["nll_base_block"])
    assert not torch.equal(overridden["nll_full_block"], reference["nll_full_block"])


@pytest.mark.parametrize("builder", ["_build_target_streams", "_build_source_stream"])
def test_compute_loss_and_metrics_reaches_the_net_only_through_the_hook(builder):
    """A remaining direct call to either stream builder would make the hook advisory: the
    override above would run, and the net would go on being fed what it always was. Checked in the
    source because a *surviving* call site produces a correct-looking run, not a failure."""
    source = inspect.getsource(SeqVaeLagAttnRwsTask.compute_loss_and_metrics)

    assert f"self.{builder}(" not in source
    assert "self._build_forward_inputs(" in source


# --------------------------------------------------------------------------------------
# Channel widths are checked against the data, not against a constant
# --------------------------------------------------------------------------------------
def test_a_stale_phase_only_c_u_is_caught_against_the_actual_batch(task, tiny_kwargs, stub_batch):
    r"""$58$ is now the with-scattering width and used to be the phase-only one, so this exact
    misconfiguration passes every config-shaped check. Only the batch can catch it, and the
    message must name the per-field widths."""
    module = task(model_kwargs=dict(tiny_kwargs, use_up_st=False, c_u=58))

    with pytest.raises(RuntimeError) as excinfo:
        module._build_source_stream(stub_batch)

    message = str(excinfo.value)
    for fragment in ("up_ph=15", "c_u=58", "use_up_st=False", "model_config.VAE_model.c_u"):
        assert fragment in message, f"{fragment!r} missing from: {message}"


def test_a_batch_from_a_pre_migration_shard_is_caught(task, stub_batch):
    """The other direction: a correct config pointed at an old-width HDF5."""
    module = task()  # c_u=58, use_up_st=True; an old shard makes the stream 43+58=101
    stub_batch.up_ph = torch.randn(stub_batch.up_st.shape[0], stub_batch.up_st.shape[1], 58)

    with pytest.raises(RuntimeError, match="source stream is 101 channels"):
        module._build_source_stream(stub_batch)


def test_the_target_width_is_checked_too(task, stub_batch):
    module = task()
    stub_batch.fhr_ph = torch.randn(stub_batch.fhr_st.shape[0], stub_batch.fhr_st.shape[1], 44)

    with pytest.raises(RuntimeError, match="target stream is 87 channels"):
        module._build_target_streams(stub_batch)


# --------------------------------------------------------------------------------------
# KL semantics and diagnostics
# --------------------------------------------------------------------------------------
def test_the_raw_kl_is_reported_separately_from_the_trained_one(
    task, stub_batch, perturb_posterior
):
    """Only ``source_conditioned_kl_raw`` may be read as an information rate; the trained one
    is free-bits floored. With a positive floor they genuinely differ, which is what keeps this
    assertion from passing vacuously."""
    module = task(hparams={"free_bits": 0.5})
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["source_conditioned_kl_train"]) > float(
        metrics["source_conditioned_kl_raw"]
    )


def test_the_latent_gap_is_zero_at_init_and_positive_once_perturbed(
    task, stub_batch, perturb_posterior
):
    """The zero-init invariant seen through the diagnostic -- and the reason every KL assertion
    in this suite perturbs first: at init the posterior *is* the prior."""
    module = task()

    _, at_init = module.compute_loss_and_metrics(stub_batch, 1, "train")
    assert float(at_init["mu_post_prior_gap_rms"]) == pytest.approx(0.0, abs=1e-6)

    perturb_posterior(module.orig_model)
    _, perturbed = module.compute_loss_and_metrics(stub_batch, 1, "train")
    assert float(perturbed["mu_post_prior_gap_rms"]) > 0.0


def test_the_gap_diagnostic_is_the_per_step_belief_shift_not_the_per_element_rms(
    task, stub_batch, perturb_posterior
):
    r"""``mu_post_prior_gap_rms`` sums over $d_z$ before the root; ``delta_mu_rms`` does not.
    The two would silently collapse into one number if the sum were dropped."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    d_z = int(module.orig_model.d_z)
    assert float(metrics["mu_post_prior_gap_rms"]) == pytest.approx(
        float(metrics["delta_mu_rms"]) * d_z**0.5, rel=1e-4
    )


def test_the_validity_mask_changes_the_loss(task, make_stub_batch_fn, perturb_posterior):
    """A weight the loss ignored would let gaps pollute every term, silently."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch_fn()

    torch.manual_seed(1)
    _, all_valid = module.compute_loss_and_metrics(batch, 1, "train")
    batch.weight[:, : batch.weight.shape[1] // 2] = 0.0
    torch.manual_seed(1)
    _, half_masked = module.compute_loss_and_metrics(batch, 1, "train")

    assert float(all_valid["nll_full_block"]) != pytest.approx(
        float(half_masked["nll_full_block"]), rel=1e-6
    )


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
        hparams={
            "beta_schedule": {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 10}
        }
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


def test_the_scheduled_beta_is_what_weights_the_kl_and_what_is_reported(
    task, stub_batch, perturb_posterior
):
    """``kld_beta`` in the metrics must be the resolved value, not the raw hparam; they differ
    the moment a schedule exists, and the plots read the reported one."""
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


def test_the_configured_beta_prior_weights_the_objective_and_is_echoed(
    task, stub_batch, perturb_posterior
):
    """The hparam reaches the loss by value, seen through the task: two identically-seeded
    steps that differ only in ``beta_prior`` must differ in the total by exactly the weighted
    prior rate, and the metric must echo the configured constant."""
    anchored = task(hparams={"beta_prior": 0.5})
    unanchored = task(hparams={"beta_prior": 0.0})
    perturb_posterior(anchored.orig_model)
    perturb_posterior(unanchored.orig_model)  # same seed in the factory -> identical weights

    torch.manual_seed(2)
    loss_anchored, metrics = anchored.compute_loss_and_metrics(stub_batch, 1, "train")
    torch.manual_seed(2)
    loss_unanchored, _ = unanchored.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["beta_prior"]) == pytest.approx(0.5)
    assert float(metrics["prior_rate"]) > 0.0
    assert float(loss_anchored - loss_unanchored) == pytest.approx(
        0.5 * float(metrics["prior_rate"]), rel=1e-4
    )


@pytest.mark.parametrize(
    ("hparam", "metric"),
    [
        ("lambda_ms", "aux_multiscale"),
        ("lambda_deriv", "aux_derivative"),
        ("lambda_boundary", "aux_boundary"),
    ],
)
def test_each_configured_shape_weight_weights_the_objective_and_is_echoed(
    task, stub_batch, perturb_posterior, hparam, metric
):
    """The ``beta_prior`` pattern, per shape term: the hparam reaches the loss by value through
    the task, the metric echoes it, and the term's own readout is what the totals differ by. The
    off run also pins the zeros-when-off contract at the task level -- an unweighted term is
    reported as an exact zero, never as its would-be value."""
    weighted = task(hparams={hparam: 0.5})
    unweighted = task(hparams={hparam: 0.0})
    perturb_posterior(weighted.orig_model)
    perturb_posterior(unweighted.orig_model)  # same seed in the factory -> identical weights

    torch.manual_seed(2)
    loss_weighted, metrics = weighted.compute_loss_and_metrics(stub_batch, 1, "train")
    torch.manual_seed(2)
    loss_unweighted, off_metrics = unweighted.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics[hparam]) == pytest.approx(0.5)
    assert float(metrics[metric]) > 0.0
    assert float(off_metrics[metric]) == 0.0
    assert float(loss_weighted - loss_unweighted) == pytest.approx(
        0.5 * float(metrics[metric]), rel=1e-4
    )


def test_the_shape_weights_default_to_zero_and_round_trip_through_the_hparams(task):
    """They land in ``self.hparams`` -- and therefore in every checkpoint -- so a run's objective
    stays recoverable from its checkpoint alone. Config-less operation resolves all three to
    ``0.0``, which is the four-term objective every pre-existing run was trained under."""
    default = task()
    configured = task(hparams={"lambda_ms": 0.13, "lambda_deriv": 0.17, "lambda_boundary": 0.19})

    for name, value in (("lambda_ms", 0.13), ("lambda_deriv", 0.17), ("lambda_boundary", 0.19)):
        assert float(default.hparams[name]) == 0.0, name
        assert float(configured.hparams[name]) == pytest.approx(value), name


def test_the_permutation_control_is_unchanged_by_the_shape_weights(
    task, make_stub_batch_fn, perturb_posterior
):
    """The control consumes only its own NLL, so the shape terms are passed 0.0 there rather
    than the configured weights -- driven at absurd weights so any leak into the shuffled
    scoring would be unmissable."""
    absurd = {"lambda_ms": 1.0e3, "lambda_deriv": 1.0e3, "lambda_boundary": 1.0e3}
    weighted = task(hparams=absurd)
    unweighted = task(hparams={name: 0.0 for name in absurd})
    perturb_posterior(weighted.orig_model)
    perturb_posterior(unweighted.orig_model)  # same seed in the factory -> identical weights

    torch.manual_seed(3)
    _, with_shape = weighted.compute_loss_and_metrics(make_stub_batch_fn(), 0, "val")
    torch.manual_seed(3)
    _, without = unweighted.compute_loss_and_metrics(make_stub_batch_fn(), 0, "val")

    for name in ("nll_shuffled_block", "kld_shuffled", "shuffle_penalty"):
        assert torch.equal(with_shape[name], without[name]), name


def test_the_permutation_control_is_unchanged_by_beta_prior(task, make_stub_batch_fn, perturb_posterior):
    """The control re-scores the full branch under a stranger's source and leaves the prior
    untouched, so its three readouts must be bitwise identical whatever the anchor weight --
    driven at an absurd weight so any leak into the shuffled scoring would be unmissable."""
    anchored = task(hparams={"beta_prior": 1.0e3})
    unanchored = task(hparams={"beta_prior": 0.0})
    perturb_posterior(anchored.orig_model)
    perturb_posterior(unanchored.orig_model)  # same seed in the factory -> identical weights

    torch.manual_seed(3)
    _, with_anchor = anchored.compute_loss_and_metrics(make_stub_batch_fn(), 0, "val")
    torch.manual_seed(3)
    _, without = unanchored.compute_loss_and_metrics(make_stub_batch_fn(), 0, "val")

    for name in ("nll_shuffled_block", "kld_shuffled", "shuffle_penalty"):
        assert torch.equal(with_anchor[name], without[name]), name


# --------------------------------------------------------------------------------------
# Peak-memory telemetry
# --------------------------------------------------------------------------------------
def _capture_peak_memory_lines(module, calls=2):
    """Fire the hook ``calls`` times and return the peak-memory log lines it emitted."""
    from loguru import logger as loguru_logger

    messages = []
    sink_id = loguru_logger.add(messages.append, level="INFO", format="{message}")
    try:
        for batch_idx in range(calls):
            module.on_train_batch_end(None, None, batch_idx)
    finally:
        loguru_logger.remove(sink_id)
    return [message for message in messages if "peak CUDA memory" in message]


def test_peak_memory_telemetry_is_silent_off_the_gpu(task):
    """On a CPU module the counters do not exist; the hook must be a no-op, not an error --
    every CPU test and the tiny smoke fit pass through it."""
    module = task()

    assert _capture_peak_memory_lines(module) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_peak_memory_telemetry_logs_exactly_once_on_the_gpu(task):
    """Once per run, not once per batch or per epoch: the number is the first step's
    high-water mark, and repeating it would bury it in a multi-day log."""
    module = task().to("cuda")

    lines = _capture_peak_memory_lines(module, calls=3)

    assert len(lines) == 1
    assert "GiB allocated" in lines[0] and "GiB reserved" in lines[0]
    assert f"rank {module.global_rank}" in lines[0]
