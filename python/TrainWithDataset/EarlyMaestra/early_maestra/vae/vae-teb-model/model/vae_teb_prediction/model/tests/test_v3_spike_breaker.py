r"""S4-T03b: the loss-spike circuit breaker watches the **main** loss, not the total.

v1's breaker EMAs the returned ``total_loss``. In v3 that value jumps every
``perm_every_n_batches`` steps, because :math:`\lambda_{\mathrm{perm}} L_{\mathrm{perm}}` is
added only on scheduled steps. Feeding the breaker the total would make its own statistic
periodic: the EMA settles on the perm-free level and every perm step then looks like a spike,
so the optimizer would silently skip exactly the steps that carry the control.

The learned-variance switch (G7) compounds this -- Gaussian NLL lives on a different scale
than the MSE the v1 spike defaults were tuned against.

Note:
    ``training_step`` reparameterises :math:`z`, so the loss is stochastic. Every step below is
    seeded from a fixed schedule; otherwise the two runs in
    :func:`test_ema_matches_a_perm_free_run` would consume different noise and the comparison
    would be meaningless.
"""
from __future__ import annotations

import math

import torch


def _install_skip_probe(pl_module):
    """Record whether each ``training_step`` decided to skip, without touching the wrapper."""
    pl_module._spike_skipped_last = False
    original = pl_module._sync_skip_decision_across_ranks

    def probe(is_spike, device):
        decided = original(is_spike, device)
        pl_module._spike_skipped_last = decided
        return decided

    pl_module._sync_skip_decision_across_ranks = probe
    return pl_module


def _drive(pl_module, batch, n_steps: int):
    """Run ``n_steps`` seeded training steps; return the per-step skip flags."""
    skips = []
    for i in range(n_steps):
        torch.manual_seed(1000 + i)  # identical reparameterisation noise across runs
        loss = pl_module.training_step(batch, i)
        skips.append(bool(pl_module._spike_skipped_last))
        assert torch.isfinite(loss), f"non-finite loss returned at step {i}"
    return skips


def _explode(pl_module, monkeypatch, factor: float):
    """Scale the main loss by ``factor`` so the breaker sees a genuine jump."""
    original = pl_module.compute_loss_and_metrics

    def scaled(batch, batch_idx, stage):
        loss, metrics = original(batch, batch_idx, stage)
        blown = loss * factor
        metrics["main_loss"] = blown.detach()
        return blown, metrics

    monkeypatch.setattr(pl_module, "compute_loss_and_metrics", scaled)


def test_periodic_perm_steps_do_not_trip_the_breaker(v3_pl, stub_batch, perturb_posterior):
    """The scheduled L_perm jump must not read as a spike once the EMA has primed."""
    pl_module = v3_pl(hparams={"loss_spike_skip": {"warmup_batches": 4, "multiplier": 5.0}})
    perturb_posterior(pl_module.orig_model, scale=0.3)  # make L_perm materially nonzero
    _install_skip_probe(pl_module)

    skips = _drive(pl_module, stub_batch, n_steps=12)

    assert not any(skips), f"breaker fired on a scheduled perm step: {skips}"
    assert pl_module._spike_skips_total == 0


def test_ema_matches_a_perm_free_run(v3_pl, stub_batch, perturb_posterior, prod_kwargs):
    r"""Two wrappers differing only in :math:`\lambda_{\mathrm{perm}}` must EMA identically.

    This is the load-bearing assertion: if the breaker read ``total_loss``, the run with the
    control active would carry a different EMA.
    """
    hparams = {"loss_spike_skip": {"warmup_batches": 2}}
    with_control = v3_pl(model_kwargs=dict(prod_kwargs, lambda_perm=0.5), hparams=hparams)
    without_control = v3_pl(model_kwargs=dict(prod_kwargs, lambda_perm=0.0), hparams=hparams)
    for module in (with_control, without_control):
        perturb_posterior(module.orig_model, scale=0.3)
        _install_skip_probe(module)

    _drive(with_control, stub_batch, n_steps=6)
    _drive(without_control, stub_batch, n_steps=6)

    assert with_control._spike_ema_loss == without_control._spike_ema_loss, (
        "the spike EMA depends on lambda_perm, so it is watching the total loss, not the main"
    )


def test_a_genuine_spike_is_still_caught(v3_pl, stub_batch, perturb_posterior, monkeypatch):
    """A large jump in the *main* loss must still skip the optimizer step."""
    pl_module = v3_pl(hparams={"loss_spike_skip": {"warmup_batches": 2, "multiplier": 5.0}})
    perturb_posterior(pl_module.orig_model)
    _install_skip_probe(pl_module)

    _drive(pl_module, stub_batch, n_steps=4)  # prime the EMA
    assert pl_module._spike_ema_loss is not None
    assert pl_module._spike_skips_total == 0

    _explode(pl_module, monkeypatch, factor=1000.0)
    torch.manual_seed(99)
    loss = pl_module.training_step(stub_batch, 5)

    assert pl_module._spike_skipped_last, "a 1000x main-loss jump was not flagged"
    assert float(loss) == 0.0, "a skipped step must return a zero-valued no-op loss"
    assert loss.requires_grad, "the no-op loss must stay attached so DDP still all-reduces"
    assert pl_module._spike_skips_total == 1


def test_non_finite_main_loss_is_always_a_spike(v3_pl, stub_batch, monkeypatch):
    pl_module = v3_pl(hparams={"loss_spike_skip": {"warmup_batches": 0}})
    _install_skip_probe(pl_module)
    _explode(pl_module, monkeypatch, factor=float("nan"))

    torch.manual_seed(0)
    loss = pl_module.training_step(stub_batch, 0)

    assert pl_module._spike_skipped_last
    assert math.isfinite(float(loss)), "the no-op loss must be finite even after a NaN step"


def test_a_spike_does_not_raise_the_bar_for_the_next_step(v3_pl, stub_batch,
                                                          perturb_posterior, monkeypatch):
    """The EMA must only absorb accepted batches, or one spike would mask the next."""
    pl_module = v3_pl(hparams={"loss_spike_skip": {"warmup_batches": 2, "multiplier": 5.0}})
    perturb_posterior(pl_module.orig_model)
    _install_skip_probe(pl_module)
    _drive(pl_module, stub_batch, n_steps=4)
    ema_before = pl_module._spike_ema_loss

    _explode(pl_module, monkeypatch, factor=1000.0)
    torch.manual_seed(99)
    pl_module.training_step(stub_batch, 5)

    assert pl_module._spike_ema_loss == ema_before, "a skipped batch polluted the EMA"


def test_breaker_can_be_disabled(v3_pl, stub_batch, monkeypatch):
    pl_module = v3_pl(hparams={"loss_spike_skip": {"enabled": False}})
    _install_skip_probe(pl_module)
    _explode(pl_module, monkeypatch, factor=1.0e6)

    for i in range(4):
        torch.manual_seed(i)
        pl_module.training_step(stub_batch, i)

    assert pl_module._spike_skips_total == 0


def test_consecutive_skips_force_accept_and_reseed(v3_pl, stub_batch, perturb_posterior,
                                                   monkeypatch):
    """After ``max_consecutive_skips`` skips in a row, force-accept one finite batch.

    The breaker takes the real gradient step, HARD-re-seeds the EMA to that batch's loss,
    and resets the run-length -- so a collapsed EMA (the near-zero Gaussian-NLL regime that
    froze the headline run) can never skip every batch indefinitely.
    """
    cap = 3
    pl_module = v3_pl(hparams={"loss_spike_skip": {
        "warmup_batches": 2, "multiplier": 5.0, "max_consecutive_skips": cap}})
    perturb_posterior(pl_module.orig_model)
    _install_skip_probe(pl_module)

    _drive(pl_module, stub_batch, n_steps=4)  # prime the EMA
    ema_before = pl_module._spike_ema_loss
    assert pl_module._spike_skips_total == 0

    _explode(pl_module, monkeypatch, factor=1000.0)

    # A skipped step returns the zero no-op; a force-accepted step returns the real loss.
    # (``_spike_skipped_last`` from the probe records the PRE-hatch decision, so it cannot
    # distinguish a force-accept -- assert on the returned loss / counters instead.)
    outs = []
    for step in range(cap + 1):
        torch.manual_seed(200 + step)
        outs.append(pl_module.training_step(stub_batch, step))

    for step in range(cap):
        assert float(outs[step]) == 0.0, f"step {step} should be skipped (zero no-op)"
    forced = outs[cap]

    assert pl_module._spike_skips_total == cap, "a forced accept must not count as a skip"
    assert pl_module._spike_forced_accepts_total == 1
    assert pl_module._spike_consecutive == 0, "the run-length must reset after a forced accept"
    # Hard re-seed (not a momentum blend): the EMA jumps to the forced step's main loss.
    assert pl_module._spike_ema_loss != ema_before
    assert pl_module._spike_ema_loss == float(forced)
    # The forced step returns the REAL loss (attached, nonzero), not the zero no-op.
    assert forced.requires_grad and torch.isfinite(forced) and float(forced) != 0.0


def test_nonfinite_is_never_force_accepted_even_at_cap(v3_pl, stub_batch, monkeypatch):
    """A non-finite loss is ALWAYS skipped -- the hatch must never force-accept a NaN.

    Force-accepting a NaN would take a NaN gradient step and, under DDP, diverge the
    accept/skip autograd branch across ranks. The finite gate must hold even past the cap.
    """
    cap = 3
    pl_module = v3_pl(hparams={"loss_spike_skip": {
        "warmup_batches": 0, "multiplier": 5.0, "max_consecutive_skips": cap}})
    _install_skip_probe(pl_module)
    _explode(pl_module, monkeypatch, factor=float("nan"))

    for step in range(cap + 3):  # well past the cap
        torch.manual_seed(300 + step)
        out = pl_module.training_step(stub_batch, step)
        assert pl_module._spike_skipped_last, f"NaN step {step} must always skip"
        assert math.isfinite(float(out)), "the no-op loss must stay finite"

    assert pl_module._spike_forced_accepts_total == 0
    assert pl_module._spike_skips_total == cap + 3
