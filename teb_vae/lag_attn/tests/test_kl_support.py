r"""Which timesteps the training KL is allowed to see.

``'full'`` masks only the warm-up prefix. ``'anchor'`` additionally drops the final $H_d$ steps,
and that difference is a research axis rather than a tidying-up.

Those tail anchors have no fully-observed forecast window, so the reconstruction term gives them
no gradient. Under ``'full'`` they are still regularised toward the prior by $\beta$, with
nothing pulling back -- so their KL decays, and the resulting droop at the end of every plotted
$K_t$ curve looks exactly like the coupling genuinely fading. It is not: it is the tail of the
sequence being pulled toward the prior by an unopposed penalty.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

_SEQ_LEN = 16
_BATCH = 2


def _latents(d_z, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(_BATCH, _SEQ_LEN, d_z, generator=generator),        # mu_prior
        torch.randn(_BATCH, _SEQ_LEN, d_z, generator=generator) * 0.3,  # logvar_prior
        torch.randn(_BATCH, _SEQ_LEN, d_z, generator=generator),        # mu_post
        torch.randn(_BATCH, _SEQ_LEN, d_z, generator=generator) * 0.3,  # logvar_post
    )


def _model(tiny_kwargs, support):
    torch.manual_seed(0)
    return SeqVaeLagAttn(**dict(tiny_kwargs, kld_support=support))


def test_support_masks_cover_the_intended_windows(tiny_kwargs):
    warmup, horizon = tiny_kwargs["warmup_period"], tiny_kwargs["horizon"]
    full = _model(tiny_kwargs, "full")._kld_support_mask(_SEQ_LEN)
    anchor = _model(tiny_kwargs, "anchor")._kld_support_mask(_SEQ_LEN)

    assert full.sum().item() == float(_SEQ_LEN - warmup)                    # [warmup, T)
    assert anchor.sum().item() == float((_SEQ_LEN - horizon) - warmup)      # [warmup, T-H)
    assert torch.all(full[:warmup] == 0.0) and torch.all(full[warmup:] == 1.0)
    assert torch.all(anchor[_SEQ_LEN - horizon :] == 0.0)
    assert torch.all(anchor[warmup : _SEQ_LEN - horizon] == 1.0)


def test_reduce_mean_counts_every_in_support_entry(tiny_kwargs):
    d_z = tiny_kwargs["d_z"]
    latents = _latents(d_z)

    for support in ("full", "anchor"):
        model = _model(tiny_kwargs, support)
        got = model._kld_loss(*latents, reduce_mean=True, free_bits=0.0)

        kld = model.kld_tensor(*latents)
        mask_btd = model._kld_support_mask(_SEQ_LEN).view(1, _SEQ_LEN, 1).expand(_BATCH, _SEQ_LEN, 1)
        expected = (kld * mask_btd).sum() / (mask_btd.sum() * d_z)
        assert torch.allclose(got, expected, atol=1e-6), f"denominator mismatch ({support})"


def test_anchor_support_excludes_a_tail_spike(tiny_kwargs):
    """The whole point of the axis, stated as a difference."""
    d_z, horizon = tiny_kwargs["d_z"], tiny_kwargs["horizon"]
    mu_prior, logvar_prior, mu_post, logvar_post = _latents(d_z, seed=7)
    mu_post = mu_post.clone()
    mu_post[:, _SEQ_LEN - horizon :, :] += 50.0  # a huge posterior drift, tail only

    anchor = _model(tiny_kwargs, "anchor")._kld_loss(
        mu_prior, logvar_prior, mu_post, logvar_post, reduce_mean=True
    )
    full = _model(tiny_kwargs, "full")._kld_loss(
        mu_prior, logvar_prior, mu_post, logvar_post, reduce_mean=True
    )
    assert full > anchor, "a tail spike should inflate full support but not anchor support"


def test_both_supports_mask_the_warmup_prefix(tiny_kwargs):
    d_z, warmup = tiny_kwargs["d_z"], tiny_kwargs["warmup_period"]
    mu_prior, logvar_prior, mu_post, logvar_post = _latents(d_z, seed=5)
    mu_post = mu_post.clone()
    mu_post[:, :warmup, :] += 50.0  # a huge drift confined to the warm-up prefix

    for support in ("full", "anchor"):
        model = _model(tiny_kwargs, support)
        spiked = model._kld_loss(mu_prior, logvar_prior, mu_post, logvar_post, reduce_mean=True)
        clean = model._kld_loss(
            mu_prior, logvar_prior, _latents(d_z, seed=5)[2], logvar_post, reduce_mean=True
        )
        assert torch.allclose(spiked, clean, atol=1e-6), f"warm-up leaked into {support}"


def test_free_bits_raises_the_aggregate(tiny_kwargs):
    latents = _latents(tiny_kwargs["d_z"])
    model = _model(tiny_kwargs, "anchor")
    floored = model._kld_loss(*latents, reduce_mean=True, free_bits=5.0)
    unfloored = model._kld_loss(*latents, reduce_mean=True, free_bits=0.0)
    assert floored.item() > unfloored.item()


def test_an_empty_support_returns_zero_rather_than_dividing_by_zero(tiny_kwargs):
    """A short sequence under 'anchor' can legitimately leave no supported step at all."""
    model = _model(tiny_kwargs, "anchor")
    d_z = tiny_kwargs["d_z"]
    short = tiny_kwargs["warmup_period"] + tiny_kwargs["horizon"]
    generator = torch.Generator().manual_seed(0)
    latents = tuple(torch.randn(_BATCH, short, d_z, generator=generator) for _ in range(4))

    assert model._kld_support_mask(short).sum().item() == 0.0
    assert model._kld_loss(*latents, reduce_mean=True).item() == 0.0


def test_transfer_entropy_scalar_and_curve_share_one_support(
    tiny_kwargs, inputs, perturb_posterior
):
    """If they could differ, the plotted curve would show a spike the reported scalar excludes.

    The posterior must be perturbed first. On a fresh model every in-support KL is exactly $0$,
    so ``scalar`` and ``nanmean(curve)`` are both $0.0$ and the comparison holds no matter what
    the two code paths compute -- including if the scalar's denominator disagreed with the
    curve's support, which is the one defect this test exists to catch.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(tiny_kwargs, kld_support="anchor")).eval()
    perturb_posterior(model)
    warmup, horizon = tiny_kwargs["warmup_period"], tiny_kwargs["horizon"]

    scalar = model.measure_transfer_entropy(*inputs, reduce_mean=True)
    curve = model.measure_transfer_entropy(*inputs, reduce_mean=False)

    assert scalar.item() > 1e-6, "the perturbation did not make the KL nonzero"
    assert torch.isnan(curve[:, :warmup, :]).all(), "warm-up not masked"
    assert torch.isnan(curve[:, _SEQ_LEN - horizon :, :]).all(), "anchor tail not masked"
    assert torch.isfinite(curve[:, warmup : _SEQ_LEN - horizon, :]).all(), "support has NaNs"
    assert torch.allclose(scalar, torch.nanmean(curve), atol=1e-6), (
        "the scalar is not the nanmean of the per-step curve"
    )


def test_full_support_transfer_entropy_masks_only_the_warmup(
    tiny_kwargs, inputs, perturb_posterior
):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(tiny_kwargs, kld_support="full")).eval()
    perturb_posterior(model)
    warmup = tiny_kwargs["warmup_period"]

    curve = model.measure_transfer_entropy(*inputs, reduce_mean=False)
    assert torch.isnan(curve[:, :warmup, :]).all()
    assert torch.isfinite(curve[:, warmup:, :]).all()
    assert curve[:, warmup:, :].abs().max().item() > 1e-6


def test_measuring_transfer_entropy_does_not_leave_the_model_in_eval_mode(tiny_kwargs, inputs):
    """It is routinely called from a plotting callback mid-training.

    A version that left the module in eval would silently disable dropout for the rest of the
    run, training a differently-regularised model than the config asked for, with no error and
    nothing in the log to show for it.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**tiny_kwargs)
    model.train()

    model.measure_transfer_entropy(*inputs)
    assert model.training, "training mode was not restored"

    model.eval()
    model.measure_transfer_entropy(*inputs)
    assert not model.training, "eval mode was not preserved"
