r"""The masked source-conditioned KL: raw vs trained, free bits, and the activity count.

The raw/train split is the units contract's sharpest edge: only the un-floored raw value may
ever be read as an information rate, so the tests pin that the floor moves exactly one of the
two, that neither sees anything outside the anchor support, and that the raw value carries no
gradient back into training.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.losses import masked_source_kl
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

_TINY = TrimmedRawGeometry(raw_len=256, decimation=16, horizon=4, warmup=2)
_B, _T, _D_Z = 2, 16, 8

_EXPECTED_KEYS = {
    "source_conditioned_kl_raw",
    "source_conditioned_kl_train",
    "kld_active_frac",
}


def _mask() -> torch.Tensor:
    forecast, _ = forecast_mask(torch.ones(_B, _T), _TINY)
    return kl_mask(forecast, _TINY)


def _kld(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(_B, _T, _D_Z, generator=generator)  # non-negative, like a real KL


def test_the_returned_names_reference_the_kl_not_transfer_entropy():
    out = masked_source_kl(_kld(), _mask())
    assert set(out) == _EXPECTED_KEYS
    for name in out:
        assert "te" not in name.split("_"), name
        assert "transfer" not in name, name


def test_with_zero_free_bits_raw_equals_train():
    out = masked_source_kl(_kld(), _mask(), free_bits=0.0)
    assert torch.equal(
        out["source_conditioned_kl_raw"], out["source_conditioned_kl_train"]
    )


def test_free_bits_floors_the_trained_kl_only():
    """Every dim below the floor: train reads the floor, raw reads the truth."""
    kld_btd = torch.full((_B, _T, _D_Z), 1.0e-3)
    out = masked_source_kl(kld_btd, _mask(), free_bits=0.01)
    assert torch.allclose(out["source_conditioned_kl_train"], torch.tensor(_D_Z * 0.01))
    assert torch.allclose(out["source_conditioned_kl_raw"], torch.tensor(_D_Z * 1.0e-3))


def test_the_raw_value_is_detached_and_the_trained_one_is_not():
    kld_btd = _kld().requires_grad_(True)
    out = masked_source_kl(kld_btd, _mask())
    assert out["source_conditioned_kl_train"].requires_grad
    assert not out["source_conditioned_kl_raw"].requires_grad


def test_values_outside_the_anchor_support_contribute_nothing():
    """The KL support is the decoded anchor set [warmup, T - H); poisoning everything outside
    it must leave both scalars bitwise unchanged."""
    kld_btd = _kld()
    mask = _mask()
    poisoned = kld_btd.clone()
    poisoned[:, : _TINY.warmup] = 1.0e6
    poisoned[:, _TINY.t_valid :] = 1.0e6

    reference = masked_source_kl(kld_btd, mask)
    altered = masked_source_kl(poisoned, mask)
    for key in ("source_conditioned_kl_raw", "source_conditioned_kl_train"):
        assert torch.equal(reference[key], altered[key])


def test_kld_active_frac_counts_dimensions_above_the_threshold():
    kld_btd = torch.zeros(_B, _T, _D_Z)
    kld_btd[..., :3] = 0.05    # clearly active
    kld_btd[..., 3:] = 1.0e-4  # collapsed
    out = masked_source_kl(kld_btd, _mask())
    assert torch.allclose(out["kld_active_frac"], torch.tensor(3.0 / _D_Z))


def test_an_empty_mask_returns_zeros_not_nan():
    out = masked_source_kl(_kld(), torch.zeros(_B, _T))
    for key in _EXPECTED_KEYS:
        assert float(out[key]) == 0.0


def test_the_model_kld_tensor_matches_the_closed_form(tiny_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**tiny_kwargs)
    generator = torch.Generator().manual_seed(5)
    mu_p = torch.randn(_B, _T, _D_Z, generator=generator)
    lv_p = torch.randn(_B, _T, _D_Z, generator=generator)
    mu_q = torch.randn(_B, _T, _D_Z, generator=generator)
    lv_q = torch.randn(_B, _T, _D_Z, generator=generator)

    got = model.kld_tensor(
        mu_prior=mu_p, logvar_prior=lv_p, mu_post=mu_q, logvar_post=lv_q
    )
    want = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0)
    assert torch.equal(got, want)
    # Identical distributions -> exactly zero, per element.
    assert float(
        model.kld_tensor(
            mu_prior=mu_p, logvar_prior=lv_p, mu_post=mu_p, logvar_post=lv_p
        ).abs().max()
    ) == 0.0


def test_the_forward_kl_readout_is_the_kld_tensor_summed_over_dims(
    tiny_kwargs, inputs, perturb_posterior
):
    """Perturbed first -- at init both sides are identically zero and this would prove
    nothing."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**tiny_kwargs).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    kld_btd = model.kld_tensor(
        mu_prior=out["mu_prior"],
        logvar_prior=out["logvar_prior"],
        mu_post=out["mu_post"],
        logvar_post=out["logvar_post"],
    )
    assert float(kld_btd.abs().max()) > 0.0
    assert torch.allclose(out["kld_per_t"], kld_btd.sum(dim=-1), atol=1e-6)
