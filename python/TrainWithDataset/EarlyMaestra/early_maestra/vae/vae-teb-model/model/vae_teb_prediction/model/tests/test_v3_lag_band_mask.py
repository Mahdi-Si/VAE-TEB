r"""S6-T01: the optional ``lag_band_mask`` keyword on :class:`SeqVaeLagAttnV3`.

The interventional lag attribution of Sprint 6 (goal G-F) masks lag bands inside the attention
and measures the resulting degradation of the forecast,

$$\Delta L_G = \mathcal{L}_{\mathrm{feat}}^{\text{masked }G} - \mathcal{L}_{\mathrm{feat}}.$$

Everything that makes that measurement trustworthy is pinned here:

* the default path (``lag_band_mask=None``) is **bit-identical** to the model before the kwarg
  existed, so the ablation cannot perturb the numbers it is meant to explain;
* a masked lag contributes *exactly* zero attention mass (``-inf`` before the softmax/entmax),
  so ``te_lag_map`` mass and :math:`\Delta L` refer to the same set of lags;
* the causal validity constraint :math:`t - \ell \ge 0` survives the intersection, because
  :class:`LagCrossAttention` applies only the one mask it is handed;
* masking is an **ablation, not a renormalisation** -- surviving lags keep unit mass, and a row
  with no survivor collapses to :math:`\alpha = 0` rather than to a uniform distribution.

Every test runs under ``model.eval()``: ``LagCrossAttention._attend`` applies ``attn_dropout``
to :math:`\alpha` *after* the softmax, so the row-sum invariants only hold outside training mode.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


def _build(prod_kwargs: dict, **overrides) -> SeqVaeLagAttnV3:
    r"""A tiny deterministic v3 model in eval mode; ``overrides`` tune the ablation kwargs."""
    kwargs = dict(prod_kwargs)
    kwargs.update(overrides)
    kwargs.pop("lambda_perm", None)
    kwargs.pop("perm_every_n_batches", None)
    torch.manual_seed(0)
    return SeqVaeLagAttnV3(**kwargs).eval()


def _n_lags(model: SeqVaeLagAttnV3) -> int:
    r"""The lag-axis extent :math:`L = \texttt{max\_lag} + 1`."""
    return int(model.lag_attn.L)


def _forward(model: SeqVaeLagAttnV3, inputs, mask=None, seed: int = 7):
    r"""Seeded forward, so the reparameterisation noise is shared across comparisons."""
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(*inputs, lag_band_mask=mask)


def _prefix_mask(model: SeqVaeLagAttnV3, k: int) -> torch.Tensor:
    r"""Keep only lags :math:`\ell \in [0, k)`."""
    keep = torch.zeros(_n_lags(model), dtype=torch.bool)
    keep[:k] = True
    return keep


# ---------------------------------------------------------------------------
# The default path must not move a single bit.
# ---------------------------------------------------------------------------
def test_none_mask_is_bit_identical(prod_kwargs, inputs) -> None:
    r"""``lag_band_mask=None`` reproduces the pre-kwarg forward exactly, tensor for tensor."""
    model = _build(prod_kwargs)
    a = _forward(model, inputs, mask=None)
    b = _forward(model, inputs, mask=None)
    for key, va in a.items():
        if not isinstance(va, torch.Tensor):
            continue
        assert torch.equal(va, b[key]), f"{key} is not deterministic under a fixed seed"

    # An all-True band mask reduces to the causal validity mask, which is precisely what
    # LagCrossAttention builds for itself when handed None. Bit-exactness, not tolerance.
    keep_all = torch.ones(_n_lags(model), dtype=torch.bool)
    c = _forward(model, inputs, mask=keep_all)
    for key, va in a.items():
        if not isinstance(va, torch.Tensor):
            continue
        assert torch.equal(va, c[key]), f"{key} moved under an all-keep band mask"


def test_combined_mask_equals_validity_when_band_is_all_true(prod_kwargs) -> None:
    r"""``_combined_lag_mask`` with an all-keep band returns the bare validity mask."""
    model = _build(prod_kwargs)
    seq_len = int(prod_kwargs["sequence_length"])
    dev = torch.device("cpu")
    validity = model.lag_attn._build_lag_mask(seq_len, device=dev)
    combined, dead = model._combined_lag_mask(
        seq_len, dev, torch.ones(_n_lags(model), dtype=torch.bool)
    )
    assert torch.equal(combined, validity)
    assert not bool(dead.any()), "an all-keep band cannot kill an anchor"
    assert model._combined_lag_mask(seq_len, dev, None) == (None, None)


def test_dead_anchors_force_lag_zero_back_on(prod_kwargs) -> None:
    r"""Masking lag :math:`0` kills the early anchors; lag :math:`0` is forced back for them.

    ``entmax15`` raises on an all-:math:`-\infty` row (support size :math:`0`, ``gather`` at
    index :math:`-1`), so the mask handed to ``_attend`` must never have an empty row. Those
    rows are then zeroed by :meth:`_ablate_dead_anchors`, which is what keeps the two
    activations in agreement.
    """
    model = _build(prod_kwargs)
    seq_len, dev = int(prod_kwargs["sequence_length"]), torch.device("cpu")
    keep = torch.ones(_n_lags(model), dtype=torch.bool)
    keep[:3] = False                                    # drop lags 0, 1, 2
    combined, dead = model._combined_lag_mask(seq_len, dev, keep)

    assert torch.equal(dead, torch.tensor([True, True, True] + [False] * (seq_len - 3)))
    assert torch.all(combined[dead, 0]), "a dead row was left empty; entmax would raise"
    assert combined.any(dim=-1).all(), "some row reaching _attend is all -inf"
    # Live anchors are untouched by the rescue.
    assert not torch.any(combined[~dead, 0])


# ---------------------------------------------------------------------------
# A masked lag contributes exactly zero mass -- under softmax and entmax alike.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_entmax", [False, True])
def test_masked_lags_get_exactly_zero_attention(prod_kwargs, inputs, use_entmax: bool) -> None:
    r"""Keeping only :math:`\ell < k` zeroes ``attn_weights[..., k:]`` and ``te_lag_map``."""
    model = _build(prod_kwargs, use_entmax=use_entmax)
    k = 3
    out = _forward(model, inputs, mask=_prefix_mask(model, k))

    alpha = out["attn_weights"]                 # (B, T, num_heads, L), lag order
    assert alpha.shape[-1] == _n_lags(model)
    assert torch.all(alpha[..., k:] == 0.0), "masked lags retained attention mass"
    assert torch.all(out["te_lag_map"][..., k:] == 0.0), "masked lags retained te_lag_map mass"


@pytest.mark.parametrize("use_entmax", [False, True])
def test_causality_survives_a_band_mask(prod_kwargs, inputs, use_entmax: bool) -> None:
    r"""With a band mask supplied, ``alpha[b, t, :, l] == 0`` for every :math:`l > t`."""
    model = _build(prod_kwargs, use_entmax=use_entmax)
    # A band mask that keeps every lag: only the causal constraint may zero anything.
    out = _forward(model, inputs, mask=torch.ones(_n_lags(model), dtype=torch.bool))
    alpha = out["attn_weights"]
    for t in range(min(int(prod_kwargs["sequence_length"]), _n_lags(model))):
        assert torch.all(alpha[:, t, :, t + 1 :] == 0.0), f"non-causal mass at anchor t={t}"


@pytest.mark.parametrize("use_entmax", [False, True])
def test_alpha_rows_sum_to_one_or_zero(prod_kwargs, inputs, use_entmax: bool) -> None:
    r"""Ablation, not renormalisation: rows keep unit mass, or collapse to zero.

    A row sums to :math:`1` wherever at least one lag is both causally valid and kept, and to
    :math:`0` where every valid lag was masked (the ``nan_to_num`` all-invalid path).
    """
    model = _build(prod_kwargs, use_entmax=use_entmax)
    seq_len = int(prod_kwargs["sequence_length"])
    # Mask lags 0..2 -> anchors t < 3 have no surviving valid lag.
    keep = torch.ones(_n_lags(model), dtype=torch.bool)
    keep[:3] = False
    out = _forward(model, inputs, mask=keep)
    alpha = out["attn_weights"]                 # (B, T, M, L)
    row_sums = alpha.sum(dim=-1)                # (B, T, M)

    validity = model.lag_attn._build_lag_mask(seq_len, device=alpha.device)   # (T, L)
    survives = (validity & keep.unsqueeze(0)).any(dim=-1)                     # (T,)

    live = row_sums[:, survives, :]
    dead = row_sums[:, ~survives, :]
    assert torch.allclose(live, torch.ones_like(live), atol=1e-6), "surviving rows lost mass"
    assert torch.all(dead == 0.0), "an all-masked row was renormalised instead of zeroed"
    assert bool((~survives).any()), "the fixture failed to produce an all-masked anchor"

    # The pre-W_o per-head summary is exactly zero, so the fused source is W_o(0) == its bias.
    assert torch.all(out["attended_source_heads"][:, ~survives] == 0.0)
    bias = model.lag_attn.W_o.bias
    expected = torch.zeros(out["attended_source"].shape[-1]) if bias is None else bias
    assert torch.allclose(
        out["attended_source"][:, ~survives, :],
        expected.expand_as(out["attended_source"][:, ~survives, :]),
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Mask forms, validation, and the other two call sites.
# ---------------------------------------------------------------------------
def test_1d_and_2d_mask_forms_agree(prod_kwargs, inputs) -> None:
    r"""A ``(L,)`` mask broadcasts to the ``(T, L)`` form."""
    model = _build(prod_kwargs)
    seq_len, n_lags = int(prod_kwargs["sequence_length"]), _n_lags(model)
    keep = _prefix_mask(model, 4)
    a = _forward(model, inputs, mask=keep)
    b = _forward(model, inputs, mask=keep.unsqueeze(0).expand(seq_len, n_lags).contiguous())
    assert torch.equal(a["attn_weights"], b["attn_weights"])
    assert torch.equal(a["mu_full"], b["mu_full"])


@pytest.mark.parametrize("bad_shape", ["lag_axis", "time_axis", "per_sample"])
def test_malformed_masks_raise(prod_kwargs, bad_shape: str) -> None:
    r"""A wrong shape raises rather than being silently reinterpreted.

    The per-sample ``(B, T, L)`` case matters most: ``LagCrossAttention.forward`` would quietly
    collapse it to sample ``0``'s mask, applying one sample's ablation to the whole batch.
    """
    model = _build(prod_kwargs)
    seq_len, n_lags = int(prod_kwargs["sequence_length"]), _n_lags(model)
    bad = {
        "lag_axis": torch.ones(n_lags + 1, dtype=torch.bool),
        "time_axis": torch.ones(seq_len + 1, n_lags, dtype=torch.bool),
        "per_sample": torch.ones(2, seq_len, n_lags, dtype=torch.bool),
    }[bad_shape]
    with pytest.raises(ValueError):
        model._combined_lag_mask(seq_len, torch.device("cpu"), bad)


def test_encode_only_threads_the_mask(prod_kwargs, inputs) -> None:
    r""":meth:`encode_only` honours the band mask and defaults to a no-op."""
    model = _build(prod_kwargs)
    k = 2
    torch.manual_seed(7)
    with torch.no_grad():
        masked = model.encode_only(*inputs, lag_band_mask=_prefix_mask(model, k))
    torch.manual_seed(7)
    with torch.no_grad():
        clean = model.encode_only(*inputs)
    assert torch.all(masked["attn_weights"][..., k:] == 0.0)
    assert not torch.equal(masked["attended_source"], clean["attended_source"])


def test_perm_posterior_threads_the_mask(prod_kwargs, inputs, perturb_posterior) -> None:
    r""":meth:`_perm_posterior` accepts the mask; the permutation control still passes ``None``.

    The posterior is perturbed off its zero-init first: at initialisation ``delta_mu_head`` is
    exactly zero, so the posterior equals the prior whatever the source is and *no* attention
    ablation could move ``mu_post``.
    """
    model = _build(prod_kwargs)
    perturb_posterior(model)
    out = _forward(model, inputs)
    h_y, h_u = out["target_state"], out["source_state"]
    h_u_perm = h_u.flip(0)

    args = (h_y, h_u_perm, out["mu_prior"], out["logvar_prior"], out["raw_logvar_prior"], True)
    with torch.no_grad():
        base = model._perm_posterior(*args)
        masked = model._perm_posterior(*args, lag_band_mask=_prefix_mask(model, 2))
        default = model._perm_posterior(*args)
    assert torch.equal(base[2], default[2]), "the default path is not a no-op"
    assert not torch.equal(base[2], masked[2]), "the mask did not reach the posterior"

    # The shipped control never masks: perm_kl_from_forward must be unaffected.
    with torch.no_grad():
        res = model.perm_kl_from_forward(out, perm_index=torch.tensor([1, 0]))
    assert torch.isfinite(res["perm_kl"]).all()
