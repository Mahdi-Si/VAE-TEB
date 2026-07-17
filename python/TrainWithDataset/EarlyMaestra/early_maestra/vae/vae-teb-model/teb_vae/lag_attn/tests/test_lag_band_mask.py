r"""Interventional lag-band masking: cutting lags out to see what the model was using.

The attention applies **exactly one** mask -- whatever it is handed *replaces* its own
causal-validity mask rather than intersecting with it. Handing it a bare band mask would
therefore silently switch off the constraint $t - \ell \ge 0$, and the model would start
attending to lags that do not exist. Nothing would raise; the ablation would simply report
nonsense. That is what :meth:`_combined_lag_mask` exists to prevent, and it is what these tests
are really about.

The other sharp edge is that ``entmax15`` and ``softmax`` disagree on an all-masked row: softmax
degrades to zeros, entmax raises. Band masking is the only thing that can produce such a row, so
the two normalisers are only interchangeable once the dead anchors are handled explicitly.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn


def _model(prod_kwargs, **overrides):
    torch.manual_seed(0)
    return SeqVaeLagAttn(**dict(prod_kwargs, **overrides)).eval()


def _band(num_lags, keep):
    mask = torch.zeros(num_lags, dtype=torch.bool)
    mask[keep] = True
    return mask


def test_no_mask_is_bit_identical_to_the_default_path(prod_kwargs, inputs):
    """The feature must cost nothing when unused, or every run pays for an analysis tool."""
    model = _model(prod_kwargs)
    torch.manual_seed(0)
    with torch.no_grad():
        without = model(*inputs)
    torch.manual_seed(0)
    with torch.no_grad():
        explicit_none = model(*inputs, lag_band_mask=None)
    assert torch.equal(without["attn_weights"], explicit_none["attn_weights"])
    assert torch.equal(without["z"], explicit_none["z"])


def test_the_causal_constraint_survives_band_masking(prod_kwargs, inputs):
    """The defect the helper exists for: a bare band mask would switch off causality."""
    model = _model(prod_kwargs)
    num_lags = prod_kwargs["max_lag"] + 1
    # Keep every lag: the band is permissive, so only causality can mask anything.
    combined, dead = model._combined_lag_mask(
        inputs[0].shape[1], inputs[0].device, torch.ones(num_lags, dtype=torch.bool)
    )
    assert combined is not None
    validity = model.lag_attn.build_lag_mask(inputs[0].shape[1])
    assert torch.equal(combined, validity)
    assert dead is not None and not bool(dead.any())


def test_masked_lags_receive_exactly_zero_attention(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    num_lags = prod_kwargs["max_lag"] + 1
    keep = [0, 1, 2]
    with torch.no_grad():
        out = model(*inputs, lag_band_mask=_band(num_lags, keep))

    dropped = [lag for lag in range(num_lags) if lag not in keep]
    assert torch.all(out["attn_weights"][..., dropped] == 0.0)


@pytest.mark.parametrize("use_entmax", [False, True])
def test_alpha_rows_sum_to_one_or_zero(prod_kwargs, inputs, use_entmax):
    """Ablation, not renormalisation -- and both normalisers must agree on that."""
    model = _model(prod_kwargs, use_entmax=use_entmax)
    num_lags = prod_kwargs["max_lag"] + 1
    with torch.no_grad():
        out = model(*inputs, lag_band_mask=_band(num_lags, [3, 4, 5]))

    sums = out["attn_weights"].sum(dim=-1)
    close_to_one = torch.isclose(sums, torch.ones_like(sums), atol=1e-5)
    close_to_zero = torch.isclose(sums, torch.zeros_like(sums), atol=1e-6)
    assert bool((close_to_one | close_to_zero).all())


@pytest.mark.parametrize("use_entmax", [False, True])
def test_a_band_excluding_lag_zero_does_not_crash_either_normaliser(
    prod_kwargs, inputs, use_entmax
):
    """entmax raises on an all-masked row; softmax does not. Only this path creates one."""
    model = _model(prod_kwargs, use_entmax=use_entmax)
    num_lags = prod_kwargs["max_lag"] + 1
    band = _band(num_lags, list(range(5, num_lags)))  # excludes lag 0

    with torch.no_grad():
        out = model(*inputs, lag_band_mask=band)
    assert torch.isfinite(out["attn_weights"]).all()
    assert torch.isfinite(out["z"]).all()


def test_dead_anchors_are_reported_and_ablated(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    seq_len = inputs[0].shape[1]
    num_lags = prod_kwargs["max_lag"] + 1
    band = _band(num_lags, list(range(5, num_lags)))

    combined, dead = model._combined_lag_mask(seq_len, inputs[0].device, band)
    assert dead is not None
    # Anchors before the band's first lag have no surviving valid lag.
    assert bool(dead[:5].all())
    assert not bool(dead[5:].any())
    # Lag 0 is forced back on there purely to keep the activation well-posed...
    assert combined is not None and bool(combined[dead, 0].all())

    # ...and the resulting rows are then discarded.
    with torch.no_grad():
        out = model(*inputs, lag_band_mask=band)
    assert torch.all(out["attn_weights"][:, dead] == 0.0)


def test_a_dead_anchor_yields_the_output_projection_bias_not_zero(prod_kwargs, inputs):
    r"""$A = W_o(0)$ is $W_o$'s *bias*, which on a trained model is not zero.

    The bias must be set nonzero first. ``initialization()`` zeroes every ``Linear`` bias, so on
    a fresh model $W_o$'s bias *is* all-zeros and this assertion could not tell the correct
    implementation from one that wrote a plain ``0.0`` at dead anchors -- the defect would only
    appear on a trained checkpoint, where it silently reintroduces the softmax/entmax
    disagreement the ablation path exists to remove.
    """
    model = _model(prod_kwargs)
    with torch.no_grad():
        model.lag_attn.W_o.bias.copy_(
            torch.randn(model.d_model, generator=torch.Generator().manual_seed(7))
        )
    assert model.lag_attn.W_o.bias.abs().max().item() > 0.0, "the probe needs a nonzero bias"

    num_lags = prod_kwargs["max_lag"] + 1
    band = _band(num_lags, list(range(5, num_lags)))
    with torch.no_grad():
        out = model(*inputs, lag_band_mask=band)

    assert torch.allclose(out["attended_source"][:, 0, :], model.lag_attn.W_o.bias, atol=1e-6)
    # The per-head summaries are pre-projection, so those really are zero.
    assert torch.all(out["attended_source_heads"][:, 0] == 0.0)


def test_a_dead_anchor_is_not_merely_zeroed(prod_kwargs, inputs):
    """The mirror of the test above: it must be able to fail."""
    model = _model(prod_kwargs)
    with torch.no_grad():
        model.lag_attn.W_o.bias.fill_(0.5)

    num_lags = prod_kwargs["max_lag"] + 1
    with torch.no_grad():
        out = model(*inputs, lag_band_mask=_band(num_lags, list(range(5, num_lags))))

    assert not torch.allclose(
        out["attended_source"][:, 0, :], torch.zeros(model.d_model), atol=1e-6
    )


def test_one_dimensional_and_two_dimensional_masks_agree(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    seq_len = inputs[0].shape[1]
    num_lags = prod_kwargs["max_lag"] + 1
    flat = _band(num_lags, [2, 3, 4])
    expanded = flat[None, :].expand(seq_len, num_lags).contiguous()

    torch.manual_seed(0)
    with torch.no_grad():
        from_flat = model(*inputs, lag_band_mask=flat)
    torch.manual_seed(0)
    with torch.no_grad():
        from_expanded = model(*inputs, lag_band_mask=expanded)
    assert torch.equal(from_flat["attn_weights"], from_expanded["attn_weights"])


def test_a_time_varying_mask_is_honoured(prod_kwargs, inputs):
    """The 2-D form is not just a broadcast convenience; it must actually vary with time."""
    model = _model(prod_kwargs)
    seq_len = inputs[0].shape[1]
    num_lags = prod_kwargs["max_lag"] + 1

    mask = torch.zeros(seq_len, num_lags, dtype=torch.bool)
    mask[:, 0] = True
    mask[seq_len // 2 :, 1] = True

    with torch.no_grad():
        out = model(*inputs, lag_band_mask=mask)

    assert torch.all(out["attn_weights"][:, : seq_len // 2, :, 1] == 0.0)
    assert torch.all(out["attn_weights"][..., 2:] == 0.0)


def test_encode_only_threads_the_mask(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    num_lags = prod_kwargs["max_lag"] + 1
    band = _band(num_lags, [0, 1])
    with torch.no_grad():
        out = model.encode_only(*inputs, lag_band_mask=band)
    assert torch.all(out["attn_weights"][..., 2:] == 0.0)


def test_a_wrong_length_mask_raises(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    with pytest.raises(ValueError, match="lag axis"):
        model(*inputs, lag_band_mask=torch.ones(3, dtype=torch.bool))


def test_a_wrong_shape_two_dimensional_mask_raises(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    num_lags = prod_kwargs["max_lag"] + 1
    with pytest.raises(ValueError, match=r"is not \(T, L\)"):
        model(*inputs, lag_band_mask=torch.ones(3, num_lags, dtype=torch.bool))


def test_a_three_dimensional_mask_raises(prod_kwargs, inputs):
    """Rejected rather than silently collapsed: per-sample masks are not expressible here."""
    model = _model(prod_kwargs)
    seq_len = inputs[0].shape[1]
    num_lags = prod_kwargs["max_lag"] + 1
    with pytest.raises(ValueError, match="must be 1-D"):
        model(*inputs, lag_band_mask=torch.ones(2, seq_len, num_lags, dtype=torch.bool))
