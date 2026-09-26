r"""Step-wise causality and prefix equivalence through the whole model.

The dataset's causality is a property of its filters and is asserted where the shards are built.
What is asserted here is the *network's*: that anchor $t$'s outputs are a function of steps up to $t$
and of nothing after. Both halves are needed, and neither implies the other -- a transform that only
reads the past, fed to a network that pools statistics over the whole sequence, produces a coupling
readout that conditions on the future.

**This is the cell where the two halves finally meet.** Everywhere else in the grid one side of the
objective still carries its own future: the two-sided cells read coefficients averaged over both
sides of $t$, and the causal-feature cells forecast a target that is itself such an average, one-sided
but group-delayed. Here the inputs carry no future and the target is the signal itself, so the claim
this file makes is the one the package exists for -- and a network-level leak would undo it silently,
with the dataset's own guarantee still intact.

Two separately falsifiable properties, both through the assembled model:

1. **Step-wise causality.** Resampling every stream strictly after $t$ leaves the forward at $t$
   bitwise unchanged -- the prior, the encoder states, and the forecasts at anchors up to $t$, which
   is the part an encoder-level test cannot reach.
2. **Prefix equivalence**, $\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$: running the model on a
   truncated sequence reproduces the full run's outputs at every surviving step. Strictly stronger
   than causality, which can hold while this fails if anything is computed relative to the sequence
   end.

**The warm-up mask cannot break either.** It is a function of $t$ and of the channel alone --
$m_{t,c} = \mathbb 1[t \ge W'_c]$, a construction-time buffer sliced to the batch's length -- so it
neither reads the data nor depends on where the sequence ends. Asserted rather than argued: the tests
below run on the *guarded* model, which is the configuration in which a length-dependent mask would
show.

``causal_norm: true`` is the qualifier, exactly as on every conv-LSTM sibling: these encoders carry a
time-pooling normaliser whose statistics would otherwise run over the whole sequence, and the flag
causalises them. That it is required is the raw-signal cell's encoder test; here it is simply on.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import (
    TINY_STRIDE,
    build,
    make_streams,
    tiny_warmup_kwargs,
)

#: Cuts inside the tiny anchor range, so at least one decoded anchor sits at or below each.
_CUTS = (6, 9, 13)

#: Prefix equivalence is exact in real arithmetic and float-exact only up to the convolutions'
#: accumulation order, which a shorter tensor changes. The sibling packages use the same bound.
_PREFIX_TOL = 1e-5


def _resample_after(stream: torch.Tensor, cut: int, seed: int) -> torch.Tensor:
    """A copy of ``stream`` with everything strictly after ``cut`` replaced by fresh noise."""
    perturbed = stream.clone()
    generator = torch.Generator().manual_seed(seed)
    perturbed[:, cut + 1 :] = torch.randn(perturbed[:, cut + 1 :].shape, generator=generator)
    return perturbed


def _forward(model, streams, *extra):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams, *extra)


@pytest.fixture(scope="module")
def kwargs():
    """The tiny guarded keyword set at a real tiling, with the causalised normalisers on."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, causal_norm=True, dropout=0.0)


@pytest.fixture(scope="module")
def model(kwargs):
    """Built once: nothing below trains and nothing below mutates it."""
    return build(kwargs).eval()


# =================================================================================================
# 1. Step-wise causality
# =================================================================================================
@pytest.mark.parametrize("cut", _CUTS)
def test_the_whole_model_reads_no_step_after_the_anchor(model, kwargs, cut: int) -> None:
    """Every stream is resampled, not only the target: ``mu_full`` reads the source through the lag
    attention, whose window runs into the strict past, so a source-side leak lands there alone. The
    final step must move, or the equalities would pass on a dead pathway."""
    streams = make_streams(kwargs)
    reference = _forward(model, streams, 1)
    moved = _forward(
        model,
        tuple(_resample_after(x, cut, seed=11 + index) for index, x in enumerate(streams)),
        1,
    )

    for key in ("mu_prior", "logvar_prior", "target_state", "source_state", "mu_post", "kld_per_t"):
        assert torch.equal(reference[key][:, : cut + 1], moved[key][:, : cut + 1]), key

    # The forecasts live on the anchor axis, so the comparison is over the anchors at or below the
    # cut rather than over a step prefix.
    index = reference["anchor_index"]
    assert torch.equal(index, moved["anchor_index"])
    early = index[0] <= cut
    assert bool(early.any()), f"no decoded anchor at or below the cut {cut}"
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert torch.equal(reference[key][:, early], moved[key][:, early]), key

    # The paired control: without it a dead pathway would pass every equality above.
    assert not torch.equal(reference["mu_prior"][:, -1], moved["mu_prior"][:, -1])
    assert not torch.equal(reference["source_state"][:, -1], moved["source_state"][:, -1])


# =================================================================================================
# 2. Prefix equivalence
# =================================================================================================
def _history_states(model, streams, length: int):
    r"""$(H^y, H^u)$ over the leading ``length`` steps, through the model's own modules.

    The encoder pathway rather than the whole forward, and deliberately: the anchor set is built from
    the constructed geometry's $T_{\mathrm{valid}}$, so a shorter sequence is refused outright at the
    decode -- which is the right behaviour and not the property under test. What prefix equivalence
    is about is the *history*, and the warm-up mask sits inside it.

    Args:
        model: The built model.
        streams: ``(y_st, y_ph, u_stream)``.
        length: How many leading steps to run.

    Returns:
        The two history states.
    """
    y_st, y_ph, u_stream = (x[:, :length] for x in streams)
    with torch.no_grad():
        target = torch.cat([y_st, y_ph], dim=-1)
        if model.target_gate is not None:
            target = model.target_gate(target)
        source = u_stream if model.source_gate is None else model.source_gate(u_stream)
        return (
            model.target_encoder(model.target_adapter(target)),
            model.source_encoder(model.source_adapter(source)),
        )


def test_running_on_a_prefix_reproduces_the_full_runs_history(model, kwargs) -> None:
    r"""$\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$: strictly stronger than causality.

    Causality can hold while this fails -- if anything were computed relative to the sequence end, or
    if the availability pattern were rebuilt against the batch's length rather than sliced from the
    constant. The guarded model is the one that would show the second.
    """
    prefix = model.sequence_length - 6
    streams = make_streams(kwargs)

    full_target, full_source = _history_states(model, streams, model.sequence_length)
    short_target, short_source = _history_states(model, streams, prefix)

    assert float((full_target[:, :prefix] - short_target).abs().max()) < _PREFIX_TOL
    assert float((full_source[:, :prefix] - short_source).abs().max()) < _PREFIX_TOL
    # Not vacuous: all-zero states, or states constant along time, would satisfy it perfectly.
    assert float(full_target.abs().max()) > _PREFIX_TOL
    assert float(full_source.abs().max()) > _PREFIX_TOL
    assert float((full_target[:, 0] - full_target[:, -1]).abs().max()) > _PREFIX_TOL


# =================================================================================================
# 3. The past-loss source gradient
# =================================================================================================
def test_a_loss_on_early_anchors_reaches_no_later_source_step(kwargs, perturb_posterior) -> None:
    """The gradient form of causality, and the one that covers the lag attention's window.

    Summing the forecast at the earliest decoded anchor and differentiating back to the source stream
    must leave every step after that anchor untouched -- otherwise the attention is reading forward,
    which no shape and no mask would report.

    Perturbed first, and that is not incidental: the posterior deltas are zero-initialised, so on a
    fresh model ``mu_full`` does not depend on the source at all and the source gradient is
    identically zero everywhere -- which would satisfy the causality half while proving nothing.
    """
    model = build(kwargs)
    perturb_posterior(model)
    model.eval()

    streams = list(make_streams(kwargs))
    streams[2] = streams[2].clone().requires_grad_(True)

    torch.manual_seed(0)
    out = model(*streams, 1)
    anchor = int(out["anchor_index"][0, 0])
    out["mu_full"][:, 0].sum().backward()

    grad = streams[2].grad
    assert grad is not None
    assert float(grad[:, anchor + 1 :].abs().max()) == 0.0
    # The control: the source at and before the anchor is read, so this is causality rather than a
    # source pathway that contributes nothing at all.
    assert float(grad[:, : anchor + 1].abs().max()) > 0.0
