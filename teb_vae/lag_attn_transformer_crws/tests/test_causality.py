r"""Step-wise causality and prefix equivalence, **unconditionally**.

The conv-LSTM cell of this row has to qualify both claims with ``causal_norm: true``: without it a
time-pooling normaliser inside each encoder mixes the whole sequence, the "prior" conditions on the
future, and the source-conditioned KL is not a coupling readout at all. This architecture has no such
normaliser -- RMSNorm reduces over channels only, the convolutions pad left, and the attention is
causal by kernel flag or explicit band mask -- and ``causal_norm`` is not a constructor keyword of
this model, so there is no flag to get wrong and no arm in which the claim fails.

That is the difference this file exists to record, and it is the reason the assertions are the
sibling's with the qualifier removed rather than a copy with a value flipped.

Note what the *inputs* add and what they do not. A causal coefficient at step $t$ is a function of
$\{x(s) : s \le t\}$, so the stored stream carries no future either -- but that is a property of the
dataset and it is proved where the dataset is built. What is proved here is the network's own
property: given whatever it is handed, $H_t$ reads no step after $t$.

Three probes, and each is paired with a control, because every one of them would pass on a model
whose pathways were dead:

1. **Step-wise causality.** Resample every stream strictly after a cut; nothing at or below the cut
   may move, and the last step must.
2. **Prefix equivalence.** $\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$, which is strictly
   stronger: it also fails if anything is computed relative to the sequence end, or if the
   availability pattern is rebuilt against the batch's length rather than sliced from a constant.
3. **The past-loss source gradient.** Differentiate the earliest decoded anchor's forecast back to
   the source stream; every later source step must be untouched. This is the one that covers the lag
   attention's window, and the posterior must be perturbed first or it is vacuous.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws

from .conftest import TINY_STRIDE, build, make_streams, tiny_warmup_kwargs

#: Cuts inside the tiny anchor range, so at least one decoded anchor sits at or below each.
_CUTS = (6, 9, 13)

#: Prefix equivalence is exact in real arithmetic and float-exact only up to the convolutions' and
#: the attention's accumulation order, which a shorter tensor changes. The sibling packages use the
#: same bound.
_PREFIX_TOL = 1e-5


def _resample_after(stream: torch.Tensor, cut: int, seed: int) -> torch.Tensor:
    """A copy of ``stream`` with everything strictly after ``cut`` replaced by fresh noise."""
    perturbed = stream.clone()
    generator = torch.Generator().manual_seed(seed)
    perturbed[:, cut + 1 :] = torch.randn(
        perturbed[:, cut + 1 :].shape, generator=generator
    )
    return perturbed


def _forward(model, streams, *extra):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams, *extra)


@pytest.fixture(scope="module")
def kwargs():
    """The tiny guarded keyword set at a real tiling. No ``causal_norm``: there is no such key."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, dropout=0.0)


@pytest.fixture(scope="module")
def model(kwargs):
    """Built once: nothing below trains and nothing below mutates it."""
    return build(kwargs).eval()


# =================================================================================================
# 0. There is no flag to qualify the claim with
# =================================================================================================
def test_causal_norm_is_not_a_keyword_of_this_constructor() -> None:
    """The architectural difference, stated where the claims below rest on it. The conv-LSTM cells
    need the key because they have a time-pooling normaliser to causalise; this one does not, so
    every claim below holds for every configuration this constructor accepts."""
    assert "causal_norm" not in inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters


def test_no_history_path_carries_a_time_pooling_normaliser(model) -> None:
    """The other half of the same statement, structurally: a statistic pooled over time on a history
    path is exactly what would make $H_t$ read its own future."""
    for name in ("target_encoder", "source_encoder", "target_adapter", "source_adapter"):
        for module in getattr(model, name).modules():
            assert not isinstance(
                module,
                (torch.nn.BatchNorm1d, torch.nn.GroupNorm, torch.nn.InstanceNorm1d),
            ), f"{name} carries {type(module).__name__}"


# =================================================================================================
# 1. Step-wise causality
# =================================================================================================
@pytest.mark.parametrize("cut", _CUTS)
def test_the_whole_model_reads_no_step_after_the_anchor(model, kwargs, cut: int) -> None:
    """Every stream is resampled, not only the target: ``mu_full`` reads the source through the lag
    attention, whose window runs into the strict past, so a source-side leak lands there alone."""
    streams = make_streams(kwargs)
    reference = _forward(model, streams, 1)
    moved = _forward(
        model,
        tuple(_resample_after(x, cut, seed=11 + index) for index, x in enumerate(streams)),
        1,
    )

    for key in (
        "mu_prior",
        "logvar_prior",
        "target_state",
        "source_state",
        "mu_post",
        "kld_per_t",
    ):
        assert torch.equal(reference[key][:, : cut + 1], moved[key][:, : cut + 1]), key

    # The forecasts live on the anchor axis, so the comparison is over the anchors at or below the
    # cut rather than over a step prefix.
    index = reference["anchor_index"]
    assert torch.equal(index, moved["anchor_index"])
    early = index[0] <= cut
    assert bool(early.any()), f"no decoded anchor at or below the cut {cut}"
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert torch.equal(reference[key][:, early], moved[key][:, early]), key


@pytest.mark.parametrize("cut", _CUTS)
def test_the_perturbation_did_reach_the_model(model, kwargs, cut: int) -> None:
    """The paired control for every equality above: without it a dead pathway would pass them all."""
    streams = make_streams(kwargs)
    reference = _forward(model, streams, 1)
    moved = _forward(
        model,
        tuple(_resample_after(x, cut, seed=11 + index) for index, x in enumerate(streams)),
        1,
    )

    assert not torch.equal(reference["mu_prior"][:, -1], moved["mu_prior"][:, -1])
    assert not torch.equal(reference["source_state"][:, -1], moved["source_state"][:, -1])


# =================================================================================================
# 2. Prefix equivalence
# =================================================================================================
def test_the_warm_up_mask_is_a_function_of_the_step_alone(model) -> None:
    """Which is what keeps it from breaking prefix equivalence: sliced to a shorter sequence it is
    the leading rows of the same constant, not a pattern recomputed against a new length."""
    adapter = model.target_adapter
    full = adapter._slice(adapter.availability, model.sequence_length)
    short = adapter._slice(adapter.availability, model.sequence_length - 5)

    assert torch.equal(short, full[: model.sequence_length - 5])
    assert not adapter.availability.requires_grad


def _history_states(model, streams, length: int):
    r"""$(H^y, H^u)$ over the leading ``length`` steps, through the model's own modules.

    The encoder pathway rather than the whole forward, and deliberately: the anchor set is built
    from the constructed geometry's $T_{\mathrm{valid}}$, so a shorter sequence is refused outright
    at the decode -- which is the right behaviour and not the property under test.

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

    Causality can hold while this fails -- if anything were computed relative to the sequence end,
    or if the availability pattern were rebuilt against the batch's length rather than sliced from
    the constant. The guarded model is the one that would show the second.
    """
    prefix = model.sequence_length - 6
    streams = make_streams(kwargs)

    full_target, full_source = _history_states(model, streams, model.sequence_length)
    short_target, short_source = _history_states(model, streams, prefix)

    assert float((full_target[:, :prefix] - short_target).abs().max()) < _PREFIX_TOL
    assert float((full_source[:, :prefix] - short_source).abs().max()) < _PREFIX_TOL


def test_the_prefix_probe_is_not_vacuous(model, kwargs) -> None:
    """A model whose states were all zeros, or constant along time, would satisfy it perfectly."""
    streams = make_streams(kwargs)
    target, source = _history_states(model, streams, model.sequence_length)

    assert float(target.abs().max()) > _PREFIX_TOL
    assert float(source.abs().max()) > _PREFIX_TOL
    assert float((target[:, 0] - target[:, -1]).abs().max()) > _PREFIX_TOL


def test_a_shorter_sequence_is_refused_rather_than_decoded_at_a_shifted_anchor_set(
    model, kwargs
) -> None:
    """The geometry is fixed at construction, so a shorter batch is a configuration error.

    Stated as a test because the architecture parent's dense slice would simply return fewer anchors
    on a short tensor, silently; a gather at construction-time indices raises instead.
    """
    streams = make_streams(kwargs)
    with pytest.raises(RuntimeError):
        _forward(model, tuple(x[:, : model.sequence_length - 6] for x in streams), 0, 1)


# =================================================================================================
# 3. The past-loss source gradient
# =================================================================================================
def test_a_loss_on_early_anchors_reaches_no_later_source_step(kwargs, perturb_posterior) -> None:
    """The gradient form of causality, and the one that covers the lag attention's window.

    Summing the forecast at the earliest decoded anchor and differentiating back to the source
    stream must leave every step after that anchor untouched -- otherwise the attention is reading
    forward, which no shape and no mask would report.

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
    # source pathway that was never connected.
    assert float(grad[:, : anchor + 1].abs().max()) > 0.0
