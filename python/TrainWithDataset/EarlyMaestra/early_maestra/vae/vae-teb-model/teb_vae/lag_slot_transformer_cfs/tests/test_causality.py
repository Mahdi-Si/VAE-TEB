r"""Nothing after an anchor changes that anchor's forecast.

The claim is **neural** causality: changing inputs after stored step $t$ cannot change the forecast
at $t$. It is measured here rather than argued from the absence of a future-reading operator,
because every mechanism that would break it -- an unfolded window that reaches forward, a
normaliser that pools over time, a lag index that wraps past the end of the record -- leaves every
shape correct.

**What this file does not establish**, and no test in this package can: raw-signal causality. The
stored coefficients are produced by filters whose support this model never sees, so token causality
over the feature grid is a statement about the model and not about the pipeline that built its
inputs. The warm-up budget bounds that exposure; it does not remove it, and auditing it is a
separate exercise on the raw prefix.

There is also a negative control here. A model whose forecasts do not move at all when the strict
future is resampled passes every causality assertion trivially, so each test requires visible
movement at the anchors that *should* move.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_FLOOR,
    TINY_SEQ_LEN,
    build_tiny_model,
    tiny_model_kwargs,
    tiny_streams,
)

#: The stored step after which the inputs are resampled. Chosen inside the decoded anchor range so
#: there are anchors on both sides of it.
CUT_STEP = 12


def trained_model(**overrides) -> SeqVaeLagResidualTrfCfs:
    """A model whose source pathway has been moved off its zero start.

    Every source-sensitivity assertion is vacuous at initialisation, where the proposal head emits
    exactly zero and the full branch is the prior. This stands in for the first optimizer steps.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model, in evaluation mode.
    """
    model = build_tiny_model(**overrides)
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        model.proposal_head.output_proj.weight.normal_(0.0, 0.4, generator=generator)
        model.proposal_head.output_proj.bias.normal_(0.0, 0.1, generator=generator)
        model.clock_proj.weight.normal_(0.0, 0.1, generator=generator)
    return model.eval()


def resample_after(tensor: torch.Tensor, step: int, seed: int) -> torch.Tensor:
    """Replace everything strictly after ``step`` with a fresh draw.

    Args:
        tensor: The stream $(B, T, C)$.
        step: The last step left untouched.
        seed: Seed for the replacement draw.

    Returns:
        A copy with its strict future resampled.
    """
    generator = torch.Generator().manual_seed(seed)
    moved = tensor.clone()
    tail = tensor.shape[1] - step - 1
    moved[:, step + 1 :] = torch.randn(
        tensor.shape[0], tail, tensor.shape[2], generator=generator
    )
    return moved


def forecasts(model: SeqVaeLagResidualTrfCfs, streams, seed: int = 0):
    """Run one dense forward under a fixed noise seed and return what it predicts.

    Args:
        model: The model.
        streams: ``(y_st, y_ph, u_stream)``.
        seed: Seed for the reparameterisation draw.

    Returns:
        The forward's dict.
    """
    torch.manual_seed(seed)
    return model(*streams, anchor_phase=0, anchor_stride=1)


# =================================================================================================
# Token causality
# =================================================================================================
@pytest.mark.parametrize("stream", [0, 1, 2])
def test_resampling_the_strict_future_leaves_earlier_anchors_bitwise_unchanged(
    stream: int,
) -> None:
    """One stream at a time, so a failure names which input reached backwards.

    Streams zero and one are the two target blocks; stream two is the source. Each is resampled
    strictly after the cut, and every anchor at or before the cut must predict exactly what it did.
    """
    model = trained_model()
    streams = list(tiny_streams())
    baseline = forecasts(model, streams)

    moved = list(streams)
    moved[stream] = resample_after(streams[stream], CUT_STEP, seed=100 + stream)
    perturbed = forecasts(model, moved)

    anchors = baseline["anchor_index"][0]
    before = anchors <= CUT_STEP
    after = anchors > CUT_STEP
    assert bool(before.any()) and bool(after.any()), "the cut leaves one side empty"

    for name in ("mu_base", "mu_full", "logvar_full", "mu_prior", "mu_post"):
        assert torch.equal(
            baseline[name][:, before], perturbed[name][:, before]
        ), f"{name} moved at an anchor at or before the cut"

    # The negative control: a model that ignored its inputs would pass the assertion above.
    assert not torch.allclose(
        baseline["mu_full"][:, after], perturbed["mu_full"][:, after]
    ), "nothing moved after the cut either; the test is vacuous"


def test_the_source_reaches_the_full_branch_and_never_the_base_branch() -> None:
    """Source purity, measured: the prior and its forecast are functions of the target alone."""
    model = trained_model()
    y_st, y_ph, u_stream = tiny_streams()
    baseline = forecasts(model, (y_st, y_ph, u_stream))

    generator = torch.Generator().manual_seed(77)
    moved_source = torch.randn(u_stream.shape, generator=generator)
    perturbed = forecasts(model, (y_st, y_ph, moved_source))

    for name in ("mu_prior", "logvar_prior", "mu_base", "logvar_base", "z_prior"):
        assert torch.equal(baseline[name], perturbed[name]), f"{name} saw the source"
    assert not torch.allclose(baseline["mu_full"], perturbed["mu_full"])


def test_the_metadata_clock_carries_no_source_value() -> None:
    """It reads stored position, and the conditioning state moves only with the target."""
    model = trained_model()
    y_st, y_ph, u_stream = tiny_streams()
    baseline = forecasts(model, (y_st, y_ph, u_stream))

    generator = torch.Generator().manual_seed(88)
    perturbed = forecasts(
        model, (y_st, y_ph, torch.randn(u_stream.shape, generator=generator))
    )
    assert torch.equal(baseline["conditioning_state"], perturbed["conditioning_state"])

    # And the clock itself is identical whatever the source stream is, including its shape-derived
    # broadcast form.
    left = model._prior_clock(u_stream)
    right = model._prior_clock(torch.randn(u_stream.shape, generator=generator))
    assert torch.equal(left, right)
    assert left.shape == (1, TINY_SEQ_LEN, model.d_model)


def test_no_future_validity_signal_can_enter_the_forward() -> None:
    """The predictor takes inputs and a tiling, and no mask of any kind.

    Future target validity selects which anchors the objective scores. If it could also reach the
    predictor, the model would be told which of its own labels exist, and every score would be
    conditioned on the answer.
    """
    import inspect

    parameters = set(inspect.signature(SeqVaeLagResidualTrfCfs.forward).parameters)
    assert parameters == {
        "self",
        "y_st",
        "y_ph",
        "u_stream",
        "anchor_phase",
        "anchor_stride",
        "selector",
        "return_proposals",
    }


# =================================================================================================
# The lag window
# =================================================================================================
def test_no_lag_reads_a_step_at_or_after_its_anchor() -> None:
    """Lag zero reads the anchor's own step, and every other lag reads strictly before it.

    Measured through the model's own mask rather than through the gather, because the mask is what
    the exposure readouts are computed against and a disagreement between the two would be
    invisible.
    """
    model = trained_model()
    mask = model.build_lag_mask(TINY_SEQ_LEN, device=torch.device("cpu"))
    assert mask.shape == (TINY_SEQ_LEN, model.n_lags)

    for step in range(TINY_SEQ_LEN):
        for lag in range(model.n_lags):
            readable = bool(mask[step, lag])
            source_step = step - lag
            assert readable == (source_step >= max(0, model.lag_floor))
            if readable:
                assert source_step <= step


def test_the_lag_floor_narrows_the_mask() -> None:
    """A floor above zero rules out stored steps that exist but lie below it."""
    unfloored = trained_model().build_lag_mask(TINY_SEQ_LEN, device=torch.device("cpu"))
    floored = trained_model(lag_floor=TINY_FLOOR).build_lag_mask(
        TINY_SEQ_LEN, device=torch.device("cpu")
    )
    assert bool((floored <= unfloored).all())
    assert int(unfloored.sum()) > int(floored.sum())
