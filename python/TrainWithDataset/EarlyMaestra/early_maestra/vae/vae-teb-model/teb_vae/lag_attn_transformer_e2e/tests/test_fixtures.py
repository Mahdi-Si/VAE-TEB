r"""The shared fixtures are themselves load-bearing, so they get their own tests.

Three failure modes here would make the rest of the suite lie silently. A stub batch whose
``weight`` stopped planting its gap would leave every mask test green whether or not the masks
work. ``perturb_posterior`` is the only thing standing between a KL assertion and vacuous truth:
the posterior delta heads are zero-initialised, so an unperturbed model passes every KL test no
matter how wrong it is. And the causality probes are the shape of every structural assertion in
this package -- if either lost its negative control, a module returning zeros would pass all of
them.

:func:`assert_raw_causal` gets the most attention because it is the one that is new here, and
because it is what this package's central claim is measured by: a probe nobody has watched fail is
a probe that proves nothing. It is driven against a genuinely causal decimator, a decimator that
reads one sample into its own future, and a dead one, and it must accept the first and reject the
other two for two different reasons.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    SEQ_LEN,
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TINY_FRONTEND_KERNELS,
    TINY_KWARGS,
    TINY_WARMUP_PERIOD,
    assert_raw_causal,
    assert_token_causal,
    make_stub_batch,
    relative_change,
    resample_after,
    resample_raw_after,
)

#: The sibling constructor keywords this model's schema deliberately does not have. There are no
#: stored feature blocks to declare a width for, so each of these would describe nothing -- and the
#: config-to-constructor sweep drops an unknown key in silence.
_SIBLING_ONLY_KEYS = ("c_y", "c_u", "use_up_st")

#: Total raw samples per token: four stride-2 stages. Restated rather than imported, because the
#: front end does not exist yet and because a probe that borrowed the module's own arithmetic could
#: not catch that arithmetic being wrong.
_TOTAL_STRIDE = 16


def _decimate(x: torch.Tensor, stride: int, *, shift: int = 0) -> torch.Tensor:
    """Mean-pool ``x`` into non-overlapping windows, optionally shifted forward.

    A stand-in for the real front end, used only to exercise the probe. At ``shift = 0`` token $t$
    averages raw samples $[st,\\, s(t+1))$, so its newest input is $s(t+1) - 1$ -- exactly the
    convention the model's anchors use. A positive shift moves the window forward, which is the
    accident a wrong decimation offset produces.

    Args:
        x: A raw batch, $(B, L)$.
        stride: Window width and hop.
        shift: Samples the window is moved forward by.

    Returns:
        The pooled sequence, $(B, L // stride, 1)$.
    """
    rolled = torch.roll(x, shifts=-shift, dims=-1)
    return rolled.unfold(-1, stride, stride).mean(dim=-1, keepdim=True)


def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble derives the repo root from this file's own depth; a wrong depth
    would resolve some unrelated directory without ever raising."""
    repo_root = Path(__file__).resolve().parents[3]
    for package in ("teb_vae", "train", "utils"):
        assert (repo_root / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {repo_root}"
        )


# ---------------------------------------------------------------------------------------
# The stub batch: this model's whole input contract
# ---------------------------------------------------------------------------------------
def test_the_stub_batch_carries_the_raw_signals_this_model_is_fed():
    """Both raw traces, not just the target one. ``up`` was the diagnostic figure's field in the
    sibling suites and is a model *input* here, so a fixture that carried only ``fhr`` would leave
    the source stream untestable."""
    batch = make_stub_batch(BATCH, SEQ_LEN)

    assert batch.fhr.shape == (BATCH, 16 * SEQ_LEN)
    assert batch.up.shape == (BATCH, 16 * SEQ_LEN)
    assert batch.weight.shape == (BATCH, SEQ_LEN)
    assert batch.fhr.shape[-1] == int(TINY_KWARGS["raw_per_step"]) * SEQ_LEN


def test_the_stub_batch_plants_its_gap_inside_the_trained_anchor_range():
    """A silently gap-free fixture would leave every mask test vacuous -- and a gap outside
    $[w,\\, T - H)$ would be trimmed away by the warm-up before any mask could see it. This
    package raises the warm-up to give the front end reach budget, which is exactly the change
    that could push the gap out of range."""
    batch = make_stub_batch(BATCH, SEQ_LEN)

    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert (batch.weight == 0.0).any()
    assert TINY_WARMUP_PERIOD <= STUB_GAP_STEP < SEQ_LEN - int(TINY_KWARGS["horizon"])


def test_tiny_dropout_is_off():
    """Nonzero dropout would make every seeded bitwise comparison in the suite flaky, and the
    bitwise half of the raw-causality probe is not a comparison that tolerates noise."""
    assert TINY_KWARGS["dropout"] == 0.0


# ---------------------------------------------------------------------------------------
# The constructor keyword sets, on their data-facing side
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_neither_kwarg_set_declares_a_stored_feature_width(kwargs):
    """A copy-pasted sibling key would be dropped by the config sweep in silence, so nothing would
    fail -- the run would simply not be the one the operator thought."""
    for key in _SIBLING_ONLY_KEYS:
        assert key not in kwargs, f"{key} describes a stored feature block this model never loads"


def test_only_the_tiny_set_pins_a_front_end_kernel_schedule():
    """The production kernels are the constructor's own default and no config sets them, so the
    shipped keyword set must not carry them either -- otherwise it would describe a configuration
    surface that does not exist."""
    assert TINY_KWARGS["frontend_kernels"] == TINY_FRONTEND_KERNELS
    assert len(TINY_FRONTEND_KERNELS) == 4  # one per stride-2 stage
    assert "frontend_kernels" not in SHIPPED_KWARGS


def test_the_tiny_warmup_gives_the_front_end_a_usable_reach_budget():
    """The budget is ``warmup_period * raw_per_step`` raw samples, and a four-stage stride-2
    cascade spends the siblings' whole budget on its decimation alone -- which is why this geometry
    raises the warm-up rather than inheriting it. The raise is only legal while the trained-anchor
    range stays non-empty, and it is only *useful* while it stays several tokens wide: a range of
    one anchor would make every masked readout in the suite a statement about a single step."""
    budget = TINY_WARMUP_PERIOD * int(TINY_KWARGS["raw_per_step"])
    trained_anchors = SEQ_LEN - int(TINY_KWARGS["horizon"]) - TINY_WARMUP_PERIOD

    assert TINY_KWARGS["warmup_period"] == TINY_WARMUP_PERIOD
    assert budget == 96
    assert trained_anchors >= 4


def test_shipped_kwargs_is_the_production_geometry_not_a_miniature():
    assert SHIPPED_KWARGS["sequence_length"] == 300
    assert SHIPPED_KWARGS["d_z"] == 48
    assert SHIPPED_KWARGS["max_lag"] == 90
    assert SHIPPED_KWARGS["warmup_period"] == 30
    assert SHIPPED_KWARGS["raw_per_step"] == 16
    assert SHIPPED_KWARGS["encoder_conv_kernels"] == (5, 9)
    assert SHIPPED_KWARGS["target_attention_blocks"] == 4
    assert SHIPPED_KWARGS["source_attention_blocks"] == 3


# ---------------------------------------------------------------------------------------
# The imported token-resolution probe
# ---------------------------------------------------------------------------------------
def test_the_token_causality_probe_fails_on_a_non_causal_module():
    """Imported, but exercised here rather than taken on trust: this suite's structural tests are
    built on it, and a probe nobody in this package has seen fail proves nothing here."""

    class _ReadsTheFuture(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flip(1).cumsum(1).flip(1)

    x = torch.randn(2, 8, 3)
    assert_token_causal(nn.Identity(), x, 3, label="identity")
    with pytest.raises(AssertionError, match="moved by"):
        assert_token_causal(_ReadsTheFuture(), x, 3, label="leaky")


def test_the_token_causality_probe_fails_on_a_dead_module():
    """The negative-control half: a module returning zeros is bit-stable everywhere."""

    class _Dead(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    with pytest.raises(AssertionError, match="never reached"):
        assert_token_causal(_Dead(), torch.randn(2, 8, 3), 3, label="dead")


def test_the_probe_tolerances_leave_a_wide_margin():
    """A gap of two orders of magnitude between "unmoved" and "moved" is what keeps the paired
    assertion from being a coin flip on float32 round-off."""
    assert CAUSALITY_TOL < MOVEMENT_TOL / 10
    assert relative_change(torch.ones(4), torch.ones(4)) == 0.0


# ---------------------------------------------------------------------------------------
# The raw-resolution probe, which is this package's own
# ---------------------------------------------------------------------------------------
def test_the_raw_resample_perturbs_the_last_axis_and_leaves_the_prefix_bitwise_identical():
    """Both halves matter. A resample that touched the prefix would report causality that was
    never tested; one that perturbed nothing would report it too."""
    x = torch.randn(2, 3, 64)

    perturbed = resample_raw_after(x, 31)

    assert torch.equal(perturbed[..., :32], x[..., :32])
    assert not torch.equal(perturbed[..., 32:], x[..., 32:])


def test_the_token_major_resample_is_the_wrong_tool_for_a_raw_batch():
    """Why this package carries its own. The imported probe perturbs axis $1$, which on a
    channel-major raw tensor is the *channel* axis: it redraws whole channels over their entire
    length, past included, and leaves the surviving channels' future untouched. A causality
    assertion built on it would be about neither causality nor this input."""
    x = torch.randn(2, 3, 64)

    token_major = resample_after(x, 1)

    # The raw past was overwritten, which no causality probe may do...
    assert not torch.equal(token_major[..., :32], x[..., :32])
    # ...while the raw future of the channels it did not touch survived untested.
    assert torch.equal(token_major[:, :2, 32:], x[:, :2, 32:])
    # The local probe gets both halves right on the same tensor.
    raw = resample_raw_after(x, 31)
    assert torch.equal(raw[..., :32], x[..., :32])
    assert not torch.equal(raw[:, :2, 32:], x[:, :2, 32:])


def test_the_raw_probe_accepts_a_right_offset_decimator():
    """The convention the model's anchors use: token $t$'s newest input sample is $s(t+1) - 1$, so
    a cut there must leave it bitwise unmoved while the last token still moves."""
    x = torch.randn(2, 16 * _TOTAL_STRIDE, dtype=torch.float64)

    for token in (3, 7):
        assert_raw_causal(
            lambda value: _decimate(value, _TOTAL_STRIDE),
            x,
            _TOTAL_STRIDE * (token + 1) - 1,
            _TOTAL_STRIDE,
            label=f"decimate@t={token}",
        )


def test_the_raw_probe_fails_on_a_one_sample_future_leak():
    """The failure a wrong decimation offset actually produces, and the reason the probe is
    bitwise: one sample out of sixteen moves a mean by roughly $6\\%$ of one sample's value, which
    no round-off threshold could be set to catch reliably."""
    x = torch.randn(2, 16 * _TOTAL_STRIDE, dtype=torch.float64)

    with pytest.raises(AssertionError, match="reads its own future"):
        assert_raw_causal(
            lambda value: _decimate(value, _TOTAL_STRIDE, shift=1),
            x,
            _TOTAL_STRIDE * 4 - 1,
            _TOTAL_STRIDE,
            label="shifted",
        )


def test_the_raw_probe_fails_on_a_dead_module():
    """The negative-control half. A dead stage is bit-stable at every cut, so without this the
    probe would certify a front end that had stopped computing anything."""
    x = torch.randn(2, 16 * _TOTAL_STRIDE, dtype=torch.float64)

    with pytest.raises(AssertionError, match="never reached"):
        assert_raw_causal(
            lambda value: torch.zeros(
                value.shape[0], value.shape[-1] // _TOTAL_STRIDE, 1, dtype=value.dtype
            ),
            x,
            _TOTAL_STRIDE * 4 - 1,
            _TOTAL_STRIDE,
            label="dead",
        )


def test_the_raw_probe_refuses_a_cut_that_names_no_token():
    """A cut past the end of the output would index the wrong token, or throw an obscure
    ``IndexError`` far from the mistake."""
    x = torch.randn(2, 8 * _TOTAL_STRIDE, dtype=torch.float64)

    with pytest.raises(AssertionError, match="beyond the"):
        assert_raw_causal(
            lambda value: _decimate(value, _TOTAL_STRIDE)[:, :2],
            x,
            _TOTAL_STRIDE * 4 - 1,
            _TOTAL_STRIDE,
            label="short output",
        )


# ---------------------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------------------
def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use."""
    markers = request.config.getini("markers")
    assert any(str(marker).startswith("slow") for marker in markers)


def test_no_fixture_files_live_in_this_module():
    """The committed shard and stats are ``lag_attn``'s; this package references them by path."""
    assert not (Path(__file__).resolve().parent / "fixtures").exists()
    shared = Path(__file__).resolve().parents[2] / "lag_attn" / "tests" / "fixtures"
    assert (shared / "tiny_shard.hdf5").is_file()
    assert (shared / "tiny_stats.hdf5").is_file()


def test_perturb_posterior_actually_changes_posterior_parameters(perturb_posterior):
    """The imported fixture must land on this model's ``posterior_head`` attribute."""

    class _StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.posterior_head = nn.Linear(4, 4)
            self.other_head = nn.Linear(4, 4)

    model = _StubModel()
    before = {name: parameter.clone() for name, parameter in model.named_parameters()}

    perturb_posterior(model)

    assert not torch.equal(model.posterior_head.weight, before["posterior_head.weight"])
    # Scoped to the posterior: perturbing the whole model would change what the KL tests mean.
    assert torch.equal(model.other_head.weight, before["other_head.weight"])
