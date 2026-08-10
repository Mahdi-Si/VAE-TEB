r"""The shared fixtures are themselves load-bearing, so they get their own tests.

Three failure modes here would make the rest of the suite lie silently. A stub batch whose
``weight`` stopped planting its gap would leave every mask test green whether or not the masks
work. ``perturb_posterior`` is the only thing standing between a KL assertion and vacuous truth:
the posterior delta heads are zero-initialised, so an unperturbed model passes every KL test no
matter how wrong it is. And the causality probe is the shape of every structural assertion in this
package -- if its negative control were dropped, a module returning zeros would pass all of them.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    BATCH,
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    SEQ_LEN,
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TINY_KWARGS,
    assert_token_causal,
    make_stub_batch,
    relative_change,
    resample_after,
)

#: The sibling constructor keywords this model's schema deliberately does not have. There is no
#: recurrent branch, no extra dilation schedule and no time-pooling normaliser left to causalise,
#: so each of these would reach nothing -- and the signature sweep drops unknown keys silently.
_SIBLING_ONLY_KEYS = (
    "lstm_layers",
    "encoder_extra_dilations",
    "encoder_extra_kernel",
    "conv_norm_groups",
    "causal_norm",
)

#: The encoder schema, every key of which varies across a planned architecture arm.
_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)


def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble derives the repo root from this file's own depth; a wrong depth
    would resolve some unrelated directory without ever raising."""
    repo_root = Path(__file__).resolve().parents[3]
    for package in ("teb_vae", "train", "utils"):
        assert (repo_root / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {repo_root}"
        )


def test_the_stub_batch_carries_raw_signals_at_sixteen_fold_length():
    batch = make_stub_batch(BATCH, SEQ_LEN)
    assert batch.fhr.shape == (BATCH, 16 * SEQ_LEN)
    assert batch.up.shape == (BATCH, 16 * SEQ_LEN)
    assert SEQ_LEN * int(TINY_KWARGS["raw_per_step"]) == batch.fhr.shape[-1]


def test_the_stub_batch_plants_its_gap():
    """A silently gap-free fixture would leave every mask test vacuous."""
    batch = make_stub_batch(BATCH, SEQ_LEN)
    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert (batch.weight == 0.0).any()
    # The gap sits inside the tiny trained-anchor range, where every mask can see it.
    assert TINY_KWARGS["warmup_period"] <= STUB_GAP_STEP
    assert STUB_GAP_STEP < SEQ_LEN - int(TINY_KWARGS["horizon"])


def test_the_stub_batch_matches_the_feature_contract():
    batch = make_stub_batch(BATCH, SEQ_LEN)
    assert batch.fhr_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.fhr_ph.shape == (BATCH, SEQ_LEN, 66)
    assert batch.up_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.up_ph.shape == (BATCH, SEQ_LEN, 15)
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == TINY_KWARGS["c_y"]
    assert batch.up_st.shape[-1] + batch.up_ph.shape[-1] == TINY_KWARGS["c_u"]


def test_tiny_dropout_is_off():
    """Nonzero dropout would make every seeded bitwise comparison in the suite flaky."""
    assert TINY_KWARGS["dropout"] == 0.0


@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_kwargs_sets_satisfy_the_constructor_invariants(kwargs):
    assert kwargs["num_heads"] * kwargs["d_head"] == kwargs["d_model"]
    assert kwargs["d_z"] % kwargs["num_heads"] == 0
    assert kwargs["warmup_period"] < kwargs["sequence_length"] - kwargs["horizon"]
    assert kwargs["raw_per_step"] == 16


@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_kwargs_sets_satisfy_the_encoder_invariants(kwargs):
    """The encoder head split must be exact and its head width even, which rotary position
    encoding requires; and the stem's two schedules must have equal length."""
    heads = int(kwargs["encoder_num_heads"])
    d_model = int(kwargs["d_model"])
    assert heads * (d_model // heads) == d_model
    assert (d_model // heads) % 2 == 0
    assert len(kwargs["encoder_conv_kernels"]) == len(kwargs["encoder_conv_dilations"])
    assert int(kwargs["source_attention_window"]) >= 1


@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_kwargs_sets_carry_the_whole_encoder_schema_and_none_of_the_siblings(kwargs):
    for key in _ENCODER_KEYS:
        assert key in kwargs, f"{key} is missing, so the arm that varies it has nothing to vary"
    for key in _SIBLING_ONLY_KEYS:
        assert key not in kwargs, f"{key} means nothing to this model and would reach nothing"


def test_the_tiny_source_bound_stays_inside_the_sequence():
    r"""$R_U = R_{\mathrm{conv}} + N_U(W_U - 1)$ must be smaller than $T$ at the tiny geometry.

    A bound that clamped at $T$ would make the measured-bound probe vacuous: every perturbation
    would be inside the window and the test could not fail.
    """
    kernels = TINY_KWARGS["encoder_conv_kernels"]
    dilations = TINY_KWARGS["encoder_conv_dilations"]
    conv_reach = 1 + sum((k - 1) * r for k, r in zip(kernels, dilations))
    blocks = int(TINY_KWARGS["source_attention_blocks"])
    window = int(TINY_KWARGS["source_attention_window"])

    assert conv_reach + blocks * (window - 1) < int(TINY_KWARGS["sequence_length"])


def test_shipped_kwargs_is_the_production_geometry_not_a_miniature():
    assert SHIPPED_KWARGS["sequence_length"] == 300
    assert SHIPPED_KWARGS["d_z"] == 64
    assert SHIPPED_KWARGS["max_lag"] == 90
    assert SHIPPED_KWARGS["encoder_conv_kernels"] == (5, 9)
    assert SHIPPED_KWARGS["encoder_conv_dilations"] == (1, 2)
    assert SHIPPED_KWARGS["encoder_d_ff"] == 512
    assert SHIPPED_KWARGS["target_attention_blocks"] == 6
    assert SHIPPED_KWARGS["source_attention_blocks"] == 3
    assert SHIPPED_KWARGS["source_attention_window"] == 16
    # The decoder side of the capacity bundle, which is shared code and therefore the half a
    # miniature of this fixture would silently lose.
    assert SHIPPED_KWARGS["decoder_hidden"] == 256
    assert SHIPPED_KWARGS["horizon_depth"] == 4
    assert SHIPPED_KWARGS["horizon_attention_blocks"] == 2


def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use."""
    markers = request.config.getini("markers")
    assert any(str(marker).startswith("slow") for marker in markers)


def test_no_fixture_files_live_in_this_module():
    """The committed shard and stats are lag_attn's; this module references them by path."""
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


def test_resample_after_leaves_the_prefix_bit_identical():
    """The probe would report causality that was never tested if it touched the prefix."""
    x = torch.randn(2, 8, 3)

    perturbed = resample_after(x, 3)

    assert torch.equal(perturbed[:, :4], x[:, :4])
    assert not torch.equal(perturbed[:, 4:], x[:, 4:])


def test_the_causality_probe_fails_on_a_non_causal_module():
    """A probe that cannot fail is not a probe.

    The failure it must catch is the one this architecture is actually exposed to: a statistic
    pooled over time. A cumulative *reverse* mean is the cheapest module with that property.
    """

    class _ReadsTheFuture(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flip(1).cumsum(1).flip(1)

    x = torch.randn(2, 8, 3)
    assert_token_causal(nn.Identity(), x, 3, label="identity")
    with pytest.raises(AssertionError, match="moved by"):
        assert_token_causal(_ReadsTheFuture(), x, 3, label="leaky")


def test_the_causality_probe_fails_on_a_dead_module():
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
