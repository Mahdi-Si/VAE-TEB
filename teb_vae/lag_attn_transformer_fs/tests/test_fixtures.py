r"""The spliced conftest is the most fragile fixture set in the family, so it gets its own tests.

Every other conftest in the family takes its keyword sets and its batch machinery from one place.
This one takes them from two, and the two halves are not interchangeable: the constructor keyword
sets describe *this architecture* and the batch machinery describes *this target*. Four things would
make the rest of the suite lie silently, and each has a test here.

1. **A keyword set from the wrong sibling.** The feature suite's shipped set carries five keywords
   this constructor does not have, and its tiny set carries none of them -- so the mistake would let
   the tiny path pass and fail only at the shipped geometry.
2. **A pattern weakened to a random draw.** Every target assertion in this suite reads the planted
   value back and recovers the $(t, c)$ it came from; against ``randn`` a transposed gather and an
   off-by-one anchor both produce correctly shaped tensors.
3. **The two halves disagreeing on the geometry they share.** The imported budget resolver and batch
   builder read $c_y$, $c_u$, the warm-up, $R$ and ``use_up_st`` off the *feature* suite's shipped
   set while every model here is built from the conv-Transformer one. The splice is sound only while
   those five agree, and nothing outside this file would notice if they stopped.
4. **``perturb_posterior`` not landing.** The posterior delta heads are zero-initialised, so an
   unperturbed model passes every KL assertion in the suite no matter how wrong it is.

The sibling suites own the fields, tolerances and probes this one inherits; what is checked here is
the splice.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_fs.tests.conftest import SHIPPED_KWARGS as FEATURE_SHIPPED_KWARGS
from teb_vae.lag_attn_fs.tests.conftest import TINY_KWARGS as FEATURE_TINY_KWARGS
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    BATCH,
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    PATTERN_STEP_SCALE,
    SEQ_LEN,
    SHARED_GEOMETRY_KEYS,
    SHIPPED_KWARGS,
    SHIPPED_REACH_BUDGET_S,
    STUB_GAP_STEP,
    TINY_KEEP_INDEX,
    TINY_KWARGS,
    build_target_gate,
    make_patterned_batch,
    patterned_feature_stream,
    relative_change,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)

#: The feature sibling's constructor keywords this architecture deliberately does not have. There is
#: no recurrent branch, no extra dilation schedule and no time-pooling normaliser left to causalise,
#: so each of these would reach nothing -- and the signature sweep drops unknown keys silently.
_SIBLING_ONLY_KEYS = (
    "lstm_layers",
    "encoder_extra_dilations",
    "encoder_extra_kernel",
    "conv_norm_groups",
    "causal_norm",
)

#: The encoder schema this architecture adds, every key of which varies across a planned arm.
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


# ---------------------------------------------------------------------------------------
# 1. The keyword sets came from the architecture sibling
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_keyword_sets_carry_the_encoder_schema_and_none_of_the_feature_siblings_keys(kwargs):
    """The splice, asserted on the sets themselves rather than on a construction that happens to
    work. A shipped set from the feature suite would fail *only* at the shipped geometry, because
    its tiny set carries none of the five keys either."""
    for key in _ENCODER_KEYS:
        assert key in kwargs, f"{key} is missing, so the arm that varies it has nothing to vary"
    for key in _SIBLING_ONLY_KEYS:
        assert key not in kwargs, f"{key} means nothing to this model and would reach nothing"


def test_the_feature_siblings_shipped_set_would_not_have_constructed():
    """The failure the test above prevents, made concrete rather than described.

    The feature suite's shipped set is the conv-LSTM one, and this constructor takes no
    ``**kwargs``, so it raises a ``TypeError`` naming the first unexpected keyword. Recorded as a
    measurement because "would not construct" is the whole reason the halves are not
    interchangeable.
    """
    assert any(key in FEATURE_SHIPPED_KWARGS for key in _SIBLING_ONLY_KEYS)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SeqVaeLagAttnTrfFs(**FEATURE_SHIPPED_KWARGS)

    # And the asymmetry: the feature tiny set carries none of the five, so it *does* construct --
    # which is exactly why the mistake would not surface on the tiny path.
    assert not any(key in FEATURE_TINY_KWARGS for key in _SIBLING_ONLY_KEYS)
    torch.manual_seed(0)
    assert SeqVaeLagAttnTrfFs(**FEATURE_TINY_KWARGS).decoder_out_channels == 109


@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_both_keyword_sets_construct_this_model(kwargs):
    """The positive direction, at both geometries, since the two paths fail separately."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**kwargs)

    assert model.geometry.t == int(kwargs["sequence_length"])
    assert model.decoder_out_channels == int(kwargs["c_y"])


@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_keyword_sets_satisfy_the_two_independent_head_constraints(kwargs):
    """Two head counts that merely coincide at the shipped configuration, and one constraint each:
    the lag-attention heads must satisfy ``num_heads * d_head == d_model`` while the *encoder* heads
    must divide ``d_model`` into an **even** width, which rotary position encoding requires. A tiny
    variant that treated them as one constraint raises at construction."""
    assert kwargs["num_heads"] * kwargs["d_head"] == kwargs["d_model"]
    assert kwargs["d_z"] % kwargs["num_heads"] == 0
    heads, d_model = int(kwargs["encoder_num_heads"]), int(kwargs["d_model"])
    assert d_model % heads == 0
    assert (d_model // heads) % 2 == 0
    assert kwargs["warmup_period"] < kwargs["sequence_length"] - kwargs["horizon"]


def test_neither_keyword_set_names_a_decoder_width():
    """``raw_per_step`` stays a geometry input -- the trimmed grid validates its raw index
    identities against it -- it simply stops being the decoder width. And there is no
    ``decoder_out_channels`` keyword at all on this constructor, so every model built from these
    kwargs derives its width from the target gate, which is the path production takes."""
    for kwargs in (TINY_KWARGS, SHIPPED_KWARGS):
        assert kwargs["raw_per_step"] == 16
        assert "decoder_out_channels" not in kwargs
        assert "target_keep_index" not in kwargs


def test_tiny_dropout_is_off():
    """Nonzero dropout would make every seeded bitwise comparison in the suite flaky."""
    assert TINY_KWARGS["dropout"] == 0.0


# ---------------------------------------------------------------------------------------
# 2. The planted pattern reads back at this suite's geometry
# ---------------------------------------------------------------------------------------
def test_every_element_of_the_pattern_is_distinct_at_this_suites_seq_len():
    """Uniqueness is what makes a mismatch legible: the value names the position it came from.
    Asserted at *this* suite's ``SEQ_LEN``, which is what the imported builder is called with."""
    stream = patterned_feature_stream(BATCH, SEQ_LEN, int(TINY_KWARGS["c_y"]))

    assert stream.shape == (BATCH, SEQ_LEN, 109)
    assert stream.unique().numel() == stream.numel()


def test_the_pattern_moves_along_both_axes():
    """A pattern constant in either axis would leave a whole class of gather bugs invisible."""
    stream = patterned_feature_stream(BATCH, SEQ_LEN, int(TINY_KWARGS["c_y"]))

    assert (stream[:, 1:, :] != stream[:, :-1, :]).all()
    assert (stream[:, :, 1:] != stream[:, :, :-1]).all()


def test_the_step_stride_exceeds_the_widest_channel_count():
    """The separation that makes a transposed read land outside the correct answer's range. Below
    the channel count the step and channel contributions would overlap, and reading step $c$ of
    channel $t$ could return a value the correct gather also produces."""
    assert PATTERN_STEP_SCALE > SHIPPED_KWARGS["c_y"]


def test_the_pattern_is_exactly_representable_in_float32_at_the_shipped_length():
    """Comparisons against it are ``torch.equal``, not ``allclose``, so this has to hold at the
    largest geometry the suite builds rather than merely at the tiny one."""
    largest = patterned_feature_stream(
        BATCH, int(SHIPPED_KWARGS["sequence_length"]), int(SHIPPED_KWARGS["c_y"])
    )

    assert torch.equal(largest, largest.double().float())
    assert float(largest.max()) < 2.0**24


def test_the_two_blocks_concatenate_back_into_one_pattern():
    """The split at the block boundary is what makes the value at channel $c$ of the concatenated
    stream equal to $c$ -- and $c$ is what the reach budget's keep-index indexes into."""
    batch = make_patterned_batch(BATCH, SEQ_LEN)
    stream = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)

    assert torch.equal(stream, patterned_feature_stream(BATCH, SEQ_LEN, int(TINY_KWARGS["c_y"])))
    assert batch.fhr_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.fhr_ph.shape == (BATCH, SEQ_LEN, 66)


def test_planting_the_pattern_leaves_the_inherited_fields_alone():
    """Only the two target blocks are overwritten; the gap, the raw traces and the source blocks
    stay exactly what every other suite in the family sees -- including the gap, which sits inside
    the tiny trained-anchor range where every mask can see it."""
    batch = make_patterned_batch(BATCH, SEQ_LEN)

    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert int(TINY_KWARGS["warmup_period"]) <= STUB_GAP_STEP
    assert STUB_GAP_STEP < SEQ_LEN - int(TINY_KWARGS["horizon"])
    assert batch.up_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.up_ph.shape == (BATCH, SEQ_LEN, 15)
    assert batch.fhr.shape == (BATCH, 16 * SEQ_LEN)


# ---------------------------------------------------------------------------------------
# 3. The two halves agree on the geometry they share
# ---------------------------------------------------------------------------------------
def test_the_shared_geometry_keys_agree_between_the_two_shipped_sets():
    """The seam of the splice, and the only thing holding it together.

    ``resolve_target_budget`` and ``make_patterned_batch`` are imported from the feature suite and
    read these five off *its* shipped set, while every model here is built from the
    conv-Transformer one. If the two ever diverged, the resolved keep-index would describe a
    different stream than the one the model declares -- and the gather is positional into the
    declared width, so it would silently take the wrong channels rather than fail.
    """
    for key in SHARED_GEOMETRY_KEYS:
        assert SHIPPED_KWARGS[key] == FEATURE_SHIPPED_KWARGS[key], key
    assert set(SHARED_GEOMETRY_KEYS) <= set(SHIPPED_KWARGS)


def test_the_locally_paired_gated_sets_are_this_architectures(shipped_gated, tiny_gated):
    """The one thing this conftest builds rather than imports: the budget's four resolved tuples
    joined to *this* architecture's keyword sets. The feature suite's own pairing would carry the
    five keywords this constructor refuses."""
    for kwargs in (shipped_gated, tiny_gated):
        for key in _SIBLING_ONLY_KEYS:
            assert key not in kwargs
        for key in _ENCODER_KEYS:
            assert key in kwargs
        for key in ("target_keep_index", "target_delays", "source_keep_index", "source_delays"):
            assert key in kwargs, key

    assert len(shipped_gated["target_keep_index"]) == 78
    assert tiny_gated["target_keep_index"] == TINY_KEEP_INDEX


def test_the_unguarded_arm_builds_no_gate_at_all():
    """``None`` must produce *nothing*, not an identity gate: the ungated model is an architectural
    baseline, and the decoder width then follows $c_y$ rather than a keep-index."""
    assert build_target_gate(None) is None
    assert shipped_gated_kwargs(None) == dict(SHIPPED_KWARGS)


def test_the_shipped_gate_emits_the_costed_width():
    gate = build_target_gate(SHIPPED_REACH_BUDGET_S)

    assert gate is not None
    assert gate.out_channels == 78
    assert gate.declared_width == SHIPPED_KWARGS["c_y"]


def test_the_tiny_guards_delays_are_nonzero_and_distinct(tiny_gated):
    """What makes the never-delayed-target assertions specific rather than "the two differ": a
    target built through this gate is wrong by a *different* number of steps in each channel."""
    delays = tiny_gated["target_delays"]

    assert len(set(delays)) == len(delays)
    assert max(delays) > 0
    assert len(delays) == len(tiny_gated["target_keep_index"])


# ---------------------------------------------------------------------------------------
# 4. The perturbation lands, and the probe tolerances leave room
# ---------------------------------------------------------------------------------------
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


def test_perturb_posterior_opens_this_models_kl(tiny_gated, inputs, perturb_posterior):
    """One step further than the stub above: on *this* model the perturbation has to produce a
    non-zero KL, without which every KL assertion in the suite is vacuously true."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**tiny_gated)
    torch.manual_seed(0)
    with torch.no_grad():
        before = model(*inputs)
    perturb_posterior(model)
    torch.manual_seed(0)
    with torch.no_grad():
        after = model(*inputs)

    assert float(before["kld_per_t"].abs().max()) == 0.0
    assert float(after["kld_per_t"].abs().max()) > 1e-6


def test_the_probe_tolerances_leave_a_wide_margin():
    """A gap of two orders of magnitude between "unmoved" and "moved" is what keeps the paired
    causality assertion from being a coin flip on float32 round-off."""
    assert CAUSALITY_TOL < MOVEMENT_TOL / 10
    assert relative_change(torch.ones(4), torch.ones(4)) == 0.0


def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use."""
    markers = request.config.getini("markers")

    assert any(str(marker).startswith("slow") for marker in markers)


def test_no_fixture_files_live_in_this_package():
    """The committed shard and stats are ``lag_attn``'s; this package references them by path."""
    assert not (Path(__file__).resolve().parent / "fixtures").exists()
    shared = Path(__file__).resolve().parents[2] / "lag_attn" / "tests" / "fixtures"

    assert (shared / "tiny_shard.hdf5").is_file()
    assert (shared / "tiny_stats.hdf5").is_file()
