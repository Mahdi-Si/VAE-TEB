r"""The planted pattern is load-bearing, so it gets its own tests.

Every target assertion in this package is made against
:func:`~teb_vae.lag_attn_fs.tests.conftest.make_patterned_batch`, and the whole point of it is
that a *wrong* gather cannot produce the right values. Three properties carry that, and each
would fail silently if the pattern were ever weakened to a random draw:

* values differ along the step axis, so an off-by-one anchor is visible;
* values differ along the channel axis, so a gather of the wrong channels is visible;
* the two axes are separated by a factor larger than the channel count, so a *transposed* gather
  -- reading step $c$ of channel $t$ -- lands on a value that exists nowhere in the correct
  answer rather than on a plausible neighbour.

The sibling's own fixture tests cover the fields this one inherits; what is checked here is the
pattern, the fields it must not have disturbed, and the two conftest helpers.
"""
from __future__ import annotations

from pathlib import Path

import torch

from teb_vae.lag_attn_fs.tests.conftest import (
    BATCH,
    PATTERN_STEP_SCALE,
    SEQ_LEN,
    SHIPPED_KWARGS,
    SHIPPED_REACH_BUDGET_S,
    STUB_GAP_STEP,
    TINY_KWARGS,
    build_target_gate,
    make_patterned_batch,
    patterned_feature_stream,
)


def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble derives the repo root from this file's own depth; a wrong depth
    would resolve some unrelated directory without ever raising."""
    repo_root = Path(__file__).resolve().parents[3]
    for package in ("teb_vae", "train", "utils"):
        assert (repo_root / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {repo_root}"
        )


def test_every_element_of_the_pattern_is_distinct():
    """Uniqueness is what makes a mismatch legible: the value names the position it came from."""
    stream = patterned_feature_stream(BATCH, SEQ_LEN, TINY_KWARGS["c_y"])
    assert stream.unique().numel() == stream.numel()


def test_the_pattern_moves_along_both_axes():
    """A pattern constant in either axis would leave a whole class of gather bugs invisible."""
    stream = patterned_feature_stream(BATCH, SEQ_LEN, TINY_KWARGS["c_y"])
    assert (stream[:, 1:, :] != stream[:, :-1, :]).all()
    assert (stream[:, :, 1:] != stream[:, :, :-1]).all()


def test_the_step_stride_exceeds_the_widest_channel_count():
    """The separation that makes a transposed read land outside the correct answer's range. At a
    stride below the channel count the step and channel contributions would overlap, and reading
    step $c$ of channel $t$ could return a value the correct gather also produces."""
    assert PATTERN_STEP_SCALE > SHIPPED_KWARGS["c_y"]


def test_the_pattern_is_exactly_representable_in_float32():
    """Comparisons against it are ``torch.equal``, not ``allclose``, so this has to hold at the
    largest geometry the suite builds rather than merely at the tiny one."""
    largest = patterned_feature_stream(
        BATCH, SHIPPED_KWARGS["sequence_length"], SHIPPED_KWARGS["c_y"]
    )
    assert torch.equal(largest, largest.double().float())
    assert float(largest.max()) < 2.0**24


def test_the_two_blocks_concatenate_back_into_one_pattern():
    """The split at the block boundary is what makes the value at channel $c$ of the concatenated
    stream equal to $c$ -- and $c$ is what the reach budget's keep-index indexes into."""
    batch = make_patterned_batch()
    stream = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
    assert torch.equal(stream, patterned_feature_stream(BATCH, SEQ_LEN, TINY_KWARGS["c_y"]))


def test_planting_the_pattern_leaves_the_inherited_fields_alone():
    """Only the two target blocks are overwritten; the gap, the raw traces and the source blocks
    stay exactly what every other suite in the family sees."""
    batch = make_patterned_batch()
    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert batch.up_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.up_ph.shape == (BATCH, SEQ_LEN, 15)
    assert batch.fhr.shape == (BATCH, 16 * SEQ_LEN)
    assert batch.up.shape == (BATCH, 16 * SEQ_LEN)


def test_the_patterned_blocks_keep_the_declared_widths():
    batch = make_patterned_batch()
    assert batch.fhr_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.fhr_ph.shape == (BATCH, SEQ_LEN, 66)
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == TINY_KWARGS["c_y"]


def test_the_keyword_sets_keep_raw_per_step_and_add_no_decoder_width():
    """``raw_per_step`` remains a geometry input -- ``TrimmedRawGeometry`` validates the raw index
    identities against it and the diagnostic page draws its first row on the raw grid -- it simply
    stops being the decoder width. And no ``decoder_out_channels`` is set here, so every model
    built from these kwargs derives its own width from the target gate rather than being told
    one, which is the path production takes."""
    for kwargs in (TINY_KWARGS, SHIPPED_KWARGS):
        assert kwargs["raw_per_step"] == 16
        assert "decoder_out_channels" not in kwargs
        assert "target_keep_index" not in kwargs


def test_the_unguarded_arm_builds_no_gate_at_all():
    """``None`` must produce *nothing*, not an identity gate: the ungated model is an
    architectural baseline, and the decoder width then follows $c_y$ rather than a keep-index."""
    assert build_target_gate(None) is None


def test_the_shipped_gate_emits_the_costed_width():
    gate = build_target_gate(SHIPPED_REACH_BUDGET_S)
    assert gate is not None
    assert gate.out_channels == 78
    assert gate.declared_width == SHIPPED_KWARGS["c_y"]


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
