r"""The encoder-attention readout: that it is the model's own attention, and what it reduces to.

This readout is the one number in the package that is **recomputed** rather than read. The model
attends through fused ``scaled_dot_product_attention``, which never materialises the probabilities,
and the forward contract is pinned at twenty keys -- so the analysis re-applies each block's own
norm, projections and rotary tables and takes an explicit softmax. That is a second implementation
of something the model already does, and a second implementation is worth exactly what its
equivalence proof is worth.

So the gate at the top of this file is the whole file's foundation: on a **perturbed** model in
float64, the recomputed probabilities contracted with $V$ and pushed through ``out_proj`` must equal
what the module actually returned. Perturbed rather than freshly initialised, because a stack that
starts near the identity -- LayerScale at $10^{-2}$ -- would pass on a badly broken recompute. Two
negative controls follow it, each breaking one operand, because an equivalence assertion that
cannot fail proves nothing about the one that passes.

Everything below the gate is arithmetic on probabilities the gate has already vouched for: the
truncation-aware ceiling, the per-anchor entropy, the mass-by-distance histogram, the composed
reach, and the analysis's own protocol, skips and figures.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.eval import run as shared_run
from teb_vae.lag_attn_rws.eval._reuse import labels, report
from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext
from teb_vae.lag_attn_rws.eval.config_schema import merge_eval_overrides, validate_eval_config
from teb_vae.lag_attn_transformer_rws.eval import encoder_attention as recompute
from teb_vae.lag_attn_transformer_rws.eval.analyses import encoder_attention as analysis
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

from .conftest import TINY_KWARGS, build_stream_encoder

#: How many segments the fixture run profiles. Small: every assertion below is about the shape and
#: the arithmetic of the readout, and the fixture's shards are white noise, so more segments buy
#: nothing but time every test in this file pays for once.
FIXTURE_CAP = 6

#: Tolerance for the float64 equivalence gate. Two contractions of the same operands in a different
#: order, so this is round-off on $O(1)$ activations and nothing else -- anything looser would stop
#: distinguishing a reordering from a wrong mask.
EQUIVALENCE_TOL = 1e-10


# =============================================================================
# Fixtures
# =============================================================================
def _perturb(module: torch.nn.Module, *, seed: int = 0, scale: float = 0.35) -> torch.nn.Module:
    """Move every parameter off its initialisation, so the equivalence gate has something to test.

    Args:
        module: The module to perturb, in place.
        seed: Seed for the perturbation.
        scale: Standard deviation of the added noise.

    Returns:
        The same module, in eval mode.
    """
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * scale
            )
    return module.eval()


@pytest.fixture(scope="module")
def blocks() -> Dict[str, Any]:
    """One perturbed target block and one perturbed source block, in float64.

    Built from the two encoders rather than constructed directly, so the windowed and full-prefix
    cases are the ones the model actually builds rather than two the test chose.
    """
    built: Dict[str, Any] = {}
    for index, stream in enumerate(recompute.STREAMS):
        encoder = build_stream_encoder(stream)
        _perturb(encoder, seed=index)
        built[stream] = encoder.attention_blocks[0].attn.double().eval()
    return built


@pytest.fixture(scope="module")
def stream_input() -> torch.Tensor:
    """A seeded $(B, T, d)$ residual stream at the tiny geometry, in float64."""
    generator = torch.Generator().manual_seed(7)
    return torch.randn(
        (3, int(TINY_KWARGS["sequence_length"]), int(TINY_KWARGS["d_model"])),
        generator=generator, dtype=torch.float64,
    )


@pytest.fixture(scope="module")
def evaluation_loader(trained_run, repointed_overrides):
    """The evaluation loader the pipeline would build, over the generated multi-class shards."""
    from train.data_module import GraphDataModule

    config = merge_eval_overrides(
        load_config(str(shared_run.resolved_config_for(Path(trained_run)))),
        repointed_overrides,
    )
    shared_run.force_single_process_loader(config)
    return GraphDataModule(config).test_dataloader()


@pytest.fixture(scope="module")
def eval_config(trained_run, repointed_overrides) -> Dict[str, Any]:
    """The validated ``eval_config`` block, with this analysis's cap set."""
    config = merge_eval_overrides(
        load_config(str(shared_run.resolved_config_for(Path(trained_run)))),
        repointed_overrides,
    )
    validated = validate_eval_config(config)
    validated["caps"] = {**(validated.get("caps") or {}), analysis.CAP_NAME: FIXTURE_CAP}
    return validated


@pytest.fixture(scope="module")
def loaded_task(trained_run):
    """The checkpoint rebuilt through this package's binding, on the CPU."""
    return shared_run.load_task(
        Path(trained_run), torch.device("cpu"), binding=TRF_BINDING
    )


@pytest.fixture(scope="module")
def analysis_run(loaded_task, evaluation_loader, eval_config, tmp_path_factory) -> Dict[str, Any]:
    """One run of the analysis against the fixture checkpoint and shards."""
    output_dir = tmp_path_factory.mktemp("encoder_attention")
    context = AnalysisContext(
        collection=None, config={}, task=loaded_task, loader=evaluation_loader
    )
    result = analysis.run_encoder_attention_analysis(
        context, eval_config=eval_config, output_dir=output_dir, probe=None
    )
    directory = Path(output_dir) / analysis.ANALYSIS_DIRNAME
    return {
        "result": result,
        "directory": directory,
        "entropy": pd.read_csv(directory / analysis.ENTROPY_FILENAME),
        "distance": pd.read_csv(directory / analysis.DISTANCE_FILENAME),
        "reach": pd.read_csv(directory / analysis.REACH_FILENAME),
        "per_recording": pd.read_csv(directory / analysis.PER_RECORDING_FILENAME),
    }


# =============================================================================
# The equivalence gate
# =============================================================================
def module_output_from(module: Any, probabilities: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Finish the attention from a probability tensor, exactly as the module's forward does.

    Args:
        module: The ``CausalSelfAttention`` block.
        probabilities: ``(B, H, T, T)``.
        x: The block's input.

    Returns:
        ``(B, T, d)``, through the module's own value projection and output projection.
    """
    batch, seq_len, _ = x.shape
    shape = (batch, seq_len, module.num_heads, module.d_head)
    value = module.v_proj(module.norm(x)).view(shape).transpose(1, 2)
    merged = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, seq_len, -1)
    return module.out_proj(merged)


@pytest.mark.parametrize("stream", list(recompute.STREAMS))
def test_the_recomputed_probabilities_reproduce_the_modules_own_output(
    blocks, stream_input, stream: str
) -> None:
    """The gate. If this holds, every reduction below is a reduction of what the model computed;
    if it does not, they are reductions of something else that happens to sum to one."""
    module = blocks[stream]
    probabilities = recompute.attention_probabilities(module, stream_input)

    rebuilt = module_output_from(module, probabilities, stream_input)
    actual = module(stream_input)

    assert torch.allclose(rebuilt, actual, rtol=0.0, atol=EQUIVALENCE_TOL), float(
        (rebuilt - actual).abs().max()
    )


@pytest.mark.parametrize("stream", list(recompute.STREAMS))
def test_the_recomputed_probabilities_are_the_kernels_own_entrywise(
    blocks, stream_input, stream: str
) -> None:
    r"""The gate above pins $P V W^O$, not $P$, and that is weaker than it looks: $V$ is
    $(T, d_h)$ with $d_h < T$, so every row of $P$ has a null space under $\cdot V$ and a wrong
    probability tensor can reproduce the module's output exactly. This closes that hole by making
    the contraction invertible -- the fused kernel is driven with the identity as its value basis,
    so what it returns *is* $P$ -- and it is driven with the module's own $Q$, $K$, mask and
    ``is_causal`` flag, so the reference is the kernel the model attends through rather than a
    second transcription of it."""
    module = blocks[stream]
    seq_len = int(stream_input.shape[1])
    query, key, _value = module._project(module.norm(stream_input))
    basis = torch.eye(seq_len, dtype=stream_input.dtype).expand(
        int(query.shape[0]), int(query.shape[1]), seq_len, seq_len
    ).contiguous()
    reference = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        basis,
        attn_mask=None if module.attn_mask is None else module.attn_mask[:seq_len, :seq_len],
        dropout_p=0.0,
        is_causal=module.is_causal,
    )

    recomputed = recompute.attention_probabilities(module, stream_input)

    assert torch.allclose(recomputed, reference, rtol=0.0, atol=EQUIVALENCE_TOL), float(
        (recomputed - reference).abs().max()
    )


def test_the_perturbation_actually_moved_the_module(stream_input) -> None:
    """Not vacuous by construction: the same block is built twice from one seed and only one copy
    is perturbed, so this measures the perturbation rather than the block's output magnitude. An
    absolute-magnitude assertion would not -- a freshly initialised block already returns $O(0.1)$,
    so it would pass at ``scale=0.0`` and the gate above would then be running on a stack close
    enough to its initialisation to hide a broken recompute."""
    # One seed either side, so the two encoders are initialised identically and `_perturb` is the
    # only thing that differs between them.
    torch.manual_seed(4242)
    untouched = build_stream_encoder("source").attention_blocks[0].attn.double().eval()
    torch.manual_seed(4242)
    perturbed = _perturb(
        build_stream_encoder("source"), seed=1
    ).attention_blocks[0].attn.double().eval()

    baseline = untouched(stream_input)
    moved = perturbed(stream_input)

    relative = float((moved - baseline).abs().max() / baseline.abs().max())
    assert relative > 0.1, relative


def test_the_wrong_mask_is_detected(blocks, stream_input) -> None:
    """First negative control: the windowed block recomputed under the full causal prefix. The
    band mask is the one operand that is not a parameter, so it is the one a recompute could get
    wrong while every projection stayed right."""
    module = blocks["source"]
    seq_len = int(stream_input.shape[1])
    hidden = module.norm(stream_input)
    shape = (int(stream_input.shape[0]), seq_len, module.num_heads, module.d_head)
    query = module.rope(module.q_proj(hidden).view(shape).transpose(1, 2))
    key = module.rope(module.k_proj(hidden).view(shape).transpose(1, 2))
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(module.d_head))
    triangle = torch.ones((seq_len, seq_len), dtype=torch.bool).tril()
    wrong = torch.softmax(scores.masked_fill(~triangle, float("-inf")), dim=-1)

    rebuilt = module_output_from(module, wrong, stream_input)

    assert not torch.allclose(rebuilt, module(stream_input), rtol=0.0, atol=EQUIVALENCE_TOL)


def test_omitting_the_rotary_encoding_is_detected(blocks, stream_input) -> None:
    """Second negative control: rotary position encoding makes a score a function of $t - j$, and
    a recompute that dropped it would still produce a valid-looking distribution over lags."""
    module = blocks["target"]
    seq_len = int(stream_input.shape[1])
    hidden = module.norm(stream_input)
    shape = (int(stream_input.shape[0]), seq_len, module.num_heads, module.d_head)
    query = module.q_proj(hidden).view(shape).transpose(1, 2)
    key = module.k_proj(hidden).view(shape).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(module.d_head))
    mask = recompute.admitted_keys(module, seq_len, device=scores.device)
    unrotated = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)

    rebuilt = module_output_from(module, unrotated, stream_input)

    assert not torch.allclose(rebuilt, module(stream_input), rtol=0.0, atol=EQUIVALENCE_TOL)


def test_train_and_eval_mode_give_the_same_probabilities(blocks, stream_input) -> None:
    """Attention-probability dropout is structurally zero in this architecture, so the mode a
    caller happens to leave the model in cannot move this readout. Asserted rather than assumed,
    because it is the one property that would make the whole analysis mode-dependent."""
    module = blocks["target"]
    module.train()
    training = recompute.attention_probabilities(module, stream_input)
    module.eval()

    assert torch.equal(training, recompute.attention_probabilities(module, stream_input))


def test_the_model_side_property_the_mode_independence_rests_on(blocks) -> None:
    """The test above compares two recomputes, and the recompute reads no dropout -- so it is mode
    independent whatever the model does, and on its own it would stay green through exactly the
    model change that breaks the claim. The claim is about the *model*: the kernel is called with
    ``dropout_p=0.0`` literally, so no mode can put dropout on the probabilities. Read off the
    source, because there is no forward output that distinguishes ``dropout_p=0.0`` from a module
    whose ``dropout.p`` is itself zero."""
    import inspect

    source = inspect.getsource(type(blocks["target"]).forward)

    assert "dropout_p=0.0" in source, source
    assert "dropout_p=self" not in source.replace(" ", ""), source


# =============================================================================
# The recompute's own properties
# =============================================================================
@pytest.mark.parametrize("stream", list(recompute.STREAMS))
def test_rows_sum_to_one_and_masked_entries_are_exactly_zero(
    blocks, stream_input, stream: str
) -> None:
    """Exactly zero rather than small: an ``-inf`` score exponentiates to zero, and a masked entry
    carrying $10^{-12}$ of mass would put mass at a distance the model cannot see."""
    module = blocks[stream]
    seq_len = int(stream_input.shape[1])
    probabilities = recompute.attention_probabilities(module, stream_input)
    mask = recompute.admitted_keys(module, seq_len, device=probabilities.device)

    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1)), atol=1e-12
    )
    assert float(probabilities[..., ~mask].abs().max()) == 0.0


def test_the_mask_comes_from_the_block_rather_than_from_a_caller(blocks) -> None:
    r"""Each block's admitted-key count is $\min(t+1, c)$ with $c = T$ for the full prefix and
    $c = W$ for the window -- which is the ceiling the entropy is read against, so a mask read from
    the wrong place would move every ratio in the run without moving anything else."""
    seq_len = int(TINY_KWARGS["sequence_length"])
    window = int(TINY_KWARGS["source_attention_window"])
    steps = torch.arange(seq_len) + 1

    target = recompute.admitted_keys(blocks["target"], seq_len, device="cpu").sum(dim=-1)
    source = recompute.admitted_keys(blocks["source"], seq_len, device="cpu").sum(dim=-1)

    assert torch.equal(target, steps.clamp(max=seq_len))
    assert torch.equal(source, steps.clamp(max=window))


def test_the_hooks_are_removed_even_when_the_pass_raises(blocks) -> None:
    """A hook that outlived a failed pass keeps firing on every later forward in the process,
    holding one batch's activations alive and refilling a store nobody is reading."""
    refs = [
        recompute.BlockRef("source", 0, blocks["source"], blocks["source"].window)
    ]
    module = blocks["source"]
    before = len(module._forward_pre_hooks)

    with pytest.raises(RuntimeError):
        with recompute.captured_block_inputs(refs):
            assert len(module._forward_pre_hooks) == before + 1
            raise RuntimeError("the pass failed")

    assert len(module._forward_pre_hooks) == before


def test_the_blocks_are_found_in_stream_and_stack_order(tiny_kwargs) -> None:
    """The order every table is written in, and the order a reader compares two runs down."""
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs)

    found = [(ref.stream, ref.index) for ref in recompute.attention_blocks(model)]

    assert found == (
        [("target", index) for index in range(int(tiny_kwargs["target_attention_blocks"]))]
        + [("source", index) for index in range(int(tiny_kwargs["source_attention_blocks"]))]
    )
    assert [ref.window for ref in recompute.attention_blocks(model)][0] is None


def test_a_model_without_the_encoders_is_refused_by_name() -> None:
    """A silent ``getattr`` default would report a model whose encoders were renamed as a model
    with no attention to describe."""

    class _Bare:
        pass

    with pytest.raises(AttributeError, match="target_encoder"):
        recompute.attention_blocks(_Bare())


# =============================================================================
# The streamed reductions
# =============================================================================
def _delta_probabilities(
    *, batch: int, heads: int, seq_len: int, offset: int
) -> torch.Tensor:
    """Build a probability tensor that puts all of each row's mass at a fixed offset.

    Built directly rather than forced through a module's weights: rotary position encoding makes
    "the weights that produce a single-offset attention" a solve rather than a construction, and
    the module path is already pinned by the equivalence gate above. What is under test here is
    the arithmetic the reductions do to a probability tensor.

    Args:
        batch: Batch size.
        heads: Head count.
        seq_len: Sequence length.
        offset: The distance $t - j$ every row puts its mass at, or at $j = 0$ where $t < $ offset.

    Returns:
        ``(batch, heads, seq_len, seq_len)``.
    """
    probabilities = torch.zeros((batch, heads, seq_len, seq_len), dtype=torch.float32)
    for step in range(seq_len):
        probabilities[:, :, step, max(step - offset, 0)] = 1.0
    return probabilities


def test_a_single_offset_attention_has_zero_entropy_and_a_delta_profile() -> None:
    """The known answer. A row with all its mass on one key carries no entropy, and its mass by
    distance is a spike at that offset -- so a reduction that reported anything else here would be
    reporting something other than what it claims to."""
    seq_len, offset = 16, 3
    probabilities = _delta_probabilities(batch=2, heads=2, seq_len=seq_len, offset=offset)
    counts = torch.arange(seq_len) + 1

    stats = recompute.block_sample_stats(
        probabilities, anchors=(offset, seq_len), admitted_counts=counts
    )

    assert np.allclose(stats.entropy_nats, 0.0, atol=1e-6)
    profile = stats.distance_mass[0, 0]
    assert profile[offset] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.delete(profile, offset), 0.0, atol=1e-9)
    assert stats.mean_distance[0, 0] == pytest.approx(float(offset), abs=1e-6)


def test_the_ceiling_is_the_mean_log_of_the_admitted_count() -> None:
    r"""$\operatorname{mean}_t \log \min(t+1, c)$, and never $\log T$: a quarter of the trained
    anchors have structurally truncated support, so a head attending uniformly over everything
    available reads as concentrated when measured against a ceiling it could not reach."""
    seq_len, window = 12, 4
    probabilities = _delta_probabilities(batch=1, heads=1, seq_len=seq_len, offset=0)
    counts = (torch.arange(seq_len) + 1).clamp(max=window)

    stats = recompute.block_sample_stats(
        probabilities, anchors=(2, seq_len), admitted_counts=counts
    )

    expected = float(np.mean(np.log(np.minimum(np.arange(2, seq_len) + 1, window))))
    assert stats.ceiling_nats[0] == pytest.approx(expected, rel=1e-9)
    assert float(np.max(stats.ceiling_nats)) < math.log(seq_len)


def test_the_entropy_is_taken_per_anchor_rather_than_on_the_averaged_profile() -> None:
    """Entropy is concave, so a mixture's entropy is at least the mean of the entropies mixed. A
    head whose focus *shifts* across the segment therefore reads as having no focus at all if the
    profile is averaged first, which is the opposite of what it does."""
    seq_len = 8
    probabilities = torch.zeros((1, 1, seq_len, seq_len))
    for step in range(seq_len):
        probabilities[0, 0, step, step] = 1.0
    counts = torch.arange(seq_len) + 1

    stats = recompute.block_sample_stats(
        probabilities, anchors=(0, seq_len), admitted_counts=counts
    )
    averaged = probabilities[0, 0].mean(dim=0)
    averaged_entropy = float(torch.special.entr(averaged).sum())

    assert stats.entropy_nats[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert averaged_entropy > 1.0


def test_an_empty_anchor_range_is_refused_rather_than_scored_as_zero() -> None:
    """A block scored over no anchors measured nothing, and a zero entropy for it would read as a
    perfectly focused head."""
    probabilities = _delta_probabilities(batch=1, heads=1, seq_len=8, offset=0)

    with pytest.raises(ValueError, match="empty"):
        recompute.block_sample_stats(
            probabilities, anchors=(4, 4), admitted_counts=torch.arange(8) + 1
        )


def test_the_accumulator_does_not_grow_with_the_number_of_segments() -> None:
    """The property that lets the cap bound time rather than memory: everything kept is a running
    sum whose shape is set by the head count and the sequence length."""
    accumulator = recompute.BlockAccumulator(n_heads=2, n_distances=8)
    stats = recompute.block_sample_stats(
        _delta_probabilities(batch=4, heads=2, seq_len=8, offset=1),
        anchors=(1, 8), admitted_counts=torch.arange(8) + 1,
    )

    shapes = []
    for _ in range(5):
        accumulator.update(stats, range(4))
        shapes.append((accumulator.entropy_sum.shape, accumulator.distance_sum.shape))

    assert len(set(shapes)) == 1
    assert accumulator.n_segments == 20
    assert accumulator.distance_sum.shape == (2, 8)


def test_an_empty_accumulator_reports_nan_rather_than_zero() -> None:
    """Zero is a measurement -- a perfectly focused head -- and a cell nothing was added to has
    not made one."""
    accumulator = recompute.BlockAccumulator(n_heads=3, n_distances=5)

    assert np.isnan(accumulator.entropy_mean).all()
    assert np.isnan(accumulator.ceiling_mean)
    assert np.isnan(accumulator.distance_profile).all()


def test_the_mass_quantile_is_a_bin_rather_than_an_interpolated_position() -> None:
    """A distance of $3.5$ steps is not a thing an attention row can have."""
    profile = np.zeros(10)
    profile[2] = 0.6
    profile[7] = 0.4

    assert recompute.mass_quantile(profile, 0.5) == 2.0
    assert recompute.mass_quantile(profile, 0.95) == 7.0
    assert np.isnan(recompute.mass_quantile(np.zeros(10), 0.5))


# =============================================================================
# The measured reach
# =============================================================================
def test_the_composed_reach_reproduces_the_structural_bound_at_the_window_edge(
    tiny_kwargs
) -> None:
    r"""The composition is the structural formula with each block's *measured* hop in place of the
    largest hop it was allowed, so a stack whose every block sat at its window edge must land
    exactly on $R_U$. That is what makes the two comparable at all."""
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs)
    geometry = recompute.stream_geometry(model, "source")
    window = int(tiny_kwargs["source_attention_window"])
    blocks = int(tiny_kwargs["source_attention_blocks"])

    at_the_edge = recompute.composed_reach(
        geometry["conv_reach_steps"], [window - 1] * blocks,
        sequence_length=int(tiny_kwargs["sequence_length"]),
    )

    assert at_the_edge == pytest.approx(float(geometry["structural_bound_steps"]))
    assert np.isnan(
        recompute.composed_reach(geometry["conv_reach_steps"], [1.0, np.nan], sequence_length=64)
    )


def test_the_composed_reach_cannot_exceed_the_segment_it_was_measured_on() -> None:
    r"""The same $\min(\cdot, T)$ the structural formula applies, and not cosmetic: on the
    full-prefix target encoder the per-block hops routinely sum past the segment, and an uncapped
    figure would report more history than the segment contains."""
    assert recompute.composed_reach(21, [200.0, 200.0, 200.0], sequence_length=300) == 300.0
    assert recompute.composed_reach(21, [5.0, 5.0], sequence_length=300) == 31.0


def test_the_unbounded_source_arm_reports_an_absent_bound_rather_than_the_sequence_length(
    tiny_kwargs
) -> None:
    """"No bound" and "a bound that happens to equal $T$" are different statements, and the arm the
    whole locality sweep is measured against is the first one."""
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**{**tiny_kwargs, "source_attention_window": None})
    geometry = recompute.stream_geometry(model, "source")

    assert geometry["structural_bound_absent"] is True
    assert geometry["structural_bound_steps"] is None
    assert geometry["attention_window"] is None
    assert recompute.stream_geometry(model, "target")["structural_bound_absent"] is True


@pytest.mark.parametrize("window", [2, 4])
def test_a_narrower_window_lowers_the_structural_bound(tiny_kwargs, window: int) -> None:
    """The axis the locality sweep varies, which every measured reach is read against."""
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**{**tiny_kwargs, "source_attention_window": window})
    geometry = recompute.stream_geometry(model, "source")

    assert geometry["structural_bound_steps"] == (
        geometry["conv_reach_steps"] + geometry["n_attention_blocks"] * (window - 1)
    )


# =============================================================================
# The analysis
# =============================================================================
def test_the_analysis_returns_the_protocol_and_declares_its_grouped_frame(analysis_run) -> None:
    result = analysis_run["result"]

    for key in ("n_samples", "composition", "plan"):
        assert key in result
    assert result["n_samples"] == FIXTURE_CAP
    assert result["plan"]["capped"] is True
    assert result["plan"]["cap"] == FIXTURE_CAP
    assert result["plan"]["cap_key"] == analysis.CAP_NAME
    declared = result["grouped_frames"]
    assert [entry["stem"] for entry in declared] == ["encoder_attention_per_recording"]
    assert list(entry["value_columns"] for entry in declared) == [list(analysis.VALUE_COLUMNS)]


def test_every_declared_file_was_written(analysis_run) -> None:
    for name in analysis_run["result"]["files"]:
        assert (analysis_run["directory"] / name).is_file(), name


def test_the_class_resolved_rows_are_in_clinical_order(analysis_run) -> None:
    """Pooled first, then HIE / acidosis / healthy -- the evaluation's one cohort order, worst
    first. Alphabetical would put ``acidosis`` first on every table, and the figure beside it reads
    its order from the same list."""
    for frame in ("entropy", "distance", "reach"):
        seen: List[str] = []
        for value in analysis_run[frame][labels.CLASS_COLUMN]:
            if not seen or seen[-1] != value:
                seen.append(str(value))
        expected = [
            name for name in [recompute.POOLED_CLASS, "hie", "acidosis", "healthy"]
            if name in set(seen)
        ]
        assert seen == expected, frame


def test_the_source_distance_mass_stops_at_the_window(analysis_run) -> None:
    r"""Exactly zero beyond $\min(W_U, T) - 1$, because the block admits no key there. The table is
    truncated at that bound, so what is asserted is the bound itself -- and the window is read off
    the run's own record rather than from a constant, because a shard shorter than $W_U$ makes the
    sequence the binding constraint and the two are not the same claim."""
    result = analysis_run["result"]
    window = result["geometry"]["source"]["attention_window"]
    reach = min(int(window), int(result["sequence_length"]))
    source = analysis_run["distance"][analysis_run["distance"]["stream"] == "source"]

    assert int(source["distance_steps"].max()) == reach - 1
    for _key, block in source.groupby(["clinical_class", "block", "head"]):
        assert float(block["mass"].sum()) == pytest.approx(1.0, abs=1e-6)


def test_the_measured_source_reach_stays_inside_its_structural_bound(analysis_run) -> None:
    """The composition is an estimate of what the encoder *uses*; the structural bound is what it
    may reach. The first cannot exceed the second, and a composition that did would be measuring
    something other than the distances the mask admits."""
    result = analysis_run["result"]
    reach = analysis_run["reach"]
    source = reach[
        (reach["stream"] == "source") & (reach["clinical_class"] == recompute.POOLED_CLASS)
    ]
    admitted = min(
        int(result["geometry"]["source"]["attention_window"]), int(result["sequence_length"])
    )

    assert len(source)
    row = source.iloc[0]
    assert row["composed_reach_p95_steps"] <= row["structural_bound_steps"]
    assert bool(row["structural_bound_absent"]) is False
    assert row["p95_distance_steps"] <= admitted - 1


def test_the_target_reach_reports_its_bound_as_absent(analysis_run) -> None:
    reach = analysis_run["reach"]
    target = reach[reach["stream"] == "target"]

    assert len(target)
    assert bool(target.iloc[0]["structural_bound_absent"]) is True
    assert pd.isna(target.iloc[0]["structural_bound_steps"])


def test_the_per_recording_reduction_is_on_recordings_rather_than_segments(analysis_run) -> None:
    """The unit every clinical question in this pipeline is asked in. A frame that was still per
    segment would hand the runner's violin fan-out one point per segment, so a recording
    contributing thirty segments would weigh thirty times one contributing one."""
    per_recording = analysis_run["per_recording"]
    result = analysis_run["result"]

    assert len(per_recording) == per_recording["guid"].nunique()
    assert len(per_recording) <= int(result["n_samples"])
    assert int(per_recording["n_segments"].sum()) == int(result["n_samples"])
    for column in analysis.VALUE_COLUMNS:
        assert column in per_recording.columns
    for column in labels.GROUP_COLUMNS:
        assert column in per_recording.columns


def test_the_headline_carries_the_six_registered_scalars(analysis_run) -> None:
    """The arm tables read the headline block and nothing else, so a measured reach that stayed in
    a CSV could not give the window sweep the measured x-axis that is half its purpose."""
    headline = analysis_run["result"]["headline"]

    assert set(headline) == set(analysis.HEADLINE_KEYS)
    for name in analysis.HEADLINE_KEYS:
        assert headline[name] is not None and np.isfinite(float(headline[name]))
    assert 0.0 <= headline["entropy_ratio_target"] <= 1.0
    assert 0.0 <= headline["entropy_ratio_source"] <= 1.0


def test_the_registered_headline_names_are_the_bindings(analysis_run) -> None:
    """The binding is where the six become headline keys; a name that drifted would leave the arm
    table reading a column no run produces."""
    registered = dict(TRF_BINDING.headline_scalars)

    for name in analysis.HEADLINE_KEYS:
        assert registered[f"encoder_attention_{name}"] == (
            analysis.ANALYSIS_DIRNAME, "headline", name
        )


def test_the_headline_resolves_out_of_the_results_block(analysis_run) -> None:
    """Through the same walker the summary uses, so a path that resolved only in the test would
    fail here rather than silently produce a block of nulls in every run."""
    results = {analysis.ANALYSIS_DIRNAME: analysis_run["result"]}

    for name, path in TRF_BINDING.headline_scalars:
        assert report._dig(results, path) is not None, name


def test_no_verdict_is_registered(analysis_run) -> None:
    """This analysis describes a mechanism rather than adjudicating a difference: a separation
    visible here is a reason to look, not a finding."""
    result = analysis_run["result"]

    assert "verdicts" not in result
    assert not any(str(key).startswith("verdict") for key in result)


def test_the_stratified_draw_reaches_every_shard(analysis_run, multi_class_shards) -> None:
    """A prefix would not: the split is eight concatenated per-subgroup files, so the first $n$
    segments are one subgroup and one clinical class -- and this analysis cuts every readout by
    clinical class, which would then have one cohort to cut into."""
    plan = analysis_run["result"]["plan"]

    assert plan["stratified_by"] == "source_file_basename"
    assert plan["n_shards_drawn"] == min(FIXTURE_CAP, len(multi_class_shards))


def test_two_runs_at_one_seed_emit_byte_identical_tables(
    loaded_task, evaluation_loader, eval_config, tmp_path_factory
) -> None:
    """The draw, the pass and every reduction are seeded or deterministic, so a re-run of the same
    checkpoint against the same split must produce the same tables -- which is what makes a
    difference between two *arms* readable as a difference between two arms."""
    digests = []
    for index in range(2):
        output_dir = tmp_path_factory.mktemp(f"repeat_{index}")
        analysis.run_encoder_attention_analysis(
            AnalysisContext(
                collection=None, config={}, task=loaded_task, loader=evaluation_loader
            ),
            eval_config=eval_config, output_dir=output_dir, probe=None,
        )
        directory = Path(output_dir) / analysis.ANALYSIS_DIRNAME
        digests.append(
            {
                name: (directory / name).read_bytes()
                for name in (
                    analysis.ENTROPY_FILENAME, analysis.DISTANCE_FILENAME,
                    analysis.REACH_FILENAME, analysis.PER_RECORDING_FILENAME,
                )
            }
        )

    assert digests[0] == digests[1]


# =============================================================================
# The two skips
# =============================================================================
def test_an_absent_cap_records_a_skip_naming_the_key(tmp_path) -> None:
    """Absence means zero, per this package's opt-in rule -- and the skip has to say which key
    would enable it, because an analysis that silently produced nothing is indistinguishable from
    one that found nothing."""
    result = analysis.run_encoder_attention_analysis(
        AnalysisContext(collection=None, config={}, task=object(), loader=object()),
        eval_config={"caps": {}, "seed": 0}, output_dir=tmp_path, probe=None,
    )

    assert result["skipped"] is True
    assert f"caps.{analysis.CAP_NAME}" in result["reason"]
    assert result["n_samples"] is None
    assert (tmp_path / analysis.ANALYSIS_DIRNAME).is_dir()


def test_an_offline_re_run_with_no_model_records_a_skip(tmp_path) -> None:
    """``--only encoder_attention`` against a finished directory has no model to recompute with.
    It records a skip and exits 0 rather than raising, which is what keeps the offline re-run
    honest about what it did not do."""
    result = analysis.run_encoder_attention_analysis(
        AnalysisContext(collection=None, config={}, task=None, loader=None),
        eval_config={"caps": {analysis.CAP_NAME: 4}, "seed": 0},
        output_dir=tmp_path, probe=None,
    )

    assert result["skipped"] is True
    assert "no model" in result["reason"] or "built neither" in result["reason"]
    assert result["plan"]["cap"] == 4


def test_a_pass_that_scored_nothing_still_writes_schemad_tables(
    monkeypatch, loaded_task, evaluation_loader, eval_config, tmp_path
) -> None:
    """A split whose shards filter down to no loadable segment is not refused anywhere -- preflight
    rejects only an empty dataset *list* -- so the pass can legitimately return zero segments. Every
    emitted table must still carry its header, because the headline block reads these frames by
    column name and a column-less frame would turn an empty measurement into a ``KeyError`` where
    the protocol asks for six nulls."""
    empty = recompute.PassResult(
        accumulators={}, per_segment=[], heatmaps={}, geometry={
            stream: recompute.stream_geometry(loaded_task.orig_model, stream)
            for stream in recompute.STREAMS
        },
        anchor_range=(0, 0), n_heads=0, seq_len=0, n_segments=0, n_batches=0,
    )
    monkeypatch.setattr(analysis, "run_encoder_attention_pass", lambda *a, **k: empty)

    result = analysis.run_encoder_attention_analysis(
        AnalysisContext(
            collection=None, config={}, task=loaded_task, loader=evaluation_loader
        ),
        eval_config=eval_config, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / analysis.ANALYSIS_DIRNAME
    for filename in (
        analysis.ENTROPY_FILENAME, analysis.DISTANCE_FILENAME, analysis.REACH_FILENAME
    ):
        frame = pd.read_csv(directory / filename)
        assert list(frame.columns), f"{filename} was written without a header"
        assert labels.CLASS_COLUMN in frame.columns
    assert set(result["headline"]) == set(analysis.HEADLINE_KEYS)
    assert all(value is None for value in result["headline"].values())


def test_a_skip_registers_the_headline_keys_as_null(tmp_path) -> None:
    """Null rather than omitted, so the arm table's column exists whether the analysis ran or not
    -- a missing column and a column of nulls read differently in a table of arms."""
    result = analysis.run_encoder_attention_analysis(
        AnalysisContext(collection=None, config={}, task=None, loader=None),
        eval_config={"caps": {analysis.CAP_NAME: 4}, "seed": 0},
        output_dir=tmp_path, probe=None,
    )

    assert set(result["headline"]) == set(analysis.HEADLINE_KEYS)
    assert all(value is None for value in result["headline"].values())


def test_the_cap_does_not_fire_the_inert_cap_warning(eval_config) -> None:
    """The shared check warns about a cap ``max_samples`` has already made unreachable. This one is
    set and used, so it must not appear -- a run whose warnings include a cap that did its job
    teaches an operator to ignore the warnings.

    Asserted against a finite ``max_samples`` rather than the committed delta's ``null``: the check
    returns early on ``None``, so silence under the shipped config would hold even if this cap were
    misspelt, and would measure the short circuit rather than the cap."""
    assert eval_config["max_samples"] is None, "the delta's null is what makes the below necessary"
    bounded = {**eval_config, "max_samples": FIXTURE_CAP * 4}

    warnings = report.check_inert_caps(bounded)

    assert not any(analysis.CAP_NAME in warning for warning in warnings)


def test_an_inert_cap_would_still_be_reported() -> None:
    """Not vacuous: the check has to fire on a cap that genuinely never bites."""
    warnings = report.check_inert_caps(
        {"max_samples": 4, "caps": {analysis.CAP_NAME: 64}}
    )

    assert any(analysis.CAP_NAME in warning for warning in warnings)
