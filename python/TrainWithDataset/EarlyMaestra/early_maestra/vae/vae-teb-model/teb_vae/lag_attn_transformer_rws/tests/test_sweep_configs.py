r"""Lint for the architecture arms: each is ``default.yaml`` plus a *declared* delta and nothing else.

Twelve arms sweep three axes. **Architecture** (A1, A2) asks what the convolution stem and the
source encoder's asymmetry are worth; the shipped ``default.yaml`` is A3, the recommended arm, and
A0 is the model this package is compared against -- ``teb_vae/lag_attn_rws`` -- not a file here.
**Source locality** sweeps $W_U \in \{8, 32, 64, \text{unbounded}\}$ around the shipped $16$, plus
the causal-input reach budget the availability representation exists to make trainable.
**Depth and width** moves $N_Y$, $N_U$ and $d_{\mathrm{ff}}$ one key at a time.

The lint holds :data:`DECLARED_DELTAS`, and it is the point of the file. Every
``configs/sweep_*.yaml`` must appear in it, and each arm's *resolved* delta against ``default.yaml``
must equal its declared keys exactly -- so a multi-key arm is a declaration rather than an
exception, and an arm that quietly acquired a second change stops being an answer to its own
question. Two arms are declared multi-key on purpose: A1 empties the stem schedules together (they
are positional against each other) while also making both streams symmetric, and A2 makes the
source encoder the target encoder in both depth and context.

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development box
-- a key that does not resolve, a stray second delta, a parameter cost that is not what the arm's
header claims, a reach budget the filter bank refuses -- rather than days into a production run.
The parameter assertions are deltas rather than absolute totals: the absolute number is pinned once,
in the design record, so a legitimate shared change to an imported downstream component cannot fail
a test here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import pytest
import tempfile

import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT = _CONFIG_DIR / "default.yaml"

_VAE = "model_config.VAE_model"
_CONV_KERNELS = f"{_VAE}.encoder_conv_kernels"
_CONV_DILATIONS = f"{_VAE}.encoder_conv_dilations"
_D_FF = f"{_VAE}.encoder_d_ff"
_TARGET_BLOCKS = f"{_VAE}.target_attention_blocks"
_SOURCE_BLOCKS = f"{_VAE}.source_attention_blocks"
_SOURCE_WINDOW = f"{_VAE}.source_attention_window"
_REACH = f"{_VAE}.causal_reach_budget_s"

#: Every arm, by file name, with the complete set of leaves it is allowed to move against
#: ``default.yaml``. An arm's resolved delta must equal its entry exactly -- neither a missing key
#: (a declaration that stopped being true) nor an extra one (a second change riding along) passes.
DECLARED_DELTAS: Dict[str, Dict[str, Any]] = {
    # Phase 1: architecture. A1 removes the stem *and* symmetrises the streams, so the only
    # difference between it and A2 is the stem itself; A2 symmetrises alone.
    "sweep_arch_a1.yaml": {
        _CONV_KERNELS: [],
        _CONV_DILATIONS: [],
        _SOURCE_BLOCKS: 4,
        _SOURCE_WINDOW: None,
    },
    "sweep_arch_a2.yaml": {_SOURCE_BLOCKS: 4, _SOURCE_WINDOW: None},
    # Phase 2: source locality, and the reach budget.
    "sweep_window_8.yaml": {_SOURCE_WINDOW: 8},
    "sweep_window_32.yaml": {_SOURCE_WINDOW: 32},
    "sweep_window_64.yaml": {_SOURCE_WINDOW: 64},
    "sweep_window_full.yaml": {_SOURCE_WINDOW: None},
    "sweep_reach_120.yaml": {_REACH: 120},
    # Phase 3: depth and width.
    "sweep_target_blocks_3.yaml": {_TARGET_BLOCKS: 3},
    "sweep_target_blocks_5.yaml": {_TARGET_BLOCKS: 5},
    "sweep_source_blocks_2.yaml": {_SOURCE_BLOCKS: 2},
    "sweep_source_blocks_4.yaml": {_SOURCE_BLOCKS: 4},
    "sweep_ff_384.yaml": {_D_FF: 384},
}

#: One causal Transformer block at the shipped widths: $4d^2 + 3 d\,d_{\mathrm{ff}} + 4d$ with
#: $d = 128$, $d_{\mathrm{ff}} = 256$. Every depth arm moves the total by exactly this.
ATTENTION_BLOCK_PARAMS = 4 * 128**2 + 3 * 128 * 256 + 4 * 128

#: Both stems together: two gated causal depthwise convolution blocks per encoder, at kernels
#: $5$ and $9$. This is the A1-to-A2 difference, and the only one.
STEM_PARAMS = 2 * (50_176 + 50_688)

#: Going $d_{\mathrm{ff}} = 256 \to 384$ across the seven attention blocks the shipped
#: architecture holds (four target, three source), at $3 d$ parameters per unit of width.
FF_384_PARAMS = 3 * 128 * (384 - 256) * 7

#: Predicted source-state reach per window arm, $R_U = \min(21 + 3(W_U - 1),\, T)$ in steps, with
#: ``None`` meaning *unbounded* rather than clamped at $T$. Pinned here so a change to the stem
#: schedule or the source depth re-costs every window arm's header loudly.
EXPECTED_SOURCE_BOUND: Dict[str, Optional[int]] = {
    "sweep_window_8.yaml": 42,
    "sweep_window_32.yaml": 114,
    "sweep_window_64.yaml": 210,
    "sweep_window_full.yaml": None,
}

#: Surviving channel counts (target, source) at the one finite reach budget this package ships,
#: measured off the analytic filter bank. Pinned so a filter-bank or channel-selection change
#: re-costs the arm rather than silently launching it against a different guard.
REACH_120_CHANNELS = (78, 29)


def _flatten(node: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a config mapping to ``{dotted path: leaf value}``.

    Non-empty dicts recurse; everything else -- scalars, lists, ``None``, an empty dict -- is a
    leaf, which matches the loader's own merge semantics: a list replaces wholesale, so a list is
    a value and never a namespace.

    Args:
        node: The mapping to flatten.
        prefix: Dotted prefix accumulated so far.

    Returns:
        One entry per leaf.
    """
    flat: Dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def undeclared_arm_files(present: Set[str], declared: Mapping[str, Any]) -> List[str]:
    """Return the ``sweep_*.yaml`` names present on disk that the table does not declare.

    Extracted so the lint's own failure mode is testable: a rule that silently matched nothing
    would leave every arm below unchecked, and the only way to know it fires is to hand it a
    file that should trip it.

    Args:
        present: File names found in the configs directory.
        declared: The declared-delta table.

    Returns:
        The undeclared names, sorted.
    """
    return sorted(name for name in present if name not in declared)


def resolved_delta(arm_flat: Mapping[str, Any], default_flat: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the leaves whose value differs between an arm and the default, with the arm's value.

    Args:
        arm_flat: The arm's flattened resolved config.
        default_flat: ``default.yaml`` flattened.

    Returns:
        ``{dotted path: arm value}`` for every differing leaf.
    """
    return {
        path: arm_flat[path]
        for path in arm_flat
        if path not in default_flat or arm_flat[path] != default_flat[path]
    }


def _resolved(name: str) -> Dict[str, Any]:
    return load_config(str(_CONFIG_DIR / name))


def _model_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a resolved config through the real driver's signature sweep.

    The arms are read the way a launch reads them -- through ``_build_model_kwargs``, which is
    also what translates ``causal_reach_budget_s`` into the four concrete channel tuples -- so a
    key that reaches nothing shows up here rather than on the production box.

    Args:
        config: A fully resolved config mapping.

    Returns:
        The constructor kwargs a launch on this config would build the net from.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
        return LagAttnTrfRwsTrainer(config_file_path=str(path))._build_model_kwargs()


@pytest.fixture(scope="module")
def default_flat() -> Dict[str, Any]:
    return _flatten(load_config(str(_DEFAULT)))


@pytest.fixture(scope="module")
def built() -> Dict[str, SeqVaeLagAttnTrfRws]:
    """Every arm and the default, constructed once at production geometry.

    Construction is what proves a config resolves to a model rather than to a set of keys, and at
    these widths it costs a few tens of milliseconds per arm, so it is done once for the module
    rather than per assertion.
    """
    names = ["default.yaml", *DECLARED_DELTAS]
    return {name: SeqVaeLagAttnTrfRws(**_model_kwargs(_resolved(name))) for name in names}


@pytest.fixture(scope="module")
def params(built) -> Dict[str, int]:
    """Measured parameter total per arm."""
    return {name: sum(p.numel() for p in model.parameters()) for name, model in built.items()}


# --------------------------------------------------------------------------------------
# The lint fires
# --------------------------------------------------------------------------------------
def test_the_lint_reports_a_sweep_file_that_no_table_entry_declares():
    """The guard on the guard. An arm outside the table would run outside every assertion below,
    which is the one failure that produces a confounded result with nothing to read it off."""
    present = {"sweep_arch_a1.yaml", "sweep_undeclared_idea.yaml"}

    assert undeclared_arm_files(present, DECLARED_DELTAS) == ["sweep_undeclared_idea.yaml"]


def test_the_lint_reports_a_declared_delta_that_does_not_match_the_file(default_flat):
    """The other half: a table entry claiming a value the file does not carry. Driven against a
    real arm with its declared value corrupted, so the comparison under test is the real one."""
    arm_flat = _flatten(_resolved("sweep_ff_384.yaml"))

    assert resolved_delta(arm_flat, default_flat) == DECLARED_DELTAS["sweep_ff_384.yaml"]
    assert resolved_delta(arm_flat, default_flat) != {_D_FF: 512}


# --------------------------------------------------------------------------------------
# The inventory is closed and every arm is its declared delta
# --------------------------------------------------------------------------------------
def test_the_configs_directory_holds_exactly_the_declared_arms():
    """Both directions: a declared arm whose file is missing, and a stray file nobody declared."""
    present = {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")}

    assert present == set(DECLARED_DELTAS)


@pytest.mark.parametrize("name", sorted(DECLARED_DELTAS))
def test_an_arm_resolves_with_its_base_consumed(name):
    """``load_config`` must both succeed and eat the ``base:`` directive; a leftover ``base`` key
    would reach the validator as an unknown key and the MLflow param dump as noise."""
    assert "base" not in _resolved(name)


@pytest.mark.parametrize("name", sorted(DECLARED_DELTAS))
def test_an_arm_moves_exactly_its_declared_leaves(name, default_flat):
    """The property the whole file exists for. The key *sets* must match too -- a typo'd override
    adds a path rather than moving one, and a config that reaches nothing trains a different
    architecture than its header describes with nothing raising."""
    arm_flat = _flatten(_resolved(name))

    assert set(arm_flat) == set(default_flat), (
        f"{name} adds or drops config paths rather than overriding one"
    )
    assert resolved_delta(arm_flat, default_flat) == DECLARED_DELTAS[name]


@pytest.mark.parametrize("name", sorted(DECLARED_DELTAS))
def test_an_arm_builds_a_model_through_the_real_driver(name, built):
    """Resolved and swept the way a launch does, so an arm whose key names no constructor argument
    fails here rather than after the run directories and the MLflow run already exist."""
    assert isinstance(built[name], SeqVaeLagAttnTrfRws)


@pytest.mark.parametrize("name", sorted(DECLARED_DELTAS))
def test_no_arm_carries_a_run_identity_of_its_own(name, default_flat):
    """Arms deliberately share the baseline's tag, MLflow experiment and run name: a per-arm
    identity would be a second delta. A run is identified by its ``TEB_RUN_STAMP`` directory and
    the resolved config written beside its checkpoints."""
    arm_flat = _flatten(_resolved(name))

    for path in (
        "general_config.tag",
        "advanced_config.tracking.mlflow.experiment_name",
        "advanced_config.tracking.mlflow.run_name",
        "advanced_config.tracking.mlflow.tags.variant",
    ):
        assert arm_flat[path] == default_flat[path]


# --------------------------------------------------------------------------------------
# Phase 1: the architecture arms
# --------------------------------------------------------------------------------------
def test_the_a1_arm_builds_no_convolution_stem(built):
    """A1 is the arm that asks what the convolutional bias is worth, so an A1 that still carried a
    stem would answer nothing. Asserted on the constructed encoders, not on the config."""
    model = built["sweep_arch_a1.yaml"]

    assert len(model.target_encoder.conv_blocks) == 0
    assert len(model.source_encoder.conv_blocks) == 0
    # The stem's reach collapses to the identity: one step, itself.
    assert model.target_encoder.conv_reach == 1
    assert model.source_encoder.conv_reach == 1


def test_both_phase_one_arms_make_the_source_encoder_the_target_encoder(built):
    """Four blocks, full causal prefix, on both streams. The source bound is *absent* under A1 and
    A2, which is what makes the stem the only difference between them."""
    for name in ("sweep_arch_a1.yaml", "sweep_arch_a2.yaml"):
        model = built[name]
        assert len(model.source_encoder.attention_blocks) == 4, name
        assert len(model.target_encoder.attention_blocks) == 4, name
        assert model.source_encoder.attention_window is None, name
        assert model.source_encoder.receptive_field is None, name


def test_the_a2_arm_keeps_the_shipped_stem(built):
    """The other side of the A1/A2 pair: A2 is A1 plus the stem and nothing else."""
    model = built["sweep_arch_a2.yaml"]

    assert len(model.target_encoder.conv_blocks) == 2
    assert len(model.source_encoder.conv_blocks) == 2
    assert model.source_encoder.conv_reach == 21


def test_a2_minus_a1_is_exactly_the_stem_cost(params):
    """The comparison the pair exists to make, in parameters. Two gated causal depthwise
    convolution blocks per encoder: $50{,}176$ at kernel $5$ plus $50{,}688$ at kernel $9$, twice."""
    assert params["sweep_arch_a2.yaml"] - params["sweep_arch_a1.yaml"] == STEM_PARAMS
    assert STEM_PARAMS == 201_728


def test_a1_against_the_shipped_configuration_is_not_the_stem_cost(params):
    """A1 also gives the source encoder a fourth attention block, so its delta against the shipped
    arm is $-201{,}728 + 164{,}352$. Stated because the naive reading -- A1 is the default without
    its stem -- is wrong and would misattribute a parameter-matched comparison."""
    delta = params["sweep_arch_a1.yaml"] - params["default.yaml"]

    assert delta == -STEM_PARAMS + ATTENTION_BLOCK_PARAMS
    assert delta == -37_376


def test_a2_against_the_shipped_configuration_is_one_attention_block(params):
    assert params["sweep_arch_a2.yaml"] - params["default.yaml"] == ATTENTION_BLOCK_PARAMS


# --------------------------------------------------------------------------------------
# Phase 2: source locality, and the reach budget
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EXPECTED_SOURCE_BOUND))
def test_a_window_arm_reports_the_bound_its_header_states(name, built):
    r"""$R_U = \min(R_{\mathrm{conv}} + N_U (W_U - 1),\, T)$, measured on the constructed encoder.
    The unbounded arm reports ``None`` rather than $T$: the bound is *absent*, and a caller reading
    a number could not tell the two apart."""
    assert built[name].source_encoder.receptive_field == EXPECTED_SOURCE_BOUND[name]


def test_a_window_arm_changes_no_parameter_count(params):
    """The window is a mask, not a width. Asserted so a window arm's result cannot be explained by
    capacity, which is the confound this sweep would otherwise carry."""
    for name in EXPECTED_SOURCE_BOUND:
        assert params[name] == params["default.yaml"], name


def test_the_window_sweep_brackets_the_lag_search_range(built):
    """The point of the sweep: the shipped $66$-step bound sits inside the $90$-step lag search
    range and the wider arms sit outside it, so the sweep spans the regime change rather than
    sampling one side of it."""
    lag_search_steps = built["default.yaml"].max_lag

    assert built["default.yaml"].source_encoder.receptive_field < lag_search_steps
    assert built["sweep_window_8.yaml"].source_encoder.receptive_field < lag_search_steps
    assert built["sweep_window_32.yaml"].source_encoder.receptive_field > lag_search_steps
    assert built["sweep_window_64.yaml"].source_encoder.receptive_field > lag_search_steps


def test_the_reach_arm_resolves_at_the_costed_channel_counts():
    """The resolution is the go/no-go: it raises on a budget that keeps no channel or whose worst
    delay outruns the warm-up. The counts are pinned on top so the arm's channel accounting is
    fixed before any GPU time is spent."""
    vae = _resolved("sweep_reach_120.yaml")["model_config"]["VAE_model"]

    budget = resolve_stream_budgets(vae)

    assert (len(budget.target_keep_index), len(budget.source_keep_index)) == REACH_120_CHANNELS


def test_the_reach_arms_worst_delay_fits_inside_the_warmup():
    """The first ``max_delay`` steps of a delayed stream are partly zero-filled, so they must fall
    inside the steps the loss already discards. The comparison is strictly greater-than, and this
    budget's worst delay is *exactly* the shipped warm-up -- the deepest admissible one, and
    therefore the hardest case for the availability representation."""
    vae = _resolved("sweep_reach_120.yaml")["model_config"]["VAE_model"]

    budget = resolve_stream_budgets(vae)

    assert budget.max_delay == vae["warmup_period"] == 30


def test_the_reach_arm_constructs_both_availability_parameters(built):
    """$W_m$ and $e_{\\mathrm{start}}$ are what make a zero-filled prefix a representation rather
    than a numerical accident, and they exist only under a finite budget. Both directions: present
    here, absent on the unguarded baseline."""
    guarded = built["sweep_reach_120.yaml"]
    unguarded = built["default.yaml"]

    for adapter in (guarded.target_adapter, guarded.source_adapter):
        assert adapter.mask_proj is not None
        assert adapter.start_embed is not None
    for adapter in (unguarded.target_adapter, unguarded.source_adapter):
        assert adapter.mask_proj is None
        assert adapter.start_embed is None


def test_the_reach_arm_narrows_the_adapters_to_the_surviving_channels(built):
    """The budget is real only if it reached the widths. A budget that resolved and was then
    dropped by the signature sweep would leave the adapters at the declared $109$ and $58$."""
    model = built["sweep_reach_120.yaml"]

    widths = (model.target_adapter.in_dim, model.source_adapter.in_dim)
    assert widths == REACH_120_CHANNELS
    assert model.source_delay_steps > 0


# --------------------------------------------------------------------------------------
# Phase 3: depth and width
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,blocks",
    [
        ("sweep_target_blocks_3.yaml", -1),
        ("sweep_target_blocks_5.yaml", 1),
        ("sweep_source_blocks_2.yaml", -1),
        ("sweep_source_blocks_4.yaml", 1),
    ],
)
def test_a_depth_arm_moves_the_total_by_exactly_one_attention_block(name, blocks, params):
    r"""$4d^2 + 3 d\,d_{\mathrm{ff}} + 4d = 164{,}352$ per block, whichever stream it is added to."""
    assert params[name] - params["default.yaml"] == blocks * ATTENTION_BLOCK_PARAMS
    assert ATTENTION_BLOCK_PARAMS == 164_352


def test_a_depth_arm_moves_one_stream_and_leaves_the_other_alone(built):
    """The arms are single-axis, checked on the constructed encoders rather than on the configs."""
    default = built["default.yaml"]

    for name, target, source in (
        ("sweep_target_blocks_3.yaml", 3, 3),
        ("sweep_target_blocks_5.yaml", 5, 3),
        ("sweep_source_blocks_2.yaml", 4, 2),
        ("sweep_source_blocks_4.yaml", 4, 4),
    ):
        model = built[name]
        assert len(model.target_encoder.attention_blocks) == target, name
        assert len(model.source_encoder.attention_blocks) == source, name
        # The window is the other axis and is held at the shipped value in every depth arm.
        assert model.source_encoder.attention_window == default.source_encoder.attention_window


def test_the_source_depth_arms_stay_inside_the_lag_search_range(built):
    """The locality property the architecture rests on: at the shipped $16$-step window, four
    source blocks is the deepest encoder whose bound still falls short of the $90$-step lag search
    range, so neither depth arm changes what the lag attention is for."""
    lag_search_steps = built["default.yaml"].max_lag

    for name, bound in (("sweep_source_blocks_2.yaml", 51), ("sweep_source_blocks_4.yaml", 81)):
        assert built[name].source_encoder.receptive_field == bound, name
        assert bound < lag_search_steps


def test_the_width_arm_moves_the_total_by_the_feed_forward_cost_alone(params, built):
    r"""$3 d \,\Delta d_{\mathrm{ff}}$ per block across the seven attention blocks the shipped
    architecture holds -- $344{,}064$ -- and nothing else: the attention projections, the stem and
    every downstream component are untouched."""
    assert params["sweep_ff_384.yaml"] - params["default.yaml"] == FF_384_PARAMS
    assert FF_384_PARAMS == 344_064

    model = built["sweep_ff_384.yaml"]
    blocks = len(model.target_encoder.attention_blocks) + len(model.source_encoder.attention_blocks)
    assert blocks == 7
    assert model.target_encoder.d_ff == 384


def test_the_model_width_is_the_same_in_every_arm(built):
    r"""$d = 128$ is held fixed across the whole study: it is the width the prior head, the
    posterior fusion, the lag attention's key-value projections and the decoder input all assume,
    and the derived encoder head width must stay even for the rotary position encoding."""
    for name, model in built.items():
        assert model.d_model == 128, name
        assert (model.d_model // model.target_encoder.num_heads) % 2 == 0, name
