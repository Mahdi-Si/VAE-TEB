r"""Lint for the sweep arms: each is ``default.yaml`` plus a *declared* delta and nothing else.

Sixteen arms sweep four axes. **Architecture** (A1, A2) asks what the convolution stem and the
source encoder's asymmetry are worth; the shipped ``default.yaml`` is A3, the recommended arm, and
A0 is the model this package is compared against -- ``teb_vae/lag_attn_rws`` -- not a file here.
**Source locality** sweeps $W_U \in \{8, 32, 64, \text{unbounded}\}$ around the shipped $16$, plus
the causal-input reach budget the availability representation exists to make trainable.
**Depth and width** moves $N_Y$, $N_U$ and $d_{\mathrm{ff}}$ one key at a time, re-centred on the
raised baseline: the target-depth arms are $4$ and $8$ around the shipped $6$, and the width arm is
the lower bracket $384$ below the shipped $512$. Both lower brackets are the values this package
shipped before the capacity revision, so each asks whether its raise was needed rather than merely
probing downwards. **The prior-anchor
weight** brackets ``beta_prior`` over three orders of magnitude around the shipped $10^{-1}$; its
$10^{-1}$ arm deliberately restates the shipped value -- pinned, so the arm survives a later
revision of the default -- and therefore declares an *empty* delta.

Six further arms are ablate-one reverts rather than sweeps: four for the bottleneck bundle, and two
for the decoder-side pair the capacity revision added -- the auxiliary shape terms in the criterion
(``sweep_aux_off.yaml``) and the self-attention over the horizon tokens
(``sweep_horizon_attn_off.yaml``). Each sets its mechanism back to the value that shipped before it
existed, so what the pair cost is read off two runs rather than inferred from one.

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
from teb_vae.lag_attn.channel_reach import resolve_stream_budgets
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
_BASE_DECODE = f"{_VAE}.base_decode"
_LOGVAR_MODE = f"{_VAE}.posterior_logvar_mode"
_SOURCE_DROPOUT = f"{_VAE}.source_dropout"
_BETA_PRIOR = f"{_VAE}.beta_prior"
_LAMBDA_MS = f"{_VAE}.lambda_ms"
_LAMBDA_DERIV = f"{_VAE}.lambda_deriv"
_LAMBDA_BOUNDARY = f"{_VAE}.lambda_boundary"
_HORIZON_ATTENTION = f"{_VAE}.horizon_attention_blocks"

#: Every arm, by file name, with the complete set of leaves it is allowed to move against
#: ``default.yaml``. An arm's resolved delta must equal its entry exactly -- neither a missing key
#: (a declaration that stopped being true) nor an extra one (a second change riding along) passes.
DECLARED_DELTAS: Dict[str, Dict[str, Any]] = {
    # Phase 1: architecture. A1 removes the stem *and* symmetrises the streams, so the only
    # difference between it and A2 is the stem itself; A2 symmetrises alone.
    "sweep_arch_a1.yaml": {
        _CONV_KERNELS: [],
        _CONV_DILATIONS: [],
        _SOURCE_BLOCKS: 6,
        _SOURCE_WINDOW: None,
    },
    "sweep_arch_a2.yaml": {_SOURCE_BLOCKS: 6, _SOURCE_WINDOW: None},
    # Phase 2: source locality, and the reach budget.
    "sweep_window_8.yaml": {_SOURCE_WINDOW: 8},
    "sweep_window_32.yaml": {_SOURCE_WINDOW: 32},
    "sweep_window_64.yaml": {_SOURCE_WINDOW: 64},
    "sweep_window_full.yaml": {_SOURCE_WINDOW: None},
    "sweep_reach_null.yaml": {_REACH: None},
    # The three ablate-one arms for the bottleneck bundle. Each reverts exactly one shipped key,
    # so the default and the arm differ in one mechanism and the attribution is not a matter of
    # reading two changes at once. The two source-dropout arms are not reverts: the default ships
    # the mechanism at the global rate, and these measure what raising it buys.
    "sweep_base_sample.yaml": {_BASE_DECODE: "sample"},
    "sweep_logvar_residual.yaml": {_LOGVAR_MODE: "residual"},
    "sweep_source_dropout_0p2.yaml": {_SOURCE_DROPOUT: 0.2},
    "sweep_source_dropout_0p3.yaml": {_SOURCE_DROPOUT: 0.3},
    # Phase 3: depth and width. The two target-depth arms bracket the shipped 6 by two blocks in
    # each direction, and the lower one is the value this package shipped before the capacity
    # revision -- so it asks whether the raise was needed rather than merely probing downwards.
    "sweep_target_blocks_4.yaml": {_TARGET_BLOCKS: 4},
    "sweep_target_blocks_8.yaml": {_TARGET_BLOCKS: 8},
    "sweep_source_blocks_2.yaml": {_SOURCE_BLOCKS: 2},
    "sweep_source_blocks_4.yaml": {_SOURCE_BLOCKS: 4},
    "sweep_ff_384.yaml": {_D_FF: 384},
    # The prior-anchor weight, bracketing the shipped 0.1 over three orders of magnitude. The
    # 0p1 arm restates the shipped value -- pinned against a later default revision -- so its
    # resolved delta against default.yaml is empty by construction. That pin is exactly what made
    # the revision from 1.0e-2 to 0.1 a two-line change here rather than a silent one: the empty
    # delta moved from one arm to the other and this table had to say so.
    "sweep_beta_prior_0p001.yaml": {_BETA_PRIOR: 1.0e-3},
    "sweep_beta_prior_0p01.yaml": {_BETA_PRIOR: 1.0e-2},
    "sweep_beta_prior_0p1.yaml": {},
    "sweep_beta_prior_1p0.yaml": {_BETA_PRIOR: 1.0},
    # The two ablate-one arms for the decoder-side bundle: the shape terms in the criterion, and
    # the self-attention over the horizon tokens in the shared decoder core. Each reverts its
    # mechanism to the value that shipped before it existed, so the pair prices the two halves of
    # the revision separately. The aux arm is declared multi-key on purpose -- the three weights
    # were shipped together and price one decision, and turning off one of three would answer a
    # question nobody has asked yet.
    "sweep_aux_off.yaml": {_LAMBDA_MS: 0.0, _LAMBDA_DERIV: 0.0, _LAMBDA_BOUNDARY: 0.0},
    "sweep_horizon_attn_off.yaml": {_HORIZON_ATTENTION: 0},
}

#: One causal Transformer block at the shipped widths: $4d^2 + 3 d\,d_{\mathrm{ff}} + 4d$ with
#: $d = 128$, $d_{\mathrm{ff}} = 512$. Every depth arm moves the total by a multiple of this.
ATTENTION_BLOCK_PARAMS = 4 * 128**2 + 3 * 128 * 512 + 4 * 128

#: One horizon self-attention block at the shipped decoder width: four bias-free
#: $d_{\mathrm{hidden}} \times d_{\mathrm{hidden}}$ projections, one ``LayerNorm`` and one scalar
#: residual gain, at $d_{\mathrm{hidden}} = 256$. The decoder is shared, so this cost is the same
#: in every package of the family that ships the blocks.
HORIZON_ATTENTION_BLOCK_PARAMS = 4 * 256**2 + 2 * 256 + 1

#: Both stems together: two gated causal depthwise convolution blocks per encoder, at kernels
#: $5$ and $9$. This is the A1-to-A2 difference, and the only one.
STEM_PARAMS = 2 * (50_176 + 50_688)

#: Going $d_{\mathrm{ff}} = 512 \to 384$ across the nine attention blocks the shipped
#: architecture holds (six target, three source), at $3 d$ parameters per unit of width. Negative:
#: the shipped width is now the larger one, so this arm is the lower bracket rather than an
#: upgrade.
FF_384_PARAMS = 3 * 128 * (384 - 512) * 9

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
    assert resolved_delta(arm_flat, default_flat) != {_D_FF: 256}


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
    """Six blocks, full causal prefix, on both streams. The source bound is *absent* under A1 and
    A2, which is what makes the stem the only difference between them.

    Read off the *shipped* target depth rather than against a literal: the arms' claim is symmetry,
    so a revision of the default's target depth that left these files behind would silently turn
    them into depth arms and this comparison would stop being about the stem."""
    shipped_depth = len(built["default.yaml"].target_encoder.attention_blocks)

    assert shipped_depth == 6
    for name in ("sweep_arch_a1.yaml", "sweep_arch_a2.yaml"):
        model = built[name]
        assert len(model.source_encoder.attention_blocks) == shipped_depth, name
        assert len(model.target_encoder.attention_blocks) == shipped_depth, name
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
    """A1 also raises the source encoder from three blocks to six, so its delta against the shipped
    arm is $-201{,}728 + 3 \\cdot 262{,}656$ -- positive. Stated because the naive reading -- A1 is
    the default without its stem -- is wrong and would misattribute a parameter-matched
    comparison."""
    delta = params["sweep_arch_a1.yaml"] - params["default.yaml"]

    assert delta == -STEM_PARAMS + 3 * ATTENTION_BLOCK_PARAMS
    assert delta == 586_240


def test_a2_against_the_shipped_configuration_is_three_attention_blocks(params):
    """The source encoder goes from the shipped three blocks to the target's six."""
    assert params["sweep_arch_a2.yaml"] - params["default.yaml"] == 3 * ATTENTION_BLOCK_PARAMS


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


def test_the_shipped_reach_budget_resolves_at_the_costed_channel_counts():
    """The resolution is the go/no-go: it raises on a budget that keeps no channel or whose worst
    delay outruns the warm-up. Read off ``default.yaml`` because the guard is now the shipped
    configuration, not an arm. The counts are pinned on top so the channel accounting is fixed
    before any GPU time is spent."""
    vae = _resolved("default.yaml")["model_config"]["VAE_model"]

    budget = resolve_stream_budgets(vae)

    assert (len(budget.target_keep_index), len(budget.source_keep_index)) == REACH_120_CHANNELS


def test_the_reach_arms_worst_delay_fits_inside_the_warmup():
    """The first ``max_delay`` steps of a delayed stream are partly zero-filled, so they must fall
    inside the steps the loss already discards. The comparison is strictly greater-than, and this
    budget's worst delay is *exactly* the shipped warm-up -- the deepest admissible one, and
    therefore the hardest case for the availability representation."""
    vae = _resolved("default.yaml")["model_config"]["VAE_model"]

    budget = resolve_stream_budgets(vae)

    assert budget.max_delay == vae["warmup_period"] == 30


def test_the_shipped_config_constructs_both_availability_parameters(built):
    """$W_m$ and $e_{\\mathrm{start}}$ are what make a zero-filled prefix a representation rather
    than a numerical accident, and they exist only under a finite budget. Both directions: present
    on the shipped guarded config, absent on the ablate-one arm that turns the guard off."""
    guarded = built["default.yaml"]
    unguarded = built["sweep_reach_null.yaml"]

    for adapter in (guarded.target_adapter, guarded.source_adapter):
        assert adapter.mask_proj is not None
        assert adapter.start_embed is not None
    for adapter in (unguarded.target_adapter, unguarded.source_adapter):
        assert adapter.mask_proj is None
        assert adapter.start_embed is None


def test_the_shipped_config_narrows_the_adapters_to_the_surviving_channels(built):
    """The budget is real only if it reached the widths. A budget that resolved and was then
    dropped by the signature sweep would leave the adapters at the declared $109$ and $58$."""
    model = built["default.yaml"]

    widths = (model.target_adapter.in_dim, model.source_adapter.in_dim)
    assert widths == REACH_120_CHANNELS
    assert model.source_delay_steps > 0

    # The ablate-one arm is the other direction: no gate, no delay, declared widths.
    unguarded = built["sweep_reach_null.yaml"]
    assert (unguarded.target_adapter.in_dim, unguarded.source_adapter.in_dim) == (109, 58)
    assert unguarded.source_delay_steps == 0


# --------------------------------------------------------------------------------------
# Phase 3: depth and width
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,blocks",
    [
        ("sweep_target_blocks_4.yaml", -2),
        ("sweep_target_blocks_8.yaml", 2),
        ("sweep_source_blocks_2.yaml", -1),
        ("sweep_source_blocks_4.yaml", 1),
    ],
)
def test_a_depth_arm_moves_the_total_by_whole_attention_blocks(name, blocks, params):
    r"""$4d^2 + 3 d\,d_{\mathrm{ff}} + 4d = 262{,}656$ per block, whichever stream it is added to.
    The target arms bracket the shipped depth by two blocks each way, the source arms by one."""
    assert params[name] - params["default.yaml"] == blocks * ATTENTION_BLOCK_PARAMS
    assert ATTENTION_BLOCK_PARAMS == 262_656


def test_a_depth_arm_moves_one_stream_and_leaves_the_other_alone(built):
    """The arms are single-axis, checked on the constructed encoders rather than on the configs."""
    default = built["default.yaml"]

    for name, target, source in (
        ("sweep_target_blocks_4.yaml", 4, 3),
        ("sweep_target_blocks_8.yaml", 8, 3),
        ("sweep_source_blocks_2.yaml", 6, 2),
        ("sweep_source_blocks_4.yaml", 6, 4),
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
    r"""$3 d \,\Delta d_{\mathrm{ff}}$ per block across the nine attention blocks the shipped
    architecture holds -- $-442{,}368$ -- and nothing else: the attention projections, the stem and
    every downstream component are untouched. Negative, because $384$ is now the lower bracket."""
    assert params["sweep_ff_384.yaml"] - params["default.yaml"] == FF_384_PARAMS
    assert FF_384_PARAMS == -442_368

    model = built["sweep_ff_384.yaml"]
    blocks = len(model.target_encoder.attention_blocks) + len(model.source_encoder.attention_blocks)
    assert blocks == 9
    assert model.target_encoder.d_ff == 384


# --------------------------------------------------------------------------------------
# The decoder-side bundle: the shape terms and the horizon attention
# --------------------------------------------------------------------------------------
def test_the_aux_arm_moves_only_the_criterion(params, built):
    """The three shape terms hold no parameters, so this arm shares a checkpoint geometry with the
    default and its result cannot be explained by capacity. Asserted on both sides -- the totals
    are equal *and* the decoder core is the same object shape -- because the whole point of the
    arm is that the only difference is what the criterion prices."""
    assert params["sweep_aux_off.yaml"] == params["default.yaml"]

    arm = built["sweep_aux_off.yaml"]
    default = built["default.yaml"]
    assert arm.horizon_core.attention_blocks == default.horizon_core.attention_blocks


def test_the_horizon_attention_arm_removes_exactly_the_two_decoder_blocks(params, built):
    r"""$4 d_{\mathrm{hidden}}^2 + 2 d_{\mathrm{hidden}} + 1 = 262{,}657$ per block at the shipped
    $256$, twice. The arm builds **no** attention module rather than an inert one, which is what
    makes the reverted decoder parameter-for-parameter the core as it was before the blocks
    existed."""
    delta = params["sweep_horizon_attn_off.yaml"] - params["default.yaml"]

    assert delta == -2 * HORIZON_ATTENTION_BLOCK_PARAMS
    assert delta == -525_314

    arm = built["sweep_horizon_attn_off.yaml"]
    assert arm.horizon_core.attention_blocks == 0
    assert arm.horizon_core.attention is None
    assert built["default.yaml"].horizon_core.attention_blocks == 2


def test_the_model_width_is_the_same_in_every_arm(built):
    r"""$d = 128$ is held fixed across the whole study: it is the width the prior head, the
    posterior fusion, the lag attention's key-value projections and the decoder input all assume,
    and the derived encoder head width must stay even for the rotary position encoding."""
    for name, model in built.items():
        assert model.d_model == 128, name
        assert (model.d_model // model.target_encoder.num_heads) % 2 == 0, name
