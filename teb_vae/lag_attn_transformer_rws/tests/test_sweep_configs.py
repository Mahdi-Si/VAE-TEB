r"""Lint for the sweep arms: each is ``default.yaml`` plus a *declared* set of moved leaves.

The arms sweep four axes -- the encoder architecture (A1, A2; the shipped ``default.yaml`` is A3,
and A0 is ``teb_vae/lag_attn_rws``), source locality and the causal-input reach budget, depth and
width, and the prior-anchor weight -- plus ablate-one reverts of the bottleneck bundle and of the
decoder-side pair (the auxiliary shape terms and the horizon self-attention).

The lint holds :data:`DECLARED_DELTAS`, and it is the point of the file. Every
``configs/sweep_*.yaml`` must appear in it, and each arm's *resolved* delta against ``default.yaml``
must move exactly its declared leaves -- so a multi-key arm is a declaration rather than an
exception, and an arm that quietly acquired a second change stops being an answer to its own
question. The values themselves live in the arm files and are not restated here.

Every arm is also built through the real driver at production geometry, and what it costs is
checked *relationally* on the constructed models: window and shape-term arms leave the parameter
count unchanged, depth arms move it by whole measured attention blocks, the width arm by the
feed-forward cost alone, and the stem and horizon-attention arms by exactly the modules they
remove. So a malformed arm -- a key that does not resolve, a stray second delta, a parameter cost
that is not what the arm is for, a reach budget the filter bank refuses -- is caught on the
development box rather than days into a production run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping

import pytest
import yaml

from teb_vae.lag_attn.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT = _CONFIG_DIR / "default.yaml"

_VAE = "model_config.VAE_model"

#: Every arm, by file name, with the complete set of leaves it is allowed to move against
#: ``default.yaml``. An arm's resolved delta must move exactly these -- neither a missing key (a
#: declaration that stopped being true) nor an extra one (a second change riding along) passes.
DECLARED_DELTAS: Dict[str, FrozenSet[str]] = {
    name: frozenset(f"{_VAE}.{key}" for key in keys)
    for name, keys in {
        # Architecture. A1 removes the stem *and* symmetrises the streams, so the only difference
        # between it and A2 is the stem itself; A2 symmetrises alone.
        "sweep_arch_a1.yaml": (
            "encoder_conv_kernels",
            "encoder_conv_dilations",
            "source_attention_blocks",
            "source_attention_window",
        ),
        "sweep_arch_a2.yaml": ("source_attention_blocks", "source_attention_window"),
        # Source locality, and the reach budget.
        "sweep_window_8.yaml": ("source_attention_window",),
        "sweep_window_32.yaml": ("source_attention_window",),
        "sweep_window_64.yaml": ("source_attention_window",),
        "sweep_window_full.yaml": ("source_attention_window",),
        "sweep_reach_null.yaml": ("causal_reach_budget_s",),
        # The ablate-one arms for the bottleneck bundle, and the two source-dropout rates.
        "sweep_base_sample.yaml": ("base_decode",),
        "sweep_logvar_residual.yaml": ("posterior_logvar_mode",),
        "sweep_source_dropout_0p2.yaml": ("source_dropout",),
        "sweep_source_dropout_0p3.yaml": ("source_dropout",),
        # Depth and width.
        "sweep_target_blocks_4.yaml": ("target_attention_blocks",),
        "sweep_target_blocks_8.yaml": ("target_attention_blocks",),
        "sweep_source_blocks_2.yaml": ("source_attention_blocks",),
        "sweep_source_blocks_4.yaml": ("source_attention_blocks",),
        "sweep_ff_384.yaml": ("encoder_d_ff",),
        # The prior-anchor weight. One arm restates the shipped value -- pinned against a later
        # default revision -- so its resolved delta against default.yaml is empty by construction.
        "sweep_beta_prior_0p001.yaml": ("beta_prior",),
        "sweep_beta_prior_0p01.yaml": ("beta_prior",),
        "sweep_beta_prior_0p1.yaml": (),
        "sweep_beta_prior_1p0.yaml": ("beta_prior",),
        # The decoder-side ablate-one arms. The aux arm is multi-key on purpose: the three shape
        # weights were shipped together and price one decision.
        "sweep_aux_off.yaml": ("lambda_ms", "lambda_deriv", "lambda_boundary"),
        "sweep_horizon_attn_off.yaml": ("horizon_attention_blocks",),
    }.items()
}

#: The arms that move only the source attention window.
_WINDOW_ARMS = (
    "sweep_window_8.yaml",
    "sweep_window_32.yaml",
    "sweep_window_64.yaml",
    "sweep_window_full.yaml",
)

#: Config paths that identify a run. Arms share the baseline's: a per-arm identity would be a
#: second delta, and a run is identified by its ``TEB_RUN_STAMP`` directory and the resolved
#: config written beside its checkpoints.
_RUN_IDENTITY_PATHS = (
    "general_config.tag",
    "advanced_config.tracking.mlflow.experiment_name",
    "advanced_config.tracking.mlflow.run_name",
    "advanced_config.tracking.mlflow.tags.variant",
)


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


def _n_params(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.fixture(scope="module")
def default_flat() -> Dict[str, Any]:
    return _flatten(load_config(str(_DEFAULT)))


@pytest.fixture(scope="module")
def built() -> Dict[str, SeqVaeLagAttnTrfRws]:
    """Every arm and the default, constructed once at production geometry through the driver.

    Construction is what proves a config resolves to a model rather than to a set of keys, and at
    these widths it costs a few tens of milliseconds per arm, so it is done once for the module.
    """
    names = ["default.yaml", *DECLARED_DELTAS]
    return {name: SeqVaeLagAttnTrfRws(**_model_kwargs(_resolved(name))) for name in names}


@pytest.fixture(scope="module")
def params(built) -> Dict[str, int]:
    """Measured parameter total per arm."""
    return {name: _n_params(model) for name, model in built.items()}


# --------------------------------------------------------------------------------------
# The inventory is closed and every arm is its declared delta
# --------------------------------------------------------------------------------------
def test_the_configs_directory_holds_exactly_the_declared_arms():
    """Both directions: a declared arm whose file is missing, and a stray file nobody declared --
    which would run outside every assertion below."""
    present = {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")}

    assert present == set(DECLARED_DELTAS)


@pytest.mark.parametrize("name", sorted(DECLARED_DELTAS))
def test_an_arm_is_the_default_plus_exactly_its_declared_leaves(name, default_flat, built):
    """``load_config`` must eat the ``base:`` directive; the key *sets* must match, because a typo'd
    override adds a path rather than moving one; the moved leaves must be exactly the declared
    ones; the run identity is the baseline's; and the arm builds through the real driver at the
    width every component downstream of the encoders assumes."""
    resolved = _resolved(name)
    arm_flat = _flatten(resolved)

    assert "base" not in resolved
    assert set(arm_flat) == set(default_flat), (
        f"{name} adds or drops config paths rather than overriding one"
    )
    moved = {path for path in arm_flat if arm_flat[path] != default_flat[path]}
    assert moved == DECLARED_DELTAS[name]
    for path in _RUN_IDENTITY_PATHS:
        assert arm_flat[path] == default_flat[path], path
    assert built[name].d_model == built["default.yaml"].d_model


# --------------------------------------------------------------------------------------
# The architecture arms
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
    """Full depth and full causal prefix on both streams. The source bound is *absent* under A1 and
    A2, which is what makes the stem the only difference between them.

    Read off the *shipped* target depth rather than against a literal: the arms' claim is symmetry,
    so a revision of the default's target depth that left these files behind would silently turn
    them into depth arms and this comparison would stop being about the stem."""
    shipped_depth = len(built["default.yaml"].target_encoder.attention_blocks)

    for name in ("sweep_arch_a1.yaml", "sweep_arch_a2.yaml"):
        model = built[name]
        assert len(model.source_encoder.attention_blocks) == shipped_depth, name
        assert len(model.target_encoder.attention_blocks) == shipped_depth, name
        assert model.source_encoder.attention_window is None, name
        assert model.source_encoder.receptive_field is None, name


def test_a2_minus_a1_is_exactly_the_stem(built, params):
    """The comparison the pair exists to make, in parameters: the two gated causal depthwise
    convolution blocks per encoder, measured on A2's own stems."""
    a2 = built["sweep_arch_a2.yaml"]
    stem = _n_params(a2.target_encoder.conv_blocks) + _n_params(a2.source_encoder.conv_blocks)

    assert stem > 0
    assert params["sweep_arch_a2.yaml"] - params["sweep_arch_a1.yaml"] == stem


# --------------------------------------------------------------------------------------
# Source locality, and the reach budget
# --------------------------------------------------------------------------------------
def test_a_window_arm_changes_no_parameter_count(params):
    """The window is a mask, not a width. Asserted so a window arm's result cannot be explained by
    capacity, which is the confound this sweep would otherwise carry."""
    for name in _WINDOW_ARMS:
        assert params[name] == params["default.yaml"], name


def test_the_window_sweep_brackets_the_lag_search_range(built):
    """The point of the sweep: the shipped bound sits inside the lag search range and the wider
    arms sit outside it, so the sweep spans the regime change rather than sampling one side of it.
    The unbounded arm reports ``None`` rather than $T$: the bound is *absent*."""
    lag_search_steps = built["default.yaml"].max_lag

    assert built["default.yaml"].source_encoder.receptive_field < lag_search_steps
    assert built["sweep_window_8.yaml"].source_encoder.receptive_field < lag_search_steps
    assert built["sweep_window_32.yaml"].source_encoder.receptive_field > lag_search_steps
    assert built["sweep_window_64.yaml"].source_encoder.receptive_field > lag_search_steps
    assert built["sweep_window_full.yaml"].source_encoder.receptive_field is None


def test_the_shipped_reach_budget_narrows_the_adapters_to_the_surviving_channels(built):
    """The resolution is the go/no-go: it raises on a budget that keeps no channel or whose worst
    delay outruns the warm-up. And the budget is real only if it reached the widths: one that
    resolved and was then dropped by the signature sweep would leave the adapters at the declared
    $c_y$ and $c_u$. The ablate-one arm is the other direction: no gate, no delay, declared
    widths."""
    vae = _resolved("default.yaml")["model_config"]["VAE_model"]
    budget = resolve_stream_budgets(vae)
    guarded = built["default.yaml"]
    unguarded = built["sweep_reach_null.yaml"]

    kept = (len(budget.target_keep_index), len(budget.source_keep_index))
    assert (guarded.target_adapter.in_dim, guarded.source_adapter.in_dim) == kept
    assert kept[0] < vae["c_y"] and kept[1] < vae["c_u"]
    assert guarded.source_delay_steps > 0

    assert (unguarded.target_adapter.in_dim, unguarded.source_adapter.in_dim) == (
        vae["c_y"],
        vae["c_u"],
    )
    assert unguarded.source_delay_steps == 0


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


# --------------------------------------------------------------------------------------
# Depth and width
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "sweep_target_blocks_4.yaml",
        "sweep_target_blocks_8.yaml",
        "sweep_source_blocks_2.yaml",
        "sweep_source_blocks_4.yaml",
        "sweep_arch_a2.yaml",
    ],
)
def test_a_depth_arm_moves_the_total_by_whole_attention_blocks(name, built, params):
    r"""$4d^2 + 3 d\,d_{\mathrm{ff}} + 4d$ per block, measured on the default's own blocks, times
    the change in block count on each stream -- and nothing else moves."""
    default, arm = built["default.yaml"], built[name]
    expected = 0
    for stream in ("target_encoder", "source_encoder"):
        shipped_blocks = getattr(default, stream).attention_blocks
        change = len(getattr(arm, stream).attention_blocks) - len(shipped_blocks)
        expected += change * _n_params(shipped_blocks[0])

    assert expected != 0
    assert params[name] - params["default.yaml"] == expected


def test_the_source_depth_arms_stay_inside_the_lag_search_range(built):
    """The locality property the architecture rests on: at the shipped window, neither source depth
    arm reaches past the lag search range, so neither changes what the lag attention is for."""
    lag_search_steps = built["default.yaml"].max_lag

    for name in ("sweep_source_blocks_2.yaml", "sweep_source_blocks_4.yaml"):
        assert built[name].source_encoder.receptive_field < lag_search_steps, name


def test_the_width_arm_moves_the_total_by_the_feed_forward_cost_alone(params, built):
    r"""$3 d \,\Delta d_{\mathrm{ff}}$ per block across every attention block of both encoders, and
    nothing else: the attention projections, the stem and every downstream component are
    untouched."""
    default, arm = built["default.yaml"], built["sweep_ff_384.yaml"]
    blocks = len(default.target_encoder.attention_blocks) + len(
        default.source_encoder.attention_blocks
    )
    delta_ff = arm.target_encoder.d_ff - default.target_encoder.d_ff

    assert delta_ff != 0
    assert arm.source_encoder.d_ff - default.source_encoder.d_ff == delta_ff
    assert params["sweep_ff_384.yaml"] - params["default.yaml"] == (
        3 * default.d_model * delta_ff * blocks
    )


# --------------------------------------------------------------------------------------
# The decoder-side bundle: the shape terms and the horizon attention
# --------------------------------------------------------------------------------------
def test_the_aux_arm_moves_only_the_criterion(params, built):
    """The three shape terms hold no parameters, so this arm shares a checkpoint geometry with the
    default and its result cannot be explained by capacity."""
    assert params["sweep_aux_off.yaml"] == params["default.yaml"]

    arm = built["sweep_aux_off.yaml"]
    default = built["default.yaml"]
    assert arm.horizon_core.attention_blocks == default.horizon_core.attention_blocks


def test_the_horizon_attention_arm_removes_exactly_the_decoder_attention(params, built):
    """The arm builds **no** attention module rather than an inert one, which is what makes the
    reverted decoder parameter-for-parameter the core as it was before the blocks existed."""
    default = built["default.yaml"]
    arm = built["sweep_horizon_attn_off.yaml"]
    removed = _n_params(default.horizon_core.attention)

    assert removed > 0
    assert params["sweep_horizon_attn_off.yaml"] - params["default.yaml"] == -removed
    assert arm.horizon_core.attention_blocks == 0
    assert arm.horizon_core.attention is None
