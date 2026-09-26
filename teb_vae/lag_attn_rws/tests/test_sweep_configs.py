r"""Lint for the calibration sweep arms, and the study's numeric collapse criterion.

Three one-variable calibration sweeps and five architecture A/B arms exercise the shipped
configuration. The calibration sweeps vary the converged KL weight ``beta_schedule.end`` over
$\{0.1, 0.3, 1.0, 3.0\}$, the latent width ``d_z`` over $\{24, 32, 48, 64, 96\}$, and the causal
input budget ``causal_reach_budget_s`` over $\{\mathrm{null}, 240, 120, 60, 32\}$ seconds. The
latent ladder brackets the shipped $64$ on both sides: $48$ is the value this package shipped
before the capacity revision, so it asks whether the raise was needed, and $96$ is the upper
bracket the raise opened. The $64$ arm restates the shipped value and therefore declares an empty
delta -- pinned, so the comparison survives a later revision of the default.
The architecture arms each flip one structural knob against the baseline: ``encoder_extra_kernel``
$\to 7$, ``conv_norm_groups`` $\to 1$, ``query_uses_logvar`` $\to$ true, ``horizon_depth``
$\to 5$, and the init-off ablation. Every arm is ``default.yaml`` plus its swept delta and
nothing else, so any pair of runs differs by one resolved key and a metric difference between
them has one explanation. Two arms are deliberate multi-key exceptions: the $240$ s reach arm
must also raise ``warmup_period`` to $60$ (that budget admits channels whose forward reach needs
a delay of up to $\lceil 240 / 4 \rceil = 60$ steps, and the budget resolver refuses a maximum
delay that outruns the loss warm-up), and the init-off arm reverts the whole three-key
initialisation-policy bundle at once. Both exceptions are encoded -- and proven structural
rather than drift -- by the tests below.

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development
box -- a key that does not resolve, a stray second delta, a value other than the declared one,
a budget the filter bank refuses -- rather than days into a production run.

**The collapse criterion.** A *completed* run is **collapsed** when either

1. ``val/source_conditioned_kl_raw`` is below :data:`KL_COLLAPSE_THRESHOLD_NATS` at every one
   of its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs, or
2. its final ``val/kld_active_frac`` is below
   :data:`KL_COLLAPSE_MIN_ACTIVE_DIMS` $/\, d_z$.

The criterion itself lives in :mod:`teb_vae.lag_attn_rws.collapse`, which depends on nothing:
the offline acceptance gate applies the same arithmetic and must stay importable without
``torch``. Its definition and rationale are recorded there; the tests below pin the behaviour
that the sweep reports depend on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn_rws.collapse import (
    KL_COLLAPSE_MIN_ACTIVE_DIMS,
    KL_COLLAPSE_PATIENCE_EPOCHS,
    KL_COLLAPSE_THRESHOLD_NATS,
    is_collapsed,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT = _CONFIG_DIR / "default.yaml"

# --------------------------------------------------------------------------------------
# The arm inventory
# --------------------------------------------------------------------------------------
_VAE = "model_config.VAE_model"
_BETA_END = f"{_VAE}.beta_schedule.end"
_D_Z = f"{_VAE}.d_z"
_REACH = f"{_VAE}.causal_reach_budget_s"
_WARMUP = f"{_VAE}.warmup_period"
# The architecture A/B arms' swept keys.
_ENC_KERNEL = f"{_VAE}.encoder_extra_kernel"
_NORM_GROUPS = f"{_VAE}.conv_norm_groups"
_QUERY_LOGVAR = f"{_VAE}.query_uses_logvar"
_HORIZON_DEPTH = f"{_VAE}.horizon_depth"
_EMBED_STD = f"{_VAE}.horizon_embed_std"
_HEAD_CALIB = f"{_VAE}.head_init_calibration"
_A_HEAD_GAIN = f"{_VAE}.a_head_gain"

#: Every arm: file name -> (swept dotted key, arm value, structurally required extras).
#: The extras entry is non-empty for the two deliberate multi-key arms -- the 240 s reach arm
#: (raised warm-up) and the init-off bundle (the three init-policy keys reverted together); see
#: the module docstring.
_ARMS: Dict[str, Any] = {
    "sweep_beta_0p1.yaml": (_BETA_END, 0.1, {}),
    "sweep_beta_0p3.yaml": (_BETA_END, 0.3, {}),
    "sweep_beta_1p0.yaml": (_BETA_END, 1.0, {}),
    "sweep_beta_3p0.yaml": (_BETA_END, 3.0, {}),
    "sweep_dz_24.yaml": (_D_Z, 24, {}),
    "sweep_dz_32.yaml": (_D_Z, 32, {}),
    "sweep_dz_48.yaml": (_D_Z, 48, {}),
    "sweep_dz_64.yaml": (_D_Z, 64, {}),
    "sweep_dz_96.yaml": (_D_Z, 96, {}),
    "sweep_reach_null.yaml": (_REACH, None, {}),
    "sweep_reach_240.yaml": (_REACH, 240, {_WARMUP: 60}),
    "sweep_reach_120.yaml": (_REACH, 120, {}),
    "sweep_reach_60.yaml": (_REACH, 60, {}),
    "sweep_reach_32.yaml": (_REACH, 32, {}),
    "sweep_enc_kernel_7.yaml": (_ENC_KERNEL, 7, {}),
    "sweep_norm_groups_1.yaml": (_NORM_GROUPS, 1, {}),
    "sweep_query_logvar.yaml": (_QUERY_LOGVAR, True, {}),
    "sweep_horizon_depth_5.yaml": (_HORIZON_DEPTH, 5, {}),
    "sweep_init_off.yaml": (_EMBED_STD, 0.02, {_HEAD_CALIB: False, _A_HEAD_GAIN: 1.0}),
}

def _flatten(node: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a config mapping to ``{dotted path: leaf value}``.

    Dicts recurse; everything else -- scalars, lists, ``None`` -- is a leaf, matching the
    loader's own merge semantics (lists replace wholesale, so a list is a value).
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


@pytest.fixture(scope="module")
def default_flat() -> Dict[str, Any]:
    return _flatten(load_config(str(_DEFAULT)))


# --------------------------------------------------------------------------------------
# The inventory is closed
# --------------------------------------------------------------------------------------
def test_the_sweep_directory_holds_exactly_the_declared_arms():
    """Both directions: a declared arm whose file is missing, and a stray ``sweep_*.yaml``
    nobody declared (which would run outside every assertion below)."""
    present = {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")}

    assert present == set(_ARMS)


# --------------------------------------------------------------------------------------
# Every arm is the default plus exactly its delta
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_differs_from_the_default_in_exactly_its_swept_delta(name, default_flat):
    """The one-variable property itself. Key sets must match exactly (a typo'd override adds
    a path rather than moving one), and the differing values must be exactly the swept key
    plus the arm's declared structural extras -- no second delta, no dropped inheritance.
    An arm restating the shipped value legitimately differs in nothing."""
    swept_key, value, extras = _ARMS[name]
    arm_flat = _flatten(_resolved(name))
    intended = {swept_key: value, **extras}

    assert set(arm_flat) == set(default_flat)
    differing = {path for path in default_flat if arm_flat[path] != default_flat[path]}
    expected = {path for path, val in intended.items() if default_flat[path] != val}
    assert differing == expected

    for path, val in intended.items():
        assert arm_flat[path] == val


@pytest.mark.parametrize(
    "name", [name for name, arm in sorted(_ARMS.items()) if arm[0] == _D_Z]
)
def test_a_latent_width_arm_satisfies_the_head_structure_constraint(name):
    """``d_z % num_heads == 0`` is a constructor invariant; caught here rather than as a
    ``ValueError`` at model build on the production box."""
    vae = _resolved(name)["model_config"]["VAE_model"]

    assert vae["d_z"] % vae["num_heads"] == 0


# --------------------------------------------------------------------------------------
# The reach arms resolve against the filter bank
# --------------------------------------------------------------------------------------
def test_the_null_reach_arm_builds_no_guard():
    """``None`` must resolve to *no* guard, not an identity one: the unguarded arm is the
    architectural baseline, with no gather and no delay module at all."""
    vae = _resolved("sweep_reach_null.yaml")["model_config"]["VAE_model"]

    assert resolve_stream_budgets(vae) is None


def test_every_finite_reach_arm_resolves_and_a_wider_budget_keeps_more_channels():
    """The resolution itself is the go/no-go: it raises on a budget that keeps no channel or
    whose worst delay outruns the arm's warm-up. And a budget is a reach ceiling, so the kept set
    can only grow with it -- an arm that kept fewer channels than a tighter one would mean the
    filter bank or the selection had stopped measuring reach."""
    finite = sorted(
        (value, name) for name, (key, value, _extras) in _ARMS.items()
        if key == _REACH and value is not None
    )
    assert len(finite) > 1, "the reach sweep needs at least two finite arms to compare"

    kept = []
    for _value, name in finite:
        budget = resolve_stream_budgets(_resolved(name)["model_config"]["VAE_model"])
        kept.append((len(budget.target_keep_index), len(budget.source_keep_index)))

    assert all(target > 0 and source > 0 for target, source in kept), kept
    for (tight_target, tight_source), (wide_target, wide_source) in zip(kept, kept[1:]):
        assert tight_target <= wide_target and tight_source <= wide_source, kept


def test_the_240s_arm_needs_its_raised_warmup():
    """The two-key exception is structural, in both directions: the resolved worst delay
    genuinely exceeds the shipped warm-up, and the same budget at the shipped warm-up is refused.
    If either half ever fails, the arm's second delta has become drift."""
    vae = _resolved("sweep_reach_240.yaml")["model_config"]["VAE_model"]

    assert resolve_stream_budgets(vae).max_delay > 30
    with pytest.raises(ValueError, match="warmup_period"):
        resolve_stream_budgets({**vae, "warmup_period": 30})


# --------------------------------------------------------------------------------------
# The collapse criterion is arithmetic, not judgement
# --------------------------------------------------------------------------------------
#: A per-anchor KL just under the collapse threshold: dead by the criterion's own definition.
_DEAD_KL = 0.95 * KL_COLLAPSE_THRESHOLD_NATS


def test_a_healthy_run_is_not_collapsed_despite_its_structural_zero_start():
    """Every run opens at exactly zero KL (zero-initialised posterior residual, beta warm-up
    from 0); the criterion must read the tail, or it calls every healthy run collapsed."""
    kl = [0.0, 0.0, 0.2, 0.9, 1.5, 1.4, 1.6]
    active_frac = [0.0, 0.0, 0.10, 0.30, 0.50, 0.50, 0.48]

    assert not is_collapsed(kl, active_frac, d_z=48)


def test_a_dead_final_stretch_of_the_kl_is_collapsed():
    kl = [0.0, 0.6, 1.1] + [_DEAD_KL] * KL_COLLAPSE_PATIENCE_EPOCHS
    active_frac = [0.0, 0.2, 0.4, 0.3, 0.2, 0.1, 0.1]

    assert is_collapsed(kl, active_frac, d_z=48)


def test_one_epoch_short_of_the_patience_is_not_collapsed():
    """Four dead final epochs, not five: the boundary case that distinguishes the patience
    from a single-epoch threshold test."""
    kl = [0.0, 0.6, 1.1, 0.5] + [_DEAD_KL] * (KL_COLLAPSE_PATIENCE_EPOCHS - 1)
    active_frac = [0.0, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3]

    assert not is_collapsed(kl, active_frac, d_z=48)


def test_too_few_active_dimensions_collapse_regardless_of_the_total_kl():
    """Clause 2 alone: a single runaway dimension can hold the total KL far above clause 1's
    threshold while the rest of the latent is dead -- collapsed into one dimension is still
    collapsed."""
    kl = [0.0, 0.8, 1.9, 2.2, 2.1, 2.3, 2.2]
    too_few = (KL_COLLAPSE_MIN_ACTIVE_DIMS - 1) / 48.0
    active_frac = [0.0, 0.3, 0.2, 0.1, too_few, too_few, too_few]

    assert is_collapsed(kl, active_frac, d_z=48)


def test_a_run_shorter_than_the_patience_is_not_judged_on_its_kl():
    """The criterion presumes a completed run of at least the patience length; a shorter
    series must not fire clause 1 on an incomplete window."""
    kl = [0.001] * (KL_COLLAPSE_PATIENCE_EPOCHS - 2)
    active_frac = [0.5] * (KL_COLLAPSE_PATIENCE_EPOCHS - 2)

    assert not is_collapsed(kl, active_frac, d_z=48)
