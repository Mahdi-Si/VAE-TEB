r"""The shipped configurations, against the constructor they build.

One failure this file exists to catch, and it is silent in both directions. The experiment driver
forwards a ``model_config.VAE_model`` key only when ``inspect.signature`` names it, so a key that
names no constructor parameter is **dropped without a word** and the run trains a different model
from the one the file describes. And a constructor parameter no configuration sets takes its
default, which may be a value no arm intends.

The second half is this architecture's own: every keyword it refuses is a parameter of its
signature, so a configuration setting one would be forwarded and would raise at startup. The
configurations must therefore not set any of them -- which is easy to violate by copying a block
from a sibling.

The rest are relational: the diagnostic page's key is the shared driver's, and every profile that
is a delta on another (the smoke twin, the short bank, the ablation switches) differs from its
parent in exactly the leaves it declares, read back off the built model where the leaf is geometry.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Set

import pytest
import torch
import yaml

from teb_vae.lag_slot_transformer_cfs.nets.core import REFUSED_KEYWORDS
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs

#: The configuration directory this package ships.
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

#: Keys under ``model_config.VAE_model`` that are the *task's* or the *resolver's* rather than the
#: constructor's. Each is consumed elsewhere by name, so it is not an orphan even though the
#: signature sweep drops it.
NON_CONSTRUCTOR_KEYS: Set[str] = {
    # The objective weights and the likelihood, read by the task from its hyperparameters.
    "beta_schedule",
    "kld_beta",
    "beta_prior",
    "lambda_full",
    "lambda_base",
    "likelihood",
    "free_bits",
    # The predictive validation monitor's draw count: a task hyperparameter the driver applies
    # after construction, by the route the seed takes, and never a constructor argument.
    "validation_mc_draws",
    # The reconstruction mixture's draw count, applied to the task the same way.
    "train_mc_draws",
    # The weight on the lag-weighted proposal ridge: an objective weight, applied the same way.
    "proposal_ridge",
    # The per-channel scored horizon's rule, resolved from the shards into
    # ``target_scored_horizon`` by the trainer.
    "target_phase_fast_cutoff_hz",
    "target_phase_fast_horizon",
    # The causal representation, resolved into the four channel tuples and the novelty vector.
    "causal_reach_budget_s",
    "causal_warmup_budget_steps",
    "causal_align_reference",
    "causal_align_reference_source",
    "causal_target_forecast_clock",
    "causal_leg_alignment",
    "causal_phase_operator",
}

#: Constructor parameters no configuration is expected to set, with the reason.
UNSET_BY_DESIGN: Set[str] = {
    # Resolved from the warm-up budget, never written by hand.
    "target_keep_index",
    "target_warmup_steps",
    "source_keep_index",
    "source_warmup_steps",
    "target_align_delays",
    "source_align_delays",
    "target_novelty_frac",
    "target_forecast_shift",
    # Resolved from the scored-horizon rule and the shards' phase-leg frequencies.
    "target_scored_horizon",
    # Weight initialisation is not a configuration decision.
    "init_weights",
} | set(REFUSED_KEYWORDS)


def load(name: str) -> Dict[str, Any]:
    """Load one configuration, resolving its ``base:`` chain by a shallow merge.

    Shallow per top-level block, which is what the shipped loader does and all this file needs:
    every key it inspects lives one level under ``model_config`` or ``general_config``.

    Args:
        name: The configuration filename.

    Returns:
        The merged mapping.
    """
    raw = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    merged = load(base_name)
    for block, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(block), dict):
            for key, inner in value.items():
                if isinstance(inner, dict) and isinstance(merged[block].get(key), dict):
                    merged[block][key].update(inner)
                else:
                    merged[block][key] = inner
        else:
            merged[block] = value
    return merged


def vae_block(name: str) -> Dict[str, Any]:
    """The model block of one configuration.

    Args:
        name: The configuration filename.

    Returns:
        The mapping under ``model_config.VAE_model``.
    """
    return load(name)["model_config"]["VAE_model"]


#: Every configuration this package ships, discovered rather than written out.
#:
#: Discovered here and written out nowhere, which is the opposite of the entry-point tuple's rule
#: and for the opposite reason: an entry point that forgot the launch convention must fail rather
#: than go unseen, while an arm added to the directory and forgotten here would be an arm nothing
#: checked. The two silent failures below -- a key naming no constructor parameter, and a key the
#: constructor refuses -- are exactly the ones a new arm is most likely to introduce, because an arm
#: is written by copying a sibling.
CONFIGS = tuple(sorted(path.name for path in CONFIG_ROOT.glob("*.yaml")))


@pytest.mark.parametrize("name", CONFIGS)
def test_every_configuration_names_only_constructor_keys_and_builds(name: str) -> None:
    """A key naming nothing is dropped by the sweep and changes no model; a refused keyword is a
    parameter, so setting one would refuse the run at startup; and what remains must pass the
    constructor's own refusals (stride against horizon, an even model width, the bounds).

    Args:
        name: The configuration to check.
    """
    block = set(vae_block(name))
    parameters = set(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters)
    orphans = sorted(block - parameters - NON_CONSTRUCTOR_KEYS)
    assert orphans == [], orphans
    assert sorted(block & set(REFUSED_KEYWORDS)) == []
    assert build_from_config(name).n_lags == int(vae_block(name)["max_lag"]) + 1


def test_the_production_config_sets_every_parameter_that_shapes_a_run() -> None:
    """The other half: a parameter no configuration sets takes a default no arm chose.

    The exceptions are named and reasoned in :data:`UNSET_BY_DESIGN` rather than left implicit.
    """
    parameters = inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters
    expected = {
        name
        for name in parameters
        if name != "self" and name not in UNSET_BY_DESIGN
    }
    missing = sorted(expected - set(vae_block("default.yaml")))
    assert missing == [], missing


def test_the_diagnostic_page_is_wired_by_the_shared_key_and_this_packages_callback() -> None:
    """The one seam whose failure is a missing figure rather than an error.

    The callback assembly reads the shared driver's key, so the configuration must name that key
    rather than one matching this package. The callback class is this package's, because the
    family's runs a forward without the per-lag proposals and hands the result to a builder that
    reads two tensors this architecture does not produce.
    """
    from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
    from teb_vae.lag_slot_transformer_cfs.plotting import LagResidualTrfCfsPlotCallback
    from teb_vae.lag_slot_transformer_cfs.trainer import LagResidualTrfCfsTrainer

    key = LagResidualTrfCfsTrainer.PLOT_CONFIG_KEY
    assert key == LagAttnRwsTrainer.PLOT_CONFIG_KEY
    assert key in load("default.yaml")["advanced_config"]["callbacks"]
    assert LagResidualTrfCfsTrainer.plot_callback_cls() is LagResidualTrfCfsPlotCallback


#: The keys that describe the data and must agree between a fixture delta and its parent.
DATA_GEOMETRY_KEYS = (
    "sequence_length",
    "horizon",
    "warmup_period",
    "causal_warmup_budget_steps",
    "c_y",
    "c_u",
    "anchor_stride",
)


def build_from_config(name: str, **overrides: Any) -> SeqVaeLagResidualTrfCfs:
    """Build a configuration's model through the same signature sweep the driver applies.

    Ungated, at the declared widths: the warm-up tuples come from the shards, which a test of the
    configuration does not open. The lag geometry, the anchors and the summation scale do not
    depend on the gate.

    Args:
        name: The configuration filename.
        **overrides: Constructor keywords to replace after the sweep.

    Returns:
        The constructed model.
    """
    parameters = set(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters)
    kwargs = {key: value for key, value in vae_block(name).items() if key in parameters}
    kwargs.update(overrides)
    torch.manual_seed(0)
    return SeqVaeLagResidualTrfCfs(**kwargs)


def test_the_short_bank_profile_builds_the_window_it_declares_and_nothing_else_moves() -> None:
    """One leaf changed, read back off the model rather than off the file.

    The bank length, the embedding rows, the summation scale and the lag-validity width all
    follow from that leaf; the anchors, the horizon and the declared widths are the parent's,
    so the two arms differ in the bank and in nothing else.
    """
    block, parent = vae_block("lag25.yaml"), vae_block("joint.yaml")
    expected_lags = int(block["max_lag"]) + 1
    for key in DATA_GEOMETRY_KEYS:
        assert block[key] == parent[key], key
    assert {key for key in block if block[key] != parent.get(key)} == {"max_lag"}

    model = build_from_config("lag25.yaml").eval()
    assert model.n_lags == expected_lags
    # One embedding row per lag-basis function where the lag identity is expanded on a basis,
    # and one per lag otherwise.
    assert model.proposal_head.lag_embedding.weight.shape[0] == (
        block["lag_basis_dim"] or expected_lags
    )
    assert model.lag_scale == pytest.approx(expected_lags ** -0.5)
    assert model.build_lag_mask(model.sequence_length).shape[1] == expected_lags

    steps, split = int(model.sequence_length), int(model.TARGET_BLOCK_SPLIT)
    y_st = torch.zeros(1, steps, split)
    y_ph = torch.zeros(1, steps, int(model.c_y) - split)
    u_stream = torch.zeros(1, steps, int(model.c_u))
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, return_proposals=True)
    dense = int(model.sequence_length) - int(model.warmup_period) - int(model.horizon)
    assert outputs["lag_valid"].shape == (1, dense, expected_lags)
    assert outputs["mean_proposals"].shape[2] == expected_lags
    assert int(outputs["anchor_index"][0, 0]) == int(model.warmup_period)
    assert int(outputs["anchor_index"][0, -1]) == int(model.sequence_length) - int(model.horizon) - 1


def test_the_short_bank_fixture_delta_keeps_the_smoke_settings_and_takes_only_the_window() -> None:
    """The fixture twin of the short-bank profile: the smoke configuration with that one leaf."""
    tiny, short = vae_block("tiny.yaml"), vae_block("tiny_lag25.yaml")
    assert {key for key in short if short[key] != tiny.get(key)} == {"max_lag"}
    assert short["max_lag"] == vae_block("lag25.yaml")["max_lag"]
    assert short["lag_chunk"] < short["max_lag"] + 1
    assert load("tiny_lag25.yaml")["dataset_config"] == load("tiny.yaml")["dataset_config"]


@pytest.mark.parametrize(
    "name,base,leaves",
    [
        ("lag25_s0_fhr.yaml", "lag25.yaml", {"zero_fhr_scattering_s0": True}),
        ("lag25_s0_up.yaml", "lag25.yaml", {"zero_up_scattering_s0": True}),
        (
            "lag25_s0_both.yaml",
            "lag25.yaml",
            {"zero_fhr_scattering_s0": True, "zero_up_scattering_s0": True},
        ),
        ("target_only_s0_fhr.yaml", "target_only.yaml", {"zero_fhr_scattering_s0": True}),
    ],
)
def test_each_ablation_profile_changes_exactly_its_declared_leaves(name, base, leaves) -> None:
    """A profile that changed a second leaf would attribute two changes to one switch."""
    block, parent = vae_block(name), vae_block(base)
    changed = {key: block[key] for key in block if block[key] != parent.get(key)}
    assert changed == leaves
    assert load(name)["general_config"]["tag"] != load(base)["general_config"]["tag"]


def test_the_tiny_config_inherits_the_data_geometry_from_the_parent() -> None:
    """Capacity is shrunk; the geometry describes the data and must not be."""
    tiny, production = vae_block("tiny.yaml"), vae_block("default.yaml")
    for key in DATA_GEOMETRY_KEYS:
        assert tiny[key] == production[key], key
    assert tiny["d_model"] < production["d_model"]
