"""The patch eval's binding contract: what runs, whose implementation runs, and which keys reconcile."""
from __future__ import annotations

import inspect
import pkgutil
from pathlib import Path

import yaml

from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_cfs.eval.run import merged_analysis_functions, unskippable_for
from teb_vae.lag_attn_transformer_patch.eval import analyses
from teb_vae.lag_attn_transformer_patch.eval import binding as binding_module
from teb_vae.lag_attn_transformer_patch.eval.binding import TRF_PATCH_BINDING

from .conftest import _REPO_ROOT

PACKAGE = "teb_vae.lag_attn_transformer_patch"


def _implementation(function):
    """The analysis behind a registry entry, through ``raw.with_unlabelled_classes``'s closure."""
    return inspect.getclosurevars(function).nonlocals.get("run", function)


def test_the_registry_runs_the_patch_implementations_in_the_shared_order() -> None:
    """Excluded analyses never run; every ``eval/analyses`` module is registered under its own name
    and runs its own entry point (so the ported and replaced names resolve to the patch code);
    ``cross_subgroup`` runs the patch sources; and the shared names keep the CFS order."""
    registry = merged_analysis_functions(TRF_PATCH_BINDING)
    always = unskippable_for(TRF_PATCH_BINDING)
    cfs = merged_analysis_functions(CFS_BINDING)

    for name in ("warmup", "spectral_skill", "band_partition"):
        assert name not in registry and name not in always, name

    modules = [info.name for info in pkgutil.iter_modules(analyses.__path__)]
    assert {"calibration", "samples", "occlusion", "time_shift"} <= set(modules)
    for name in modules:
        assert name in registry, f"eval/analyses/{name}.py is not registered"
        assert _implementation(registry[name]).__module__ == f"{PACKAGE}.eval.analyses.{name}", name
    assert registry["cross_subgroup"] is binding_module.run_cross_subgroup_analysis

    shared = [name for name in registry if name in cfs]
    assert shared == [name for name in cfs if name in registry]
    assert list(registry)[-1] == "cross_subgroup"


def test_every_geometry_key_is_a_constructor_parameter_and_a_default_config_key() -> None:
    """``preflight.reconcile`` silently skips a key missing from either side, so a key that is not
    both is a reconciliation that never happens."""
    keys = TRF_PATCH_BINDING.geometry_keys
    parameters = set(inspect.signature(TRF_PATCH_BINDING.model_cls.__init__).parameters)
    config = yaml.safe_load(
        (Path(_REPO_ROOT) / "teb_vae" / "lag_attn_transformer_patch" / "configs" / "default.yaml").read_text()
    )
    declared = set(config["model_config"]["VAE_model"])

    assert len(set(keys)) == len(keys) > 0
    assert set(keys) <= parameters, sorted(set(keys) - parameters)
    assert set(keys) <= declared, sorted(set(keys) - declared)
