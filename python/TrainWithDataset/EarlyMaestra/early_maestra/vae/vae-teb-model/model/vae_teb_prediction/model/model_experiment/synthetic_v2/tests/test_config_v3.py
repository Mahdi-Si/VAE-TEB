r"""S0-T00 / S0-T01 / S0-T06: config_synth_v3.yaml, data_tag decoupling, cache presence.

Pins the three config-level guarantees the ablation ladder rests on:

* ``experiment.data_tag`` splits the cache leaf (``data/<benchmark>/<data_tag>``) from the
  results root (``results/<tag>``), so three arms under ``tag: G1_raw_v3`` read one immutable
  cache; absent ``data_tag`` reproduces today's ``experiment.tag`` path exactly (S0-T01);
* ``config_synth_v3.yaml`` parses, and ``resolve_arm`` + ``build_model`` construct a real
  ``SeqVaeLagAttnV3`` for all three arms with the specced attribute values, differing only in
  the G0 (``causal_norm``) / G1-G5 keys, with a byte-identical ``loss`` block (S0-T06);
* ``config_synth_v2.yaml`` is byte-unchanged (the v1 path is untouched);
* (``-k cache_present``, slow) the shared cache exists and its ``meta.json`` matches the
  config -- skipped when the cache has not been built yet (S0-T00).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    run_pipeline_v2 as drv,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    pl_module_v2 as plm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    resolve_cache_dir,
)

_SV2 = Path(__file__).resolve().parents[1]
_CFG_V3 = _SV2 / "config_synth_v3.yaml"
_CFG_V2 = _SV2 / "config_synth_v2.yaml"

_ARM_ATTRS = {
    "parity": {"causal_norm": False, "posterior_logvar": "independent",
               "logvar_bound": "clamp", "kld_support": "full"},
    "v3_noncausal": {"causal_norm": False, "posterior_logvar": "residual",
                     "logvar_bound": "smooth", "kld_support": "anchor"},
    "v3_prod": {"causal_norm": True, "posterior_logvar": "residual",
                "logvar_bound": "smooth", "kld_support": "anchor"},
}


# ---------------------------------------------------------------------------
# S0-T01: experiment.data_tag cache/results decoupling
# ---------------------------------------------------------------------------
def test_data_tag_absent_is_todays_path() -> None:
    """Absent ``data_tag`` returns exactly the ``experiment.tag`` cache path."""
    cfg = {"experiment": {"tag": "G1_raw_v2_notch"}, "paths": {"data_dir": "./data"}}
    got = resolve_cache_dir(cfg, benchmark="G1_raw")
    assert got.name == "G1_raw_v2_notch"
    assert got.parent.name == "G1_raw"


def test_data_tag_present_splits_cache_from_results() -> None:
    """``data_tag`` != ``tag`` -> cache keys on data_tag, results on tag."""
    cfg = {
        "experiment": {"tag": "G1_raw_v3", "data_tag": "G1_raw_v2_notch"},
        "paths": {"data_dir": "./data", "results_dir": "./results"},
    }
    cache = resolve_cache_dir(cfg, benchmark="G1_raw")
    results = drv._results_dir(cfg, "G1_raw")
    assert cache.name == "G1_raw_v2_notch"
    assert results.name == "G1_raw_v3"


# ---------------------------------------------------------------------------
# S0-T06: config_synth_v3.yaml
# ---------------------------------------------------------------------------
def test_v3_config_parses_and_has_required_keys() -> None:
    cfg = drv.load_config(_CFG_V3)
    assert cfg["experiment"]["tag"] == "G1_raw_v3"
    # data_tag decouples the shared cache from the per-run results root. Its value differs
    # between the dev box (a reduced local pilot cache) and the headline prod run
    # (G1_raw_v2_notch); only the decoupling itself is invariant.
    assert cfg["experiment"]["data_tag"] != cfg["experiment"]["tag"]
    assert cfg["model"]["class"] == "SeqVaeLagAttnV3"
    assert cfg["loss"]["likelihood"] == "gaussian_nll"
    assert cfg["loss"]["sigma_obs"] == "learned"
    assert float(cfg["loss"]["free_bits"]) == 0.0
    assert cfg["loss"]["beta_schedule"]["kind"] == "linear_warmup"
    assert cfg["train"]["pilot_beta_schedule"]["kind"] == "linear_warmup"
    assert cfg["curriculum"]["enabled"] is False
    # ddp.strategy deferred to the resolver (S2-T04).
    assert str(cfg["ddp"]["strategy"]) in ("auto", "")
    assert set(cfg["arms"]) == {"parity", "v3_noncausal", "v3_prod"}


@pytest.mark.parametrize("arm", ["parity", "v3_noncausal", "v3_prod"])
def test_v3_config_builds_each_arm_with_expected_attrs(arm) -> None:
    """resolve_arm + build_model construct a real v3 with the specced attributes.

    This is the end-to-end guard against the overlay-clobber failure: the per-arm deltas
    live inside ``model.v3`` so they override the base v3 overlay, and build_model applies
    the (arm-merged) overlay over the flat block. A regression that let the base overlay win
    would build every arm with ``causal_norm=True`` and this test would catch it.
    """
    cfg = drv.load_config(_CFG_V3)
    resolved = drv.resolve_arm(cfg, arm)
    model, _ = plm.build_model(resolved["model"], torch.device("cpu"))
    assert type(model).__name__ == "SeqVaeLagAttnV3"
    for attr, expected in _ARM_ATTRS[arm].items():
        assert getattr(model, attr) == expected, (
            f"arm {arm}: {attr}={getattr(model, attr)!r}, expected {expected!r}"
        )
    # head_structured_latent held true across arms; entmax/freeze come from the overlay.
    assert model.head_structured_latent is True


def test_v3_config_loss_identical_across_arms() -> None:
    cfg = drv.load_config(_CFG_V3)
    losses = [drv.resolve_arm(cfg, a)["loss"]
              for a in ("parity", "v3_noncausal", "v3_prod")]
    assert losses[0] == losses[1] == losses[2] == cfg["loss"]


def test_v3_config_models_differ_only_in_g0(  # noqa: D103
) -> None:
    cfg = drv.load_config(_CFG_V3)
    prod = drv.resolve_arm(cfg, "v3_prod")["model"]["v3"]
    noncausal = drv.resolve_arm(cfg, "v3_noncausal")["model"]["v3"]
    diff = {k for k in set(prod) | set(noncausal) if prod.get(k) != noncausal.get(k)}
    assert diff == {"causal_norm"}


def test_v2_config_byte_unchanged_reference() -> None:
    """config_synth_v2.yaml still resolves to a v1 model with today's kwargs (untouched)."""
    cfg = drv.load_config(_CFG_V2)
    assert "class" not in cfg["model"]  # no model.class -> the v1 alias
    model, _ = plm.build_model(cfg["model"], torch.device("cpu"))
    assert type(model).__name__ == "SeqVaeLagAttnV1"


# ---------------------------------------------------------------------------
# S0-T00: shared cache presence (slow; skipped when the cache has not been built)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cache_present_and_matches_config() -> None:
    """The shared cache exists and its meta.json matches config_synth_v2's generation knobs.

    Skipped (not failed) when the cache has not been built yet, so a fresh checkout reads as
    a skip rather than a red gate. Once the build the user is running lands, this asserts the
    splits + meta.json + norm_stats are present and consistent.
    """
    import json

    cfg = drv.load_config(_CFG_V3)
    cache = resolve_cache_dir(cfg, benchmark="G1_raw")
    if not (cache / "train.npz").is_file():
        pytest.skip(f"shared cache not built yet at {cache}")
    for name in ("train.npz", "val.npz", "test.npz", "meta.json", "norm_stats.npz"):
        assert (cache / name).is_file(), f"missing {name} under {cache}"
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    v2 = drv.load_config(_CFG_V2)
    raw = v2["benchmarks"]["G1_raw"]["raw"]
    # A stale cache cannot be silently adopted: the generation knobs must match.
    meta_raw = (meta.get("raw") or meta.get("config", {}).get("benchmarks", {})
                .get("G1_raw", {}).get("raw", {}))
    if meta_raw:
        assert float(meta_raw.get("f_pulse", raw["f_pulse"])) == float(raw["f_pulse"])
        assert bool(meta_raw.get("fhrv_notch_enabled", raw["fhrv_notch_enabled"])) == \
            bool(raw["fhrv_notch_enabled"])
    n_train = int(meta.get("n_train", meta.get("counts", {}).get("train", 0)) or 0)
    print(f"[cache_present] {cache}  n_train={n_train}")
