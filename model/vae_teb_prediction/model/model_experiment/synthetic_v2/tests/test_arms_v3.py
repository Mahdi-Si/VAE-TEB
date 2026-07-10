r"""S0-T04 / S0-T05: arm resolution, arm-scoped run dirs, and the ``--arm`` CLI seam.

The three ``synthetic_v3`` arms (``parity`` / ``v3_noncausal`` / ``v3_prod``) share one
cache, one seed set, and one objective, differing only in a handful of ``model`` kwargs.
These tests pin the plumbing that keeps that true:

* ``resolve_arm`` deep-merges ``arms.<name>`` over the base ``model`` / ``loss`` blocks so
  the three resolved ``model`` blocks differ in *exactly* the intended keys and the
  resolved ``loss`` block is byte-identical across arms (S0-T04a);
* ``_run_dir`` arm-scopes the run root to ``results/<tag>/<arm>/`` and reproduces
  ``_results_dir`` exactly for ``arm=None`` (S0-T04b);
* ``_select_arm`` enforces the CLI contract (unknown / multi-arm / single-arm-default);
* ``_stage_subprocess_cmd`` carries ``--arm`` into the DDP re-exec (S0-T05);
* ``save_checkpoint_v2`` stamps both ``model_class`` and ``arm`` into the blob (S0-T04c).
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


def _v3_arm_config() -> dict:
    """A minimal config carrying the three-arm ladder over a shared base ``model``/``loss``."""
    return {
        "experiment": {
            "tag": "G1_raw_v3",
            "data_tag": "G1_raw_v2_notch",
            "benchmark": "G1_raw",
        },
        "paths": {"results_dir": "./results", "data_dir": "./data"},
        "model": {
            "class": "SeqVaeLagAttnV3",
            "d_model": 128,
            "causal_norm": True,
            "posterior_logvar": "residual",
            "logvar_bound": "smooth",
            "kld_support": "anchor",
            "lag_bias_init": "alibi_decay",
            "use_entmax": True,
        },
        "loss": {"likelihood": "gaussian_nll", "sigma_obs": "learned", "free_bits": 0.0},
        "arms": {
            "parity": {"model": {"causal_norm": False, "posterior_logvar": "independent",
                                 "logvar_bound": "clamp", "kld_support": "full",
                                 "lag_bias_init": "normal", "use_entmax": False}},
            "v3_noncausal": {"model": {"causal_norm": False}},
            "v3_prod": {"model": {"causal_norm": True}},
        },
    }


# ---------------------------------------------------------------------------
# S0-T04a: resolve_arm
# ---------------------------------------------------------------------------
def test_resolve_arm_sets_intended_keys() -> None:
    cfg = _v3_arm_config()
    assert drv.resolve_arm(cfg, "v3_prod")["model"]["causal_norm"] is True
    assert drv.resolve_arm(cfg, "v3_noncausal")["model"]["causal_norm"] is False
    par = drv.resolve_arm(cfg, "parity")["model"]
    assert par["posterior_logvar"] == "independent"
    assert par["logvar_bound"] == "clamp"
    assert par["kld_support"] == "full"
    assert par["lag_bias_init"] == "normal"
    assert par["use_entmax"] is False
    # The merge is deep: base keys survive an arm that sets only a couple of them.
    assert par["d_model"] == 128


def test_resolve_arm_none_is_identity() -> None:
    cfg = _v3_arm_config()
    assert drv.resolve_arm(cfg, None) is cfg


def test_resolve_arm_models_differ_only_in_intended_keys() -> None:
    cfg = _v3_arm_config()
    prod = drv.resolve_arm(cfg, "v3_prod")["model"]
    parity = drv.resolve_arm(cfg, "parity")["model"]
    noncausal = drv.resolve_arm(cfg, "v3_noncausal")["model"]

    # parity vs v3_prod: the six latent/encoder-machinery keys.
    diff = {k for k in set(prod) | set(parity) if prod.get(k) != parity.get(k)}
    assert diff == {"causal_norm", "posterior_logvar", "logvar_bound",
                    "kld_support", "lag_bias_init", "use_entmax"}
    # v3_noncausal vs v3_prod: only causal_norm (the G0 isolation).
    diff2 = {k for k in set(prod) | set(noncausal) if prod.get(k) != noncausal.get(k)}
    assert diff2 == {"causal_norm"}


def test_resolve_arm_loss_identical_across_arms() -> None:
    cfg = _v3_arm_config()
    losses = [drv.resolve_arm(cfg, a)["loss"]
              for a in ("parity", "v3_noncausal", "v3_prod")]
    assert losses[0] == losses[1] == losses[2] == cfg["loss"]


def test_resolve_arm_unknown_raises() -> None:
    with pytest.raises(ValueError):
        drv.resolve_arm(_v3_arm_config(), "nope")


# ---------------------------------------------------------------------------
# S0-T04b: _run_dir
# ---------------------------------------------------------------------------
def test_run_dir_arm_scoped() -> None:
    cfg = _v3_arm_config()
    rd = drv._run_dir(cfg, "G1_raw", "v3_prod")
    assert rd.name == "v3_prod"
    assert rd.parent.name == "G1_raw_v3"


def test_run_dir_none_matches_results_dir() -> None:
    cfg = _v3_arm_config()
    assert drv._run_dir(cfg, "G1_raw", None) == drv._results_dir(cfg, "G1_raw")


# ---------------------------------------------------------------------------
# S0-T05: _select_arm + subprocess
# ---------------------------------------------------------------------------
def test_select_arm_explicit_and_default() -> None:
    cfg = _v3_arm_config()
    assert drv._select_arm(cfg, "v3_prod") == "v3_prod"
    assert drv._select_arm({"arms": {"only": {}}}, None) == "only"  # single-arm default
    assert drv._select_arm({}, None) is None  # no arms block (v1 / v2 path)


def test_select_arm_multi_arm_without_flag_exits() -> None:
    with pytest.raises(SystemExit):
        drv._select_arm(_v3_arm_config(), None)


def test_select_arm_unknown_flag_exits() -> None:
    with pytest.raises(SystemExit):
        drv._select_arm(_v3_arm_config(), "bogus")


def test_subprocess_cmd_carries_arm() -> None:
    cmd = drv._stage_subprocess_cmd(
        Path("cfg.yaml"), "train", devices=8, pilot=True, arm="v3_prod",
        max_samples=64,
    )
    assert "--config" in cmd
    assert cmd[cmd.index("--arm") + 1] == "v3_prod"
    assert cmd[cmd.index("--max-samples") + 1] == "64"
    assert "--pilot" in cmd
    # Arm-less path omits the flag (v1 / v2 back-compat).
    cmd0 = drv._stage_subprocess_cmd(Path("cfg.yaml"), "train")
    assert "--arm" not in cmd0


# ---------------------------------------------------------------------------
# S0-T04c: save_checkpoint_v2 stamps arm + model_class (write-site provenance)
# ---------------------------------------------------------------------------
def test_save_checkpoint_stamps_arm_and_class(tmp_path) -> None:
    model_cfg = {
        "sequence_length": 16, "d_model": 16, "d_z": 8, "horizon": 4,
        "warmup_period": 2, "c_y": 87, "c_u": 101, "use_up_st": True,
        "max_lag": 8, "num_heads": 4, "d_head": 4, "lstm_layers": 1,
        "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
    }
    model, kwargs = plm.build_model(model_cfg, torch.device("cpu"))
    ckpt_path = tmp_path / "final.ckpt"
    plm.save_checkpoint_v2(
        ckpt_path, model=model, model_kwargs=kwargs, config={"experiment": {}},
        data_meta={}, epoch=1, val_loss=float("nan"),
        loss_settings={}, latent_stats_fitted=False, arm="v3_prod",
    )
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert blob["arm"] == "v3_prod"
    assert blob["model_class"] == "SeqVaeLagAttnV1"  # the committed alias

    # Arm-less save records arm=None (v1 / v2 path unchanged).
    plm.save_checkpoint_v2(
        ckpt_path, model=model, model_kwargs=kwargs, config={"experiment": {}},
        data_meta={}, epoch=1, val_loss=float("nan"),
        loss_settings={}, latent_stats_fitted=False,
    )
    blob0 = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert blob0["arm"] is None


# ---------------------------------------------------------------------------
# S2-T06: cross-arm seed determinism (the arms must share one cache order)
# ---------------------------------------------------------------------------
def test_arms_consume_different_rng_during_construction() -> None:
    """The arms are structurally different, so model init leaves a DIFFERENT RNG state.

    This is *why* ``train_v2`` re-seeds after construction: without it, each arm would get a
    different DataLoader shuffle order at the same seed and a gamma difference could no longer
    be attributed to the model alone (Section 7).
    """
    import lightning as pl

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E501
        resolve_arm as _resolve,
    )

    cfg = _v3_arm_config()
    # Give the arms real (small) constructor dims so they actually build.
    cfg["model"] = {**cfg["model"], "sequence_length": 16, "d_model": 16, "d_z": 8,
                    "horizon": 4, "warmup_period": 2, "c_y": 87, "c_u": 101,
                    "max_lag": 8, "num_heads": 4, "d_head": 4, "lstm_layers": 1,
                    "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True}
    # Move the arm deltas into model.v3 (the real config layout) so build_model applies them.
    cfg["model"]["v3"] = {k: cfg["model"].pop(k) for k in
                          ("causal_norm", "posterior_logvar", "logvar_bound",
                           "kld_support", "lag_bias_init", "use_entmax")}
    cfg["arms"] = {a: {"model": {"v3": d["model"]}} for a, d in cfg["arms"].items()}

    states = {}
    for arm in ("parity", "v3_prod"):
        pl.seed_everything(0, workers=True)
        plm.build_model(_resolve(cfg, arm)["model"], torch.device("cpu"))
        states[arm] = torch.randn(3).tolist()
    assert states["parity"] != states["v3_prod"], (
        "expected the arms to consume different RNG draws; if this ever becomes equal the "
        "post-construction re-seed is redundant but harmless"
    )


def test_reseed_after_construction_restores_common_rng_state() -> None:
    """Re-seeding after construction (as train_v2 does) makes the post-model RNG identical."""
    import lightning as pl

    cfg = {"sequence_length": 16, "d_model": 16, "d_z": 8, "horizon": 4,
           "warmup_period": 2, "c_y": 87, "c_u": 101, "max_lag": 8, "num_heads": 4,
           "d_head": 4, "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0],
           "head_structured_latent": True}
    v3_prod = {**cfg, "class": "SeqVaeLagAttnV3",
               "v3": {"causal_norm": True, "posterior_logvar": "residual"}}
    parity = {**cfg, "class": "SeqVaeLagAttnV3",
              "v3": {"causal_norm": False, "posterior_logvar": "independent",
                     "lag_bias_init": "normal", "use_entmax": False}}

    states = []
    for model_cfg in (parity, v3_prod):
        pl.seed_everything(0, workers=True)
        plm.build_model(model_cfg, torch.device("cpu"))
        pl.seed_everything(0, workers=True)  # <- the train_v2 re-seed
        states.append(torch.randn(3).tolist())
    assert states[0] == states[1], (
        "post-construction re-seed must leave every arm with the same RNG state, so the "
        "DataLoader shuffle (and hence the cache order) is identical across arms"
    )


@pytest.mark.slow
def test_first_batch_guids_identical_across_arms(shared_cache_dir) -> None:
    """End-to-end: all three arms see the SAME first training batch at a fixed seed.

    Reproduces ``train_v2``'s sequence (seed -> build arm model -> re-seed -> build loader) and
    compares the per-sample ``guid`` order. This is the assertion Section 7 actually rests on.
    """
    import lightning as pl

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v2 import (  # noqa: E501
        SyntheticTEDataModuleV2,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E501
        load_config,
    )

    assert shared_cache_dir is not None, "no cache found; build it first (S0-T00)"
    cfg = load_config(Path(__file__).resolve().parents[1] / "config_synth_v3.yaml")
    cfg["dataset"] = {**(cfg.get("dataset") or {}), "num_workers": 0,
                      "persistent_workers": False, "pin_memory": False}

    guids = {}
    for arm in ("parity", "v3_noncausal", "v3_prod"):
        pl.seed_everything(0, workers=True)
        plm.build_model(drv.resolve_arm(cfg, arm)["model"], torch.device("cpu"))
        pl.seed_everything(0, workers=True)  # the train_v2 re-seed
        dm = SyntheticTEDataModuleV2(cfg, batch_size=4, cache_dir=shared_cache_dir)
        dm.setup("fit")
        guids[arm] = list(next(iter(dm.train_dataloader()))["guid"])

    assert guids["parity"] == guids["v3_noncausal"] == guids["v3_prod"], (
        f"arms saw different first batches: {guids}"
    )
