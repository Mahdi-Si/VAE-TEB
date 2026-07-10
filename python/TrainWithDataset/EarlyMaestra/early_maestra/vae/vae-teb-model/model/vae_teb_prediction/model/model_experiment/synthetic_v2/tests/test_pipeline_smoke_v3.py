r"""S4-T07: per-arm eval + report on the pilot cache, end to end.

Three arms that differ only in ``model_kwargs`` must produce three *distinguishable* artifact
trees. Before Sprint 4 they did not: ``metrics.json`` recorded no arm and no model class, the
report header named neither, and ``run_eval`` rebuilt the model from the **config** rather than
from the graded checkpoint -- so grading arm B's weights under arm C's config would have loaded
a structurally-compatible but wrong architecture without complaint.

This module asserts the artifacts exist, name their arm and class, carry the Sprint 3/4 gates,
and never print the retired ``-> 0`` claim for a KL ratio.

Everything here is `slow`: it reads real pilot checkpoints trained on the local cache. Both are
skipped (loudly, with a reason) when absent, rather than passing vacuously.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    run_pipeline_v2 as rp,
)

pytestmark = pytest.mark.slow

_SV2 = Path(__file__).resolve().parents[1]
_CONFIG = _SV2 / "config_synth_v3.yaml"
_ARMS = ("parity", "v3_noncausal", "v3_prod")
_SPLIT = "val"


def _run_root(arm: str) -> Path:
    config = rp.load_config(_CONFIG)
    return rp._run_dir(config, config["experiment"]["benchmark"], arm)


def _require_graded(arm: str) -> Path:
    split_dir = _run_root(arm) / _SPLIT
    if not (split_dir / "metrics.json").is_file():
        pytest.skip(
            f"arm {arm!r} has not been graded on the {_SPLIT} split. Run:\n"
            f"  --stage train --pilot --arm {arm}\n"
            f"  --stage eval --arm {arm} --split {_SPLIT}"
        )
    return split_dir


def _metrics(arm: str) -> dict:
    return json.loads((_require_graded(arm) / "metrics.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm", _ARMS)
def test_per_arm_artifacts_exist(arm: str) -> None:
    split_dir = _require_graded(arm)
    for name in ("metrics.json", "per_sample_eval.npz", "report.md"):
        assert (split_dir / name).is_file(), f"{arm}/{_SPLIT}/{name} missing"


@pytest.mark.parametrize("arm", _ARMS)
def test_each_arm_names_itself(arm: str) -> None:
    r"""S4-T03: the arm and the model class are stamped everywhere they can be traced from."""
    split_dir = _require_graded(arm)
    m = _metrics(arm)
    assert m["arm"] == arm
    assert m["model_class"] == "SeqVaeLagAttnV3"
    assert m["model_kwargs"] and m["cache_dir"] and m["ckpt"]

    with np.load(split_dir / "per_sample_eval.npz", allow_pickle=False) as z:
        assert str(z["arm"]) == arm
        assert str(z["model_class"]) == "SeqVaeLagAttnV3"

    header = "\n".join((split_dir / "report.md").read_text(encoding="utf-8").splitlines()[:8])
    assert arm in header and "SeqVaeLagAttnV3" in header


def test_the_three_arms_are_distinguishable() -> None:
    r"""Structurally identical arms, three different artifact trees.

    ``parity`` reduces v3 to v1's latent machinery (``kld_support: full``,
    ``posterior_logvar: independent``); the two v3 arms use anchor-aligned KL support and a
    residual posterior. Those differences must be visible in the graded output, not merely in
    the config that launched it.
    """
    got = {a: _metrics(a) for a in _ARMS}
    assert got["parity"]["calibration"]["kld_support"] == "full"
    assert got["v3_noncausal"]["calibration"]["kld_support"] == "anchor"
    assert got["v3_prod"]["calibration"]["kld_support"] == "anchor"
    # Every arm rebuilt as v3 -- NOT silently as the committed v1 alias (S1-T04).
    assert {m["model_class"] for m in got.values()} == {"SeqVaeLagAttnV3"}
    assert got["parity"]["model_kwargs"]["posterior_logvar"] == "independent"
    assert got["v3_prod"]["model_kwargs"]["posterior_logvar"] == "residual"
    assert got["v3_prod"]["model_kwargs"]["causal_norm"] is True
    assert got["v3_noncausal"]["model_kwargs"]["causal_norm"] is False


@pytest.mark.parametrize("arm", _ARMS)
def test_sprint_3_and_4_gates_are_populated(arm: str) -> None:
    m = _metrics(arm)
    pc = m["prediction_controls"]
    assert set(pc["controls"]) == {"shuffle", "reverse"}
    assert pc["n_signal_cells"] == 12          # 15 cells, 3 of them null
    for key in ("feat_loss", "base_loss", "ordering_pass", "ordering_pass_frac"):
        assert key in pc["overall"], key

    cal = m["calibration"]
    knc = cal["kbar_at_null_cells"]
    assert knc["n_cells"] == 3
    assert knc["ci_lo"] <= knc["mean"] <= knc["ci_hi"]
    assert isinstance(knc["pass"], bool)

    # The KL-space ratio is retained, annotated, and never expected to vanish.
    for ctrl, res in m["null_controls"].items():
        assert res["expected_to_vanish"] is False
        assert "Finding F2" in res["note"], ctrl


@pytest.mark.parametrize("arm", _ARMS)
def test_out_of_support_flag_tracks_kld_support(arm: str) -> None:
    r"""S4-T01: only the anchor-support arms flag ``kbar_full``; ``parity`` flags nothing."""
    m = _metrics(arm)
    flagged = m["calibration"]["kld_variants"]["kbar_full"]["out_of_support"]
    assert flagged is (m["calibration"]["kld_support"] == "anchor")
    # kbar_full is flagged, never dropped.
    assert m["calibration"]["kld_variants"]["kbar_full"]["n"] > 0
    with np.load(_require_graded(arm) / "per_sample_eval.npz", allow_pickle=False) as z:
        assert "kbar_full" in z.files


@pytest.mark.parametrize("arm", _ARMS)
def test_no_report_claims_a_kl_ratio_should_reach_zero(arm: str) -> None:
    text = (_require_graded(arm) / "report.md").read_text(encoding="utf-8")
    for banned in ("→ 0", "-> 0"):
        assert banned not in text, f"{arm} report still claims a KL ratio reaches 0"
    assert "Readouts (not gates)" in text
    assert "Finding F2" in text
    # The training-time perm control must never appear in an eval report.
    assert "feat_loss_perm" not in text


@pytest.mark.parametrize("arm", _ARMS)
def test_report_dagger_appears_only_under_anchor_support(arm: str) -> None:
    m = _metrics(arm)
    text = (_require_graded(arm) / "report.md").read_text(encoding="utf-8")
    expect_dagger = m["calibration"]["kld_support"] == "anchor"
    assert ("Out of support" in text) is expect_dagger


# ---------------------------------------------------------------------------
# S8-T02: every plugin stage lands its artifact, and the cross-arm index reads them
# ---------------------------------------------------------------------------
def _require_stage(arm: str, filename: str, stage: str) -> Path:
    path = _require_graded(arm) / filename
    if not path.is_file():
        pytest.skip(f"arm {arm!r} lacks {filename}. Run: --stage {stage} --arm {arm} "
                    f"--split {_SPLIT}")
    return path


@pytest.mark.parametrize("arm", _ARMS)
def test_calibration_folds_into_metrics_json(arm: str) -> None:
    r"""Sprint 5 folds its block in beside -- never over -- the gamma-vs-TE fit."""
    m = _metrics(arm)
    block = m.get("calibration_predictive")
    if block is None:
        pytest.skip(f"arm {arm!r} lacks calibration_predictive. Run: --stage calibration")
    assert "gamma_inj" in m["calibration"], "the gamma fit must survive the fold"
    for key in ("nll_mean", "crps_mean", "coverage_90", "by_te", "null_cell_coverage"):
        assert key in block, key


@pytest.mark.parametrize("arm", _ARMS)
def test_lag_intervention_json_is_well_formed(arm: str) -> None:
    payload = json.loads(
        _require_stage(arm, "lag_intervention.json", "lag_intervention").read_text("utf-8")
    )
    assert payload["arm"] == arm and payload["split"] == _SPLIT
    assert payload["overall"]["n_signal_cells"] == 12
    # An all-keep mask must reduce to the causal validity mask -- exactly, not approximately.
    assert payload["noop_max_abs_delta"] <= payload["noop_atol"]
    # S8-T01 extended rho_by_band to the per-cell true band, which `arms_report` reads.
    for band in ("inband", "outband"):
        assert band in payload["rho_by_band"], band
        assert set(payload["rho_by_band"][band]) == {"rho", "ci", "n_cells", "gated"}


@pytest.mark.parametrize("arm", _ARMS)
def test_cmi_json_is_well_formed_and_the_latent_recovery_is_model_free(arm: str) -> None:
    r"""``cmi_latent`` never reads the checkpoint, so its recovery is identical across arms."""
    payload = json.loads(_require_stage(arm, "cmi.json", "cmi").read_text("utf-8"))
    assert payload["arm"] == arm and payload["split"] == _SPLIT
    assert payload["ceiling_nats"] == pytest.approx(np.log(payload["estimator"]
                                                           ["contrastive_batch"]))
    rec = payload["recovery"]
    assert rec["available"] is True
    assert rec["n_cells"] >= 8
    assert rec["spearman_cmi_te_inj"] > 0.8
    assert rec["max_abs_null_cmi"] < 0.05
    assert rec["factor2_pass_frac"] == 1.0
    assert (_require_graded(arm) / "cmi_table.csv").is_file()


def test_cmi_latent_recovery_is_identical_across_arms() -> None:
    r"""The ``latent`` config reads only regenerated latents -- the arms cannot move it."""
    rhos = {}
    for arm in _ARMS:
        payload = json.loads(_require_stage(arm, "cmi.json", "cmi").read_text("utf-8"))
        rhos[arm] = payload["recovery"]["spearman_cmi_te_inj"]
    assert len(set(rhos.values())) == 1, f"a model-free statistic moved across arms: {rhos}"


def test_cmi_bias_is_flagged_unreliable_exactly_on_the_non_causal_arms() -> None:
    r"""The G0 leak, detected by an instrument that never reads the KL.

    ``causal_norm: false`` pools ``GroupNorm`` statistics across time, so ``target_state[b, t]``
    carries a per-sample factor. The fit/eval split is by sample, so a regression on it does not
    transfer and the held-out ``cond_r2_v`` goes negative -- on every cell of both non-causal arms,
    and on none of ``v3_prod``'s.
    """
    verdicts, r2 = {}, {}
    for arm in _ARMS:
        payload = json.loads(_require_stage(arm, "cmi.json", "cmi").read_text("utf-8"))
        bias = payload["overall"]["cmi_bias"]
        verdicts[arm] = bias["reliable"]
        r2[arm] = payload["overall"]["cond_r2_feature_model"]["v"]

    assert verdicts == {"parity": False, "v3_noncausal": False, "v3_prod": True}, verdicts
    assert r2["parity"] < 0 and r2["v3_noncausal"] < 0, r2
    assert r2["v3_prod"] > 0, r2


def test_arms_report_tabulates_every_arm_at_the_tag_root() -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    config = rp.load_config(_CONFIG)
    tag_root = rp._results_dir(config, config["experiment"]["benchmark"])
    path = tag_root / "arms_report.md"
    if not path.is_file():
        pytest.skip("arms_report.md absent. Run: --stage arms_report")
    text = path.read_text(encoding="utf-8")

    header = next(ln for ln in text.splitlines() if ln.startswith("| arm |"))
    cells = [c.strip() for c in header.strip("|").split("|")]
    assert cells == ["arm", "model_class"] + [c[0] for c in ar.ARMS_REPORT_COLUMNS]

    for arm in _ARMS:
        assert f"[`{arm}`]({arm}/{_SPLIT}/report.md)" in text
    # It lives at the arm-INDEPENDENT root, one level above every arm.
    assert not any((tag_root / arm / "arms_report.md").exists() for arm in _ARMS)


def test_the_dict_driver_dispatches_arms_report_once_after_the_arm_sweep() -> None:
    r"""``arms_report`` is cross-arm and model-free, so it runs **once**, arm-less, last.

    The generic per-arm plugin loop skips it (``not spec.model_dependent``) and ``main()`` gives it
    ``arm=None``. Without the explicit post-sweep block in ``run_pipeline`` it would be dispatched
    nowhere -- which is what the S8-T01 task card's "registry entry" alone would have produced. A
    dry run exercises the wiring without touching a checkpoint.
    """
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = rp.run_pipeline({
            "config_path": str(_CONFIG), "dry_run": True, "split": _SPLIT, "arms": list(_ARMS),
            "stages": {"r0_realizability": False, "build": False, "data_previews": False,
                       "beta_select": False, "train": False, "eval": True, "test_plots": False,
                       "calibration": True, "lag_intervention": True, "cmi": True,
                       "report": True, "arms_report": True},
        })

    assert status["arms_report"] == "dry-run"
    assert not any(k.startswith("arms_report[") for k in status), "arms_report ran per arm"
    # ..while the arm-scoped analysis stages ran once per arm.
    for name in ("eval", "calibration", "lag_intervention", "cmi", "report"):
        assert sum(1 for k in status if k.startswith(f"{name}[")) == len(_ARMS), name
    # Model-free stages carry no arm suffix at all.
    assert "data_previews" in status and not any(k.startswith("data_previews[") for k in status)


def test_the_model_free_artifacts_are_written_once_not_per_arm() -> None:
    r"""``realizability.json`` and the data-story gallery belong to the tag, not to an arm."""
    config = rp.load_config(_CONFIG)
    tag_root = rp._results_dir(config, config["experiment"]["benchmark"])
    for arm in _ARMS:
        arm_dir = tag_root / arm
        if not arm_dir.is_dir():
            continue
        assert not (arm_dir / "realizability.json").exists(), arm
        assert not (arm_dir / "arms_report.md").exists(), arm


def test_the_results_tree_is_fully_gitignored() -> None:
    r"""S8-T02: ``git status --porcelain`` must be empty after a pilot run.

    Checks the rules rather than the working tree, so it does not depend on what happens to be on
    disk when it runs. Every artifact the pipeline writes under ``results/`` must be ignored.
    """
    import subprocess

    config = rp.load_config(_CONFIG)
    tag_root = rp._results_dir(config, config["experiment"]["benchmark"])
    candidates = [
        tag_root / "arms_report.md",
        tag_root / "realizability.json",
        tag_root / "v3_prod" / "best.ckpt",
        tag_root / "v3_prod" / "val" / "metrics.json",
        tag_root / "v3_prod" / "val" / "cmi.json",
        tag_root / "v3_prod" / "val" / "cmi_table.csv",
        tag_root / "v3_prod" / "val" / "lag_intervention.json",
        tag_root / "v3_prod" / "val" / "per_sample_eval.npz",
        tag_root / "v3_prod" / "val" / "report.md",
        tag_root / "v3_prod" / "val" / "figures" / "cmi_comparison.pdf",
        tag_root / "v3_prod" / "val" / "calibration" / "reliability.pdf",
        tag_root / "v3_prod" / "logs" / "version_0" / "metrics.csv",
    ]
    # ``git check-ignore`` echoes back, verbatim, whichever of its arguments are ignored -- so feed
    # it repo-relative POSIX paths and compare against those, not against resolved absolutes.
    rel = [p.resolve().relative_to(Path(_REPO_ROOT).resolve()).as_posix() for p in candidates]
    out = subprocess.run(
        ["git", "check-ignore", "--no-index", *rel],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=120,
    )
    ignored = {ln.strip().replace("\\", "/") for ln in out.stdout.splitlines() if ln.strip()}
    missing = [p for p in rel if p not in ignored]
    assert not missing, "these generated artifacts are NOT gitignored: " + ", ".join(missing)
