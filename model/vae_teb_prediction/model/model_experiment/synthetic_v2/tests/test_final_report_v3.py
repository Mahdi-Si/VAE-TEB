r"""S3-T05: the report's headline gates show the control that actually discriminates.

The old headline table printed ``| null_ratio (shuffle) -> 0 | 1.05 |`` -- a plausible number
under a caption contradicting it, from which a reader would conclude the model ignores the
source. Now:

* the **headline gates** carry the prediction-space ordering
  $\mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}} < \mathcal L_{\mathrm{feat}}^{\pi(U)}$,
  rendered as ``pass`` / ``FAIL``;
* the KL ratio moves to a ``## Readouts (not gates)`` table carrying the Finding-F2 note;
* the training-time ``feat_loss_perm`` (a ``source_state`` derangement) and the eval-time
  ``feat_loss_shuffle`` (an input-stream corruption) never share a table;
* a legacy ``metrics.json`` without ``prediction_controls`` renders ``n/a``, not a traceback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    eval_v2,
    final_report_v2 as fr,
)


def _pred_block(*, passing: bool = True) -> Dict[str, Any]:
    fsh = 3.0 if passing else 1.5     # passing: feat < base < feat_shuffle
    return {
        "controls": ["shuffle", "reverse"],
        "n_signal_cells": 2,
        "overall": {
            "feat_loss": 1.0, "base_loss": 2.0,
            "feat_loss_shuffle": fsh, "shuffle_penalty_shuffle": fsh - 1.0,
            "ordering_pass_shuffle": passing,
            "feat_loss_reverse": 2.8, "shuffle_penalty_reverse": 1.8,
            "ordering_pass_reverse": True,
            "ordering_pass": passing, "ordering_pass_frac": 1.0 if passing else 0.5,
        },
        "per_cell": {},
    }


def _null_block() -> Dict[str, Any]:
    return {
        "shuffle": {"mean_ratio": 1.05, "expected_to_vanish": False,
                    "note": eval_v2.NULL_RATIO_NOTE, "per_cell": {}},
        "reverse": {"mean_ratio": 1.08, "expected_to_vanish": False,
                    "note": eval_v2.NULL_RATIO_NOTE, "per_cell": {}},
    }


# ---------------------------------------------------------------------------
# final_report_v2 section helpers
# ---------------------------------------------------------------------------
def test_headline_rows_render_the_ordering_verdict() -> None:
    rows = "\n".join(fr._prediction_controls_rows(_pred_block(passing=True)))
    assert "ordering gate (shuffle)" in rows
    assert "pass" in rows and "FAIL" not in rows
    assert "penalty 2" in rows          # 3.0 - 1.0


def test_headline_rows_render_a_failure_loudly() -> None:
    rows = "\n".join(fr._prediction_controls_rows(_pred_block(passing=False)))
    assert "**FAIL**" in rows
    assert "50%" in rows                # ordering_pass_frac = 0.5


def test_headline_rows_degrade_on_a_legacy_metrics_json() -> None:
    rows = "\n".join(fr._prediction_controls_rows({}))
    assert "n/a" in rows


def test_readouts_section_carries_the_f2_note_and_no_arrow_zero() -> None:
    text = "\n".join(fr._readouts_section(_null_block()))
    assert "## Readouts (not gates)" in text
    assert "Finding F2" in text
    assert "near 1.0" in text
    assert "1.05" in text and "1.08" in text
    assert "→ 0" not in text and "-> 0" not in text


def test_readouts_section_degrades_when_absent() -> None:
    text = "\n".join(fr._readouts_section({}))
    assert "n/a" in text


def test_full_markdown_has_no_arrow_zero_caption_for_a_kl_ratio() -> None:
    r"""The whole rendered report must never claim a KL ratio should reach 0."""
    metrics = {
        "run_tag": "unit", "split": "test", "n_samples": 100, "n_cells": 3,
        "ckpt": "final.ckpt",
        "calibration": {"gamma_inj": 0.9, "alpha_inj": 0.05, "r2_inj": 0.98},
        "lag_recovery": {"mean_lag_mass": 0.8, "lag_mass_threshold": 0.7},
        "prediction_controls": _pred_block(),
        "null_controls": _null_block(),
        "frac_phi": {"mean": 1.1, "min": 0.9, "max": 1.3},
        "per_cell": [],
    }
    lines = fr._render_markdown(
        {"experiment": {"tag": "G1_raw_v3"}}, "G1_raw", Path("."), metrics,
        None, None, None, [], split="test")
    text = "\n".join(lines)
    assert "null_ratio" not in text or "→ 0" not in text
    assert "→ 0" not in text
    assert "## Headline gates" in text and "## Readouts (not gates)" in text
    # The training-time perm control never appears in an eval report.
    assert "feat_loss_perm" not in text


# ---------------------------------------------------------------------------
# eval_v2.write_report fallback
# ---------------------------------------------------------------------------
def test_write_report_keeps_the_heading_but_retires_the_caption(tmp_path) -> None:
    r"""``## Null controls`` survives (a test pins it) while its ``-> 0`` caption does not."""
    metrics = {
        "run_tag": "unit", "ckpt": "final.ckpt", "split": "test",
        "n_samples": 10, "n_cells": 2, "warmup": 5, "T": 300, "horizon": 30,
        "calibration": {"gamma_inj": 1.0, "alpha_inj": 0.0, "r2_inj": 1.0,
                        "monotonic_inj": True},
        "lag_recovery": {"mean_lag_mass": 0.9, "lag_mass_threshold": 0.8},
        "prediction_controls": _pred_block(),
        "null_controls": _null_block(),
        "null_probe": {"null_te_scat_mean": -0.5},
        "frac_phi": {"mean": 1.0, "min": 0.9, "max": 1.1},
    }
    text = eval_v2.write_report(metrics, tmp_path).read_text(encoding="utf-8")
    assert "## Null controls" in text                     # the pinned heading
    assert "-> 0)" not in text                            # ...but not the old caption
    assert "Finding F2" in text
    assert "## Prediction controls" in text
    assert "ordering = pass" in text
    assert "feat_loss_perm" not in text


def test_write_report_degrades_without_prediction_controls(tmp_path) -> None:
    metrics = {
        "run_tag": "unit", "ckpt": "c", "split": "test", "n_samples": 1, "n_cells": 1,
        "warmup": 5, "T": 300, "horizon": 30,
        "calibration": {}, "lag_recovery": {}, "null_controls": {},
        "null_probe": {}, "frac_phi": {},
    }
    text = eval_v2.write_report(metrics, tmp_path).read_text(encoding="utf-8")
    assert "## Prediction controls" in text
    assert "n/a (no prediction controls in metrics)" in text


# ---------------------------------------------------------------------------
# S4-T06: three figure-path anchors
# ---------------------------------------------------------------------------
def _build_tree(tmp_path: Path, *, arm: str | None) -> tuple[dict, Path, Path, Path]:
    r"""Lay out a results tree and return ``(config, tag_root, run_root, split_dir)``.

    The data-story gallery is written at the tag root; ``training_curves.html`` at the run
    root; a per-split figure in the split dir. That is exactly what the pipeline produces.
    """
    config = {"experiment": {"tag": "G1_raw_v3"}, "paths": {"results_dir": str(tmp_path)}}
    root = tmp_path / "G1_raw_v3"
    run_root = root / arm if arm else root
    split_dir = run_root / "test"
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "figures" / "raw_preview.pdf").write_bytes(b"%PDF-1.4\n")
    (run_root / "figures").mkdir(parents=True, exist_ok=True)
    (run_root / "figures" / "training_curves.html").write_text("<html/>", encoding="utf-8")
    split_dir.mkdir(parents=True, exist_ok=True)
    return config, root, run_root, split_dir


def test_tag_root_is_resolved_absolutely_not_by_walking_parents(tmp_path) -> None:
    r"""``.parent`` has a different depth per layout; ``tag_root`` must not care."""
    config, root, _, _ = _build_tree(tmp_path, arm="v3_prod")
    assert fr.tag_root(config, "G1_raw") == root


def test_arm_layout_resolves_shared_gallery_and_training_curves(tmp_path) -> None:
    config, root, run_root, split_dir = _build_tree(tmp_path, arm="v3_prod")
    report = fr.final_report_v2(config, benchmark="G1_raw", out_dir=split_dir, split="test",
                                render_headline=False)
    text = report.read_text(encoding="utf-8")

    # The shared data-story gallery is two levels up (``../../figures``), NOT the arm's.
    assert "../../figures" in text
    assert (root / "figures").is_dir()
    # The training curve is one level up (the run root), not in the split dir.
    assert "../figures/training_curves.html" in text
    # Both relative links must resolve to a file/dir that actually exists on disk.
    for rel in ("../../figures", "../figures/training_curves.html"):
        assert (split_dir / rel).resolve().exists(), rel


def test_v1_arm_less_layout_still_resolves(tmp_path) -> None:
    config, root, run_root, split_dir = _build_tree(tmp_path, arm=None)
    assert run_root == root
    report = fr.final_report_v2(config, benchmark="G1_raw", out_dir=split_dir, split="test",
                                render_headline=False)
    text = report.read_text(encoding="utf-8")
    assert "../figures" in text
    assert (split_dir / "../figures").resolve().exists()
    assert (split_dir / "../figures/training_curves.html").resolve().exists()


def test_split_index_names_the_tag_not_the_arm(tmp_path) -> None:
    r"""``tag = results_dir.name`` printed the ARM name as the experiment tag under a v3 config."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    config, root, run_root, split_dir = _build_tree(tmp_path, arm="v3_prod")
    path = rp._write_split_index(run_root, ["test"], config=config, benchmark="G1_raw")
    text = path.read_text(encoding="utf-8")
    assert "`G1_raw_v3`" in text
    assert "`v3_prod`" not in text
    # ..and the data-story gallery link points one level up, out of the arm dir.
    assert "(../figures/)" in text
    assert (run_root / "../figures").resolve().exists()


def test_split_index_arm_less_links_figures_locally(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    config, root, run_root, _ = _build_tree(tmp_path, arm=None)
    text = rp._write_split_index(run_root, ["test"], config=config,
                                 benchmark="G1_raw").read_text(encoding="utf-8")
    assert "(figures/)" in text and "(../figures/)" not in text


# ---------------------------------------------------------------------------
# S8-T01: the cross-arm index
# ---------------------------------------------------------------------------
_ARMS = ("parity", "v3_noncausal", "v3_prod")


def _metrics_json(arm: str) -> Dict[str, Any]:
    return {
        "arm": arm,
        "model_class": "SeqVaeLagAttnV3",
        "calibration": {
            "gamma_inj": 0.5, "gamma_scat": 0.25, "alpha_inj": 0.1, "r2_inj": 0.9,
            "spearman_inj": 0.95,
            "kbar_at_null_cells": {"mean": 0.01, "pass": True},
        },
        "prediction_controls": {
            "overall": {"feat_loss": 1.0, "base_loss": 1.4,
                        "shuffle_penalty_shuffle": 0.3, "ordering_pass": True},
        },
        "null_controls": {"shuffle": {"mean_ratio": 1.04}},
        "lag_recovery": {"mean_lag_mass": 0.82},
        "calibration_predictive": {"nll_mean": 1.30, "crps_mean": 0.52, "coverage_90": 0.91},
    }


def _lag_json() -> Dict[str, Any]:
    return {"overall": {"inband_gate_pass": True},
            "rho_by_band": {"inband": {"rho": 0.71, "ci": [0.2, 0.9], "n_cells": 15,
                                       "gated": True}}}


def _cmi_json() -> Dict[str, Any]:
    return {"overall": {"rho_kbar_cmi_feature_model": {"rho": 0.33},
                        "cmi_bias": {"estimate": -0.06, "reliable": True},
                        "cond_r2_feature_model": {"u": 0.19, "v": 0.07}},
            "recovery": {"spearman_cmi_te_inj": 0.96}}


def _arms_tree(tmp_path: Path, arms=_ARMS, *, split: str = "val",
               omit: tuple = ()) -> Path:
    r"""A ``results/<tag>/<arm>/<split>/`` tree; ``omit`` drops named artifacts everywhere."""
    root = tmp_path / "G1_raw_v3"
    for arm in arms:
        d = root / arm / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.md").write_text("# per-split report", encoding="utf-8")
        payloads = {"metrics.json": _metrics_json(arm),
                    "lag_intervention.json": _lag_json(),
                    "cmi.json": _cmi_json()}
        for name, payload in payloads.items():
            if name in omit:
                continue
            if name == "metrics.json" and "calibration_predictive" in omit:
                payload = {k: v for k, v in payload.items() if k != "calibration_predictive"}
            (d / name).write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_arms_report_header_matches_the_constant(tmp_path) -> None:
    r"""The emitted header IS the column-to-source mapping, not a parallel list that can drift."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    root = _arms_tree(tmp_path)
    text = ar.build_arms_report(_ARMS, root, split="val", tag="G1_raw_v3")
    header = next(ln for ln in text.splitlines() if ln.startswith("| arm |"))
    cells = [c.strip() for c in header.strip("|").split("|")]
    assert cells == ["arm", "model_class"] + [c[0] for c in ar.ARMS_REPORT_COLUMNS]


def test_arms_report_renders_one_two_or_three_arms(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    for n in (1, 2, 3):
        arms = _ARMS[:n]
        root = _arms_tree(tmp_path / f"n{n}", arms)
        text = ar.build_arms_report(arms, root, split="val")
        rows = [ln for ln in text.splitlines() if ln.startswith("| [`")]
        assert len(rows) == n
        for arm in arms:
            assert f"[`{arm}`]" in text


def test_arms_report_resolves_every_column_when_all_sources_exist(tmp_path) -> None:
    r"""No column reads ``n/a`` on a complete tree -- the mapping is live, not aspirational."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    root = _arms_tree(tmp_path)
    text = ar.build_arms_report(_ARMS, root, split="val")
    row = next(ln for ln in text.splitlines() if ln.startswith("| [`parity`]"))
    assert "n/a" not in row, row
    # The derived column: base_loss - feat_loss = 1.4 - 1.0
    assert "0.4" in row
    assert "pass" in row  # the three boolean gate columns


@pytest.mark.parametrize(
    "omit, blanked",
    [
        (("cmi.json",), ["rho_kbar_cmi", "cmi_bias", "cmi_bias_ok",
                         "cond_r2_target_state", "cmi_recovery_rho"]),
        (("lag_intervention.json",), ["inband_gate_pass", "rho_deltaL_attn"]),
        (("calibration_predictive",), ["nll", "crps", "coverage_0.9"]),
        (("metrics.json",), ["gamma_inj", "pred_gain", "ordering_pass", "mean_lag_mass"]),
    ],
)
def test_arms_report_missing_source_is_na_per_column(tmp_path, omit, blanked) -> None:
    r"""Each absent artifact blanks exactly its own columns, and nothing else."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    root = _arms_tree(tmp_path, omit=omit)
    text = ar.build_arms_report(_ARMS, root, split="val")
    header = next(ln for ln in text.splitlines() if ln.startswith("| arm |"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    row = next(ln for ln in text.splitlines() if ln.startswith("| [`v3_prod`]"))
    values = [c.strip() for c in row.strip("|").split("|")]

    for name in blanked:
        assert values[cols.index(name)] == "n/a", f"{name} should be n/a when {omit} is absent"
    # Columns sourced elsewhere still resolve.
    survivors = set(cols[2:]) - set(blanked)
    if "metrics.json" not in omit:
        assert values[cols.index("gamma_inj")] != "n/a"
    for name in survivors & {"rho_deltaL_attn", "cmi_bias"}:
        if _ARTIFACT_OF[name] not in omit:
            assert values[cols.index(name)] != "n/a", name


_ARTIFACT_OF = {"rho_deltaL_attn": "lag_intervention.json", "cmi_bias": "cmi.json"}


def test_arms_report_row_links_to_the_arms_own_report(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    root = _arms_tree(tmp_path)
    text = ar.build_arms_report(_ARMS, root, split="val")
    for arm in _ARMS:
        assert f"[`{arm}`]({arm}/val/report.md)" in text
        assert (root / arm / "val" / "report.md").is_file()


def test_arms_report_degrades_when_no_arm_is_graded(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    root = tmp_path / "G1_raw_v3"
    root.mkdir(parents=True)
    text = ar.build_arms_report(_ARMS, root, split="val")
    assert "n/a" in text and "--stage eval" in text


def test_arms_report_notes_a_partially_graded_ladder(tmp_path) -> None:
    r"""An ungraded arm is named explicitly, not silently rendered as a row of dashes."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    root = _arms_tree(tmp_path, ("parity", "v3_prod"))
    text = ar.build_arms_report(_ARMS, root, split="val")
    assert "`v3_noncausal`: not graded" in text
    row = next(ln for ln in text.splitlines() if ln.startswith("| [`v3_noncausal`]"))
    values = [c.strip() for c in row.strip("|").split("|")]
    assert values[2:] == ["n/a"] * len(ar.ARMS_REPORT_COLUMNS)   # every data column
    assert values[1] == "`n/a`"                                  # model_class too
    # ..while the graded arms keep their numbers.
    assert "0.5" in next(ln for ln in text.splitlines() if ln.startswith("| [`parity`]"))


def test_dig_resolves_paths_and_derived_differences() -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    obj = {"a": {"b": 3.0, "c": 1.25}}
    assert ar._dig(obj, "a.b") == 3.0
    assert ar._dig(obj, "a.b - a.c") == pytest.approx(1.75)
    assert ar._dig(obj, "a.missing") is None
    assert ar._dig(obj, "a.b - a.missing") is None
    assert ar._dig(None, "a.b") is None
    assert ar._dig({"a": 5}, "a.b") is None      # a scalar mid-path, not a dict
    assert ar._dig({"a": [1, 2]}, "a.0") is None  # a list mid-path


def test_a_zero_is_a_value_and_a_false_is_a_verdict() -> None:
    r"""``0.0`` must not render as ``n/a``, and ``False`` must render as a loud failure.

    A gamma of exactly zero -- which is what the pilot arms produce -- is the *finding*. Rendering
    it as "no data" would erase it.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar

    assert ar._dig({"a": {"b": 0.0}}, "a.b") == 0.0
    assert ar._dig({"a": {"b": False}}, "a.b") is False
    assert ar._dig({"a": {"b": None}}, "a.b") is None

    assert ar._render_value("gamma_inj", 0.0) == "0"
    assert ar._render_value("gamma_inj", None) == "n/a"
    assert ar._render_value("null_gate", False) == "**FAIL**"
    assert ar._render_value("null_gate", True) == "pass"
    assert ar._render_value("null_gate", None) == "n/a"
    # Every gate column renders a verdict, never a bare `True`/`False`.
    for column in ar._BOOL_COLUMNS:
        assert ar._render_value(column, True) == "pass"
        assert ar._render_value(column, False) == "**FAIL**"
    assert ar._BOOL_COLUMNS <= {c[0] for c in ar.ARMS_REPORT_COLUMNS}


def test_arms_report_stage_is_registered_model_free_and_non_fatal() -> None:
    r"""Cross-arm and model-free: ``--stage arms_report`` must not require ``--arm``."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    rp._load_stage_plugins()
    spec = rp._STAGE_REGISTRY["arms_report"]
    assert spec.run is ar.run_arms_report_stage
    assert spec.model_dependent is False
    assert spec.fatal is False
    assert "arms_report" in rp.stage_names()


def test_arms_report_is_inert_on_an_arm_less_v1_config(tmp_path) -> None:
    r"""``arms_report`` is ``default_on``, so a v1 run dispatches it. It must do nothing, quietly.

    The v1 / v2 path has no ``arms`` block. The stage logs a warning, writes no file, and returns
    ``0`` -- it must never emit an empty table or take the run down.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    config = {"experiment": {"tag": "G1_raw_v2_notch", "benchmark": "G1_raw"},
              "paths": {"results_dir": str(tmp_path)}}  # no `arms` block
    ctx = rp.StageContext(config=config, benchmark="G1_raw", arm=None, split="val")
    assert ar.run_arms_report_stage(ctx) == 0
    assert not list(tmp_path.rglob("arms_report*.md"))

    # ..and the arm-less run dir is the tag root itself, with no arm level inserted.
    assert rp._run_dir(config, "G1_raw", None) == rp._results_dir(config, "G1_raw")
    assert rp.resolve_arm(config, None) == config


def test_arms_report_stage_writes_at_the_tag_root(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v3 as ar
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    root = _arms_tree(tmp_path)
    config = {
        "experiment": {"tag": "G1_raw_v3", "benchmark": "G1_raw"},
        "paths": {"results_dir": str(tmp_path)},
        "arms": {a: {} for a in _ARMS},
    }
    ctx = rp.StageContext(config=config, benchmark="G1_raw", arm=None, split="val")
    assert ar.run_arms_report_stage(ctx) == 0
    out = root / "arms_report.md"
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Cross-arm report" in text
    assert "G0 alone" in text            # the preamble names the two contrasts
    assert "ordinal within a fixed lag only" in text   # the S1-T05 gamma_scat footnote
