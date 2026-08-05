r"""The offline acceptance gate and the arm tables, tested from files alone.

Everything here drives ``eval/verify.py`` the way an operator does: with a ``summary.json`` and,
for the arm comparison, a directory of finished-run shapes on disk. No model is built anywhere in
this file -- that is the module's one non-negotiable property, and the AST layering test proves
it on the import graph while this file proves the behaviour.

The synthetic summaries are deliberately minimal rather than copies of a real one: each test
states exactly which keys the criterion under test reads, so a summary-schema change that moves
one of them fails here by name instead of surfacing as a wall of ``INCONCLUSIVE``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from teb_vae.lag_attn_rws.eval import verify


# =============================================================================
# The restated constants are pinned to their canonical owners
# =============================================================================
def test_the_restated_names_are_pinned_to_their_canonical_owners():
    """``verify`` restates four names rather than importing them, because their owners pull in
    ``torch``, the logging stack or the trainer. Each restatement is pinned here, so a rename at
    the owner fails a test instead of silently splitting the two."""
    from teb_vae.lag_attn_rws import trainer
    from teb_vae.lag_attn_rws.eval import metrics, report_seam

    assert verify.SUMMARY_FILENAME == report_seam.SUMMARY_FILENAME
    assert verify.RESOLVED_CONFIG_FILENAME == trainer.RESOLVED_CONFIG_FILENAME
    assert verify.RWS_VERDICTS == metrics.PROMOTED_VERDICTS
    # The pred_gap column the gate reads must be a registered headline scalar, and the two CSV
    # series the collapse criterion consumes must be metrics the trainer actually tracks.
    assert verify.PRED_GAP_COLUMN in {name for name, _ in report_seam.HEADLINE_SCALARS}
    assert verify.KL_SERIES_COLUMN in trainer._TRACKED_METRICS
    assert verify.ACTIVE_FRAC_COLUMN in trainer._TRACKED_METRICS


# =============================================================================
# Synthetic summaries
# =============================================================================
def clean_summary(**overrides: Any) -> Dict[str, Any]:
    """A summary every criterion passes on, carrying exactly the keys the gate reads."""
    verdicts = [
        {"name": name, "status": "PASS", "criterion": "c", "detail": "d", "values": {}}
        for name in verify.RWS_VERDICTS
    ]
    summary: Dict[str, Any] = {
        "exit_code": 0,
        "failed": [],
        "checkpoint": None,
        "run_context": {"n_parameters": 3371725, "train_epoch": 412},
        "preflight": {
            "checks": {
                "weights_loaded": {
                    "passed": True,
                    "witnesses_with_evidence": ["delta_heads"],
                    "max_abs_weight": {"delta_heads": 0.1, "film_generators": 0.0},
                }
            }
        },
        "results": {
            "headline": {
                "pred_gap_mc_nats": 1.5,
                "d_base_mc_nats": 700.0,
                "source_conditioned_kl_raw_nats": 2.0,
                "kl_active_dims": 12,
                "source_margin_nats": 3.0,
                "prior_rate_nats": 4.0,
                "mean_logvar_prior": -1.5,
                "logvar_prior_floor_frac": 0.05,
                **{f"verdict_{name}": "PASS" for name in verify.RWS_VERDICTS},
            },
            "verdicts": verdicts,
            "sanity": {
                "checks": {
                    "per_file_counts": {"verdict": "pass", "detail": "all shards contributed"},
                    "headline_finite": {"verdict": "pass", "detail": "every scalar finite"},
                },
                "failed": [],
                "n_inconclusive": 0,
            },
        },
    }
    summary.update(overrides)
    return summary


def test_a_clean_summary_passes_every_criterion():
    report = verify.verify(clean_summary())

    assert report["passed"] is True
    assert report["failed"] == [] and report["inconclusive"] == []
    assert report["n_passed"] == len(verify.CRITERIA)


def test_a_failed_model_verdict_fails_the_gate():
    summary = clean_summary()
    for verdict in summary["results"]["verdicts"]:
        if verdict["name"] == "predictive_improvement":
            verdict["status"] = "FAIL"

    report = verify.verify(summary)

    assert report["passed"] is False
    assert "verdict_predictive_improvement" in report["failed"]


def test_a_failed_step_fails_the_gate():
    report = verify.verify(clean_summary(exit_code=1, failed=["forecast"]))

    assert report["passed"] is False
    assert "exit_code" in report["failed"]


def test_a_failed_sanity_check_fails_the_gate():
    summary = clean_summary()
    summary["results"]["sanity"]["failed"] = ["kl_identity"]

    report = verify.verify(summary)

    assert "sanity_block" in report["failed"]


def test_inconclusive_is_reported_and_never_counted_as_a_pass():
    """An empty summary can satisfy nothing -- and must not read as satisfying anything. Every
    criterion is inconclusive, none is counted passed, and the rendered report says the
    verification is partial."""
    report = verify.verify({})

    assert report["n_passed"] == 0
    assert len(report["inconclusive"]) == len(verify.CRITERIA)
    assert report["failed"] == []
    assert "partial" in verify.format_report(report)


def test_a_skipped_verdict_is_inconclusive_not_passed():
    """The standing case: an ``mse`` checkpoint's calibration verdict is INCONCLUSIVE, and the
    gate must carry that through rather than counting it either way."""
    summary = clean_summary()
    for verdict in summary["results"]["verdicts"]:
        if verdict["name"] == "calibration_near_nominal":
            verdict["status"] = "INCONCLUSIVE"

    report = verify.verify(summary)

    assert "verdict_calibration_near_nominal" in report["inconclusive"]
    assert report["passed"] is True  # no failure -- but the report says it is partial
    assert report["n_passed"] == len(verify.CRITERIA) - 1


def test_the_gate_names_the_pred_gap_column_it_reads():
    """Two ``pred_gap`` columns exist and the gate must say which one it means -- in the report
    record, in the criterion's own detail, and on the console."""
    report = verify.verify(clean_summary())

    assert report["pred_gap_column_read"] == "pred_gap_mc_nats"
    assert "pred_gap_mc_nats" in report["criteria"]["headline_pred_gap"]["detail"]
    assert "pred_gap_mc_nats" in verify.format_report(report)


def test_a_missing_or_skipped_preflight_is_inconclusive():
    summary = clean_summary()
    del summary["preflight"]
    assert verify.check_weights_loaded(summary)["verdict"] == "INCONCLUSIVE"

    skipped = clean_summary(preflight={"skipped": True, "reason": "no model"})
    assert verify.check_weights_loaded(skipped)["verdict"] == "INCONCLUSIVE"


def test_main_returns_the_shell_exit_code_and_writes_the_json_report(tmp_path):
    passing = tmp_path / "summary.json"
    passing.write_text(json.dumps(clean_summary()), encoding="utf-8")
    json_out = tmp_path / "report.json"

    assert verify.main(passing, json_out) == 0
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["passed"] is True and written["summary_path"] == str(passing)

    failing = tmp_path / "failing.json"
    failing.write_text(json.dumps(clean_summary(exit_code=1)), encoding="utf-8")
    assert verify.main(failing) == 1


def test_the_gate_mirrors_a_real_runs_recorded_verdicts(evaluated):
    """Against the session's real run: every verdict criterion's outcome is the recorded status,
    verbatim -- the gate re-reads the model's own criteria rather than re-deriving them, so a
    disagreement here means the gate is reading the wrong paths."""
    summary = evaluated["summary"]

    report = verify.verify(summary)

    recorded = {v["name"]: v["status"] for v in summary["results"]["verdicts"]}
    for name in verify.RWS_VERDICTS:
        assert report["criteria"][f"verdict_{name}"]["verdict"] == recorded[name]
    # And the structural criteria all resolved on a real summary -- nothing INCONCLUSIVE among
    # them means every path the gate reads exists in the artifact a run actually writes.
    for name in ("exit_code", "per_file_counts", "weights_loaded", "headline_pred_gap",
                 "headline_finite", "sanity_block"):
        assert report["criteria"][name]["verdict"] != "INCONCLUSIVE", name


# =============================================================================
# The arm comparison
# =============================================================================
#: Eight epochs of healthy validation series: the KL well above the collapse threshold.
_HEALTHY_KL = [0.0, 0.5, 2.0, 4.0, 5.0, 5.5, 5.2, 5.4]
_HEALTHY_ACTIVE = [0.0, 0.1, 0.4, 0.5, 0.55, 0.6, 0.6, 0.58]

#: The same length, dead at the tail: below 0.02 nats at every one of the last five epochs.
_COLLAPSED_KL = [0.0, 0.5, 1.0, 0.01, 0.005, 0.003, 0.002, 0.001]
_COLLAPSED_ACTIVE = [0.0, 0.1, 0.2, 0.02, 0.01, 0.01, 0.01, 0.01]


def write_arm(
    root: Path,
    name: str,
    *,
    beta: float,
    beta_prior: float = 1.0e-2,
    d_z: int = 48,
    reach: Optional[Any] = None,
    kept: Optional[Tuple[int, int]] = None,
    pred_gap: float = 1.0,
    kl_series: Optional[List[float]] = None,
    active_series: Optional[List[float]] = None,
    with_csv: bool = True,
) -> Path:
    """Write one finished-run shape: summary, resolved config, and the training CSV."""
    results_dir = root / name / "eval_results"
    results_dir.mkdir(parents=True)
    train_root = root / name / "train_run"
    checkpoint = train_root / "model_checkpoints" / "lag-attn-rws-epoch=07.ckpt"

    summary = clean_summary(checkpoint=str(checkpoint))
    summary["results"]["headline"]["pred_gap_mc_nats"] = pred_gap
    (results_dir / verify.SUMMARY_FILENAME).write_text(
        json.dumps(summary), encoding="utf-8"
    )

    config = {
        "model_config": {
            "VAE_model": {
                "beta_schedule": {"kind": "linear_warmup", "start": 0.0, "end": beta},
                "beta_prior": beta_prior,
                "d_z": d_z,
                "causal_reach_budget_s": reach,
                "c_y": 109,
                "c_u": 58,
                "encoder_extra_kernel": 15,
                "query_uses_logvar": False,
                "horizon_depth": 3,
                "horizon_embed_std": 0.8,
                "head_init_calibration": True,
                "a_head_gain": 2.0,
            }
        }
    }
    if kept is not None:
        # What the trainer records beside the config: the channels that survived the budget, as
        # against the shard widths `c_y`/`c_u` above, which the budget never changes.
        config["model_config"]["resolved_causal_budget"] = {
            "target_channels_kept": kept[0],
            "source_channels_kept": kept[1],
        }
    (results_dir / verify.RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    if with_csv:
        kl = list(_HEALTHY_KL if kl_series is None else kl_series)
        active = list(_HEALTHY_ACTIVE if active_series is None else active_series)
        csv_dir = train_root / "train_results"
        csv_dir.mkdir(parents=True)
        lines = [f"{verify.EPOCH_COLUMN},{verify.KL_SERIES_COLUMN},{verify.ACTIVE_FRAC_COLUMN}"]
        lines += [f"{epoch},{kl[epoch]},{active[epoch]}" for epoch in range(len(kl))]
        (csv_dir / verify.METRICS_HISTORY_FILENAME).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return results_dir


def _section(document: str, heading: str) -> str:
    """One ``##`` section of the emitted markdown, up to the next heading."""
    start = document.index(heading)
    end = document.find("\n## ", start + 1)
    return document[start:] if end < 0 else document[start:end]


def test_three_arms_key_the_beta_table_by_the_swept_value_not_the_directory(tmp_path):
    """The directory names here are deliberate lies -- each names a *different* beta than its
    config carries -- so a row keyed off the name would be caught immediately."""
    write_arm(tmp_path, "beta_9p9", beta=0.1)
    write_arm(tmp_path, "beta_0p1", beta=1.0)
    write_arm(tmp_path, "some_run", beta=0.3)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## KL-weight sweep")
    rows = [line for line in section.splitlines() if line.startswith("| ")][1:]

    keys = [row.split("|")[1].strip() for row in rows if not row.startswith("|---")]
    assert keys == ["0.1", "0.3", "1"], keys
    # And the misleadingly named directories appear only as identification, in value order.
    assert "beta_9p9" in rows[0] and "some_run" in rows[1] and "beta_0p1" in rows[2]


def test_the_four_prior_anchor_arms_resolve_into_their_own_section(tmp_path):
    """Without a section keyed on ``beta_prior`` the swept arms appear in no generated table and
    the study becomes hand transcription -- which is what every other sweep here has a section to
    avoid. The four are the shipped bracket, over three orders of magnitude."""
    for name, weight in (
        ("bp_0p001", 1.0e-3), ("bp_0p01", 1.0e-2), ("bp_0p1", 0.1), ("bp_1p0", 1.0),
    ):
        write_arm(tmp_path, name, beta=1.0, beta_prior=weight)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Prior-anchor weight sweep")
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    keys = [row.split("|")[1].strip() for row in rows[1:] if not row.startswith("|---")]

    assert keys == ["0.001", "0.01", "0.1", "1"], keys
    # The columns the study is read down, in the order it is read in: the floor first, the base
    # forecast next, the coupling columns last.
    header = rows[0]
    for column in (
        "`beta_prior`", "`logvar_prior_floor_frac`", "`mean_logvar_prior`", "`prior_rate`",
        "`d_base_mc_nats`", "`pred_gap`", "`abs(pred_gap)/K`", "`source_margin`", "Verdicts",
    ):
        assert column in header, column


def test_every_generated_table_is_well_formed_markdown(tmp_path):
    """Every table's header, delimiter and body rows must agree on their cell count.

    A header *label* that contains an unescaped ``|`` splits into extra cells, and a table whose
    delimiter row no longer matches its header is not a table at all under GitHub-flavoured
    markdown -- the whole section degrades to a paragraph of pipes. The failure is invisible from
    the code, invisible to a per-label ``in header`` assertion (the substring is still there), and
    surfaces only when somebody opens the document the multi-day arms are read from. So the check
    is structural and runs over the whole emitted document rather than over one section.
    """
    write_arm(tmp_path, "low", beta=0.1, beta_prior=1.0e-3, d_z=24, reach=60, kept=(59, 23))
    write_arm(tmp_path, "high", beta=1.0, beta_prior=1.0, d_z=64, pred_gap=-4.0)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0

    def cells(line: str) -> int:
        """Cell count of one markdown row, by the same split a renderer applies."""
        return len(line.strip().strip("|").split("|"))

    lines = out.read_text(encoding="utf-8").splitlines()
    tables = 0
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("|") or not set(lines[index + 1].strip()) <= set("|-: "):
            continue
        if not lines[index + 1].startswith("|"):
            continue
        tables += 1
        width = cells(line)
        assert cells(lines[index + 1]) == width, (
            f"line {index + 1}: header has {width} cells, delimiter has "
            f"{cells(lines[index + 1])} -- this section will not render as a table:\n{line}"
        )
        for body in lines[index + 2 :]:
            if not body.startswith("|"):
                break
            assert cells(body) == width, f"line {index + 1}: row width {cells(body)} != {width}"

    # A guard on the guard: a scan that matched no tables would pass on a broken document.
    assert tables >= 5, f"only {tables} tables found; the scan is not reaching the sections"


def test_the_amplification_ratio_carries_the_sign_of_the_gap(tmp_path):
    """The ratio is |pred_gap| / K, and the same magnitude means opposite things either side of
    zero: nats of forecast degradation per nat of rate where the gap is negative, nats of gain
    where it is positive. A bare number would invite exactly the comparison it must not be read
    as, so the cell says which."""
    write_arm(tmp_path, "helping", beta=1.0, beta_prior=0.1, pred_gap=4.0)
    write_arm(tmp_path, "hurting", beta=1.0, beta_prior=1.0, pred_gap=-4.0)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Prior-anchor weight sweep")

    helping = next(line for line in section.splitlines() if "helping" in line)
    hurting = next(line for line in section.splitlines() if "hurting" in line)
    # K is 2.0 in the fixture summary, so both are 2 -- and they are not the same finding.
    assert "2 (gain)" in helping
    assert "2 (cost)" in hurting


def test_the_verdict_triple_is_rendered_as_one_cell(tmp_path):
    """The three predictive verdicts together, because FAIL / PASS / FAIL is one state -- no
    predictive gain, a positive source margin, and therefore no specificity -- and three separate
    columns invite reading the first alone."""
    results_dir = write_arm(tmp_path, "mixed", beta=1.0, beta_prior=0.1)
    summary = json.loads((results_dir / verify.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    summary["results"]["headline"].update({
        "verdict_predictive_improvement": "FAIL",
        "verdict_source_margin_positive": "PASS",
        "verdict_source_specificity": "FAIL",
    })
    (results_dir / verify.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Prior-anchor weight sweep")

    assert "FAIL / PASS / FAIL" in section


def test_a_collapsed_arm_is_marked_never_omitted(tmp_path):
    write_arm(tmp_path, "healthy", beta=0.1)
    write_arm(
        tmp_path, "dead", beta=0.3,
        kl_series=_COLLAPSED_KL, active_series=_COLLAPSED_ACTIVE,
    )
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## KL-weight sweep")

    dead_row = next(line for line in section.splitlines() if "dead" in line)
    assert "**collapsed**" in dead_row
    healthy_row = next(line for line in section.splitlines() if "healthy" in line)
    assert "**collapsed**" not in healthy_row and "| no |" in healthy_row


def test_an_arm_missing_its_metrics_csv_is_reported_incomplete_not_rowless(tmp_path):
    write_arm(tmp_path, "complete", beta=0.1)
    write_arm(tmp_path, "no_csv", beta=0.3, with_csv=False)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")

    # The row is present, keyed by its swept value, with the gap number it does have...
    section = _section(document, "## KL-weight sweep")
    row = next(line for line in section.splitlines() if "no_csv" in line)
    assert row.split("|")[1].strip() == "0.3"
    # ...and what could not be read is named, in the row and in the incomplete list.
    assert "incomplete" in row
    assert "## Incomplete runs" in document
    assert verify.METRICS_HISTORY_FILENAME in _section(document, "## Incomplete runs")


def test_the_reach_table_reads_the_kept_channel_widths_from_the_config(tmp_path):
    write_arm(tmp_path, "unguarded", beta=1.0, reach=None)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Reach budget sweep")

    row = next(line for line in section.splitlines() if "unguarded" in line)
    assert row.split("|")[1].strip() == "null"  # explicit null, not "(absent)"
    assert "109 / 58" in row


def test_the_reach_table_reports_the_surviving_channels_not_the_declared_widths(tmp_path):
    """The column is the sweep's whole subject, and `c_y`/`c_u` cannot express it.

    Two startup guards force the declared widths to equal the shard's, for every arm that can
    train -- so rendering them prints ``109 / 58`` in every row and tells the reader the reach
    budget pruned nothing. The surviving counts are what the trainer resolved and recorded.
    """
    write_arm(tmp_path, "reach_null", beta=1.0, reach=None)
    write_arm(tmp_path, "reach_120", beta=1.0, reach=120, kept=(78, 29))
    write_arm(tmp_path, "reach_240", beta=1.0, reach=240, kept=(94, 43))
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Reach budget sweep")
    cells = {
        line.split("|")[1].strip(): line.split("|")[2].strip()
        for line in section.splitlines()
        if line.startswith("|") and line.split("|")[1].strip() in {"null", "120", "240"}
    }

    assert cells["120"] == "78 / 29"
    assert cells["240"] == "94 / 43"
    # The unguarded arm has no record to read, and there the declared widths are the truth.
    assert cells["null"] == "109 / 58"
    assert len(set(cells.values())) == 3, "a constant column is the defect this pins"


def test_an_empty_directory_is_an_error_not_an_empty_document(tmp_path):
    assert verify.compare_arms(tmp_path, tmp_path / "arms.md") == 1
    assert not (tmp_path / "arms.md").exists()


def test_the_cli_dispatches_between_the_gate_and_the_tables(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(clean_summary()), encoding="utf-8")

    assert verify._cli([str(summary)]) == 0
    assert verify._cli([str(summary), "--runs", str(tmp_path)]) == 2  # both is a usage error
    assert verify._cli([]) == 2  # neither is too


def test_the_emitted_document_never_uses_the_refused_name(tmp_path):
    """The arm tables are an artifact, and the naming rule that binds every run artifact binds
    them too: the coupling readout is not called a transfer entropy anywhere in the output."""
    write_arm(tmp_path, "arm", beta=1.0)
    out = tmp_path / "arms.md"
    verify.compare_arms(tmp_path, out)

    lowered = out.read_text(encoding="utf-8").lower()
    assert "transfer entropy" not in lowered and "te_lag" not in lowered
