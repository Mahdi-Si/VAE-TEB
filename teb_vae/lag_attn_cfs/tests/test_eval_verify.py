r"""The offline acceptance gate, the arm tables and the cross-cell table, tested from files alone.

Everything here drives ``eval/verify.py`` the way an operator does: with a ``summary.json`` and,
for the tables, a directory of finished-run shapes on disk. No model is built anywhere in this
file except in the two pinning tests at the top, which reach for the registries this module
restates -- that is the module's one non-negotiable property, and the AST layering test proves it
on the import graph while this file proves the behaviour.

The synthetic summaries are deliberately minimal rather than copies of a real one: each test
states exactly which keys the criterion under test reads, so a summary-schema change that moves
one of them fails here by name instead of surfacing as a wall of ``INCONCLUSIVE``.

**No test here asserts a direction or a magnitude.** The numbers in the synthetic summaries are
arbitrary shapes chosen to distinguish two renderings; what is asserted is which key was read,
which cell it landed in, and what the module does when it is missing.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest
import yaml

from teb_vae.lag_attn_cfs.eval import verify
from teb_vae.lag_attn_transformer_cfs.eval import verify as trf_verify


# =================================================================================================
# The restated constants are pinned to their canonical owners
# =================================================================================================
def test_the_restated_names_are_pinned_to_their_canonical_owners() -> None:
    """``verify`` restates these names rather than importing them, because their owners pull in
    ``torch``, the logging stack or the trainer. Each restatement is pinned here, so a rename at
    the owner fails a test instead of silently splitting the two."""
    from teb_vae.lag_attn_cfs.eval import metrics, report_seam
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

    assert verify.SUMMARY_FILENAME == report_seam.SUMMARY_FILENAME
    assert verify.RESOLVED_CONFIG_FILENAME == RESOLVED_CONFIG_FILENAME
    assert verify.CFS_VERDICTS == metrics.PROMOTED_VERDICTS
    # Ten rather than the raw pipeline's eight, and the two extra are the ones only this cell can
    # have. Named rather than counted: a registry that lost one and gained another would keep the
    # count.
    assert verify.CFS_VERDICTS[-2:] == (
        "coupling_exceeds_availability_clock",
        "anchor_geometry_intact",
    )
    # The two CSV series the collapse criterion consumes must be metrics the trainer tracks.
    assert verify.KL_SERIES_COLUMN in LagAttnCfsTrainer.TRACKED_METRICS
    assert verify.ACTIVE_FRAC_COLUMN in LagAttnCfsTrainer.TRACKED_METRICS


def _headline_columns_read_by(module: Any) -> Set[str]:
    """Every column name a module hands to ``_headline_cell``, read off its source.

    Derived rather than listed: a hand-kept list is one that goes stale the first time a column is
    added, and the column that goes unchecked is then exactly the new one.

    Args:
        module: The module to scan.

    Returns:
        The resolved column names. A module-level constant is resolved through the module; a
        literal is taken as written.
    """
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    names: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
            node.func, "id", ""
        )
        if called != "_headline_cell" or len(node.args) < 2:
            continue
        column = node.args[1]
        if isinstance(column, ast.Constant):
            names.add(str(column.value))
        elif isinstance(column, ast.Name):
            names.add(str(getattr(module, column.id)))
        elif isinstance(column, ast.Attribute):
            names.add(str(getattr(module, column.attr)))
    return names


@pytest.mark.parametrize(
    "module",
    [verify, trf_verify],
    ids=["cfs", "transformer_cfs"],
)
def test_every_numeric_cell_comes_from_the_headline_block(module: Any) -> None:
    """The headline block is the one surface the reporting layer promises to keep resolvable, so a
    column reading anything else is a permanently ``(missing)`` cell -- and no test on synthetic
    summaries would catch it, because the synthetic ones carry whatever this file puts in them.

    The scan is over the source rather than over a list, so a column added tomorrow is checked
    tomorrow. Both cells' table modules, because the second one renders its own sweep section.
    """
    from teb_vae.lag_attn_cfs.eval.binding import HEADLINE_SCALARS as EXTRA_SCALARS
    from teb_vae.lag_attn_cfs.eval.report_seam import HEADLINE_SCALARS as SHARED_SCALARS

    registered = {name for name, _ in SHARED_SCALARS} | {name for name, _ in EXTRA_SCALARS}
    read = _headline_columns_read_by(module)

    assert read, "the scan found no headline column at all, so it is checking nothing"
    assert read <= registered, sorted(read - registered)


def test_the_cfs_only_columns_come_through_the_binding_rather_than_the_shared_registry() -> None:
    """Where each column is registered is not a detail: every path in the shared tuple must
    resolve on a run of *every* model that uses this pipeline, so a cfs-only entry there would
    read as a number the other cells failed to produce."""
    from teb_vae.lag_attn_cfs.eval.binding import HEADLINE_SCALARS as EXTRA_SCALARS
    from teb_vae.lag_attn_cfs.eval.report_seam import HEADLINE_SCALARS as SHARED_SCALARS

    cfs_only = {
        verify.CLOCK_MARGIN_COLUMN,
        verify.ANCHORS_PER_SAMPLE_COLUMN,
        verify.TARGET_WARM_FRAC_COLUMN,
    }
    shared = {
        verify.PRED_GAP_COLUMN,
        verify.D_BASE_COLUMN,
        verify.D_FULL_COLUMN,
        verify.KL_COLUMN,
        verify.ACTIVE_DIMS_COLUMN,
        verify.LAG_PEAK_COLUMN,
    }

    assert cfs_only <= {name for name, _ in EXTRA_SCALARS}
    assert cfs_only & {name for name, _ in SHARED_SCALARS} == set()
    assert shared <= {name for name, _ in SHARED_SCALARS}


def test_the_two_model_class_strings_are_pinned_to_the_two_bindings() -> None:
    """Restated as strings because the bindings name ``torch`` modules and the gate may import
    neither. A class rename must fail here rather than silently emptying the cross-cell table."""
    from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
    from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

    assert verify.BASELINE_MODEL_CLASS == CFS_BINDING.model_cls.__name__
    assert verify.COMPARISON_MODEL_CLASS == TRF_CFS_BINDING.model_cls.__name__


# =================================================================================================
# Synthetic summaries
# =================================================================================================
def clean_summary(**overrides: Any) -> Dict[str, Any]:
    """A summary every criterion passes on, carrying exactly the keys the gate reads."""
    verdicts = [
        {"name": name, "status": "PASS", "criterion": "c", "detail": "d", "values": {}}
        for name in verify.CFS_VERDICTS
    ]
    summary: Dict[str, Any] = {
        "exit_code": 0,
        "failed": [],
        "checkpoint": None,
        "run_context": {
            "n_parameters": 5143262,
            "train_epoch": 12,
            "model_class": verify.BASELINE_MODEL_CLASS,
        },
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
                "pred_gap_rmse_pct": 0.4,
                "pred_gap_mc_likelihood_pct": 0.1,
                "d_base_mc_nats": 2100.0,
                "d_full_mc_nats": 2098.5,
                "source_conditioned_kl_raw_nats": 2.0,
                "kl_active_dims": 12,
                "kl_argmax_lag_step": 7,
                "coupling_minus_clock_nats": 0.75,
                "anchors_per_sample": 152,
                "target_warm_frac": 1.0,
                **{f"verdict_{name}": "PASS" for name in verify.CFS_VERDICTS},
            },
            "verdicts": verdicts,
            "sanity": {
                "checks": {
                    "per_file_counts": {"verdict": "pass", "detail": "all shards contributed"},
                    "headline_finite": {"verdict": "pass", "detail": "every scalar finite"},
                    "argmax_lag": {
                        "verdict": "pass",
                        "detail": "the KL attribution peaks strictly inside the attainable window",
                    },
                },
                "failed": [],
                "n_inconclusive": 0,
            },
        },
    }
    summary.update(overrides)
    return summary


# =================================================================================================
# The gate
# =================================================================================================
def test_a_clean_summary_passes_every_criterion() -> None:
    report = verify.verify(clean_summary())

    assert report["passed"] is True
    assert report["failed"] == [] and report["inconclusive"] == []
    assert report["n_passed"] == len(verify.CRITERIA)


def test_the_two_cfs_only_verdicts_are_criteria_of_this_gate() -> None:
    """The gate exists to refuse a run whose verdicts failed, and these two are the verdicts this
    cell alone can have -- a gate that dropped them would pass a run whose coupling readout was
    indistinguishable from an availability clock."""
    for name in ("coupling_exceeds_availability_clock", "anchor_geometry_intact"):
        assert f"verdict_{name}" in dict(verify.CRITERIA)


def test_a_failed_model_verdict_fails_the_gate() -> None:
    summary = clean_summary()
    for verdict in summary["results"]["verdicts"]:
        if verdict["name"] == "anchor_geometry_intact":
            verdict["status"] = "FAIL"

    report = verify.verify(summary)

    assert report["passed"] is False
    assert "verdict_anchor_geometry_intact" in report["failed"]


def test_a_failed_step_fails_the_gate() -> None:
    report = verify.verify(clean_summary(exit_code=1, failed=["forecast"]))

    assert report["passed"] is False
    assert "exit_code" in report["failed"]


def test_a_failed_sanity_check_fails_the_gate() -> None:
    """The asymmetry the gate exists for: the run's own exit code deliberately does not move on a
    failed sanity check, so a run that exited 0 is refused here."""
    summary = clean_summary()
    summary["results"]["sanity"]["failed"] = ["kl_identity"]

    report = verify.verify(summary)

    assert summary["exit_code"] == 0
    assert "sanity_block" in report["failed"]
    assert report["passed"] is False


def test_inconclusive_is_reported_and_never_counted_as_a_pass() -> None:
    """An empty summary can satisfy nothing -- and must not read as satisfying anything. Every
    criterion is inconclusive, none is counted passed, and the rendered report says the
    verification is partial."""
    report = verify.verify({})

    assert report["n_passed"] == 0
    assert len(report["inconclusive"]) == len(verify.CRITERIA)
    assert report["failed"] == []
    assert "partial" in verify.format_report(report)


def test_the_unset_clock_threshold_is_inconclusive_not_passed() -> None:
    """This cell's standing case rather than an edge case. ``clock_margin_min_nats`` ships unset,
    so the availability-clock verdict is INCONCLUSIVE on every run until somebody sets it -- and
    the gate must carry that through rather than counting it either way."""
    summary = clean_summary()
    for verdict in summary["results"]["verdicts"]:
        if verdict["name"] == "coupling_exceeds_availability_clock":
            verdict["status"] = "INCONCLUSIVE"

    report = verify.verify(summary)

    assert "verdict_coupling_exceeds_availability_clock" in report["inconclusive"]
    assert report["passed"] is True  # no failure -- but the report says it is partial
    assert report["n_passed"] == len(verify.CRITERIA) - 1


def test_the_gate_names_the_pred_gap_column_it_reads() -> None:
    """Two ``pred_gap`` columns exist and the gate must say which one it means -- in the report
    record, in the criterion's own detail, and on the console."""
    report = verify.verify(clean_summary())

    assert report["pred_gap_column_read"] == "pred_gap_mc_nats"
    assert "pred_gap_mc_nats" in report["criteria"]["headline_pred_gap"]["detail"]
    assert "pred_gap_mc_nats" in verify.format_report(report)


def test_a_missing_or_skipped_preflight_is_inconclusive() -> None:
    summary = clean_summary()
    del summary["preflight"]
    assert verify.check_weights_loaded(summary)["verdict"] == "INCONCLUSIVE"

    skipped = clean_summary(preflight={"skipped": True, "reason": "no model"})
    assert verify.check_weights_loaded(skipped)["verdict"] == "INCONCLUSIVE"


def test_main_returns_the_shell_exit_code_and_writes_the_json_report(tmp_path) -> None:
    passing = tmp_path / "summary.json"
    passing.write_text(json.dumps(clean_summary()), encoding="utf-8")
    json_out = tmp_path / "report.json"

    assert verify.main(passing, json_out) == 0
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["passed"] is True and written["summary_path"] == str(passing)

    failing = tmp_path / "failing.json"
    failing.write_text(json.dumps(clean_summary(exit_code=1)), encoding="utf-8")
    assert verify.main(failing) == 1


@pytest.mark.slow
def test_the_gate_mirrors_a_real_runs_recorded_verdicts(collected_run) -> None:
    """Against the session's real run: every verdict criterion's outcome is the recorded status,
    verbatim -- the gate re-reads the model's own criteria rather than re-deriving them, so a
    disagreement here means the gate is reading the wrong paths."""
    summary = collected_run["summary"]

    report = verify.verify(summary)

    recorded = {entry["name"]: entry["status"] for entry in summary["results"]["verdicts"]}
    for name in verify.CFS_VERDICTS:
        assert report["criteria"][f"verdict_{name}"]["verdict"] == recorded[name]
    # And the structural criteria all resolved on a real summary -- nothing INCONCLUSIVE among
    # them means every path the gate reads exists in the artifact a run actually writes.
    for name in ("exit_code", "per_file_counts", "weights_loaded", "headline_pred_gap",
                 "headline_finite", "sanity_block"):
        assert report["criteria"][name]["verdict"] != "INCONCLUSIVE", name


# =================================================================================================
# The arm tables
# =================================================================================================
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
    anchor_stride: int = 15,
    warmup_period: int = 133,
    horizon: int = 15,
    horizon_depth: int = 4,
    d_z: int = 64,
    model_class: Optional[str] = verify.BASELINE_MODEL_CLASS,
    d_base: float = 2100.0,
    pred_gap: float = 1.0,
    kl_series: Optional[List[float]] = None,
    active_series: Optional[List[float]] = None,
    with_csv: bool = True,
    csv_columns: Optional[List[str]] = None,
) -> Path:
    """Write one finished-run shape: summary, resolved config, and the training CSV.

    Args:
        root: Directory the run shapes are written under.
        name: The run's directory name -- identification only; nothing keys off it.
        anchor_stride: The tiling arm's swept value.
        warmup_period: The floor arm's swept value.
        horizon: The horizon arm's swept value.
        horizon_depth: The decoder-depth arm's swept value.
        d_z: The latent width the collapse criterion's second clause is applied against.
        model_class: What ``run_context`` records; ``None`` writes a run that recorded none.
        d_base: The base branch's block score, which the cross-cell selection rule reads.
        pred_gap: The headline coupling readout.
        kl_series: The per-epoch KL column. ``None`` writes the healthy series.
        active_series: The per-epoch active fraction. ``None`` writes the healthy series.
        with_csv: Whether to write the training CSV at all.
        csv_columns: Which of the three columns to write. ``None`` writes all three.

    Returns:
        The run's ``eval_results`` directory.
    """
    results_dir = root / name / "eval_results"
    results_dir.mkdir(parents=True)
    train_root = root / name / "train_run"
    checkpoint = train_root / "model_checkpoints" / "lag-attn-cfs-epoch=07.ckpt"

    summary = clean_summary(checkpoint=str(checkpoint))
    summary["run_context"]["model_class"] = model_class
    summary["results"]["headline"]["pred_gap_mc_nats"] = pred_gap
    summary["results"]["headline"]["d_base_mc_nats"] = d_base
    (results_dir / verify.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")

    config = {
        "model_config": {
            "VAE_model": {
                "anchor_stride": anchor_stride,
                "warmup_period": warmup_period,
                "horizon": horizon,
                "horizon_depth": horizon_depth,
                "d_z": d_z,
                "c_y": 102,
                "c_u": 51,
                "causal_warmup_budget_steps": 134,
            }
        }
    }
    (results_dir / verify.RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    if with_csv:
        kl = list(_HEALTHY_KL if kl_series is None else kl_series)
        active = list(_HEALTHY_ACTIVE if active_series is None else active_series)
        columns = (
            [verify.EPOCH_COLUMN, verify.KL_SERIES_COLUMN, verify.ACTIVE_FRAC_COLUMN]
            if csv_columns is None else list(csv_columns)
        )
        by_column = {
            verify.EPOCH_COLUMN: [float(index) for index in range(len(kl))],
            verify.KL_SERIES_COLUMN: kl,
            verify.ACTIVE_FRAC_COLUMN: active,
        }
        csv_dir = train_root / "train_results"
        csv_dir.mkdir(parents=True)
        lines = [",".join(columns)]
        lines += [
            ",".join(str(by_column[column][index]) for column in columns)
            for index in range(len(kl))
        ]
        (csv_dir / verify.METRICS_HISTORY_FILENAME).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return results_dir


def _section(document: str, heading: str) -> str:
    """One ``##`` section of the emitted markdown, up to the next heading."""
    start = document.index(heading)
    end = document.find("\n## ", start + 1)
    return document[start:] if end < 0 else document[start:end]


def test_three_arms_key_the_stride_table_by_the_swept_value_not_the_directory(tmp_path) -> None:
    """The directory names here are deliberate lies -- each names a *different* stride than its
    config carries -- so a row keyed off the name would be caught immediately."""
    write_arm(tmp_path, "stride_30", anchor_stride=1)
    write_arm(tmp_path, "stride_1", anchor_stride=15)
    write_arm(tmp_path, "some_run", anchor_stride=5)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Anchor tiling sweep")
    rows = [line for line in section.splitlines() if line.startswith("| ")][1:]

    keys = [row.split("|")[1].strip() for row in rows if not row.startswith("|---")]
    assert keys == ["1", "5", "15"], keys
    # And the misleadingly named directories appear only as identification, in value order.
    assert "stride_30" in rows[0] and "some_run" in rows[1] and "stride_1" in rows[2]


@pytest.mark.parametrize(
    "heading, axis, values",
    [
        ("## Anchor tiling sweep", "anchor_stride", (1, 15)),
        ("## Anchor floor sweep", "warmup_period", (133, 150)),
        ("## Horizon sweep", "horizon", (15, 30)),
        ("## Decoder depth sweep", "horizon_depth", (3, 4)),
    ],
)
def test_each_of_the_four_shipped_arms_resolves_into_its_own_section(
    tmp_path, heading: str, axis: str, values
) -> None:
    """The four shipped ``sweep_*.yaml`` arms, one section each. Without a section keyed on its
    axis a swept arm appears in no generated table and the study becomes hand transcription."""
    for index, value in enumerate(values):
        write_arm(tmp_path, f"arm_{index}", **{axis: value})
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), heading)
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    keys = [row.split("|")[1].strip() for row in rows[1:] if not row.startswith("|---")]

    assert keys == [str(value) for value in sorted(values)], keys
    assert f"`{axis}`" in rows[0]


def test_the_horizon_section_refuses_a_level_comparison_in_the_document(tmp_path) -> None:
    """The rule is emitted rather than only stated in the docstring, because the comparison it
    forbids is the one a reader makes by reflex: a block score is per anchor over H*C_keep
    coefficients, so twice the horizon is twice the block and larger nats for that reason alone."""
    write_arm(tmp_path, "h15", horizon=15)
    write_arm(tmp_path, "h30", horizon=30)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Horizon sweep")

    assert verify.HORIZON_LEVEL_RULE in section
    # And the two level columns are labelled in the header itself, so a reader who skipped the
    # paragraph still meets the refusal at the column they were about to read.
    header = next(line for line in section.splitlines() if line.startswith("| `horizon`"))
    assert header.count("not comparable") == 2
    # The scale-free columns are what the axis *is* readable on, so they have to be there.
    assert "`pred_gap_rmse_pct`" in header


def test_every_generated_table_is_well_formed_markdown(tmp_path) -> None:
    """Every table's header, delimiter and body rows must agree on their cell count.

    A header *label* that contains an unescaped ``|`` splits into extra cells, and a table whose
    delimiter row no longer matches its header is not a table at all under GitHub-flavoured
    markdown -- the whole section degrades to a paragraph of pipes. The failure is invisible from
    the code, invisible to a per-label ``in header`` assertion (the substring is still there), and
    surfaces only when somebody opens the document the multi-day arms are read from. So the check
    is structural and runs over the whole emitted document rather than over one section.
    """
    write_arm(tmp_path, "base", anchor_stride=15)
    write_arm(tmp_path, "trf", model_class=verify.COMPARISON_MODEL_CLASS, pred_gap=-4.0)
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
        for body in lines[index + 2:]:
            if not body.startswith("|"):
                break
            assert cells(body) == width, f"line {index + 1}: row width {cells(body)} != {width}"

    # A guard on the guard: a scan that matched no tables would pass on a broken document.
    assert tables >= 6, f"only {tables} tables found; the scan is not reaching the sections"


def test_a_collapsed_arm_is_marked_never_omitted(tmp_path) -> None:
    write_arm(tmp_path, "healthy", anchor_stride=15)
    write_arm(
        tmp_path, "dead", anchor_stride=1,
        kl_series=_COLLAPSED_KL, active_series=_COLLAPSED_ACTIVE,
    )
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Anchor tiling sweep")

    dead_row = next(line for line in section.splitlines() if "dead" in line)
    assert "**collapsed**" in dead_row
    healthy_row = next(line for line in section.splitlines() if "healthy" in line)
    assert "**collapsed**" not in healthy_row and "| no |" in healthy_row


def test_a_run_whose_active_fraction_series_is_absent_is_unknown_not_healthy(tmp_path) -> None:
    """The criterion has two clauses and the second reads the final active fraction. A CSV
    carrying only the KL column can answer with clause 1 alone -- and a one-clause answer rendered
    as "no" is a verdict the run did not support, in the cell an operator scans a sweep table
    down."""
    write_arm(
        tmp_path, "no_active", anchor_stride=15,
        csv_columns=[verify.EPOCH_COLUMN, verify.KL_SERIES_COLUMN],
    )
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")
    row = next(
        line for line in _section(document, "## Anchor tiling sweep").splitlines()
        if "no_active" in line
    )

    assert "unknown" in row
    assert "| no |" not in row
    assert verify.ACTIVE_FRAC_COLUMN in row
    # ...and the same absence is named in the incomplete list, not only inside one cell.
    assert verify.ACTIVE_FRAC_COLUMN in _section(document, "## Incomplete runs")


def test_an_arm_missing_its_metrics_csv_is_reported_incomplete_not_rowless(tmp_path) -> None:
    write_arm(tmp_path, "complete", anchor_stride=15)
    write_arm(tmp_path, "no_csv", anchor_stride=1, with_csv=False)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")

    # The row is present, keyed by its swept value, with the gap number it does have...
    section = _section(document, "## Anchor tiling sweep")
    row = next(line for line in section.splitlines() if "no_csv" in line)
    assert row.split("|")[1].strip() == "1"
    # ...and what could not be read is named, in the row and in the incomplete list.
    assert "unknown" in row
    assert "## Incomplete runs" in document
    assert verify.METRICS_HISTORY_FILENAME in _section(document, "## Incomplete runs")


def test_an_empty_directory_is_an_error_not_an_empty_document(tmp_path) -> None:
    assert verify.compare_arms(tmp_path, tmp_path / "arms.md") == 1
    assert not (tmp_path / "arms.md").exists()


# =================================================================================================
# The cross-cell table
# =================================================================================================
def test_the_cross_cell_table_keys_rows_on_the_recorded_model_class(tmp_path) -> None:
    """Never on the directory name: a rename must not relabel which architecture produced a run,
    and the class is stamped in exactly one place a finished run keeps."""
    write_arm(tmp_path, "run_a", model_class=verify.COMPARISON_MODEL_CLASS)
    write_arm(tmp_path, "run_b", model_class=verify.BASELINE_MODEL_CLASS)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-cell comparison")
    rows = [line for line in section.splitlines() if line.startswith("| ")][1:]
    keys = [row.split("|")[1].strip() for row in rows if not row.startswith("|---")]

    # The baseline cell first, so the row the selection rule is stated against is met first.
    assert keys == [verify.BASELINE_MODEL_CLASS, verify.COMPARISON_MODEL_CLASS]
    assert "run_b" in rows[0] and "run_a" in rows[1]


def test_the_selection_rule_and_the_clock_margin_reach_the_document(tmp_path) -> None:
    """Both for the same reason: the table is what a threshold gets set from later, and a table of
    two architectures' KLs invites exactly the ranking the rule forbids."""
    write_arm(tmp_path, "cfs")
    write_arm(tmp_path, "trf", model_class=verify.COMPARISON_MODEL_CLASS)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-cell comparison")

    assert verify.SELECTION_RULE in section
    # The measurement the unset verdict does not gate, carried whatever the threshold says.
    assert f"`{verify.CLOCK_MARGIN_COLUMN}`" in section
    assert "0.75" in section


def test_a_row_whose_base_reconstruction_is_worse_is_marked_not_dropped(tmp_path) -> None:
    """The selection rule's mechanism. A suppressed arm reads as an arm that was never run, so the
    cell carries a marker and the footnote says what the marker means."""
    write_arm(tmp_path, "baseline", d_base=2000.0)
    write_arm(tmp_path, "worse", model_class=verify.COMPARISON_MODEL_CLASS, d_base=2500.0)
    write_arm(tmp_path, "better", model_class=verify.COMPARISON_MODEL_CLASS, d_base=1900.0)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-cell comparison")

    worse_row = next(line for line in section.splitlines() if "worse/" in line)
    better_row = next(line for line in section.splitlines() if "better/" in line)
    assert verify._D0_MARKER in worse_row
    assert verify._D0_MARKER not in better_row
    assert verify._D0_FOOTNOTE in section


def test_the_lag_peak_is_never_quoted_without_the_runs_verdict_on_it(tmp_path) -> None:
    """An argmax is defined on a flat profile and on a censored one exactly as it is on a real
    peak, so a bare number is the misreading the run's own ``argmax_lag`` check exists to prevent.
    Three states, three renderings -- and a run that never checked says so rather than reading as
    checked."""
    write_arm(tmp_path, "checked")
    degenerate = write_arm(tmp_path, "degenerate")
    summary = json.loads((degenerate / verify.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    summary["results"]["sanity"]["checks"]["argmax_lag"] = {
        "verdict": "fail", "detail": "the argmax lag is 0, so the attribution never looks back",
    }
    (degenerate / verify.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")

    unchecked = write_arm(tmp_path, "unchecked")
    summary = json.loads((unchecked / verify.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    del summary["results"]["sanity"]["checks"]["argmax_lag"]
    (unchecked / verify.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")

    out = tmp_path / "arms.md"
    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-cell comparison")
    rows = {
        name: next(line for line in section.splitlines() if f"{name}/" in line)
        for name in ("checked", "degenerate", "unchecked")
    }

    assert "7 (inside the window)" in rows["checked"]
    assert "**degenerate**" in rows["degenerate"]
    assert "never looks back" in rows["degenerate"]
    assert "(not checked)" in rows["unchecked"]


def test_a_directory_with_one_cell_still_emits_the_table_and_says_so(tmp_path) -> None:
    """A comparison with one side missing is a fact about the directory that was handed in;
    dropping the table would report it as a fact about the models."""
    write_arm(tmp_path, "only_cfs")
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-cell comparison")

    assert "Only one architecture is present here" in section
    assert verify.BASELINE_MODEL_CLASS in section


def test_a_run_recording_no_model_class_is_keyed_unknown_rather_than_guessed(tmp_path) -> None:
    """The offline re-run path genuinely does not know what produced it -- there is no checkpoint
    to read the stamp from -- and a row guessed from the directory name would be the one error
    this table's keying rule exists to prevent."""
    write_arm(tmp_path, "offline_rerun", model_class=None)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")
    section = _section(document, "## Cross-cell comparison")

    assert verify.UNRECORDED_MODEL in section
    assert "model_class" in _section(document, "## Incomplete runs")


def test_the_verdict_family_is_rendered_as_one_cell(tmp_path) -> None:
    """Four verdicts together, because FAIL / PASS / FAIL / INCONCLUSIVE is one state -- no
    predictive gain, a positive source margin, therefore no specificity, and a clock margin nobody
    has set a threshold for -- and four separate columns invite reading the first alone."""
    results_dir = write_arm(tmp_path, "mixed")
    summary = json.loads((results_dir / verify.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    summary["results"]["headline"].update({
        "verdict_predictive_improvement": "FAIL",
        "verdict_source_margin_positive": "PASS",
        "verdict_source_specificity": "FAIL",
        "verdict_coupling_exceeds_availability_clock": "INCONCLUSIVE",
    })
    (results_dir / verify.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-cell comparison")

    assert "FAIL / PASS / FAIL / INCONCLUSIVE" in section


# =================================================================================================
# What the tables must not do
# =================================================================================================
def test_no_cross_target_table_against_the_feature_target_cell_is_produced(tmp_path) -> None:
    """Deliberately out of scope, and asserted rather than trusted to stay so: the blocks differ
    (1470 against 2340 coefficients) and so do the horizons, so a level comparison against
    ``lag_attn_fs`` would invite exactly the reading both DESIGN.md records forbid."""
    write_arm(tmp_path, "cfs")
    write_arm(tmp_path, "trf", model_class=verify.COMPARISON_MODEL_CLASS)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")

    assert "lag_attn_fs" not in document
    assert "SeqVaeLagAttnFs" not in document
    # And the module offers no way to build one: the only cross-anything table is the cross-cell
    # one, whose two keys are the two cfs cells.
    assert not any(
        name.startswith("build_cross") and name != "build_cross_cell_table"
        for name in dir(verify)
    )
    assert {verify.BASELINE_MODEL_CLASS, verify.COMPARISON_MODEL_CLASS} == {
        "SeqVaeLagAttnCfs", "SeqVaeLagAttnTrfCfs"
    }


def test_the_emitted_document_never_uses_the_refused_names(tmp_path) -> None:
    """The tables are an artifact, and the naming rules that bind every run artifact bind them
    too: the coupling readout is not called a transfer entropy anywhere in the output, and there
    is no bpm anywhere in this pipeline."""
    write_arm(tmp_path, "arm")
    out = tmp_path / "arms.md"
    verify.compare_arms(tmp_path, out)

    lowered = out.read_text(encoding="utf-8").lower()
    assert "transfer entropy" not in lowered and "te_lag" not in lowered
    assert "bpm" not in lowered


def test_the_cli_dispatches_between_the_gate_and_the_tables(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(clean_summary()), encoding="utf-8")

    assert verify._cli([str(summary)]) == 0
    assert verify._cli([str(summary), "--runs", str(tmp_path)]) == 2  # both is a usage error
    assert verify._cli([]) == 2  # neither is too
