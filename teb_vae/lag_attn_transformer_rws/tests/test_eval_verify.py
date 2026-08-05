r"""The acceptance gate, this model's arm tables and the cross-model comparison, from files alone.

Everything here drives ``eval/verify.py`` the way an operator does: with a ``summary.json`` and,
for the tables, a directory of finished-run *shapes* on disk. No model is built anywhere in this
file -- that is the module's one non-negotiable property, and the AST layering test proves it on
the import graph while this file proves the behaviour.

The synthetic runs are deliberately minimal rather than copies of a real one: each test states
exactly which keys the cell under test reads, so a summary-schema change that moves one of them
fails here by name instead of surfacing as a wall of ``(missing)``.

``clean_summary`` comes from the sibling's verify suite by import. The gate *is* the sibling's, so
the summary shape it reads is the sibling's too, and a second copy of a forty-line fixture would be
a second thing to keep in step with the schema.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import yaml

from teb_vae.lag_attn_rws.eval import verify as shared_verify
from teb_vae.lag_attn_rws.tests.test_eval_verify import clean_summary
from teb_vae.lag_attn_transformer_rws.eval import verify

#: Eight epochs of healthy validation series, and the same length dead at the tail. The collapse
#: verdict is the shared criterion's, so these only have to be on the right side of it.
_HEALTHY_KL = [0.0, 0.5, 2.0, 4.0, 5.0, 5.5, 5.2, 5.4]
_HEALTHY_ACTIVE = [0.0, 0.1, 0.4, 0.5, 0.55, 0.6, 0.6, 0.58]
_COLLAPSED_KL = [0.0, 0.5, 1.0, 0.01, 0.005, 0.003, 0.002, 0.001]
_COLLAPSED_ACTIVE = [0.0, 0.1, 0.2, 0.02, 0.01, 0.01, 0.01, 0.01]


# =============================================================================
# The restated names are pinned to the registries that own them
# =============================================================================
def test_every_headline_column_the_tables_read_is_a_registered_headline_scalar():
    """The tables read the headline block and nothing else, so a column named here that no
    analysis registers is a column that renders ``(missing)`` on every run forever. Pinned against
    both registries: the shared one, and this model's own additions through its binding."""
    from teb_vae.lag_attn_rws.eval import report_seam
    from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

    shared_names = {name for name, _ in report_seam.HEADLINE_SCALARS}
    local_names = {name for name, _ in TRF_BINDING.headline_scalars}

    for column in (
        verify.D_BASE_COLUMN,
        verify.D_FULL_COLUMN,
        verify.KL_COLUMN,
        verify.LAG_PEAK_COLUMN,
        verify.ACTIVE_DIMS_COLUMN,
        verify.PRED_GAP_COLUMN,
    ):
        assert column in shared_names, column
    # The measured reach is this model's own, and belongs to neither the shared registry nor a
    # hand-written list: it exists because `encoder_attention` registers it.
    for column in (verify.REACH_MEDIAN_COLUMN, verify.REACH_P95_COLUMN):
        assert column in local_names, column
        assert column not in shared_names, column


def test_the_two_model_class_names_are_the_bindings_own():
    """The table keys on strings because the module may not import ``torch``; the strings are
    pinned to the classes so a rename fails here rather than emptying a column."""
    from teb_vae.lag_attn_rws.eval.run import RWS_BINDING
    from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

    assert verify.BASELINE_MODEL_CLASS == RWS_BINDING.model_cls.__name__
    assert verify.THIS_MODEL_CLASS == TRF_BINDING.model_cls.__name__
    assert verify.BASELINE_MODEL_CLASS != verify.THIS_MODEL_CLASS


def test_the_lag_readability_verdict_the_table_quotes_is_a_check_the_run_records():
    """``_lag_peak_cell`` re-reads the run's own ``argmax_lag`` check rather than re-deriving it.
    A rename in the sanity registry would make every lag cell read "not checked", which is exactly
    the shape a silent failure takes here."""
    from teb_vae.lag_attn_rws.eval import report_seam

    sanity = report_seam.build_sanity({}, {})

    assert "argmax_lag" in sanity["checks"]


def test_the_swept_paths_name_keys_the_shipped_arms_actually_set():
    """Each axis is an axis only if an arm file varies it. Read from the committed sweep configs,
    so an arm renamed or dropped is reported here rather than leaving a table of one row."""
    configs = Path(__file__).resolve().parents[1] / "configs"
    swept: Dict[str, Any] = {}
    for path in sorted(configs.glob("sweep_*.yaml")):
        block = (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        for key, value in ((block.get("model_config") or {}).get("VAE_model") or {}).items():
            swept.setdefault(key, []).append(value)

    for axis in (
        verify.SWEPT_WINDOW,
        verify.SWEPT_TARGET_BLOCKS,
        verify.SWEPT_SOURCE_BLOCKS,
        verify.SWEPT_D_FF,
        verify.SWEPT_REACH,
        # The prior-anchor axis belongs here even though the objective term is shared rather than
        # this encoder's: the four arms that sweep it ship in THIS package, so this is the only
        # suite in which renaming or retiring one can be caught.
        verify.SWEPT_BETA_PRIOR,
    ):
        assert axis[-1] in swept, f"no shipped arm varies {axis[-1]!r}"
    # And the stem arm, which is the pair of empty lists rather than a scalar, is covered by the
    # architecture row rather than by an axis of its own.
    assert [] in swept["encoder_conv_kernels"]
    for path in verify.ARCHITECTURE_KEYS:
        assert path[:-1] == ("model_config", "VAE_model")


# =============================================================================
# The gate
# =============================================================================
def test_the_gate_passes_a_clean_summary_and_refuses_a_failed_one(tmp_path):
    """Delegated in full, so what is asserted here is the delegation and the exit codes -- the
    criteria themselves are the sibling's and are tested there."""
    passing = tmp_path / "summary.json"
    passing.write_text(json.dumps(clean_summary()), encoding="utf-8")
    json_out = tmp_path / "report.json"

    assert verify.main(passing, json_out) == 0
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["passed"] is True
    assert written["pred_gap_column_read"] == verify.PRED_GAP_COLUMN

    failing = tmp_path / "failing.json"
    failing.write_text(json.dumps(clean_summary(exit_code=1, failed=["forecast"])), "utf-8")
    assert verify.main(failing) == 1


def test_a_failed_sanity_check_exits_non_zero_even_though_every_step_completed(tmp_path):
    """The asymmetry the gate exists for: the runner reports whether a step *raised*, and a run
    whose numbers do not hold together exits 0 there."""
    summary = clean_summary()
    summary["results"]["sanity"]["failed"] = ["kl_identity"]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    assert verify.main(path) == 1


def test_the_parser_names_this_package():
    assert "lag_attn_transformer_rws" in verify.build_parser().prog


def test_the_cli_dispatches_between_the_gate_and_the_tables(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(clean_summary()), encoding="utf-8")

    assert verify._cli([str(summary)]) == 0
    assert verify._cli([str(summary), "--runs", str(tmp_path)]) == 2  # both is a usage error
    assert verify._cli([]) == 2  # neither is too


# =============================================================================
# Synthetic finished runs
# =============================================================================
def write_run(
    root: Path,
    name: str,
    *,
    model_class: Optional[str] = verify.THIS_MODEL_CLASS,
    window: Any = 16,
    window_absent: bool = False,
    target_blocks: int = 4,
    source_blocks: int = 3,
    d_ff: int = 256,
    conv_kernels: Sequence[int] = (5, 9),
    reach: Optional[Any] = None,
    kept: Optional[Tuple[int, int]] = None,
    d_base: float = 700.0,
    pred_gap: float = 1.0,
    beta_prior: float = 0.1,
    reach_median: Optional[float] = None,
    reach_p95: Optional[float] = None,
    lag_peak: Optional[int] = None,
    lag_check: Optional[Dict[str, Any]] = None,
    kl_series: Optional[List[float]] = None,
    active_series: Optional[List[float]] = None,
    with_csv: bool = True,
) -> Path:
    """Write one finished-run shape: summary, resolved config and the training CSV.

    Only the keys the tables read are set. ``model_class=None`` writes a ``run_context`` without
    one, which is what a run evaluated with no checkpoint leaves behind.
    """
    results_dir = root / name / "eval_results"
    results_dir.mkdir(parents=True)
    train_root = root / name / "train_run"
    checkpoint = train_root / "model_checkpoints" / "epoch=07.ckpt"

    summary = clean_summary(checkpoint=str(checkpoint))
    summary["run_context"] = {
        "n_parameters": 4_100_000,
        "train_epoch": 7,
        "model_class": model_class,
    }
    headline = summary["results"]["headline"]
    headline[verify.D_BASE_COLUMN] = d_base
    headline[verify.D_FULL_COLUMN] = d_base - pred_gap
    headline[verify.PRED_GAP_COLUMN] = pred_gap
    if reach_median is not None:
        headline[verify.REACH_MEDIAN_COLUMN] = reach_median
    if reach_p95 is not None:
        headline[verify.REACH_P95_COLUMN] = reach_p95
    if lag_peak is not None:
        # The peak is a headline scalar; the sanity check below is the run's verdict on whether
        # it means anything. Both come off the same `results.lag` block in a real run.
        headline[verify.LAG_PEAK_COLUMN] = lag_peak
    if lag_check is not None:
        summary["results"]["sanity"]["checks"]["argmax_lag"] = lag_check
    (results_dir / verify.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")

    model: Dict[str, Any] = {
        "d_z": 48,
        "beta_prior": beta_prior,
        "causal_reach_budget_s": reach,
        "c_y": 109,
        "c_u": 58,
        "encoder_conv_kernels": list(conv_kernels),
        "encoder_conv_dilations": [1, 2][: len(conv_kernels)],
        "encoder_num_heads": 4,
        "encoder_d_ff": d_ff,
        "target_attention_blocks": target_blocks,
        "source_attention_blocks": source_blocks,
    }
    if not window_absent:
        model["source_attention_window"] = window
    config: Dict[str, Any] = {"model_config": {"VAE_model": model}}
    if kept is not None:
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
        lines = [
            f"{shared_verify.EPOCH_COLUMN},{shared_verify.KL_SERIES_COLUMN},"
            f"{verify.ACTIVE_FRAC_COLUMN}"
        ]
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


def _rows(section: str) -> List[str]:
    """The data rows of the one table in a section, header and rule dropped."""
    return [
        line for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ][1:]


def _cells(row: str) -> List[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _by_run(section: str) -> Dict[str, List[str]]:
    """The section's rows, keyed by the run directory's own name.

    The run column carries the display path a scan produced -- ``<name>/eval_results`` -- and the
    tests name the directory they wrote, so the first component is the key.
    """
    return {_cells(row)[-1].split("/")[0]: _cells(row) for row in _rows(section)}


# =============================================================================
# This model's arm tables
# =============================================================================
def test_the_window_table_is_keyed_by_the_config_value_not_the_directory(tmp_path):
    """The directory names here are deliberate lies -- each names a different window than its
    config carries -- so a row keyed off the name would be caught immediately. ``null`` sorts with
    the non-numeric keys and is a *value*: the unbounded arm."""
    write_run(tmp_path, "window_64", window=8)
    write_run(tmp_path, "window_8", window=32)
    write_run(tmp_path, "unbounded", window=None)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Source window sweep")
    rows = _rows(section)

    assert [_cells(row)[0] for row in rows] == ["8", "32", "null"]
    assert "window_64" in rows[0] and "window_8" in rows[1] and "unbounded" in rows[2]


def test_an_arm_that_lost_the_window_key_is_not_keyed_like_the_unbounded_one(tmp_path):
    """The distinction that matters more here than in the sibling: ``null`` is the arm the whole
    locality family is measured against, and an arm file that dropped the key is a mis-built arm.
    Folding the two together would rank a defect as a measurement."""
    write_run(tmp_path, "unbounded", window=None)
    write_run(tmp_path, "broken", window_absent=True)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Source window sweep")
    keys = {run: cells[0] for run, cells in _by_run(section).items()}

    assert keys["unbounded"] == "null"
    assert keys["broken"] == "(absent)"


def test_the_window_table_carries_the_measured_reach_beside_the_configured_window(tmp_path):
    """Half the point of the encoder-attention analysis: a window is what the arm was *given*, and
    the reach columns are what its encoder used. A run without the analysis says ``(missing)``
    rather than reporting a zero it never measured."""
    write_run(tmp_path, "measured", window=16, reach_median=9.0, reach_p95=14.0)
    write_run(tmp_path, "unmeasured", window=32)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    by_run = _by_run(_section(out.read_text(encoding="utf-8"), "## Source window sweep"))

    assert by_run["measured"][1] == "9" and by_run["measured"][2] == "14"
    assert by_run["unmeasured"][1] == "(missing)" and by_run["unmeasured"][2] == "(missing)"


def test_every_arm_appears_in_every_sweep_table(tmp_path):
    """Which arms belong to which sweep is a fact about the directory an operator handed in. A
    module that filtered by "the axis this arm looks like it varies" would drop rows on the arms
    that vary two."""
    write_run(tmp_path, "window_8", window=8)
    write_run(tmp_path, "target_5", target_blocks=5)
    write_run(tmp_path, "ff_384", d_ff=384)
    write_run(tmp_path, "reach_120", reach=120, kept=(78, 29))
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    document = out.read_text(encoding="utf-8")

    for heading in (
        "## Source window sweep",
        "## Target encoder depth sweep",
        "## Source encoder depth sweep",
        "## Feed-forward width sweep",
        "## Reach budget sweep",
        "## Encoder architecture arms",
    ):
        by_run = _by_run(_section(document, heading))
        assert set(by_run) == {"window_8", "target_5", "ff_384", "reach_120"}, heading


def test_the_prior_anchor_arms_resolve_into_their_own_section(tmp_path):
    """The prior-anchor sweep is rendered by this package's own ``build_arm_tables``, so it needs
    its own test here.

    The objective term is shared with the comparison model, but the four arms that sweep it ship in
    *this* package and are run here, so this copy of the section is what the study is read from --
    and until this test existed, deleting the whole block left the suite at its exact pass count.
    Three regressions it closes: the section going missing or being renamed, ``SWEPT_BETA_PRIOR``
    keying on the wrong config path (every row would collapse onto one inherited value), and a typo
    in any of the four headline names inlined in this block, which renders ``(missing)`` rather
    than raising.
    """
    write_run(tmp_path, "bp_low", beta_prior=1.0e-3)
    write_run(tmp_path, "bp_shipped", beta_prior=0.1)
    write_run(tmp_path, "bp_high", beta_prior=1.0)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Prior-anchor weight sweep")

    rows = _rows(section)
    # Keyed by each run's own resolved beta_prior, in numeric order -- not by directory name.
    assert [_cells(row)[0] for row in rows] == ["0.001", "0.1", "1"], section
    # Every headline name in this block resolves against a real summary. `(missing)` is what a
    # mistyped column renders as, and it would otherwise ship as a table of blanks.
    assert "(missing)" not in section, section

    # The header is the line above the delimiter row -- this section carries prose first, so a
    # fixed offset would drift the moment that prose is edited.
    lines = section.splitlines()
    header = next(
        lines[index - 1] for index, line in enumerate(lines) if line.startswith("|---")
    )
    for column in (
        "`beta_prior`", "`logvar_prior_floor_frac`", "`mean_logvar_prior`", "`prior_rate`",
        "`pred_gap`", "`abs(pred_gap)/K`", "`source_margin`", "Verdicts",
    ):
        assert column in header, column


def test_the_reach_table_reports_the_surviving_channels_not_the_declared_widths(tmp_path):
    """The column is that sweep's whole subject and ``c_y``/``c_u`` cannot express it: two startup
    guards force the declared widths to equal the shard's for every arm that can train."""
    write_run(tmp_path, "reach_null", reach=None)
    write_run(tmp_path, "reach_120", reach=120, kept=(78, 29))
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Reach budget sweep")
    kept = {cells[0]: cells[1] for cells in _by_run(section).values()}

    assert kept["120"] == "78 / 29"
    assert kept["null"] == "109 / 58"


def test_the_architecture_row_names_all_seven_encoder_knobs(tmp_path):
    """A row naming all of them is readable without knowing in advance which one this arm flipped
    -- and the stem arm is a pair of empty lists rather than a scalar, so it has no axis of its
    own and this row is where it is legible."""
    write_run(tmp_path, "stem_free", conv_kernels=[])
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    row = _rows(_section(out.read_text(encoding="utf-8"), "## Encoder architecture arms"))[0]

    for path in verify.ARCHITECTURE_KEYS:
        assert f"{path[-1]}=" in row
    assert "encoder_conv_kernels=[]" in row


def test_a_collapsed_arm_is_marked_never_omitted(tmp_path):
    write_run(tmp_path, "healthy", window=8)
    write_run(
        tmp_path, "dead", window=16, kl_series=_COLLAPSED_KL, active_series=_COLLAPSED_ACTIVE
    )
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Source window sweep")

    assert "**collapsed**" in next(row for row in _rows(section) if "dead" in row)
    assert "| no |" in next(row for row in _rows(section) if "healthy" in row)


def test_an_arm_missing_its_metrics_csv_is_reported_incomplete_not_rowless(tmp_path):
    write_run(tmp_path, "complete", window=8)
    write_run(tmp_path, "no_csv", window=16, with_csv=False)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")

    row = next(row for row in _rows(_section(document, "## Source window sweep"))
               if "no_csv" in row)
    assert _cells(row)[0] == "16"  # keyed by the value it does carry
    assert "incomplete" in row
    assert "## Incomplete runs" in document
    assert verify.METRICS_HISTORY_FILENAME in _section(document, "## Incomplete runs")


def test_an_empty_directory_is_an_error_not_an_empty_document(tmp_path):
    assert verify.compare_arms(tmp_path, tmp_path / "arms.md") == 1
    assert not (tmp_path / "arms.md").exists()


# =============================================================================
# The cross-model table
# =============================================================================
def test_two_models_runs_are_keyed_by_the_class_each_run_recorded(tmp_path):
    """The comparison the package exists to make. The directory names are again lies, and the
    baseline sorts first so the row the selection rule is stated against is met first."""
    write_run(tmp_path, "transformer_looking_name", model_class=verify.BASELINE_MODEL_CLASS)
    write_run(tmp_path, "baseline_looking_name", model_class=verify.THIS_MODEL_CLASS)
    out = tmp_path / "arms.md"

    assert verify.compare_arms(tmp_path, out) == 0
    section = _section(out.read_text(encoding="utf-8"), "## Cross-model comparison")
    rows = _rows(section)

    assert [_cells(row)[0] for row in rows] == [
        verify.BASELINE_MODEL_CLASS, verify.THIS_MODEL_CLASS
    ]
    assert _cells(rows[0])[-1].startswith("transformer_looking_name")
    assert _cells(rows[1])[-1].startswith("baseline_looking_name")


def test_a_single_model_directory_still_emits_the_table_and_says_so(tmp_path):
    """An omitted table reads as a comparison that could not be made; the note says which side is
    missing, which is a fact about the directory rather than about the models."""
    write_run(tmp_path, "trf_a", model_class=verify.THIS_MODEL_CLASS)
    write_run(tmp_path, "trf_b", model_class=verify.THIS_MODEL_CLASS)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Cross-model comparison")

    assert len(_rows(section)) == 2
    assert "Only one architecture is present here" in section
    assert verify.THIS_MODEL_CLASS in section


def test_a_run_recording_no_model_class_is_keyed_unknown_and_reported_incomplete(tmp_path):
    """A run evaluated with no checkpoint has no stamp to copy. Guessing from the directory name
    is the one error the keying rule exists to prevent, so the row says it does not know."""
    write_run(tmp_path, "offline_rerun", model_class=None)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    document = out.read_text(encoding="utf-8")
    row = _rows(_section(document, "## Cross-model comparison"))[0]

    assert _cells(row)[0] == verify.UNRECORDED_MODEL
    assert "carries no `model_class`" in _section(document, "## Cross-model comparison")
    assert "no model_class" in _section(document, "## Incomplete runs")


def test_the_cross_model_row_carries_the_numbers_the_architecture_question_is_asked_with(tmp_path):
    """$D_0$, $D_1$, the gap, the KL, the lag peak, the active fraction, the collapse verdict, the
    epoch count and the parameter count -- each read from the artifact rather than recomputed."""
    write_run(
        tmp_path, "trf", d_base=680.0, pred_gap=2.5, lag_peak=12,
        lag_check={"verdict": "pass", "detail": "peaks strictly inside"},
    )
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    cells = _cells(_rows(_section(out.read_text(encoding="utf-8"), "## Cross-model"))[0])

    assert cells[1] == "4100000"          # parameters
    assert cells[2] == "680" and cells[3] == "677.5"   # D0, D1
    assert cells[4] == "2.5"              # pred_gap
    assert cells[6].startswith("12 (inside the window)")  # the lag peak and its verdict
    assert cells[7] == "0.58"             # final active fraction
    assert cells[8] == "no" and cells[9] == "7"        # collapse verdict, epochs


def test_a_degenerate_lag_peak_is_marked_beside_the_argmax(tmp_path):
    """An argmax is defined on a flat profile exactly as it is on a real peak, so the number is
    never quoted without the run's own verdict on whether it means anything."""
    write_run(
        tmp_path, "edge", lag_peak=90, lag_check={
            "verdict": "fail",
            "detail": "the argmax lag sits at the largest attainable lag (90)",
        },
    )
    write_run(tmp_path, "unchecked", lag_check={})
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    rows = {
        run: cells[6]
        for run, cells in _by_run(
            _section(out.read_text(encoding="utf-8"), "## Cross-model")
        ).items()
    }

    assert "**degenerate**" in rows["edge"] and "largest attainable lag" in rows["edge"]
    assert rows["unchecked"].endswith("(not checked)")


# =============================================================================
# The selection rule
# =============================================================================
def test_the_selection_rule_is_in_the_emitted_document(tmp_path):
    """In the artifact rather than only in the source: a table of two architectures' KLs invites
    exactly the ranking the rule forbids, and a reader has only the table."""
    write_run(tmp_path, "trf")
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Cross-model comparison")

    assert verify.SELECTION_RULE in section
    assert "Do not select on KL magnitude" in section
    assert verify.BASELINE_MODEL_CLASS in section  # which row is the baseline is explicit


def test_a_row_worse_than_the_baseline_on_d0_is_flagged_never_dropped(tmp_path):
    """A suppressed arm reads as an arm that was not run, so the row stays and the cell is marked.
    The threshold is the best baseline run in the directory: the rule asks whether this arm is
    competitive with what the comparison architecture can do."""
    write_run(tmp_path, "baseline", model_class=verify.BASELINE_MODEL_CLASS, d_base=700.0)
    write_run(tmp_path, "trf_better", d_base=690.0, pred_gap=2.0)
    write_run(tmp_path, "trf_worse", d_base=760.0, pred_gap=9.0)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    document = out.read_text(encoding="utf-8")
    section = _section(document, "## Cross-model comparison")
    gaps = {run: cells[4] for run, cells in _by_run(section).items()}

    assert gaps["trf_worse"] == f"9 {verify._D0_MARKER}"
    assert gaps["trf_better"] == "2"
    assert gaps["baseline"] == "1"
    assert verify._D0_FOOTNOTE in section
    # Flagged, not dropped: every run is still a row.
    assert len(_rows(section)) == 3


def test_a_non_finite_d0_neither_sets_the_threshold_nor_trips_it(tmp_path):
    """A NaN $D_0$ is not a worse reconstruction, it is an unusable one, and the run's own
    ``headline_finite`` check is where that is reported. Ranking against it -- or flagging on it --
    would turn a broken run into a threshold."""
    write_run(tmp_path, "baseline_nan", model_class=verify.BASELINE_MODEL_CLASS,
              d_base=float("nan"))
    write_run(tmp_path, "trf_nan", d_base=float("nan"))
    write_run(tmp_path, "trf", d_base=800.0)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Cross-model comparison")

    # No finite baseline exists, so nothing is flagged -- including the run that would be worse
    # than any of them.
    assert verify._D0_MARKER not in section
    assert len(_rows(section)) == 3


def test_no_row_is_flagged_when_the_directory_holds_no_baseline_run(tmp_path):
    """Nothing to be worse than. A flag raised against an absent baseline would be an arbitrary
    threshold wearing the rule's clothes."""
    write_run(tmp_path, "trf_a", d_base=900.0)
    write_run(tmp_path, "trf_b", d_base=700.0)
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)
    section = _section(out.read_text(encoding="utf-8"), "## Cross-model comparison")

    assert verify._D0_MARKER not in section
    assert verify.baseline_d_base(
        [{"model_class": verify.THIS_MODEL_CLASS, "headline": {verify.D_BASE_COLUMN: 700.0}}]
    ) is None


# =============================================================================
# The module's own properties
# =============================================================================
def test_neither_this_gate_nor_the_one_it_delegates_to_imports_torch():
    """Proved on the import graph, one level of delegation deep: this module is a thin dispatcher
    over the sibling's, so a ``torch`` import *there* would break the property here just as
    surely. The layering test walks this module; nothing else walks what it calls."""
    for module in (verify, shared_verify):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        names: List[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        offending = [name for name in names if name == "torch" or name.startswith("torch.")]
        assert offending == [], f"{module.__name__} imports {offending}"


def test_the_emitted_document_never_uses_the_refused_name(tmp_path):
    """The tables are an artifact, and the naming rule that binds every run artifact binds them
    too: the coupling readout is not called a transfer entropy anywhere in the output."""
    write_run(tmp_path, "trf")
    out = tmp_path / "arms.md"
    verify.compare_arms(tmp_path, out)

    lowered = out.read_text(encoding="utf-8").lower()
    assert "transfer entropy" not in lowered and "te_lag" not in lowered


@pytest.mark.parametrize("heading", [
    "## Arm inventory",
    "## Source window sweep",
    "## Target encoder depth sweep",
    "## Source encoder depth sweep",
    "## Feed-forward width sweep",
    "## Reach budget sweep",
    "## Encoder architecture arms",
    "## Prior-anchor weight sweep",
    "## Cross-model comparison",
])
def test_the_document_carries_every_section_under_its_own_heading(tmp_path, heading):
    write_run(tmp_path, "trf")
    out = tmp_path / "arms.md"

    verify.compare_arms(tmp_path, out)

    assert heading in out.read_text(encoding="utf-8")
