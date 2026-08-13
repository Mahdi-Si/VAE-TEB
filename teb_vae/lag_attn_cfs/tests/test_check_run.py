r"""The run checker is what turns a registered criterion into a verdict, so it is what needs testing.

``RESULTS.md`` registers five tier-1 criteria before the headline run and three of them are
statements about *every logged row*; one is a $10^{-6}$ relative recomposition of two channel-axis
splits against each other, which nobody can honestly check by reading a multi-thousand-row CSV. The
checker exists so the verdict is produced by code -- and a checker nobody tests is a verdict nobody
should trust, because the failure mode is a green report on a void run rather than an exception.

Every test below plants **one** failure in an otherwise passing history and asserts that the checker
finds that criterion and only that criterion. The two properties that make the tool worth having are
asserted directly and are easy to lose in a rewrite:

* an unevaluated criterion is reported as unevaluated rather than as satisfied -- criterion 5 needs
  two run directories and cannot be read from one;
* a tier-2 value never moves the exit code, whatever it reads. There is no prior against which a
  threshold on any of them could have been calibrated, so a checker that gated on one would be
  enforcing a guess.

The anchor band is derived from the run's own resolved configuration rather than hard-coded, which
the geometry test exercises with a configuration none of the shipped arms uses: a checker carrying
the shipped $[10, 11]$ and $152$ as literals would pass a horizon arm for the wrong reason.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest
import yaml

from teb_vae.lag_attn_cfs import check_run
from teb_vae.lag_attn_cfs.check_run import (
    FAIL,
    NOT_EVALUATED,
    PASS,
    RUN_ARGS,
    build_parser,
    resolve_anchor_band,
)

from .conftest import (
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
)

#: How many epochs a synthetic history carries. Enough that the tail direction has a window and
#: that "every logged row" is a statement about more than one row.
ROWS = 12

#: The anchor counts the shipped geometry implies, written out **here** rather than imported, so
#: the checker's own derivation is compared against a hand computation rather than against itself.
#: $T_{\mathrm{valid}} = 300 - 15 = 285$, floor $133$, so $152$ anchors and
#: $\lceil 152/15 \rceil = 11$ tiles at phase $0$ against $10$ at the last.
SHIPPED_DENSE_ANCHORS = 152
SHIPPED_TILE_BAND = (10, 11)


def _row(epoch: int) -> Dict[str, Any]:
    """One passing epoch of a synthetic metric history.

    The two channel-axis splits are built to recompose to each other exactly, which is what
    criterion 4 checks: three warm-up tertiles summing to the same number as the two stored-block
    columns. They are given *different* per-column values so a checker comparing the wrong pair
    would not pass by accident.

    Args:
        epoch: The epoch number, which several series drift with so a tail direction exists.

    Returns:
        The row, keyed by column name.
    """
    row: Dict[str, Any] = {"epoch": epoch}
    for stage in ("train", "val"):
        row[f"{stage}/target_warm_frac"] = 1.0
        row[f"{stage}/pred_gap_warm_lo"] = 0.25
        row[f"{stage}/pred_gap_warm_mid"] = 0.5
        row[f"{stage}/pred_gap_warm_hi"] = 0.25
        row[f"{stage}/pred_gap_st"] = 0.75
        row[f"{stage}/pred_gap_ph"] = 0.25
        row[f"{stage}/source_conditioned_kl_raw"] = 0.1 * epoch
        row[f"{stage}/kld_active_frac"] = 0.66
        row[f"{stage}/logvar_prior_floor_frac"] = 0.01
        row[f"{stage}/source_lag_warmth_frac_st"] = 0.4
        row[f"{stage}/source_lag_warmth_frac_ph"] = 0.02
    # The tiling: a training step's value is the batch mean over per-segment phases, so it sits
    # strictly inside the band, while both evaluation stages decode every valid anchor.
    row["train/anchors_per_sample"] = 10.4
    row["val/anchors_per_sample"] = float(SHIPPED_DENSE_ANCHORS)
    row["train/total_loss"] = 1500.0 - epoch
    row["train/spike_skipped"] = 0.0
    row["val/kld_source_null"] = 0.02 * epoch
    row["val/shuffle_penalty"] = 0.3
    return row


def write_run(
    directory: Path,
    *,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    geometry: Optional[Dict[str, Any]] = None,
    with_config: bool = True,
) -> Path:
    """Write a synthetic run directory in the layout the training entry point produces.

    Args:
        directory: The run root; created if absent.
        rows: The metric rows. Defaults to :data:`ROWS` passing epochs.
        geometry: ``model_config.VAE_model`` leaves to override in the resolved configuration.
        with_config: Whether to write the resolved configuration at all. A run can legitimately
            lack it -- the driver's write of that file is non-fatal -- which is the case the
            ``--config`` argument exists for.

    Returns:
        The run root.
    """
    history = list(rows if rows is not None else [_row(epoch) for epoch in range(ROWS)])
    results = directory / "train_results"
    results.mkdir(parents=True, exist_ok=True)
    with open(results / "metrics_history.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    if with_config:
        vae = dict(
            sequence_length=SHIPPED_SEQUENCE_LENGTH,
            horizon=SHIPPED_HORIZON,
            warmup_period=SHIPPED_WARMUP_PERIOD,
            anchor_stride=SHIPPED_HORIZON,
        )
        vae.update(geometry or {})
        checkpoints = directory / "model_checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "resolved_config.yaml").write_text(
            yaml.safe_dump({"model_config": {"VAE_model": vae}}), encoding="utf-8"
        )
    return directory


def broken(column: str, value: Any, *, epoch: int = 5) -> List[Dict[str, Any]]:
    """A passing history with one cell of one row replaced.

    One cell rather than a whole column, deliberately: every tier-1 criterion is a statement about
    *every* logged row, so a checker that looked at the last row alone -- or at an aggregate --
    would pass a history whose failure sits in the middle.

    Args:
        column: The column to break.
        value: The value to put there.
        epoch: Which row to break.

    Returns:
        The rows.
    """
    rows = [_row(index) for index in range(ROWS)]
    rows[epoch][column] = value
    return rows


def statuses(capsys) -> Dict[int, str]:
    """Parse the printed report back into ``{criterion number: status}``.

    Args:
        capsys: pytest's capture fixture, read after a ``main`` call.

    Returns:
        The verdicts by criterion number.
    """
    found: Dict[int, str] = {}
    for line in capsys.readouterr().out.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        status, remainder = stripped[1:].split("]", 1)
        found[int(remainder.strip().split(".", 1)[0])] = status.strip()
    return found


# =================================================================================================
# The Run-button convention
# =================================================================================================
def test_the_module_ships_a_launch_dict_whose_keys_are_all_arguments():
    """Without one there is nothing to fill in and the Run button can only fail; a key that is not
    a ``dest`` silently does nothing, which is the failure the one launch path with no command line
    to misspell in cannot otherwise surface."""
    dests = {action.dest for action in build_parser()._actions if action.dest != "help"}

    assert isinstance(RUN_ARGS, dict)
    assert set(RUN_ARGS) == dests, (
        f"only in RUN_ARGS: {sorted(set(RUN_ARGS) - dests)}, "
        f"only on the parser: {sorted(dests - set(RUN_ARGS))}"
    )


def test_no_argument_is_required_and_none_carries_an_argparse_default():
    """``required=True`` fires before the launch dict is consulted, so it makes the Run button
    unusable whatever the dict says; and the merge reads any non-``None`` parsed value as having
    come from the command line, so an argparse default makes that key's entry unreachable -- the
    operator edits the dict, nothing changes, and nothing says why."""
    actions = [action for action in build_parser()._actions if action.dest != "help"]

    assert [action.dest for action in actions if action.required] == []
    assert {action.dest: action.default for action in actions if action.default is not None} == {}


def test_the_entry_point_returns_its_exit_code_rather_than_exiting(tmp_path):
    """``main`` returns the code and ``_cli`` is what hands it to ``sys.exit``. A ``sys.exit`` in
    ``main`` would make it uncallable from a test or from another entry point."""
    write_run(tmp_path)

    assert check_run.main(run_dir=str(tmp_path)) == 0
    assert check_run._cli(["--run-dir", str(tmp_path)]) == 0


def test_a_missing_run_directory_refuses_rather_than_raising(tmp_path, capsys):
    """Named, and non-zero. An operator who pointed the checker at the wrong directory must get a
    sentence about the directory rather than a traceback about a file."""
    code = check_run.main(run_dir=str(tmp_path / "absent"))

    assert code == 2
    assert "no metric history" in capsys.readouterr().out


def test_the_refusal_names_both_ways_to_supply_the_run_directory(capsys):
    """Required-ness is enforced after the merge, so the message has to name the launch dict as
    well as the flag -- the Run button is the path that has no flag to read."""
    code = check_run._cli([])

    assert code == 2
    printed = capsys.readouterr().out
    assert "--run-dir" in printed and "RUN_ARGS" in printed


# =================================================================================================
# Tier 1: one planted failure at a time
# =================================================================================================
def test_a_clean_run_passes_every_derivable_criterion(tmp_path, capsys):
    """The baseline the planted-failure tests are read against. Four criteria pass and the fifth is
    reported as not evaluated, which is not a pass."""
    write_run(tmp_path)

    code = check_run.main(run_dir=str(tmp_path))

    assert code == 0
    assert statuses(capsys) == {1: PASS, 2: PASS, 3: PASS, 4: PASS, 5: NOT_EVALUATED}


@pytest.mark.parametrize(
    "number, column, value",
    [
        # A stamped provenance column: anything but exactly 1.0 means the checkpoint was built by
        # code predating the constructor's budget-and-floor refusal.
        (1, "val/target_warm_frac", 0.999999),
        # The tiling is not the one the configuration states.
        (2, "train/anchors_per_sample", 8.0),
        (2, "val/anchors_per_sample", 151.0),
        # The loss went non-finite, which the breaker's own guard should have caught first.
        (3, "train/total_loss", float("nan")),
        # The breaker latched: the margin is mis-tuned for this objective.
        (3, "train/spike_skipped", 0.04),
        # The two channel-axis splits stopped decomposing the same quantity.
        (4, "train/pred_gap_warm_mid", 0.6),
        (4, "val/pred_gap_ph", 0.35),
    ],
)
def test_each_tier_one_criterion_catches_its_own_planted_failure(
    tmp_path, capsys, number, column, value
):
    """One cell of one row, in an otherwise passing history. The criterion that owns it fails, the
    others do not, and the exit code follows."""
    write_run(tmp_path, rows=broken(column, value))

    code = check_run.main(run_dir=str(tmp_path))

    found = statuses(capsys)
    assert code == 1
    assert found[number] == FAIL, f"{column} did not trip criterion {number}"
    assert [key for key, status in found.items() if status == FAIL] == [number]


@pytest.mark.parametrize(
    "absent",
    [("train/target_warm_frac", "val/target_warm_frac"), ("val/target_warm_frac",)],
    ids=["both stages", "one stage"],
)
def test_an_absent_column_fails_rather_than_passing_vacuously(tmp_path, capsys, absent):
    """The failure a checker written against ``if value != expected`` has: a run whose column is
    missing entirely has no offending row, so every per-row test passes over nothing.

    Both parametrisations matter. The criterion is a statement about every logged row of **both**
    stages, so a column present on one of them is not half a pass — it is a criterion that was not
    evaluated where it was absent."""
    rows = [_row(epoch) for epoch in range(ROWS)]
    for row in rows:
        for column in absent:
            del row[column]
    write_run(tmp_path, rows=rows)

    code = check_run.main(run_dir=str(tmp_path))

    assert code == 1
    assert statuses(capsys)[1] == FAIL


def test_the_recomposition_tolerance_admits_float32_noise(tmp_path, capsys):
    """The criterion is $10^{-6}$ *relative*, not exact: the two splits are float32 reductions of
    the same per-element scores in a different order, so an exact test would fail every real run."""
    write_run(tmp_path, rows=broken("train/pred_gap_warm_mid", 0.5 + 1e-8))

    code = check_run.main(run_dir=str(tmp_path))

    assert code == 0
    assert statuses(capsys)[4] == PASS


# =================================================================================================
# The geometry the anchor count is read against
# =================================================================================================
def test_the_anchor_band_is_the_hand_computed_one_at_the_shipped_geometry():
    """Against a hand computation rather than against the checker's own arithmetic reapplied."""
    band = resolve_anchor_band(
        {
            "model_config": {
                "VAE_model": {
                    "sequence_length": SHIPPED_SEQUENCE_LENGTH,
                    "horizon": SHIPPED_HORIZON,
                    "warmup_period": SHIPPED_WARMUP_PERIOD,
                    "anchor_stride": SHIPPED_HORIZON,
                }
            }
        }
    )

    assert (band.train_low, band.train_high) == SHIPPED_TILE_BAND
    assert band.dense == SHIPPED_DENSE_ANCHORS


def test_the_band_follows_the_run_rather_than_the_shipped_numbers(tmp_path, capsys):
    """A checker carrying $[10, 11]$ and $152$ as literals would pass an arm at another horizon,
    floor or stride for the wrong reason -- and three shipped arms move one of the three. Here the
    history is the shipped one and the configuration is not, so the counts must be *rejected*."""
    write_run(tmp_path, geometry={"horizon": 30, "anchor_stride": 30})

    code = check_run.main(run_dir=str(tmp_path))

    assert code == 1
    assert statuses(capsys)[2] == FAIL


def test_a_run_with_no_resolved_configuration_fails_the_geometry_criterion(tmp_path, capsys):
    """Not a pass and not a crash. The driver's write of that file is deliberately non-fatal, so a
    run can lack it -- and a criterion that cannot be evaluated must not read as satisfied. The
    message names the argument that recovers it."""
    write_run(tmp_path, with_config=False)

    code = check_run.main(run_dir=str(tmp_path))

    printed = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] 2." in printed
    assert "--config" in printed


def test_a_malformed_resolved_configuration_fails_the_geometry_criterion(tmp_path, capsys):
    """A truncated config is what a killed run leaves behind, which is when this gets pointed at a
    directory at all.

    ``yaml.YAMLError`` derives from ``Exception``, not from any of the three the caller catches, so
    an unparseable file took the whole report down with a traceback instead of failing the one
    criterion that needs the geometry -- the opposite of what the guard exists for.
    """
    run = write_run(tmp_path)
    (run / "model_checkpoints" / "resolved_config.yaml").write_text(
        "model_config:\n  VAE_model:\n   horizon: [15\n", encoding="utf-8"
    )

    code = check_run.main(run_dir=str(run))

    printed = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] 2." in printed
    assert "does not parse as YAML" in printed
    # Not vacuous: every other criterion was still read off the same run.
    assert "[PASS] 1." in printed


def test_the_configuration_may_be_supplied_from_outside_the_run_directory(tmp_path):
    """The other half of the same case: a resolved configuration recovered from elsewhere makes the
    geometry criterion evaluable again."""
    run = write_run(tmp_path / "run", with_config=False)
    elsewhere = write_run(tmp_path / "reference")

    assert check_run.main(
        run_dir=str(run),
        config=str(elsewhere / "model_checkpoints" / "resolved_config.yaml"),
    ) == 0


# =================================================================================================
# Criterion 5: two evaluations
# =================================================================================================
def test_two_identical_run_directories_satisfy_the_determinism_criterion(tmp_path, capsys):
    """Identical anchor indices are necessary and not sufficient -- the reparameterisation draw and
    the permutation generator move too -- so the comparison is over the whole row set."""
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b")

    code = check_run.main(run_dir=str(first), second_run_dir=str(second))

    assert code == 0
    assert statuses(capsys)[5] == PASS


def test_a_mismatched_pair_fails_the_determinism_criterion(tmp_path, capsys):
    """Below the last printed digit is exactly where a re-tiled anchor set or a moved draw would
    show, which is why the rows are compared as text rather than parsed first."""
    first = write_run(tmp_path / "a")
    second = write_run(tmp_path / "b", rows=broken("val/source_conditioned_kl_raw", 0.5000001))

    code = check_run.main(run_dir=str(first), second_run_dir=str(second))

    assert code == 1
    assert statuses(capsys)[5] == FAIL


def test_a_second_directory_with_a_different_column_set_fails(tmp_path, capsys):
    """A metric surface that changed between the two evaluations is a difference in what was
    measured, not in what was measured *to*, and reads as identical to any row-by-row comparison
    that intersected the columns first."""
    first = write_run(tmp_path / "a")
    rows = [_row(epoch) for epoch in range(ROWS)]
    for row in rows:
        row["val/an_extra_column"] = 1.0
    second = write_run(tmp_path / "b", rows=rows)

    code = check_run.main(run_dir=str(first), second_run_dir=str(second))

    assert code == 1
    assert statuses(capsys)[5] == FAIL


# =================================================================================================
# Tier 2 is reported, never gated
# =================================================================================================
def test_every_tier_two_number_is_printed_beside_its_name(tmp_path, capsys):
    """The tier a number belongs to has to be visible in the output, or the distinction the record
    draws between the two survives only in the record."""
    write_run(tmp_path)

    check_run.main(run_dir=str(tmp_path))

    printed = capsys.readouterr().out
    assert "Tier 1 -- must hold" in printed
    assert "reported and interpreted" in printed
    for name in (
        "val/source_conditioned_kl_raw",
        "val/kld_source_null",
        "val/kld_active_frac",
        "val/logvar_prior_floor_frac",
        "val/shuffle_penalty",
        "val/source_lag_warmth_frac_st",
        "val/source_lag_warmth_frac_ph",
        "val/pred_gap_warm spread",
    ):
        assert name in printed, name
    # The one comparison the record calls the most important number on the page: the coupling
    # readout beside the floor the availability clock alone induces.
    assert "attributable to source variation" in printed
    # And the trajectory, because criterion 6 asks whether the readout is still rising at the end.
    assert "rising by" in printed


def test_extreme_tier_two_values_do_not_move_the_exit_code(tmp_path, capsys):
    """A collapsed latent, a pinned prior scale, a coupling readout identical to its own null floor
    and a source warmth of zero -- every tier-2 alarm at once. There is no prior against which a
    threshold on any of them could have been calibrated, so the run is still scored on tier 1
    alone."""
    rows = [_row(epoch) for epoch in range(ROWS)]
    for row in rows:
        row["val/kld_active_frac"] = 0.0
        row["val/logvar_prior_floor_frac"] = 1.0
        row["val/source_conditioned_kl_raw"] = 0.0
        row["val/kld_source_null"] = 0.0
        row["val/source_lag_warmth_frac_st"] = 0.0
        row["val/source_lag_warmth_frac_ph"] = 0.0
        row["val/shuffle_penalty"] = -5.0
    write_run(tmp_path, rows=rows)

    code = check_run.main(run_dir=str(tmp_path))

    assert code == 0
    assert statuses(capsys)[1] == PASS


def test_an_absent_tier_two_column_is_reported_as_absent(tmp_path, capsys):
    """Rather than as zero. A column the framework never emitted and one that read zero are
    different findings, and the second is the one that would be interpreted."""
    rows = [_row(epoch) for epoch in range(ROWS)]
    for row in rows:
        del row["val/kld_source_null"]
    write_run(tmp_path, rows=rows)

    code = check_run.main(run_dir=str(tmp_path))

    assert code == 0
    assert "val/kld_source_null: absent" in capsys.readouterr().out
