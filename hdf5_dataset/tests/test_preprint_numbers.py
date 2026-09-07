r"""The fidelity numbers printed in the model preprint, against the run that measured them.

A document that compiles is not a document that is right.  The four cells' ``test_docs.py``
already apply this technique to their ``DESIGN.md``: read the one section that states a measured
number, recompute it from the artefact it came from, and fail on drift.  The preprints had no
equivalent, and their fidelity table is exactly the kind of number that goes stale silently --
it is a median over a rebuilt shard, so an operator who re-runs the comparison tool and does not
re-read the table leaves a published correlation describing a dataset that no longer exists.

**Two absences, two different meanings.**

*The measurement CSV.*  ``output/`` is git-ignored, so a fresh clone and the production box have
never run the comparison tool.  Absent is a skip, exactly as it is for every other measurement-
dependent test in this package.  What the skip must not cover is a CSV built against a different
shard or at a different leg alignment, and it does not: that case is already refused by
``test_causal_torch.py::test_the_measurement_csv_describes_this_bank_and_this_shard``, which runs
off the same artefact and the same provenance record.

*The preprint tree.*  ``teb_vae/lag_attn_cfs/docs/latex_lag_attn_cfs/`` is not tracked either, so
a checkout that has the code and not the document is a real configuration rather than a broken
one.  Absent is a skip; **present and disagreeing is a failure**, and the message names the block,
the column, the stated value and the measured one.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from hdf5_dataset.tests.conftest import MEASUREMENTS_PATH, requires_measurements

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The model preprint's transform section, which is where the fidelity table lives.
TRANSFORM_SECTION = (
    _REPO_ROOT
    / "teb_vae"
    / "lag_attn_cfs"
    / "docs"
    / "latex_lag_attn_cfs"
    / "sections"
    / "transform.tex"
)

requires_preprint = pytest.mark.skipif(
    not TRANSFORM_SECTION.exists(), reason=f"no model preprint at {TRANSFORM_SECTION}"
)

#: The drop rule's threshold: a channel whose warm-up exceeds a stored segment never reaches a
#: file, so the table's medians are taken over the stored channels alone.
STORED_STEPS = 330

#: The table's numeric columns, in the order they appear in the row, against the CSV column each
#: is the median of.  Positional rather than parsed from the header, because a LaTeX ``tabular``
#: header is a layout object and reading one is how a test starts passing for the wrong reason;
#: the header phrases are asserted separately, below, so a column reordering fails here rather
#: than silently comparing the wrong pair.
FIDELITY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("unaligned", "r_at_predicted_lag"),
    ("envelope-aligned", "r_at_predicted_lag_envelope"),
    ("at the best lag", "r_at_best_lag"),
    ("at zero lag", "r_at_zero_lag"),
)

#: Blocks in the order the table lists them, and whether each has a pair of legs to align.  A
#: scattering channel is one filter, so its aligned entry is a structural absence rather than an
#: unmeasured one, and the table prints a dash there.
FIDELITY_ROWS: Tuple[Tuple[str, bool], ...] = (
    ("fhr_st", False),
    ("up_st", False),
    ("fhr_ph", True),
    ("up_ph", True),
)

#: How far a printed median may sit from the measured one.  Three decimals are printed, so half a
#: unit in the last place is $0.0005$; the allowance is $0.002$, which absorbs a rounding
#: convention without admitting a stale value -- the numbers this catches move by tenths.
FIDELITY_TOLERANCE = 0.002

#: One row of ``tab:transformfidelity``: a block name, then its cells up to the row terminator.
_ROW = re.compile(r"\\texttt\{(fhr|up)\\_(st|ph)\}((?:[^\\]|\\(?!\\))*)\\\\")

#: A signed decimal anywhere in a cell, whether or not it is wrapped in ``\mathbf``.
_NUMBER = re.compile(r"-?\d+\.\d+")


def _fidelity_table() -> str:
    r"""Return the whole ``table`` environment that carries ``tab:transformfidelity``.

    Anchored on the environment rather than on ``\toprule``: a caption may precede or follow its
    label, and a rule-anchored slice that guessed wrong would silently read the *neighbouring*
    table and compare a correct document against the wrong numbers.

    Returns:
        The environment, ``\begin{table}`` to ``\end{table}`` inclusive.

    Raises:
        AssertionError: If the label is absent, which means the table was renamed or removed and
            this test must move with it.
    """
    text = TRANSFORM_SECTION.read_text(encoding="utf-8")
    assert "tab:transformfidelity" in text, (
        f"{TRANSFORM_SECTION.name} carries no tab:transformfidelity. The fidelity table is what "
        f"this test exists to check; if it moved, move this test with it."
    )
    label = text.index("tab:transformfidelity")
    start = text.rindex(r"\begin{table}", 0, label)
    stop = text.index(r"\end{table}", label) + len(r"\end{table}")
    return text[start:stop]


def _measured_medians() -> Dict[str, Dict[str, float]]:
    """Median of each fidelity column, per block, over the stored channels.

    Mirrors ``gen_model_data.py::write_fidelity``'s restriction exactly: the same drop-rule
    threshold over the same CSV, so the table, the figure and this check are three readings of
    one number rather than three numbers.

    Returns:
        ``{block: {csv_column: median}}``.  A column that is ``nan`` for every channel of a block
        -- the aligned column on a scattering block -- is absent from that block's mapping.
    """
    rows: Dict[str, List[Dict[str, str]]] = {}
    with MEASUREMENTS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["block"], []).append(row)

    def value(row: Dict[str, str], key: str) -> float:
        raw = row[key]
        return float("nan") if raw in ("", "nan") else float(raw)

    out: Dict[str, Dict[str, float]] = {}
    for block, _has_legs in FIDELITY_ROWS:
        stored = [
            row for row in rows[block] if value(row, "causal_warmup_steps") <= STORED_STEPS
        ]
        medians: Dict[str, float] = {}
        for _phrase, column in FIDELITY_COLUMNS:
            finite = np.array(
                [value(row, column) for row in stored if not np.isnan(value(row, column))]
            )
            if finite.size:
                medians[column] = float(np.median(finite))
        out[block] = medians
    return out


@requires_preprint
def test_the_fidelity_table_names_its_columns_in_the_order_this_test_reads_them() -> None:
    """The header phrases, in order, so a reordered table fails here rather than comparing wrongly.

    The row grammar below is positional -- it takes the $n$-th number of a row as the $n$-th
    column -- which is only safe while the header says what that order is.  Asserting the phrases
    separately is what keeps a silent column swap from turning into a silently passing comparison.
    """
    body = _fidelity_table()
    positions = []
    for phrase, _column in FIDELITY_COLUMNS:
        assert phrase in body, (
            f"tab:transformfidelity has no {phrase!r} column. This test reads its columns "
            f"positionally and cannot tell which number is which without the header saying so."
        )
        positions.append(body.index(phrase))
    assert positions == sorted(positions), (
        f"tab:transformfidelity names its columns in a different order than "
        f"{[phrase for phrase, _ in FIDELITY_COLUMNS]}; the positional read below would compare "
        f"the wrong pairs."
    )


@requires_preprint
@requires_measurements
def test_the_preprint_fidelity_table_matches_the_measured_medians() -> None:
    r"""Every correlation printed in ``tab:transformfidelity``, against the CSV it came from.

    This is the check the document could not make for itself.  The table states four medians per
    block over four blocks, and each is a median over the stored channels of the comparison run --
    reproducible in three lines, and otherwise unverifiable by anything but a reader's memory.

    The two scattering rows carry a dash in the aligned column and are asserted to: a scattering
    channel is one filter, so there are no legs to align, and printing a number there would be
    inventing one.
    """
    body = _fidelity_table()
    measured = _measured_medians()

    found = {}
    for match in _ROW.finditer(body):
        block = f"{match.group(1)}_{match.group(2)}"
        found[block] = match.group(3)
    assert set(found) == {name for name, _ in FIDELITY_ROWS}, (
        f"tab:transformfidelity lists {sorted(found)}, not "
        f"{sorted(name for name, _ in FIDELITY_ROWS)}"
    )

    for block, has_legs in FIDELITY_ROWS:
        cells = found[block]
        # The first number in the row is the "usable" fraction's numerator, which is not a
        # correlation; the fraction is dropped by taking numbers with a decimal point only.
        numbers = [float(text) for text in _NUMBER.findall(cells)]
        expected_count = len(FIDELITY_COLUMNS) if has_legs else len(FIDELITY_COLUMNS) - 1
        assert len(numbers) == expected_count, (
            f"{block} row of tab:transformfidelity carries {len(numbers)} decimals, expected "
            f"{expected_count}: {cells.strip()!r}"
        )
        if not has_legs:
            assert "---" in cells, (
                f"{block} has no pair of legs to align, so its aligned cell must be a dash "
                f"rather than a number: {cells.strip()!r}"
            )

        columns = [
            column
            for _phrase, column in FIDELITY_COLUMNS
            if has_legs or column != "r_at_predicted_lag_envelope"
        ]
        for column, stated in zip(columns, numbers):
            assert column in measured[block], (
                f"{block} states {stated} for {column}, but every channel's value in "
                f"{MEASUREMENTS_PATH.name} is nan"
            )
            assert stated == pytest.approx(measured[block][column], abs=FIDELITY_TOLERANCE), (
                f"tab:transformfidelity states {stated} for {block} / {column}; "
                f"{MEASUREMENTS_PATH} measures {measured[block][column]:.4f}. Re-read the table "
                f"against the run, or re-run the comparison tool -- one of the two is stale."
            )


@requires_preprint
@requires_measurements
def test_the_preprint_states_the_alignment_beats_the_shipped_arm_on_both_phase_blocks() -> None:
    """The claim the table is printed to support, checked as a claim and not as four digits.

    A table can be numerically correct and still be quoted backwards in the paragraph beside it.
    The direction is the finding: on both phase blocks the envelope-aligned median at the
    predicted delay exceeds the unaligned one by a wide margin, and on neither scattering block
    does the question arise.
    """
    measured = _measured_medians()
    for block, has_legs in FIDELITY_ROWS:
        if not has_legs:
            continue
        aligned = measured[block]["r_at_predicted_lag_envelope"]
        shipped = measured[block]["r_at_predicted_lag"]
        assert aligned > 5.0 * abs(shipped), (
            f"{block}: aligned {aligned:.4f} against shipped {shipped:.4f}. The preprint states "
            f"the alignment as the repair; on this measurement it is not one."
        )
