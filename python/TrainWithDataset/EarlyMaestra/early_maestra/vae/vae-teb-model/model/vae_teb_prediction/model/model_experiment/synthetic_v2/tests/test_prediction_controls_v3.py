r"""S3-T03: the prediction-space control replaces the KL null ratio as the headline gate.

The retired gate asked for $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}} \to 0$. No honest
model can pass it: $\mathrm{KL}(q \,\|\, p)$ says "the source moved my belief", not
"...correctly", and a deranged source is still a source. The replacement scores the *forecast*,
which is checkable against the real future:

.. math::

    \mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}} < \mathcal L_{\mathrm{feat}}^{\pi(U)}

On the $\mathrm{TE}_{\mathrm{inj}} = 0$ null cells there is no true source to be wrong about, so
the three losses must *coincide* instead of ordering.

These tests drive :func:`eval_v2.prediction_controls` on hand-built per-sample arrays, so both a
passing and a failing arrangement are constructed exactly rather than hoped for.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import eval_v2  # noqa: E402


def _arrs(cells) -> dict:
    r"""Build per-sample arrays from ``[(cell_id, te_inj, feat, base, feat_shuffle), ...]``.

    Two samples per cell (identical), so the per-cell ``nanmean`` reduction is exercised
    without introducing sampling noise into the assertions.
    """
    cid, te, feat, base, fsh = [], [], [], [], []
    for c, t, f, b, s in cells:
        for _ in range(2):
            cid.append(c); te.append(t); feat.append(f); base.append(b); fsh.append(s)
    return {
        "cell_id": np.asarray(cid, dtype=np.int64),
        "te_inj": np.asarray(te, dtype=np.float64),
        "feat_loss": np.asarray(feat, dtype=np.float64),
        "base_loss": np.asarray(base, dtype=np.float64),
        "feat_loss_shuffle": np.asarray(fsh, dtype=np.float64),
        "shuffle_penalty_shuffle": np.asarray(fsh, dtype=np.float64)
        - np.asarray(feat, dtype=np.float64),
    }


def test_passing_arrangement_on_signal_cells() -> None:
    r"""``feat < base < feat_shuffle`` on every signal cell -> the gate passes."""
    arrs = _arrs([
        (1, 1.0, 1.0, 2.0, 3.0),
        (2, 2.0, 0.5, 2.0, 5.0),
        (3, 3.0, 0.2, 2.0, 6.0),
    ])
    res = eval_v2.prediction_controls(arrs, ["shuffle"])
    assert res["controls"] == ["shuffle"]
    assert res["n_signal_cells"] == 3
    assert res["overall"]["ordering_pass"] is True
    assert res["overall"]["ordering_pass_shuffle"] is True
    assert res["overall"]["ordering_pass_frac"] == 1.0
    assert res["overall"]["shuffle_penalty_shuffle"] > 0.0
    for c in (1, 2, 3):
        assert res["per_cell"][c]["ordering_pass_shuffle"] is True
        assert res["per_cell"][c]["shuffle_penalty_shuffle"] > 0.0
        assert res["per_cell"][c]["null_cell_consistent"] is None


def test_failing_arrangement_when_a_wrong_source_is_not_worse() -> None:
    r"""A corrupted source that forecasts no worse than the true one fails the gate.

    Cell 2's ``feat_loss_shuffle`` sits *below* its ``base_loss``: the model's source pathway
    adds capacity but the specific source content is not what it is exploiting.
    """
    arrs = _arrs([
        (1, 1.0, 1.0, 2.0, 3.0),   # passes
        (2, 2.0, 0.5, 2.0, 1.5),   # feat < feat_shuffle < base -> fails the right inequality
    ])
    res = eval_v2.prediction_controls(arrs, ["shuffle"])
    assert res["per_cell"][1]["ordering_pass_shuffle"] is True
    assert res["per_cell"][2]["ordering_pass_shuffle"] is False
    assert res["overall"]["ordering_pass"] is False
    assert res["overall"]["ordering_pass_frac"] == 0.5


def test_failing_arrangement_when_the_source_does_not_help() -> None:
    r"""``base <= feat`` fails the LEFT inequality: the source buys no prediction gain."""
    arrs = _arrs([(1, 1.0, 2.0, 2.0, 9.0)])
    res = eval_v2.prediction_controls(arrs, ["shuffle"])
    assert res["per_cell"][1]["ordering_pass_shuffle"] is False
    assert res["overall"]["ordering_pass"] is False


def test_null_cells_are_checked_for_coincidence_not_ordering() -> None:
    r"""On ``te_inj == 0`` the three losses must collapse together; the gate is not asserted."""
    arrs = _arrs([
        (0, 0.0, 2.0, 2.0, 2.0),        # null: all three coincide
        (1, 1.0, 1.0, 2.0, 3.0),        # signal: passes
    ])
    res = eval_v2.prediction_controls(arrs, ["shuffle"])
    assert res["n_signal_cells"] == 1
    assert res["per_cell"][0]["ordering_pass_shuffle"] is None
    assert res["per_cell"][0]["null_cell_consistent"] is True
    # The null cell must not drag the overall gate down.
    assert res["overall"]["ordering_pass"] is True


def test_null_cell_inconsistency_is_reported() -> None:
    r"""A null cell whose losses diverge is flagged (the source is being used where it can't be)."""
    arrs = _arrs([(0, 0.0, 1.0, 2.0, 5.0)])
    res = eval_v2.prediction_controls(arrs, ["shuffle"])
    assert res["per_cell"][0]["null_cell_consistent"] is False
    assert res["n_signal_cells"] == 0
    assert res["overall"]["ordering_pass"] is False   # no signal cells -> nothing passed


def test_null_cell_tolerance_is_relative_to_base_loss() -> None:
    r"""``null_tol`` scales with the cell's ``base_loss``: the losses are MSEs of arbitrary scale.

    The same 2% spread must read as consistent whether the losses live near 1.0 or near 100.0.
    An absolute tolerance would call the second cell wildly inconsistent for no physical reason.
    """
    small = _arrs([(0, 0.0, 1.00, 1.00, 1.02)])
    large = _arrs([(0, 0.0, 100.0, 100.0, 102.0)])
    for arrs in (small, large):
        res = eval_v2.prediction_controls(arrs, ["shuffle"], null_tol=0.05)
        assert res["per_cell"][0]["null_cell_consistent"] is True
    # ...and a 10% spread trips it at either scale.
    for arrs in (_arrs([(0, 0.0, 1.0, 1.0, 1.1)]), _arrs([(0, 0.0, 100.0, 100.0, 110.0)])):
        res = eval_v2.prediction_controls(arrs, ["shuffle"], null_tol=0.05)
        assert res["per_cell"][0]["null_cell_consistent"] is False


def test_absent_control_arrays_are_skipped() -> None:
    r"""A control with no ``feat_loss_<ctrl>`` column is dropped rather than raising."""
    arrs = _arrs([(1, 1.0, 1.0, 2.0, 3.0)])
    res = eval_v2.prediction_controls(arrs, ["shuffle", "reverse"])
    assert res["controls"] == ["shuffle"]
    assert "ordering_pass_reverse" not in res["overall"]


def test_null_ratios_are_annotated_as_a_non_vanishing_readout() -> None:
    r"""``null_ratios`` keeps its numbers but can no longer be read as a ``-> 0`` gate."""
    arrs = {
        "cell_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "te_inj": np.asarray([0.0, 0.0, 2.0, 2.0]),
        "kbar": np.asarray([1.0, 1.0, 2.0, 2.0]),
        "kbar_shuffle": np.asarray([1.0, 1.0, 2.1, 2.1]),
    }
    res = eval_v2.null_ratios(arrs, ["shuffle"])
    # The number itself is unchanged: 2.1 / 2.0 over the single signal cell.
    assert res["shuffle"]["mean_ratio"] == 1.05
    assert res["shuffle"]["expected_to_vanish"] is False
    assert "Finding F2" in res["shuffle"]["note"]
    # The pre-existing readers keep working.
    assert set(res["shuffle"]["per_cell"]) == {0, 1}
