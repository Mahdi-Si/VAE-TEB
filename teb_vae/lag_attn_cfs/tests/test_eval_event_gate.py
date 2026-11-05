r"""Known answers for the event-gate states and the ``event_gate`` analysis.

The analysis must keep two shut states apart: **forced shut** ($o_t = 0$, the UA mask removed every
key) and **model shut** ($o_t = 1$, $g_t < g_{\mathrm{shut}}$). Both have an applied gate of $0$,
so a test that checked the applied gate alone would pass on an analysis that merged them.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval.analyses import event_gate
from teb_vae.lag_attn_cfs.gate_states import (
    FORCED_SHUT,
    MODEL_SHUT,
    OPEN,
    PARTLY_OPEN,
    gate_states,
    run_lengths,
)


def test_each_anchor_lands_in_the_state_its_gate_and_mask_give() -> None:
    """A closed key set wins over any learned value; otherwise the two thresholds split $g_t$."""
    learned = np.array([0.1, 0.5, 0.9, 0.95, np.nan])
    opened = np.array([1, 1, 1, 0, 1], dtype=bool)
    states = gate_states(learned, opened)
    assert states.tolist() == [MODEL_SHUT, PARTLY_OPEN, OPEN, FORCED_SHUT, -1]
    # Without a UA mask no anchor can be forced shut.
    assert FORCED_SHUT not in gate_states(learned[:4]).tolist()


def test_run_lengths_count_each_unbroken_run() -> None:
    """Three runs of lengths 2, 1 and 3, read in time order."""
    flags = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert run_lengths(flags).tolist() == [2, 1, 3]
    assert run_lengths(np.zeros(4, dtype=bool)).size == 0


def _context(per_anchor: pd.DataFrame) -> types.SimpleNamespace:
    """An analysis context over one per-anchor table and a matching per-sample table."""
    per_sample = pd.DataFrame(
        {"guid": sorted(per_anchor["guid"].unique()), "clinical_class": "healthy"}
    )
    return types.SimpleNamespace(
        collection=types.SimpleNamespace(per_anchor=per_anchor, per_sample=per_sample)
    )


def _table(with_learned: bool = True) -> pd.DataFrame:
    """Eight anchors of one segment: 2 forced shut, 2 model shut, 2 partly open, 2 open."""
    learned = np.array([0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
    opened = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
    frame = pd.DataFrame({
        "guid": "G0", "epoch": -3600.0, "anchor": np.arange(10, 18),
        "event_gate": learned * opened, "source_open": opened,
        "kld_per_t": np.arange(8, dtype=float), "mc_pred_gap": np.arange(8, dtype=float),
    })
    if with_learned:
        frame["event_gate_learned"] = learned
    return frame


def test_the_analysis_reports_each_state_share_and_the_model_shut_run(tmp_path) -> None:
    """A quarter of the anchors per state, and one model-shut run of two steps."""
    result = event_gate.run_event_gate_analysis(
        _context(_table()), eval_config={"bootstrap_resamples": 20}, output_dir=tmp_path
    )
    headline = result["headline"]
    for name in ("forced_shut_frac", "model_shut_frac", "partly_open_frac", "open_frac"):
        assert headline[name] == 0.25, name
    assert result["blinks"]["n"] == 1
    assert result["blinks"]["median_s"] == 2 * event_gate.SECONDS_PER_STEP
    by_state = {row["state"]: row for row in result["states"]}
    assert by_state["model_shut"]["kl_nats"] == 2.5
    assert by_state["open"]["gain_nats"] == 6.5


def test_a_run_without_the_learned_gate_recovers_the_same_states(tmp_path) -> None:
    """Where a key is open the applied gate is the learned one, so every state count is exact."""
    with_learned = event_gate.run_event_gate_analysis(
        _context(_table()), eval_config={"bootstrap_resamples": 20}, output_dir=tmp_path / "a"
    )
    recovered = event_gate.run_event_gate_analysis(
        _context(_table(with_learned=False)), eval_config={"bootstrap_resamples": 20},
        output_dir=tmp_path / "b",
    )
    assert recovered["plan"]["learned_gate_recovered"] is True
    assert recovered["headline"] == with_learned["headline"]


def test_a_model_with_no_gate_and_no_mask_records_a_skip(tmp_path) -> None:
    """Nothing to classify is a recorded skip, not a table of zeros."""
    frame = pd.DataFrame({"guid": ["G0"], "epoch": [-3600.0], "anchor": [1], "kld_per_t": [0.0]})
    result = event_gate.run_event_gate_analysis(
        _context(frame), eval_config={}, output_dir=tmp_path
    )
    assert result["skipped"] is True
    assert result["n_samples"] is None
