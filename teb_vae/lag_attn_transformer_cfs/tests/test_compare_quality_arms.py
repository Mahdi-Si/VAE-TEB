r"""The quality-arm comparison reads its ``event`` window from the evaluations it compares.

A hard-coded window would split the anchors at a reach neither evaluation used, so the default is
the ``eval_config.event_lag_window_s`` both arms resolved, and a disagreement is refused.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn_transformer_cfs.eval.compare_quality_arms import resolve_event_window_s


def _arm(window):
    return {"table": None, "switches": {}, "event_lag_window_s": window}


def test_the_window_is_the_one_both_evaluations_resolved() -> None:
    assert resolve_event_window_s(None, _arm(120.0), _arm(120.0)) == 120.0
    # One arm without a dumped config still names the window of the other.
    assert resolve_event_window_s(None, _arm(160.0), _arm(None)) == 160.0


def test_an_explicit_window_wins() -> None:
    assert resolve_event_window_s(90.0, _arm(120.0), _arm(160.0)) == 90.0


@pytest.mark.parametrize("windows", [(120.0, 160.0), (None, None)])
def test_two_windows_or_none_are_refused(windows) -> None:
    with pytest.raises(ValueError, match="event-window-s"):
        resolve_event_window_s(None, _arm(windows[0]), _arm(windows[1]))
