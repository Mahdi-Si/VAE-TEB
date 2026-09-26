r"""The diagnostic page's forecast rows are bound with this model's own facts.

The page's builders live in the conv-LSTM causal cell and are tested there; this package writes no
figure module of its own. What its composition can get wrong is the *binding*: the forecast rows
are bound with values the page cannot recover from the arrays it is handed -- which declared channel
each decoder output is, where the two stored blocks meet on that axis, the stride a training step
tiles at, and the likelihood, coverage floor and density structure the per-window score row scores
under -- and each has to be read off **this** net and this run's objective. A wrong one draws a real
forecast at the wrong time, or scores a window under another density, with no shape error in it.
The rendered page itself is exercised by ``test_heatmap_timestamps.py``.
"""
from __future__ import annotations

from teb_vae.lag_attn_cfs import sample_page as causal_page

from .conftest import CAUSAL_ST_WIDTH, TINY_STRIDE


def test_the_forecast_rows_are_the_causal_builder_bound_with_this_nets_facts(task):
    """By object identity on the underlying function, and every bound value compared against where
    the objective takes it rather than against a literal."""
    module = task()
    rows = module.forecast_rows
    model = module.orig_model

    assert rows.func is causal_page.causal_forecast_rows
    assert rows.keywords["keep_index"] is model.target_gate.keep_index
    assert rows.keywords["block_split"] == CAUSAL_ST_WIDTH
    assert rows.keywords["training_stride"] == model.anchor_stride == TINY_STRIDE
    assert rows.keywords["likelihood"] == module.hparams["likelihood"]
    assert rows.keywords["coverage_floor"] == model.coverage_floor
    assert rows.keywords["target_forecast_shift"] == model.target_forecast_shift
    # The rest of the objective's density, off the net: ``None`` on a net that built neither term.
    assert rows.keywords["cell_mask"] is None and rows.keywords["ar_coef"] is None
    # No resolved budget on a hand-built task, so the forecast rows' axis cannot state a constant.
    assert rows.keywords["forecast_clock_delay_s"] is None
