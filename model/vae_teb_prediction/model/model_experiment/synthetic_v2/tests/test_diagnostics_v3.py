r"""S2-T03: v3 diagnostics passthrough.

The v3 training metrics dict gains ``kld_raw`` / ``kld_train`` / ``kld_active_frac`` (the
bottleneck-health signals) and ``mean_logvar_{full,base}`` (variance-collapse monitors),
each key-presence guarded and -- crucially -- omitted rather than passed as ``None``
(``_log_metrics`` cannot log a ``None``). A v1 run logs exactly today's set, and the new
names reach ``training_curves.html`` via ``_HTML_METRIC_ORDER``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    visualize_v2,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    SyntheticSeqVaeLagAttnV2Pl,
    build_model,
)

_CPU = torch.device("cpu")
_TINY = {
    "sequence_length": 32, "d_model": 16, "d_z": 8, "horizon": 4, "warmup_period": 2,
    "c_y": 87, "c_u": 101, "use_up_st": True, "max_lag": 8, "num_heads": 4, "d_head": 4,
    "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
}
_V3 = {**_TINY, "class": "SeqVaeLagAttnV3",
       "v3": {"posterior_logvar": "residual", "kld_support": "anchor",
              "logvar_bound": "smooth"}}
_V3_KEYS = ("kld_raw", "kld_train", "kld_active_frac", "mean_logvar_full",
            "mean_logvar_base")


def _batch(B=2, T=32):
    torch.manual_seed(0)
    return types.SimpleNamespace(
        fhr_st=torch.randn(B, T, 43), fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43), up_ph=torch.randn(B, T, 58),
        weight=torch.ones(B, T))


def _metrics(cfg, *, likelihood, sigma_obs, free_bits=0.0):
    model, _ = build_model(cfg, _CPU)
    wrap = SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood=likelihood, sigma_obs=sigma_obs,
        free_bits=free_bits, detach_baseline_in_full=True)
    wrap.eval()
    with torch.no_grad():
        _, metrics = wrap.compute_loss_and_metrics(_batch(), 0, "train")
    return metrics


def test_v3_metrics_present_and_not_none() -> None:
    metrics = _metrics(_V3, likelihood="gaussian_nll", sigma_obs="learned", free_bits=0.2)
    for key in _V3_KEYS:
        assert key in metrics, f"v3 metric {key} missing"
    # No metric is None at logging time (the guard omits None rather than passing it).
    for key, val in metrics.items():
        assert val is not None, f"metric {key} is None"


def test_v1_metrics_are_todays_set() -> None:
    """A v1 (mse) run logs none of the v3-ONLY signals, and never a None."""
    metrics = _metrics(_TINY, likelihood="mse", sigma_obs=1.0)
    # The raw/train KL split and the active-frac are v3-only (the free-bits objective).
    for key in ("kld_raw", "kld_train", "kld_active_frac"):
        assert key not in metrics, f"v1 run unexpectedly logged {key}"
    # No metric is ever None (the guard omits None rather than passing it -- the S2-T03 fix
    # for the previously-unconditional ``mean_logvar_*`` entries).
    for key, val in metrics.items():
        assert val is not None, f"v1 metric {key} is None"


def test_html_metric_order_includes_v3_signals() -> None:
    """The new names are in the legend-ordering hint (so they render in order)."""
    for key in ("kld_raw", "kld_train", "kld_active_frac"):
        assert key in visualize_v2._HTML_METRIC_ORDER


def test_html_renders_new_traces(tmp_path) -> None:
    """A synthetic metrics.csv carrying the v3 columns renders them into the HTML."""
    plot = getattr(visualize_v2, "plot_loss_curves_html", None)
    if plot is None:  # pragma: no cover - plotting optional
        import pytest
        pytest.skip("plot_loss_curves_html not available")
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        "epoch,train/kld_raw_epoch,train/kld_train_epoch,train/kld_active_frac_epoch\n"
        "0,0.0,0.20,0.1\n1,0.05,0.20,0.3\n2,0.12,0.22,0.6\n",
        encoding="utf-8",
    )
    out_stem = tmp_path / "training_curves"
    result = plot(csv, out_stem)
    if not result:  # plotly missing -> non-fatal, nothing rendered
        import pytest
        pytest.skip("plotly not installed; HTML not emitted")
    html_files = list(tmp_path.glob("*.html"))
    assert html_files, "no HTML emitted"
    text = html_files[0].read_text(encoding="utf-8", errors="ignore")
    assert "kld_raw" in text and "kld_active_frac" in text
