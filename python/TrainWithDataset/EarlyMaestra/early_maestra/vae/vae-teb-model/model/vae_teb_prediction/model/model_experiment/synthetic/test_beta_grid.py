r"""Pytest checks for the ``beta_grid`` mode (beta x M x TE sweep).

Covers the three pieces that do **not** require a trained checkpoint:

* :func:`gpu_pool._cells_beta_grid` enumerates ``len(settings) x len(beta)``
  cells with the $\beta$-namespaced ``beta_grid/<base>/b<token>`` run-tag and a
  distinct ``loss.kld_beta`` patch per cell.
* :func:`beta_sweep.run_beta_grid` with ``train_missing=False`` /
  ``build_missing=False`` walks the skip path cleanly, writing an empty
  ``summary.csv`` / ``analysis.json`` without training.
* :func:`beta_sweep._make_beta_grid_plots` renders the three multi-line figures
  from synthetic rows (no model / checkpoint needed).

The $(M, \mathrm{TE})$ grid is narrowed to a single cell and the MC sample
budget is tiny so :func:`evaluate_te.enumerate_sweep` (the slowest step, a
bisection over the analytic block TE) finishes in a few seconds. Run from the
repo root with ``python -m pytest``.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    beta_sweep as bs,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    gpu_pool as gp,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_active_benchmark,
)

_CONFIG_PATH = Path(__file__).resolve().parent / "config_synth.yaml"


def _tiny_grid_config(data_dir: Path, results_dir: Path) -> Dict[str, Any]:
    """Load ``config_synth.yaml`` narrowed to a 1-cell G2 grid + 2 betas.

    G2 (smooth ARX) is used rather than G1 because its analytic block TE is
    closed-form (``c_for_mean_te_block_arx``) -- exact, fast and free of the
    Monte-Carlo noise that makes G1's bisection unreliable at a tiny sample
    budget. The enumeration / skip-path logic under test is benchmark-agnostic.

    Args:
        data_dir: Absolute path used as ``paths.data_dir``.
        results_dir: Absolute path used as ``paths.results_dir``.

    Returns:
        The benchmark-resolved config with a single ``(M=1, TE=0.1)`` sweep cell
        and a 2-value ``beta_sweep.grid``.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["experiment"]["benchmark"] = "G2"
    config = resolve_active_benchmark(raw)
    config["paths"]["data_dir"] = str(data_dir)
    config["paths"]["results_dir"] = str(results_dir)
    config["runtime"]["device"] = "cpu"
    config["beta_sweep"]["grid"] = [1.0e-4, 1.0e-2]
    config["beta_sweep"]["beta_grid"] = {
        "m_grid": [1],
        "target_te_grid": [0.1],
    }
    return config


def test_cells_beta_grid_enumeration(tmp_path):
    """``_cells_beta_grid`` crosses each (M, TE) setting with every beta."""
    config = _tiny_grid_config(tmp_path / "data", tmp_path / "results")
    cells = gp._cells_beta_grid(config, build=False)

    betas = config["beta_sweep"]["grid"]
    # m_grid=[1] x target_te_grid=[0.1] -> 1 setting; x 2 betas -> 2 cells.
    assert len(cells) == 1 * len(betas)
    for cell in cells:
        assert cell.benchmark == "G2"
        assert cell.run_tag.startswith("beta_grid/")
        assert cell.run_tag.split("/")[1].endswith("_m1")  # base run-tag, M=1
        assert "loss.kld_beta" in cell.patches
    # The beta patches are exactly the configured grid (distinct per cell).
    patched = sorted(c.patches["loss.kld_beta"] for c in cells)
    assert patched == sorted(float(b) for b in betas)
    # All cells share the single (beta-independent) dataset cache.
    assert len({c.data_tag for c in cells}) == 1


def test_run_beta_grid_skip_path(tmp_path):
    """``run_beta_grid`` writes empty artifacts when nothing is built/trained."""
    config = _tiny_grid_config(tmp_path / "data", tmp_path / "results")
    result = bs.run_beta_grid(
        config, build_missing=False, train_missing=False,
    )
    assert result["axis"] == "beta_grid"
    assert result["rows"] == []
    # 1 setting x 2 betas were all skipped (no cached dataset).
    assert len(result["skipped"]) == len(config["beta_sweep"]["grid"])
    out_dir = Path(result["out_dir"])
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "analysis.json").is_file()
    # No plots when < 2 rows were evaluated.
    assert not (out_dir / "kbar_vs_beta__byTE.pdf").is_file()


def _fake_rows() -> List[Dict[str, Any]]:
    """Synthetic per-cell rows spanning a 2 x 2 x 2 (M, TE, beta) grid."""
    rows: List[Dict[str, Any]] = []
    for m in (1, 4):
        for te in (0.1, 1.0):
            for beta in (1.0e-4, 1.0e-2):
                rows.append({
                    "M": m, "target_te": te, "beta": beta,
                    "te_true": te, "k_bar": te * (1.0 + 0.1 * m) - beta,
                })
    return rows


def test_make_beta_grid_plots_writes_figures(tmp_path):
    """``_make_beta_grid_plots`` renders the three multi-line figures."""
    bs._make_beta_grid_plots(_fake_rows(), tmp_path)
    for stem in (
        "kbar_vs_beta__byTE", "kbar_vs_beta__byM", "kbar_vs_te__byBeta",
    ):
        assert (tmp_path / f"{stem}.pdf").is_file(), stem
        assert (tmp_path / f"{stem}.png").is_file(), stem
