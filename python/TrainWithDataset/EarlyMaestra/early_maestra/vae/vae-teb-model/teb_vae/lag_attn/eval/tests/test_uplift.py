r"""Tests for the uplift analysis.

Two cases carry this file, and **both** are required -- either alone passes on a broken analysis.

The **positive case** runs a ``perturb_full_pathway`` model and requires a non-zero uplift. An
untouched model would pass a "there is an uplift column" test vacuously, because
``_zero_init_delta_heads`` zeroes ``residual_decoder.mean_head`` and $\delta\mu_{\mathrm{src}}$
is then identically zero *whatever* $z$ is.

The **negative case** runs an untouched model under ``likelihood='mse'`` and requires the uplift
to be exactly zero with the flag set. The ``mse`` restriction is not incidental: under
``gaussian_nll`` with ``sigma_obs='learned'`` the full and baseline losses read different
variance heads, so they differ even when the mean correction is zero, and this case would fail
on a perfectly correct analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval.analyses import uplift as uplift_analysis

#: Objective overrides putting the analysis under plain squared error -- see the module docstring.
MSE_OBJECTIVE = {"likelihood": "mse", "sigma_obs": 1.0}


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-sample frame)``."""
    torch.manual_seed(7)
    summary = uplift_analysis.run_uplift_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir,
        probe={"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4},
    )
    frame = pd.read_csv(
        Path(output_dir) / uplift_analysis.ANALYSIS_DIRNAME / "per_sample.csv"
    )
    return summary, frame


def test_a_live_pathway_produces_a_nonzero_uplift(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The positive case. An untouched model would pass a weaker test vacuously."""
    runner = make_eval_runner(perturb=True, output_dir=tmp_path / "runner")
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "live"
    )

    assert set(["l_full", "l_base", "uplift_abs", "uplift_rel"]) <= set(frame.columns)
    assert len(frame) == 4
    assert np.abs(frame["uplift_abs"].to_numpy()).max() > 0.0
    assert not summary["near_zero_uplift"]


def test_a_dead_pathway_under_mse_produces_exactly_zero_uplift_and_sets_the_flag(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""The negative case, and the reason it must be run under ``'mse'``.

    On an untouched model ``residual_decoder.mean_head`` is zero, so $\mu_{\mathrm{full}} =
    \mu_{\mathrm{base}}$ exactly and the squared errors are identical. Under ``gaussian_nll``
    with a learned $\sigma$ they would still differ, through the variance heads alone.
    """
    runner = make_eval_runner(
        perturb=False, hparams=MSE_OBJECTIVE, output_dir=tmp_path / "runner"
    )
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "dead"
    )

    assert frame["uplift_abs"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    assert frame["l_full"].to_numpy() == pytest.approx(frame["l_base"].to_numpy())
    assert summary["near_zero_uplift"] is True


def test_the_learned_variance_heads_move_the_uplift_even_with_a_dead_mean_pathway(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Why a near-zero uplift is flagged rather than treated as a verdict.

    The same untouched model that gives exactly zero uplift under ``'mse'`` gives a non-zero one
    under ``gaussian_nll`` with ``sigma_obs='learned'``, because the two losses read different
    variance heads. Reading that difference as evidence about the mean pathway would be wrong in
    both directions.
    """
    runner = make_eval_runner(
        perturb=False,
        hparams={"likelihood": "gaussian_nll", "sigma_obs": "learned"},
        output_dir=tmp_path / "runner",
    )
    _, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "nll")
    assert np.abs(frame["uplift_abs"].to_numpy()).max() > 0.0


def test_the_summary_reports_the_positive_fraction_and_the_objective(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The objective travels with the number, so a reader knows how to read it."""
    runner = make_eval_runner(hparams=MSE_OBJECTIVE, output_dir=tmp_path / "runner")
    summary, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "summary"
    )

    assert 0.0 <= summary["positive_fraction"] <= 1.0
    assert summary["likelihood"] == "mse"
    assert summary["n_samples"] == 4
    assert summary["composition"] == {"tiny_shard.hdf5": 4}
    assert Path(summary["figure"]).stat().st_size > 0
