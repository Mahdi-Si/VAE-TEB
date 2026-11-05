r"""Put the verdicts and the main readouts of several official evaluation runs side by side.

Reads ``<run>/eval_results/summary.json`` for each run. The readouts are per-anchor block scores in
nats: ``mean_*`` decodes the latent mean, ``mc_*`` scores the Monte Carlo predictive
$-\log \tfrac1K \sum_k \exp(-\mathrm{NLL}(z_k))$, and ``pred_gap`` is base minus full (positive
means the source helps). Calibration lists the observed coverage and the relative tail error at
$1\sigma$, $2\sigma$ and $3\sigma$.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "runs": {
        "orig": common.RESULTS / "official_eval" / "orig",
        "v3": common.RESULTS / "official_eval" / "v3",
        "v4": common.RESULTS / "official_eval" / "v4",
        "v5": common.RESULTS / "official_eval" / "v5",
        "v6": common.RESULTS / "official_eval" / "v6",
    },
    "out": common.RESULTS / "official_eval" / "comparison.md",
}
READOUTS = ["mean_nll_base_block", "mean_nll_full_block", "mc_nll_base_block", "mc_nll_full_block",
            "mc_nll_shuffled_block", "mc_pred_gap", "mc_pred_gap_se", "mean_pred_gap", "pred_gap",
            "source_conditioned_kl_raw", "kld_source_null", "source_conditioned_kl_shuffled_raw",
            "nll_persistence_block"]


def load(run):
    """Load one run's summary.

    Args:
        run: The run's output directory.

    Returns:
        The parsed ``summary.json``.
    """
    return json.load(open(Path(run) / "eval_results" / "summary.json"))


def calib_row(label, values):
    """Format one calibration row.

    Args:
        label: Row label.
        values: One formatted cell per run.

    Returns:
        A Markdown table row.
    """
    return f"| {label} | " + " | ".join(values) + " |"


def main(a):
    """Print and write a Markdown table of verdicts, readouts and calibration.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    runs = {t: Path(p) for t, p in a["runs"].items() if (Path(p) / "eval_results" / "summary.json").exists()}
    S = {t: load(p)["results"] for t, p in runs.items()}
    tags = list(S)
    lines = ["| verdict | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    names = []
    for t in tags:
        names += [v["name"] for v in S[t]["verdicts"] if v["name"] not in names]
    for n in names:
        row = [next((v["status"] for v in S[t]["verdicts"] if v["name"] == n), "-") for t in tags]
        lines.append(f"| {n} | " + " | ".join(row) + " |")

    lines += ["", "| readout | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    for k in READOUTS:
        vals = [S[t]["readouts"].get(k) for t in tags]
        lines.append(f"| {k} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")

    cal = {t: S[t]["calibration"] for t in tags}
    lines += ["", "| calibration | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    lines.append(calib_row("coverage 1/2/3 sigma",
                           ["/".join(f"{x['observed']:.3f}" for x in cal[t]["coverage"]) for t in tags]))
    lines.append(calib_row("relative tail error",
                           ["/".join(f"{x['relative_tail_error']:.2f}" for x in cal[t]["coverage"]) for t in tags]))
    lines.append(calib_row("mean z^2", [f"{cal[t]['mean_standardised_sq']:.3f}" for t in tags]))
    lines.append(calib_row("PIT max deviation", [f"{cal[t]['pit']['max_cdf_deviation']:.3f}" for t in tags]))
    lines.append(calib_row("normalised CRPS", [f"{cal[t]['crps_normalised']:.4f}" for t in tags]))
    text = "\n".join(lines) + "\n"
    print(text)
    Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(a["out"]).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(RUN_ARGS)
