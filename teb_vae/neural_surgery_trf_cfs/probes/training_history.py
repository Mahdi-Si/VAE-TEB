r"""Summarise the run's ``metrics_history.csv`` at selected epochs.

Run inline in session 2. Prints the validation loss, the block NLLs, the source gains, the KL and
its source-null part, the decoder log-variance floor fraction, the gradient norm and the learning
rate at a fixed list of epochs, and the best epochs of the monitored metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "history": common.HISTORY,
    "epochs": [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 430, 450, 486, 500, 524],
    "out_csv": common.RESULTS / "session2" / "training_history_selected.csv",
}
COLUMNS = ["val/total_loss", "val/nll_full_block", "val/nll_base_block", "val/pred_gap", "val/pred_gap_mean",
           "val/source_conditioned_kl_raw", "val/kld_source_null", "val/shuffle_penalty",
           "val/mean_logvar_full", "val/logvar_full_floor_frac", "val/mean_logvar_prior", "val/event_gate_mean",
           "val/pred_gap_st", "val/pred_gap_ph", "val/pred_gap_tau_first", "val/pred_gap_tau_last",
           "val/kld_active_frac", "train/nll_full_block", "train/grad_norm", "train/spike_skipped", "lr"]


def main(args):
    """Print the selected epochs and the best epochs, and save the selected rows as CSV.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    d = pd.read_csv(args["history"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    sel = d.loc[d.epoch.isin(args["epochs"]), ["epoch"] + COLUMNS]
    for chunk in (COLUMNS[:8], COLUMNS[8:16], COLUMNS[16:]):
        print(sel[["epoch"] + chunk].round(4).to_string(index=False))
    for metric, mode in (("val/total_loss", "min"), ("val/nll_full_block", "min"), ("val/pred_gap", "max")):
        idx = d[metric].idxmin() if mode == "min" else d[metric].idxmax()
        print(f"best {metric}: epoch {int(d.loc[idx, 'epoch'])} value {d.loc[idx, metric]:.3f}")
    print("val - train NLL over the last 50 epochs:",
          float((d["val/nll_full_block"] - d["train/nll_full_block"]).tail(50).mean()))
    print("spike-skipped batches:", float(d["train/spike_skipped"].sum()))
    Path(args["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(args["out_csv"], index=False)


if __name__ == "__main__":
    main(RUN_ARGS)
