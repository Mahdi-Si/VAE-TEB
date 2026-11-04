r"""Compare input and output weight norms of the scattering and the phase channels.

Run inline in session 2. The column norm $\lVert W_{:,c} \rVert_2$ of ``target_adapter.linear`` says
how strongly input channel $c$ enters the target encoder; the row norm $\lVert W_{c,:} \rVert_2$ of
``decoder.mean_head`` says how strongly output channel $c$ reads the decoder features. On the
epoch-524 checkpoint both are about three to six times smaller for the 44 ``fhr_ph`` channels than
for the 32 ``fhr_st`` channels, and several phase input columns are near $0.01$.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "checkpoint": common.CHECKPOINT,
    "out": common.RESULTS / "session2" / "weight_norms.json",
}


def main(args):
    """Read the checkpoint's state dict on the CPU and report the norms.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    ck = torch.load(args["checkpoint"], map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    n_st = common.N_ST
    cols = sd["target_adapter.linear.weight"].norm(dim=0).numpy()
    rows = sd["decoder.mean_head.weight"].norm(dim=1).numpy()
    src = sd["source_adapter.linear.weight"].norm(dim=0).numpy()
    out = {
        "target_adapter_col_norm_st": cols[:n_st].tolist(),
        "target_adapter_col_norm_ph": cols[n_st:].tolist(),
        "target_adapter_mean_st": float(cols[:n_st].mean()),
        "target_adapter_mean_ph": float(cols[n_st:].mean()),
        "mean_head_row_norm_mean_st": float(rows[:n_st].mean()),
        "mean_head_row_norm_mean_ph": float(rows[n_st:].mean()),
        "source_adapter_col_norm": src.tolist(),
    }
    print({k: (np.round(v, 3) if isinstance(v, float) else "...") for k, v in out.items()})
    Path(args["out"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args["out"], "w"), indent=1)


if __name__ == "__main__":
    main(RUN_ARGS)
