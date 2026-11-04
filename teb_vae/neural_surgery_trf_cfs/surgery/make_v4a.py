r"""Write v4a: the v4 checkpoint without the round-4 posterior variance match.

Run inline in session 2. Round 4 step 2 added $c_d$ to the bias of each posterior log-variance
head. This script subtracts the same $c_d$, read from the round-4 log, so v4a keeps only the latent
affine of step 1. The posterior has 4 heads of 16 latent dimensions each; both the ``model.`` and
the ``_orig_model.`` copies of the state dict are edited.

v4a is the better checkpoint for single-sample scores; v4 is the better one for Monte Carlo
predictive scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "checkpoint": common.checkpoint_variant("v4"),
    "out_ckpt": common.checkpoint_variant("v4a"),
    "round4_json": common.RESULTS / "surgery" / "round4" / "v4.json",
}


def main(a):
    """Subtract the posterior shift from the v4 state dict and save v4a.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    assert Path(a["out_ckpt"]).resolve() != Path(a["checkpoint"]).resolve()
    c = torch.tensor(json.load(open(a["round4_json"]))["post_shift"])
    ck = torch.load(a["checkpoint"], map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    n_heads = sum(1 for k in sd if k.startswith("model.posterior_head.logvar_post_head.") and k.endswith(".bias"))
    k = c.numel() // n_heads
    edited = 0
    for pre in ("model.", "_orig_model."):
        for i in range(n_heads):
            key = f"{pre}posterior_head.logvar_post_head.{i}.bias"
            if key in sd:
                sd[key] = sd[key] - c[i * k:(i + 1) * k].to(sd[key].dtype)
                edited += 1
    ck["surgery_v4"]["post_shift_kept"] = False
    ck["surgery_v4"]["variant"] = "v4a: latent decoupling only, no posterior variance match"
    torch.save(ck, a["out_ckpt"])
    print("edited", edited, "bias tensors; saved", a["out_ckpt"])


if __name__ == "__main__":
    main(RUN_ARGS)
