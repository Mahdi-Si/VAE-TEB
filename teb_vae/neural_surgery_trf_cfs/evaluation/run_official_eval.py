r"""Run the official evaluation pipeline on the test half for each checkpoint.

Each run is a subprocess of ``python -m teb_vae.lag_attn_transformer_cfs.eval.run`` with the
test-half override file. Analyses that do not bear on the surgery are skipped to save time; the
verdicts, readouts and calibration still run. One run takes about 20 to 40 minutes on one GPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.evaluation import make_test_half_overrides  # noqa: E402

RUN_ARGS = {
    "checkpoints": {
        "orig": common.CHECKPOINT,
        "v3": common.checkpoint_variant("v3"),
        "v4": common.checkpoint_variant("v4"),
        "v5": common.checkpoint_variant("v5"),
        "v6": common.checkpoint_variant("v6"),
    },
    "out_root": common.RESULTS / "official_eval",
    "skip": "sufficiency,attribution,samples,recording_traces,cross_subgroup,time_to_delivery,"
            "second_stage,trajectory",
    "python": sys.executable,
}


def main(a):
    """Write the overrides once, then evaluate each checkpoint into ``out_root/<tag>``.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    overrides = make_test_half_overrides.main(make_test_half_overrides.RUN_ARGS)
    root = Path(a["out_root"])
    root.mkdir(parents=True, exist_ok=True)
    for tag, ckpt in a["checkpoints"].items():
        out = root / tag
        cmd = [a["python"], "-m", "teb_vae.lag_attn_transformer_cfs.eval.run",
               "--checkpoint", str(ckpt), "--overrides", str(overrides),
               "--output-dir", str(out), "--skip", a["skip"]]
        with open(root / f"{tag}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=str(common.REPO), stdout=log, stderr=subprocess.STDOUT)
        print(tag, "exit", code, flush=True)


if __name__ == "__main__":
    main(RUN_ARGS)
