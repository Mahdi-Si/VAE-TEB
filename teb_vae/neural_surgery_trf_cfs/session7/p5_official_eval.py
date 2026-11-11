r"""Phase 5: run the official evaluation on the test half for the control and the s7 stack.

The official verdicts come from ``teb_vae.lag_attn_transformer_cfs.eval.run``, which reads the
run configuration beside the checkpoint. This script therefore builds one small run folder per
arm under ``out/phase5/official/<arm>/``:

* the fine-tuned checkpoint of the test pass (seed 0), copied under a run-style name;
* ``resolved_config.yaml``: the epoch-424 configuration, with the rung-7 keys added for the
  ``s7stack`` arm so the configuration and the checkpoint agree;
* ``eval_overrides_test.yaml``: the local shards and stand-in statistics, restricted to the
  recordings with split code 1 or 3.

It then runs the evaluation for each arm in a subprocess, skipping the analyses that cost much
time and do not feed a verdict, and writes ``out/phase5/official/verdicts.json`` with both arms'
verdicts and headline read-outs side by side.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Rung-7 keys of ``model_config.VAE_model``; the same values as ``init_s7.py`` stamps.
S7_CONFIG_KEYS: Dict[str, Any] = {
    "source_running_gap_keys": "remove",
    "source_valid_tolerance_s": 16.0,
    "decoder_logvar_clamp": [-9.0, 3.0],
    "forecast_student_nu": 8.0,
}

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "arms": {"control": {}, "s7stack": S7_CONFIG_KEYS},
    "seed": 0,
    "codes": [1, 3],
    "skip": "sufficiency,attribution,samples,recording_traces,cross_subgroup,spectral_skill,lag_high_kl",
    "run": True,                 # False: only collect the verdicts of finished runs
    "device": "cuda:0",
}
ROOT = S.OUT / "phase5" / "official"


def prepare(arm: str, extra_keys: Dict[str, Any], seed: int, codes: List[int]) -> Dict[str, Path]:
    """Build the run folder of one arm.

    Args:
        arm: The arm name; its checkpoint is ``out/phase4/pilots_test/<arm>_s<seed>.ckpt``.
        extra_keys: Keys to add to ``model_config.VAE_model`` of the configuration.
        seed: The seed of the fine-tune.
        codes: Split codes of the recordings to evaluate.

    Returns:
        ``{'checkpoint', 'overrides', 'output'}``.

    Raises:
        FileNotFoundError: If the fine-tuned checkpoint does not exist.
    """
    source = S.OUT / "phase4" / "pilots_test" / f"{arm}_s{seed}.ckpt"
    if not source.exists():
        raise FileNotFoundError(f"{source}: run the test pass with --save-checkpoint first")
    folder = ROOT / arm
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / f"lag-attn-trf-cfs-{arm}-ft.ckpt"
    shutil.copyfile(source, checkpoint)
    config = yaml.safe_load(S.CONFIG.read_text())
    config["model_config"]["VAE_model"].update(extra_keys)
    (folder / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    overrides = yaml.safe_load(S.OVERRIDES.read_text())
    overrides["dataset_config"]["dataloader_config"]["dataset_kwargs"]["allowed_guids"] = P.guids_with_code(codes)
    overrides_path = folder / "eval_overrides_test.yaml"
    overrides_path.write_text(yaml.safe_dump(overrides, sort_keys=False))
    return {"checkpoint": checkpoint, "overrides": overrides_path, "output": folder / "eval"}


def read_verdicts(output: Path) -> Dict[str, Any]:
    """Read the verdicts and the headline read-outs of a finished evaluation.

    Args:
        output: The evaluation output directory.

    Returns:
        ``{'verdicts': {name: {'status', 'values'}}, 'exit_code', 'n_recordings'}``, or an
        ``error`` entry when the run left no summary.
    """
    summary_path = output / "eval_results" / "summary.json"
    if not summary_path.exists():
        return {"error": f"no summary at {summary_path}"}
    summary = json.loads(summary_path.read_text())
    results = summary.get("results", {})
    return {
        "verdicts": {item["name"]: {"status": item["status"], "values": item.get("values")}
                     for item in results.get("verdicts", [])},
        "exit_code": summary.get("exit_code"), "n_recordings": results.get("n_recordings"),
        "failed_steps": summary.get("failed"),
    }


def main(args: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare, run and compare.

    Args:
        args: :data:`RUN_ARGS` or the command-line equivalent.

    Returns:
        ``{arm: verdict record}``.
    """
    report: Dict[str, Any] = {}
    for arm, extra_keys in args["arms"].items():
        paths = prepare(arm, extra_keys, int(args["seed"]), list(args["codes"]))
        if args["run"]:
            command = [sys.executable, "-m", "teb_vae.lag_attn_transformer_cfs.eval.run",
                       "--checkpoint", str(paths["checkpoint"]), "--overrides", str(paths["overrides"]),
                       "--output-dir", str(paths["output"]), "--device", args["device"], "--skip", args["skip"]]
            with open(paths["output"].parent / "eval.log", "w") as log:
                code = subprocess.call(command, cwd=str(S.REPO), stdout=log, stderr=subprocess.STDOUT)
            print(f"{arm}: evaluation exit code {code}")
        report[arm] = read_verdicts(paths["output"])
    (ROOT / "verdicts.json").write_text(json.dumps(report, indent=1))
    names = sorted({name for record in report.values() for name in record.get("verdicts", {})})
    arms = list(report)
    print(f"{'verdict':42s} " + " ".join(f"{arm:>14s}" for arm in arms))
    for name in names:
        print(f"{name:42s} " + " ".join(f"{report[arm].get('verdicts', {}).get(name, {}).get('status', '-'):>14s}"
                                        for arm in arms))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true", default=not RUN_ARGS["run"])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parsed = parser.parse_args()
    main(dict(RUN_ARGS, run=not parsed.no_run, device=parsed.device))
