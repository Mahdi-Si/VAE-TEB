r"""Evaluate, verify and compare the two arms of the quality-mask experiment from one place.

The experiment trains the model twice on the same quality-version-4 shards: with the quality masks
(``configs/prod_q4_masked.yaml``) and without them (``configs/prod_q4_no_quality_mask.yaml``). This
module takes the exact checkpoint and the exact output directory of each arm, and then runs, in
order:

1. ``eval/run.py`` on each checkpoint under ``eval/configs/eval_overrides_prod_q4.yaml``, the two
   arms in parallel on two devices by default. Each arm's console output goes to
   ``<output_dir>/run_quality_arms_eval.log``.
2. ``eval/verify.py`` on each arm's ``eval_results/summary.json``. Its report goes to
   ``<output_dir>/run_quality_arms_verify.txt`` and to the console.
3. ``eval/compare_quality_arms.py`` on the two output directories.

No step relies on a directory name to tell the arms apart. Before anything runs, the module reads
the resolved config that each checkpoint's training run wrote, and refuses unless the masked
checkpoint trained with ``source_quality_mask``, ``event_weight_ua_valid`` and
``target_quality_mask`` all on, and the control with all three off. It also refuses one directory
for two roles, and one checkpoint for both arms.

Run it from the PyCharm Run button after filling in :data:`RUN_ARGS`, or:

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_cfs.eval.run_quality_arms \
        --masked-checkpoint <masked run>/model_checkpoints/<best>.ckpt \
        --control-checkpoint <control run>/model_checkpoints/<best>.ckpt \
        --masked-output-dir <dir> --control-output-dir <dir> --comparison-output-dir <dir>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/eval/run_quality_arms.py`` -> up four.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and the package imports below cannot be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_rws.eval.run import RESULTS_DIRNAME, resolved_config_for  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval import compare_quality_arms  # noqa: E402

EVAL_MODULE = "teb_vae.lag_attn_transformer_cfs.eval.run"
VERIFY_MODULE = "teb_vae.lag_attn_transformer_cfs.eval.verify"
DEFAULT_OVERRIDES = "teb_vae/lag_attn_transformer_cfs/eval/configs/eval_overrides_prod_q4.yaml"
EVAL_LOG = "run_quality_arms_eval.log"
VERIFY_LOG = "run_quality_arms_verify.txt"
STAGES = ("eval", "verify", "compare")
ARMS = ("masked", "control")
#: The switches that identify an arm, and the value each arm trained with.
ARM_SWITCHES = ("source_quality_mask", "event_weight_ua_valid", "target_quality_mask")
EXPECTED = {"masked": True, "control": False}


def _absolute(path: str) -> Path:
    """Resolve a path against the repository root, so the Run button's working directory is moot."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate


def check_arm(arm: str, checkpoint: Path) -> Dict[str, Any]:
    """Read the switches a checkpoint trained with, and refuse a checkpoint of the other arm.

    Args:
        arm: ``"masked"`` or ``"control"``.
        checkpoint: The checkpoint given for that arm.

    Returns:
        ``{switch: value}`` for every name in :data:`ARM_SWITCHES`.

    Raises:
        SystemExit: If the checkpoint is absent, or any switch differs from the arm's value.
    """
    if not checkpoint.is_file():
        raise SystemExit(f"{arm}_checkpoint {checkpoint} is not a file.")
    with open(resolved_config_for(checkpoint), encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    vae = (config.get("model_config") or {}).get("VAE_model") or {}
    switches = {name: vae.get(name) for name in ARM_SWITCHES}
    wrong = {name: value for name, value in switches.items() if bool(value) != EXPECTED[arm]}
    if wrong:
        raise SystemExit(
            f"{arm}_checkpoint {checkpoint} did not train as the {arm} arm: {wrong}, expected "
            f"{EXPECTED[arm]} for {list(ARM_SWITCHES)}. Were the two checkpoints swapped?"
        )
    return switches


def _check_distinct(paths: Dict[str, Path]) -> None:
    """Refuse one path in two roles, which would let one arm's results overwrite the other's."""
    seen: Dict[Path, str] = {}
    for role, path in paths.items():
        key = path.resolve()
        if key in seen:
            raise SystemExit(f"{role} and {seen[key]} are the same path, {path}.")
        seen[key] = role


def _eval_command(checkpoint: Path, output_dir: Path, overrides: Path, device: str) -> List[str]:
    """Return the ``eval/run.py`` command line for one arm."""
    return [sys.executable, "-m", EVAL_MODULE, "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir), "--overrides", str(overrides), "--device", device]


def run_evaluations(jobs: Dict[str, Dict[str, Any]], overrides: Path, sequential: bool) -> None:
    """Run ``eval/run.py`` for every arm, in parallel unless *sequential*.

    Args:
        jobs: ``{arm: {'checkpoint', 'output_dir', 'device'}}``.
        overrides: The evaluation override file both arms use.
        sequential: Run one arm after the other instead of both at once.

    Raises:
        SystemExit: If an evaluation exits non-zero, naming its log.
    """
    running = []
    for arm, job in jobs.items():
        job["output_dir"].mkdir(parents=True, exist_ok=True)
        log_path = job["output_dir"] / EVAL_LOG
        command = _eval_command(job["checkpoint"], job["output_dir"], overrides, job["device"])
        print(f"[{arm}] {' '.join(command)}\n[{arm}] log: {log_path}", flush=True)
        log = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        running.append((arm, process, log, log_path))
        if sequential:
            process.wait()
    failed = []
    for arm, process, log, log_path in running:
        code = process.wait()
        log.close()
        print(f"[{arm}] eval exited {code}", flush=True)
        if code != 0:
            failed.append(f"{arm} (see {log_path})")
    if failed:
        raise SystemExit(f"evaluation failed for {', '.join(failed)}.")


def run_verify(arm: str, output_dir: Path) -> int:
    """Run ``eval/verify.py`` on one arm's summary and keep its report beside the results.

    Args:
        arm: ``"masked"`` or ``"control"``.
        output_dir: That arm's evaluation output directory.

    Returns:
        The verifier's exit code: non-zero when a sanity check failed.
    """
    summary = output_dir / RESULTS_DIRNAME / "summary.json"
    result = subprocess.run([sys.executable, "-m", VERIFY_MODULE, str(summary)], cwd=_REPO_ROOT,
                            capture_output=True, text=True)
    report = result.stdout + result.stderr
    (output_dir / VERIFY_LOG).write_text(report, encoding="utf-8")
    print(f"===== [{arm}] verify exited {result.returncode} =====\n{report}", flush=True)
    return result.returncode


def main(masked_checkpoint: Optional[str] = None, control_checkpoint: Optional[str] = None,
         masked_output_dir: Optional[str] = None, control_output_dir: Optional[str] = None,
         comparison_output_dir: Optional[str] = None, overrides: Optional[str] = None,
         masked_device: Optional[str] = None, control_device: Optional[str] = None,
         sequential: Optional[bool] = None, stages: Optional[str] = None) -> int:
    """Check both arms, then run the selected stages.

    Args:
        masked_checkpoint: The masked arm's checkpoint. Needed by the ``eval`` stage.
        control_checkpoint: The control's checkpoint. Needed by the ``eval`` stage.
        masked_output_dir: The masked arm's evaluation output directory.
        control_output_dir: The control's evaluation output directory.
        comparison_output_dir: Where the comparison reports go.
        overrides: The evaluation override file. Default: :data:`DEFAULT_OVERRIDES`.
        masked_device: The masked arm's device. Default: ``cuda:0``.
        control_device: The control's device. Default: ``cuda:1``.
        sequential: Evaluate one arm after the other. Default: ``False``.
        stages: A comma-separated subset of :data:`STAGES`. Default: all three.

    Returns:
        ``0``, or ``1`` when the verifier reported a failed check for either arm.

    Raises:
        SystemExit: If a stage's path is missing, one path has two roles, a checkpoint trained as
            the other arm, or an evaluation failed.
    """
    selected = STAGES if stages is None else tuple(s.strip() for s in stages.split(",") if s.strip())
    unknown = sorted(set(selected) - set(STAGES))
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {list(STAGES)}.")
    given = {"masked_output_dir": masked_output_dir, "control_output_dir": control_output_dir,
             "comparison_output_dir": comparison_output_dir,
             "masked_checkpoint": masked_checkpoint, "control_checkpoint": control_checkpoint}
    needed = ["masked_output_dir", "control_output_dir", "comparison_output_dir"]
    if "eval" in selected:
        needed += ["masked_checkpoint", "control_checkpoint"]
    refusal = missing_required(given, needed)
    if refusal:
        raise SystemExit(refusal)
    roles = {name: _absolute(str(given[name])) for name in needed}
    _check_distinct(roles)
    outputs = {arm: roles[f"{arm}_output_dir"] for arm in ARMS}

    if "eval" in selected:
        devices = {"masked": masked_device or "cuda:0", "control": control_device or "cuda:1"}
        for arm in ARMS:
            print(f"[{arm}] trained with {check_arm(arm, roles[f'{arm}_checkpoint'])}", flush=True)
        jobs = {arm: {"checkpoint": roles[f"{arm}_checkpoint"], "output_dir": outputs[arm],
                      "device": devices[arm]} for arm in ARMS}
        run_evaluations(jobs, _absolute(overrides or DEFAULT_OVERRIDES), bool(sequential))

    code = 0
    if "verify" in selected:
        codes = [run_verify(arm, outputs[arm]) for arm in ARMS]
        code = 1 if any(codes) else 0

    if "compare" in selected:
        compare_quality_arms.main(masked_dir=str(outputs["masked"]),
                                  control_dir=str(outputs["control"]),
                                  output_dir=str(roles["comparison_output_dir"]))
    return code


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; every ``dest`` is also a :data:`RUN_ARGS` key."""
    p = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_cfs.eval.run_quality_arms",
        description="Evaluate, verify and compare the masked and control arms.")
    p.add_argument("--masked-checkpoint", dest="masked_checkpoint")
    p.add_argument("--control-checkpoint", dest="control_checkpoint")
    p.add_argument("--masked-output-dir", dest="masked_output_dir")
    p.add_argument("--control-output-dir", dest="control_output_dir")
    p.add_argument("--comparison-output-dir", dest="comparison_output_dir")
    p.add_argument("--overrides", dest="overrides")
    p.add_argument("--masked-device", dest="masked_device")
    p.add_argument("--control-device", dest="control_device")
    p.add_argument("--sequential", dest="sequential", action="store_const", const=True,
                   default=None)
    p.add_argument("--stages", dest="stages", help=f"Comma-separated subset of {list(STAGES)}.")
    return p


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run; :func:`main` refuses a missing path."""
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # ============================ FILL IN: the checkpoints ============================
    # str -- each arm's best val/total_loss checkpoint, the full path. Each run directory is
    #        <out_dir_base>/<stamp>-<tag>/; the file is model_checkpoints/lag-attn-trf-cfs-epoch=NN.ckpt.
    #        The arm each checkpoint trained as is read from its own resolved config and checked.
    "masked_checkpoint": None,
    "control_checkpoint": None,

    # ============================== The output folders ==============================
    # str -- one folder per arm and one for the comparison; three different paths.
    "masked_output_dir": r"/data/deid/isilon/MS_model/new_vae_teb_cross_attention/lag_attn_transformer_cfs/q4_eval_masked",
    "control_output_dir": r"/data/deid/isilon/MS_model/new_vae_teb_cross_attention/lag_attn_transformer_cfs/q4_eval_no_quality_mask",
    "comparison_output_dir": r"/data/deid/isilon/MS_model/new_vae_teb_cross_attention/lag_attn_transformer_cfs/q4_quality_arms_comparison",

    # ========================= Everything below has a default =========================
    # str | None -- the evaluation override file both arms use.  (None -> eval_overrides_prod_q4.yaml)
    "overrides": None,
    # str | None -- one device per arm.  (None -> "cuda:0" and "cuda:1")
    "masked_device": None,
    "control_device": None,
    # bool | None -- True evaluates one arm after the other on its own device.  (None -> parallel)
    "sequential": None,
    # str | None -- e.g. "compare" to re-run only the comparison on finished evaluations, or
    #        "verify,compare". The checkpoints are needed only by "eval".  (None -> all three)
    "stages": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
