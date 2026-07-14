r"""S0-T05 / S7-T01..T03: staged orchestrator for ``synthetic_v4`` (raw-model validation).

A *dedicated* driver (not a plugin on ``run_pipeline_v2``) because the v2/v3 orchestrator reads
the v2/v3-schema ``config["model"]["horizon"]`` / ``resolved_model_class_name(config["model"])``,
which a ``model_raw``-schema v4 config does not carry. The **schema-agnostic** pieces
(:func:`load_config`, :func:`resolve_arm`, ``_run_dir``, ``_results_dir``, ``resolve_cache_dir``)
are imported and reused; only the stage registry is forked so v4 stages register into their own
table rather than the v2 one.

Each later sprint's stage module (``realizability_v4``, ``data_previews_v4``, ``build_dataset_v4``,
``eval_v4`` ...) calls :func:`register_stage_v4` at import; :func:`_load_stage_plugins` imports
them warn-don't-gate, so a not-yet-landed stage is simply absent from ``--stage`` rather than an
import error.

Sprint 7 adds the sweep driver on top of the skeleton (S7-T01/T02/T03):

* **Model-free stages** (``build`` / ``realizability`` / ``data_previews`` / ``arms_report``) run
  **once** at the arm-less ``results/<tag>/`` root.
* **Per-arm stages** (``train`` / ``test_plots`` / ``eval`` / ``report``) loop over
  ``config.arms`` when no explicit ``--arm`` is given, else run the single named arm (this second
  path is also the DDP-safe subprocess re-entry point).
* **Split-scoped stages** (``eval`` / ``test_plots`` / ``report``) additionally fan out over every
  cached split, writing per-split artifacts under ``results/<tag>/<arm>/<split>/``.
* ``--dry-run`` prints the resolved plan (arms + per-arm run dirs) and dispatches nothing.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_MODULE_FILE = Path(__file__).resolve()
_MODULE_DIR = _MODULE_FILE.parent
_PKG = "model.vae_teb_prediction.model.model_experiment.synthetic_v2"
_QUALNAME = f"{_PKG}.run_pipeline_v4"

# Repo root importable whether run as a script or as ``-m`` (six levels up), then alias ``__main__``
# under the dotted name so a stage plugin's ``register_stage_v4`` reaches ONE registry object.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if __name__ == "__main__" and _QUALNAME not in sys.modules:
    sys.modules[_QUALNAME] = sys.modules[__name__]

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.arms_v4 import (  # noqa: E402
    list_arms,
    resolve_arm_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (  # noqa: E402
    load_config,
    resolve_cache_dir,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E402
    _results_dir,
    _run_dir,
)

#: The v4 default ``--config``.
_DEFAULT_CONFIG = _MODULE_DIR / "config_synth_v4.yaml"

#: Dataset splits, in the canonical fan-out order.
_ALL_SPLITS = ("train", "val", "test")

#: Model-dependent stages that additionally fan out over every cached split, writing per-split
#: artifacts under ``results/<tag>/<arm>/<split>/``. (``train`` is per-arm but split-independent;
#: ``arms_report`` resolves splits internally, so neither belongs here.)
_SPLIT_SCOPED_STAGES = frozenset({"eval", "test_plots", "report"})


# =============================================================================
# Stage registry (forked; fresh table so v4 stages never collide with v2's).
# =============================================================================
@dataclass(frozen=True)
class StageContextV4:
    r"""Everything a v4 stage runner needs, from the CLI dispatcher.

    Attributes:
        config: The parsed config tree, **already arm-resolved** by the dispatcher.
        benchmark: Active benchmark key under ``benchmarks``.
        arm: Resolved arm name, or ``None`` (arm-less / model-free stages).
        split: A single split name, or ``None`` for "every cached split".
        pilot: Short/pilot run (small grids, few samples).
        args: The raw argparse namespace when dispatched from the CLI, else ``None``.
    """

    config: Dict[str, Any]
    benchmark: str
    arm: Optional[str] = None
    split: Optional[str] = None
    pilot: bool = False
    args: Optional[argparse.Namespace] = None

    def run_dir(self) -> Path:
        r"""``results/<tag>/<arm>/`` (or ``results/<tag>/`` when arm-less).

        This is the **arm root** -- checkpoints (``final.ckpt`` / ``best.ckpt``) live here and are
        split-independent, so training writes and every stage discovers the checkpoint from it.
        """
        return _run_dir(self.config, self.benchmark, self.arm)

    def output_dir(self) -> Path:
        r"""The per-artifact output root, split-scoped when a split is set (S7-T03).

        A split-scoped stage (``eval`` / ``test_plots`` / ``report``) writes its per-split
        artifacts under ``results/<tag>/<arm>/<split>/`` so ``--split all`` grades every split into
        a distinct directory without collision; with ``split is None`` this collapses to
        :meth:`run_dir` (the single-arm / pre-fan-out layout the Sprint 4-6 stages assumed).
        """
        base = self.run_dir()
        return base / self.split if self.split else base

    def results_dir(self) -> Path:
        r"""``results/<tag>/`` -- the arm-independent root."""
        return _results_dir(self.config, self.benchmark)


@dataclass(frozen=True)
class StageSpecV4:
    r"""One registered v4 pipeline stage.

    Attributes:
        name: Stage key used by ``--stage``.
        run: ``(StageContextV4) -> int`` runner.
        order: Sort order in ``--help`` / execution (lower first).
        model_dependent: Whether output depends on the model (arm-scoped). Model-free stages run
            once at the arm-less ``results/<tag>/`` root.
        fatal: When ``False``, an exception is caught, logged, and the run continues (0).
        help: One-line ``--help`` description.
    """

    name: str
    run: Callable[[StageContextV4], int]
    order: int = 100
    model_dependent: bool = False
    fatal: bool = True
    help: str = ""


_STAGE_REGISTRY_V4: "OrderedDict[str, StageSpecV4]" = OrderedDict()


def register_stage_v4(spec: StageSpecV4) -> None:
    r"""Register a v4 stage, keeping the registry sorted by ``spec.order``.

    Raises:
        ValueError: When ``spec.name`` is already registered.
    """
    if spec.name in _STAGE_REGISTRY_V4:
        raise ValueError(f"stage {spec.name!r} is already registered")
    _STAGE_REGISTRY_V4[spec.name] = spec
    for key in sorted(_STAGE_REGISTRY_V4, key=lambda k: _STAGE_REGISTRY_V4[k].order):
        _STAGE_REGISTRY_V4.move_to_end(key)


def stage_names_v4() -> List[str]:
    r"""Registered stage keys, in execution order."""
    return list(_STAGE_REGISTRY_V4)


def _dispatch_stage_v4(spec: StageSpecV4, ctx: StageContextV4) -> int:
    r"""Run one stage, honouring its ``fatal`` policy."""
    if spec.fatal:
        return spec.run(ctx)
    try:
        return spec.run(ctx)
    except Exception as exc:  # noqa: BLE001 -- opt-in stages never gate the run
        print(f"[{spec.name}] failed (non-fatal): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


#: Stage modules that register on import. Missing modules degrade to "stage unavailable" (a
#: not-yet-landed sprint), never an import error. Sprints 1+ append here.
_STAGE_PLUGIN_MODULES_V4 = (
    "realizability_v4",
    "data_previews_v4",
    "build_dataset_v4",
    "trainer_v4",
    "eval_runner_v4",
    "eval_v4",
    "final_report_v4",
    "arms_report_v4",
)


def _load_stage_plugins() -> None:
    r"""Import each v4 stage module so its module-level ``register_stage_v4`` fires (idempotent)."""
    for name in _STAGE_PLUGIN_MODULES_V4:
        try:
            importlib.import_module(f"{_PKG}.{name}")
        except ModuleNotFoundError:
            continue  # stage not landed yet
        except Exception as exc:  # noqa: BLE001 -- a broken optional stage must not gate startup
            print(f"[plugin] {name} failed to import: {type(exc).__name__}: {exc}", file=sys.stderr)


# =============================================================================
# Split fan-out + DDP-safe subprocess dispatch (S7-T02 / S7-T03)
# =============================================================================
def _resolve_splits(config: Dict[str, Any], benchmark: str,
                    split: Optional[str]) -> List[str]:
    r"""Resolve which cached splits a split-scoped stage should process (S7-T03).

    ``split`` of ``None`` / ``"all"`` selects **every** split whose cache ``.npz`` exists (in
    ``train, val, test`` order); an explicit name restricts to that one. Falls back to ``["val"]``
    when nothing is discoverable yet (e.g. before a build), matching the shared eval runner's
    default split.

    Args:
        config: The (arm-resolved) config tree.
        benchmark: Active benchmark key under ``benchmarks``.
        split: ``None`` / ``"all"`` for every present split, or a single split name.

    Returns:
        The ordered list of split names to process.
    """
    present: List[str] = []
    try:
        cache_dir = Path(resolve_cache_dir(config, benchmark=benchmark))
        present = [s for s in _ALL_SPLITS if (cache_dir / f"{s}.npz").is_file()]
    except Exception:  # noqa: BLE001 -- cache path not resolvable yet (pre-build)
        present = []
    if split in (None, "", "all", "ALL"):
        return present or ["val"]
    return [str(split)]


def _stage_subprocess_cmd_v4(config_path: Any, stage: str, *,
                             arm: Optional[str] = None, pilot: bool = False,
                             split: Optional[str] = None) -> List[str]:
    r"""Build the argv vector that re-enters this driver for one arm's ``stage`` (S7-T02).

    The train stage is dispatched as a fresh subprocess so Lightning owns a clean DDP process
    group per arm (re-entering the *same* module file via ``-m`` keeps one stage registry). Only
    flags this driver's own parser accepts are appended -- ``--arm`` / ``--pilot`` / ``--split`` --
    so the child is a single-arm, in-process dispatch (never a nested sweep).

    Args:
        config_path: Path to ``config_synth_v4.yaml`` (threaded to the child verbatim).
        stage: The stage the child runs (typically ``"train"``).
        arm: The arm to resolve in the child (its explicit ``--arm`` disables the child's sweep).
        pilot: Forward ``--pilot`` when set.
        split: Forward ``--split`` when set.

    Returns:
        The subprocess argv list, ``[python, -m, <qualname>, --config, ..., --stage, ...]``.
    """
    cmd: List[str] = [sys.executable, "-m", _QUALNAME,
                      "--config", str(config_path), "--stage", stage]
    if arm:
        cmd += ["--arm", str(arm)]
    if pilot:
        cmd += ["--pilot"]
    if split:
        cmd += ["--split", str(split)]
    return cmd


def _run_subprocess(cmd: Sequence[str], *, dry_run: bool = False) -> int:
    r"""Run a child driver process from the repo root; ``dry_run`` prints and returns 0."""
    if dry_run:
        print("[dry-run] " + " ".join(str(c) for c in cmd))
        return 0
    proc = subprocess.run(list(cmd), cwd=str(_REPO_ROOT))
    return int(proc.returncode)


# =============================================================================
# Sweep driver (S7-T01)
# =============================================================================
def _dispatch_for_arm(config: Dict[str, Any], benchmark: str, spec: StageSpecV4,
                      arm: Optional[str], args: argparse.Namespace) -> int:
    r"""Resolve ``arm`` and dispatch ``spec`` once (or once per cached split when split-scoped).

    Args:
        config: The base (un-arm-resolved) config tree.
        benchmark: Active benchmark key.
        spec: The stage to dispatch.
        arm: The arm to resolve, or ``None`` (model-free / arm-less).
        args: The parsed CLI namespace (``pilot`` / ``split`` / ``dry_run``).

    Returns:
        ``0`` on success, else the first non-zero stage return code (worst-case OR).
    """
    arm_cfg = resolve_arm_v4(config, arm)
    if spec.name in _SPLIT_SCOPED_STAGES:
        rc = 0
        for s in _resolve_splits(arm_cfg, benchmark, args.split):
            ctx = StageContextV4(config=arm_cfg, benchmark=benchmark, arm=arm,
                                 split=s, pilot=args.pilot, args=args)
            rc = rc or _dispatch_stage_v4(spec, ctx)
        return rc
    ctx = StageContextV4(config=arm_cfg, benchmark=benchmark, arm=arm,
                         split=args.split, pilot=args.pilot, args=args)
    return _dispatch_stage_v4(spec, ctx)


def _sweep_arms(config: Dict[str, Any], args: argparse.Namespace) -> List[Optional[str]]:
    r"""The arm list a model-dependent sweep iterates.

    An explicit ``--arm`` selects that single arm; otherwise the sweep iterates every configured arm
    **in the config's declared order** (not alphabetically), so the ``config_synth_v4.yaml`` author's
    intent holds: the direct headline arm ``prod`` runs first and the ``am_carrier_prod`` probe --
    which reads a *separate* cache that a direct-only run has not built -- runs last. ``[None]`` when
    the config declares no ``arms`` block.
    """
    if args.arm:
        return [args.arm]
    return list((config.get("arms") or {}).keys()) or [None]


def run_sweep(args: argparse.Namespace, config: Dict[str, Any], benchmark: str,
              spec: StageSpecV4) -> int:
    r"""Drive one stage across the arm ladder (S7-T01).

    Model-free stages run **once** at the arm-less root; model-dependent stages loop the arms,
    dispatching ``train`` as a DDP-safe subprocess per arm and every other stage in-process (with
    split fan-out for the split-scoped stages). ``--dry-run`` prints the resolved plan and
    dispatches nothing.

    Args:
        args: The parsed CLI namespace.
        config: The base config tree (arms resolved per-arm inside).
        benchmark: Active benchmark key.
        spec: The stage to sweep.

    Returns:
        ``0`` on success, else the first non-zero return code.
    """
    if not spec.model_dependent:
        if args.dry_run:
            print(f"[plan] stage={spec.name!r} (model-free) -> {_run_dir(config, benchmark)}")
            return 0
        return _dispatch_for_arm(config, benchmark, spec, None, args)

    arms = _sweep_arms(config, args)
    if args.dry_run:
        print(f"[plan] stage={spec.name!r} over {len(arms)} arm(s):")
        for arm in arms:
            splits = (_resolve_splits(resolve_arm_v4(config, arm), benchmark, args.split)
                      if spec.name in _SPLIT_SCOPED_STAGES else [None])
            for s in splits:
                run_dir = _run_dir(config, benchmark, arm)
                tail = f"/{s}" if s else ""
                print(f"  - arm={arm!r:>20} -> {run_dir}{tail}")
        return 0

    rc = 0
    for arm in arms:
        if spec.name == "train":
            cmd = _stage_subprocess_cmd_v4(args.config, "train", arm=arm,
                                           pilot=args.pilot, split=args.split)
            rc = rc or _run_subprocess(cmd)
        else:
            rc = rc or _dispatch_for_arm(config, benchmark, spec, arm, args)
    return rc


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    r"""Build the argparse parser (stage plugins are loaded first so ``--stage`` lists them)."""
    _load_stage_plugins()
    parser = argparse.ArgumentParser(
        prog="run_pipeline_v4",
        description="synthetic_v4 raw-model validation pipeline (staged).",
    )
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG),
                        help="path to config_synth_v4.yaml")
    parser.add_argument("--stage", choices=stage_names_v4(), default=None,
                        help="stage to run: " + (", ".join(stage_names_v4()) or "(none registered)"))
    parser.add_argument("--arm", default=None,
                        help="arm to resolve for model-dependent stages")
    parser.add_argument("--pilot", action="store_true",
                        help="short/pilot run (small grids, few samples)")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"],
                        help="a single split, or omit for all cached splits")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved arm/split plan and dispatch nothing")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    r"""Drive one stage: sweep the arm ladder, or dispatch a single arm (S7-T01).

    A model-dependent stage with no explicit ``--arm`` (or any ``--dry-run``) enters
    :func:`run_sweep`; an explicit ``--arm`` (the subprocess re-entry / single-arm path) and every
    model-free stage dispatch once. Prints help when given no ``--stage``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.stage:
        parser.print_help()
        return 0

    config = load_config(args.config)
    benchmark = str(config.get("experiment", {}).get("benchmark", "G1_raw_v4"))

    spec = _STAGE_REGISTRY_V4.get(args.stage)
    if spec is None:  # argparse ``choices`` normally prevents this
        print(f"[stage] unknown stage {args.stage!r}; registered: {stage_names_v4()}",
              file=sys.stderr)
        return 2

    # Sweep when: a dry-run plan is requested, OR a model-dependent stage was invoked without an
    # explicit arm and the config declares an arm ladder. An explicit ``--arm`` (the subprocess
    # re-entry) and every model-free stage take the single-dispatch path below.
    if args.dry_run or (spec.model_dependent and args.arm is None and bool(list_arms(config))):
        return run_sweep(args, config, benchmark, spec)

    arm = args.arm if spec.model_dependent else None
    return _dispatch_for_arm(config, benchmark, spec, arm, args)


# =============================================================================
# Edit-and-run driver (no CLI args -> run the toggled stages; mirrors run_pipeline_v2.py)
# =============================================================================
#: Edit these, then just hit ▶ Run (no CLI args needed). The *config* controls WHAT each stage does
#: (grid, arms' deltas, model, beta schedule, thresholds); this dict controls WHICH stages + arms run,
#: on HOW MANY GPUs. Nothing runs unless toggled on. Passing ANY CLI argument bypasses this and uses
#: the argparse CLI in :func:`main` instead, so both entry styles coexist.
#:
#: PROD / FINAL (default below): ``pilot: False`` runs the FULL config grid, full training with
#: **multi-GPU DDP** over ``cuda_devices``, into the config's real tag/paths -- ``train`` is dispatched
#: through :func:`run_sweep` as a DDP-safe subprocess per arm (so Lightning owns a clean 8-GPU process
#: group; the subprocess re-enters ``--stage train --arm X``, never this driver).
#: DEV smoke: set ``pilot: True`` -> tiny grid + a single GPU + a few epochs, isolated under
#: ``pilot_v4_out/`` so it never clobbers a headline run.
_PIPELINE: Dict[str, Any] = {
    "pilot": False,
    # GPUs for a PROD run (Lightning ``devices``; >1 -> DDP). Ignored in pilot mode (forced to [0]).
    # This is the prod 8x A6000 box; trim the list to match the machine you launch on.
    "cuda_devices": [0, 1, 2, 3, 4, 5, 6, 7],
    # Per-epoch raw FHR/UP forecast plots DURING training. Off by default: it imports ``utils.style``
    # (a repo module that must be present/importable on the box -- note ``utils/style.py`` is untracked,
    # so a git checkout or a partial deploy will not have it), and per-epoch PDF plotting across a
    # multi-arm DDP sweep is expensive. The report / test_plots stages produce the figures instead.
    # Set True only if ``utils/style.py`` is importable on this machine.
    "raw_plotting": False,
    # Arms to sweep, in this order. [] -> every configured arm. This is the full DIRECT-cache headline
    # ladder; ``am_carrier_prod`` reads a SEPARATE am cache (build config_synth_v4_am.yaml first), so
    # it is run in its own pass, not here.
    "arms": ["prod", "frontend_noncausal", "single_stride", "no_antialias", "no_gated",
             "disable_source"],
    "split": "val",                 # split to grade (eval / report / arms_report); None -> all splits
    "stages": {                     # flip any to False to skip it (stages are resumable)
        "build": True,              #   generate the raw .npz cache (needed before train/eval)
        "realizability": True,      #   model-free te_raw gate -> realizability.json
        "data_previews": False,     #   raw FHR/UP overlay figures per cell
        "train": True,              #   train each arm (DDP subprocess per arm when >1 GPU)
        "eval": True,               #   grade each arm -> <arm>/<split>/metrics.json
        "report": True,             #   per-arm markdown report + figures
        "arms_report": True,        #   one cross-arm gate table at the tag root
    },
    # Pilot-only grid (ignored when pilot is False -> the config's full grid is built instead).
    "te_grid": [0.0, 2.0], "lag_grid": [8], "n_per_split": {"train": 8, "val": 4, "test": 4},
}

#: Split-scoped stages that take a ``--split`` (train is split-independent; model-free stages ignore it).
_SPLIT_FLAG_STAGES = frozenset({"eval", "report", "arms_report"})
#: Execution order for the edit-and-run driver (``build`` is handled specially, before the rest).
_DRIVER_STAGE_ORDER = ("realizability", "data_previews", "train", "eval", "report", "arms_report")


def _edit_and_run() -> int:
    r"""Run the stages toggled in :data:`_PIPELINE` with no CLI args (hit ▶ Run).

    Every model-dependent stage is dispatched through :func:`main` **without** an explicit ``--arm``,
    so the tested :func:`run_sweep` machinery drives the arm loop: ``train`` becomes a DDP-safe
    subprocess per arm (each subprocess lets Lightning own a clean multi-GPU DDP process group over
    ``cuda_devices``), while ``eval`` / ``report`` run in-process per arm with split fan-out.
    Crucially, ``train`` is **never** called in-process here -- an in-process multi-GPU fit would make
    Lightning re-launch this arg-less script and re-enter :func:`_edit_and_run` on every rank.

    In ``pilot`` mode the run is forced to a single GPU and retargeted to an isolated ``pilot_v4`` tag
    under ``pilot_v4_out/``. Returns ``0``; raises on the first non-zero stage return code.
    """
    import copy

    import yaml

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v4 import (
        build_all_v4,
    )

    plan = _PIPELINE
    config = copy.deepcopy(load_config(str(_DEFAULT_CONFIG)))
    benchmark = str(config.get("experiment", {}).get("benchmark", "G1_raw_v4"))
    arms = list(plan["arms"]) if plan["arms"] else list((config.get("arms") or {}).keys())

    # GPU selection: pilot forces a single device; a prod run uses the configured list (>1 -> DDP).
    devices = [0] if plan["pilot"] else list(plan["cuda_devices"])
    config.setdefault("general_config", {})["cuda_devices"] = devices

    if plan["pilot"]:
        out_dir = _MODULE_DIR / "pilot_v4_out"
        config["experiment"] = {**config.get("experiment", {}),
                                "tag": "pilot_v4", "data_tag": "pilot_v4_direct"}
        config["paths"] = {"data_dir": str(out_dir / "data"),
                           "results_dir": str(out_dir / "results")}
        config.setdefault("general_config", {}).setdefault("folders_config", {})[
            "out_dir_base"] = str(out_dir / "train_out")
        config.setdefault("advanced_config", {}).setdefault("tracking", {}).setdefault(
            "mlflow", {})["enabled"] = False
        cfg_path = out_dir / "config_run_v4.yaml"
    else:
        cfg_path = _MODULE_DIR / "config_run_v4.yaml"

    # Narrow the arm block to the selection so a no-arm sweep (train / eval / arms_report) iterates
    # exactly these arms in this order.
    orig_arms = config.get("arms", {}) or {}
    config["arms"] = {arm: orig_arms.get(arm, {}) for arm in arms}

    # Keep training headless: no per-epoch raw plotting unless explicitly enabled (avoids the
    # ``utils.style`` import; the figures come from the report / test_plots stages).
    config.setdefault("advanced_config", {}).setdefault("callbacks", {}).setdefault(
        "raw_plotting", {})["enabled"] = bool(plan.get("raw_plotting", False))

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    base = ["--config", str(cfg_path)]
    pilot_flag = ["--pilot"] if plan["pilot"] else []
    split_flag = ["--split", plan["split"]] if plan["split"] else []
    stages = plan["stages"]

    def _banner(msg: str) -> None:
        print(f"\n{'=' * 78}\n>>> {msg}\n{'=' * 78}", flush=True)

    def _run(extra: List[str], label: str) -> None:
        _banner(label)
        rc = main(base + extra)
        if rc != 0:
            raise RuntimeError(f"stage failed ({label}): return code {rc}")

    mode = "tiny pilot grid, 1 GPU" if plan["pilot"] else f"full grid, {len(devices)} GPU(s) (DDP)"
    print(f"[run] {'PILOT' if plan['pilot'] else 'PROD'} sweep: arms={arms} ({mode})")

    # build is a direct call (so the pilot grid override applies; prod builds the config's full grid).
    if stages.get("build"):
        _banner(f"build cache  arms={arms}")
        cache_dir = Path(resolve_cache_dir(config, benchmark=benchmark))
        overrides = (dict(grid_override={"target_te_grid": plan["te_grid"],
                                         "lag_grid": plan["lag_grid"]},
                          n_override=dict(plan["n_per_split"])) if plan["pilot"] else {})
        build_all_v4(config, benchmark=benchmark, out_dir=cache_dir, **overrides)
        print(f"[build] cache -> {cache_dir}")

    # Every other toggled stage -> main() WITHOUT --arm, so run_sweep drives the arm loop and, for
    # train, the DDP-safe subprocess-per-arm dispatch.
    for name in _DRIVER_STAGE_ORDER:
        if not stages.get(name):
            continue
        extra = ["--stage", name, *pilot_flag]
        if name in _SPLIT_FLAG_STAGES:
            extra += split_flag
        _run(extra, name)

    root = _results_dir(config, benchmark)
    _banner("DONE -- artifacts")
    print(f"  results root : {root}")
    print(f"  arms report  : {root / 'arms_report_v4.md'}")
    for arm in arms:
        run_dir = _run_dir(config, benchmark, arm)
        print(f"  [{arm}] ckpt   : {run_dir / 'final.ckpt'}")
        print(f"  [{arm}] metrics: {run_dir / (plan['split'] or 'val') / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    # No CLI args -> run the _PIPELINE toggles (hit ▶ Run). Any argument -> the argparse CLI in
    # main() (also the DDP-safe `--stage train` subprocess re-entry). Mirrors run_pipeline_v2.py.
    raise SystemExit(main() if len(sys.argv) > 1 else _edit_and_run())
