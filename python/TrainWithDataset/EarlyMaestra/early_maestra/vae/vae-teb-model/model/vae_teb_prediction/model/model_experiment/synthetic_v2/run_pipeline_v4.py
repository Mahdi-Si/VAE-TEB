r"""S0-T05: staged orchestrator skeleton for ``synthetic_v4`` (raw-model validation).

A *dedicated* driver (not a plugin on ``run_pipeline_v2``) because the v2/v3 orchestrator reads
the v2/v3-schema ``config["model"]["horizon"]`` / ``resolved_model_class_name(config["model"])``,
which a ``model_raw``-schema v4 config does not carry. The **schema-agnostic** pieces
(:func:`load_config`, :func:`resolve_arm`, ``_run_dir``, ``_results_dir``) are imported and reused;
only the stage registry is forked so v4 stages register into their own table rather than the v2
one.

Each later sprint's stage module (``realizability_v4``, ``data_previews_v4``, ``build_dataset_v4``,
``eval_v4`` ...) calls :func:`register_stage_v4` at import; :func:`_load_stage_plugins` imports
them warn-don't-gate, so a not-yet-landed stage is simply absent from ``--stage`` rather than an
import error. This Sprint-0 skeleton dispatches a single stage (no arm sweep, no DDP subprocess
yet) and prints help when given no stage.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
    resolve_arm_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (  # noqa: E402
    load_config,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E402
    _results_dir,
    _run_dir,
)

#: The v4 default ``--config``.
_DEFAULT_CONFIG = _MODULE_DIR / "config_synth_v4.yaml"


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
        r"""``results/<tag>/<arm>/`` (or ``results/<tag>/`` when arm-less)."""
        return _run_dir(self.config, self.benchmark, self.arm)

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
    "eval_v4",
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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    r"""Dispatch a single stage (or print help when no ``--stage`` is given)."""
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

    arm = args.arm if spec.model_dependent else None
    config = resolve_arm_v4(config, arm)
    ctx = StageContextV4(config=config, benchmark=benchmark, arm=arm,
                         split=args.split, pilot=args.pilot, args=args)
    return _dispatch_stage_v4(spec, ctx)


if __name__ == "__main__":
    raise SystemExit(main())
