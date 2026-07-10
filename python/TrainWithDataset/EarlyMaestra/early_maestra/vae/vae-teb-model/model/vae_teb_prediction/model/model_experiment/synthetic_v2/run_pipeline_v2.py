r"""End-to-end driver for the ``synthetic_v2`` pipeline.

Two equivalent interfaces (the second mirrors ``synthetic/run_mixed_pipeline``):

1. **Edit-and-run dict (no argparse).** Run the file with **no arguments** and the
   ``PIPELINE`` dict in ``__main__`` drives every enabled stage in
   :data:`_STAGE_ORDER` -- ``r0_realizability`` -> ``build`` -> ``data_previews``
   -> ``train`` (+``beta_select``) -> ``eval`` -> ``test_plots`` -> ``report``, plus
   the ``solve_te`` / ``am_check`` / ``recover`` / ``scatter_preview`` diagnostics. Edit
   the dict's ``stages`` toggles and knobs, then::

       .venv/Scripts/python.exe .../synthetic_v2/run_pipeline_v2.py

   Programmatically, build a config dict and call :func:`run_pipeline` directly.

2. **Argparse CLI (per-stage / one-off hooks).** Passing any argument dispatches to
   :func:`main`: ``--solve-te 2.0 8`` (prints $B$, achieved block TE, per-step SNR
   via the ported inverter :func:`analytic_te.B_y_for_mean_te_block_state_space`),
   ``--am-check``, ``--scatter-preview``, ``--recover``, and ``--stage <name>``. This
   CLI is also what the DDP-safe ``train`` / ``beta_select`` subprocesses re-enter, so
   it is retained alongside the dict driver. See ``README.md`` for per-stage commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Make the repo root importable whether this file is run as a script
# (``python .../run_pipeline_v2.py``) or as a module
# (``python -m ...run_pipeline_v2`` / ``importlib.import_module``). The repo root
# is six levels up: synthetic_v2 -> model_experiment -> model ->
# vae_teb_prediction -> model -> <repo root>.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Canonical dotted name of this module.
_QUALNAME = "model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2"

# Under ``python run_pipeline_v2.py`` (or ``-m``) this file is bound as ``__main__``. The stage
# plugins loaded by :func:`_load_stage_plugins` reach ``register_stage`` through the *dotted*
# name, which would import a SECOND copy of this module with its own, empty ``_STAGE_REGISTRY``
# -- so every plugin stage would register into a registry nobody reads, and ``--stage
# calibration`` would be rejected as an invalid choice. Aliasing ``__main__`` under the dotted
# name first makes both spellings resolve to one module object.
if __name__ == "__main__" and _QUALNAME not in sys.modules:
    sys.modules[_QUALNAME] = sys.modules[__name__]

import yaml  # noqa: E402  (import after the sys.path bootstrap)

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    resolve_cache_dir,
    solve_cell_coupling,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (  # noqa: E402
    am_separation_from_config,
    generate_cell_raw,
)

_MODULE_FILE = Path(__file__).resolve()
_MODULE_DIR = _MODULE_FILE.parent
#: The argparse ``--config`` default. Stays v1/v2 so the CLI and its regression tests are
#: unchanged; the v3 ablation ladder is entered by pointing ``--config`` (or the ``PIPELINE``
#: dict's ``config_path``) at :data:`_DEFAULT_CONFIG_V3`.
_DEFAULT_CONFIG = _MODULE_DIR / "config_synth_v2.yaml"
_DEFAULT_CONFIG_V3 = _MODULE_DIR / "config_synth_v3.yaml"

# =============================================================================
# Stage registry (S4-T05)
# =============================================================================
# Every stage table -- the ``--stage`` choices, the dict driver's execution order, its
# on-by-default set, and the arm-scoped set -- is DERIVED from this one registry, so adding a
# stage means calling :func:`register_stage` from the module that implements it and editing
# nothing here or in ``main()``. That is what lets the Sprint 5/6/7 analyses
# (``calibration`` / ``lag_intervention`` / ``cmi``) land as three file-disjoint modules
# instead of three edits to the same dispatch block.


@dataclass(frozen=True)
class StageContext:
    r"""Everything a stage function needs, from either driver (CLI or the ``PIPELINE`` dict).

    Attributes:
        config: The parsed config tree, **already arm-resolved** by the dispatcher.
        benchmark: Active benchmark key under ``benchmarks``.
        arm: Resolved arm name; ``None`` for the arm-less v1/v2 layout and for every
            model-free stage.
        ckpt: Explicit checkpoint path, else the stage resolves best/final under the run dir.
        split: A single split name, or ``None`` for "every cached split".
        analysis_samples: Per-sample diagnostic PDFs to emit (``test_plots``).
        max_samples: Optional per-split sample cap for the expensive analysis stages.
        pilot: Short-run smoke (``train``) / pilot grid (``build``, ``r0_realizability``).
        full: Opt into the full locked grid where ``pilot`` is the default.
        args: The raw argparse namespace when dispatched from the CLI, else ``None``. Only
            the built-in ``train`` / ``beta_select`` stages read it (for ``_train_overrides``);
            registered analysis stages must use the explicit fields above so they work under
            both drivers.
    """

    config: Dict[str, Any]
    benchmark: str
    arm: Optional[str] = None
    ckpt: Optional[str] = None
    split: Optional[str] = None
    analysis_samples: int = 4
    max_samples: Optional[int] = None
    pilot: bool = False
    full: bool = False
    args: Optional[argparse.Namespace] = None

    def run_dir(self) -> Path:
        r"""``results/<tag>/<arm>/`` (or ``results/<tag>/`` when arm-less)."""
        return _run_dir(self.config, self.benchmark, self.arm)

    def tag_root(self) -> Path:
        r"""``results/<tag>/`` -- the arm-independent root (data-story figures, arms_report)."""
        return _results_dir(self.config, self.benchmark)

    def splits(self) -> List[str]:
        r"""The splits this stage should process."""
        return list(_resolve_splits(self.config, self.benchmark, self.split))


@dataclass(frozen=True)
class StageSpec:
    r"""One registered pipeline stage.

    Attributes:
        name: Stage key, used by ``--stage`` and by ``PIPELINE['stages']``.
        order: Execution order in the dict driver (lower runs first).
        default_on: Whether the dict driver runs it when ``PIPELINE['stages']`` omits the key.
        model_dependent: Whether its output depends on the model, and is therefore scoped to
            ``results/<tag>/<arm>/`` and repeated per arm. Model-free stages run once at the
            arm-less root and never require ``--arm``.
        run: ``(StageContext) -> int``. ``None`` for the four diagnostic stages that the dict
            driver handles inline and that ``--stage`` never dispatches.
        cli: Whether ``--stage`` exposes it.
        fatal: When ``False``, an exception is caught, logged as ``failed (non-fatal)``, and
            the run continues. The append-only analysis stages register with ``fatal=False``
            so a diverging CMI fit can never abort a headline run.
        help: One-line description for ``--help``.
    """

    name: str
    order: int
    default_on: bool
    model_dependent: bool
    run: Optional[Callable[[StageContext], int]] = None
    cli: bool = True
    fatal: bool = True
    help: str = ""


_STAGE_REGISTRY: "OrderedDict[str, StageSpec]" = OrderedDict()


def register_stage(spec: StageSpec) -> None:
    r"""Register a pipeline stage, keeping the registry sorted by ``spec.order``.

    Args:
        spec: The stage to register.

    Raises:
        ValueError: When ``spec.name`` is already registered.
    """
    if spec.name in _STAGE_REGISTRY:
        raise ValueError(f"stage {spec.name!r} is already registered")
    _STAGE_REGISTRY[spec.name] = spec
    for key in sorted(_STAGE_REGISTRY, key=lambda k: _STAGE_REGISTRY[k].order):
        _STAGE_REGISTRY.move_to_end(key)


def stage_names() -> List[str]:
    r"""Stage keys exposed by ``--stage``, in execution order."""
    return [n for n, s in _STAGE_REGISTRY.items() if s.cli]


def stage_order() -> Tuple[str, ...]:
    r"""Every registered stage key, in dict-driver execution order."""
    return tuple(_STAGE_REGISTRY)


def stage_defaults() -> Dict[str, bool]:
    r"""``{stage: default_on}`` for the dict driver."""
    return {n: s.default_on for n, s in _STAGE_REGISTRY.items()}


def model_dependent_stages() -> frozenset:
    r"""Stages whose output depends on the model, and which are therefore arm-scoped."""
    return frozenset(n for n, s in _STAGE_REGISTRY.items() if s.model_dependent)


#: Modules that register extra stages on import. Each is optional: a missing or broken
#: analysis module degrades to "that stage is unavailable", never to an import error at
#: startup. Sprints 5/6/7 append to this tuple and touch nothing else in this file.
_STAGE_PLUGIN_MODULES: Tuple[str, ...] = (
    "calibration_v3",
    "lag_intervention_v3",
    "cmi_v3",
    "arms_report_v3",
)
_PLUGINS_LOADED = False


def _load_stage_plugins() -> None:
    r"""Import the optional analysis modules so their :func:`register_stage` calls fire.

    Idempotent, and warn-don't-gate: a plugin that fails to import (missing optional
    dependency, syntax error under development) prints a warning and leaves its stage
    unregistered, rather than taking the whole driver down with it.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    import importlib
    for mod in _STAGE_PLUGIN_MODULES:
        try:
            importlib.import_module(
                f"model.vae_teb_prediction.model.model_experiment.synthetic_v2.{mod}"
            )
        except ModuleNotFoundError:
            continue  # not shipped yet (Sprints 5-7)
        except Exception as exc:  # noqa: BLE001 -- a broken plugin must not gate the driver
            print(f"[stages][warn] plugin {mod} failed to import: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)


def load_config(path: Any) -> Dict[str, Any]:
    r"""Load and parse ``config_synth_v2.yaml``.

    Args:
        path: Path to the YAML config.

    Returns:
        The parsed config as a nested ``dict``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _results_dir(config: Dict[str, Any], benchmark: str) -> Path:
    r"""Resolve the ``results/<tag>/`` output directory for this run.

    Uses ``experiment.tag`` (falling back to the benchmark name) under
    ``paths.results_dir`` (relative paths resolve against this module's directory).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key (fallback tag).

    Returns:
        The ``results/<tag>`` directory as an absolute :class:`Path` (not created).
    """
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    if not results_dir.is_absolute():
        results_dir = _MODULE_DIR / results_dir
    return results_dir / tag


def _run_dir(config: Dict[str, Any], benchmark: str,
             arm: Optional[str] = None) -> Path:
    r"""Resolve the per-run output root ``results/<tag>/<arm>/`` (or ``results/<tag>/``).

    Model-dependent artifacts (checkpoints, per-split ``metrics.json`` /
    ``per_sample_eval.npz`` / reports) are arm-scoped so the three ``synthetic_v3`` arms
    coexist under one ``experiment.tag``. The split-independent, model-free artifacts
    (``realizability.json``, the data-story gallery, ``recovery.json``) stay at the
    arm-less ``results/<tag>/`` root and are written once. ``arm=None`` reproduces
    :func:`_results_dir` exactly, preserving the v1 / v2 single-arm layout.

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key (fallback tag).
        arm: The arm name, or ``None`` for the arm-less run root.

    Returns:
        The run root :class:`Path` (not created).
    """
    base = _results_dir(config, benchmark)
    return base / str(arm) if arm else base


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    r"""Recursively merge ``over`` into a deep copy of ``base`` (``over`` wins).

    Nested dicts are merged key-by-key rather than replaced wholesale, so an arm delta
    that sets a single ``model`` kwarg does not drop the rest of the base ``model`` block.
    """
    out = deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def resolve_arm(config: Dict[str, Any], arm: Optional[str]) -> Dict[str, Any]:
    r"""Deep-merge ``arms.<arm>`` over the base ``model`` / ``loss`` blocks.

    The three ``synthetic_v3`` arms (``parity`` / ``v3_noncausal`` / ``v3_prod``) differ
    only in a handful of ``model`` kwargs while sharing one cache, one seed set, and one
    objective. ``arms.<name>`` carries just those per-arm deltas; this pure resolver merges
    them (arm wins; nested dicts merged, not replaced) over the base config so the rest of
    the pipeline sees a single flat config. ``arm=None`` (or a config with no ``arms``
    block for that name) returns the input unchanged, preserving the v1 / v2 single-arm
    path. The input is never mutated; the ``arms`` block is left intact for provenance.

    Args:
        config: The parsed config tree (with an optional ``arms`` block).
        arm: The arm name to resolve, or ``None`` for the base config.

    Returns:
        A new config dict with the arm delta merged in (or the input when ``arm`` is
        ``None``).

    Raises:
        ValueError: If ``arm`` is given but not present in the ``arms`` block.
    """
    if arm is None:
        return config
    arms = config.get("arms") or {}
    if arm not in arms:
        raise ValueError(
            f"unknown arm {arm!r}; configured arms: {sorted(arms)}"
        )
    return _deep_merge(config, dict(arms[arm] or {}))


# Stages whose output depends on the model (and are therefore arm-scoped) are declared by
# ``StageSpec.model_dependent`` and read back through :func:`model_dependent_stages`. The
# remaining stages (``build`` / ``r0_realizability`` / ``data_previews`` / ...) are model-free
# and run once at the arm-less ``results/<tag>/`` root.


def _select_arm(config: Dict[str, Any], arm_flag: Optional[str]) -> Optional[str]:
    r"""Resolve the effective arm for a model-dependent stage from ``--arm`` + the config.

    Enforces the arm contract at the CLI boundary: an explicit ``--arm`` must name a
    configured arm; with no flag, a single-arm config defaults to its sole arm, a config
    with no ``arms`` block runs arm-less (the v1 / v2 path), and a multi-arm config demands
    an explicit choice rather than silently grading only one arm.

    Args:
        config: The parsed config tree.
        arm_flag: The ``--arm`` value (``None`` when the flag is absent).

    Returns:
        The resolved arm name, or ``None`` for the arm-less path.

    Raises:
        SystemExit: On an unknown ``--arm``, or a multi-arm config with no ``--arm`` (a
            non-zero exit with an actionable message).
    """
    arms = config.get("arms") or {}
    if arm_flag:
        if arm_flag not in arms:
            raise SystemExit(
                f"[arm] unknown --arm {arm_flag!r}; configured arms: "
                f"{sorted(arms) or '(none)'}"
            )
        return arm_flag
    if not arms:
        return None
    if len(arms) == 1:
        return next(iter(arms))
    raise SystemExit(
        f"[arm] this config defines {len(arms)} arms {sorted(arms)}; pass --arm NAME "
        "to select one."
    )


def _flatten_leaves(tree: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    r"""Flatten a nested dict to ``{'a.b.c': leaf}`` so a nested arm delta reads at a glance."""
    out: Dict[str, Any] = {}
    for key, val in tree.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(_flatten_leaves(val, f"{path}."))
        else:
            out[path] = val
    return out


def _print_arm_plan(config: Dict[str, Any], arm: Optional[str]) -> None:
    r"""Print an arm's resolved model class and only the ``model`` kwargs it overrides (dry-run).

    The per-arm deltas live inside ``arms.<name>.model.v3`` (they must override the base ``v3``
    overlay, which ``build_model`` applies last), so the delta is flattened to leaf paths rather
    than printing the whole merged overlay.
    """
    resolved = resolve_arm(config, arm)
    cls = str((resolved.get("model") or {}).get("class", "SeqVaeLagAttn (alias -> v1)"))
    label = arm if arm is not None else "(arm-less)"
    arm_model = ((config.get("arms") or {}).get(arm or "", {}) or {}).get("model") or {}
    delta = _flatten_leaves(dict(arm_model)) if arm else {}
    suffix = f"  delta={delta}" if delta else ("  (no model delta vs base)" if arm else "")
    print(f"[pipeline]   arm {label}: class={cls}{suffix}")


def solve_te(
    config: Dict[str, Any],
    target_te: float,
    delay: int,
    *,
    benchmark: str = "G1_raw",
) -> Dict[str, Any]:
    r"""Solve the coupling $B$ for a cell authored by ``(target_te, D)``.

    Thin CLI wrapper that delegates to
    :func:`build_dataset_v2.solve_cell_coupling` (the single owner of the inverter
    call), so the ``--solve-te`` demo and the dataset build solve identically.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        target_te: Target injected block TE in nats ($\ge 0$; ``0`` is a null cell).
        delay: Fixed source->target lag $D$ in decimated steps.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        The inverter's result dict augmented with ``snr_per_step``: keys
        ``B_y``, ``B_y_scalar``, ``te_block``, ``te_per_step``, ``n_iter``,
        ``snr_per_step``.
    """
    return solve_cell_coupling(config, target_te, delay, benchmark=benchmark)


def _print_solution(target_te: float, delay: int, solution: Dict[str, Any]) -> None:
    r"""Pretty-print an inverter solution to stdout.

    Args:
        target_te: The requested target block TE (nats).
        delay: The requested fixed lag $D$.
        solution: The dict returned by :func:`solve_te`.
    """
    snr = solution["snr_per_step"]
    print(f"[solve-te] cell  target_te={target_te:g} nats   D={delay} steps")
    print(f"  B_y_scalar   = {solution['B_y_scalar']:.6f}")
    print(f"  te_block     = {solution['te_block']:.4f} nats   (achieved)")
    print(f"  te_per_step  = {solution['te_per_step']:.4f} nats")
    print(f"  SNR/step     = {snr:.4f}   ({100.0 * snr:.2f}%)")
    print(f"  n_iter       = {solution['n_iter']}")


def _print_am_check(result: Dict[str, Any]) -> None:
    r"""Pretty-print an AM-separation pre-check result to stdout.

    Args:
        result: The dict returned by
            :func:`raw_generators.am_separation_from_config`.
    """
    verdict = "ADEQUATE" if result["adequate"] else "MARGINAL"
    print(f"[am-check] AM-separation pre-check (S1-T04)  ->  {verdict}")
    print(f"  margin_peak   = {result['margin_peak']:.3f}   (want >= 1)")
    print(f"  margin_edge   = {result['margin_edge']:.3f}")
    print(f"  preservation  = {result['preservation']:.3f}   (frac_Phi pre-estimate)")
    print(f"  sigma_wav     = {result['sigma_wav_hz']:.5f} Hz")
    print(f"  f_env_peak    = {result['f_env_peak']:.5f} Hz   (edge {result['f_env_edge']:.5f} Hz)")
    print(f"  mod_depth_rms = {result['mod_depth_rms']:.3f}")
    print(f"  {result['recommendation']}")


def scatter_preview(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    target_te: float = 2.0,
    delay: int = 8,
    n: int = 16,
) -> Dict[str, Any]:
    r"""Transform one strong ``am_carrier`` cell and write a scattering heatmap (S2-T04).

    Solves the coupling for a strong cell, generates ``n`` raw pairs, runs the real
    scattering transform + normalisation via
    :class:`scattering_adapter.ScatteringAdapter`, prints the four field shapes and the
    fs-correct coupled channel, and writes the :func:`visualize_v2.plot_scattering_heatmap`
    figure under ``results_dir/<tag>/figures/``.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        target_te: Target injected block TE (nats) for the preview cell.
        delay: Fixed lag $D$ (decimated steps) for the preview cell.
        n: Number of raw pairs to transform (for a stable per-channel z-score).

    Returns:
        A dict with the written figure paths (``figures``), the ``coupled`` channel info,
        the field ``shapes``, and the mean $|{\mathrm{corr}}|$ of the coupled ``up_st``
        channel with the decimated latent ``c[15:315]`` (``coupled_corr``).
    """
    # Local imports: these pull torch / matplotlib, kept out of the fast --solve-te path.
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.scattering_adapter import (
        ScatteringAdapter,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2 import (
        plot_scattering_heatmap,
    )

    solution = solve_te(config, target_te, delay, benchmark=benchmark)
    b_scalar = float(solution["B_y_scalar"])
    raw = generate_cell_raw(
        n, B=b_scalar, D=delay, config=config, benchmark=benchmark,
        seed=int(config.get("seeds", {}).get("dgp", 0)), te_inj=float(solution["te_block"]),
    )

    adapter = ScatteringAdapter(config, benchmark=benchmark)
    fields, _ = adapter.transform_and_normalise(raw["fhr_raw"], raw["up_raw"])
    coupled = adapter.coupled_channel_indices()

    # Coupled up_st channel vs the decimated source latent on the trimmed grid [15:315].
    idx = int(coupled["up_st"])
    c_slice = raw["latents"]["c"][:, 15:315]
    chan = fields["up_st"][:, :, idx]
    c_c = c_slice - c_slice.mean(axis=1, keepdims=True)
    ch_c = chan - chan.mean(axis=1, keepdims=True)
    denom = (
        (c_c ** 2).sum(axis=1) ** 0.5 * (ch_c ** 2).sum(axis=1) ** 0.5
    )
    corr = ((c_c * ch_c).sum(axis=1) / (denom + 1e-12))
    coupled_corr = float(abs(corr).mean())

    out_stem = _results_dir(config, benchmark) / "figures" / "scattering_heatmap_preview"
    figures = plot_scattering_heatmap(
        fields["fhr_st"], fields["up_st"], out_stem,
        fhr_ph=fields["fhr_ph"], up_ph=fields["up_ph"],
        coupled_idx=idx, center_freqs=adapter.center_freqs_np, fs=adapter.fs,
    )

    shapes = {name: tuple(arr.shape) for name, arr in fields.items()}
    print("[scatter-preview] strong am_carrier cell "
          f"(target_te={target_te:g}, D={delay}, B={b_scalar:.4f}, te_block={solution['te_block']:.4f})")
    for name in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
        print(f"  {name:7s} shape = {shapes[name]}")
    print(f"  coupled st channel = {idx}  ({coupled['hz']:.5f} Hz, xi={coupled['xi']:.5f})")
    print(f"  |corr(up_st ch{idx}, c[15:315])| mean = {coupled_corr:.3f}")
    for path in figures:
        print(f"  wrote {path}")
    return {"figures": figures, "coupled": coupled, "shapes": shapes, "coupled_corr": coupled_corr}


def data_previews(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    target_te: float = 2.0,
    delay: int = 8,
    n: int = 16,
    include_null: bool = True,
) -> Dict[str, Any]:
    r"""Render the data-domain figure gallery for one strong cell (S7 previews).

    Generates one strong cell (solved for ``target_te`` at lag ``delay``), runs the real
    scattering transform + normalisation, and writes the four data-domain figures into
    ``results/<tag>/figures/`` so a plain ``run_pipeline_v2.py`` emits them alongside the
    model-grading gallery: the annotated raw preview
    (:func:`visualize_v2.plot_raw_preview`), the scattering heatmap
    (:func:`visualize_v2.plot_scattering_heatmap`), the latent / AM envelope-carrier
    decomposition (:func:`visualize_v2.plot_latent_am_decomposition`), and the headline
    raw$+$scattering paired preview (:func:`visualize_v2.plot_raw_scatter_paired`). When
    ``include_null`` a $B=0$ null cell's raw preview is also written for contrast. Supersedes
    the single-heatmap ``scatter_preview`` (which stays available as a lightweight diagnostic).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        target_te: Target injected block TE (nats) for the strong preview cell.
        delay: Fixed lag $D$ (decimated steps) for the preview cell.
        n: Number of raw pairs to transform (for a stable per-channel z-score).
        include_null: Also render a $B=0$ null-cell raw preview for contrast.

    Returns:
        A dict with the written figure paths (``figures``) and the ``coupled`` channel info.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.scattering_adapter import (
        ScatteringAdapter,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2 import (
        plot_am_separation,
        plot_band_spectra,
        plot_latent_am_decomposition,
        plot_latent_coupling,
        plot_raw_preview,
        plot_raw_scatter_paired,
        plot_scattering_heatmap,
        plot_te_authoring,
    )

    bench = config["benchmarks"][benchmark]
    render_mode = str(bench.get("raw", {}).get("render_mode", "am_carrier"))
    seed = int(config.get("seeds", {}).get("dgp", 0))

    solution = solve_te(config, target_te, delay, benchmark=benchmark)
    b_scalar = float(solution["B_y_scalar"])
    te_block = float(solution["te_block"])
    raw = generate_cell_raw(
        n, B=b_scalar, D=delay, config=config, benchmark=benchmark, seed=seed,
        te_inj=te_block, render_mode=render_mode,
    )
    adapter = ScatteringAdapter(config, benchmark=benchmark)
    fields, _ = adapter.transform_and_normalise(raw["fhr_raw"], raw["up_raw"])
    coupled = adapter.coupled_channel_indices()
    idx = int(coupled["up_st"])

    figs_dir = _results_dir(config, benchmark) / "figures"
    meta = raw["meta"]
    f_pulse = float(meta.get("f_pulse", bench.get("raw", {}).get("f_pulse", 0.06)))
    prev_meta = {"te_inj": te_block, "D": delay, "B": b_scalar, "f_pulse": f_pulse}

    written: List[Any] = []
    written += plot_raw_preview(
        raw["fhr_raw"], raw["up_raw"], figs_dir / "raw_preview", meta=prev_meta, fs=adapter.fs,
        fhr_ph=fields["fhr_ph"], up_ph=fields["up_ph"],
    )
    written += plot_scattering_heatmap(
        fields["fhr_st"], fields["up_st"], figs_dir / "scattering_heatmap",
        fhr_ph=fields["fhr_ph"], up_ph=fields["up_ph"],
        coupled_idx=idx, center_freqs=adapter.center_freqs_np, fs=adapter.fs,
    )
    written += plot_latent_am_decomposition(
        raw["latents"], figs_dir / "latent_am_decomposition", fs=adapter.fs,
        f_pulse=f_pulse, meta=meta,
    )
    written += plot_raw_scatter_paired(
        raw["fhr_raw"], raw["up_raw"], fields["fhr_st"], fields["up_st"],
        figs_dir / "raw_scatter_paired", coupled_idx=idx,
        latent_c=raw["latents"]["c"], latent_d=raw["latents"]["d"],
        fhr_ph=fields["fhr_ph"], up_ph=fields["up_ph"],
        center_freqs=adapter.center_freqs_np, fs=adapter.fs, meta=meta,
    )

    # Data-generation *story* figures (the controls behind the previews above): the
    # frequency recipe (§4-§5), the coupling pathway / lag (§6), the carrier de-risk
    # (§7), and the TE control law (§9). All four are self-contained from this single
    # strong cell + ``config`` -- no realizability preflight required; ``te_authoring``
    # is the only one that runs its own (modest) Monte-Carlo.
    horizon = int(config["model"]["horizon"])
    written += plot_band_spectra(
        raw["fhr_raw"], raw["up_raw"], figs_dir / "band_spectra", fs=adapter.fs, meta=meta,
    )
    written += plot_latent_coupling(
        raw["latents"], figs_dir / "latent_coupling", D=delay, horizon=horizon, fs=adapter.fs,
    )
    written += plot_am_separation(
        config, figs_dir / "am_separation", benchmark=benchmark,
    )
    written += plot_te_authoring(
        config, figs_dir / "te_authoring", benchmark=benchmark, delay=delay,
    )

    if include_null:
        raw0 = generate_cell_raw(
            max(2, n // 2), B=0.0, D=delay, config=config, benchmark=benchmark,
            seed=seed + 1, te_inj=0.0, render_mode=render_mode,
        )
        # Transform the null cell too so its raw preview also carries the phase-harmonic
        # panels -- the contrast (coupling present vs absent) is the point of the null.
        fields0, _ = adapter.transform_and_normalise(raw0["fhr_raw"], raw0["up_raw"])
        written += plot_raw_preview(
            raw0["fhr_raw"], raw0["up_raw"], figs_dir / "raw_preview_null",
            meta={"te_inj": 0.0, "D": delay, "B": 0.0, "f_pulse": f_pulse}, fs=adapter.fs,
            fhr_ph=fields0["fhr_ph"], up_ph=fields0["up_ph"],
        )

    print(f"[data-previews] strong cell target_te={target_te:g} D={delay} "
          f"B={b_scalar:.4f} render={render_mode}  coupled ch {idx} ({coupled['hz']:.5f} Hz)")
    for path in written:
        print(f"  wrote {path}")
    return {"figures": written, "coupled": coupled}


def _train_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    r"""Build the :func:`pl_module_v2.train_v2` overrides from the CLI + ``train`` config.

    ``--pilot`` overlays the short-run knobs (``train.pilot_*``) so the training path
    can be smoke-tested on the real cache; explicit ``--epochs`` / ``--devices`` win
    over both the pilot and the config defaults.

    Args:
        config: The parsed config tree.
        args: The parsed CLI namespace (uses ``pilot``, ``epochs``, ``devices``).

    Returns:
        The overrides dict for :func:`pl_module_v2.train_v2`.
    """
    train_cfg = config.get("train", {}) or {}
    overrides: Dict[str, Any] = {"pilot": bool(args.pilot)}
    if args.pilot:
        overrides["epochs"] = int(train_cfg.get("pilot_epochs", 3))
        overrides["limit_train_batches"] = int(train_cfg.get("pilot_limit_train_batches", 4))
        overrides["limit_val_batches"] = int(train_cfg.get("pilot_limit_val_batches", 2))
        overrides["batch_size"] = int(train_cfg.get("pilot_batch_size", 16))
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    overrides["devices"] = _resolve_train_devices(
        config, devices=args.devices, pilot=bool(args.pilot)
    )
    if getattr(args, "max_samples", None) is not None:
        overrides["max_samples"] = int(args.max_samples)
    return overrides


def _resolve_train_devices(
    config: Dict[str, Any], *, devices: Any = None, pilot: bool = False,
) -> Any:
    r"""Resolve the Lightning ``devices`` spec for a training stage.

    Precedence: an explicit ``devices`` (``--devices`` / ``PIPELINE["devices"]``) beats
    everything; a ``pilot`` run takes the pilot-oriented ``train.devices``; a headline run
    takes ``ddp.devices`` -- the documented multi-GPU knob -- falling back to
    ``train.devices``.

    Before this, ``ddp.devices`` was never read anywhere and ``train.devices`` (``1``) was
    the only fallback, so a headline train without an explicit ``--devices`` silently
    trained on a single GPU on an 8-GPU box.

    Both drivers share this, so the printed plan matches what the subprocess receives.

    Args:
        config: The parsed config tree.
        devices: An explicit devices spec, or ``None`` to resolve from the config.
        pilot: Whether this is a short ``--pilot`` training smoke.

    Returns:
        The Lightning ``devices`` spec (an int count or a comma-separated GPU list).
    """
    if devices is not None:
        return devices
    train_cfg = config.get("train") or {}
    if pilot:
        return train_cfg.get("devices", 1)
    ddp_devices = (config.get("ddp") or {}).get("devices")
    if ddp_devices is not None:
        return ddp_devices
    return train_cfg.get("devices", 1)


def _print_train_result(result: Dict[str, Any]) -> None:
    r"""Pretty-print a :func:`pl_module_v2.train_v2` result to stdout.

    Args:
        result: The dict returned by :func:`pl_module_v2.train_v2`.
    """
    metrics = result.get("metrics", {})
    print(f"[train] finished {result.get('epochs')} epoch(s)")
    for key in ("train/total_loss", "val/total_loss", "train/kld_nats"):
        if key in metrics:
            print(f"  {key:18s} = {metrics[key]:.4f}")
    print(f"  latent_stats_n = {result.get('n_stats')}")
    if result.get("checkpoint"):
        print(f"  checkpoint     -> {result['checkpoint']}")
    if result.get("best"):
        print(f"  best           -> {result['best']}")
    if result.get("metrics_csv"):
        print(f"  metrics.csv    -> {result['metrics_csv']}")
    for path in result.get("figures", []):
        print(f"  wrote {path}")


def _print_eval_metrics(metrics: Dict[str, Any]) -> None:
    r"""Pretty-print the Sprint 6 evaluation gates to stdout.

    Args:
        metrics: The dict returned by :func:`eval_v2.run_eval`.
    """
    cal = metrics.get("calibration", {})
    lag = metrics.get("lag_recovery", {})
    nul = metrics.get("null_controls", {})
    frac = metrics.get("frac_phi", {})

    def _f(x: Any) -> str:
        return "n/a" if x is None else f"{float(x):.4g}"

    print(
        f"[eval] split={metrics.get('split')} "
        f"n_samples={metrics.get('n_samples')} n_cells={metrics.get('n_cells')}"
    )
    print(
        f"  calibration: gamma_inj={_f(cal.get('gamma_inj'))} "
        f"gamma_scat={_f(cal.get('gamma_scat'))} "
        f"alpha_inj={_f(cal.get('alpha_inj'))} "
        f"R2_inj={_f(cal.get('r2_inj'))} R2_scat={_f(cal.get('r2_scat'))} "
        f"monotonic_inj={cal.get('monotonic_inj')}"
    )
    print(
        f"  lag_recovery: mean_LagMass={_f(lag.get('mean_lag_mass'))} "
        f"(thr {_f(lag.get('lag_mass_threshold'))}, pass={lag.get('mean_lag_mass_pass')}) "
        f"frac_within_tol={_f(lag.get('frac_within_tol'))}"
    )
    for ctrl, res in nul.items():
        print(f"  null[{ctrl}]: mean null_ratio={_f(res.get('mean_ratio'))}")
    print(f"  frac_Phi (signal): mean={_f(frac.get('mean'))}")


def _resolve_run_ckpt(results_dir: Path, ckpt: Optional[str]) -> Path:
    r"""Resolve a run's checkpoint from the run root (explicit > best > final).

    Used by the eval and test-plots bridges. The checkpoint is written by the train
    stage into the **run root** ``results/<tag>/`` (never into a per-split subfolder),
    so callers must pass that root here even when their per-split ``out_dir`` differs.

    Args:
        results_dir: The run root directory (``results/<tag>/``).
        ckpt: Optional explicit checkpoint path.

    Returns:
        The resolved checkpoint :class:`Path`.

    Raises:
        FileNotFoundError: If ``ckpt`` is given but missing, or no ``best.ckpt`` /
            ``final.ckpt`` exists under ``results_dir``.
    """
    if ckpt:
        p = Path(ckpt)
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    for name in ("best.ckpt", "final.ckpt"):
        p = results_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"no checkpoint under {results_dir} (best.ckpt / final.ckpt); pass --ckpt"
    )


def _resolve_split_npz(cache_dir: Path, split: str) -> Path:
    r"""Resolve a cache split ``.npz`` (requested split, falling back test->val->train)."""
    order = [split] + [s for s in ("test", "val", "train") if s != split]
    for s in order:
        p = cache_dir / f"{s}.npz"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"no cache split under {cache_dir} (looked for {order}); run --stage build first."
    )


def _build_runner_and_loader(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    arm: Optional[str] = None,
    ckpt: Optional[str] = None,
    split: str = "test",
    out_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
    attach_raw_provider: bool = False,
) -> Tuple[Any, Any, str, Path]:
    r"""Build a ``TestRunner`` + v2 ``DataLoader`` from a checkpoint (S4-T04).

    The shared seam behind every model-dependent analysis stage: ``test_plots``,
    ``calibration`` (S5), ``lag_intervention`` (S6) and ``cmi`` (S7) all need exactly this
    pair, and nothing about it is specific to any of them.

    The model is rebuilt from the checkpoint's own ``model_class`` + ``model_kwargs`` and
    cross-checked against the class the config expects, so grading arm B's checkpoint under
    arm C's config raises rather than silently loading the wrong architecture (S1-T04). The
    three arms are structurally identical except for ``posterior_logvar``, so a strict
    state-dict load would not always catch the swap.

    Note:
        ``attach_raw_provider`` is off by default. The provider is only needed by the
        ``test_plots`` raw panel, it regenerates waveforms on demand, and importing it drags in
        the scattering adapter (and hence ``kymatio``). Keeping it opt-in lets the analysis
        stages import this helper on a machine without ``kymatio`` installed.

    Args:
        config: The parsed config tree (already arm-resolved by the caller).
        benchmark: Active benchmark key under ``benchmarks``.
        arm: Resolved arm name, used only to locate ``results/<tag>/<arm>/`` when ``out_dir``
            is omitted.
        ckpt: Explicit checkpoint path, else ``best.ckpt`` / ``final.ckpt`` under the run dir.
        split: Cache split to load (``test`` / ``val`` / ``train``).
        out_dir: Directory the runner writes its artifacts under (defaults to the run dir).
        batch_size: Loader batch size (defaults to ``optim.batch_size``).
        attach_raw_provider: Regenerate raw 4 Hz FHR/UP per sample (``test_plots`` only). The
            provider is lazy, so only the rows the caller actually consumes are regenerated.

    Returns:
        ``(runner, loader, used_split, ckpt_path)``.

    Raises:
        FileNotFoundError: When no checkpoint can be resolved.
        RuntimeError: When the checkpoint's weights do not load into the rebuilt model.
    """
    import torch  # local: pulls the model / testing stack

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        resolve_cache_dir,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v2 import (
        SyntheticTEDatasetV2,
        make_dataloader,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (
        rebuild_model_from_checkpoint,
        resolved_model_class_name,
    )
    from model.vae_teb_prediction.testing.base import TestRunner
    from train.graph_models_utils import load_checkpoint_strict

    run_dir = _run_dir(config, benchmark, arm)
    results_dir = run_dir if out_dir is None else Path(out_dir)
    ckpt_path = _resolve_run_ckpt(run_dir, ckpt)
    cache_dir = resolve_cache_dir(config, benchmark=benchmark)
    npz = _resolve_split_npz(cache_dir, split)
    bs = int(config.get("optim", {}).get("batch_size", 32)) if batch_size is None \
        else int(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Rebuild the EXACT architecture from the checkpoint's embedded model_class + model_kwargs
    # (weights_only=False: a trusted local checkpoint carrying non-tensor metadata), then load
    # the state dict from the SAME already-deserialised blob (load_checkpoint_strict accepts an
    # object containing a state_dict) so the file is not read/unpickled twice.
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    expected_class = resolved_model_class_name(config["model"])
    model, _ = rebuild_model_from_checkpoint(blob, device, expected_class=expected_class)
    if load_checkpoint_strict(model, blob) is None:
        raise RuntimeError(f"could not load v2 checkpoint {ckpt_path} into the model")
    model.eval()

    results_dir.mkdir(parents=True, exist_ok=True)
    runner = TestRunner(
        model=model,
        device=device,
        output_dir=results_dir,
        warmup_steps=int(getattr(model, "warmup_period", 30)),
        horizon=int(getattr(model, "horizon", 30)),
        max_lag=int(getattr(model, "max_lag", 90)),
        use_up_st=bool(getattr(model, "use_up_st", True)),
    )

    used_split = Path(npz).stem
    raw_provider = None
    if attach_raw_provider:
        # Deterministic raw-waveform regenerator for the per-sample diagnostic's first panel
        # (regenerated on demand from ``sample_raw_index``; the raw is never cached).
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E501
            make_raw_provider,
        )
        try:
            raw_provider = make_raw_provider(
                config, used_split, benchmark=benchmark, cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- the raw panel is a nicety, never a blocker
            print(f"[runner] raw provider unavailable ({exc}); raw panel will be empty.")
            raw_provider = None

    dataset = SyntheticTEDatasetV2(npz, raw_provider=raw_provider)
    loader = make_dataloader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    return runner, loader, used_split, ckpt_path


def run_test_plots(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    arm: Optional[str] = None,
    ckpt: Optional[str] = None,
    split: str = "test",
    analysis_samples: int = 4,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""Drive a v2 cache split through the standard testing per-sample diagnostics (S7-T07).

    Builds a v2 ``DataLoader`` from the requested split and runs the **standard**
    ``testing`` per-sample analyses (:func:`testing.analyses.qualitative.run_sample_diagnostics`
    and :func:`testing.analyses.kld_lag_diagnostics.run_kld_lag_diagnostics`) on it, so the
    usual ``samples_diag/`` PDFs + ``sample_metrics.csv`` are produced with the synthetic-TE
    provenance ($\mathrm{TE}_{\mathrm{inj}}$, $\mathrm{TE}_{\mathrm{scat}}$,
    $\mathrm{frac}_\Phi$, true lag) rendered by the S7-T06 metadata bridge -- **no HDF5 paths
    or ``stats.hdf5`` needed**.

    Note:
        The runner is built **directly** from the ``vae_teb_lag_attn_v1`` model that
        ``synthetic_v2`` trains (via the checkpoint's embedded ``model_kwargs``), rather than
        through :func:`testing.run_tests.run_full_test_pipeline`. The latter's
        :meth:`TestRunner.from_checkpoint` rebuilds the model from
        ``testing.base``'s import, which is pinned to the *legacy* ``vae_teb_lag_attn_old``
        architecture and cannot align a v1 checkpoint. Constructing the runner here keeps the
        shared testing pipeline untouched (additive) while reusing its exact analysis code.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        arm: Resolved arm name (``None`` for the arm-less v1/v2 layout).
        ckpt: Optional explicit checkpoint path (else best/final under the run dir).
        split: Which cache split to plot (default ``test``; falls back to val/train).
        analysis_samples: Number of TE-annotated per-sample diagnostic PDFs to emit.
        out_dir: Optional override for the run directory (defaults to the run dir).

    Returns:
        A dict ``{'out_dir': Path, 'samples_dir': Path, 'sample_diagnostics': ...,
        'kld_lag_diagnostics': ...}``.
    """
    from model.vae_teb_prediction.testing.analyses.kld_lag_diagnostics import (
        run_kld_lag_diagnostics,
    )
    from model.vae_teb_prediction.testing.analyses.qualitative import run_sample_diagnostics

    results_dir = _run_dir(config, benchmark, arm) if out_dir is None else Path(out_dir)
    out = results_dir / "test_plots"
    # Only the first ``analysis_samples`` rows are diagnosed, so a small batch keeps the raw
    # regeneration to the few cells actually plotted (the loader isn't fully consumed).
    batch_size = int(config.get("optim", {}).get("batch_size", 32))
    plot_batch_size = max(1, min(batch_size, int(analysis_samples)))
    runner, loader, _used_split, _ckpt_path = _build_runner_and_loader(
        config, benchmark=benchmark, arm=arm, ckpt=ckpt, split=split, out_dir=out,
        batch_size=plot_batch_size, attach_raw_provider=True,
    )

    samples_dir = out / "samples_diag"
    sample_res = run_sample_diagnostics(
        runner, loader, max_samples=int(analysis_samples), output_dir=samples_dir
    )
    # The KLD + lag diagnostic (already TE-aware for synthetic batches) is a non-fatal bonus.
    kld_lag_res: Optional[Dict[str, Any]] = None
    try:
        kld_lag_res = run_kld_lag_diagnostics(
            runner, loader, max_samples=int(analysis_samples),
            output_dir=out / "samples_kld_lag",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[test_plots] kld_lag diagnostics skipped: {exc}")

    print(f"[test_plots] wrote sample diagnostics -> {samples_dir}")
    return {
        "out_dir": out,
        "samples_dir": samples_dir,
        "sample_diagnostics": sample_res,
        "kld_lag_diagnostics": kld_lag_res,
    }


# =============================================================================
# Per-split evaluation / reporting (grade EVERY dataset split, not just test)
# =============================================================================

# The dataset splits, in report order. A default run grades and plots every split
# whose cache exists (train AND val AND test) into its OWN results subfolder
# ``results/<tag>/<split>/`` so we can see how the model does on each; an explicit
# ``--split NAME`` restricts to one.
_ALL_SPLITS = ("train", "val", "test")


def _resolve_splits(config: Dict[str, Any], benchmark: str,
                    split: Optional[str]) -> List[str]:
    r"""Resolve which dataset splits to evaluate / plot.

    ``split`` of ``None`` / ``"all"`` selects **every** split whose cache ``.npz`` exists
    (in ``train, val, test`` order), so a default run grades train, val AND test; an
    explicit name restricts to that one split. Falls back to ``["test"]`` when nothing is
    discoverable (e.g. before a build), matching the prior single-split default.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        split: ``None`` / ``"all"`` for every present split, or a single split name.

    Returns:
        The ordered list of split names to process.
    """
    present: List[str] = []
    try:
        cache_dir = resolve_cache_dir(config, benchmark=benchmark)
        present = [s for s in _ALL_SPLITS if (cache_dir / f"{s}.npz").is_file()]
    except Exception:  # noqa: BLE001 -- cache path not resolvable yet (pre-build)
        present = []
    if split in (None, "", "all", "ALL"):
        return present or ["test"]
    return [str(split)]


def _split_dir(results_dir: Path, split: str) -> Path:
    r"""The per-split output directory ``results/<tag>/<split>/`` (created)."""
    d = results_dir / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def _eval_splits(config: Dict[str, Any], benchmark: str, *, ckpt: Optional[str],
                 splits: Sequence[str], results_dir: Path,
                 arm: Optional[str] = None) -> Dict[str, Any]:
    r"""Run ``eval_v2.run_eval`` for each split into its own ``<run_dir>/<split>/`` dir.

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key.
        ckpt: Explicit checkpoint path, or ``None`` for auto-discovery.
        splits: The splits to grade.
        results_dir: The run's ``results/<tag>/<arm>/`` root.
        arm: Resolved arm name, stamped into each split's ``metrics.json`` (S4-T03).

    Returns:
        ``{split: metrics_dict}`` for each graded split.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
        run_eval,
    )
    # Resolve the checkpoint ONCE from the run root: the train stage writes best/final.ckpt
    # into results/<tag>/<arm>/, but each split evaluates into its own
    # results/<tag>/<arm>/<split>/ ``out_dir``. Passing that per-split ``out_dir`` as the
    # checkpoint search root would (and did) fail auto-discovery, so we hand every split the
    # resolved explicit path.
    ckpt_path = str(_resolve_run_ckpt(results_dir, ckpt))
    out: Dict[str, Any] = {}
    for s in splits:
        sdir = _split_dir(results_dir, s)
        print(f"[eval:{s}] grading -> {sdir / 'metrics.json'}")
        metrics = run_eval(config, benchmark=benchmark, ckpt=ckpt_path, split=s,
                           out_dir=sdir, arm=arm)
        _print_eval_metrics(metrics)
        out[s] = metrics
    return out


def _test_plots_splits(config: Dict[str, Any], benchmark: str, *, ckpt: Optional[str],
                       analysis_samples: int, splits: Sequence[str],
                       results_dir: Path, arm: Optional[str] = None) -> None:
    r"""Render the standard-testing per-sample diagnostics for each split (guarded)."""
    # Resolve the checkpoint from the run root (best/final live in results/<tag>/<arm>/, not
    # in the per-split out_dir passed below). Guarded so a missing checkpoint only warns --
    # these diagnostics must never gate the run.
    try:
        ckpt_path = str(_resolve_run_ckpt(results_dir, ckpt))
    except FileNotFoundError as exc:
        print(f"[test_plots][warn] {exc}")
        return
    for s in splits:
        sdir = _split_dir(results_dir, s)
        try:
            run_test_plots(config, benchmark=benchmark, arm=arm, ckpt=ckpt_path, split=s,
                           analysis_samples=analysis_samples, out_dir=sdir)
        except Exception as exc:  # noqa: BLE001 -- diagnostics only, never fatal
            print(f"[test_plots:{s}][warn] {type(exc).__name__}: {exc}")


def _report_splits(config: Dict[str, Any], benchmark: str, *, splits: Sequence[str],
                   results_dir: Path) -> Dict[str, Path]:
    r"""Assemble a full report + figure gallery per split, then a cross-split index.

    Each split's report and figures land in ``results/<tag>/<split>/`` (so every dataset
    gets its own gallery); a top-level ``results/<tag>/report.md`` cross-links them and
    tabulates the headline gates side by side so the splits are directly comparable.

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key.
        splits: The splits to report.
        results_dir: The run's ``results/<tag>/`` root.

    Returns:
        ``{split: report_path}``.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
        final_report_v2,
    )
    paths: Dict[str, Path] = {}
    for s in splits:
        sdir = _split_dir(results_dir, s)
        p = final_report_v2(config, benchmark=benchmark, out_dir=sdir, split=s)
        print(f"[report:{s}] wrote {p}")
        paths[s] = p
    _write_split_index(results_dir, list(splits), config=config, benchmark=benchmark)
    return paths


def _write_split_index(results_dir: Path, splits: Sequence[str], *,
                       config: Optional[Dict[str, Any]] = None,
                       benchmark: str = "G1_raw") -> Path:
    r"""Write a top-level ``report.md`` cross-linking every split's report + gate table.

    Reads each split's ``metrics.json`` and tabulates the headline gates (γ vs TE_inj /
    TE_scat, per-sample slope, LagMass, null ratios) side by side, so "how are we doing on
    train vs val vs test" is answerable at a glance.

    The index is written at the **run** root (``results/<tag>/<arm>/`` under a v3 config), so
    two of its links depend on the layout (S4-T06): ``figures/training_curves.html`` is local
    to the run root, while the split-independent data-generation gallery lives one level up at
    the *tag* root. The heading likewise names the experiment tag, not the arm directory.

    Args:
        results_dir: The run root (``results/<tag>/<arm>/``, or ``results/<tag>/`` arm-less).
        splits: The processed splits.
        config: The parsed config tree, used to resolve the tag root. When omitted the tag is
            read from the directory name (the pre-arm behaviour).
        benchmark: Active benchmark key.

    Returns:
        The written ``report.md`` :class:`Path`.
    """
    import json
    import os

    def _g(x: Any) -> str:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return "n/a"
        return f"{xf:.4g}" if xf == xf else "n/a"

    if config is not None:
        root = _results_dir(config, benchmark)
        tag = root.name
    else:
        root, tag = results_dir, results_dir.name
    # ``figures`` when the index sits at the tag root, ``../figures`` when it sits in an arm dir.
    shared_rel = os.path.relpath(root / "figures", results_dir).replace(os.sep, "/")
    lines: List[str] = [
        f"# synthetic_v2 per-split report index — `{tag}`", "",
        "Every dataset split is graded and plotted into its own subfolder so the model's "
        "behaviour can be compared across splits (over-/under-fitting shows up as a "
        "train-vs-test gap in these gates).", "",
        "| split | n | γ_inj (cell) | γ_scat (cell) | γ_inj (sample) | mean LagMass | "
        "null (shuffle) | null (reverse) | report |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in splits:
        m_path = results_dir / s / "metrics.json"
        d: Dict[str, Any] = {}
        if m_path.is_file():
            try:
                d = json.loads(m_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                d = {}
        cal = d.get("calibration", {}) or {}
        lag = d.get("lag_recovery", {}) or {}
        nul = d.get("null_controls", {}) or {}
        link = f"[`{s}/report.md`]({s}/report.md)" if m_path.is_file() else "_(no metrics)_"
        lines.append(
            f"| {s} | {d.get('n_samples', 'n/a')} | {_g(cal.get('gamma_inj'))} "
            f"| {_g(cal.get('gamma_scat'))} | {_g(cal.get('gamma_inj_sample'))} "
            f"| {_g(lag.get('mean_lag_mass'))} "
            f"| {_g((nul.get('shuffle') or {}).get('mean_ratio'))} "
            f"| {_g((nul.get('reverse') or {}).get('mean_ratio'))} | {link} |"
        )
    lines += ["", "Split-independent data-generation figures (raw / scattering / latent / "
              f"TE-authoring) live in [`{shared_rel}/`]({shared_rel}/); interactive training "
              "curves for this run in "
              "[`figures/training_curves.html`](figures/training_curves.html).", ""]
    path = results_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote cross-split index {path}")
    return path


# =============================================================================
# Edit-and-run dict driver (no argparse needed; mirrors synthetic/run_mixed_pipeline)
# =============================================================================

# The canonical stage order and the on-by-default set are DERIVED from ``_STAGE_REGISTRY``
# via :func:`stage_order` / :func:`stage_defaults` (S4-T05). ``run_pipeline`` validates
# ``PIPELINE['stages']`` against ``stage_order()`` so a typo fails loudly instead of silently
# skipping a stage, and a plugin-registered analysis stage becomes a valid key automatically.

#: Stages the dict driver dispatches with bespoke inline logic (subprocess re-exec, cache-hit
#: skipping, per-split fan-out). Everything else registered as ``model_dependent`` is handed to
#: :func:`_dispatch_stage` generically at the tail of the per-arm loop.
_BUILTIN_PIPELINE_STAGES: frozenset = frozenset({
    "solve_te", "am_check", "recover", "r0_realizability", "build", "data_previews",
    "scatter_preview", "beta_select", "train", "eval", "test_plots", "report",
})


def _banner(step: int, total: int, name: str, note: str = "") -> None:
    r"""Print a uniform stage banner.

    Args:
        step: 1-based stage index.
        total: Total number of stages.
        name: Stage name.
        note: Optional short status note appended to the banner.
    """
    line = "=" * 78
    suffix = f"  ({note})" if note else ""
    print(f"\n{line}\n[pipeline] stage {step}/{total}: {name}{suffix}\n{line}")


def _run_subprocess(cmd: List[str], *, dry_run: bool) -> None:
    r"""Run a child process from the repo root, streaming its output.

    Training and the beta sweep are driven as child processes so Lightning's DDP
    launcher re-executes the scoped ``--stage train`` / ``--stage beta_select``
    command (never the whole edit-and-run pipeline) in every worker rank -- the same
    rationale as ``synthetic/run_mixed_pipeline`` subprocessing ``train_ddp``.

    Args:
        cmd: The full command (``sys.executable run_pipeline_v2.py --stage ...``).
        dry_run: When ``True``, only print the command.

    Raises:
        RuntimeError: If the child exits with a non-zero return code.
    """
    print(f"[pipeline] $ {' '.join(cmd)}")
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            f"pipeline subprocess failed (exit {proc.returncode}): {' '.join(cmd)}"
        )


def _warn_if_checkpoint_is_stale(
    ckpt_path: Path, config: Dict[str, Any], *, epochs: Optional[int] = None,
) -> None:
    r"""Print a loud warning when an existing checkpoint does not match the live config.

    ``train`` skips silently when ``final.ckpt`` exists (which is what makes a long
    multi-arm run resumable). The failure mode that guards against a wasted week is the
    inverse: a leftover **pilot** checkpoint under the same ``experiment.tag`` makes every
    downstream stage grade a 400-step model and emit a complete, plausible-looking
    ``arms_report.md``. The checkpoint records the epoch it stopped at and the
    $\beta$ schedule actually used, so both are checked against the config here.

    Never raises: this is a readout, and a checkpoint from a legitimately resumed run
    should not abort the pipeline.

    Args:
        ckpt_path: The existing ``final.ckpt``.
        config: The arm-resolved config tree.
        epochs: The pipeline's epoch override, or ``None`` for ``optim.epochs``.
    """
    try:
        import torch

        blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 -- a readout must never break the run
        print(f"[pipeline]   (could not inspect {ckpt_path.name}: {exc})")
        return

    stopped_at = blob.get("epoch")
    want_epochs = int(epochs) if epochs is not None else int(config["optim"]["epochs"])
    ckpt_beta = (blob.get("loss_settings") or {}).get("beta_schedule")
    cfg_beta = (config.get("loss") or {}).get("beta_schedule")

    print(f"[pipeline]   existing ckpt: arm={blob.get('arm')!r} "
          f"class={blob.get('model_class')!r} stopped_at_epoch={stopped_at}")

    warnings: List[str] = []
    if isinstance(stopped_at, int) and stopped_at + 1 < want_epochs:
        warnings.append(
            f"it stopped at epoch {stopped_at + 1} but the config asks for {want_epochs} "
            f"-- this looks like a PILOT checkpoint"
        )
    if ckpt_beta != cfg_beta:
        warnings.append(
            f"its beta_schedule {ckpt_beta} != the config's {cfg_beta}"
        )
    for msg in warnings:
        print(f"[pipeline]   WARNING: {msg}")
    if warnings:
        print("[pipeline]   WARNING: every downstream stage will grade THIS checkpoint. "
              "Delete the arm's results dir, or set force_retrain=True.")


def _stage_subprocess_cmd(
    config_path: Path,
    stage: str,
    *,
    devices: Any = None,
    epochs: Optional[int] = None,
    pilot: bool = False,
    arm: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[str]:
    r"""Assemble a ``--stage {train,beta_select}`` subprocess command (DDP-safe).

    Args:
        config_path: Path to the YAML config handed to the child.
        stage: ``train`` or ``beta_select``.
        devices: Lightning devices spec (``None`` keeps the config/train default).
        epochs: Optional epoch override.
        pilot: Pass ``--pilot`` for the short training smoke.
        arm: The resolved ``synthetic_v3`` arm; forwarded as ``--arm`` so the DDP
            re-exec trains and checkpoints under the same arm.
        max_samples: Optional per-split sample cap, forwarded as ``--max-samples``.

    Returns:
        The command list.
    """
    cmd = [
        sys.executable, str(_MODULE_FILE),
        "--config", str(config_path), "--stage", stage,
    ]
    if pilot:
        cmd.append("--pilot")
    if devices is not None:
        cmd += ["--devices", str(devices)]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    if arm is not None:
        cmd += ["--arm", str(arm)]
    if max_samples is not None:
        cmd += ["--max-samples", str(int(max_samples))]
    return cmd


def run_pipeline(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    r"""Run the ``synthetic_v2`` stages in order from an edit-and-run config dict.

    This is the no-argparse driver (mirrors
    ``synthetic/run_mixed_pipeline.run_pipeline``): edit the ``PIPELINE`` dict in
    ``__main__`` (or build one in Python and call this) and every enabled stage runs
    in :data:`_STAGE_ORDER`. Cheap / in-process stages call their worker functions
    directly; ``train`` and ``beta_select`` are driven as scoped subprocesses so a
    multi-GPU DDP re-exec never re-runs the whole pipeline.

    Recognised ``pipeline`` keys:

    * ``config_path`` -- YAML config (default: the sibling ``config_synth_v2.yaml``).
    * ``benchmark`` -- active benchmark (``None`` -> ``experiment.benchmark``).
    * ``pilot`` -- ``True`` uses the pilot grid for ``r0_realizability`` / ``build``;
      ``False`` (default) uses the full locked mix grid.
    * ``force_rebuild`` -- ``True`` regenerates cached ``build`` parts (``resume=False``).
    * ``force_retrain`` -- ``True`` retrains even when ``final.ckpt`` exists.
    * ``train_pilot`` -- ``True`` runs the short training smoke (``--pilot``).
    * ``devices`` / ``epochs`` -- training overrides forwarded to the subprocess.
    * ``arms`` -- the ``synthetic_v3`` arm sweep for the model-dependent stages
      (``None`` -> every arm in the config's ``arms`` block, or one arm-less v1/v2 run).
    * ``ckpt`` / ``split`` / ``analysis_samples`` -- ``eval`` / ``test_plots`` knobs.
    * ``solve_te_args`` -- ``(target_te, D)`` for the ``solve_te`` stage.
    * ``scatter_preview`` -- sub-dict ``{target_te, delay, n}`` for that stage.
    * ``dry_run`` -- print the plan without executing.
    * ``stages`` -- ``{stage: bool}`` toggles overriding :data:`_STAGE_DEFAULTS`.

    Args:
        pipeline: The edit-and-run configuration (see the ``PIPELINE`` dict in
            ``__main__`` for every key and its default).

    Returns:
        ``{stage: status}`` plus ``benchmark`` and ``results_dir``. ``status`` is one
        of ``done`` / ``skipped (disabled)`` / ``skipped (exists)`` / ``dry-run`` /
        ``failed (non-fatal)``.

    Raises:
        ValueError: On an unknown stage key, an unknown benchmark, or a ``solve_te``
            stage enabled without ``solve_te_args``.
        RuntimeError: When a subprocess stage exits non-zero.
    """
    t_start = time.time()
    _load_stage_plugins()   # so a plugin-registered analysis stage is a valid PIPELINE key
    order = stage_order()
    stages = stage_defaults()
    stages.update(pipeline.get("stages") or {})
    unknown = set(stages) - set(order)
    if unknown:
        raise ValueError(
            f"unknown stage keys {sorted(unknown)}; valid stages are {list(order)}."
        )

    dry_run = bool(pipeline.get("dry_run", False))
    config_path = Path(pipeline.get("config_path", _DEFAULT_CONFIG))
    config = load_config(config_path)
    benchmark = (
        pipeline.get("benchmark")
        or config.get("experiment", {}).get("benchmark", "G1_raw")
    )
    if benchmark not in config.get("benchmarks", {}):
        raise ValueError(
            f"benchmark {benchmark!r} has no matching block under 'benchmarks'."
        )

    pilot = bool(pipeline.get("pilot", False))
    force_rebuild = bool(pipeline.get("force_rebuild", False))
    force_retrain = bool(pipeline.get("force_retrain", False))
    train_pilot = bool(pipeline.get("train_pilot", False))
    devices = pipeline.get("devices")
    # Resolve once, here, so the printed plan, the banner and the subprocess command all
    # agree. ``None`` -> ddp.devices for a headline run, train.devices under train_pilot.
    resolved_devices = _resolve_train_devices(config, devices=devices, pilot=train_pilot)
    epochs = pipeline.get("epochs")
    ckpt = pipeline.get("ckpt")
    # ``split=None`` (the default) grades/plots EVERY available split (train, val, test)
    # into its own ``results/<tag>/<split>/`` subfolder; an explicit name restricts to one.
    split = pipeline.get("split", None)
    analysis_samples = int(pipeline.get("analysis_samples", 4))
    results_dir = _results_dir(config, benchmark)
    n_stages = len(order)
    status: Dict[str, Any] = {}

    def _sb(name: str, note: str = "") -> None:
        _banner(order.index(name) + 1, n_stages, name, note)

    print(
        f"[pipeline] synthetic_v2 edit-and-run\n"
        f"           config      = {config_path}\n"
        f"           benchmark   = {benchmark}   tag = {config.get('experiment', {}).get('tag')}\n"
        f"           results     = {results_dir}\n"
        f"           pilot grid  = {pilot}   dry_run = {dry_run}"
    )

    # --- solve_te (diagnostic query) -----------------------------------------
    _sb("solve_te")
    if not stages["solve_te"]:
        status["solve_te"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    else:
        args = pipeline.get("solve_te_args")
        if not args or len(args) != 2:
            raise ValueError(
                "stages.solve_te is enabled but PIPELINE['solve_te_args'] = "
                "(target_te, D) is not set."
            )
        te, d = float(args[0]), int(args[1])
        if dry_run:
            status["solve_te"] = "dry-run"
            print(f"[pipeline] would solve B for target_te={te:g}, D={d}")
        else:
            _print_solution(te, d, solve_te(config, te, d, benchmark=benchmark))
            status["solve_te"] = "done"

    # --- am_check (diagnostic) -----------------------------------------------
    _sb("am_check")
    if not stages["am_check"]:
        status["am_check"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["am_check"] = "dry-run"
        print("[pipeline] would run the AM-separation pre-check.")
    else:
        _print_am_check(am_separation_from_config(config, benchmark=benchmark))
        status["am_check"] = "done"

    # --- recover (opt-in render-knob sweep) ----------------------------------
    _sb("recover")
    if not stages["recover"]:
        status["recover"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["recover"] = "dry-run"
        print(f"[pipeline] would sweep render knobs -> {results_dir / 'recovery.json'}")
    else:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
            sweep_render_knobs,
        )
        sweep_render_knobs(config, benchmark=benchmark, out_dir=results_dir)
        status["recover"] = "done"

    # --- r0_realizability (three-TE de-risk gate) ----------------------------
    _sb("r0_realizability", note="pilot" if pilot else "full grid")
    if not stages["r0_realizability"]:
        status["r0_realizability"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["r0_realizability"] = "dry-run"
        print(f"[pipeline] would probe realizability -> {results_dir / 'realizability.json'}")
    else:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
            run_realizability_preflight,
        )
        run_realizability_preflight(
            config, benchmark=benchmark, pilot=pilot, out_dir=results_dir,
        )
        status["r0_realizability"] = "done"

    # --- build (generate -> scatter -> normalise -> cache) -------------------
    _sb("build", note="pilot" if pilot else "full grid")
    if not stages["build"]:
        status["build"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["build"] = "dry-run"
        print(f"[pipeline] would build the {'pilot' if pilot else 'full'} cache "
              f"(resume={not force_rebuild}).")
    else:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
            build_all,
        )
        cache_dir = resolve_cache_dir(config, benchmark=benchmark)
        if cache_is_complete(cache_dir) and not force_rebuild:
            # build_all rewrites every split .npz unconditionally; never do that to a
            # populated cache without an explicit force_rebuild.
            print(f"[build] cache up to date -> {cache_dir}")
            status["build"] = "skipped (cache up to date)"
        else:
            out_dir = build_all(
                config, benchmark=benchmark, pilot=pilot, resume=not force_rebuild,
            )
            print(f"[build] wrote cache -> {out_dir}")
            status["build"] = "done"

    # --- data_previews (raw + scattering + latent gallery) -------------------
    _sb("data_previews")
    if not stages["data_previews"]:
        status["data_previews"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["data_previews"] = "dry-run"
        print("[pipeline] would render the raw / scattering / latent preview gallery.")
    else:
        dp = dict(pipeline.get("data_previews") or {})
        # Figures only -- never gate the run on a plotting failure.
        try:
            data_previews(
                config, benchmark=benchmark,
                target_te=float(dp.get("target_te", 2.0)),
                delay=int(dp.get("delay", 8)),
                n=int(dp.get("n", 16)),
                include_null=bool(dp.get("include_null", True)),
            )
            status["data_previews"] = "done"
        except Exception as exc:  # noqa: BLE001 -- diagnostics only
            print(f"[pipeline][warn] data_previews failed: {type(exc).__name__}: {exc}")
            status["data_previews"] = "failed (non-fatal)"

    # --- scatter_preview (diagnostic heatmap) --------------------------------
    _sb("scatter_preview")
    if not stages["scatter_preview"]:
        status["scatter_preview"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["scatter_preview"] = "dry-run"
        print("[pipeline] would transform a strong cell and write the scattering heatmap.")
    else:
        sp = dict(pipeline.get("scatter_preview") or {})
        scatter_preview(
            config, benchmark=benchmark,
            target_te=float(sp.get("target_te", 2.0)),
            delay=int(sp.get("delay", 8)),
            n=int(sp.get("n", 16)),
        )
        status["scatter_preview"] = "done"

    # --- model-dependent stages, swept per arm -------------------------------
    # The model-free stages above run ONCE at the arm-less results/<tag>/ root. The stages
    # below consume or produce a checkpoint, so they repeat for every configured arm into
    # results/<tag>/<arm>/. ``arms`` defaults to every arm in the config's ``arms`` block
    # (so a v3 config sweeps parity / v3_noncausal / v3_prod), or to a single arm-less run
    # for a v1 / v2 config.
    arm_list = list(pipeline.get("arms") or list(config.get("arms") or {}) or [None])

    # Which splits to grade / plot: every available split by default (each into its own
    # results/<tag>/<arm>/<split>/ subfolder), or the one requested via ``split``.
    splits = _resolve_splits(config, benchmark, split)

    for arm in arm_list:
        arm_cfg = resolve_arm(config, arm)
        arm_dir = _run_dir(config, benchmark, arm)
        arm_note = f"arm={arm}" if arm is not None else ""

        def _asb(name: str, note: str = "", _arm_note: str = arm_note) -> None:
            joined = ", ".join(x for x in (_arm_note, note) if x)
            _banner(order.index(name) + 1, n_stages, name, joined)

        def _skey(name: str, _arm: Optional[str] = arm) -> str:
            return f"{name}[{_arm}]" if _arm is not None else name

        if dry_run and arm is not None:
            _print_arm_plan(config, arm)

        # --- beta_select (opt-in KL sweep; DDP-safe subprocess) --------------
        _asb("beta_select")
        if not stages["beta_select"]:
            status[_skey("beta_select")] = "skipped (disabled)"
            print("[pipeline] disabled.")
        else:
            _run_subprocess(
                _stage_subprocess_cmd(
                    config_path, "beta_select", devices=resolved_devices, epochs=epochs,
                    pilot=train_pilot, arm=arm,
                ),
                dry_run=dry_run,
            )
            status[_skey("beta_select")] = "dry-run" if dry_run else "done"

        # --- train (fit -> checkpoint; DDP-safe subprocess) ------------------
        ckpt_final = arm_dir / "final.ckpt"
        _asb("train", note=f"devices={resolved_devices}"
                          + (", pilot" if train_pilot else ""))
        if not stages["train"]:
            status[_skey("train")] = "skipped (disabled)"
            print("[pipeline] disabled.")
        elif ckpt_final.is_file() and not force_retrain:
            status[_skey("train")] = "skipped (exists)"
            print(f"[pipeline] checkpoint exists, skipping training: {ckpt_final}\n"
                  f"           (set force_retrain=True to retrain)")
            _warn_if_checkpoint_is_stale(ckpt_final, arm_cfg, epochs=epochs)
        else:
            _run_subprocess(
                _stage_subprocess_cmd(
                    config_path, "train", devices=resolved_devices, epochs=epochs,
                    pilot=train_pilot, arm=arm,
                ),
                dry_run=dry_run,
            )
            status[_skey("train")] = "dry-run" if dry_run else "done"

        # --- eval (grade the checkpoint, per split) --------------------------
        _asb("eval", note=f"splits={splits}")
        if not stages["eval"]:
            status[_skey("eval")] = "skipped (disabled)"
            print("[pipeline] disabled.")
        elif dry_run:
            status[_skey("eval")] = "dry-run"
            print(f"[pipeline] would grade the checkpoint on splits {splits} -> "
                  f"{arm_dir / '<split>' / 'metrics.json'}")
        else:
            _eval_splits(arm_cfg, benchmark, ckpt=ckpt, splits=splits,
                         results_dir=arm_dir, arm=arm)
            status[_skey("eval")] = "done"

        # --- test_plots (standard testing per-sample diagnostics, per split) -
        _asb("test_plots", note=f"splits={splits}")
        if not stages["test_plots"]:
            status[_skey("test_plots")] = "skipped (disabled)"
            print("[pipeline] disabled.")
        elif dry_run:
            status[_skey("test_plots")] = "dry-run"
            print(f"[pipeline] would render {analysis_samples} TE-annotated sample PDFs "
                  f"per split {splits}.")
        else:
            # Figures only -- never gate the run on a plotting failure.
            _test_plots_splits(arm_cfg, benchmark, ckpt=ckpt, arm=arm,
                               analysis_samples=analysis_samples, splits=splits,
                               results_dir=arm_dir)
            status[_skey("test_plots")] = "done"

        # --- plugin-registered analysis stages (calibration / lag_intervention / cmi) ----
        # Append-only: each was contributed by its own module via ``register_stage`` and is
        # dispatched generically here, so adding one touches neither this loop nor ``main()``.
        #
        # These run BEFORE ``report`` on purpose. Each plugin either folds a block into the
        # split's ``metrics.json`` (calibration -> ``calibration_predictive``) or drops a
        # side-car JSON the report's registered section reads (lag_intervention). Dispatching
        # them after ``report`` -- as this loop originally did -- rendered every new section
        # as ``n/a`` on a one-shot pipeline pass. Their ``StageSpec.order`` therefore only
        # sequences them among *themselves*; their position relative to the builtins is fixed
        # here.
        for name in order:
            spec = _STAGE_REGISTRY[name]
            if name in _BUILTIN_PIPELINE_STAGES or not spec.model_dependent or spec.run is None:
                continue
            _asb(name)
            if not stages.get(name, spec.default_on):
                status[_skey(name)] = "skipped (disabled)"
                print("[pipeline] disabled.")
                continue
            if dry_run:
                status[_skey(name)] = "dry-run"
                print(f"[pipeline] would run stage {name} on splits {splits}.")
                continue
            ctx = StageContext(
                config=arm_cfg, benchmark=benchmark, arm=arm, ckpt=ckpt, split=split,
                analysis_samples=analysis_samples,
                max_samples=pipeline.get("max_samples"),
            )
            rc = _dispatch_stage(spec, ctx)
            status[_skey(name)] = "done" if rc == 0 else f"exit {rc}"

        # --- report (assemble the markdown report + gallery, per split) ------
        # Last, so it sees every analysis block the plugin stages just wrote.
        _asb("report", note=f"splits={splits}")
        if not stages["report"]:
            status[_skey("report")] = "skipped (disabled)"
            print("[pipeline] disabled.")
        elif dry_run:
            status[_skey("report")] = "dry-run"
            print(f"[pipeline] would assemble a report per split {splits} under {arm_dir}.")
        else:
            _report_splits(arm_cfg, benchmark, splits=splits, results_dir=arm_dir)
            report_path = arm_dir / "report.md"  # the cross-split index
            print(f"[report] wrote {report_path}")
            status[_skey("report")] = "done"

    # --- cross-arm, model-free plugin stages (arms_report) ---------------------
    # These read *every* arm's artifacts and write once at the arm-less results/<tag>/ root, so
    # they belong after the sweep, not inside it. The generic per-arm loop above skips them
    # (`not spec.model_dependent`), and `main()` gives them `arm=None`, so this block is the only
    # place the dict driver dispatches them.
    for name in order:
        spec = _STAGE_REGISTRY[name]
        if (name in _BUILTIN_PIPELINE_STAGES or spec.model_dependent
                or spec.run is None):
            continue
        _banner(order.index(name) + 1, n_stages, name)
        if not stages.get(name, spec.default_on):
            status[name] = "skipped (disabled)"
            print("[pipeline] disabled.")
            continue
        if dry_run:
            status[name] = "dry-run"
            print(f"[pipeline] would run stage {name} across arms {arm_list}.")
            continue
        ctx = StageContext(
            config=config, benchmark=benchmark, arm=None, split=split,
            analysis_samples=analysis_samples, max_samples=pipeline.get("max_samples"),
        )
        rc = _dispatch_stage(spec, ctx)
        status[name] = "done" if rc == 0 else f"exit {rc}"

    # --- summary --------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"[pipeline] finished in {elapsed / 60.0:.1f} min")
    for name, st in status.items():
        print(f"  {name:24s} {st}")
    print(f"[pipeline] artifacts under {results_dir} (arms: "
          f"{[a for a in arm_list if a is not None] or 'none'})")
    status["benchmark"] = benchmark
    status["results_dir"] = str(results_dir)
    return status


def build_parser() -> argparse.ArgumentParser:
    r"""Build the ``run_pipeline_v2`` argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` exposing ``--solve-te`` and
        ``--am-check`` (implemented) and ``--stage`` (registered for later sprints).
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline_v2",
        description=(
            "synthetic_v2 driver. Implemented: --solve-te, --am-check, "
            "--scatter-preview, --recover, and --stage {r0_realizability, build, "
            "data_previews, train (+--pilot), beta_select, eval, test_plots, report}."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to config_synth_v2.yaml (default: the sibling config).",
    )
    parser.add_argument(
        "--solve-te",
        nargs=2,
        metavar=("TARGET_TE", "D"),
        help=(
            "Solve the coupling B for a cell authored by a target injected block "
            "TE (nats) at a fixed lag D (decimated steps); prints B, TE, SNR."
        ),
    )
    parser.add_argument(
        "--am-check",
        action="store_true",
        help=(
            "Run the AM-separation analytic pre-check (S1-T04) on the active "
            "benchmark's raw/scattering config; prints the margin, preservation, and "
            "modulation depth."
        ),
    )
    parser.add_argument(
        "--scatter-preview",
        action="store_true",
        help=(
            "Transform one strong am_carrier cell through the real scattering + "
            "normalisation (S2-T04): prints the four field shapes and the fs-correct "
            "coupled channel, and writes a scattering heatmap under results/<tag>/figures/."
        ),
    )
    # ``--stage`` choices come from the registry, so a plugin-registered analysis stage is
    # dispatchable (and documented) without editing this parser.
    _load_stage_plugins()
    parser.add_argument(
        "--stage",
        choices=stage_names(),
        help="Pipeline stage to run. " + "; ".join(
            f"{n} ({_STAGE_REGISTRY[n].help})" for n in stage_names()
            if _STAGE_REGISTRY[n].help
        ) + ".",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="With --stage train / beta_select, override the number of epochs.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "With --stage train / beta_select, the Lightning devices spec: an int (first "
            "N GPUs), a comma list ('0,1,2'), or 1 for single-GPU (default). >1 selects DDP."
        ),
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "With --stage r0_realizability / build, use the small pilot grid (the "
            "default for these stages; kept for explicitness)."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "With --stage r0_realizability / build, run the FULL locked mix grid at "
            "mix.n_per_cell_{train,val,test} instead of the pilot grid (expensive). "
            "Mutually exclusive with --pilot."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "With --stage build, regenerate over a COMPLETE cache. Without it, build "
            "reports 'cache up to date' and exits without touching the .npz files; "
            "build_all always rewrites every split, so this guards the production cache."
        ),
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help=(
            "Run the frac_Phi recovery sweep (S3-T06) over the render knobs "
            "(f_pulse / am_offset_ratio / omega); writes recovery.json under "
            "results/<tag>/ and prints the chosen setting."
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "With --stage eval / test_plots, an explicit checkpoint path; otherwise "
            "best.ckpt / final.ckpt under results/<tag>/ is auto-discovered."
        ),
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "test", "val", "train"],
        help=(
            "With --stage eval / test_plots / report, which split(s) to process. Default "
            "'all' grades and plots EVERY cached split (train, val, test) into its own "
            "results/<tag>/<split>/ subfolder; a single name restricts to that split."
        ),
    )
    parser.add_argument(
        "--analysis-samples",
        type=int,
        default=4,
        help=(
            "With --stage test_plots, the number of per-sample TE-annotated diagnostic "
            "PDFs to emit through the standard testing pipeline (default 4)."
        ),
    )
    parser.add_argument(
        "--arm",
        default=None,
        help=(
            "The synthetic_v3 ablation arm (e.g. 'parity' / 'v3_noncausal' / 'v3_prod') "
            "resolved by deep-merging arms.<name> over the base model/loss blocks. "
            "Model-dependent stages write under results/<tag>/<arm>/. Required when the "
            "config defines more than one arm; a single-arm config defaults to it; a "
            "config with no 'arms' block ignores it (the v1 / v2 path)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional per-split cap on the number of samples (forwarded to the stages "
            "that honour it; leaves the full grid when unset)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    r"""CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A process exit code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.solve_te is not None:
        config = load_config(args.config)
        raw_te, raw_delay = args.solve_te
        try:
            target_te = float(raw_te)
        except ValueError:
            parser.error(f"--solve-te TARGET_TE must be a number, got {raw_te!r}")
        try:
            delay = int(raw_delay)
        except ValueError:
            parser.error(f"--solve-te D must be an integer, got {raw_delay!r}")
        if target_te < 0.0:
            parser.error(f"--solve-te TARGET_TE must be >= 0, got {target_te}")
        if delay < 1:
            parser.error(f"--solve-te D must be >= 1 decimated steps, got {delay}")
        # Honour the config's active-benchmark selector (experiment.benchmark).
        benchmark = config.get("experiment", {}).get("benchmark", "G1_raw")
        if benchmark not in config.get("benchmarks", {}):
            parser.error(
                f"experiment.benchmark={benchmark!r} has no matching block under "
                "'benchmarks'."
            )
        try:
            solution = solve_te(config, target_te, delay, benchmark=benchmark)
        except ValueError as exc:
            # e.g. the target TE lies outside the inverter's [lo, hi] bracket; report it
            # as a clean CLI error like the other --solve-te input checks, not a traceback.
            parser.error(f"--solve-te could not solve target_te={target_te} D={delay}: {exc}")
        _print_solution(target_te, delay, solution)
        return 0

    if args.am_check:
        config = load_config(args.config)
        benchmark = config.get("experiment", {}).get("benchmark", "G1_raw")
        if benchmark not in config.get("benchmarks", {}):
            parser.error(
                f"experiment.benchmark={benchmark!r} has no matching block under "
                "'benchmarks'."
            )
        _print_am_check(am_separation_from_config(config, benchmark=benchmark))
        return 0

    if args.scatter_preview:
        config = load_config(args.config)
        benchmark = config.get("experiment", {}).get("benchmark", "G1_raw")
        if benchmark not in config.get("benchmarks", {}):
            parser.error(
                f"experiment.benchmark={benchmark!r} has no matching block under "
                "'benchmarks'."
            )
        scatter_preview(config, benchmark=benchmark)
        return 0

    if args.recover:
        config = load_config(args.config)
        benchmark = config.get("experiment", {}).get("benchmark", "G1_raw")
        if benchmark not in config.get("benchmarks", {}):
            parser.error(
                f"experiment.benchmark={benchmark!r} has no matching block under "
                "'benchmarks'."
            )
        # Local import: eval_v2 pulls torch/kymatio via the scattering adapter.
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
            sweep_render_knobs,
        )
        sweep_render_knobs(
            config, benchmark=benchmark, out_dir=_results_dir(config, benchmark)
        )
        return 0

    if args.stage is not None:
        config = load_config(args.config)
        benchmark = config.get("experiment", {}).get("benchmark", "G1_raw")
        if benchmark not in config.get("benchmarks", {}):
            parser.error(
                f"experiment.benchmark={benchmark!r} has no matching block under "
                "'benchmarks'."
            )
        if args.pilot and args.full:
            parser.error("--pilot and --full are mutually exclusive.")
        spec = _STAGE_REGISTRY.get(args.stage)
        if spec is None or spec.run is None:
            parser.error(f"stage {args.stage!r} is registered but has no runner.")
            return 2  # unreachable; parser.error exits
        # Resolve the arm ONLY for model-dependent stages, then deep-merge its delta over
        # the base model / loss blocks. The model-free stages (build / r0_realizability /
        # data_previews) run once at the arm-less ``results/<tag>/`` root and never require
        # --arm, even under a multi-arm config.
        arm = _select_arm(config, args.arm) if spec.model_dependent else None
        config = resolve_arm(config, arm)
        ctx = StageContext(
            config=config, benchmark=benchmark, arm=arm, ckpt=args.ckpt,
            split=args.split, analysis_samples=args.analysis_samples,
            max_samples=getattr(args, "max_samples", None),
            pilot=bool(args.pilot), full=bool(args.full), args=args,
        )
        return _dispatch_stage(spec, ctx)

    parser.print_help()
    return 0


def _dispatch_stage(spec: StageSpec, ctx: StageContext) -> int:
    r"""Run one stage, honouring its ``fatal`` policy.

    A ``fatal=False`` stage that raises is reported as ``failed (non-fatal)`` and returns 0,
    so a diverging CMI fit or a broken calibration figure cannot abort a headline run. This is
    the same warn-don't-gate convention ``_test_plots_splits`` already applies per split.

    Args:
        spec: The registered stage.
        ctx: The resolved stage context.

    Returns:
        The stage's exit code (``0`` for a swallowed non-fatal failure).
    """
    assert spec.run is not None
    if spec.fatal:
        return spec.run(ctx)
    try:
        return spec.run(ctx)
    except Exception as exc:  # noqa: BLE001 -- opt-in analyses never gate the run
        print(f"[{spec.name}] failed (non-fatal): {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 0


# -----------------------------------------------------------------------------
# Built-in stage runners (registered at import; ``main()`` never names them)
# -----------------------------------------------------------------------------
def _stage_r0_realizability(ctx: StageContext) -> int:
    r"""Three-TE model-free de-risk pre-flight -> ``results/<tag>/realizability.json``."""
    # Default to the pilot grid: the full mix grid generates tens of thousands of scattering
    # passes and is opt-in via --full only.
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
        run_realizability_preflight,
    )
    run_realizability_preflight(
        ctx.config, benchmark=ctx.benchmark, pilot=not ctx.full, out_dir=ctx.tag_root(),
    )
    return 0


_CACHE_ARTIFACTS = ("train.npz", "val.npz", "test.npz", "meta.json", "norm_stats.npz")


def cache_is_complete(cache_dir: Path) -> bool:
    r"""Whether ``cache_dir`` holds a complete, loadable split cache.

    Args:
        cache_dir: The resolved cache leaf (see :func:`resolve_cache_dir`).

    Returns:
        ``True`` when every artifact in :data:`_CACHE_ARTIFACTS` is present.
    """
    return all((cache_dir / name).is_file() for name in _CACHE_ARTIFACTS)


def _print_cache_contents(out_dir: Path) -> None:
    r"""Print the split / meta / stats paths under a built cache leaf."""
    for split in ("train", "val", "test"):
        npz = out_dir / f"{split}.npz"
        if npz.is_file():
            print(f"  {split:5s} -> {npz}")
    print(f"  meta   -> {out_dir / 'meta.json'}")
    print(f"  stats  -> {out_dir / 'norm_stats.npz'}")


def _stage_build(ctx: StageContext) -> int:
    r"""Enumerate -> generate -> scatter -> normalise -> cache.

    Refuses to touch a complete cache without ``--force-rebuild``. ``build_all``
    unconditionally re-fits the normaliser and rewrites every split ``.npz``, and the
    Stage-1 parts are keyed on ``(split, cell_id)`` alone, so a default (``pilot``)
    build against a populated production ``data_tag`` would silently overwrite it with
    the small pilot grid -- a different cell count *and* different lags.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        build_all,
    )
    # Print the resolved cache leaf BEFORE any generation, so a stale or typo'd
    # ``experiment.data_tag`` (which would silently rebuild ~12k samples into the wrong
    # directory) is visible up front rather than after the work.
    cache_dir = resolve_cache_dir(ctx.config, benchmark=ctx.benchmark)
    print(f"[build] cache dir -> {cache_dir}")
    force = bool(getattr(ctx.args, "force_rebuild", False))
    if cache_is_complete(cache_dir) and not force:
        print(f"[build] cache up to date -> {cache_dir}")
        print("[build] nothing to do (pass --force-rebuild to regenerate in place).")
        _print_cache_contents(cache_dir)
        return 0
    grid = "full locked mix grid" if ctx.full else "PILOT grid (pass --full for the locked mix grid)"
    if cache_is_complete(cache_dir):
        print(f"[build] WARNING: --force-rebuild will OVERWRITE the complete cache at {cache_dir}")
    print(f"[build] grid -> {grid}")
    out_dir = build_all(ctx.config, benchmark=ctx.benchmark, pilot=not ctx.full)
    print(f"[build] wrote cache -> {out_dir}")
    _print_cache_contents(out_dir)
    return 0


def _stage_data_previews(ctx: StageContext) -> int:
    r"""Render the data-domain gallery (raw + scattering + latent) into ``<tag>/figures/``."""
    data_previews(ctx.config, benchmark=ctx.benchmark)
    return 0


def _stage_train(ctx: StageContext) -> int:
    r"""Fit the model on the cached splits -> checkpoint + loss curves."""
    # pl_module_v2 pulls torch / lightning, so import it lazily here.
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (
        train_v2,
    )
    assert ctx.args is not None, "the train stage needs the CLI namespace for _train_overrides"
    overrides = _train_overrides(ctx.config, ctx.args)
    overrides["arm"] = ctx.arm
    _print_train_result(train_v2(ctx.config, overrides, benchmark=ctx.benchmark))
    return 0


def _stage_beta_select(ctx: StageContext) -> int:
    r"""Pick the least-collapsed KL weight over ``beta_select.beta_grid``."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (
        beta_select,
    )
    assert ctx.args is not None, "the beta_select stage needs the CLI namespace"
    overrides = _train_overrides(ctx.config, ctx.args)
    # Explicitly invoking the stage force-runs it even when disabled in config.
    overrides["force"] = True
    overrides["arm"] = ctx.arm
    result = beta_select(ctx.config, overrides, benchmark=ctx.benchmark)
    print(f"[beta_select] selected beta = {result['selected_beta']:g}")
    for row in result["results"]:
        print(f"  beta={row['beta']:.3e}  kld_nats={row['kld_nats']:.4f}  "
              f"total_loss={row['total_loss']:.4f}")
    if result.get("out_path"):
        print(f"  wrote {result['out_path']}")
    return 0


def _stage_eval(ctx: StageContext) -> int:
    r"""Grade a trained checkpoint -> per-split ``metrics.json`` + ``per_sample_eval.npz``."""
    results_dir = ctx.run_dir()
    _eval_splits(ctx.config, ctx.benchmark, ckpt=ctx.ckpt, splits=ctx.splits(),
                 results_dir=results_dir, arm=ctx.arm)
    print(f"  wrote per-split metrics under {results_dir / '<split>'}")
    return 0


def _stage_test_plots(ctx: StageContext) -> int:
    r"""Bridge each cache split through the standard-testing per-sample diagnostics."""
    _test_plots_splits(ctx.config, ctx.benchmark, ckpt=ctx.ckpt, arm=ctx.arm,
                       analysis_samples=ctx.analysis_samples, splits=ctx.splits(),
                       results_dir=ctx.run_dir())
    return 0


def _stage_report(ctx: StageContext) -> int:
    r"""Assemble a markdown report + figure gallery per split, plus a cross-split index."""
    results_dir = ctx.run_dir()
    splits = ctx.splits()
    if not any((results_dir / s / "metrics.json").is_file() for s in splits):
        print("[report] WARNING: no per-split metrics.json found; the reports' calibration / "
              "lag / control gates will read 'n/a'. Run --stage eval first.", file=sys.stderr)
    _report_splits(ctx.config, ctx.benchmark, splits=splits, results_dir=results_dir)
    print(f"[report] wrote per-split reports + cross-split index under {results_dir}")
    return 0


#: The built-in stages, registered at import so ``stage_names()`` / ``stage_order()`` are
#: populated before ``build_parser`` or ``run_pipeline`` read them.
_BUILTIN_STAGE_SPECS: Tuple[StageSpec, ...] = (
    # order   name                default_on  model_dependent  cli
    StageSpec("solve_te", 0, False, False, None, cli=False,
              help="quick coupling query (dict driver only; CLI uses --solve-te)"),
    StageSpec("am_check", 1, False, False, None, cli=False,
              help="AM envelope-vs-wavelet separation pre-check (dict driver only)"),
    StageSpec("recover", 2, False, False, None, cli=False,
              help="frac_Phi render-knob sweep (dict driver only)"),
    StageSpec("r0_realizability", 3, True, False, _stage_r0_realizability,
              help="three-TE model-free de-risk pre-flight"),
    StageSpec("build", 4, True, False, _stage_build,
              help="generate -> scatter -> normalise -> cache"),
    StageSpec("data_previews", 5, True, False, _stage_data_previews,
              help="raw + scattering + latent figure gallery"),
    StageSpec("scatter_preview", 6, False, False, None, cli=False,
              help="scattering heatmap (dict driver only; CLI uses --scatter-preview)"),
    StageSpec("beta_select", 7, False, True, _stage_beta_select,
              help="KL-weight sweep (opt-in)"),
    StageSpec("train", 8, True, True, _stage_train,
              help="fit the model -> checkpoint + loss curves"),
    StageSpec("eval", 9, True, True, _stage_eval,
              help="grade the checkpoint -> metrics.json"),
    StageSpec("test_plots", 10, True, True, _stage_test_plots,
              help="standard-testing per-sample TE-annotated diagnostics"),
    StageSpec("report", 11, True, True, _stage_report,
              help="markdown report + figure gallery"),
)

for _builtin_spec in _BUILTIN_STAGE_SPECS:
    register_stage(_builtin_spec)


if __name__ == "__main__":
    # ----- edit-and-run configuration (no CLI; edit and run the file) --------
    # Runs with NO command-line args -> the PIPELINE dict below drives every stage
    # in _STAGE_ORDER (mirrors synthetic/run_mixed_pipeline). Passing ANY argument
    # (e.g. `--stage train`, `--solve-te 2.0 8`) instead dispatches to the argparse
    # CLI in main() -- that same CLI is what the DDP-safe train/beta_select
    # subprocesses re-enter, so it is kept alongside the dict driver.
    # HEADLINE synthetic_v3 three-arm run (S8-T03). To go back to the v1/v2 path, set
    # ``config_path`` to ``_DEFAULT_CONFIG`` and ``arms`` to ``None``.
    PIPELINE: Dict[str, Any] = {
        # --- identifiers ------------------------------------------------------
        "config_path": _DEFAULT_CONFIG_V3,  # config_synth_v3.yaml (the ablation ladder)
        "benchmark": None,                # None -> experiment.benchmark (G1_raw)
        # --- grid / build behaviour -------------------------------------------
        "pilot": False,                   # r0_realizability + build grid: True=pilot,
                                          #   False=full locked mix grid
        "force_rebuild": False,           # True -> regenerate cached build parts
                                          #   (resume=False). Leave False: `build` refuses
                                          #   to overwrite a complete cache without it.
        # --- training knobs ----------------------------------------------------
        "train_pilot": False,             # True -> short training smoke (--pilot)
        "force_retrain": False,           # True -> retrain even when final.ckpt exists
                                          #   (False makes the whole run resumable per arm)
        "devices": None,                  # None -> ddp.devices (headline, =8) /
                                          #   train.devices (--pilot, =1). An int, or
                                          #   "0,1,2,3"; >1 selects DDP (subprocessed).
        "epochs": None,                   # None -> optim.epochs (=100)
        "arms": ["v3_prod", "parity", "v3_noncausal"],
                                          # v3_prod first: it is the headline claim, so a
                                          #   run that dies overnight still yields it.
                                          #   None -> every arm in the config's 'arms' block.
        # --- eval / test_plots knobs ------------------------------------------
        "ckpt": None,                     # None -> best/final under results/<tag>/<arm>/
        "split": None,                    # eval / test_plots / report split; None -> ALL
                                          #   cached splits (train, val, test), each into
                                          #   its own results/<tag>/<arm>/<split>/ subfolder
        "analysis_samples": 4,            # TE-annotated per-sample PDFs in test_plots
        "max_samples": None,              # None -> each stage's own config cap
                                          #   (cmi.max_samples=512, calibration=2000)
        # --- diagnostic-stage settings ----------------------------------------
        "solve_te_args": None,            # (target_te, D) required iff stages.solve_te
        "scatter_preview": {"target_te": 2.0, "delay": 8, "n": 16},
        "data_previews": {"target_te": 2.0, "delay": 8, "n": 16, "include_null": True},
        # --- behaviour ---------------------------------------------------------
        "dry_run": False,                 # print the plan (incl. subprocess cmds) only
        # --- stage toggles (executed in _STAGE_ORDER) -------------------------
        # Model-free stages run once; model-dependent stages run once per arm, and the
        # analysis plugins run before `report` so its sections are populated.
        "stages": {
            "solve_te": False,            # quick coupling query -- needs solve_te_args
            "am_check": False,            # AM-separation pre-check (diagnostic)
            "recover": False,             # frac_Phi render-knob sweep (opt-in)
            "r0_realizability": False,    # OFF: at pilot=False this regenerates the full
                                          #   12k grid through the scattering transform
                                          #   (~1 h) to re-answer a question S1-T05 already
                                          #   settled (frac_phi is 2.4-5.9x; eval.
                                          #   realizability.fatal is false). Flip on for a
                                          #   fresh benchmark.
            "build": True,                # no-op cache-hit check when data_tag exists
            "data_previews": True,        # raw + scattering + latent gallery (cheap, once)
            "scatter_preview": False,     # scattering heatmap (diagnostic; superseded)
            "beta_select": False,         # KL-weight sweep (opt-in; beta is set in config)
            "train": True,                # fit -> checkpoint (DDP-safe subprocess, per arm)
            "eval": True,                 # grade -> metrics.json (gamma, controls, lag)
            "calibration": True,          # G-E: NLL / CRPS / coverage, stratified by TE
            "lag_intervention": True,     # G-F: leave-one-lag-band-out delta_L (opt-in)
            "cmi": True,                  # G-G: neural CMI in absolute nats (opt-in)
            "test_plots": True,           # standard testing per-sample diagnostics
            "report": True,               # markdown report + figures, per arm per split
            "arms_report": True,          # THE cross-arm table (once, after the sweep)
        },
    }

    if len(sys.argv) > 1:
        # Any CLI argument -> argparse mode (also the entry the train / beta_select
        # subprocesses and the documented one-off hooks re-enter).
        raise SystemExit(main())
    run_pipeline(PIPELINE)
