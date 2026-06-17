r"""``run_mixed_pipeline`` -- one-file, end-to-end driver for the ``G1_mix``
mixed-population experiment.

Runs every stage of the v3 mixed-population validation
(``model_validation_v3_mixed.md``) **in the correct order** from a single
edit-and-run configuration -- no ``argparse``, no ``sys.argv``. Edit the
``PIPELINE`` dict in ``__main__`` and run the file::

    python -m model.vae_teb_prediction.model.model_experiment.synthetic.run_mixed_pipeline
    # or directly (the repo root is bootstrapped onto sys.path):
    python run_mixed_pipeline.py

Stage order (each individually skippable via ``PIPELINE['stages']``):

    1. ``build_in_mix``     -- the in-mix pool ``data/G1_mix/<tag>/``
                               (:func:`mixed_dataset.build_g1_mix`).
    2. ``build_holdout``    -- the interior held-out test-only cache
                               ``data/G1_mix/<tag>_holdout/``.
    3. ``build_extrap``     -- the $M$-extrapolation test-only caches
                               ``data/G1_mix/<tag>_extrap_m<M>/`` for every
                               ``mix.holdout_m`` value (or ``PIPELINE['extrap_m']``).
    4. ``data_previews``    -- journal-quality data-anatomy figures for the
                               in-mix cache (:mod:`visualize_mixed`): per-channel
                               source/target panels with the true lag walk,
                               colour-matched lag sections, the TE $\times$
                               lag-band gallery and the channel atlas, under
                               ``data/G1_mix/<tag>/previews/``. Non-fatal.
    5. ``beta_calibration`` -- OPTIONAL (heavy: one pooled training per
                               $\beta$). Runs :mod:`mixed_calibration` in a
                               subprocess, then reads the selected
                               $\beta^\star = \arg\min_\beta
                               \operatorname{mean}_M|\gamma_M - 1|
                               + \lambda_\alpha \operatorname{mean}_M|\alpha_M|$
                               from ``results/G1_mix/mixed_calibration/calibration.json``.
    6. ``train``            -- the final pooled model via :mod:`train_ddp` in a
                               subprocess (so Lightning's DDP launcher re-executes
                               the *training* module, never this pipeline).
                               ``PIPELINE['resume_ckpt']`` optionally continues
                               training from a previous run's synthetic-format
                               checkpoint (``results/G1_mix/<run_tag>/final.ckpt``
                               or ``best.ckpt``; weights only, fresh optimizer);
                               ``None`` trains from scratch.
    7. ``eval_in_mix``      -- :func:`mixed_eval.evaluate_mixed` on the in-mix +
                               interior-holdout caches
                               (``results/G1_mix/<run_tag>/mixed_eval/``).
    8. ``eval_extrap``      -- one :func:`mixed_eval.evaluate_mixed` pass per
                               extrapolation cache, each writing to its own
                               ``mixed_eval_extrap_m<M>/`` subdirectory (each
                               pass also writes its ``per_sample.csv`` with the
                               extrapolation rows under ``split=holdout``,
                               consumed by ``combined_figures``).
    9. ``combined_figures`` -- the combined per-sample scatter suite pooled
                               over every eval pass (all $M$ colours in one
                               figure), re-rendered purely from the CSV/JSON
                               artifacts (:func:`mixed_eval.render_combined_per_sample_scatter`)
                               into ``results/G1_mix/<run_tag>/combined_figures/``.
                               Cheap and non-fatal.
   10. ``per_cell_diagnostics`` -- OPTIONAL (heavy: a forward / LOLO pass per
                               cell). The faithful single-cell ``lag_recovery`` /
                               ``evaluate_te`` analyses run once per sub-population
                               (:func:`mixed_per_cell_diag.run_mixed_per_cell_diag`),
                               writing per-cell figures + a cross-cell rollup under
                               ``results/G1_mix/<run_tag>/per_cell/<cache>/``.
   10. ``pipeline_tests``   -- the broad model-diagnostic pipeline
                               (:mod:`run_pipeline_tests` $\to$
                               ``testing.run_tests.run_full_test_pipeline``:
                               histograms, forecast quality, attention / lag
                               diagnostics, KL-PCA, residual usage, ...) on the
                               **same checkpoint** stages 6-7 evaluate -- i.e.
                               the $\beta^\star$-trained ``<ckpt_name>`` --
                               against the in-mix test split. Outputs land in
                               ``results/G1_mix/<run_tag>/testing_pipeline/<output_tag>/``.

Why subprocesses for stages 5-6: Lightning's multi-GPU DDP launcher re-executes
``sys.executable + sys.argv`` in every worker rank. If training ran in-process,
each rank would re-run this *entire pipeline* (builds, evals, ...). Driving the
training CLI in a child process keeps the re-executed module scoped to
``train_ddp`` / ``mixed_calibration``, which are designed for it. Builds and
evaluations are single-process and run in-process.

$\beta$ resolution for the final training (``PIPELINE['beta']``):

    * ``None``   -- use ``loss.kld_beta`` from the config, *unless* the
      ``beta_calibration`` stage ran in this invocation, in which case the
      freshly selected $\beta^\star$ is used (and printed).
    * ``float``  -- explicit override (e.g. a previously selected $\beta^\star$).
    * ``"auto"`` -- read $\beta^\star$ from an existing
      ``results/G1_mix/mixed_calibration/calibration.json`` (errors loudly if
      the calibration has not been run yet).

Idempotence: dataset builds skip complete caches (``force_rebuild`` overrides),
and training skips when ``results/G1_mix/<run_tag>/<ckpt_name>`` already exists
(``force_retrain`` overrides), so the file can be re-run after an interruption
and it continues from the first missing artifact.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- repo-root bootstrap so `python run_mixed_pipeline.py` works directly ----
# (running via `python -m ...synthetic.run_mixed_pipeline` already has the
# repo root on sys.path; this is a no-op in that case).
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.vae_teb_prediction.model.model_experiment.synthetic.mixed_dataset import (  # noqa: E402
    _HOLDOUT_SUFFIX,
    _mix_block,
    build_g1_mix,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.mixed_eval import (  # noqa: E402
    evaluate_mixed,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (  # noqa: E402
    apply_path_overrides,
    load_config,
    resolve_active_benchmark,
    resolve_user_path,
)

_MODULE_BASE = "model.vae_teb_prediction.model.model_experiment.synthetic"
_BENCHMARK = "G1_mix"
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Canonical stage order. ``run_pipeline`` validates the PIPELINE['stages'] keys
# against this tuple so a typo fails loudly instead of silently skipping.
_STAGE_ORDER = (
    "build_in_mix",
    "build_holdout",
    "build_extrap",
    "data_previews",
    "beta_calibration",
    "train",
    "eval_in_mix",
    "eval_extrap",
    "combined_figures",
    "per_cell_diagnostics",
    "pipeline_tests",
)

# Defaults applied when PIPELINE['stages'] omits a key. Everything runs except
# the two heavy opt-in stages -- the beta sweep (one pooled training per beta)
# and the per-cell diagnostics (a forward / LOLO pass per cell across every
# cache) are far too expensive to start by accident; they must be opted into.
_STAGE_DEFAULTS = {
    name: name not in ("beta_calibration", "per_cell_diagnostics")
    for name in _STAGE_ORDER
}


# =============================================================================
# Small helpers
# =============================================================================

def _banner(step: int, total: int, name: str, note: str = "") -> None:
    """Print a uniform stage banner.

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
    """Run a child process from the repo root, streaming its output.

    Args:
        cmd: The full command (``sys.executable -m <module> ...``).
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
            f"pipeline subprocess failed (exit {proc.returncode}): "
            f"{' '.join(cmd)}"
        )


def _read_beta_star(results_root: Path) -> float:
    r"""Read the selected $\beta^\star$ from the mixed-calibration artifacts.

    Args:
        results_root: Resolved ``paths.results_dir``.

    Returns:
        The selected $\beta^\star$.

    Raises:
        FileNotFoundError: If ``calibration.json`` does not exist yet.
        ValueError: If the file carries no finite ``beta_star``.
    """
    path = results_root / _BENCHMARK / "mixed_calibration" / "calibration.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"beta='auto' but no calibration artifact at {path}. Enable the "
            f"'beta_calibration' stage (or run mixed_calibration) first, or "
            f"set PIPELINE['beta'] to an explicit float."
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    beta_star = float((payload.get("selected") or {}).get("beta_star", float("nan")))
    if not (beta_star == beta_star):  # NaN check
        raise ValueError(f"calibration artifact {path} carries no finite beta_star.")
    return beta_star


def _resolve_extrap_ms(pipeline: Dict[str, Any], config: Dict[str, Any]) -> List[int]:
    """Resolve the extrapolation-$M$ list (pipeline override or ``mix.holdout_m``).

    Args:
        pipeline: The PIPELINE dict.
        config: The resolved config.

    Returns:
        The list of untrained $M$ values to build / evaluate (possibly empty).
    """
    override = pipeline.get("extrap_m")
    if override is not None:
        return [int(m) for m in override]
    return [int(m) for m in _mix_block(config).get("holdout_m", [])]


def _calibration_cmd(
    pipeline: Dict[str, Any], config_path: Path
) -> List[str]:
    """Assemble the ``mixed_calibration`` subprocess command.

    The pools are built by stages 1-2, so ``--no-build`` is always passed.

    Args:
        pipeline: The PIPELINE dict (reads the ``calibration`` sub-block).
        config_path: Path to the YAML config handed to the child.

    Returns:
        The command list.
    """
    cal = dict(pipeline.get("calibration") or {})
    mode = str(cal.get("mode", "task_parallel"))
    cmd = [
        sys.executable, "-m", f"{_MODULE_BASE}.mixed_calibration",
        "--config", str(config_path), "--no-build", "--mode", mode,
    ]
    if mode == "task_parallel":
        gpus = cal.get("gpus") or [0]
        cmd += ["--gpus", ",".join(str(int(g)) for g in gpus)]
    else:  # ddp fallback ('devices': None in the sub-block falls back too)
        devices = cal.get("devices") or pipeline.get("devices", 1)
        cmd += ["--devices", str(devices)]
    if cal.get("betas"):
        cmd += ["--betas", ",".join(f"{float(b):g}" for b in cal["betas"])]
    if cal.get("epochs") is not None:
        cmd += ["--epochs", str(int(cal["epochs"]))]
    for flag, key in (("--data-dir", "data_dir"), ("--results-dir", "results_dir")):
        if pipeline.get(key):
            cmd += [flag, str(pipeline[key])]
    return cmd


def _train_cmd(
    pipeline: Dict[str, Any],
    config_path: Path,
    beta: Optional[float],
    *,
    tag: str,
    run_tag: str,
) -> List[str]:
    """Assemble the ``train_ddp`` subprocess command.

    Args:
        pipeline: The PIPELINE dict.
        config_path: Path to the YAML config handed to the child.
        beta: Resolved KL weight for this run; ``None`` keeps the config value.
        tag: Resolved in-mix cache tag.
        run_tag: Resolved results-directory tag (defaulted to ``tag`` by the
            caller, so an omitted ``PIPELINE['run_tag']`` is fine).

    Returns:
        The command list.
    """
    cmd = [
        sys.executable, "-m", f"{_MODULE_BASE}.train_ddp",
        "--config", str(config_path),
        "--tag", tag,
        "--run-tag", run_tag,
        "--devices", str(pipeline.get("devices", 1)),
    ]
    if beta is not None:
        cmd += ["--beta", f"{float(beta):g}"]
    for flag, key in (
        ("--epochs", "epochs"),
        ("--batch-size", "batch_size"),
        ("--lr", "lr"),
        ("--seed", "seed"),
        ("--early-stop-patience", "early_stop_patience"),
        ("--resume-ckpt", "resume_ckpt"),
        ("--data-dir", "data_dir"),
        ("--results-dir", "results_dir"),
    ):
        if pipeline.get(key) is not None:
            cmd += [flag, str(pipeline[key])]
    return cmd


# =============================================================================
# Pipeline driver
# =============================================================================

def run_pipeline(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    r"""Run the full ``G1_mix`` pipeline in order, honouring the stage toggles.

    Args:
        pipeline: The edit-and-run configuration (see the ``PIPELINE`` dict in
            ``__main__`` for every key and its semantics).

    Returns:
        ``{stage: status}`` plus ``beta_used`` and the key artifact paths.
        ``status`` is one of ``done`` / ``skipped (disabled)`` /
        ``skipped (exists)`` / ``dry-run``.

    Raises:
        ValueError: On an unknown stage key (fails loud on typos).
        RuntimeError: When a subprocess stage exits non-zero.
    """
    t_start = time.time()
    stages = dict(_STAGE_DEFAULTS)
    stages.update(pipeline.get("stages") or {})
    unknown = set(stages) - set(_STAGE_ORDER)
    if unknown:
        raise ValueError(
            f"unknown stage keys {sorted(unknown)}; valid stages are "
            f"{list(_STAGE_ORDER)}."
        )
    dry_run = bool(pipeline.get("dry_run", False))
    force_rebuild = bool(pipeline.get("force_rebuild", False))
    force_retrain = bool(pipeline.get("force_retrain", False))
    ckpt_name = str(pipeline.get("ckpt_name", "final.ckpt"))
    tag = str(pipeline["tag"])
    run_tag = str(pipeline.get("run_tag") or tag)
    config_path = Path(pipeline.get("config_path", _DEFAULT_CONFIG))

    # Resolved config for the in-process stages (builds + evals). The training
    # and calibration subprocesses re-load the same YAML themselves.
    config = load_config(config_path)
    config["experiment"]["benchmark"] = _BENCHMARK
    apply_path_overrides(config, {
        "data_dir": pipeline.get("data_dir"),
        "results_dir": pipeline.get("results_dir"),
    })
    config = resolve_active_benchmark(config)
    config["experiment"]["tag"] = tag

    data_root = resolve_user_path(config["paths"]["data_dir"])
    results_root = resolve_user_path(config["paths"]["results_dir"])
    extrap_ms = _resolve_extrap_ms(pipeline, config)
    holdout_tag = tag + _HOLDOUT_SUFFIX
    n_stages = len(_STAGE_ORDER)
    status: Dict[str, Any] = {}

    def _stage_banner(name: str, note: str = "") -> None:
        """Banner with the stage index derived from :data:`_STAGE_ORDER`."""
        _banner(_STAGE_ORDER.index(name) + 1, n_stages, name, note)

    print(
        f"[pipeline] G1_mix end-to-end run\n"
        f"           config      = {config_path}\n"
        f"           in-mix tag  = '{tag}'   run tag = '{run_tag}'\n"
        f"           data root   = {data_root}\n"
        f"           results     = {results_root}\n"
        f"           extrap M    = {extrap_ms}\n"
        f"           dry_run     = {dry_run}"
    )

    # --- 1. in-mix pool -------------------------------------------------------
    _stage_banner("build_in_mix")
    if not stages["build_in_mix"]:
        status["build_in_mix"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["build_in_mix"] = "dry-run"
        print(f"[pipeline] would build data/{_BENCHMARK}/{tag}/")
    else:
        build_g1_mix(config, force=force_rebuild)
        status["build_in_mix"] = "done"

    # --- 2. interior held-out cache -------------------------------------------
    _stage_banner("build_holdout")
    if not stages["build_holdout"]:
        status["build_holdout"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["build_holdout"] = "dry-run"
        print(f"[pipeline] would build data/{_BENCHMARK}/{holdout_tag}/")
    else:
        build_g1_mix(config, force=force_rebuild, holdout=True)
        status["build_holdout"] = "done"

    # --- 3. M-extrapolation caches ---------------------------------------------
    _stage_banner("build_extrap", note=f"M in {extrap_ms}" if extrap_ms else "none configured")
    if not stages["build_extrap"] or not extrap_ms:
        status["build_extrap"] = "skipped (disabled)" if not stages["build_extrap"] else "skipped (no holdout_m)"
        print("[pipeline] disabled or no extrapolation M configured.")
    elif dry_run:
        status["build_extrap"] = "dry-run"
        for m in extrap_ms:
            print(f"[pipeline] would build data/{_BENCHMARK}/{tag}_extrap_m{m}/")
    else:
        for m in extrap_ms:
            build_g1_mix(config, force=force_rebuild, extrap_m=int(m))
        status["build_extrap"] = "done"

    # --- 4. data-anatomy previews (visualize_mixed) -----------------------------
    in_mix_cache = data_root / _BENCHMARK / tag
    _stage_banner("data_previews")
    if not stages["data_previews"]:
        status["data_previews"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["data_previews"] = "dry-run"
        print(f"[pipeline] would render data-anatomy previews -> "
              f"{in_mix_cache / 'previews'}")
    else:
        # Lazy import (pulls matplotlib); a preview failure must never gate
        # the training / evaluation stages.
        try:
            from model.vae_teb_prediction.model.model_experiment.synthetic.visualize_mixed import (
                render_mixed_previews,
            )
            render_mixed_previews(in_mix_cache)
            status["data_previews"] = "done"
        except Exception as exc:  # noqa: BLE001 -- diagnostics only
            print(f"[pipeline][warn] data previews failed: "
                  f"{type(exc).__name__}: {exc}")
            status["data_previews"] = "failed (non-fatal)"

    # --- 5. beta calibration sweep (optional, heavy) ---------------------------
    beta = pipeline.get("beta")
    _stage_banner("beta_calibration")
    if stages["beta_calibration"]:
        _run_subprocess(_calibration_cmd(pipeline, config_path), dry_run=dry_run)
        status["beta_calibration"] = "dry-run" if dry_run else "done"
        if not dry_run and beta is None:
            beta = _read_beta_star(results_root)
            print(f"[pipeline] using freshly selected beta* = {beta:.3e} "
                  f"for the final training run.")
    else:
        status["beta_calibration"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    if beta == "auto":
        try:
            beta = _read_beta_star(results_root)
            print(f"[pipeline] beta='auto' -> beta* = {beta:.3e} from "
                  f"mixed_calibration/calibration.json")
        except (FileNotFoundError, ValueError):
            if not dry_run:
                raise
            # A dry run must never abort: report the unresolved beta and
            # fall back to the config default for the printed plan.
            beta = None
            print("[pipeline] beta='auto' but no calibration artifact yet -- "
                  "a real run would fail here unless the beta_calibration "
                  "stage runs first.")
    if beta is None and not stages["beta_calibration"]:
        print(
            f"[pipeline][note] training at the config default "
            f"loss.kld_beta={config['loss'].get('kld_beta')}. The v3 protocol "
            f"recommends a *selected* beta (enable the 'beta_calibration' "
            f"stage or set PIPELINE['beta']) -- an unselected small beta "
            f"leaves the KL nearly free, so K-bar will overshoot the TE scale "
            f"(gamma >> 1)."
        )
    status["beta_used"] = (
        float(beta) if beta is not None else float(config["loss"].get("kld_beta"))
    )

    # --- 6. final pooled training (DDP-safe subprocess) ------------------------
    resume_ckpt = pipeline.get("resume_ckpt")
    _stage_banner(
        "train",
        note=f"devices={pipeline.get('devices', 1)}"
             + (f", resume from {resume_ckpt}" if resume_ckpt else ""),
    )
    run_dir = results_root / _BENCHMARK / run_tag
    ckpt_path = run_dir / ckpt_name
    if not stages["train"]:
        status["train"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif ckpt_path.is_file() and not force_retrain:
        status["train"] = "skipped (exists)"
        print(f"[pipeline] checkpoint exists, skipping training: {ckpt_path}\n"
              f"           (set force_retrain=True to retrain)")
        if resume_ckpt:
            print("[pipeline][note] resume_ckpt is set but training was "
                  "skipped -- continue into a NEW run_tag or set "
                  "force_retrain=True.")
    else:
        # Fail before launching the subprocess so a typo in the path does not
        # cost a DDP spin-up (dry runs only print the would-be command).
        if resume_ckpt and not dry_run and not Path(resume_ckpt).is_file():
            raise FileNotFoundError(
                f"PIPELINE['resume_ckpt'] not found: {resume_ckpt}. Provide "
                f"the synthetic-format checkpoint of a previous run, e.g. "
                f"results/G1_mix/<previous_run_tag>/final.ckpt."
            )
        _run_subprocess(
            _train_cmd(
                pipeline, config_path,
                None if beta is None else float(beta),
                tag=tag, run_tag=run_tag,
            ),
            dry_run=dry_run,
        )
        status["train"] = "dry-run" if dry_run else "done"

    # --- 6. per-group eval: in-mix (+ interior holdout iff it was built) --------
    # Only pull the interior held-out cache into the eval when stage 2 actually
    # built it; with `build_holdout` disabled the in-mix `test.npz` is the sole
    # evaluation set (and `evaluate_mixed` skips its held-out block on `None`),
    # so no "held-out cache skipped" FileNotFoundError is emitted.
    eval_holdout_tag = holdout_tag if stages["build_holdout"] else None
    _stage_banner("eval_in_mix", note=f"ckpt={ckpt_name}")
    if not stages["eval_in_mix"]:
        status["eval_in_mix"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["eval_in_mix"] = "dry-run"
        ho_note = f" + '{eval_holdout_tag}'" if eval_holdout_tag else ""
        print(f"[pipeline] would evaluate run '{run_tag}' on '{tag}'{ho_note} "
              f"-> {run_dir / 'mixed_eval'}")
    else:
        evaluate_mixed(
            config, run_tag=run_tag, in_mix_tag=tag, holdout_tag=eval_holdout_tag,
            ckpt_name=ckpt_name,
        )
        status["eval_in_mix"] = "done"

    # --- 7. per-group eval: M-extrapolation caches ------------------------------
    _stage_banner("eval_extrap", note=f"M in {extrap_ms}" if extrap_ms else "none configured")
    if not stages["eval_extrap"] or not extrap_ms:
        status["eval_extrap"] = "skipped (disabled)" if not stages["eval_extrap"] else "skipped (no holdout_m)"
        print("[pipeline] disabled or no extrapolation M configured.")
    elif dry_run:
        status["eval_extrap"] = "dry-run"
        for m in extrap_ms:
            print(f"[pipeline] would evaluate '{tag}_extrap_m{m}' -> "
                  f"{run_dir / f'mixed_eval_extrap_m{m}'}")
    else:
        for m in extrap_ms:
            extrap_tag = f"{tag}_extrap_m{m}"
            # in_mix_light: the canonical in-mix diagnostics already live in
            # mixed_eval/ (stage 7); these passes only need the in-mix
            # calibration to score the extrapolation cells.
            evaluate_mixed(
                config, run_tag=run_tag, in_mix_tag=tag,
                holdout_tag=extrap_tag, ckpt_name=ckpt_name,
                out_subdir=f"mixed_eval_extrap_m{m}", in_mix_light=True,
            )
        status["eval_extrap"] = "done"

    # --- 9. combined per-sample figures across every eval pass ------------------
    _stage_banner("combined_figures")
    if not stages["combined_figures"]:
        status["combined_figures"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["combined_figures"] = "dry-run"
        print(f"[pipeline] would render the combined per-sample figures from "
              f"the eval CSVs -> {run_dir / 'combined_figures'}")
    else:
        # CSV/JSON -> matplotlib only (no checkpoint, no GPU); a failure must
        # never gate the remaining stages -- same policy as data_previews.
        try:
            from model.vae_teb_prediction.model.model_experiment.synthetic.mixed_eval import (
                render_combined_per_sample_scatter,
            )
            written = render_combined_per_sample_scatter(run_dir)
            status["combined_figures"] = "done" if written else "skipped (no CSVs)"
        except Exception as exc:  # noqa: BLE001 -- figures only
            print(f"[pipeline][warn] combined figures failed: "
                  f"{type(exc).__name__}: {exc}")
            status["combined_figures"] = "failed (non-fatal)"

    # --- 10. faithful per-cell single-cell diagnostics (opt-in, heavy) ----------
    pcd_raw = pipeline.get("per_cell_diagnostics")
    if pcd_raw is not None and not isinstance(pcd_raw, dict):
        # Same stage-name collision guard as `pipeline_tests`: the toggle lives
        # under 'stages'; a top-level bool would otherwise be coerced to {} and
        # the (expensive) stage would silently run with defaults.
        raise ValueError(
            "PIPELINE['per_cell_diagnostics'] must be a dict of stage settings; "
            "to disable the stage set "
            "PIPELINE['stages']['per_cell_diagnostics'] = False."
        )
    _stage_banner("per_cell_diagnostics", note=f"ckpt={ckpt_name}")
    if not stages["per_cell_diagnostics"]:
        status["per_cell_diagnostics"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["per_cell_diagnostics"] = "dry-run"
        print(f"[pipeline] would run per-cell lag_recovery / evaluate_te "
              f"diagnostics on {ckpt_path}\n           -> {run_dir / 'per_cell'}")
    else:
        # Lazy import (pulls matplotlib via the reused single-cell renderers).
        from model.vae_teb_prediction.model.model_experiment.synthetic.mixed_per_cell_diag import (
            run_mixed_per_cell_diag,
        )
        eval_blk = config["benchmarks"][_BENCHMARK].get("eval", {}) or {}
        pcd = dict(eval_blk.get("per_cell_diag", {}) or {})
        pcd.update({
            k: v for k, v in dict(pcd_raw or {}).items() if v is not None
        })
        extrap_tags = [f"{tag}_extrap_m{m}" for m in extrap_ms]
        run_mixed_per_cell_diag(
            config, run_tag=run_tag, in_mix_tag=tag, holdout_tag=holdout_tag,
            extrap_tags=extrap_tags, ckpt_name=ckpt_name,
            caches=tuple(pcd.get("caches", ["in_mix"])),
            run_lag_recovery=bool(pcd.get("run_lag_recovery", True)),
            run_eval_te=bool(pcd.get("run_eval_te", True)),
            run_width_sweep=bool(pcd.get("run_width_sweep", False)),
            width_grid=tuple(pcd.get("width_grid", [1, 5, 10, 20])),
            n_per_cell=pcd.get("n_per_cell"),
        )
        status["per_cell_diagnostics"] = "done"

    # --- 11. broad model-diagnostic testing pipeline -----------------------------
    pt_raw = pipeline.get("pipeline_tests")
    if pt_raw is not None and not isinstance(pt_raw, dict):
        # Guard the stage-name collision: the toggle lives under 'stages';
        # a top-level `"pipeline_tests": False` would otherwise be silently
        # coerced to {} and the (expensive) stage would still run.
        raise ValueError(
            "PIPELINE['pipeline_tests'] must be a dict of stage settings; to "
            "disable the stage set PIPELINE['stages']['pipeline_tests'] = False."
        )
    pt = dict(pt_raw or {})
    pt_output_tag = str(pt.get("output_tag") or Path(ckpt_name).stem)
    pt_out_dir = run_dir / "testing_pipeline" / pt_output_tag
    test_npz = data_root / _BENCHMARK / tag / "test.npz"
    _stage_banner("pipeline_tests", note=f"ckpt={ckpt_name}")
    if not stages["pipeline_tests"]:
        status["pipeline_tests"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["pipeline_tests"] = "dry-run"
        print(f"[pipeline] would run the testing pipeline on {ckpt_path}\n"
              f"           against {test_npz}\n"
              f"           -> {pt_out_dir}")
    else:
        # Lazy import: this pulls in the full testing/ package (heavy), and
        # the stage may well be skipped.
        from model.vae_teb_prediction.model.model_experiment.synthetic.run_pipeline_tests import (
            run_synthetic_pipeline_tests,
        )
        # Absolute checkpoint_path + data_npz so the pipeline-level
        # data_dir / results_dir overrides apply (run_pipeline_tests reads
        # paths.* from the raw YAML and would miss them otherwise). With
        # checkpoint_path set, outputs land at
        # <run_dir>/testing_pipeline/<output_tag>/ by its own convention.
        pt_analysis = pt.get("analysis_samples")
        run_synthetic_pipeline_tests(
            output_tag=pt_output_tag,
            checkpoint_path=ckpt_path,
            data_npz=test_npz,
            config_path=config_path,
            device=pt.get("device"),
            max_samples=pt.get("max_samples"),
            analysis_samples=10 if pt_analysis is None else int(pt_analysis),
            batch_size=pt.get("batch_size"),
            skip_up_effect=bool(pt.get("skip_up_effect", False)),
            skip_frequency_band=bool(pt.get("skip_frequency_band", False)),
            skip_attention=bool(pt.get("skip_attention", False)),
            skip_forecast_heatmaps=bool(pt.get("skip_forecast_heatmaps", False)),
            skip_kld_pca=bool(pt.get("skip_kld_pca", False)),
            skip_interactive=bool(pt.get("skip_interactive", False)),
        )
        status["pipeline_tests"] = "done"

    # --- summary ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"[pipeline] finished in {elapsed / 60.0:.1f} min")
    for name in _STAGE_ORDER:
        print(f"  {name:18s} {status.get(name, '?')}")
    print(f"  beta used          {status['beta_used']:g}")
    print("[pipeline] artifacts:")
    print(f"  caches      {data_root / _BENCHMARK}")
    if stages["data_previews"]:
        print(f"  previews    {in_mix_cache / 'previews'}")
    print(f"  run         {run_dir}")
    print(f"  eval        {run_dir / 'mixed_eval'}")
    for m in extrap_ms:
        print(f"  extrap eval {run_dir / f'mixed_eval_extrap_m{m}'}")
    if stages["combined_figures"]:
        print(f"  combined    {run_dir / 'combined_figures'}")
    if stages["per_cell_diagnostics"]:
        print(f"  per-cell    {run_dir / 'per_cell'}")
    if stages["pipeline_tests"]:
        print(f"  diagnostics {pt_out_dir}")
    if stages["beta_calibration"]:
        print(f"  calibration {results_root / _BENCHMARK / 'mixed_calibration'}")
    status["run_dir"] = str(run_dir)
    return status


if __name__ == "__main__":
    # ----- edit-and-run configuration (no CLI; edit and run the file) --------
    PIPELINE: Dict[str, Any] = {
        # --- identifiers ------------------------------------------------------
        "config_path": _DEFAULT_CONFIG,  # config_synth.yaml
        "tag": "G1_mix_base",        # in-mix cache tag: data/G1_mix/<tag>/
        "run_tag": "G1_mix_base",    # results dir: results/G1_mix/<run_tag>/
        # --- compute ------------------------------------------------------------
        "devices": 1,                # train_ddp devices: 8 on the Linux box,
                                     # 1 on the Windows dev laptop; also accepts
                                     # explicit indices, e.g. "0,1,2,3"
        # --- training knobs (None -> config value) -------------------------------
        "beta": None,                # None -> loss.kld_beta (or beta* when the
                                     #   beta_calibration stage runs);
                                     # float -> explicit KL weight;
                                     # "auto" -> read beta* from an existing
                                     #   results/G1_mix/mixed_calibration/calibration.json
        "epochs": None,              # None -> optim.epochs
        "batch_size": None,          # None -> optim.batch_size (PER-GPU)
        "lr": None,                  # None -> optim.lr
        "seed": None,                # None -> experiment.seed
        "early_stop_patience": None, # int -> enable EarlyStopping on val loss
        "resume_ckpt": None,         # None -> train from scratch.
                                     # Path  -> CONTINUE training from that
                                     #   checkpoint's weights. Provide the
                                     #   SYNTHETIC-format checkpoint written by
                                     #   train_ddp, i.e.
                                     #   results/G1_mix/<previous_run_tag>/final.ckpt
                                     #   (or best.ckpt) -- NOT a Lightning
                                     #   lightning_ckpts/lightning_best-*.ckpt.
                                     #   Weights only: the optimizer / LR
                                     #   schedule / epoch counter start fresh.
                                     #   Every training hyperparameter (lr,
                                     #   scheduled LR changes lr_milestones /
                                     #   lr_gamma, weight_decay, beta) is taken
                                     #   FRESH from config_synth.yaml at resume
                                     #   time -- edit its optim:/loss: blocks to
                                     #   override them; a milestone at epoch N
                                     #   then fires at epoch N of the resumed run.
                                     #   Training is still skipped when
                                     #   <run_dir>/<ckpt_name> already exists,
                                     #   so continue into a NEW run_tag or set
                                     #   force_retrain=True.
        "ckpt_name": "final.ckpt",   # checkpoint evaluated by stages 6-7
        # --- extrapolation axis ---------------------------------------------------
        "extrap_m": [],              # [] -> NO M-extrapolation caches (build_extrap /
                                     #   eval_extrap become no-ops, per-cell extrap_tags
                                     #   empty). We evaluate ONLY the in-mix test split.
                                     # None -> config mix.holdout_m (e.g. [4, 64]);
                                     # or an explicit list like [64]
        # --- paths (None -> config paths.*) ---------------------------------------
        "data_dir": None,
        "results_dir": None,
        # --- behaviour -------------------------------------------------------------
        "force_rebuild": False,      # rebuild caches even when complete
        "force_retrain": False,      # retrain even when <ckpt_name> exists
        "dry_run": False,            # print the plan (incl. subprocess cmds) only
        # --- stage toggles (always executed in this order) -------------------------
        "stages": {
            "build_in_mix": True,    # 1. data/G1_mix/<tag>/
            # NOTE: build_holdout / build_extrap / eval_extrap default to True in
            # _STAGE_DEFAULTS -- they must be set to explicit False here (NOT
            # commented out, which would fall back to the True default).
            "build_holdout": False,  # 2. DISABLED: no separate interior holdout
                                     #    cache; we use the in-mix test split only.
            "build_extrap": False,   # 3. DISABLED: no M-extrapolation caches
                                     #    (also zeroed by extrap_m=[]).
            "data_previews": True,   # 4. data-anatomy figures (visualize_mixed)
                                     #    -> data/G1_mix/<tag>/previews/
            "beta_calibration": False,  # 5. OPTIONAL beta sweep (one pooled
                                        #    training per beta -- expensive)
            "train": True,           # 6. final pooled model (train_ddp)
            "eval_in_mix": True,     # 7. mixed_eval on the in-mix test split
            "eval_extrap": False,    # 8. DISABLED: no extrapolation eval passes
            "combined_figures": True,  # 9. combined per-sample scatter suite
                                     #    pooled over every eval pass (CSV ->
                                     #    figures only; cheap, non-fatal)
            "per_cell_diagnostics": False,  # 10. OPTIONAL faithful per-cell
                                     #    lag_recovery / evaluate_te reproduction
                                     #    (forward / LOLO pass per cell -- heavy;
                                     #    the evaluate_te-style kbar_vs_te plots
                                     #    already render by default in stage 7-8)
            "pipeline_tests": True,  # 11. broad testing/ diagnostics on the
                                     #    same <ckpt_name> (run_pipeline_tests)
        },
        # --- beta_calibration sub-stage settings (stage 4 only) --------------------
        "calibration": {
            "mode": "task_parallel",  # "task_parallel" (one beta per GPU slot)
                                      # or "ddp" (sequential, all GPUs per beta)
            "gpus": [0],              # task-parallel GPU slot list
                                      # ([0,1,2,3,4,5,6,7] on the Linux box)
            "devices": None,          # ddp-mode device spec (None -> 'devices')
            "betas": None,            # None -> config mix_calibration.beta_grid
            "epochs": None,           # None -> optim.epochs
        },
        # --- per_cell_diagnostics sub-stage settings (stage 9 only) ----------------
        # Each None falls back to config benchmarks.G1_mix.eval.per_cell_diag.
        "per_cell_diagnostics": {
            "caches": ["in_mix"],     # in-mix only (holdout / extrap caches are not
                                      #   built); None -> config (e.g.
                                      #   [in_mix, holdout, extrap])
            "run_lag_recovery": None, # None -> config (analyze_lag_recovery per cell)
            "run_eval_te": None,      # None -> config (evaluate_checkpoint per cell)
            "run_width_sweep": None,  # None -> config (LOLO window-width sweep; heavy)
            "width_grid": None,       # None -> config [1, 5, 10, 20]
            "n_per_cell": None,       # None -> config (whole cell test split)
        },
        # --- pipeline_tests sub-stage settings (stage 10 only) ---------------------
        "pipeline_tests": {
            "output_tag": None,       # None -> ckpt_name stem ('final'); leaf
                                      # under <run_dir>/testing_pipeline/
            "max_samples": None,      # None -> whole in-mix test split
            "analysis_samples": 10,   # per-sample diagnostic PDFs
            "batch_size": None,       # None -> optim.batch_size
            "device": None,           # None -> auto (cuda if available)
            "skip_up_effect": False,
            "skip_frequency_band": False,
            "skip_attention": False,
            "skip_forecast_heatmaps": False,
            "skip_kld_pca": False,
            "skip_interactive": False,
        },
    }

    run_pipeline(PIPELINE)
