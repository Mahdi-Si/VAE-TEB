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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Make the repo root importable whether this file is run as a script
# (``python .../run_pipeline_v2.py``) or as a module
# (``python -m ...run_pipeline_v2`` / ``importlib.import_module``). The repo root
# is six levels up: synthetic_v2 -> model_experiment -> model ->
# vae_teb_prediction -> model -> <repo root>.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402  (import after the sys.path bootstrap)

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    solve_cell_coupling,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (  # noqa: E402
    am_separation_from_config,
    generate_cell_raw,
)

_MODULE_FILE = Path(__file__).resolve()
_MODULE_DIR = _MODULE_FILE.parent
_DEFAULT_CONFIG = _MODULE_DIR / "config_synth_v2.yaml"

# Pipeline stages registered for ``--help`` visibility. All implemented:
# ``r0_realizability`` (S3), ``build`` (S4), ``data_previews`` (S7 figure gallery),
# ``train`` + ``beta_select`` (S5), ``eval`` (S6), ``test_plots`` + ``report`` (S7).
_STAGES = [
    "build",
    "r0_realizability",
    "data_previews",
    "train",
    "beta_select",
    "eval",
    "test_plots",
    "report",
]


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
        plot_latent_am_decomposition,
        plot_raw_preview,
        plot_raw_scatter_paired,
        plot_scattering_heatmap,
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
        raw["fhr_raw"], raw["up_raw"], figs_dir / "raw_preview", meta=prev_meta, fs=adapter.fs
    )
    written += plot_scattering_heatmap(
        fields["fhr_st"], fields["up_st"], figs_dir / "scattering_heatmap",
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
        center_freqs=adapter.center_freqs_np, fs=adapter.fs, meta=meta,
    )
    if include_null:
        raw0 = generate_cell_raw(
            max(2, n // 2), B=0.0, D=delay, config=config, benchmark=benchmark,
            seed=seed + 1, te_inj=0.0, render_mode=render_mode,
        )
        written += plot_raw_preview(
            raw0["fhr_raw"], raw0["up_raw"], figs_dir / "raw_preview_null",
            meta={"te_inj": 0.0, "D": delay, "B": 0.0, "f_pulse": f_pulse}, fs=adapter.fs,
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
    overrides: Dict[str, Any] = {}
    if args.pilot:
        overrides["epochs"] = int(train_cfg.get("pilot_epochs", 3))
        overrides["limit_train_batches"] = int(train_cfg.get("pilot_limit_train_batches", 4))
        overrides["limit_val_batches"] = int(train_cfg.get("pilot_limit_val_batches", 2))
        overrides["batch_size"] = int(train_cfg.get("pilot_batch_size", 16))
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    overrides["devices"] = args.devices if args.devices is not None else train_cfg.get("devices", 1)
    return overrides


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


def _resolve_test_ckpt(results_dir: Path, ckpt: Optional[str]) -> Path:
    r"""Resolve the checkpoint for the test-plots bridge (explicit > best > final).

    Args:
        results_dir: The run directory (``results/<tag>/``).
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


def run_test_plots(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
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
        ckpt: Optional explicit checkpoint path (else best/final under ``results/<tag>/``).
        split: Which cache split to plot (default ``test``; falls back to val/train).
        analysis_samples: Number of TE-annotated per-sample diagnostic PDFs to emit.
        out_dir: Optional override for the run directory (defaults to ``results/<tag>/``).

    Returns:
        A dict ``{'out_dir': Path, 'samples_dir': Path, 'sample_diagnostics': ...,
        'kld_lag_diagnostics': ...}``.
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
        build_model,
    )
    from model.vae_teb_prediction.testing.analyses.kld_lag_diagnostics import (
        run_kld_lag_diagnostics,
    )
    from model.vae_teb_prediction.testing.analyses.qualitative import run_sample_diagnostics
    from model.vae_teb_prediction.testing.base import TestRunner
    from train.graph_models_utils import load_checkpoint_strict

    results_dir = _results_dir(config, benchmark) if out_dir is None else Path(out_dir)
    ckpt_path = _resolve_test_ckpt(results_dir, ckpt)
    cache_dir = resolve_cache_dir(config, benchmark=benchmark)
    npz = _resolve_split_npz(cache_dir, split)
    batch_size = int(config.get("optim", {}).get("batch_size", 32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact v1 architecture from the checkpoint's embedded model_kwargs
    # (weights_only=False: a trusted local checkpoint carrying non-tensor metadata), then
    # load the state dict from the SAME already-deserialised blob (load_checkpoint_strict
    # accepts an object containing a state_dict) so the file is not read/unpickled twice.
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model, _ = build_model(dict(blob["model_kwargs"]), device)
    if load_checkpoint_strict(model, blob) is None:
        raise RuntimeError(f"could not load v2 checkpoint {ckpt_path} into the model")
    model.eval()

    out = results_dir / "test_plots"
    out.mkdir(parents=True, exist_ok=True)
    runner = TestRunner(
        model=model,
        device=device,
        output_dir=out,
        warmup_steps=int(getattr(model, "warmup_period", 30)),
        horizon=int(getattr(model, "horizon", 30)),
        max_lag=int(getattr(model, "max_lag", 90)),
        use_up_st=bool(getattr(model, "use_up_st", True)),
    )

    dataset = SyntheticTEDatasetV2(npz)
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
# Edit-and-run dict driver (no argparse needed; mirrors synthetic/run_mixed_pipeline)
# =============================================================================

# Canonical stage order. ``run_pipeline`` validates ``PIPELINE['stages']`` against
# this tuple so a typo fails loudly instead of silently skipping a stage.
_STAGE_ORDER = (
    "solve_te",          # quick coupling query for one (target_te, D) -- diagnostic
    "am_check",          # AM envelope-vs-wavelet separation pre-check -- diagnostic
    "recover",           # frac_Phi render-knob sweep -- opt-in tuning
    "r0_realizability",  # three-TE de-risk pre-flight (writes realizability.json)
    "build",             # generate -> scatter -> normalise -> cache
    "data_previews",     # annotated raw + scattering + latent gallery for a strong cell
    "scatter_preview",   # scattering heatmap of a strong cell -- diagnostic
    "beta_select",       # KL-weight sweep -- opt-in (DDP-safe subprocess)
    "train",             # fit the model -> checkpoint + loss curves (DDP-safe subprocess)
    "eval",              # grade the checkpoint -> metrics.json
    "test_plots",        # standard testing per-sample TE-annotated diagnostics
    "report",            # assemble the markdown report + figure gallery
)

# Defaults applied when ``PIPELINE['stages']`` omits a key: the core
# build -> train -> eval -> test_plots -> report path is ON; the diagnostics /
# opt-in tuning stages are OFF (they must be enabled explicitly).
_STAGE_DEFAULTS = {
    name: name in ("r0_realizability", "build", "data_previews", "train", "eval",
                   "test_plots", "report")
    for name in _STAGE_ORDER
}


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


def _stage_subprocess_cmd(
    config_path: Path,
    stage: str,
    *,
    devices: Any = None,
    epochs: Optional[int] = None,
    pilot: bool = False,
) -> List[str]:
    r"""Assemble a ``--stage {train,beta_select}`` subprocess command (DDP-safe).

    Args:
        config_path: Path to the YAML config handed to the child.
        stage: ``train`` or ``beta_select``.
        devices: Lightning devices spec (``None`` keeps the config/train default).
        epochs: Optional epoch override.
        pilot: Pass ``--pilot`` for the short training smoke.

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
    stages = dict(_STAGE_DEFAULTS)
    stages.update(pipeline.get("stages") or {})
    unknown = set(stages) - set(_STAGE_ORDER)
    if unknown:
        raise ValueError(
            f"unknown stage keys {sorted(unknown)}; valid stages are "
            f"{list(_STAGE_ORDER)}."
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
    epochs = pipeline.get("epochs")
    ckpt = pipeline.get("ckpt")
    split = str(pipeline.get("split", "test"))
    analysis_samples = int(pipeline.get("analysis_samples", 4))
    results_dir = _results_dir(config, benchmark)
    n_stages = len(_STAGE_ORDER)
    status: Dict[str, Any] = {}

    def _sb(name: str, note: str = "") -> None:
        _banner(_STAGE_ORDER.index(name) + 1, n_stages, name, note)

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

    # --- beta_select (opt-in KL sweep; DDP-safe subprocess) ------------------
    _sb("beta_select")
    if not stages["beta_select"]:
        status["beta_select"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    else:
        _run_subprocess(
            _stage_subprocess_cmd(
                config_path, "beta_select", devices=devices, epochs=epochs,
                pilot=train_pilot,
            ),
            dry_run=dry_run,
        )
        status["beta_select"] = "dry-run" if dry_run else "done"

    # --- train (fit -> checkpoint; DDP-safe subprocess) ----------------------
    ckpt_final = results_dir / "final.ckpt"
    _sb("train", note=f"devices={devices if devices is not None else 1}"
                      + (", pilot" if train_pilot else ""))
    if not stages["train"]:
        status["train"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif ckpt_final.is_file() and not force_retrain:
        status["train"] = "skipped (exists)"
        print(f"[pipeline] checkpoint exists, skipping training: {ckpt_final}\n"
              f"           (set force_retrain=True to retrain)")
    else:
        _run_subprocess(
            _stage_subprocess_cmd(
                config_path, "train", devices=devices, epochs=epochs,
                pilot=train_pilot,
            ),
            dry_run=dry_run,
        )
        status["train"] = "dry-run" if dry_run else "done"

    # --- eval (grade the checkpoint) -----------------------------------------
    _sb("eval", note=f"split={split}")
    if not stages["eval"]:
        status["eval"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["eval"] = "dry-run"
        print(f"[pipeline] would grade the checkpoint -> {results_dir / 'metrics.json'}")
    else:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
            run_eval,
        )
        metrics = run_eval(
            config, benchmark=benchmark, ckpt=ckpt, split=split, out_dir=results_dir,
        )
        _print_eval_metrics(metrics)
        status["eval"] = "done"

    # --- test_plots (standard testing per-sample diagnostics) ----------------
    _sb("test_plots", note=f"split={split}")
    if not stages["test_plots"]:
        status["test_plots"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["test_plots"] = "dry-run"
        print(f"[pipeline] would render {analysis_samples} TE-annotated sample PDFs.")
    else:
        # Figures only -- never gate the run on a plotting failure.
        try:
            run_test_plots(
                config, benchmark=benchmark, ckpt=ckpt, split=split,
                analysis_samples=analysis_samples,
            )
            status["test_plots"] = "done"
        except Exception as exc:  # noqa: BLE001 -- diagnostics only
            print(f"[pipeline][warn] test_plots failed: {type(exc).__name__}: {exc}")
            status["test_plots"] = "failed (non-fatal)"

    # --- report (assemble the markdown report) -------------------------------
    _sb("report")
    if not stages["report"]:
        status["report"] = "skipped (disabled)"
        print("[pipeline] disabled.")
    elif dry_run:
        status["report"] = "dry-run"
        print(f"[pipeline] would assemble the report under {results_dir}.")
    else:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
            final_report_v2,
        )
        report_path = final_report_v2(config, benchmark=benchmark)
        print(f"[report] wrote {report_path}")
        status["report"] = "done"

    # --- summary --------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"[pipeline] finished in {elapsed / 60.0:.1f} min")
    for name in _STAGE_ORDER:
        print(f"  {name:18s} {status.get(name, '?')}")
    print(f"[pipeline] artifacts under {results_dir}")
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
    parser.add_argument(
        "--stage",
        choices=_STAGES,
        help=(
            "Pipeline stage to run. Implemented: r0_realizability (S3-T05, the three-TE "
            "de-risk pre-flight), build (S4, generate -> scatter -> normalise -> cache), "
            "train (S5-T02, fit the model -> checkpoint + loss curves; add --pilot for a "
            "short smoke), beta_select (S5-T03, pick the least-collapsed KL over "
            "beta_select.beta_grid), eval (S6, grade a checkpoint -> metrics.json), "
            "data_previews (render the raw / scattering / latent figure gallery for a "
            "strong cell), and report (assemble report.md + the full figure gallery)."
        ),
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
        default="test",
        choices=["test", "val", "train"],
        help=(
            "With --stage eval / test_plots, the split to use (default 'test'; falls "
            "back to val/train when the requested split is not cached)."
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
        if args.stage == "r0_realizability":
            # Default to the pilot grid: the full mix grid generates tens of thousands of
            # scattering passes and is opt-in via --full only.
            from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
                run_realizability_preflight,
            )
            run_realizability_preflight(
                config, benchmark=benchmark, pilot=not args.full,
                out_dir=_results_dir(config, benchmark),
            )
            return 0
        if args.stage == "build":
            # S4: enumerate -> generate -> scatter -> normalise -> cache. Default to the
            # pilot grid (a quick smoke); the locked full mix grid is opt-in via --full.
            from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
                build_all,
            )
            out_dir = build_all(config, benchmark=benchmark, pilot=not args.full)
            print(f"[build] wrote cache -> {out_dir}")
            for split in ("train", "val", "test"):
                npz = out_dir / f"{split}.npz"
                if npz.is_file():
                    print(f"  {split:5s} -> {npz}")
            print(f"  meta   -> {out_dir / 'meta.json'}")
            print(f"  stats  -> {out_dir / 'norm_stats.npz'}")
            return 0
        if args.stage == "data_previews":
            # Render the data-domain gallery (raw + scattering + latent) for a strong cell
            # into results/<tag>/figures/. Pulls torch / kymatio via the adapter, so the
            # import lives inside data_previews.
            data_previews(config, benchmark=benchmark)
            return 0
        if args.stage == "train":
            # S5-T02: fit the unchanged model on the cached splits -> checkpoint + loss
            # curves. --pilot runs a short smoke (few epochs, a handful of batches) on the
            # real cache; the full headline run is --stage train with no --pilot.
            # pl_module_v2 pulls torch / lightning, so import it lazily here.
            from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (
                train_v2,
            )
            overrides = _train_overrides(config, args)
            result = train_v2(config, overrides, benchmark=benchmark)
            _print_train_result(result)
            return 0
        if args.stage == "beta_select":
            # S5-T03: pick the least-collapsed KL weight over beta_select.beta_grid.
            # Explicitly invoking the stage force-runs it even when disabled in config.
            from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (
                beta_select,
            )
            overrides = _train_overrides(config, args)
            overrides["force"] = True
            result = beta_select(config, overrides, benchmark=benchmark)
            print(f"[beta_select] selected beta = {result['selected_beta']:g}")
            for row in result["results"]:
                print(
                    f"  beta={row['beta']:.3e}  kld_nats={row['kld_nats']:.4f}  "
                    f"total_loss={row['total_loss']:.4f}"
                )
            if result.get("out_path"):
                print(f"  wrote {result['out_path']}")
            return 0
        if args.stage == "eval":
            # S6: grade a trained checkpoint -> metrics.json (calibration vs TE_inj /
            # TE_scat, lag recovery, null-control collapse). eval_v2 pulls torch / the
            # loader + model stack, so import it lazily here.
            from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
                run_eval,
            )
            out_dir = _results_dir(config, benchmark)
            metrics = run_eval(
                config, benchmark=benchmark, ckpt=args.ckpt, split=args.split,
                out_dir=out_dir,
            )
            _print_eval_metrics(metrics)
            print(f"  wrote {out_dir / 'metrics.json'}")
            return 0
        if args.stage == "test_plots":
            # S7-T07: bridge a v2 cache split through the standard testing pipeline so the
            # per-sample diagnostics carry the synthetic TE provenance. Imports the model /
            # testing stack lazily via run_test_plots.
            run_test_plots(
                config, benchmark=benchmark, ckpt=args.ckpt, split=args.split,
                analysis_samples=args.analysis_samples,
            )
            return 0
        if args.stage == "report":
            # S7-T05: assemble the full markdown report + headline figure from meta.json /
            # metrics.json / realizability.json / the figure gallery / the standard-testing
            # sample diagnostics. Degrades gracefully when an artifact is missing (so it can
            # run before the headline train/eval); the minimal eval_v2.write_report remains
            # the internal metrics-only fallback.
            from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
                final_report_v2,
            )
            # Graceful degradation is intentional (the report can run before the headline
            # eval), but warn loudly when metrics.json is absent so a skipped/failed eval is
            # not mistaken for a completed report.
            if not (_results_dir(config, benchmark) / "metrics.json").is_file():
                print(
                    "[report] WARNING: no metrics.json found; the report's calibration / "
                    "lag / null gates will read 'n/a'. Run --stage eval for the full report.",
                    file=sys.stderr,
                )
            report_path = final_report_v2(config, benchmark=benchmark)
            print(f"[report] wrote {report_path}")
            return 0
        raise NotImplementedError(
            f"stage '{args.stage}' lands in a later sprint; Sprints 3-6 implement "
            "'r0_realizability', 'build', 'train', 'beta_select', 'eval', and 'report'."
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    # ----- edit-and-run configuration (no CLI; edit and run the file) --------
    # Runs with NO command-line args -> the PIPELINE dict below drives every stage
    # in _STAGE_ORDER (mirrors synthetic/run_mixed_pipeline). Passing ANY argument
    # (e.g. `--stage train`, `--solve-te 2.0 8`) instead dispatches to the argparse
    # CLI in main() -- that same CLI is what the DDP-safe train/beta_select
    # subprocesses re-enter, so it is kept alongside the dict driver.
    PIPELINE: Dict[str, Any] = {
        # --- identifiers ------------------------------------------------------
        "config_path": _DEFAULT_CONFIG,   # config_synth_v2.yaml
        "benchmark": None,                # None -> experiment.benchmark (G1_raw)
        # --- grid / build behaviour -------------------------------------------
        "pilot": False,                   # r0_realizability + build grid: True=pilot,
                                          #   False=full locked mix grid
        "force_rebuild": False,           # True -> regenerate cached build parts
                                          #   (resume=False); False keeps the resume skip
        # --- training knobs ----------------------------------------------------
        "train_pilot": False,             # True -> short training smoke (--pilot)
        "force_retrain": False,           # True -> retrain even when final.ckpt exists
        "devices": 1,                     # Lightning devices: 1 local, 8 on the A6000 box,
                                          #   or "0,1,2,3"; >1 selects DDP (subprocessed)
        "epochs": None,                   # None -> optim.epochs
        # --- eval / test_plots knobs ------------------------------------------
        "ckpt": None,                     # None -> best/final under results/<tag>/
        "split": "test",                  # eval / test_plots split (falls back val/train)
        "analysis_samples": 4,            # TE-annotated per-sample PDFs in test_plots
        # --- diagnostic-stage settings ----------------------------------------
        "solve_te_args": None,            # (target_te, D) required iff stages.solve_te
        "scatter_preview": {"target_te": 2.0, "delay": 8, "n": 16},
        "data_previews": {"target_te": 2.0, "delay": 8, "n": 16, "include_null": True},
        # --- behaviour ---------------------------------------------------------
        "dry_run": False,                 # print the plan (incl. subprocess cmds) only
        # --- stage toggles (executed in _STAGE_ORDER) -------------------------
        "stages": {
            "solve_te": False,            # quick coupling query -- needs solve_te_args
            "am_check": False,            # AM-separation pre-check (diagnostic)
            "recover": False,             # frac_Phi render-knob sweep (opt-in)
            "r0_realizability": True,     # three-TE de-risk pre-flight
            "build": True,                # generate -> scatter -> normalise -> cache
            "data_previews": True,        # raw + scattering + latent gallery
            "scatter_preview": False,     # scattering heatmap (diagnostic; superseded)
            "beta_select": False,         # KL-weight sweep (opt-in)
            "train": True,                # fit -> checkpoint (DDP-safe subprocess)
            "eval": True,                 # grade -> metrics.json
            "test_plots": True,           # standard testing per-sample diagnostics
            "report": True,               # markdown report + figures
        },
    }

    if len(sys.argv) > 1:
        # Any CLI argument -> argparse mode (also the entry the train / beta_select
        # subprocesses and the documented one-off hooks re-enter).
        raise SystemExit(main())
    run_pipeline(PIPELINE)
