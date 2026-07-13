r"""Generate & plot sample raw FHR/UP waveforms in the carrier-free ``direct`` render mode.

Demonstrates ``render_mode='direct'`` (``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` §7.4):
the coupled UP-contraction → FHR-deceleration latent pair is rendered *straight* onto the
$4\,\mathrm{Hz}$ raw grid with **no** pulse-shape carrier and **no** amplitude modulation, so
the traces look like clinical FHR/UP recordings rather than the AM-modulated carrier the
default ``am_carrier`` render uses to keep the injected transfer entropy concentrated in a
single, known scattering channel.

Run from the **repo root** with the project interpreter::

    .venv/Scripts/python.exe model/vae_teb_prediction/model/model_experiment/synthetic_v2/preview_direct_mode.py

It solves **one** coupled cell (target block TE, fixed lag $D$) *once*, then renders that
same cell three ways — ``am_carrier`` (the default), ``direct`` one-sided (the default direct
shape), and ``direct`` bipolar — and writes, under
``results/<tag>/figures/direct_preview/``:

* ``raw_preview_<mode>.{pdf,png}`` — the annotated two-panel raw FHR/UP preview per mode
  (via :func:`visualize_v2.plot_raw_preview`);
* ``latent_decomposition_direct.{pdf,png}`` — the latent / (flat-carrier) decomposition for
  the direct render (via :func:`visualize_v2.plot_latent_am_decomposition`);
* ``compare_am_vs_direct.{pdf,png}`` — a 2×2 am-carrier-vs-direct comparison of one sample.

The script is pure ``numpy`` + ``matplotlib`` (the inverter solve and the raw render never
touch ``torch`` / ``kymatio``), so it runs in a few seconds on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable whether run as a script or a module (six levels up:
# synthetic_v2 -> model_experiment -> model -> vae_teb_prediction -> model -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    solve_cell_coupling,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (  # noqa: E402
    generate_cell_raw,
)

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _MODULE_DIR / "config_synth_v2.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    r"""Parse ``config_synth_v2.yaml`` into a nested dict.

    Args:
        path: Path to the YAML config.

    Returns:
        The parsed config tree.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _results_dir(config: Dict[str, Any], benchmark: str) -> Path:
    r"""Resolve ``results/<tag>/`` (tag falls back to the benchmark name).

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key (fallback tag).

    Returns:
        The ``results/<tag>`` directory (absolute; not created).
    """
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    if not results_dir.is_absolute():
        results_dir = _MODULE_DIR / results_dir
    return results_dir / tag


def _plot_comparison(
    raw_am: Dict[str, Any],
    raw_direct: Dict[str, Any],
    out_stem: Path,
    *,
    fs: float,
    sample: int = 0,
) -> List[Path]:
    r"""Write a 2×2 ``am_carrier``-vs-``direct`` raw comparison for one sample.

    Rows are FHR (top) and UP (bottom); columns are the AM-carrier render (left) and the
    carrier-free direct render (right). The physiological baseline / resting tone is marked
    on each panel so the fast AM carrier oscillation (left) reads clearly against the slow
    one-sided contraction / deceleration deflections of the direct render (right).

    Args:
        raw_am: The :func:`raw_generators.generate_cell_raw` dict for the ``am_carrier`` render.
        raw_direct: The same for the ``direct`` render (same coupled cell / seed).
        out_stem: Output path stem (``.pdf`` / ``.png`` are appended).
        fs: Raw sampling rate in Hz (for the time axis).
        sample: Row index to plot.

    Returns:
        The list of written file paths.
    """
    # Local import: pulls matplotlib (and applies the house style) only when plotting.
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import plot_style_v2 as ps

    import matplotlib.pyplot as plt

    fhr_am = np.asarray(raw_am["fhr_raw"], dtype=float)[sample]
    up_am = np.asarray(raw_am["up_raw"], dtype=float)[sample]
    fhr_di = np.asarray(raw_direct["fhr_raw"], dtype=float)[sample]
    up_di = np.asarray(raw_direct["up_raw"], dtype=float)[sample]
    t_min = np.arange(fhr_am.shape[-1]) / fs / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.5), sharex=True)
    (ax_fhr_am, ax_fhr_di), (ax_up_am, ax_up_di) = axes

    panels = (
        (ax_fhr_am, fhr_am, ps.COLOR_BLUE, "FHR (bpm)", "am_carrier"),
        (ax_fhr_di, fhr_di, ps.COLOR_BLUE, "FHR (bpm)", "direct"),
        (ax_up_am, up_am, ps.COLOR_VERMILLION, "UP (mmHg)", "am_carrier"),
        (ax_up_di, up_di, ps.COLOR_VERMILLION, "UP (mmHg)", "direct"),
    )
    for ax, trace, color, ylabel, mode in panels:
        ax.plot(t_min, trace, color=color, lw=0.6)
        ax.axhline(float(np.mean(trace)), color=ps.COLOR_GRAY, lw=0.6, ls=":")
        ax.set_ylabel(ylabel)
        ax.set_title(mode)
        ps.tighten_xaxis(ax, t_min)
        ps.style_axes(ax)
    ax_up_am.set_xlabel("time (min)")
    ax_up_di.set_xlabel("time (min)")

    meta = raw_direct.get("meta", {})
    te_inj = meta.get("te_inj")
    delay = meta.get("D")
    te_txt = "n/a" if te_inj is None else f"{te_inj:.2f} nats"
    fig.suptitle(
        f"am_carrier (left) vs carrier-free direct (right)   "
        f"TE_inj={te_txt}, D={delay}"
    )
    return ps.save_figure(fig, out_stem)


def build_previews(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    target_te: float = 2.0,
    delay: int = 8,
    n: int = 6,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""Solve one coupled cell and render/plot it in ``am_carrier`` and ``direct`` modes.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        target_te: Target injected block TE (nats) for the preview cell.
        delay: Fixed lag $D$ (decimated steps) for the preview cell.
        n: Number of raw pairs to render (the AM envelope uses the batch-pooled latent std,
            so a handful gives a stable amplitude scale).
        out_dir: Output directory; defaults to ``results/<tag>/figures/direct_preview``.

    Returns:
        A dict with the solved ``B`` / achieved ``te_block``, the per-mode ``meta`` dicts,
        and the flat list of written figure ``paths``.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2 import (
        plot_latent_am_decomposition,
        plot_raw_preview,
    )

    seed = int(config.get("seeds", {}).get("dgp", config.get("seeds", {}).get("base_seed", 0)))
    fs = float(config["benchmarks"][benchmark]["raw"]["fs"])
    if out_dir is None:
        out_dir = _results_dir(config, benchmark) / "figures" / "direct_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Solve the coupling ONCE; B is a property of the latent DGP, independent of the raw
    # render, so all three renders below use the identical (B, D) coupled cell.
    solution = solve_cell_coupling(config, target_te, delay, benchmark=benchmark)
    b_scalar = float(solution["B_y_scalar"])
    te_block = float(solution["te_block"])

    # Render the same cell three ways. ``direct`` reads raw.direct.one_sided from config; we
    # additionally force the bipolar variant via a shallow config copy so both are shown.
    import copy

    cfg_bipolar = copy.deepcopy(config)
    cfg_bipolar["benchmarks"][benchmark]["raw"].setdefault("direct", {})["one_sided"] = False

    renders = {
        "am_carrier": generate_cell_raw(
            n, B=b_scalar, D=delay, config=config, benchmark=benchmark, seed=seed,
            te_inj=te_block, render_mode="am_carrier",
        ),
        "direct_one_sided": generate_cell_raw(
            n, B=b_scalar, D=delay, config=config, benchmark=benchmark, seed=seed,
            te_inj=te_block, render_mode="direct",
        ),
        "direct_bipolar": generate_cell_raw(
            n, B=b_scalar, D=delay, config=cfg_bipolar, benchmark=benchmark, seed=seed,
            te_inj=te_block, render_mode="direct",
        ),
    }

    written: List[Path] = []
    for name, raw in renders.items():
        prev_meta = {
            "te_inj": te_block, "D": delay, "B": b_scalar,
            "f_pulse": raw["meta"].get("f_pulse"),
        }
        written += plot_raw_preview(
            raw["fhr_raw"], raw["up_raw"], out_dir / f"raw_preview_{name}",
            meta=prev_meta, fs=fs,
        )

    # Latent / (flat-carrier) decomposition for the default one-sided direct render.
    written += plot_latent_am_decomposition(
        renders["direct_one_sided"]["latents"], out_dir / "latent_decomposition_direct",
        fs=fs, f_pulse=float(renders["direct_one_sided"]["meta"].get("f_pulse", 0.06)),
        meta=renders["direct_one_sided"]["meta"],
    )

    # Side-by-side am-carrier vs (one-sided) direct comparison of one sample.
    written += _plot_comparison(
        renders["am_carrier"], renders["direct_one_sided"],
        out_dir / "compare_am_vs_direct", fs=fs,
    )

    print(
        f"[direct-preview] cell target_te={target_te:g} D={delay} "
        f"B={b_scalar:.4f} te_block={te_block:.4f} (n={n})"
    )
    for path in written:
        print(f"  wrote {path}")
    return {
        "B_y_scalar": b_scalar,
        "te_block": te_block,
        "meta": {name: raw["meta"] for name, raw in renders.items()},
        "paths": written,
    }


def main(argv: Optional[List[str]] = None) -> None:
    r"""CLI entry point for the direct-mode preview.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="Generate & plot sample raw FHR/UP waveforms in the carrier-free "
                    "'direct' render mode (vs the default am_carrier)."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help="Path to config_synth_v2.yaml.")
    parser.add_argument("--benchmark", default=None,
                        help="Benchmark key under `benchmarks:` (default: experiment.benchmark).")
    parser.add_argument("--target-te", type=float, default=2.0,
                        help="Target injected block TE (nats) for the preview cell.")
    parser.add_argument("--delay", type=int, default=8,
                        help="Fixed source->target lag D (decimated steps).")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of raw pairs to render.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: results/<tag>/figures/direct_preview).")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    benchmark = args.benchmark or str(config.get("experiment", {}).get("benchmark", "G1_raw"))
    build_previews(
        config, benchmark=benchmark, target_te=args.target_te,
        delay=args.delay, n=args.n, out_dir=args.out,
    )


if __name__ == "__main__":
    main()
