r"""One-shot dataset builder CLI -- generate, persist, and preview (Decision D7).

This script generates a synthetic benchmark dataset **exactly once** and caches
it so every downstream training run (every $\beta$, every hyper-parameter
setting) reuses identical samples -- a precondition for a valid $\beta$-sweep
comparison.

Behaviour:
    1. Read ``config_synth.yaml`` (the ``data`` / ``model`` / ``experiment``
       blocks) and call :func:`generators.gen_delayed_gaussian`.
    2. Write to ``<data_dir>/<benchmark>/<tag>/``:
        * ``train.npz``, ``val.npz``, ``test.npz`` -- the tensor splits, each
          holding the five native fields ``fhr_st / fhr_ph / up_st / up_ph /
          weight`` (uncompressed, so they stay memory-mappable);
        * ``meta.json`` -- the analytic ``te_true``, ``true_lag_band``, RNG
          seeds, and the informative/distractor channel map;
        * ``preview.pdf`` -- a visual summary produced via :mod:`visualize`.
    3. Be idempotent: skip generation when the cache exists unless ``--force``.

Run modes (project convention -- see Decision D9 in
``synthetic_te_validation_plan.md``): the script supports **both** a CLI and an
edit-and-run ``__main__``, auto-detected from whether any command-line argument
is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.build_dataset
        ... [--config PATH] [--force] [--tag TAG] [--easy] [--a A] [--m M]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly::

        python -m ...synthetic.build_dataset

The ``--tag/--easy/--a/--m`` overrides let later phases build variant datasets
(e.g. the easy/``a=0`` proof-of-life splits) without editing the config.
``--easy`` can only switch the easy variant *on*; pass a distinct ``--tag`` so
the variant does not clobber the baseline cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
    gen_ar_gaussian,
    gen_delayed_gaussian,
    gen_delayed_xor,
    gen_two_lag_gaussian,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_active_benchmark,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.visualize import (
    make_preview,
)

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.data_dir`` config value is resolved relative to ``model_experiment/``.
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_SPLIT_FILES = ("train.npz", "val.npz", "test.npz", "meta.json")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse the synthetic-experiment YAML config.

    The active benchmark block is overlaid onto the flat ``data`` / ``sweep``
    keys via :func:`train_minimal.resolve_active_benchmark`.

    Args:
        config_path: Path to ``config_synth.yaml``.

    Returns:
        The parsed, benchmark-resolved config as a nested dict.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return resolve_active_benchmark(config)


# Benchmark -> generator dispatch. Benchmark G reuses ``gen_delayed_gaussian``
# (with ``reverse_roles=True`` supplied by :func:`_build_gen_kwargs`).
_GENERATORS = {
    "A": gen_delayed_gaussian,
    "B": gen_ar_gaussian,
    "C": gen_delayed_xor,
    "E": gen_two_lag_gaussian,
    "G": gen_delayed_gaussian,
}


def _build_gen_kwargs(
    benchmark: str, data: Dict[str, Any], model: Dict[str, Any],
    c_y: int, c_u: int,
) -> Dict[str, Any]:
    r"""Assemble the keyword arguments for the active benchmark's generator.

    Each benchmark generator takes a different parameter set (B adds ``rho`` /
    ``burn_in``; C swaps ``a`` / ``sigma2`` for ``q`` / ``obs_noise``; E takes
    two delays / coefficients / channel counts; G is Benchmark A with
    ``reverse_roles=True``). This helper reads only the keys the chosen
    generator accepts.

    Args:
        benchmark: Benchmark id (``A`` / ``B`` / ``C`` / ``E`` / ``G``).
        data: The (benchmark-resolved) ``data`` config block.
        model: The ``model`` config block (for ``horizon``).
        c_y: Target channel count.
        c_u: Source channel count.

    Returns:
        A dict of keyword arguments for ``_GENERATORS[benchmark]`` (excluding
        ``n`` and ``seed``, which :func:`_write_split` supplies per split).

    Raises:
        ValueError: If ``benchmark`` is not one of A / B / C / E / G.
    """
    common: Dict[str, Any] = {
        "T": int(data["sequence_length"]),
        "c_y": c_y,
        "c_u": c_u,
        "horizon": int(model["horizon"]),
        "standardize": True,
    }
    if benchmark in ("A", "G"):
        kw = {
            **common,
            "delay": int(data["delay"]),
            "a": float(data["a"]),
            "sigma2": float(data["sigma2"]),
            "M": int(data["M"]),
            "easy_variant": bool(data.get("easy_variant", False)),
        }
        if benchmark == "G":
            kw["reverse_roles"] = True
        return kw
    if benchmark == "B":
        return {
            **common,
            "delay": int(data["delay"]),
            "a": float(data["a"]),
            "sigma2": float(data["sigma2"]),
            "M": int(data["M"]),
            "rho": float(data["rho"]),
            "burn_in": int(data.get("burn_in", 200)),
            "easy_variant": bool(data.get("easy_variant", False)),
        }
    if benchmark == "C":
        return {
            **common,
            "delay": int(data["delay"]),
            "q": float(data["q"]),
            "M": int(data["M"]),
            "obs_noise": float(data.get("obs_noise", 0.1)),
            "easy_variant": bool(data.get("easy_variant", False)),
        }
    if benchmark == "E":
        return {
            **common,
            "delay1": int(data["delay1"]),
            "delay2": int(data["delay2"]),
            "a1": float(data["a1"]),
            "a2": float(data["a2"]),
            "sigma2": float(data["sigma2"]),
            "M1": int(data["M1"]),
            "M2": int(data["M2"]),
        }
    raise ValueError(
        f"build_dataset: unknown benchmark {benchmark!r} "
        f"(expected one of A, B, C, E, G)."
    )


def _write_split(
    out_dir: Path,
    split: str,
    n: int,
    seed: int,
    generator: Any,
    gen_kwargs: Dict[str, Any],
    split_channels: Dict[str, int],
) -> Dict[str, Any]:
    r"""Generate one split, split $Y/U$ into native fields, and ``.npz`` it.

    Args:
        out_dir: Destination directory for ``<split>.npz``.
        split: Split name (``train`` / ``val`` / ``test``).
        n: Number of samples for this split.
        seed: RNG seed for the generator (independent per split).
        generator: The benchmark's generator function from :data:`_GENERATORS`.
        gen_kwargs: Generator arguments (``T``, the DGP params, channels ...).
        split_channels: Channel counts ``{c_y_st, c_y_ph, c_u_st, c_u_ph}``.

    Returns:
        The generator ``meta`` dict for this split (identical across splits up
        to the RNG seed).
    """
    Y, U, meta = generator(n=n, seed=seed, **gen_kwargs)
    Y_np = Y.numpy()
    U_np = U.numpy()
    c_y_st = split_channels["c_y_st"]
    c_y_ph = split_channels["c_y_ph"]
    c_u_st = split_channels["c_u_st"]
    c_u_ph = split_channels["c_u_ph"]
    T = int(gen_kwargs["T"])

    np.savez(
        out_dir / f"{split}.npz",
        fhr_st=np.ascontiguousarray(Y_np[..., :c_y_st]),
        fhr_ph=np.ascontiguousarray(Y_np[..., c_y_st : c_y_st + c_y_ph]),
        up_st=np.ascontiguousarray(U_np[..., :c_u_st]),
        up_ph=np.ascontiguousarray(U_np[..., c_u_st : c_u_st + c_u_ph]),
        weight=np.ones((n, T), dtype=np.float32),
    )
    print(f"  [{split:5s}] n={n:6d}  seed={seed}  ->  {split}.npz")
    return meta


def build_dataset(
    config: Dict[str, Any], *, force: bool = False
) -> Path:
    r"""Generate, persist, and preview a synthetic benchmark dataset.

    Args:
        config: The parsed ``config_synth.yaml`` (possibly with CLI overrides
            already applied to ``experiment`` / ``data``).
        force: Regenerate even when a complete cache already exists.

    Returns:
        The cache directory ``<data_dir>/<benchmark>/<tag>/``.

    Raises:
        ValueError: If the configured channel counts are inconsistent.
    """
    exp = config["experiment"]
    data = config["data"]
    model = config["model"]

    benchmark = str(exp["benchmark"])
    tag = str(exp["tag"])
    base_seed = int(exp["seed"])

    c_y_st, c_y_ph = int(data["c_y_st"]), int(data["c_y_ph"])
    c_u_st, c_u_ph = int(data["c_u_st"]), int(data["c_u_ph"])
    c_y, c_u = c_y_st + c_y_ph, c_u_st + c_u_ph
    if c_y != int(model["c_y"]) or c_u != int(model["c_u"]):
        raise ValueError(
            f"channel mismatch: data block gives c_y={c_y}, c_u={c_u} but "
            f"model block gives c_y={model['c_y']}, c_u={model['c_u']}."
        )

    data_root = (_EXPERIMENT_DIR / str(config["paths"]["data_dir"])).resolve()
    out_dir = data_root / benchmark / tag

    if not force and all((out_dir / f).is_file() for f in _SPLIT_FILES):
        print(f"cache exists, skipping: {out_dir}  (use --force to rebuild)")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = _GENERATORS.get(benchmark)
    if generator is None:
        raise ValueError(
            f"build_dataset: unknown benchmark {benchmark!r} "
            f"(expected one of {sorted(_GENERATORS)})."
        )
    gen_kwargs = _build_gen_kwargs(benchmark, data, model, c_y, c_u)
    split_channels = {
        "c_y_st": c_y_st, "c_y_ph": c_y_ph,
        "c_u_st": c_u_st, "c_u_ph": c_u_ph,
    }
    split_sizes = {
        "train": int(data["n_train"]),
        "val": int(data["n_val"]),
        "test": int(data["n_test"]),
    }
    # Independent per-split seeds so train / val / test never overlap.
    split_seeds = {
        "train": base_seed + 0,
        "val": base_seed + 1,
        "test": base_seed + 2,
    }

    print(f"building benchmark {benchmark} (tag '{tag}') -> {out_dir}")
    train_meta: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        meta = _write_split(
            out_dir, split, split_sizes[split], split_seeds[split],
            generator, gen_kwargs, split_channels,
        )
        if split == "train":
            train_meta = meta

    # meta.json: the generator ground truth (split-invariant) + cache bookkeeping.
    meta_out = dict(train_meta)
    meta_out.pop("seed", None)
    meta_out["tag"] = tag
    meta_out["split_sizes"] = split_sizes
    meta_out["split_seeds"] = split_seeds
    meta_out["channel_map"] = {
        "fhr_st": [0, c_y_st],
        "fhr_ph": [c_y_st, c_y_st + c_y_ph],
        "up_st": [0, c_u_st],
        "up_ph": [c_u_st, c_u_st + c_u_ph],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2)
    band = meta_out["true_lag_band"]
    band_str = f"{band[0]}..{band[-1]}" if band else "(none)"
    print(f"  [meta ] te_true={meta_out['te_true']:.4f} nats "
          f"(per-step {meta_out['te_per_step']:.4f})  "
          f"lag_band={band_str}  ->  meta.json")

    pdf = make_preview(out_dir, meta_out)
    print(f"  [plot ] -> {pdf.name}")
    print(f"done: {out_dir}")
    return out_dir


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply CLI / in-file overrides in place onto the parsed config.

    Args:
        config: The parsed config dict (mutated in place).
        overrides: A mapping with optional keys ``tag``, ``easy``, ``a``,
            ``m``, ``q``, ``rho``. ``a`` / ``m`` drive the Gaussian (A / B / E)
            sweep cells, ``q`` the XOR (C) sweep cells and ``rho`` the
            Benchmark-B rho-null datasets. Both ``vars(argparse.Namespace)``
            (CLI mode) and the in-file ``RUN_CONFIG`` dict (edit-and-run mode)
            satisfy this -- see Decision D9 in
            ``synthetic_te_validation_plan.md``.
    """
    if overrides.get("tag") is not None:
        config["experiment"]["tag"] = overrides["tag"]
    if overrides.get("easy"):
        config["data"]["easy_variant"] = True
    if overrides.get("a") is not None:
        config["data"]["a"] = overrides["a"]
    if overrides.get("m") is not None:
        config["data"]["M"] = overrides["m"]
    if overrides.get("q") is not None:
        config["data"]["q"] = overrides["q"]
    if overrides.get("rho") is not None:
        config["data"]["rho"] = overrides["rho"]


def main() -> None:
    """CLI entry point: parse arguments, load config, build the dataset."""
    parser = argparse.ArgumentParser(
        description="Generate, persist, and preview a synthetic TE dataset."
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="regenerate even if a complete cache already exists",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="override experiment.tag (cache subdirectory name)",
    )
    parser.add_argument(
        "--easy", action="store_true",
        help="force the easy variant (all channels informative)",
    )
    parser.add_argument(
        "--a", type=float, default=None, help="override data.a (A / B / G)",
    )
    parser.add_argument(
        "--m", type=int, default=None, help="override data.M",
    )
    parser.add_argument(
        "--q", type=float, default=None, help="override data.q (XOR, C)",
    )
    parser.add_argument(
        "--rho", type=float, default=None, help="override data.rho (AR, B)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    _apply_overrides(config, vars(args))
    build_dataset(config, force=args.force)


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision D9)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly.
    #
    # RUN_CONFIG keys mirror the CLI flags; `None` means "fall back to
    # config_synth.yaml". For any setting with no override key, edit the
    # loaded `config` dict directly at the marked spot below.
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "tag": None,        # None -> config experiment.tag (cache subdir name)
        "easy": False,      # True -> force the easy variant (all channels)
        "a": None,          # None -> config data.a   (Gaussian A / B / G)
        "m": None,          # None -> config data.M
        "q": None,          # None -> config data.q   (XOR, benchmark C)
        "rho": None,        # None -> config data.rho (AR, benchmark B)
        "force": False,     # True -> rebuild even if a complete cache exists
    }

    if len(sys.argv) > 1:
        main()              # CLI mode -- argparse
    else:
        config = load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["data"]["n_train"] = 2000
        # ---------------------------------------------------------------------
        _apply_overrides(config, RUN_CONFIG)
        build_dataset(config, force=RUN_CONFIG["force"])

