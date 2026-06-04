r"""One-shot dataset builder CLI -- generate, persist, and preview (Decision V2-D2).

This script generates a synthetic v2 benchmark dataset **exactly once** and
caches it so every downstream training run (every $\beta$, every
hyper-parameter setting) reuses identical samples -- a precondition for a
valid $\beta$-sweep comparison.

Behaviour:
    1. Read ``config_synth.yaml`` (the ``data`` / ``model`` / ``experiment``
       blocks) and call the active benchmark's generator from
       :data:`_GENERATORS` (one of ``gen_state_space_oscillator``,
       ``gen_smooth_arx``, ``gen_regime_switch_smooth``).
    2. Write to ``<data_dir>/<benchmark>/<tag>/``:
        * ``train.npz``, ``val.npz``, ``test.npz`` -- the tensor splits, each
          holding the five native fields ``fhr_st / fhr_ph / up_st / up_ph /
          weight`` (uncompressed, so they stay memory-mappable);
        * ``meta.json`` -- the analytic ``te_true``, ``true_lag_band``, RNG
          seeds, and the informative/distractor channel map;
        * ``preview.pdf`` -- a visual summary produced via :mod:`visualize`.
    3. Be idempotent: skip generation when the cache exists unless ``--force``.

Run modes (project convention -- see Decision V2-D8 in
``model_validation_v2_plan.md``): the script supports **both** a CLI and an
edit-and-run ``__main__``, auto-detected from whether any command-line argument
is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.build_dataset
        ... [--config PATH] [--force] [--tag TAG] [--easy] [--m M]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly::

        python -m ...synthetic.build_dataset

The ``--tag/--easy/--m`` overrides let later phases build variant datasets
without editing the config. ``--easy`` can only switch the easy variant *on*;
pass a distinct ``--tag`` so the variant does not clobber the baseline cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
    DEFAULT_DECOMP_PARAMS,
    gen_regime_switch_smooth,
    gen_smooth_arx,
    gen_state_space_oscillator,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    apply_path_overrides,
    resolve_active_benchmark,
    resolve_user_path,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.visualize import (
    make_dataset_gallery,
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


# Benchmark -> generator dispatch (v2). G1-rev shares G1's generator and only
# differs in the `reverse_roles=True` flag (set via config, not via a dedicated
# function). G1_twoband shares G1's generator with two-delay channel splits.
# G2_wrong_delay / G2_zero_coupling share G2's generator and differ only in
# `delay` / `c` (null-control variants consumed by :mod:`null_controls`). v1
# entries (A/B/C/E/G) were removed (V2-D7).
_GENERATORS: Dict[str, Any] = {
    "G1":               gen_state_space_oscillator,
    "G1-rev":           gen_state_space_oscillator,
    "G1_twoband":       gen_state_space_oscillator,
    "G2":               gen_smooth_arx,
    "G2_wrong_delay":   gen_smooth_arx,
    "G2_zero_coupling": gen_smooth_arx,
    "G3":               gen_regime_switch_smooth,
}


def _resolve_channel_decomp(
    data: Dict[str, Any], c_y: int, c_u: int, benchmark: str,
) -> Dict[str, Any]:
    r"""Build a fully-specified ``channel_decomp`` dict from the YAML config.

    Single source of truth for ``m``: ``data['M']`` (the existing informative-
    channel count). The decomposition holds ``n_smallnoise`` and ``n_noise``
    fixed at their YAML-configured values so the small-noise / pure-noise loss
    floor is invariant across an ``M`` sweep, and derives:

    $$
    n_{\text{self}} = c_y - m - n_{\text{smallnoise}},
    \qquad
    n_{\text{dist}} = c_u - m_{\text{source}} - n_{\text{noise}},
    $$

    where $m_{\text{source}} = m$ for G1 / G2 and
    $m_{\text{source}} = m\cdot K_{\text{classes}}$ for G3. Raises
    ``ValueError`` if either derived size is negative.

    Args:
        data: The benchmark-resolved ``data`` config block. Must contain
            ``M`` and a ``channel_decomp`` sub-block (with ``target`` and
            ``source`` keys); falls back to :data:`DEFAULT_DECOMP_PARAMS`
            for any missing knob.
        c_y, c_u: Channel counts from the same ``data`` block.
        benchmark: Active benchmark id. For G3, the source-side TE width is
            ``M * K_classes``.

    Returns:
        A dict accepted by the generators' ``channel_decomp`` kwarg and by
        :func:`generators._validate_channel_decomp`. Always contains
        ``m``, ``m_source``, ``n_self``, ``n_smallnoise``, ``n_dist``,
        ``n_noise``, ``sigma_smallnoise``, ``ar1_fraction``, and the
        four range tuples.

    Raises:
        ValueError: If ``M`` is missing, or if the resolved ``n_self`` /
            ``n_dist`` would be negative.
    """
    M = int(data["M"])
    if benchmark == "G3":
        m_source = M * int(data.get("K_classes", 1))
    else:
        m_source = M

    raw = data.get("channel_decomp") or {}
    raw_target = raw.get("target") or {}
    raw_source = raw.get("source") or {}

    def _pick(d: Dict[str, Any], key: str, fallback_key: Optional[str] = None) -> Any:
        if key in d:
            return d[key]
        fb = key if fallback_key is None else fallback_key
        return DEFAULT_DECOMP_PARAMS[fb]

    n_smallnoise = int(_pick(raw_target, "n_smallnoise"))
    n_noise = int(_pick(raw_source, "n_noise"))
    sigma_smallnoise = float(_pick(raw_target, "sigma_smallnoise"))
    ar1_fraction = float(_pick(raw_target, "ar1_fraction"))
    # Target / source rho ranges are keyed by sub-block, not by global name.
    rho_range_self = tuple(_pick(raw_target, "rho_range", "rho_range_self"))
    rho_range_dist = tuple(_pick(raw_source, "rho_range", "rho_range_dist"))
    osc_period_range = tuple(_pick(raw_target, "osc_period_range"))
    osc_amp_range = tuple(_pick(raw_target, "osc_amp_range"))

    n_self = c_y - M - n_smallnoise
    n_dist = c_u - m_source - n_noise
    if n_self < 0:
        raise ValueError(
            f"channel_decomp budget closure failed for target: "
            f"c_y={c_y} - M={M} - n_smallnoise={n_smallnoise} = {n_self} < 0. "
            f"Lower n_smallnoise, raise c_y, or lower M."
        )
    if n_dist < 0:
        raise ValueError(
            f"channel_decomp budget closure failed for source: "
            f"c_u={c_u} - m_source={m_source} - n_noise={n_noise} = "
            f"{n_dist} < 0. Lower n_noise, raise c_u, or lower M "
            f"(G3: lower M*K_classes={m_source})."
        )
    return {
        "m":                M,
        "n_self":           int(n_self),
        "n_smallnoise":     n_smallnoise,
        "m_source":         m_source,
        "n_dist":           int(n_dist),
        "n_noise":          n_noise,
        "sigma_smallnoise": sigma_smallnoise,
        "ar1_fraction":     ar1_fraction,
        "rho_range_self":   rho_range_self,
        "rho_range_dist":   rho_range_dist,
        "osc_period_range": osc_period_range,
        "osc_amp_range":    osc_amp_range,
    }


def _build_gen_kwargs(
    benchmark: str, data: Dict[str, Any], model: Dict[str, Any],
    c_y: int, c_u: int,
) -> Dict[str, Any]:
    r"""Assemble the keyword arguments for the active benchmark's generator.

    Branches on ``benchmark`` and copies the relevant fields from the
    benchmark-resolved ``data`` block into the kwargs dict the generator
    expects. ``n`` and ``seed`` are *not* included -- they are supplied per
    split by :func:`_write_split`. ``c_y`` and ``c_u`` (the resolved native
    channel counts) and ``horizon`` (from the ``model`` block) are injected
    so the generator output matches the model's input contract.

    Args:
        benchmark: Active benchmark id (one of the keys of :data:`_GENERATORS`).
        data: The (benchmark-resolved) ``data`` config block.
        model: The ``model`` config block (for ``horizon``).
        c_y: Target channel count.
        c_u: Source channel count.

    Returns:
        A dict of keyword arguments for ``_GENERATORS[benchmark]`` (excluding
        ``n`` and ``seed``).

    Raises:
        ValueError: If ``benchmark`` is not a registered v2 benchmark.
    """
    horizon = int(model["horizon"])
    T = int(data["sequence_length"])
    if benchmark in ("G1", "G1-rev", "G1_twoband"):
        # G1-rev is G1 with reverse_roles=True (flag set in the config block);
        # G1_twoband (Sprint 4.5) is G1 with len(delays) > 1 spread across M.
        # The generator demands ``len(oscillators) == len(delays) == len(B_y)
        # == M``: when the config supplies a single spec we tile it M times,
        # and when the config supplies k > 1 specs with M % k == 0 we repeat
        # each entry M / k times (so e.g. delays=[35, 85], M=4 -> [35, 35, 85,
        # 85]). Asymmetric splits across delays must be set explicitly.
        M = int(data["M"])

        def _tile(lst: List[Any], name: str) -> List[Any]:
            if len(lst) == M:
                return lst
            if len(lst) == 1:
                return lst * M
            if M % len(lst) == 0:
                per = M // len(lst)
                return [item for item in lst for _ in range(per)]
            raise ValueError(
                f"benchmark {benchmark!r}: cannot tile {name} of length "
                f"{len(lst)} to M={M}. Provide a length matching M, length 1, "
                f"or a length that divides M evenly."
            )

        oscillators = _tile(
            [tuple(pair) for pair in data["oscillators"]], "oscillators",
        )
        B_y = _tile([float(b) for b in data["B_y"]], "B_y")
        kwargs: Dict[str, Any] = {
            "T": T,
            "oscillators": oscillators,
            "target_ar": float(data["target_ar"]),
            "B_y": B_y,
            "sigma2_y": float(data["sigma2_y"]),
            "sigma2_eta": data["sigma2_eta"],   # scalar or sequence; generator handles both
            "M": M,
            "c_y": c_y,
            "c_u": c_u,
            "horizon": horizon,
            "K_history": (None if data.get("K_history") is None
                          else int(data["K_history"])),
            "easy_variant": bool(data.get("easy_variant", False)),
            "standardize": bool(data.get("standardize", True)),
            "reverse_roles": bool(data.get("reverse_roles", False)),
            "te_n_samples": int(data.get("te_n_samples", 50_000)),
            "channel_decomp": _resolve_channel_decomp(data, c_y, c_u, benchmark),
        }
        # Delay mode: variable (`delay_min`/`delay_max`) XOR fixed per-channel
        # (`delays`, used by the multi-band variant). In variable mode the lag
        # drifts within each signal as a random walk when `delay_walk` is set.
        if data.get("delay_min") is not None or data.get("delay_max") is not None:
            kwargs["delay_min"] = int(data["delay_min"])
            kwargs["delay_max"] = int(data["delay_max"])
            kwargs["delay_walk"] = bool(data.get("delay_walk", False))
            kwargs["delay_walk_step_prob"] = float(
                data.get("delay_walk_step_prob", 0.02)
            )
        else:
            kwargs["delays"] = _tile([int(d) for d in data["delays"]], "delays")
        return kwargs
    if benchmark in ("G2", "G2_wrong_delay", "G2_zero_coupling"):
        # G2_zero_coupling overrides `c` to 0; G2_wrong_delay overrides the
        # delay range to lags >> max_lag + horizon. All share G2's kwarg
        # surface -- only the YAML values differ.
        kwargs = {
            "T": T,
            "rho_u": float(data["rho_u"]),
            "rho_y": float(data["rho_y"]),
            "c": float(data["c"]),
            "sigma2_eta": float(data["sigma2_eta"]),
            "sigma2_eps": float(data["sigma2_eps"]),
            "M": int(data["M"]),
            "c_y": c_y,
            "c_u": c_u,
            "horizon": horizon,
            "K_history": (None if data.get("K_history") is None
                          else int(data["K_history"])),
            "burn_in": (None if data.get("burn_in") is None
                        else int(data["burn_in"])),
            "easy_variant": bool(data.get("easy_variant", False)),
            "standardize": bool(data.get("standardize", True)),
            "reverse_roles": bool(data.get("reverse_roles", False)),
            "channel_decomp": _resolve_channel_decomp(data, c_y, c_u, benchmark),
        }
        # Variable delay (`delay_min`/`delay_max`) XOR fixed scalar (`delay`).
        # In variable mode the lag drifts within each signal as a random walk
        # when `delay_walk` is set (the G2 controls override it to false).
        if data.get("delay_min") is not None or data.get("delay_max") is not None:
            kwargs["delay_min"] = int(data["delay_min"])
            kwargs["delay_max"] = int(data["delay_max"])
            kwargs["delay_walk"] = bool(data.get("delay_walk", False))
            kwargs["delay_walk_step_prob"] = float(
                data.get("delay_walk_step_prob", 0.02)
            )
        else:
            kwargs["delay"] = int(data["delay"])
        return kwargs
    if benchmark == "G3":
        return {
            "T": T,
            "K_classes": int(data["K_classes"]),
            "p_switch": float(data["p_switch"]),
            "delta": int(data["delta"]),
            "M": int(data["M"]),
            "omega_grid": (None if data.get("omega_grid") is None
                           else [float(w) for w in data["omega_grid"]]),
            "amp_grid": (None if data.get("amp_grid") is None
                         else [float(a) for a in data["amp_grid"]]),
            "sigma2_y": float(data.get("sigma2_y", 0.1)),
            "sigma2_u": float(data.get("sigma2_u", 0.1)),
            "c_y": c_y,
            "c_u": c_u,
            "horizon": horizon,
            "shared_regime": bool(data.get("shared_regime", False)),
            "template_period_min": int(data.get("template_period_min", 40)),
            "standardize": bool(data.get("standardize", True)),
            "channel_decomp": _resolve_channel_decomp(data, c_y, c_u, benchmark),
        }
    raise ValueError(
        f"build_dataset: unknown benchmark {benchmark!r} "
        f"(expected one of {sorted(_GENERATORS)})."
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

    # Per-sample, per-step ground-truth lag d_{i,t} (n, T). Pulled out of
    # ``meta`` (it is a large, non-JSON array) and written into this split's
    # ``.npz`` aligned to its samples. ``None`` for zero-TE controls; absent
    # from legacy generators -> the lag-attention overlay simply isn't drawn.
    true_lag_tt = meta.pop("true_lag_tt", None)
    extra_arrays: Dict[str, Any] = {}
    if true_lag_tt is not None:
        extra_arrays["true_lag_tt"] = np.ascontiguousarray(
            np.asarray(true_lag_tt, dtype=np.int16)
        )

    np.savez(
        out_dir / f"{split}.npz",
        fhr_st=np.ascontiguousarray(Y_np[..., :c_y_st]),
        fhr_ph=np.ascontiguousarray(Y_np[..., c_y_st : c_y_st + c_y_ph]),
        up_st=np.ascontiguousarray(U_np[..., :c_u_st]),
        up_ph=np.ascontiguousarray(U_np[..., c_u_st : c_u_st + c_u_ph]),
        weight=np.ones((n, T), dtype=np.float32),
        **extra_arrays,
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

    # ``paths.data_dir`` can be relative (joined with model_experiment/),
    # absolute on any drive, or use ``~`` / ``$VAR``. See
    # :func:`train_minimal.resolve_user_path` for the rules.
    data_root = resolve_user_path(config["paths"]["data_dir"])
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
    # Stamp the active benchmark key over whatever the generator wrote. This
    # disambiguates G1 from its reverse-roles variant G1-rev (both produced by
    # ``gen_state_space_oscillator``) so downstream consumers see the real
    # cache identity.
    meta_out["benchmark"] = benchmark
    meta_out["split_sizes"] = split_sizes
    meta_out["split_seeds"] = split_seeds
    meta_out["channel_map"] = {
        "fhr_st": [0, c_y_st],
        "fhr_ph": [c_y_st, c_y_st + c_y_ph],
        "up_st": [0, c_u_st],
        "up_ph": [c_u_st, c_u_st + c_u_ph],
    }
    # `channel_decomp` (resolved sizes) and `channel_layout` (per-block index
    # lists) are already populated by the generators in `train_meta`; we
    # re-stamp them here so the cache-bookkeeping `meta_out` always carries
    # them next to `channel_map`. Downstream evaluators read these to colour-
    # code TE / self / smallnoise channels without re-deriving the layout.
    if "channel_decomp" in train_meta:
        meta_out["channel_decomp"] = train_meta["channel_decomp"]
    if "channel_layout" in train_meta:
        meta_out["channel_layout"] = train_meta["channel_layout"]
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2)
    band = meta_out["true_lag_band"]
    band_str = f"{band[0]}..{band[-1]}" if band else "(none)"
    print(f"  [meta ] te_true={meta_out['te_true']:.4f} nats "
          f"(per-step {meta_out['te_per_step']:.4f})  "
          f"lag_band={band_str}  ->  meta.json")

    pdf = make_preview(out_dir, meta_out)
    print(f"  [plot ] -> {pdf.name}")
    gallery = make_dataset_gallery(out_dir, meta_out)
    n_figs = len({Path(p).stem for p in gallery})
    print(f"  [figs ] -> figures/  ({n_figs} figures, {len(gallery)} files)")
    print(f"done: {out_dir}")
    return out_dir


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply CLI / in-file overrides in place onto the parsed config.

    Args:
        config: The parsed config dict (mutated in place).
        overrides: A mapping with optional keys ``tag``, ``easy``, ``m``.
            ``m`` overrides ``data.M`` (number of informative channels) and
            applies to every v2 benchmark. Both ``vars(argparse.Namespace)``
            (CLI mode) and the in-file ``RUN_CONFIG`` dict (edit-and-run mode)
            satisfy this -- see Decision V2-D8 in
            ``model_validation_v2_plan.md``.
    """
    if overrides.get("tag") is not None:
        config["experiment"]["tag"] = overrides["tag"]
    if overrides.get("easy"):
        config["data"]["easy_variant"] = True
    if overrides.get("m") is not None:
        config["data"]["M"] = overrides["m"]
    # data_dir / results_dir overrides -> config["paths"] (None -> YAML default).
    apply_path_overrides(config, overrides)


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
        "--m", type=int, default=None, help="override data.M",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, dest="data_dir",
        help="override paths.data_dir (absolute/relative path, ~, or $VAR); "
             "None -> config paths.data_dir",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, dest="results_dir",
        help="override paths.results_dir (same format as --data-dir); "
             "None -> config paths.results_dir",
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
        "tag": None,         # None -> config experiment.tag (cache subdir name)
        "easy": False,       # True -> force the easy variant (all channels)
        "m": None,           # None -> config data.M (informative channels)
        "force": False,      # True -> rebuild even if a complete cache exists
        "data_dir": None,    # None -> config paths.data_dir
        "results_dir": None, # None -> config paths.results_dir
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

