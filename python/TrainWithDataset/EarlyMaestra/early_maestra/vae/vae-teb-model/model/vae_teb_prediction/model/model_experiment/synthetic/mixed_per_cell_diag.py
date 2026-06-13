r"""``mixed_per_cell_diag`` -- faithful per-sub-population reproduction of the
single-cell ``lag_recovery`` / ``evaluate_te`` analyses on a ``G1_mix`` model.

``mixed_eval.evaluate_mixed`` already ports most of ``evaluate_te`` and the LOLO
core of ``lag_recovery`` to the heterogeneous pool, grouped per cell. This
module closes the remaining gap: it runs the **actual** single-cell entry points
(:func:`lag_recovery.analyze_lag_recovery`,
:func:`evaluate_te.evaluate_checkpoint`, and optionally
:func:`lag_recovery.sweep_window_widths`) once **per cell**, so the faithful
single-cell figures (attention heatmap, LOLO bar / overlay, per-checkpoint
diagnostics, the window-width sweep) are produced for every sub-population.

It reuses the per-cell loader (:func:`mixed_eval._cell_subset_loader`) and a
single-cell-style meta (:func:`mixed_eval._cell_eval_meta`) and injects them
into those entry points via their ``model_ckpt`` / ``loader_meta`` / ``out_dir``
hooks -- so the model is loaded **once** and no one-cell cache is materialised on
disk. Per cell it writes::

    results/G1_mix/<run_tag>/per_cell/<cache>/cell_<id>/lag_recovery/...
    results/G1_mix/<run_tag>/per_cell/<cache>/cell_<id>/eval_te/...
    results/G1_mix/<run_tag>/per_cell/<cache>/cell_<id>/width_sweep/...   (optional)

and a cross-cell rollup per cache::

    results/G1_mix/<run_tag>/per_cell/<cache>/eval_te_aggregate.csv
    results/G1_mix/<run_tag>/per_cell/<cache>/eval_te_metrics.json
    results/G1_mix/<run_tag>/per_cell/<cache>/lag_recovery_aggregate.csv
    results/G1_mix/<run_tag>/per_cell/<cache>/{kbar_vs_te,per_dim_kl_heatmap,...}

This is a library entry point driven by :mod:`run_mixed_pipeline`'s opt-in
``per_cell_diagnostics`` stage; it is intentionally heavy (a forward / LOLO pass
per cell) and therefore default-off in the pipeline.
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te import (
    _aggregate_metrics,
    _make_checkpoint_figure,
    _make_plots,
    _make_sweep_extras,
    evaluate_checkpoint,
    load_eval_checkpoint,
    write_metrics_json,
    write_summary_csv,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.lag_recovery import (
    analyze_lag_recovery,
    sweep_window_widths,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.mixed_eval import (
    _cell_eval_meta,
    _cell_subset_loader,
    _cells_by_id,
    _load_mixed,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_device,
    resolve_user_path,
)

_BENCHMARK = "G1_mix"

# Which cache groups the stage understands. ``in_mix`` / ``holdout`` resolve to a
# single tag each; ``extrap`` fans over every ``extrap_tags`` entry.
_VALID_CACHES = ("in_mix", "holdout", "extrap")


def _extrap_label(tag: str) -> str:
    """Derive a compact sub-folder label from an extrapolation cache tag.

    Args:
        tag: A cache tag such as ``G1_mix_base_extrap_m64``.

    Returns:
        ``extrap_m64`` for the example above, else ``extrap_<tag>``.
    """
    if "_extrap_m" in tag:
        return "extrap_m" + tag.rsplit("_extrap_m", 1)[-1]
    return f"extrap_{tag}"


def _resolve_targets(
    caches: Sequence[str],
    in_mix_tag: str,
    holdout_tag: Optional[str],
    extrap_tags: Sequence[str],
) -> List[Tuple[str, str]]:
    """Resolve the requested cache groups into ``(tag, label)`` pairs.

    Args:
        caches: Subset of :data:`_VALID_CACHES`.
        in_mix_tag: In-mix cache tag.
        holdout_tag: Interior held-out cache tag (or ``None``).
        extrap_tags: $M$-extrapolation cache tags.

    Returns:
        Ordered ``(tag, label)`` pairs; the label is the per-cache sub-folder.

    Raises:
        ValueError: On an unknown cache key.
    """
    unknown = set(caches) - set(_VALID_CACHES)
    if unknown:
        raise ValueError(
            f"unknown cache keys {sorted(unknown)}; valid: {list(_VALID_CACHES)}."
        )
    targets: List[Tuple[str, str]] = []
    if "in_mix" in caches and in_mix_tag:
        targets.append((in_mix_tag, "in_mix"))
    if "holdout" in caches and holdout_tag:
        targets.append((holdout_tag, "holdout"))
    if "extrap" in caches:
        for t in extrap_tags:
            targets.append((str(t), _extrap_label(str(t))))
    return targets


def _write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of (possibly heterogeneous) dict rows as a union-column CSV.

    Args:
        rows: Row dicts; the header is the ordered union of their keys.
        path: Destination CSV path (parent created).
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _eval_te_per_cell(
    ckpt_path: Path,
    config: Dict[str, Any],
    model_ckpt: Tuple[Any, Dict[str, Any]],
    loader_meta: Tuple[Any, Dict[str, Any]],
    cell: Dict[str, Any],
    cell_id: int,
    data_tag: str,
    out_dir: Path,
    device: Any,
) -> Dict[str, Any]:
    r"""Run :func:`evaluate_te.evaluate_checkpoint` for one cell + write its artifacts.

    Replicates ``evaluate_te``'s ``single`` mode (summary CSV / metrics JSON /
    per-checkpoint diagnostics figure) into the cell's own directory, and
    stamps the cell identity onto the row so the cross-cell rollup labels are
    unique (the bare ``run_tag`` is shared by every cell).

    Args:
        ckpt_path: The real checkpoint path (provenance only -- ``model_ckpt``
            supplies the loaded model).
        config: Parsed config.
        model_ckpt: Pre-loaded ``(model, ckpt)``.
        loader_meta: Per-cell ``(loader, meta)``.
        cell: Manifest cell dict (for the M / TE / band overrides).
        cell_id: The cell id.
        data_tag: Row ``data_tag`` label.
        out_dir: Destination ``.../eval_te`` directory.
        device: Compute device.

    Returns:
        The flat metrics row (cell-stamped).
    """
    row = evaluate_checkpoint(
        ckpt_path, config, device=device, data_tag=data_tag,
        model_ckpt=model_ckpt, loader_meta=loader_meta,
    )
    # Cell-stamp: unique run_tag (figure labels / per_dim_kl keys collide
    # otherwise) and authoritative M / TE / band from the manifest.
    te_true = float(
        cell.get("te_cell_realised", cell.get("target_te", float("nan")))
    )
    row["cell_id"] = int(cell_id)
    row["M"] = int(cell.get("M", row.get("M") or 0))
    row["band"] = str(cell.get("band", ""))
    row["target_te"] = float(cell.get("target_te", float("nan")))
    row["te_true"] = te_true
    row["run_tag"] = f"cell{int(cell_id):03d}_M{row['M']}_te{te_true:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _aggregate_metrics([row])
    write_summary_csv([row], out_dir / "summary.csv")
    write_metrics_json(metrics, [row], out_dir / "metrics.json")
    try:
        _make_checkpoint_figure(row, out_dir)
    except Exception as exc:  # noqa: BLE001 -- a figure must never gate the run
        print(f"[per_cell_diag] eval_te figure (cell {cell_id}) skipped: {exc}")
    return row


def _render_eval_aggregate(
    eval_rows: List[Dict[str, Any]], out_dir: Path,
) -> None:
    """Write the cross-cell eval_te aggregate CSV / JSON / sweep figures.

    Reuses ``evaluate_te``'s own sweep renderers (:func:`evaluate_te._make_plots`,
    :func:`evaluate_te._make_sweep_extras`) so the per-cell rollup matches the
    standalone ``--mode sweep`` figures (``kbar_vs_te`` / ``per_dim_kl_heatmap`` /
    ``null_control``).

    Args:
        eval_rows: One cell-stamped row per cell.
        out_dir: Per-cache destination directory.
    """
    if not eval_rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _aggregate_metrics(eval_rows)
    write_summary_csv(eval_rows, out_dir / "eval_te_aggregate.csv")
    write_metrics_json(metrics, eval_rows, out_dir / "eval_te_metrics.json")
    if len(eval_rows) >= 2:
        try:
            _make_plots(eval_rows, metrics, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[per_cell_diag] eval_te kbar_vs_te figures skipped: {exc}")
        try:
            _make_sweep_extras(eval_rows, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[per_cell_diag] eval_te sweep-extra figures skipped: {exc}")


def _process_cache(
    config: Dict[str, Any],
    tag: str,
    label: str,
    *,
    model_ckpt: Tuple[Any, Dict[str, Any]],
    ckpt_path: Path,
    base_dir: Path,
    device: Any,
    horizon: int,
    bs: int,
    n_per_cell: Optional[int],
    run_lag_recovery: bool,
    run_eval_te: bool,
    run_width_sweep: bool,
    width_grid: Sequence[int],
) -> Dict[str, Any]:
    r"""Run the per-cell diagnostics for one cache (in-mix / holdout / extrap).

    Args:
        config: Parsed config.
        tag: Cache tag under ``data/G1_mix/``.
        label: Per-cache sub-folder name under ``per_cell/``.
        model_ckpt: Pre-loaded ``(model, ckpt)``.
        ckpt_path: The real checkpoint path (provenance).
        base_dir: ``results/G1_mix/<run_tag>/per_cell``.
        device: Compute device.
        horizon, bs: Model / loader geometry.
        n_per_cell: Optional per-cell sample cap.
        run_lag_recovery, run_eval_te, run_width_sweep: Stage toggles.
        width_grid: Window-width grid for the sweep.

    Returns:
        ``{"tag", "label", "n_cells", "out_dir"}`` (or a ``skipped`` reason).
    """
    out_dir = base_dir / label
    try:
        ds, manifest, cellids = _load_mixed(config, tag)
    except FileNotFoundError as exc:
        print(f"[per_cell_diag] cache '{tag}' skipped: {exc}")
        return {"tag": tag, "label": label, "skipped": "cache not found"}
    cells = _cells_by_id(manifest)
    T = int(ds.meta.get("sequence_length", config["model"]["sequence_length"]))
    print(f"[per_cell_diag] cache '{tag}' ({label}): {len(cells)} cells ...")

    eval_rows: List[Dict[str, Any]] = []
    lag_rows: List[Dict[str, Any]] = []
    n_done = 0
    for cid, cell in sorted(cells.items()):
        loader = _cell_subset_loader(ds, cid, cellids, bs, cap=n_per_cell)
        if loader is None:
            continue
        meta = _cell_eval_meta(cell, horizon, T)
        cell_dir = out_dir / f"cell_{int(cid):03d}"
        data_tag = f"{tag}__cell{int(cid)}"
        try:
            if run_lag_recovery:
                res = analyze_lag_recovery(
                    ckpt_path, config, device=device, data_tag=data_tag,
                    n_ablation_samples=n_per_cell,
                    model_ckpt=model_ckpt, loader_meta=(loader, meta),
                    out_dir=cell_dir / "lag_recovery",
                )
                lag_rows.append({"cell_id": int(cid), **res["row"]})
            if run_eval_te:
                eval_rows.append(_eval_te_per_cell(
                    ckpt_path, config, model_ckpt, (loader, meta),
                    cell, int(cid), data_tag, cell_dir / "eval_te", device,
                ))
            if run_width_sweep:
                sweep_window_widths(
                    ckpt_path, config, widths=tuple(width_grid), device=device,
                    data_tag=data_tag, n_ablation_samples=n_per_cell,
                    model_ckpt=model_ckpt, loader_meta=(loader, meta),
                    out_dir=cell_dir / "width_sweep",
                )
            n_done += 1
        except Exception as exc:  # noqa: BLE001 -- one cell must not kill the run
            print(f"[per_cell_diag] cell {cid} ({label}) failed: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    if run_eval_te:
        _render_eval_aggregate(eval_rows, out_dir)
    if run_lag_recovery:
        _write_rows_csv(lag_rows, out_dir / "lag_recovery_aggregate.csv")
    print(f"[per_cell_diag] cache '{label}' done: {n_done} cells -> {out_dir}")
    return {"tag": tag, "label": label, "n_cells": n_done, "out_dir": str(out_dir)}


def run_mixed_per_cell_diag(
    config: Dict[str, Any],
    *,
    run_tag: str,
    in_mix_tag: str,
    holdout_tag: Optional[str] = None,
    extrap_tags: Sequence[str] = (),
    ckpt_name: str = "final.ckpt",
    caches: Sequence[str] = ("in_mix",),
    device: Any = None,
    run_lag_recovery: bool = True,
    run_eval_te: bool = True,
    run_width_sweep: bool = False,
    width_grid: Sequence[int] = (1, 5, 10, 20),
    n_per_cell: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Run the faithful per-cell ``lag_recovery`` / ``evaluate_te`` diagnostics.

    Loads the ``G1_mix`` checkpoint **once** and, for each requested cache,
    runs the injected single-cell entry points per cell (see the module
    docstring), writing per-cell subdirectories plus a cross-cell rollup.

    Args:
        config: Resolved config carrying ``benchmarks.G1_mix`` and ``paths``.
        run_tag: Training run under ``results/G1_mix/<run_tag>/``.
        in_mix_tag: In-mix cache tag.
        holdout_tag: Interior held-out cache tag (or ``None``).
        extrap_tags: $M$-extrapolation cache tags (e.g.
            ``["G1_mix_base_extrap_m64"]``).
        ckpt_name: Checkpoint file name under the run dir.
        caches: Which cache groups to process (subset of
            ``{"in_mix", "holdout", "extrap"}``).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        run_lag_recovery, run_eval_te: Per-analysis toggles.
        run_width_sweep: Whether to run the (heavy) LOLO window-width sweep.
        width_grid: Window-width grid for the sweep.
        n_per_cell: Optional cap on the samples used per cell.

    Returns:
        ``{"run_tag", "ckpt", "out_root", "caches": [per-cache summaries]}``.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: On an unknown cache key.
    """
    device = device or resolve_device(config.get("runtime", {}))
    results_root = resolve_user_path(config["paths"]["results_dir"])
    run_dir = results_root / _BENCHMARK / run_tag
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"per_cell_diag: checkpoint not found: {ckpt_path}. Train the "
            f"G1_mix model (run_mixed_pipeline 'train' stage) first."
        )
    model_ckpt = load_eval_checkpoint(ckpt_path, device)

    model_cfg = config["model"]
    horizon = int(model_cfg["horizon"])
    eval_cfg = config["benchmarks"][_BENCHMARK].get("eval", {}) or {}
    bs = int(eval_cfg.get("batch_size") or config.get("optim", {}).get("batch_size", 32))

    targets = _resolve_targets(caches, in_mix_tag, holdout_tag, extrap_tags)
    base_dir = run_dir / "per_cell"
    print(
        f"[per_cell_diag] G1_mix per-cell diagnostics\n"
        f"           ckpt        = {ckpt_path}\n"
        f"           caches      = {[lbl for _, lbl in targets]}\n"
        f"           lag_recovery={run_lag_recovery}  eval_te={run_eval_te}  "
        f"width_sweep={run_width_sweep}\n"
        f"           n_per_cell  = {n_per_cell}\n"
        f"           out         = {base_dir}"
    )

    cache_summaries: List[Dict[str, Any]] = []
    for tag, label in targets:
        cache_summaries.append(_process_cache(
            config, tag, label,
            model_ckpt=model_ckpt, ckpt_path=ckpt_path, base_dir=base_dir,
            device=device, horizon=horizon, bs=bs,
            n_per_cell=n_per_cell,
            run_lag_recovery=run_lag_recovery, run_eval_te=run_eval_te,
            run_width_sweep=run_width_sweep, width_grid=width_grid,
        ))

    print(f"[per_cell_diag] finished -> {base_dir}")
    return {
        "run_tag": run_tag,
        "ckpt": str(ckpt_path),
        "out_root": str(base_dir),
        "caches": cache_summaries,
    }
