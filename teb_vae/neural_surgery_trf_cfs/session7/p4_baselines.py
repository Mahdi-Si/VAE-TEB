r"""Phase 4: linear baseline forecasters, scored per anchor on the cells the model is scored on.

The pilot fine-tunes of the model are compared with four linear forecasters anchor by anchor, so
the baselines must use the same anchors and the same cells as the model evaluation. This script
fits the baselines on the *fit* recordings (split code 0) and writes one squared-error table per
role.

**Anchors.** The anchors are those of :func:`s7_interventions.matched_forward` at stride 5 and
phase 0 with ``anchor_valid`` true. A feature uses only values stored at or before the anchor
step $t$, namely the lags $t, t-1, \dots, t-P+1$ with $P = 6$.

**Cells.** A cell $(\tau, c)$ of anchor $a$ is *counted* when $M_{\tau c}\, m_{a\tau} > 0$, where
$M$ is the model's ``cell_mask`` and $m$ the anchor's ``step_mask``. The block $c < 32$ holds the
scattering coefficients and the block $c \ge 32$ the phase coefficients. Squared errors are summed
over counted cells only, in the model's normalised units.

**Forecasters.** With $y_t \in \mathbb{R}^{76}$ the kept target channels and $u_t \in \mathbb{R}^{42}$
the kept source channels (a source value is set to zero where the validity mask closes it):

* ``persistence``: $\hat y_{t+\tau+1} = y_t$;
* ``own_ar``: per channel $c$, a direct multi-step ridge from $y_{t-p,c}$, $p < P$, to
  $y_{t+\tau+1,c}$ for every $\tau$;
* ``ridge_y``: one direct ridge from the $P$ lags of all 76 target channels to every cell;
* ``ridge_yu``: ``ridge_y`` plus the $P$ lags of all 42 source channels.

**Fit.** For a fixed horizon step $\tau$ the set of rows that enter the fit does not depend on the
channel, because $M$ is a fixed grid and only $m_{a\tau}$ varies with the row. So one weighted
normal equation per $\tau$ serves all channels:

$$W_\tau = \left(X_\tau^\top X_\tau + \lambda N_\tau I\right)^{-1} X_\tau^\top Y_\tau ,$$

where $X_\tau$ holds the rows with $m_{a\tau} > 0$, $N_\tau$ is their count and the intercept is
penalised like every other weight. The fit therefore needs only the sufficient statistics
$X^\top X$, $X^\top Y$ and $Y^\top Y$, which one pass over the fit recordings accumulates. A
held-out squared error follows from the statistics of the held-out part alone,

$$\mathrm{SSE}(W) = \operatorname{tr}\!\left(W^\top G W\right) - 2 \operatorname{tr}\!\left(W^\top X^\top Y\right) + \operatorname{tr}\!\left(Y^\top Y\right).$$

**Penalty.** The penalty $\lambda$ is chosen per forecaster by a two-fold split *inside* code 0:
the fit recordings divide by the parity of $\lfloor \operatorname{crc32}(\mathrm{guid}) / 4 \rfloor$,
each half is fitted and scored on the other, and the $\lambda$ with the least pooled held-out
squared error per counted cell wins. Codes 1, 2 and 3 never enter the fit or the choice.

**Outputs** (in ``out/phase4``): ``baselines_fit.parquet`` (code 0, in-sample),
``baselines_select.parquet`` (code 2), ``baselines_test_SEALED.parquet`` (codes 1 and 3, written
and never read here) and ``baseline_weights.npz``.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass flags.
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.lag_attn_transformer_cfs.eval.linear_baselines import _lagged  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs. ``limit`` stops after that many batches (``None`` reads all).
RUN_ARGS: Dict[str, Any] = {
    "device": "cuda:0",
    "limit": None,
    "anchor_stride": 5,
    "lags": 6,
    "lambdas": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    "out_dir": str(S.OUT / "phase4"),
}

#: Forecasters in table order.
NAMES = ("persistence", "own_ar", "ridge_y", "ridge_yu")
#: Horizon prefixes with their own sums: ``sse0`` is $\tau = 0$ and ``sse5`` is $\tau < 5$.
PREFIXES = (("sse", None), ("sse0", 1), ("sse5", 5))
#: Width of the scattering block among the kept target channels.
N_SCATTERING = 32

F64 = torch.float64


class Stats:
    r"""Sufficient statistics of the weighted ridge fits for one half of the fit recordings.

    Attributes:
        gram: $X^\top X$ per horizon step, ``(H, F, F)``, with features ordered
            ``[y lags, 1, u lags]``.
        xty: $X^\top Y$ per horizon step, ``(H, F, C)``.
        yty: $\sum Y^2$ per cell, ``(H, C)``.
        n_rows: Rows entering the fit at each step, ``(H,)``.
        own_gram: Per-channel Gram of ``[own lags, 1]``, ``(H, C, P+1, P+1)``.
        own_xty: Per-channel cross term, ``(H, C, P+1)``.
    """

    def __init__(self, horizon: int, channels: int, lags: int, width_u: int) -> None:
        """Allocate zeroed statistics.

        Args:
            horizon: $H$.
            channels: $C$, the kept target channels.
            lags: $P$.
            width_u: $C_u$, the kept source channels.
        """
        width = lags * channels + 1 + lags * width_u
        self.gram = torch.zeros(horizon, width, width, dtype=F64)
        self.xty = torch.zeros(horizon, width, channels, dtype=F64)
        self.yty = torch.zeros(horizon, channels, dtype=F64)
        self.n_rows = torch.zeros(horizon, dtype=F64)
        self.own_gram = torch.zeros(horizon, channels, lags + 1, lags + 1, dtype=F64)
        self.own_xty = torch.zeros(horizon, channels, lags + 1, dtype=F64)

    def add(self, x: torch.Tensor, own: torch.Tensor, target: torch.Tensor, step: torch.Tensor) -> None:
        """Accumulate one batch of rows.

        Args:
            x: Full features ``(n, F)``.
            own: Per-channel features ``(n, P+1, C)`` with the intercept last.
            target: Targets ``(n, H, C)``, zero outside the counted cells.
            step: Row weights ``(n, H)`` in $\\{0, 1\\}$.
        """
        for tau in range(step.shape[1]):
            pick = step[:, tau] > 0
            if not bool(pick.any()):
                continue
            xs, ys, os_ = x[pick], target[pick, tau], own[pick]
            self.gram[tau] += xs.T @ xs
            self.xty[tau] += xs.T @ ys
            self.yty[tau] += (ys * ys).sum(0)
            self.n_rows[tau] += float(pick.sum())
            self.own_gram[tau] += torch.einsum("npc,nqc->cpq", os_, os_)
            self.own_xty[tau] += torch.einsum("npc,nc->cp", os_, ys)

    def __iadd__(self, other: "Stats") -> "Stats":
        """Add another half's statistics in place."""
        for name in ("gram", "xty", "yty", "n_rows", "own_gram", "own_xty"):
            getattr(self, name).add_(getattr(other, name))
        return self


def solve(gram: torch.Tensor, xty: torch.Tensor, n_rows: torch.Tensor, lam: float) -> torch.Tensor:
    r"""Solve $W = (G + \lambda N I)^{-1} X^\top Y$ for every leading index.

    Args:
        gram: ``(..., F, F)`` with one leading index per horizon step.
        xty: ``(..., F, K)`` or ``(..., F)`` for a single output.
        n_rows: Row count per horizon step, ``(H,)``, broadcast against the leading indices.
        lam: $\lambda$.

    Returns:
        The weights with the shape of ``xty``.
    """
    width = gram.shape[-1]
    scale = (lam * n_rows.clamp(min=1.0)).reshape(-1, *([1] * (gram.dim() - 1)))
    system = gram + scale * torch.eye(width, dtype=gram.dtype)
    vector = xty.dim() == gram.dim() - 1
    solution = torch.linalg.solve(system, xty.unsqueeze(-1) if vector else xty)
    return solution.squeeze(-1) if vector else solution


def heldout_sse(weights: torch.Tensor, stats: Stats, cell_mask: torch.Tensor, kind: str, width_y: int) -> Tuple[float, float]:
    """Squared error of fitted weights on a held-out half, from its statistics alone.

    Args:
        weights: ``(H, F, C)`` for the ridge kinds, ``(H, C, P+1)`` for ``own_ar``.
        stats: Statistics of the held-out half.
        cell_mask: ``(H, C)`` float.
        kind: ``'own_ar'``, ``'ridge_y'`` or ``'ridge_yu'``.
        width_y: Number of leading features that ``ridge_y`` uses.

    Returns:
        ``(sse, n_cells)`` over the counted cells.
    """
    if kind == "own_ar":
        quad = torch.einsum("tcp,tcpq,tcq->tc", weights, stats.own_gram, weights)
        cross = torch.einsum("tcp,tcp->tc", weights, stats.own_xty)
    else:
        cut = width_y if kind == "ridge_y" else weights.shape[1]
        gram, xty = stats.gram[:, :cut, :cut], stats.xty[:, :cut]
        quad = (weights * (gram @ weights)).sum(1)
        cross = (weights * xty).sum(1)
    cells = quad - 2.0 * cross + stats.yty
    return float((cells * cell_mask).sum()), float((cell_mask * stats.n_rows[:, None]).sum())


def choose_penalty(halves: Tuple[Stats, Stats], cell_mask: torch.Tensor, lambdas: Sequence[float],
                   width_y: int) -> Dict[str, Any]:
    """Pick $\\lambda$ per forecaster by fitting each half and scoring the other.

    Args:
        halves: Statistics of the two inner halves of code 0.
        cell_mask: ``(H, C)`` float.
        lambdas: The grid.
        width_y: Leading feature count of ``ridge_y``.

    Returns:
        ``{'best': {kind: lambda}, 'curve': {kind: [mse per lambda]}}``.
    """
    curve: Dict[str, List[float]] = {kind: [] for kind in ("own_ar", "ridge_y", "ridge_yu")}
    for lam in lambdas:
        for kind in curve:
            total, count = 0.0, 0.0
            for fit, test in (halves, halves[::-1]):
                if kind == "own_ar":
                    w = solve(fit.own_gram, fit.own_xty, fit.n_rows, lam)
                elif kind == "ridge_y":
                    w = solve(fit.gram[:, :width_y, :width_y], fit.xty[:, :width_y], fit.n_rows, lam)
                else:
                    w = solve(fit.gram, fit.xty, fit.n_rows, lam)
                sse, n = heldout_sse(w, test, cell_mask, kind, width_y)
                total, count = total + sse, count + n
            curve[kind].append(total / count)
    best = {kind: float(lambdas[int(np.argmin(values))]) for kind, values in curve.items()}
    return {"best": best, "curve": curve}


def fit_weights(stats: Stats, lam: Dict[str, float], width_y: int) -> Dict[str, torch.Tensor]:
    """Fit the three ridge forecasters on the whole fit set.

    Args:
        stats: Statistics of all code-0 rows.
        lam: Chosen penalty per forecaster.
        width_y: Leading feature count of ``ridge_y``.

    Returns:
        ``{'own_ar': (H, C, P+1), 'ridge_y': (H, width_y, C), 'ridge_yu': (H, F, C)}``.
    """
    return {
        "own_ar": solve(stats.own_gram, stats.own_xty, stats.n_rows, lam["own_ar"]),
        "ridge_y": solve(stats.gram[:, :width_y, :width_y], stats.xty[:, :width_y], stats.n_rows, lam["ridge_y"]),
        "ridge_yu": solve(stats.gram, stats.xty, stats.n_rows, lam["ridge_yu"]),
    }


class Rows:
    """The rows of one batch that the baselines read, flattened over valid anchors.

    Attributes:
        guid: GUID per row, ``(n,)`` object array.
        epoch: Batch ``epoch`` per row.
        anchor: Anchor step per row.
        x: Features ``[y lags, 1, u lags]``, ``(n, F)``, float64.
        own: Per-channel features, ``(n, P+1, C)`` with the intercept last.
        last: $y_t$, ``(n, C)``.
        target: Targets ``(n, H, C)``, zero outside the counted cells.
        counted: Counted-cell mask ``(n, H, C)``.
        step: Step mask as $\\{0, 1\\}$, ``(n, H)``.
    """

    def __init__(self, task: Any, model: Any, batch: Any, anchor_stride: int, lags: int,
                 cell_mask: torch.Tensor) -> None:
        """Build the rows with the matched forward.

        Args:
            task: The loaded task.
            model: The network.
            batch: One batch on the device.
            anchor_stride: Spacing between decoded anchors.
            lags: $P$.
            cell_mask: ``(H, C)`` bool, on the model's device.
        """
        with torch.no_grad():
            m = I.matched_forward(task, model, batch, anchor_stride=anchor_stride, anchor_phase=0)
            y_st, y_ph, _, _, _, source_valid = m.inputs
            y = model.target_gate(torch.cat([y_st, y_ph], dim=-1))
            keep = None if model.source_gate is None else model.source_gate.keep_index
            valid_u = source_valid if keep is None else source_valid.index_select(-1, keep)
            u = m.source * valid_u.to(m.source.dtype)
            valid = m.anchor_valid.to(torch.bool)
            if bool((m.anchors[valid] < lags - 1).any()):
                raise ValueError("an anchor lies before the first full lag window")
            anchors = m.anchors.to(torch.long).clamp(min=lags - 1)
            y_lags, u_lags = _lagged(y, anchors, lags), _lagged(u, anchors, lags)  # (B, A, P, C)
            step = (m.step_mask > 0)
            counted = step[..., None] & cell_mask[None, None]
            target = torch.where(counted, m.target, torch.zeros_like(m.target))
            nan_history = (~torch.isfinite(y_lags).all(-1).all(-1)) | (~torch.isfinite(u_lags).all(-1).all(-1))
            y_lags, u_lags = torch.nan_to_num(y_lags), torch.nan_to_num(u_lags)
            self.n_nonfinite = int((nan_history & valid).sum())

            size = y_lags.shape[0]
            x = torch.cat([y_lags.flatten(2), torch.ones(size, y_lags.shape[1], 1, device=y.device),
                           u_lags.flatten(2)], dim=-1)
            own = torch.cat([y_lags, torch.ones(size, y_lags.shape[1], 1, y_lags.shape[-1], device=y.device)], dim=2)
            count = valid.shape[1]
            guids = np.repeat(np.asarray([S.guid_text(g) for g in batch["guid"]], dtype=object), count)
            flat = valid.reshape(-1)
            cpu = valid.reshape(-1).cpu().numpy()
            self.guid = guids[cpu]
            self.epoch = np.repeat(batch["epoch"].float().cpu().numpy(), count)[cpu]
            self.anchor = m.anchors.reshape(-1).cpu().numpy()[cpu]
            self.x = x.reshape(-1, x.shape[-1])[flat].to("cpu", F64)
            self.own = own.reshape(-1, *own.shape[2:])[flat].to("cpu", F64)
            self.last = y_lags[:, :, 0].reshape(-1, y_lags.shape[-1])[flat].to("cpu", F64)
            self.target = target.reshape(-1, *target.shape[2:])[flat].to("cpu", F64)
            self.counted = counted.reshape(-1, *counted.shape[2:])[flat].cpu()
            self.step = step.reshape(-1, step.shape[-1])[flat].to("cpu", F64)

    def take(self, index: torch.Tensor) -> "Rows":
        """Return a shallow copy restricted to the rows ``index``.

        Args:
            index: Long tensor of row positions.

        Returns:
            A new :class:`Rows` view.
        """
        new = object.__new__(Rows)
        pos = index.numpy()
        new.guid, new.epoch, new.anchor = self.guid[pos], self.epoch[pos], self.anchor[pos]
        for name in ("x", "own", "last", "target", "counted", "step"):
            setattr(new, name, getattr(self, name)[index])
        new.n_nonfinite = 0
        return new


def predictions(rows: Rows, weights: Dict[str, torch.Tensor], width_y: int) -> Dict[str, torch.Tensor]:
    """Predict the ``(n, H, C)`` forecast block with every forecaster.

    Args:
        rows: The rows to predict.
        weights: Output of :func:`fit_weights`.
        width_y: Leading feature count of ``ridge_y``.

    Returns:
        ``{forecaster: (n, H, C)}`` in the order of :data:`NAMES`.
    """
    horizon = weights["own_ar"].shape[0]
    return {
        "persistence": rows.last[:, None, :].expand(-1, horizon, -1),
        "own_ar": torch.einsum("npc,tcp->ntc", rows.own, weights["own_ar"]),
        "ridge_y": torch.einsum("nf,tfc->ntc", rows.x[:, :width_y], weights["ridge_y"]),
        "ridge_yu": torch.einsum("nf,tfc->ntc", rows.x, weights["ridge_yu"]),
    }


def score_rows(rows: Rows, weights: Dict[str, torch.Tensor], width_y: int) -> pd.DataFrame:
    """Squared-error sums per anchor and block for every forecaster.

    Args:
        rows: The rows to score.
        weights: Output of :func:`fit_weights`.
        width_y: Leading feature count of ``ridge_y``.

    Returns:
        One row per anchor with the columns of the table schema, except ``cohort`` and ``code``.
    """
    blocks = {"st": slice(0, N_SCATTERING), "ph": slice(N_SCATTERING, None)}
    frame: Dict[str, Any] = {"guid": rows.guid, "epoch": rows.epoch, "anchor": rows.anchor}

    def reduce(cells: torch.Tensor, block: slice, upto: Optional[int]) -> np.ndarray:
        part = cells[:, :upto, block]
        return part.sum(dim=(1, 2)).numpy()

    counted = rows.counted.to(F64)
    for prefix, upto in PREFIXES:
        for tag, block in blocks.items():
            if prefix == "sse":
                frame[f"n_{tag}"] = reduce(counted, block, upto).astype(np.int64)
            else:
                frame[f"n{prefix[3:]}_{tag}"] = reduce(counted, block, upto).astype(np.int64)
    for name, pred in predictions(rows, weights, width_y).items():
        cells = torch.where(rows.counted, (pred - rows.target) ** 2, torch.zeros((), dtype=F64))
        for prefix, upto in PREFIXES:
            for tag, block in blocks.items():
                frame[f"{prefix}_{tag}_{name}"] = reduce(cells, block, upto)
    return pd.DataFrame(frame)


def inner_half(guid: str) -> int:
    r"""Inner half of a fit recording, $\lfloor \operatorname{crc32}(\mathrm{guid}) / 4 \rfloor \bmod 2$.

    Args:
        guid: The recording GUID.

    Returns:
        0 or 1.
    """
    return (zlib.crc32(S.guid_text(guid).encode()) // 4) % 2


def batches(task: Any, cfg: dict, dev: Any, model: Any, anchor_stride: int, lags: int,
            cell_mask: torch.Tensor, limit: Optional[int], tag: str):
    """Iterate :class:`Rows` per batch with a progress line.

    Args:
        task: The loaded task.
        cfg: The merged configuration.
        dev: Device.
        model: The network.
        anchor_stride: Spacing between decoded anchors.
        lags: $P$.
        cell_mask: ``(H, C)`` bool on the device.
        limit: Number of batches or ``None``.
        tag: Label for the progress line.

    Yields:
        One :class:`Rows` per batch.
    """
    start = time.time()
    for index, batch in enumerate(S.iter_batches(task, cfg, dev, limit=limit)):
        yield Rows(task, model, batch, anchor_stride, lags, cell_mask)
        if index % 20 == 0:
            print(f"[{tag}] batch {index}  {time.time() - start:.0f}s", flush=True)


def _ratio(total: float, count: float) -> float:
    r"""Return ``total / count``, or NaN when no cell was counted (for example $\tau = 0$ scattering).

    Args:
        total: Summed squared error.
        count: Number of counted cells.

    Returns:
        The mean squared error.
    """
    return float(total) / float(count) if count > 0 else float("nan")


def summarise(table: pd.DataFrame, code: int) -> None:
    """Print the MSE per block and the linear source gain for one split code.

    Args:
        table: The per-anchor table of that code.
        code: The split code, for the heading.
    """
    groups = {"all": table["cohort"].notna(), "healthy": table["cohort"] == "healthy",
              "acidosis+hie": table["cohort"].isin(["acidosis", "hie"])}
    for label, pick in groups.items():
        sub = table[pick]
        if sub.empty:
            continue
        print(f"\n== code {code} | {label} | {len(sub)} anchors, {sub['guid'].nunique()} recordings ==")
        print(f"{'forecaster':<13}{'MSE st':>10}{'MSE ph':>10}{'MSE tau0 st':>13}{'MSE tau0 ph':>13}")
        for name in NAMES:
            cols = [_ratio(sub[f"sse_st_{name}"].sum(), sub["n_st"].sum()),
                    _ratio(sub[f"sse_ph_{name}"].sum(), sub["n_ph"].sum()),
                    _ratio(sub[f"sse0_st_{name}"].sum(), sub["n0_st"].sum()),
                    _ratio(sub[f"sse0_ph_{name}"].sum(), sub["n0_ph"].sum())]
            print(f"{name:<13}" + "".join(f"{v:>{w}.5f}" if np.isfinite(v) else f"{'n/a':>{w}}"
                                          for v, w in zip(cols, (10, 10, 13, 13))))
        recording = (sub["cohort"] + "/" + sub["guid"]).to_numpy()
        for tag, num in (("st", "n_st"), ("ph", "n_ph")):
            keep = sub[num].to_numpy() > 0
            delta = (sub[f"sse_{tag}_ridge_yu"] - sub[f"sse_{tag}_ridge_y"]).to_numpy()[keep]
            weight = sub[num].to_numpy()[keep].astype(float)
            result = S.paired_bootstrap(recording[keep], delta / weight, weight)
            print(f"ridge_yu - ridge_y, {tag}: {S.format_ci(result, 5)}")
        both = sub["n_st"] + sub["n_ph"]
        keep = both.to_numpy() > 0
        delta = ((sub["sse_st_ridge_yu"] + sub["sse_ph_ridge_yu"] - sub["sse_st_ridge_y"] - sub["sse_ph_ridge_y"])
                 .to_numpy()[keep])
        weight = both.to_numpy()[keep].astype(float)
        print("ridge_yu - ridge_y, both: " + S.format_ci(S.paired_bootstrap(recording[keep], delta / weight, weight), 5))


def main(device: str, limit: Optional[int], anchor_stride: int, lags: int, lambdas: Sequence[float],
         out_dir: str) -> Dict[str, Any]:
    """Fit the baselines on code 0 and write the three per-anchor tables and the weights.

    Args:
        device: Torch device string.
        limit: Number of batches per pass, or ``None`` for every batch.
        anchor_stride: Spacing between decoded anchors.
        lags: $P$.
        lambdas: Ridge penalty grid.
        out_dir: Output directory.

    Returns:
        A dictionary with the chosen penalties and the penalty curve.
    """
    start = time.time()
    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model
    cell_mask = model.forecast_likelihood_kwargs()["cell_mask"].bool().to(dev)
    horizon, channels = cell_mask.shape
    keep_target = model.target_gate.keep_index.cpu().numpy()
    if int((keep_target < 36).sum()) != N_SCATTERING or not bool((keep_target[:N_SCATTERING] < 36).all()):
        raise ValueError("the kept target channels are not 32 scattering followed by phase channels")
    width_u = model.source_gate.out_channels if model.source_gate is not None else task.orig_model.c_u
    width_y = lags * channels + 1
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cell_cpu = cell_mask.cpu().to(F64)

    # Pass 1: sufficient statistics of the fit recordings, by inner half.
    halves = (Stats(horizon, channels, lags, width_u), Stats(horizon, channels, lags, width_u))
    n_nonfinite = 0
    for rows in batches(task, cfg, dev, model, anchor_stride, lags, cell_mask, limit, "stats"):
        n_nonfinite += rows.n_nonfinite
        code = np.array([S.guid_hash(g) for g in rows.guid])
        half = np.array([inner_half(g) for g in rows.guid])
        for h in (0, 1):
            pick = torch.as_tensor(np.nonzero((code == S.FIT_CODE) & (half == h))[0])
            if pick.numel():
                sub = rows.take(pick)
                halves[h].add(sub.x, sub.own, sub.target, sub.step)
    torch.cuda.empty_cache()
    print(f"pass 1 done {time.time() - start:.0f}s; rows per half (tau=0): "
          f"{[int(h.n_rows[0]) for h in halves]}; anchors with a non-finite history: {n_nonfinite}")

    choice = choose_penalty(halves, cell_cpu, lambdas, width_y)
    print("inner two-fold pooled MSE per counted cell:")
    for kind, values in choice["curve"].items():
        print(f"  {kind:<9}" + "  ".join(f"{lam:g}:{v:.5f}" for lam, v in zip(lambdas, values)))
    print("chosen penalty:", choice["best"])
    full = Stats(horizon, channels, lags, width_u)
    full += halves[0]
    full += halves[1]
    weights = fit_weights(full, choice["best"], width_y)
    np.savez(out / "baseline_weights.npz", own_ar=weights["own_ar"].numpy().astype(np.float32),
             ridge_y=weights["ridge_y"].numpy().astype(np.float32),
             ridge_yu=weights["ridge_yu"].numpy().astype(np.float32),
             lam_own_ar=choice["best"]["own_ar"], lam_ridge_y=choice["best"]["ridge_y"],
             lam_ridge_yu=choice["best"]["ridge_yu"], lambdas=np.asarray(lambdas), lags=lags,
             cell_mask=cell_cpu.numpy().astype(np.uint8), target_keep_index=keep_target,
             source_keep_index=model.source_gate.keep_index.cpu().numpy(),
             n_fit_rows_per_tau=full.n_rows.numpy(),
             layout=np.asarray("ridge_*: features = [target lags 0..P-1 (lag-major, 76 channels each), 1, "
                               "source lags 0..P-1 (42 channels each)]; ridge_y uses the first "
                               "76*P+1; own_ar: features = [own lags 0..P-1, 1]"))
    del halves, full

    # Pass 2: score every anchor of every code.
    frames: List[pd.DataFrame] = []
    for rows in batches(task, cfg, dev, model, anchor_stride, lags, cell_mask, limit, "score"):
        frames.append(score_rows(rows, weights, width_y))
    table = pd.concat(frames, ignore_index=True)
    table["cohort"] = table["guid"].map(S.cohort_of_guids())
    table["code"] = table["guid"].map(S.guid_hash).astype(int)
    lead = ["guid", "epoch", "anchor", "cohort", "code", "n_st", "n_ph"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    table["epoch"] = table["epoch"].astype(float)
    table["anchor"] = table["anchor"].astype(int)

    paths = {
        "baselines_fit.parquet": table["code"] == S.FIT_CODE,
        "baselines_select.parquet": table["code"] == S.SELECT_CODE,
        "baselines_test_SEALED.parquet": table["code"].isin(S.TEST_CODES),
    }
    for name, pick in paths.items():
        table[pick].reset_index(drop=True).to_parquet(out / name, index=False)
        print("wrote", out / name)
    print("wrote", out / "baseline_weights.npz")
    summarise(table[table["code"] == S.FIT_CODE], S.FIT_CODE)
    summarise(table[table["code"] == S.SELECT_CODE], S.SELECT_CODE)
    print(f"\nruntime {time.time() - start:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return choice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--limit", type=int, default=RUN_ARGS["limit"])
    parser.add_argument("--anchor-stride", dest="anchor_stride", type=int, default=RUN_ARGS["anchor_stride"])
    parser.add_argument("--lags", type=int, default=RUN_ARGS["lags"])
    parser.add_argument("--lambdas", type=float, nargs="+", default=RUN_ARGS["lambdas"])
    parser.add_argument("--out-dir", dest="out_dir", default=RUN_ARGS["out_dir"])
    main(**vars(parser.parse_args()))
