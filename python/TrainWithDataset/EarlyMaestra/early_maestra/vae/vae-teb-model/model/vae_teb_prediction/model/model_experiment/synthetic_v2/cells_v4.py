r"""S1-T01: the concentrated cell grid for ``synthetic_v4``.

The concentrated recipe (memory: G1_mix / concentrated-ladder) plants a **recoverable** per-step
TE by using a single pathway ($M=1$), a short identifiable fixed lag $D$ (default $D=8$, inside the
$H=30$ horizon so the true past-source band $\mathcal L^\star = \{\max(0,D-H),\dots,D-1\}$ lives
fully inside the forecast window), and a TE ladder $\mathrm{TE}_{\mathrm{inj}} \in
\{0,0.5,1,2,3\}$ block-nats with dedicated null cells ($B=0$). Each ladder level may be replicated
``cells_per_level`` times (independent generation seeds, same coupling $B$) so a level is
represented by several i.i.d. cells.

Coupling is solved **once per distinct** $(\mathrm{TE}_{\mathrm{inj}}, D)$ by the reused analytic
inverter (:func:`solve_cell_coupling` -> :func:`B_y_for_mean_te_block_state_space`), so the label
is exact. Only ``lag_mode: fixed`` is supported in the first cut; band lag is deferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    solve_cell_coupling,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellV4:
    r"""One cell of the concentrated single-pathway v4 grid -- $(\mathrm{TE}_{\mathrm{inj}}, D)$.

    Attributes:
        cell_id: Stable index within the pool (contiguous over kept cells, null cells included).
        target_te: The *requested* injected block TE in nats ($\ge 0$; $0$ is a null anchor).
        D: The fixed source$\to$target lag in decimated steps ($\ge 1$).
        B_y_scalar: The solved coupling $B$ (``0`` for a null cell).
        te_block_realised: The *achieved* block TE at ``B_y_scalar`` -- the exact
            $\mathrm{TE}_{\mathrm{inj}}$ label (``0`` for a null).
        is_null: Whether this is a null ($B=0$) cell.
        level_index: Index of the $(\mathrm{TE}_{\mathrm{inj}}, D)$ level in enumeration order.
        replicate: Which i.i.d. copy of the level this cell is (``0..cells_per_level-1``).
    """

    cell_id: int
    target_te: float
    D: int
    B_y_scalar: float
    te_block_realised: float
    is_null: bool
    level_index: int = 0
    replicate: int = 0


def enumerate_cells_v4(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    target_te_grid: Optional[Sequence[float]] = None,
    lag_grid: Optional[Sequence[int]] = None,
    cells_per_level: Optional[int] = None,
) -> Tuple[List[CellV4], List[Dict[str, Any]]]:
    r"""Enumerate and solve the concentrated cell grid (fixed lag only).

    Crosses ``target_te_grid`` with ``lag_grid`` (each a fixed $D$), replicating every level
    ``cells_per_level`` times. Null cells ($\mathrm{TE}_{\mathrm{inj}}=0$) are kept without invoking
    the inverter ($B=0$). Signal cells solve the coupling once per distinct $(\mathrm{TE}, D)$ (the
    replicas share $B$); a cell whose bracket misses the target is **logged and dropped** (collected
    in ``dropped``) rather than aborting. ``cell_id`` is contiguous over kept cells.

    Args:
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        target_te_grid: Override for ``mix.target_te_grid`` (e.g. a pilot grid).
        lag_grid: Override for ``mix.lag_grid``.
        cells_per_level: Override for ``mix.cells_per_level`` (default $1$).

    Returns:
        ``(cells, dropped)`` -- the kept :class:`CellV4` list and a list of
        ``{'target_te', 'D', 'reason'}`` dicts for unsolvable cells.

    Raises:
        ValueError: If ``mix.lag_mode`` is not ``fixed`` (band lag is deferred).
    """
    bench = config["benchmarks"][benchmark]
    mix = bench["mix"]
    lag_mode = str(mix.get("lag_mode", "fixed"))
    if lag_mode != "fixed":
        raise ValueError(
            f"enumerate_cells_v4: only lag_mode='fixed' is supported in the first cut, "
            f"got {lag_mode!r} (band lag is deferred)."
        )

    te_grid = list(mix["target_te_grid"] if target_te_grid is None else target_te_grid)
    lags = list(mix["lag_grid"] if lag_grid is None else lag_grid)
    reps = int(mix.get("cells_per_level", 1) if cells_per_level is None else cells_per_level)
    if reps < 1:
        raise ValueError(f"enumerate_cells_v4: cells_per_level must be >= 1, got {reps}.")

    cells: List[CellV4] = []
    dropped: List[Dict[str, Any]] = []
    solved_cache: Dict[Tuple[float, int], Dict[str, Any]] = {}
    next_id = 0
    level_index = 0

    for target_te in te_grid:
        target_te = float(target_te)
        for D in lags:
            D = int(D)
            is_null = target_te == 0.0
            if is_null:
                b_scalar, te_real = 0.0, 0.0
            else:
                key = (round(target_te, 9), D)
                solution = solved_cache.get(key)
                if solution is None:
                    try:
                        solution = solve_cell_coupling(config, target_te, D, benchmark=benchmark)
                    except ValueError as exc:
                        logger.warning(
                            "enumerate_cells_v4: dropping unsolvable cell "
                            "(target_te=%g, D=%d): %s", target_te, D, exc,
                        )
                        dropped.append({"target_te": target_te, "D": D, "reason": str(exc)})
                        continue
                    solved_cache[key] = solution
                b_scalar = float(solution["B_y_scalar"])
                te_real = float(solution["te_block"])

            for r in range(reps):
                cells.append(
                    CellV4(
                        cell_id=next_id, target_te=target_te, D=D,
                        B_y_scalar=b_scalar, te_block_realised=te_real,
                        is_null=is_null, level_index=level_index, replicate=r,
                    )
                )
                next_id += 1
            level_index += 1

    logger.info(
        "enumerate_cells_v4: %d cells kept (%d null, %d signal), %d dropped; cells_per_level=%d.",
        len(cells),
        sum(1 for c in cells if c.is_null),
        sum(1 for c in cells if not c.is_null),
        len(dropped), reps,
    )
    return cells, dropped
