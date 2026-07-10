r"""Sprint 7 (goal G-G): a neural CMI estimate, calibrated in absolute nats.

Everything else in this pipeline grades the model's own KL surrogate $\bar K$ against the injected
transfer entropy. That is a *self-report*: $\bar K$ is a quantity the model computes, so a
calibration $\bar K = \alpha + \gamma\,\mathrm{TE}$ tells us the surrogate tracks the truth, not
that the source is genuinely informative about the target's future. This module estimates the
underlying conditional mutual information

$$I\bigl(U_{\le t};\, Y^{+}_{t,1:H} \,\bigm|\, c_t\bigr)$$

directly, with a critic that never sees $\bar K$.

Three estimator configurations
------------------------------

The spec asks for two *conditioning sets* -- the ground-truth latent history and the model's own
``target_state`` -- with $U$ and $Y^{+}$ held fixed, so that

$$\mathrm{bias} \;=\; \mathrm{CMI}(\texttt{target\_state}) \;-\; \mathrm{CMI}(\texttt{ground\_truth})$$

isolates the conditioning summary. But if $U$ and $Y^{+}$ are the *scattering features*, the
resulting nats are comparable to $\mathrm{TE}_{\mathrm{scat}}$, not to $\mathrm{TE}_{\mathrm{inj}}$
-- and S1-T05 established that $\mathrm{TE}_{\mathrm{scat}}$ is ordinal *within a fixed lag only*,
inflated $2.4\text{--}5.9\times$, with an absolute scale set by the probe's free ridge. An
absolute-nats claim measured there would be meaningless.

So there are **three** configurations, not two:

======================  ==========================  ==========================  ====================
config                  $U_{\le t}$                 $Y^{+}_{t,1:H}$             $c_t$
======================  ==========================  ==========================  ====================
``latent``              GT source latent $c$ window  GT target latent $d$ future  GT $d$ history
``feature_gt``          ``u_stream`` window          ``build_future_target``      GT $d$ history
``feature_model``       *same as* ``feature_gt``     *same as* ``feature_gt``     ``target_state[t]``
======================  ==========================  ==========================  ====================

``latent`` lives entirely in the data-generating process's own coordinates, so it is directly
comparable to $\mathrm{TE}_{\mathrm{inj}}$ in **nats** and is where the estimator validates itself
(S7-T03). ``feature_gt`` $\to$ ``feature_model`` changes *only* $c_t$, so ``bias`` means what the
spec says it means. Neither ``latent`` nor ``feature_gt`` touches the checkpoint at all -- the
absolute-nats recovery check is therefore **model-free**, and holds even on a pilot checkpoint whose
source pathway never switched on.

How the conditioning enters, and two estimators that do not work
---------------------------------------------------------------

The obvious estimator -- one critic $f(u, [y^{+}, c])$ with negatives drawn by permuting $u$ within
the contrastive batch -- does **not** estimate the CMI. Permuting $u$ breaks its pairing with the
*whole* of $(y^{+}, c)$, so the negatives come from $p(u)\,p(y^{+}, c)$ and the bound converges to
$I\bigl(U; (Y^{+}, C)\bigr)$. On dependent draws that is inflated by $I(U; C)$: the source's own
past is correlated with the target's past through the very coupling being measured.

The textbook repair is the CCMI construction (Mukherjee et al.), which uses the chain rule
$I(U; (Y^{+}, C)) = I(U; C) + I(U; Y^{+} \mid C)$ to write the CMI as a difference of two InfoNCE
bounds. **That does not work here either, and it was measured failing.** On the Gaussian recovery
ladder both terms saturate against their common $\log K$ ceiling long before the difference becomes
informative -- at $B = 1.2$, $\hat I(U;(Y^{+},C)) = 3.20$ and $\hat I(U;C) = 2.79$ against a ceiling
of $\log 32 = 3.466$ -- so the difference is a difference of two clipped numbers and comes out
*non-monotone* in the coupling ($0.39, 0.46, 0.48, 0.35, 0.41$ as the true TE climbs
$0.82 \to 2.56$). The nuisance term $I(U;C)$ is simply far larger than the CMI being extracted.

What this module actually does
------------------------------

Condition by **residualisation**, then run a *single* InfoNCE on the residuals:

$$\tilde U \;=\; U - \hat{\mathbb{E}}_{\mathrm{lin}}[U \mid c_t], \qquad
  \tilde Y^{+} \;=\; Y^{+} - \hat{\mathbb{E}}_{\mathrm{lin}}[Y^{+} \mid c_t], \qquad
  \widehat{\mathrm{CMI}} \;=\; \hat I_{\mathrm{NCE}}\bigl(\tilde U;\, \tilde Y^{+}\bigr),$$

$$\hat I_{\mathrm{NCE}}(X; Z) \;=\; \mathbb{E}\Bigl[f(x_i, z_i)
   - \log \tfrac{1}{K}\textstyle\sum_{j=1}^{K} e^{f(x_j, z_i)}\Bigr] \;\le\; \log K,$$

with the regression coefficients fitted on the fit partition only. For jointly Gaussian
$(U, Y^{+}, C)$ the residuals are independent of $C$ and carry exactly the partial covariance, so
$I(\tilde U; \tilde Y^{+}) = I(U; Y^{+} \mid C)$ **exactly** -- and the ``latent`` configuration *is*
jointly Gaussian, by construction of the data-generating process. That is what licenses the
absolute-nats claim, and it is why the claim is made there and nowhere else.

The estimate is now the CMI itself, so the $\log K$ ceiling applies to it directly: at
$K = 32$ the ceiling is $3.47$ nats, just above the $\mathrm{TE}_{\mathrm{inj}} = 3.0$ cell. It is a
**lower bound**, and like every InfoNCE bound it loosens as the target approaches the ceiling.
Measured on the Gaussian ladder against the exact closed form: ratio $0.89$ at
$\mathrm{TE} = 0.82$, $0.84$ at $1.71$, $0.75$ at $2.56$. On an independent pair ($B = 0$) it reads
$-0.027$ -- a *lower bound on zero* estimated from finite data fluctuates about zero, which is why
S7-T03's null acceptance is $\lvert\widehat{\mathrm{CMI}}\rvert < 0.05$ rather than
$\widehat{\mathrm{CMI}} \ge 0$.

For the two ``feature_*`` configurations the residualisation removes only the **linear** dependence
on $c_t$, so $\hat I$ is a conditional-linear surrogate rather than a strict CMI. This is why those
two configurations carry only rank-level claims -- which is what the spec already demanded of
``feature_model`` for an unrelated reason (its conditioning is the model's own state).
:func:`gaussian_cmi_closed_form` provides the exact reference in the Gaussian case, and the
``latent`` config reports it alongside the neural estimate so the estimator stays auditable.

Five implementation facts drive the shape of this module
--------------------------------------------------------

* **Grid alignment is load-bearing.** ``raw_generators.generate_cell_raw`` simulates the coupled
  latents $(c, d)$ on the decimated grid of length $T_{\mathrm{tot}} = n_{\mathrm{raw}}/16 = 330$,
  but the cached features and ``target_state`` occupy the trimmed analysis window of length
  $300$, i.e. ``[TRIM_STEPS : T_tot - TRIM_STEPS]``. So :func:`make_latent_provider` crops by
  ``TRIM_STEPS`` **in decimated steps** -- not in raw samples, as ``make_raw_provider`` does --
  giving $t_{\mathrm{model}} = t_{\mathrm{latent}} - 15$. With true lags $D \in \{8, 12, 20\}$, a
  15-step misalignment exceeds the smallest lag and would silently invalidate every nats claim.

* **``runner.inference_mode()`` poisons autograd.** It is ``torch.inference_mode()``; tensors
  created inside are *inference tensors* and can never participate in a backward pass, even after
  ``.detach()`` or ``.clone()``. The critic needs gradients. :func:`collect_cmi_anchors` is
  therefore the only thing inside that block, and it exports every anchor through
  ``.cpu().numpy()``; every fit in :func:`run_cmi_comparison` happens outside it.

* **The InfoNCE ceiling is $\log K$, and it brackets the top cell.** At the shipped
  ``contrastive_batch = 32`` the ceiling is $\log 32 \approx 3.47$ nats, which sits just above the
  $\mathrm{TE}_{\mathrm{inj}} = 3.0$ cell. No absolute-nats claim is made for any cell whose
  injected TE exceeds ``ceiling_claim_frac`` $\times$ ceiling (default $0.8 \times 3.47 = 2.77$);
  those cells are stamped ``near_ceiling`` and the figure draws the ceiling line explicitly.

* **Early stopping is what makes the estimate non-vacuous**, not an optimisation nicety. Trained to
  convergence on a few thousand *correlated* anchors, the critic memorises the pairing and its
  held-out bound collapses -- measured at $-22$ nats on a $\mathrm{TE}_{\mathrm{inj}} = 0$ cell
  whose true CMI is $0$. Valid, and useless. See :meth:`NeuralCMIEstimator._train`.

* **The feature configs need the coupled-channel reduction.** At ``u_channels: all`` the source
  anchor is $24 \times 101 = 2424$-dimensional and the future target $30 \times 87 = 2610$; the
  bound reads $-5$ to $-17$ nats. Slicing to ``meta.coupled_channel`` -- the same slice
  ``eval_v2.measure_te_scat`` makes -- restores it. See :func:`_channel_index`.

Reading the numbers on a pilot: ``cmi_latent`` should track $\mathrm{TE}_{\mathrm{inj}}$ regardless
of training, because it never reads the model. ``cmi_feature_model``, ``bias`` and
$\rho(\bar K, \mathrm{CMI})$ will be near-zero on a 400-step pilot whose source pathway never
switched on -- a training-progress signal, not a verdict (Section 11).

One readout is **not** a training-progress signal. ``cond_r2_v``, the held-out $R^2$ of $Y^{+}$ on
the conditioning summary, is negative on every cell of a ``causal_norm: false`` arm and positive on
every cell of ``v3_prod``, because a time-pooling ``GroupNorm`` makes ``target_state[b, t]`` carry
per-sample statistics that cannot transfer across the by-sample fit/eval split. That is the G0 leak,
seen without ever reading the model's KL -- and it is why ``cmi_bias`` is flagged unreliable on
those arms (:func:`_bias_reliability`).
"""
from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v2 import (
    build_u_stream,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
    _bootstrap_ci,
    _clean_window_mean,
    _jsonable,
    _spearman_finite,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
    SectionContext,
    SectionSpec,
    _fmt,
    register_section,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
    StageContext,
    StageSpec,
    _build_runner_and_loader,
    _split_dir,
    register_stage,
)

_STAGE_ORDER = 14
_SECTION_ORDER = 40

#: The three estimator configurations, in report order. See the module docstring.
CONFIG_LATENT = "latent"
CONFIG_FEATURE_GT = "feature_gt"
CONFIG_FEATURE_MODEL = "feature_model"
_ALL_CONFIGS: Tuple[str, ...] = (CONFIG_LATENT, CONFIG_FEATURE_GT, CONFIG_FEATURE_MODEL)

#: Configurations that require the ground-truth latent provider. When it cannot be built the
#: stage drops these and still runs ``feature_model``.
_LATENT_DEPENDENT: Tuple[str, ...] = (CONFIG_LATENT, CONFIG_FEATURE_GT)

#: Bootstrap resamples for every CMI confidence interval (cluster bootstrap over samples).
_N_BOOT_CMI: int = 2000

#: Below this many cells a rank correlation over cells is reported, never gated. Matches
#: ``lag_intervention_v3._MIN_CELLS_FOR_RHO``.
_MIN_CELLS_FOR_RHO: int = 8

#: S7-T03 recovery thresholds. Literal starting values, revisable; a test may not assert against
#: an undefined "documented factor", so these are named constants and the test imports them.
_RECOVER_MIN_DEPENDENT: float = 0.30
_RECOVER_MAX_INDEPENDENT: float = 0.05
_RECOVER_MIN_SPEARMAN: float = 0.80
_RECOVER_FACTOR: float = 2.0
_RECOVER_TE_MAX_FOR_ABSOLUTE: float = 2.0


# ---------------------------------------------------------------------------
# S7-T01: deterministic ground-truth latent provider, with the decimated-grid crop
# ---------------------------------------------------------------------------
def make_latent_provider(
    config: Dict[str, Any],
    split: str,
    *,
    benchmark: str = "G1_raw",
    cache_dir: Optional[Path] = None,
) -> Callable[[int, int], Tuple[np.ndarray, np.ndarray]]:
    r"""Build a memoised ground-truth-latent regenerator keyed by ``(cell_id, raw_index)``.

    The cache stores only the $300$-step scattering features; the coupled latents $(c, d)$ that
    ``raw_generators.generate_cell_raw`` simulates under ``raw["latents"]`` are never persisted (the
    ``.npz`` schema is frozen). Because generation is fully seed-deterministic and each cached row
    stamps its within-cell index (``sample_raw_index``), the exact latent pair for any shuffled cache
    row is regenerable on demand -- exactly as :func:`build_dataset_v2.make_raw_provider` regenerates
    the raw waveforms, and through the identical :func:`build_dataset_v2.generate_pilot_samples`
    call, so the two providers cannot drift.

    **The crop is in decimated steps, not raw samples.** This is the one substantive difference from
    :func:`~build_dataset_v2.make_raw_provider`, and it is load-bearing. The latents live on the
    decimated grid of length $T_{\mathrm{tot}} = n_{\mathrm{raw}} / 16 = 330$; the model's features
    and ``target_state`` occupy the trimmed window
    ``[TRIM_STEPS : T_tot - TRIM_STEPS]`` of length ``sequence_length`` $= 300$. The raw provider
    slices $\bigl[15 \cdot 16,\; 5280 - 15 \cdot 16\bigr)$ because its samples are on the $4\,$Hz
    grid; this provider slices $[15, 315)$. Cropping in the wrong units would shift the conditioning
    by $15$ steps -- more than the smallest true lag $D = 8$ -- and silently invalidate every
    absolute-nats claim.

    **Fully cache-authoritative.** ``generate_cell_raw`` scales the AM envelope by the *batch-pooled*
    latent standard deviation, so a row's latents depend on the ``n`` passed to it. The per-cell
    ``n`` therefore comes from the cache's own row counts (the ``sample_cell_id`` bincount), and the
    solved $B$ / seeds / render mode / ``n_raw`` from ``meta.json`` -- never from the live config,
    which supplies only the generation-invariant latent physics.

    Args:
        config: The parsed ``config_synth_v3.yaml`` tree.
        split: The cache split the rows come from (``train`` / ``val`` / ``test``); selects the
            generation seed offset and the split's ``.npz``.
        benchmark: Active benchmark key under ``benchmarks``.
        cache_dir: Override for the directory holding ``meta.json`` + ``<split>.npz`` (defaults to
            :func:`build_dataset_v2.resolve_cache_dir`).

    Returns:
        A **total** callable ``provider(cell_id, raw_index) -> (c_win, d_win)`` returning two 1-D
        ``float64`` arrays of length ``sequence_length``: the source latent $c$ (AR(2) contraction
        strength) and the target latent $d$ (AR(1) deceleration depth, coupled as
        $d_k = A_y d_{k-1} + B\,c_{k-D} + \varepsilon_k$). On any failure (unknown cell,
        out-of-range row, regeneration error) it returns NaN windows rather than raising, so a caller
        can skip those anchors without special-casing. The function carries a ``window_length``
        attribute.

    Raises:
        FileNotFoundError: When ``meta.json`` or ``<split>.npz`` is absent.
        KeyError: When the split ``.npz`` lacks ``sample_cell_id``.
        ValueError: When ``meta.json`` has no cells, or the decimated grid geometry is inconsistent
            with ``TRIM_STEPS`` (the alignment assertion).
    """
    # Local imports: ``scattering_adapter`` pulls kymatio, and ``build_dataset_v2`` pulls the
    # generators. Neither belongs at module import time -- the fast unit tests must not need them.
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        CellV2,
        generate_pilot_samples,
        resolve_cache_dir,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (
        DECIMATION,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.scattering_adapter import (
        TRIM_STEPS,
    )

    cdir = resolve_cache_dir(config, benchmark=benchmark) if cache_dir is None else Path(cache_dir)

    meta_path = cdir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"make_latent_provider: no meta.json under {cdir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    cells_by_id: Dict[int, CellV2] = {}
    for c in meta.get("cells", []) or []:
        cid = int(c["cell_id"])
        cells_by_id[cid] = CellV2(
            cell_id=cid, target_te=float(c.get("target_te", 0.0)), D=int(c["D"]),
            B_y_scalar=float(c.get("B_y_scalar", 0.0)),
            te_block_realised=float(c.get("te_block_realised", c.get("te_inj", 0.0))),
        )
    if not cells_by_id:
        raise ValueError(f"make_latent_provider: meta.json under {cdir} has no cells")

    seeds_meta = meta.get("seeds", {}) or {}
    base_seed = int(seeds_meta.get("dgp", seeds_meta.get("base_seed", 0)))
    render_mode = meta.get("render_mode")
    n_raw = int(((meta.get("raw") or {}).get("n_raw"))
                or config["benchmarks"][benchmark]["raw"]["n_raw"])
    seq_len = int(meta.get("sequence_length")
                  or config["benchmarks"][benchmark]["data"]["sequence_length"])

    # --- the alignment invariant, asserted rather than assumed ---------------------------
    if n_raw % int(DECIMATION) != 0:
        raise ValueError(
            f"make_latent_provider: n_raw={n_raw} is not a multiple of DECIMATION={DECIMATION}"
        )
    t_tot = n_raw // int(DECIMATION)
    if t_tot - 2 * int(TRIM_STEPS) != seq_len:
        raise ValueError(
            f"make_latent_provider: decimated grid T_tot={t_tot} minus 2*TRIM_STEPS="
            f"{2 * int(TRIM_STEPS)} is {t_tot - 2 * int(TRIM_STEPS)}, not sequence_length="
            f"{seq_len}. The latent crop would be misaligned with the model's window."
        )
    assert int(TRIM_STEPS) == (t_tot - seq_len) // 2, "t_model = t_latent - TRIM_STEPS"
    crop = slice(int(TRIM_STEPS), t_tot - int(TRIM_STEPS))

    # Per-cell sample count straight from the cache: the pooled-std AM amplitude (and hence the
    # latent scaling) depends on the ``n`` the build passed to ``generate_cell_raw``.
    split_npz = cdir / f"{split}.npz"
    if not split_npz.is_file():
        raise FileNotFoundError(f"make_latent_provider: no {split}.npz under {cdir}")
    with np.load(split_npz) as npz:
        if "sample_cell_id" not in npz.files:
            raise KeyError(f"make_latent_provider: {split}.npz lacks sample_cell_id")
        cell_ids = np.asarray(npz["sample_cell_id"]).astype(np.int64)
    n_by_cell = {int(cid): int(np.count_nonzero(cell_ids == cid))
                 for cid in np.unique(cell_ids)}

    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    nan_win = np.full(seq_len, np.nan, dtype=np.float64)

    def provider(cell_id: int, raw_index: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return the cropped $(c, d)$ latent pair for one row; NaN windows on any failure."""
        cid, ri = int(cell_id), int(raw_index)
        try:
            if cid not in cells_by_id or cid not in n_by_cell:
                raise KeyError(f"unknown cell_id {cid}")
            if cid not in cache:
                raw = generate_pilot_samples(
                    cells_by_id[cid], int(n_by_cell[cid]), split, config,
                    benchmark=benchmark, base_seed=base_seed, render_mode=render_mode,
                )
                latents = raw["latents"]
                cache[cid] = (
                    np.ascontiguousarray(latents["c"][:, crop], dtype=np.float64),
                    np.ascontiguousarray(latents["d"][:, crop], dtype=np.float64),
                )
            c_all, d_all = cache[cid]
            if not 0 <= ri < c_all.shape[0]:
                raise IndexError(f"raw_index {ri} out of range for cell {cid} (n={c_all.shape[0]})")
            return c_all[ri].copy(), d_all[ri].copy()
        except Exception as exc:  # noqa: BLE001 -- a missing anchor is skipped, never fatal
            logger.warning(
                "make_latent_provider: latents unavailable for (cell={}, row={}): {}", cid, ri, exc
            )
            return nan_win.copy(), nan_win.copy()

    provider.window_length = seq_len  # type: ignore[attr-defined]
    return provider


# ---------------------------------------------------------------------------
# S7-T02: the InfoNCE critic and the CCMI difference estimator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CMIConfig:
    r"""Capacity, optimisation and inference knobs for :class:`NeuralCMIEstimator`.

    Attributes:
        critic_width: Hidden width of each critic MLP.
        critic_depth: Number of hidden layers in each branch encoder.
        embed_dim: Dimension the two branches project to before the score head. Keeps the
            score head small when $U$ or $Y^{+}$ is high-dimensional.
        contrastive_batch: The number of candidates $K$ in each InfoNCE softmax. **This sets the
            ceiling**: $\hat I_{\mathrm{NCE}} \le \log K$. At the shipped $K = 32$ the ceiling is
            $\approx 3.47$ nats, which sits just above the $\mathrm{TE}_{\mathrm{inj}} = 3.0$ cell.
        n_iters: Gradient-step **budget**. Early stopping normally ends the fit far short of it,
            so this is a cap rather than a tuning knob.
        lr: Adam learning rate.
        fit_frac: Fraction of *samples* (not anchors) used to fit; the rest is held out for the
            reported estimate and its bootstrap.
        val_frac: Fraction of the *fit* samples carved off as an early-stopping validation slice.
            Without it the critic memorises the fit anchors and the held-out InfoNCE collapses --
            measured at $-22$ nats on a null cell after 3000 unstopped steps.
        eval_every: Validation cadence, in gradient steps.
        patience: Consecutive non-improving validations before the fit stops. The best-scoring
            critic state is restored.
        n_boot: Block-bootstrap resamples of the held-out contrastive blocks.
        residual_ridge: Penalty for the $x \sim c$ regression, as a fraction of the Gram diagonal's
            mean. Keeps the residualisation well-posed when $c_t$ is high-dimensional (the
            ``feature_model`` conditioning is ``d_h``-dimensional) relative to the anchor count.
        ceiling_claim_frac: A cell whose $\mathrm{TE}_{\mathrm{inj}}$ exceeds this fraction of the
            ceiling is stamped ``near_ceiling`` and carries no absolute-nats claim.
        seed: Seeds torch and numpy; two runs at one seed agree bitwise.
        device: ``"cpu"`` or ``"cuda"``.
    """

    critic_width: int = 64
    critic_depth: int = 1
    embed_dim: int = 16
    contrastive_batch: int = 32
    n_iters: int = 6000
    lr: float = 1.0e-3
    fit_frac: float = 0.5
    val_frac: float = 0.25
    eval_every: int = 50
    patience: int = 15
    n_boot: int = _N_BOOT_CMI
    residual_ridge: float = 1.0e-6
    ceiling_claim_frac: float = 0.8
    seed: int = 0
    device: str = "cpu"


class _Critic(torch.nn.Module):
    r"""A separable InfoNCE critic $f(x, z) = \langle g_\theta(x),\, h_\phi(z) \rangle / \sqrt{e}$.

    Two MLP encoders project $x$ and $z$ into a shared ``embed_dim`` space; the score is their
    scaled inner product. Separability is what makes the $K \times K$ score matrix a single
    matrix multiply, so the full contrastive softmax costs one ``matmul`` per step rather than
    $K^{2}$ forward passes.

    Args:
        x_dim: Dimension of the first variable.
        z_dim: Dimension of the second variable.
        cfg: The estimator config (supplies width, depth and ``embed_dim``).
    """

    def __init__(self, x_dim: int, z_dim: int, cfg: CMIConfig) -> None:
        super().__init__()
        self.g = self._mlp(x_dim, cfg)
        self.h = self._mlp(z_dim, cfg)
        self._scale = float(math.sqrt(cfg.embed_dim))

    @staticmethod
    def _mlp(in_dim: int, cfg: CMIConfig) -> torch.nn.Module:
        r"""``critic_depth`` ReLU layers of width ``critic_width``, then a linear to ``embed_dim``."""
        layers: List[torch.nn.Module] = []
        d = int(in_dim)
        for _ in range(max(1, int(cfg.critic_depth))):
            layers += [torch.nn.Linear(d, int(cfg.critic_width)), torch.nn.ReLU()]
            d = int(cfg.critic_width)
        layers.append(torch.nn.Linear(d, int(cfg.embed_dim)))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""Score matrix $(B_x, B_z)$ of every $x_i$ against every $z_j$."""
        return (self.g(x) @ self.h(z).T) / self._scale


def infonce_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    r"""The InfoNCE lower bound from a square score matrix whose diagonal is the positive pair.

    $$\hat I_{\mathrm{NCE}} \;=\; \frac{1}{K}\sum_{i=1}^{K}
        \Bigl[ f(x_i, z_i) \;-\; \log \frac{1}{K} \sum_{j=1}^{K} e^{f(x_j, z_i)} \Bigr]
        \;\le\; \log K .$$

    The inner sum runs over the *first* index (candidate $x$'s against a fixed $z_i$), which is the
    "permute $U$" convention: negatives are other rows' $x$ paired with this row's $z$.

    Args:
        scores: A $(K, K)$ tensor with ``scores[i, j] = f(x_i, z_j)``.

    Returns:
        A scalar tensor holding the bound, in nats.
    """
    k = scores.shape[0]
    positives = scores.diagonal()
    # logsumexp down the columns: for each z_j, marginalise over candidate x_i.
    log_denominator = torch.logsumexp(scores, dim=0) - math.log(k)
    return (positives - log_denominator).mean()


def _ridge_lstsq(design: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    r"""Ridge-stabilised least squares, penalty relative to the Gram diagonal's mean.

    Args:
        design: $(n, p)$ regressors.
        targets: $(n, q)$ responses.
        ridge: Penalty as a fraction of ``mean(diag(design.T @ design))``.

    Returns:
        The $(p, q)$ coefficient matrix.
    """
    gram = design.T @ design
    scale = float(np.mean(np.diag(gram))) if gram.size else 1.0
    gram = gram + float(ridge) * max(scale, 1e-12) * np.eye(gram.shape[0])
    return np.linalg.solve(gram, design.T @ targets)


def gaussian_cmi_closed_form(u: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    r"""Exact $I(U; V \mid C)$ for jointly Gaussian variables, from residual covariances.

    $$I(U; V \mid C) \;=\; \tfrac12 \Bigl(
        \ln\det \Sigma_{\tilde U} + \ln\det \Sigma_{\tilde V}
        - \ln\det \Sigma_{[\tilde U, \tilde V]} \Bigr),$$

    with $\tilde U, \tilde V$ the OLS residuals of $U, V$ on $C$. Used as the audit reference for
    the ``latent`` configuration, whose variables are Gaussian by construction. On the Sprint-7
    recovery ladder this agrees with the independent Monte-Carlo block TE
    (:func:`analytic_te.te_block_state_space_gaussian`) to within $1\text{--}2\%$, which is what
    validates the anchor windowing in :func:`latent_anchor_rows`.

    Args:
        u: $(N, u_{\dim})$ source anchors.
        v: $(N, v_{\dim})$ future-target anchors.
        c: $(N, c_{\dim})$ conditioning anchors.

    Returns:
        The conditional mutual information in nats, or ``nan`` when a covariance is singular.
    """
    design = np.column_stack([np.asarray(c, dtype=np.float64), np.ones(len(c))])
    resid_u = u - design @ _ridge_lstsq(design, np.asarray(u, dtype=np.float64), 1e-9)
    resid_v = v - design @ _ridge_lstsq(design, np.asarray(v, dtype=np.float64), 1e-9)

    def _logdet(x: np.ndarray) -> float:
        cov = np.atleast_2d(np.cov(x, rowvar=False))
        sign, val = np.linalg.slogdet(cov)
        return float(val) if sign > 0 else float("nan")

    return 0.5 * (
        _logdet(resid_u) + _logdet(resid_v) - _logdet(np.column_stack([resid_u, resid_v]))
    )


class NeuralCMIEstimator:
    r"""Residualise on $c_t$, then bound $I(\tilde U; \tilde Y^{+})$ with a single InfoNCE critic.

    $$\tilde U = U - \hat{\mathbb{E}}_{\mathrm{lin}}[U \mid c_t], \quad
      \tilde Y^{+} = Y^{+} - \hat{\mathbb{E}}_{\mathrm{lin}}[Y^{+} \mid c_t], \quad
      \widehat{\mathrm{CMI}} = \hat I_{\mathrm{NCE}}(\tilde U;\, \tilde Y^{+}) \;\le\; \log K .$$

    For jointly Gaussian $(U, Y^{+}, C)$ the residuals are independent of $C$ and carry exactly the
    partial covariance, so the bound targets $I(U; Y^{+} \mid C)$ *exactly*. That holds for the
    ``latent`` configuration by construction of the data-generating process, and is what licenses
    its absolute-nats claim. For the ``feature_*`` configurations only the linear dependence on
    $c_t$ is removed, so the quantity is a conditional-linear surrogate and carries rank-level
    claims only.

    The module docstring records why the two obvious alternatives were rejected -- a single critic
    over $(U, [Y^{+}, C])$ estimates the joint, and the CCMI difference of two InfoNCE bounds was
    measured saturating against its own ceiling on this data.

    .. warning::
        $\hat I_{\mathrm{NCE}}$ is a **lower bound**, and it loosens as the target approaches
        $\log K$. Measured against the exact closed form on the Gaussian ladder: ratio $0.89$ at
        $\mathrm{TE} = 0.82$ nats, $0.84$ at $1.71$, $0.75$ at $2.56$. On independent draws it
        fluctuates about zero and can be slightly negative.

    Two design choices matter for honesty:

    * **Fit and evaluate on disjoint samples.** The residual regression *and* the critic are fitted
      on the fit partition only; the reported value is held out. InfoNCE evaluated on its own
      training anchors is optimistically biased, and a residualisation fitted on the evaluation
      anchors would leak.
    * **The bootstrap resamples held-out contrastive blocks, not anchors.** Anchors from one sample
      share that sample's noise realisation. Blocks are drawn from held-out samples and the critic
      stays frozen, so no refit happens inside the bootstrap.

    Args:
        u_dim: Dimension of $U_{\le t}$.
        v_dim: Dimension of $Y^{+}_{t,1:H}$.
        c_dim: Dimension of the conditioning summary $c_t$.
        cfg: Capacity / optimisation knobs.
    """

    def __init__(self, u_dim: int, v_dim: int, c_dim: int, cfg: CMIConfig) -> None:
        self.cfg = cfg
        self.u_dim, self.v_dim, self.c_dim = int(u_dim), int(v_dim), int(c_dim)
        self.device = torch.device(cfg.device)
        torch.manual_seed(int(cfg.seed))
        self.critic = _Critic(self.u_dim, self.v_dim, cfg).to(self.device)
        self.trace: List[float] = []
        self.val_trace: List[float] = []

    # -- ceiling bookkeeping ------------------------------------------------------------
    @property
    def ceiling_nats(self) -> float:
        r"""$\log K$, the largest value the InfoNCE bound -- and hence the estimate -- can attain."""
        return float(math.log(int(self.cfg.contrastive_batch)))

    def near_ceiling(self, te: float) -> bool:
        r"""Whether an injected TE is close enough to $\log K$ that no nats claim may be made.

        Args:
            te: The cell's injected transfer entropy, in nats.

        Returns:
            ``True`` when ``te > ceiling_claim_frac * ceiling_nats``. At the shipped $K = 32$ and
            ``ceiling_claim_frac = 0.8`` the threshold is $2.77$ nats, so the
            $\mathrm{TE}_{\mathrm{inj}} = 3.0$ cell is flagged and the $2.0$ cell is not.
        """
        if not np.isfinite(te):
            return False
        return bool(float(te) > float(self.cfg.ceiling_claim_frac) * self.ceiling_nats)

    # -- internals ----------------------------------------------------------------------
    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        r"""Numpy -> a fresh float32 tensor on the estimator's device (never an inference tensor)."""
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=self.device)

    @staticmethod
    def _split_by_group(groups: np.ndarray, fit_frac: float,
                        rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        r"""Partition anchors by their *sample* id, so no sample straddles fit and eval."""
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_fit = max(1, int(round(len(uniq) * float(fit_frac))))
        if n_fit >= len(uniq):
            n_fit = max(1, len(uniq) - 1)
        fit_groups = set(uniq[:n_fit].tolist())
        is_fit = np.fromiter((g in fit_groups for g in groups), dtype=bool, count=len(groups))
        return np.where(is_fit)[0], np.where(~is_fit)[0]

    def _residualise(
        self, x: np.ndarray, c: np.ndarray, fit_idx: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        r"""Remove $\hat{\mathbb{E}}_{\mathrm{lin}}[x \mid c]$, coefficients from ``fit_idx`` only.

        Returns:
            ``(residual, r2)`` where ``r2`` is the fraction of ``x``'s total variance the
            conditioning explained on the held-out anchors -- a diagnostic for how strong the
            conditioning summary is.
        """
        design = np.column_stack([c, np.ones(len(c))])
        beta = _ridge_lstsq(design[fit_idx], x[fit_idx], float(self.cfg.residual_ridge))
        resid = x - design @ beta
        total = float(np.var(x, axis=0).sum())
        left = float(np.var(resid, axis=0).sum())
        r2 = float(1.0 - left / total) if total > 1e-12 else float("nan")
        return resid, r2

    def _train(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        train_idx: np.ndarray,
        val_blocks: Sequence[np.ndarray],
        generator: torch.Generator,
    ) -> Dict[str, Any]:
        r"""Maximise the InfoNCE bound, early-stopping on a held-out slice of the fit partition.

        **Early stopping is not an optimisation nicety here; it is what makes the estimate
        non-vacuous.** An InfoNCE critic trained to convergence on a few thousand correlated
        anchors memorises the pairing, and its *held-out* bound then collapses far below zero --
        measured at $-22$ nats on a $\mathrm{TE}_{\mathrm{inj}} = 0$ cell after 3000 unstopped
        steps, against a true CMI of $0$. Such a bound is still valid (any critic gives a lower
        bound) but it is useless. Restoring the critic that scored best on the validation slice
        brings the null cells back to $\lvert \hat I \rvert < 0.05$.

        Args:
            x: All residualised source anchors, on device.
            z: All residualised future-target anchors, on device.
            train_idx: Anchor indices the critic may fit on.
            val_blocks: Contrastive blocks of the validation slice (disjoint samples).
            generator: Torch generator for the minibatch draws.

        Returns:
            ``{"best_iter", "stopped_iter", "best_val"}``.
        """
        k = min(int(self.cfg.contrastive_batch), int(len(train_idx)))
        if k < 2:
            raise ValueError(f"contrastive batch of {k} is degenerate; need >= 2 fit anchors")
        opt = torch.optim.Adam(self.critic.parameters(), lr=float(self.cfg.lr))
        x_tr, z_tr = x[train_idx], z[train_idx]

        best_val, best_state, best_iter, stale = -float("inf"), None, 0, 0
        step = 0
        self.critic.train()
        for step in range(1, int(self.cfg.n_iters) + 1):
            idx = torch.randint(0, x_tr.shape[0], (k,), generator=generator, device=self.device)
            loss = -infonce_lower_bound(self.critic(x_tr[idx], z_tr[idx]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.trace.append(float(-loss.detach()))

            if val_blocks and step % int(self.cfg.eval_every) == 0:
                self.critic.eval()
                score = float(self._eval_blocks(x, z, val_blocks).mean())
                self.critic.train()
                self.val_trace.append(score)
                if score > best_val + 1e-4:
                    best_val, best_iter, stale = score, step, 0
                    best_state = copy.deepcopy(self.critic.state_dict())
                else:
                    stale += 1
                    if stale >= int(self.cfg.patience):
                        break

        if best_state is not None:
            self.critic.load_state_dict(best_state)
        self.critic.eval()
        return {"best_iter": int(best_iter), "stopped_iter": int(step),
                "best_val": float(best_val) if np.isfinite(best_val) else None}

    @torch.no_grad()
    def _eval_blocks(self, x: torch.Tensor, z: torch.Tensor,
                     blocks: Sequence[np.ndarray]) -> np.ndarray:
        r"""One held-out InfoNCE value per contrastive block."""
        out = []
        for block in blocks:
            if len(block) < 2:
                continue
            idx = torch.as_tensor(block, dtype=torch.long, device=self.device)
            out.append(float(infonce_lower_bound(self.critic(x[idx], z[idx]))))
        return np.asarray(out, dtype=np.float64)

    def _blocks(self, index: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        r"""Chop a shuffled anchor index into ``contrastive_batch``-sized blocks."""
        k = int(self.cfg.contrastive_batch)
        shuffled = index.copy()
        rng.shuffle(shuffled)
        n_blocks = len(shuffled) // k
        if n_blocks == 0:
            return [shuffled] if len(shuffled) >= 2 else []
        return [shuffled[i * k: (i + 1) * k] for i in range(n_blocks)]

    def _converged(self) -> bool:
        r"""Did the validation score ever improve on its first reading?"""
        if len(self.val_trace) < 2:
            if len(self.trace) < 20:
                return False
            tenth = max(1, len(self.trace) // 10)
            return bool(np.mean(self.trace[-tenth:]) > np.mean(self.trace[:tenth]))
        return bool(max(self.val_trace) > self.val_trace[0])

    # -- public API ---------------------------------------------------------------------
    def fit_estimate(
        self,
        u: np.ndarray,
        v: np.ndarray,
        c: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Residualise, fit the critic on one sample partition, estimate on the other.

        Args:
            u: $(N, u_{\dim})$ source-history anchors.
            v: $(N, v_{\dim})$ future-target anchors.
            c: $(N, c_{\dim})$ conditioning anchors.
            groups: $(N,)$ integer sample id per anchor. Anchors sharing an id share a noise
                realisation, so they must not straddle the fit/eval split nor be resampled
                independently.

        Returns:
            A dict with ``estimate`` (nats), ``ci_lo`` / ``ci_hi`` (block bootstrap percentile
            interval), ``ceiling_nats``, ``at_ceiling`` (the estimate is within $1\%$ of
            $\log K$), ``cond_r2_u`` / ``cond_r2_v`` (variance of $U$ / $Y^{+}$ explained by
            $c_t$), ``n_anchors``, ``n_groups_fit``, ``n_groups_eval``, ``n_blocks``,
            ``best_iter`` / ``stopped_iter`` (early stopping), and ``converged``.

        Raises:
            ValueError: When array lengths disagree, fewer than three samples are present, or the
                held-out partition cannot form a contrastive block.
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        groups = np.asarray(groups)
        if not (len(u) == len(v) == len(c) == len(groups)):
            raise ValueError("u, v, c and groups must share a length")
        if len(np.unique(groups)) < 3:
            raise ValueError(
                "need >= 3 distinct samples: one partition each for fitting, early stopping, "
                "and the held-out estimate"
            )

        rng = np.random.default_rng(int(self.cfg.seed))
        generator = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed))
        fit_idx, eval_idx = self._split_by_group(groups, self.cfg.fit_frac, rng)

        # Carve the early-stopping validation slice out of the FIT samples, again by group, so
        # the reported held-out estimate never informed the stopping decision.
        train_idx, val_idx = self._split_by_group(
            groups[fit_idx], 1.0 - float(self.cfg.val_frac), rng
        )
        train_idx, val_idx = fit_idx[train_idx], fit_idx[val_idx]

        # The residual regression is part of the model: fit it on the training anchors only.
        resid_u, r2_u = self._residualise(u, c, train_idx)
        resid_v, r2_v = self._residualise(v, c, train_idx)
        x_t, z_t = self._tensor(resid_u), self._tensor(resid_v)

        val_blocks = self._blocks(val_idx, np.random.default_rng(int(self.cfg.seed) + 11))
        stop = self._train(x_t, z_t, train_idx, val_blocks, generator)

        blocks = self._blocks(eval_idx, rng)
        values = self._eval_blocks(x_t, z_t, blocks)
        if values.size == 0:
            raise ValueError("held-out partition has too few anchors to form a contrastive block")

        estimate = float(values.mean())
        ci_lo, ci_hi = self._bootstrap_ci(values, rng)
        ceiling = self.ceiling_nats
        return {
            "estimate": estimate,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ceiling_nats": ceiling,
            "at_ceiling": bool(estimate > 0.99 * ceiling),
            "cond_r2_u": r2_u,
            "cond_r2_v": r2_v,
            "n_anchors": int(len(u)),
            "n_groups_fit": int(len(np.unique(groups[train_idx]))),
            "n_groups_val": int(len(np.unique(groups[val_idx]))),
            "n_groups_eval": int(len(np.unique(groups[eval_idx]))),
            "n_blocks": int(len(blocks)),
            "best_iter": stop["best_iter"],
            "stopped_iter": stop["stopped_iter"],
            "converged": self._converged(),
        }

    def _bootstrap_ci(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> Tuple[Optional[float], Optional[float]]:
        r"""Percentile CI of the mean held-out InfoNCE value, resampling blocks with replacement.

        Blocks are the resampling unit: each is built from held-out samples and contributes one
        InfoNCE reading. The critic is frozen, so this measures evaluation variance, not fit
        variance -- stated plainly because a CI that refitted the critic would be much wider.

        Returns:
            ``(ci_lo, ci_hi)``, or ``(None, None)`` when fewer than three blocks exist.
        """
        n = int(values.size)
        if n < 3:
            return None, None
        idx = rng.integers(0, n, size=(int(self.cfg.n_boot), n))
        means = values[idx].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Anchor windows -- shared by the collector and the S7-T03 recovery check
# ---------------------------------------------------------------------------
def valid_anchor_range(seq_len: int, u_lookback: int, c_lookback: int, horizon: int) -> range:
    r"""The anchors $t$ at which every window fits inside a length-``seq_len`` row.

    Needs ``max(u_lookback, c_lookback) - 1`` steps of history and ``horizon`` steps of future.
    """
    lo = max(int(u_lookback), int(c_lookback)) - 1
    hi = int(seq_len) - int(horizon)
    return range(max(lo, 0), max(hi, 0))


def latent_anchor_rows(
    c: np.ndarray,
    d: np.ndarray,
    anchors: Sequence[int],
    *,
    u_lookback: int,
    c_lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Slice one row's latents into $(U_{\le t},\, Y^{+}_{t,1:H},\, c_t)$ anchor triples.

    $$U_{\le t} = \bigl(c_{t-L_u+1}, \dots, c_t\bigr), \qquad
      Y^{+}_{t,1:H} = \bigl(d_{t+1}, \dots, d_{t+H}\bigr), \qquad
      c_t = \bigl(d_{t-L_c+1}, \dots, d_t\bigr).$$

    This is the ``latent`` configuration: every variable lives in the data-generating process's own
    coordinates, so the resulting CMI is the injected block transfer entropy and is comparable to
    $\mathrm{TE}_{\mathrm{inj}}$ in nats. Matching the label means matching its conditioning depth:
    ``build_dataset_v2`` solves each cell's coupling at ``data.K_history``, so ``u_lookback`` and
    ``c_lookback`` default to that value rather than to the (much shorter) feature-config lookback.

    Args:
        c: Source latent row, shape $(T,)$.
        d: Target latent row, shape $(T,)$.
        anchors: The anchor indices $t$ to extract.
        u_lookback: $L_u$, the source history depth.
        c_lookback: $L_c$, the conditioning history depth.
        horizon: $H$, the forecast horizon.

    Returns:
        ``(u, v, cond)`` of shapes $(A, L_u)$, $(A, H)$, $(A, L_c)$.
    """
    lu, lc, h = int(u_lookback), int(c_lookback), int(horizon)
    idx = np.asarray(anchors, dtype=np.int64)
    u = c[idx[:, None] - np.arange(lu - 1, -1, -1)[None, :]]
    v = d[idx[:, None] + np.arange(1, h + 1)[None, :]]
    cond = d[idx[:, None] - np.arange(lc - 1, -1, -1)[None, :]]
    return u, v, cond


def recover_synthetic(
    *,
    B: float,
    D: int,
    horizon: int,
    k_history: int,
    n: int,
    t_tot: int,
    cfg: CMIConfig,
    r: float = 0.92,
    w: float = 0.20,
    target_ar: float = 0.50,
    sigma2_y: float = 1.0,
    sigma2_eta: float = 1.0,
    seed: int = 0,
    anchor_stride: int = 4,
) -> Dict[str, Any]:
    r"""Estimate the CMI on freshly simulated latents whose analytic block TE is known.

    Drives :func:`raw_generators.simulate_latent_pair` -- the same simulator the injected-TE label
    is solved against -- so the estimate and the target are two readings of one process. With
    ``B = 0`` the source is disconnected and the true CMI is exactly zero.

    Args:
        B: Coupling strength; ``0.0`` gives an independent (null) pair.
        D: Source$\to$target delay in decimated steps.
        horizon: Block horizon $H$.
        k_history: Conditioning / source-history depth, matching the analytic label's
            ``K_history``.
        n: Number of simulated samples (each becomes a bootstrap cluster).
        t_tot: Simulated sequence length.
        cfg: Estimator knobs.
        r: Source pole radius.
        w: Source pole angle, rad/step.
        target_ar: Target self-coefficient $A_y$.
        sigma2_y: Target innovation variance.
        sigma2_eta: Source innovation variance.
        seed: Simulation seed.
        anchor_stride: Keep every ``anchor_stride``-th anchor, to decorrelate the pool.

    Returns:
        The :meth:`NeuralCMIEstimator.fit_estimate` dict, plus ``te_analytic`` (the independent
        Monte-Carlo block TE from :func:`analytic_te.te_block_state_space_gaussian`) and
        ``cmi_gauss_exact`` (:func:`gaussian_cmi_closed_form` on the very anchors the estimator
        saw). The two references agree to $1\text{--}2\%$, which is what proves the anchor windowing
        reproduces the block-TE estimand rather than some neighbouring quantity.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.analytic_te import (
        te_block_state_space_gaussian,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (
        simulate_latent_pair,
    )

    c_all, d_all = simulate_latent_pair(
        int(n), int(t_tot), r=r, w=w, target_ar=target_ar, B=float(B), D=int(D),
        sigma2_y=sigma2_y, sigma2_eta=sigma2_eta, seed=int(seed),
    )
    te_analytic = float(
        te_block_state_space_gaussian(
            [(r, w)], target_ar, [int(D)], [float(B)], sigma2_y, sigma2_eta, int(horizon),
            K_history=int(k_history), n_samples=20_000, seed=int(seed) + 1,
        )
    )

    anchors = list(valid_anchor_range(t_tot, k_history, k_history, horizon))[::int(anchor_stride)]
    u_rows, v_rows, c_rows, groups = [], [], [], []
    for i in range(int(n)):
        u, v, cond = latent_anchor_rows(
            c_all[i], d_all[i], anchors,
            u_lookback=k_history, c_lookback=k_history, horizon=horizon,
        )
        u_rows.append(u)
        v_rows.append(v)
        c_rows.append(cond)
        groups.append(np.full(len(anchors), i, dtype=np.int64))

    u_all = np.concatenate(u_rows)
    v_all = np.concatenate(v_rows)
    c_all_anchor = np.concatenate(c_rows)
    est = NeuralCMIEstimator(int(k_history), int(horizon), int(k_history), cfg)
    out = est.fit_estimate(u_all, v_all, c_all_anchor, np.concatenate(groups))
    out["te_analytic"] = te_analytic
    out["cmi_gauss_exact"] = gaussian_cmi_closed_form(u_all, v_all, c_all_anchor)
    out["near_ceiling"] = est.near_ceiling(te_analytic)
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
#: Documented defaults; ``benchmarks.<bm>.eval.cmi`` overrides. Resolved in S7-T03 and recorded in
#: Section 11 of ``SYNTHETIC_V3_SPEC_AND_SPRINTS.md``.
_DEFAULT_CMI: Dict[str, Any] = {
    "configs": list(_ALL_CONFIGS),
    "latent_lookback": 24,     # >= max(D) = 20; longer accrues a log-det bias
    "u_lookback": 24,          # feature configs: the source window depth
    "c_lookback": 24,          # feature configs: the ground-truth conditioning depth
    "u_channels": "coupled",   # "all" | "coupled" (meta.coupled_channel.up_st)
    "y_channels": "coupled",   # "all" | "coupled" (meta.coupled_channel.fhr_st)
    "anchor_stride": 3,
    "max_anchors_per_cell": 12_000,
    "max_samples": 512,
    "critic_width": 64,
    "critic_depth": 1,
    "embed_dim": 16,
    "contrastive_batch": 32,   # ceiling = log(32) = 3.4657 nats
    "n_iters": 6000,
    "eval_every": 50,
    "patience": 15,
    "val_frac": 0.25,
    "fit_frac": 0.5,
    "lr": 1.0e-3,
    "n_boot": _N_BOOT_CMI,
    "ceiling_claim_frac": 0.8,
    "min_cells_for_rho": _MIN_CELLS_FOR_RHO,
}


def _cmi_cfg(config: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    r"""Read ``benchmarks.<benchmark>.eval.cmi``, falling back to :data:`_DEFAULT_CMI`."""
    bench = (config.get("benchmarks") or {}).get(benchmark) or {}
    cfg = ((bench.get("eval") or {}).get("cmi")) or {}
    merged = dict(_DEFAULT_CMI)
    merged.update({k: v for k, v in cfg.items() if v is not None})
    merged["configs"] = [str(c) for c in merged["configs"]]
    unknown = set(merged["configs"]) - set(_ALL_CONFIGS)
    if unknown:
        raise ValueError(f"eval.cmi.configs has unknown entries {sorted(unknown)}")
    return merged


def _estimator_config(cfg: Dict[str, Any], *, seed: int, device: str) -> CMIConfig:
    r"""Project the stage's config block onto the estimator's dataclass."""
    return CMIConfig(
        critic_width=int(cfg["critic_width"]), critic_depth=int(cfg["critic_depth"]),
        embed_dim=int(cfg["embed_dim"]), contrastive_batch=int(cfg["contrastive_batch"]),
        n_iters=int(cfg["n_iters"]), lr=float(cfg["lr"]), fit_frac=float(cfg["fit_frac"]),
        val_frac=float(cfg["val_frac"]), eval_every=int(cfg["eval_every"]),
        patience=int(cfg["patience"]), n_boot=int(cfg["n_boot"]),
        ceiling_claim_frac=float(cfg["ceiling_claim_frac"]), seed=int(seed), device=device,
    )


# ---------------------------------------------------------------------------
# S7-T04a/b: the collector -- one clean forward per batch, anchors exported to numpy
# ---------------------------------------------------------------------------
def _window_index(anchors: np.ndarray, lookback: int) -> np.ndarray:
    r"""$(A, L)$ index of the lookback window ending at each anchor, oldest first."""
    return anchors[:, None] - np.arange(int(lookback) - 1, -1, -1)[None, :]


def _channel_index(meta: Dict[str, Any], mode: str, n_channels: int, key: str) -> np.ndarray:
    r"""Which feature channels enter an anchor: all of them, or the single coupled one.

    ``"all"`` keeps the full stream. ``"coupled"`` keeps only the channel the injection rides on
    (``meta.coupled_channel.<key>``) -- a **model-free** dimensionality reduction, and the same one
    ``eval_v2.measure_te_scat`` makes when it slices ``slice_coupled_channels`` before estimating
    $\mathrm{TE}_{\mathrm{scat}}$.

    It is not cosmetic. At ``"all"`` the source anchor is $24 \times 101 = 2424$-dimensional and the
    future-target anchor $30 \times 87 = 2610$-dimensional, against a few thousand *correlated*
    anchors per cell; the critic memorises and its held-out bound goes to $-5$ nats or worse, which
    is a valid but vacuous lower bound. At ``"coupled"`` the anchors are $24$- and $30$-dimensional
    and the bound is informative. The cost of the reduction is an assumption -- that the transform
    did not spread the coupling beyond its own channel -- which is exactly the assumption
    $\mathrm{TE}_{\mathrm{scat}}$ already makes.

    Args:
        meta: The cache manifest.
        mode: ``"all"`` or ``"coupled"``.
        n_channels: The stream's channel count.
        key: The ``coupled_channel`` entry to read (``"up_st"`` or ``"fhr_st"``).

    Returns:
        The channel indices to keep.
    """
    if str(mode) == "all":
        return np.arange(int(n_channels))
    if str(mode) == "coupled":
        idx = int(((meta.get("coupled_channel") or {}).get(key, 0)))
        return np.asarray([idx], dtype=np.int64)
    raise ValueError(f"eval.cmi channel mode must be 'all' or 'coupled', got {mode!r}")


def collect_cmi_anchors(
    model: Any,
    runner: Any,
    loader: Any,
    latent_provider: Optional[Callable[[int, int], Tuple[np.ndarray, np.ndarray]]],
    *,
    cfg: Dict[str, Any],
    configs: Sequence[str],
    meta: Dict[str, Any],
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    r"""Extract per-cell anchor pools for every requested estimator configuration.

    Runs **one clean forward per batch** inside ``runner.inference_mode()`` and immediately exports
    everything through ``.cpu().numpy()``. That is not defensive style: ``inference_mode`` produces
    *inference tensors*, which can never enter an autograd graph, so a critic fitted on them would
    fail at ``backward()``. Every fit therefore happens outside, in :func:`run_cmi_comparison`.

    Anchors are restricted to the model's **clean window** -- the same
    :func:`eval_v2._clean_window_mean` mask ``eval`` uses for $\bar K$ -- so
    $\rho(\bar K, \mathrm{CMI})$ compares like with like.

    Args:
        model: The rebuilt model (only ``target_state`` is read from its forward).
        runner: The configured ``TestRunner``.
        loader: Dataloader over the evaluation subset.
        latent_provider: ``(cell_id, raw_index) -> (c, d)``, or ``None`` when the ground-truth
            latents are unavailable (the two latent-dependent configs are then skipped).
        cfg: The resolved ``eval.cmi`` block.
        configs: Which of ``latent`` / ``feature_gt`` / ``feature_model`` to build.
        meta: The cache manifest, for ``coupled_channel``.
        max_samples: Cap on samples consumed.
        seed: Seeds the per-cell anchor subsample.

    Returns:
        ``(cells, n_samples_seen)`` where ``cells`` maps ``cell_id`` to
        ``{"te_inj", "te_scat", "delay", "kbar", "n_rows", "n_anchors", "groups",
        "<config>": {"u", "v", "c"}}``, all arrays ``float64`` / ``int64`` numpy.

    Raises:
        ValueError: When a latent-dependent config is requested without a ``latent_provider``.
    """
    want_latent = CONFIG_LATENT in configs
    want_feat_gt = CONFIG_FEATURE_GT in configs
    want_feat_model = CONFIG_FEATURE_MODEL in configs
    if (want_latent or want_feat_gt) and latent_provider is None:
        raise ValueError(
            f"configs {sorted(set(configs) & set(_LATENT_DEPENDENT))} need the ground-truth latent "
            "provider, but none was supplied. The stage drops them when it cannot be built; a "
            "direct caller must do the same."
        )

    warmup, horizon = int(runner.warmup_steps), int(runner.horizon)
    lu, lc = int(cfg["u_lookback"]), int(cfg["c_lookback"])
    ll = int(cfg["latent_lookback"])
    stride = max(1, int(cfg["anchor_stride"]))

    pools: Dict[int, Dict[str, Any]] = {}
    n_seen = 0

    def _pool(cid: int) -> Dict[str, Any]:
        if cid not in pools:
            pools[cid] = {"cols": {}, "groups": [], "kbar": [], "te_inj": [], "te_scat": [],
                          "delay": [], "n_rows": 0}
        return pools[cid]

    def _push(pool: Dict[str, Any], key: str, arr: np.ndarray) -> None:
        pool["cols"].setdefault(key, []).append(arr)

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            y_st, y_ph = batch.fhr_st, batch.fhr_ph
            u_stream = build_u_stream(batch)
            delay_t = torch.as_tensor(batch.delay)
            out = model(y_st, y_ph, u_stream)
            kbar_b, valid = _clean_window_mean(
                out["kld_per_t"], delay_t, warmup=warmup, horizon=horizon
            )
            y_plus = runner.build_future_target(batch)          # (B, T-H, H, C_y)
            target_state = out["target_state"]                  # (B, T, d_h)

            # Everything below leaves inference-mode territory immediately.
            valid_np = valid.detach().cpu().numpy()
            kbar_np = kbar_b.detach().cpu().numpy().astype(np.float64)
            u_np = u_stream.detach().cpu().numpy().astype(np.float64)
            yp_np = y_plus.detach().cpu().numpy().astype(np.float64)
            ts_np = target_state.detach().cpu().numpy().astype(np.float64)
            cid_np = torch.as_tensor(batch.cell_id).cpu().numpy().astype(np.int64)
            raw_index = getattr(batch, "raw_index", None)
            if raw_index is None:
                if want_latent or want_feat_gt:
                    # Defaulting to zeros here would regenerate cell `cid`'s *first* row's latents
                    # for every sample -- a silent, plausible-looking wrong answer.
                    raise KeyError(
                        "the batch carries no `raw_index`, so the ground-truth latents cannot be "
                        "matched to their cached rows. Rebuild the cache, or drop the "
                        f"{sorted(_LATENT_DEPENDENT)} configs."
                    )
                ridx_np = np.zeros(len(cid_np), dtype=np.int64)
            else:
                ridx_np = torch.as_tensor(raw_index).cpu().numpy().astype(np.int64)
            te_np = torch.as_tensor(batch.te_true).cpu().numpy().astype(np.float64)
            ts_scat = getattr(batch, "te_scat", None)
            tes_np = (torch.as_tensor(ts_scat).cpu().numpy().astype(np.float64)
                      if ts_scat is not None else np.full(len(cid_np), np.nan))
            dly_np = delay_t.cpu().numpy().astype(np.int64)

            src_ch = _channel_index(meta, cfg["u_channels"], u_np.shape[-1], "up_st")
            tgt_ch = _channel_index(meta, cfg["y_channels"], yp_np.shape[-1], "fhr_st")
            n_valid_anchor = yp_np.shape[1]     # T - H

            for b in range(len(cid_np)):
                cid = int(cid_np[b])
                pool = _pool(cid)
                # Anchors: inside the clean window AND with a full lookback and future.
                lo = max(lu, lc, ll) - 1
                ok = np.where(valid_np[b][:n_valid_anchor])[0]
                anchors = ok[ok >= lo][::stride]
                if anchors.size < 4:
                    continue

                if want_latent or want_feat_gt:
                    c_lat, d_lat = latent_provider(cid, int(ridx_np[b]))
                    if not np.isfinite(c_lat).all() or not np.isfinite(d_lat).all():
                        continue
                    if want_latent:
                        u, v, cc = latent_anchor_rows(
                            c_lat, d_lat, anchors, u_lookback=ll, c_lookback=ll, horizon=horizon
                        )
                        _push(pool, f"{CONFIG_LATENT}_u", u)
                        _push(pool, f"{CONFIG_LATENT}_v", v)
                        _push(pool, f"{CONFIG_LATENT}_c", cc)
                    if want_feat_gt:
                        _push(pool, f"{CONFIG_FEATURE_GT}_c", d_lat[_window_index(anchors, lc)])

                if want_feat_gt or want_feat_model:
                    win = _window_index(anchors, lu)                     # (A, Lu)
                    u_feat = u_np[b][win][:, :, src_ch].reshape(len(anchors), -1)
                    v_feat = yp_np[b][anchors][:, :, tgt_ch].reshape(len(anchors), -1)
                    if want_feat_gt:
                        _push(pool, f"{CONFIG_FEATURE_GT}_u", u_feat)
                        _push(pool, f"{CONFIG_FEATURE_GT}_v", v_feat)
                    if want_feat_model:
                        _push(pool, f"{CONFIG_FEATURE_MODEL}_u", u_feat)
                        _push(pool, f"{CONFIG_FEATURE_MODEL}_v", v_feat)
                        _push(pool, f"{CONFIG_FEATURE_MODEL}_c", ts_np[b][anchors])

                pool["groups"].append(np.full(len(anchors), pool["n_rows"], dtype=np.int64))
                pool["n_rows"] += 1
                pool["kbar"].append(float(kbar_np[b]))
                pool["te_inj"].append(float(te_np[b]))
                pool["te_scat"].append(float(tes_np[b]))
                pool["delay"].append(int(dly_np[b]))
            n_seen += len(cid_np)

    # Concatenate, subsample, and drop the scaffolding.
    rng = np.random.default_rng(int(seed))
    result: Dict[int, Dict[str, Any]] = {}
    cap = int(cfg["max_anchors_per_cell"])
    for cid, pool in pools.items():
        if pool["n_rows"] < 3:
            logger.warning("cmi: cell {} has only {} usable rows; skipping.", cid, pool["n_rows"])
            continue
        groups = np.concatenate(pool["groups"])
        cols = {k: np.concatenate(v) for k, v in pool["cols"].items()}
        if len(groups) > cap:
            keep = np.sort(rng.choice(len(groups), size=cap, replace=False))
            groups = groups[keep]
            cols = {k: v[keep] for k, v in cols.items()}
        entry: Dict[str, Any] = {
            "cell_id": cid,
            "n_rows": int(pool["n_rows"]),
            "n_anchors": int(len(groups)),
            "te_inj": float(np.mean(pool["te_inj"])),
            "te_scat": float(np.mean(pool["te_scat"])),
            "delay": int(np.median(pool["delay"])),
            "kbar": float(np.mean(pool["kbar"])),
            "groups": groups,
        }
        for name in configs:
            keys = (f"{name}_u", f"{name}_v", f"{name}_c")
            if all(k in cols for k in keys):
                entry[name] = {"u": cols[keys[0]], "v": cols[keys[1]], "c": cols[keys[2]]}
        result[cid] = entry
    return result, n_seen


# ---------------------------------------------------------------------------
# S7-T04a/b: fit per cell per config, difference the conditionings, correlate against kbar
# ---------------------------------------------------------------------------
def _bootstrap_rho_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000, rng=None
) -> Tuple[Optional[float], Optional[float]]:
    r"""Percentile CI of Spearman $\rho$, resampling the **cells** jointly.

    ``eval_v2._bootstrap_ci`` bootstraps a mean; a correlation needs its pairs resampled together,
    so the statistic is recomputed on each resampled cell set. Mirrors
    ``lag_intervention_v3._bootstrap_rho_ci``.
    """
    gen = rng if rng is not None else np.random.default_rng()
    n = int(np.asarray(x).size)
    if n < 3:
        return None, None
    idx = gen.integers(0, n, size=(int(n_boot), n))
    rhos = [_spearman_finite(x[row], y[row]) for row in idx]
    finite = np.asarray([r for r in rhos if r is not None], dtype=np.float64)
    if finite.size < 2:
        return None, None
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(lo), float(hi)


def _difference_ci(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    r"""An interval for $a - b$ from two independent percentile intervals.

    The two estimates are fitted on the same anchors but by **independent critics**, and their
    bootstraps resample independently. Combining the half-widths in quadrature is the usual
    normal-approximation move; it is reported as an interval on the difference, never as a
    hypothesis test.
    """
    for k in ("ci_lo", "ci_hi"):
        if a.get(k) is None or b.get(k) is None:
            return None, None
    ha = (a["ci_hi"] - a["ci_lo"]) / 2.0
    hb = (b["ci_hi"] - b["ci_lo"]) / 2.0
    half = math.sqrt(ha * ha + hb * hb)
    delta = a["estimate"] - b["estimate"]
    return float(delta - half), float(delta + half)


#: A held-out $R^2$ below this is a *negative* $R^2$ up to sampling noise. Measured separation on
#: the pilot is enormous (see :func:`_bias_reliability`), so the exact value is not delicate.
_COND_R2_TOL: float = 0.0


def _bias_reliability(model_block: Dict[str, Any]) -> Dict[str, Any]:
    r"""Is the ``target_state`` residualisation well-posed enough for ``bias`` to mean anything?

    The gate is ``cond_r2_v``: the **held-out** $R^2$ of a linear fit of the target's own future
    $Y^{+}$ on the conditioning summary $c_t$. A summary that claims to represent the target's past
    must predict the target's future; if it cannot do so *out of sample*, the residual
    $\tilde Y^{+}$ is inflated by fit error and the InfoNCE bound on it is vacuous rather than
    over-conditioned. ``bias`` then compares a working conditioning against a broken one, and means
    nothing.

    This is not hypothetical, and the separation is total. Under ``causal_norm: false`` the target
    encoder's ``GroupNorm`` reduces over $(C/G, T)$ -- **across time** -- so ``target_state[b, t]``
    carries a per-sample normalising statistic. The fit/eval split is by sample, so a regression
    fitted on the fit samples cannot transfer. Measured per cell on the pilot ``val`` split:

    ==================  =====================  ===================
    arm                 ``causal_norm``        ``cond_r2_v``
    ==================  =====================  ===================
    ``parity``          ``false``              negative on **15/15** cells ($-0.05$ to $-0.74$)
    ``v3_noncausal``    ``false``              negative on **15/15** cells ($-0.04$ to $-0.74$)
    ``v3_prod``         ``true``               positive on **15/15** cells ($+0.003$ to $+0.13$)
    ==================  =====================  ===================

    So this flag is, incidentally, a detector for the G0 leak that never looks at the KL -- and it
    fires on a model whose source pathway never opened.

    ``cond_r2_u`` is reported but **not** gated on: it measures how well the target's past predicts
    the *source's* past, which on a $\mathrm{TE}_{\mathrm{inj}} = 0$ cell is genuinely zero, so a
    small negative value there is correct behaviour rather than a defect (``v3_prod`` reads
    $-0.05, -0.04, -0.06$ on exactly its three null cells).

    Args:
        model_block: The ``feature_model`` estimate dict.

    Returns:
        ``{"reliable": bool, "reason": str | None}``.
    """
    r2_v = model_block.get("cond_r2_v")
    if r2_v is not None and np.isfinite(r2_v) and r2_v < _COND_R2_TOL:
        return {
            "reliable": False,
            "reason": (
                f"held-out cond_r2_v = {r2_v:.3f} < {_COND_R2_TOL}: the residualisation on "
                "`target_state` does not generalise across samples, so this bias measures a "
                "non-transferable conditioning summary, not a worse one"
            ),
        }
    return {"reliable": True, "reason": None}


def run_cmi_comparison(
    cells: Dict[int, Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    configs: Sequence[str],
    horizon: int,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    r"""Fit the estimator per cell per configuration and assemble the comparison.

    Runs **outside** ``inference_mode`` (see :func:`collect_cmi_anchors`). The ``latent`` config
    also reports :func:`gaussian_cmi_closed_form`, which is exact for its Gaussian variables and
    therefore audits the neural estimate at no cost.

    ``bias`` = ``cmi_feature_model`` $-$ ``cmi_feature_gt``. Both share $U$ and $Y^{+}$ exactly;
    only $c_t$ differs, so the difference is attributable to the conditioning summary. Because
    ``feature_model``'s conditioning is a function of the model, only **rank-level** corroboration
    is claimed for it and for the bias.

    Args:
        cells: The per-cell anchor pools from :func:`collect_cmi_anchors`.
        cfg: The resolved ``eval.cmi`` block.
        configs: Which configurations were collected.
        horizon: $H$, used only to size the estimator.
        seed: Seeds the estimator and every bootstrap.
        device: Torch device for the critics.

    Returns:
        A JSON-able summary with ``per_cell``, ``overall`` and ``recovery``.
    """
    if not cells:
        return {"error": "no cells produced usable anchors"}

    est_cfg = _estimator_config(cfg, seed=seed, device=device)
    rng = np.random.default_rng(int(seed))
    per_cell: Dict[str, Dict[str, Any]] = {}

    for cid in sorted(cells):
        entry = cells[cid]
        record: Dict[str, Any] = {
            "cell_id": int(cid),
            "n_rows": entry["n_rows"],
            "n_anchors": entry["n_anchors"],
            "te_inj": entry["te_inj"],
            "te_scat": entry["te_scat"],
            "delay": entry["delay"],
            "kbar": entry["kbar"],
        }
        for name in configs:
            pool = entry.get(name)
            if pool is None:
                continue
            est = NeuralCMIEstimator(
                pool["u"].shape[1], pool["v"].shape[1], pool["c"].shape[1], est_cfg
            )
            try:
                out = est.fit_estimate(pool["u"], pool["v"], pool["c"], entry["groups"])
            except ValueError as exc:  # too few samples / degenerate block
                logger.warning("cmi: cell {} config {} failed ({})", cid, name, exc)
                continue
            out["near_ceiling"] = est.near_ceiling(entry["te_inj"])
            record[f"cmi_{name}"] = out
            if name == CONFIG_LATENT:
                record["cmi_latent_gauss_exact"] = gaussian_cmi_closed_form(
                    pool["u"], pool["v"], pool["c"]
                )

        gt, mdl = record.get(f"cmi_{CONFIG_FEATURE_GT}"), record.get(f"cmi_{CONFIG_FEATURE_MODEL}")
        if gt is not None and mdl is not None:
            lo, hi = _difference_ci(mdl, gt)
            record["bias"] = {
                "estimate": float(mdl["estimate"] - gt["estimate"]), "ci_lo": lo, "ci_hi": hi,
                **_bias_reliability(mdl),
            }
        per_cell[str(cid)] = record

    return {
        "per_cell": per_cell,
        "overall": _overall(per_cell, configs, cfg, rng),
        "recovery": _recovery(per_cell, cfg),
        "ceiling_nats": float(math.log(int(cfg["contrastive_batch"]))),
        "ceiling_claim_frac": float(cfg["ceiling_claim_frac"]),
        "configs": list(configs),
        "estimator": {k: cfg[k] for k in (
            "critic_width", "critic_depth", "embed_dim", "contrastive_batch", "n_iters",
            "eval_every", "patience", "val_frac", "fit_frac", "lr", "n_boot",
            "latent_lookback", "u_lookback", "c_lookback", "u_channels", "y_channels",
            "anchor_stride", "max_anchors_per_cell",
        )},
        "horizon": int(horizon),
    }


def _overall(per_cell: Dict[str, Dict[str, Any]], configs: Sequence[str],
             cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    r"""Cross-cell rank correlations and the pooled bias.

    $\rho(\bar K, \mathrm{CMI})$ is computed **separately per conditioning**: the three
    configurations answer different questions and pooling them would blur exactly the distinction
    Sprint 7 exists to draw.
    """
    cells = list(per_cell.values())
    n_cells = len(cells)
    gated = n_cells >= int(cfg["min_cells_for_rho"])
    overall: Dict[str, Any] = {
        "n_cells": n_cells,
        "n_signal_cells": sum(1 for c in cells if c["te_inj"] > 0.0),
        "rho_reported_not_gated": not gated,
    }

    kbar = np.asarray([c["kbar"] for c in cells], dtype=np.float64)
    te = np.asarray([c["te_inj"] for c in cells], dtype=np.float64)
    for name in configs:
        vals = np.asarray(
            [c.get(f"cmi_{name}", {}).get("estimate", np.nan) for c in cells], dtype=np.float64
        )
        rho = _spearman_finite(kbar, vals)
        lo, hi = _bootstrap_rho_ci(kbar, vals, n_boot=int(cfg["n_boot"]), rng=rng) \
            if rho is not None else (None, None)
        overall[f"rho_kbar_cmi_{name}"] = {
            "rho": rho, "ci": [lo, hi], "n_cells": n_cells, "gated": bool(gated),
        }

    if CONFIG_LATENT in configs:
        vals = np.asarray(
            [c.get(f"cmi_{CONFIG_LATENT}", {}).get("estimate", np.nan) for c in cells],
            dtype=np.float64,
        )
        rho = _spearman_finite(vals, te)
        lo, hi = _bootstrap_rho_ci(vals, te, n_boot=int(cfg["n_boot"]), rng=rng) \
            if rho is not None else (None, None)
        overall["rho_cmi_te_inj_latent"] = {"rho": rho, "ci": [lo, hi], "n_cells": n_cells}

    with_bias = [c for c in cells if "bias" in c]
    biases = np.asarray([c["bias"]["estimate"] for c in with_bias], dtype=np.float64)
    if biases.size >= 2:
        mean, lo, hi = _bootstrap_ci(biases, n_boot=int(cfg["n_boot"]), rng=rng)
        unreliable = [c for c in with_bias if not c["bias"].get("reliable", True)]
        overall["cmi_bias"] = {
            "estimate": mean, "ci_lo": lo, "ci_hi": hi, "n_cells": int(biases.size),
            "reliable": not unreliable,
            "n_cells_unreliable": len(unreliable),
            "reason": unreliable[0]["bias"]["reason"] if unreliable else None,
        }
    else:
        overall["cmi_bias"] = None

    # The conditioning's held-out explanatory power, averaged over cells: the diagnostic that says
    # whether `cmi_bias` may be read at all.
    for name in configs:
        r2v = np.asarray([c.get(f"cmi_{name}", {}).get("cond_r2_v", np.nan) for c in cells],
                         dtype=np.float64)
        r2u = np.asarray([c.get(f"cmi_{name}", {}).get("cond_r2_u", np.nan) for c in cells],
                         dtype=np.float64)
        if np.isfinite(r2v).any():
            overall[f"cond_r2_{name}"] = {"u": float(np.nanmean(r2u)), "v": float(np.nanmean(r2v))}
    return overall


def _recovery(per_cell: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    r"""How well the model-free ``latent`` estimate reproduces $\mathrm{TE}_{\mathrm{inj}}$."""
    cells = [c for c in per_cell.values() if f"cmi_{CONFIG_LATENT}" in c]
    if not cells:
        return {"available": False}
    te = np.asarray([c["te_inj"] for c in cells], dtype=np.float64)
    hat = np.asarray([c[f"cmi_{CONFIG_LATENT}"]["estimate"] for c in cells], dtype=np.float64)
    ceiling = math.log(int(cfg["contrastive_batch"]))
    claim = float(cfg["ceiling_claim_frac"]) * ceiling
    absolute = (te > 0.0) & (te <= claim)
    ratios = hat[absolute] / te[absolute] if absolute.any() else np.asarray([])
    null = te <= 0.0
    return {
        "available": True,
        "n_cells": int(len(cells)),
        "spearman_cmi_te_inj": _spearman_finite(hat, te),
        "n_absolute_claim_cells": int(absolute.sum()),
        "factor2_pass_frac": (float(np.mean((ratios >= 0.5) & (ratios <= 2.0)))
                              if ratios.size else None),
        "min_ratio": float(ratios.min()) if ratios.size else None,
        "max_ratio": float(ratios.max()) if ratios.size else None,
        "max_abs_null_cmi": float(np.abs(hat[null]).max()) if null.any() else None,
        "near_ceiling_cells": [int(c["cell_id"]) for c in cells
                               if c[f"cmi_{CONFIG_LATENT}"].get("near_ceiling")],
    }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
_CSV_BASE = ("cell_id", "delay", "te_inj", "te_scat", "kbar", "n_rows", "n_anchors")


def write_cmi_table(summary: Dict[str, Any], path: Path) -> Path:
    r"""Write one row per cell to ``cmi_table.csv``, one column group per configuration."""
    per_cell = summary.get("per_cell") or {}
    configs = summary.get("configs") or []
    header = list(_CSV_BASE)
    for name in configs:
        header += [f"cmi_{name}", f"cmi_{name}_lo", f"cmi_{name}_hi", f"cmi_{name}_near_ceiling"]
        if name == CONFIG_LATENT:
            header.append("cmi_latent_gauss_exact")
    header += ["bias", "bias_lo", "bias_hi"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for cid in sorted(per_cell, key=lambda k: int(k)):
            cell = per_cell[cid]
            row: List[Any] = [cell.get(k) for k in _CSV_BASE]
            for name in configs:
                block = cell.get(f"cmi_{name}") or {}
                row += [block.get("estimate"), block.get("ci_lo"), block.get("ci_hi"),
                        block.get("near_ceiling")]
                if name == CONFIG_LATENT:
                    row.append(cell.get("cmi_latent_gauss_exact"))
            bias = cell.get("bias") or {}
            row += [bias.get("estimate"), bias.get("ci_lo"), bias.get("ci_hi")]
            writer.writerow(row)
    return path


def _render_figure(summary: Dict[str, Any], figures_dir: Path) -> None:
    r"""Draw the CMI-vs-TE overlay with the ceiling line; never fatal."""
    try:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
            visualize_v2 as viz,
        )

        figures_dir.mkdir(parents=True, exist_ok=True)
        viz.plot_cmi_comparison(summary, figures_dir / "cmi_comparison")
    except Exception as exc:  # noqa: BLE001 -- a bad figure must not lose the JSON
        logger.warning("cmi: figure failed ({})", exc)


def run_cmi_stage(ctx: StageContext) -> int:
    r"""Estimate the CMI for one arm, on every requested split.

    Degrades rather than raises. When the ground-truth latent provider cannot be built, the two
    latent-dependent configurations (``latent``, ``feature_gt``) are dropped with a logged skip and
    the run continues on ``feature_model`` alone -- which is exactly the position the production
    ``testing/`` pipeline is in, since real data has no ground-truth latents.

    Args:
        ctx: The arm-resolved stage context. ``ctx.max_samples`` overrides the configured cap.

    Returns:
        ``0`` on success. Registered ``fatal=False``, so a raise here is logged and the run
        continues.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        resolve_cache_dir,
    )

    cfg = _cmi_cfg(ctx.config, ctx.benchmark)
    max_samples = ctx.max_samples if ctx.max_samples is not None else cfg["max_samples"]
    seed = int(((ctx.config.get("seeds") or {}).get("base_seed", 0)))
    run_dir = ctx.run_dir()
    cache_dir = resolve_cache_dir(ctx.config, benchmark=ctx.benchmark)
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    for split in ctx.splits():
        runner, loader, used_split, ckpt_path = _build_runner_and_loader(
            ctx.config, benchmark=ctx.benchmark, arm=ctx.arm, ckpt=ctx.ckpt, split=split,
        )
        split_dir = _split_dir(run_dir, used_split)
        model = runner.model

        configs = list(cfg["configs"])
        latent_provider = None
        if any(name in configs for name in _LATENT_DEPENDENT):
            try:
                latent_provider = make_latent_provider(
                    ctx.config, used_split, benchmark=ctx.benchmark, cache_dir=cache_dir
                )
            except Exception as exc:  # noqa: BLE001 -- degrade, do not gate
                logger.warning(
                    "cmi[{}/{}]: ground-truth latents unavailable ({}); dropping {} and "
                    "continuing on the model-conditioned estimate only.",
                    ctx.arm or "-", used_split, exc, list(_LATENT_DEPENDENT),
                )
                configs = [c for c in configs if c not in _LATENT_DEPENDENT]
        if not configs:
            logger.warning("cmi[{}/{}]: no estimator configs remain; skipping.",
                           ctx.arm or "-", used_split)
            continue

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "cmi[{}/{}]: configs={} from {} (max_samples={}, device={})",
            ctx.arm or "-", used_split, configs, ckpt_path.name, max_samples, device,
        )

        cells, n_seen = collect_cmi_anchors(
            model, runner, loader, latent_provider,
            cfg=cfg, configs=configs, meta=meta, max_samples=max_samples, seed=seed,
        )
        n_total = len(getattr(loader, "dataset", []) or [])
        summary = run_cmi_comparison(
            cells, cfg=cfg, configs=configs, horizon=int(runner.horizon),
            seed=seed, device=device,
        )
        summary["arm"] = ctx.arm
        summary["model_class"] = type(model).__name__
        summary["split"] = used_split
        summary["n_samples"] = int(n_seen)
        summary["n_samples_skipped"] = max(n_total - int(n_seen), 0)
        summary["latent_provider_available"] = latent_provider is not None

        with (split_dir / "cmi.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(summary), handle, indent=2)
        write_cmi_table(summary, split_dir / "cmi_table.csv")
        _render_figure(summary, split_dir / "figures")

        rec = summary.get("recovery") or {}
        if rec.get("available"):
            logger.info(
                "cmi[{}/{}]: rho(CMI_latent, TE_inj) = {} over {} cells; "
                "null |CMI| <= {}; factor-2 pass {}",
                ctx.arm or "-", used_split, rec.get("spearman_cmi_te_inj"), rec.get("n_cells"),
                rec.get("max_abs_null_cmi"), rec.get("factor2_pass_frac"),
            )
        bias = (summary.get("overall") or {}).get("cmi_bias")
        if bias:
            logger.info("cmi[{}/{}]: model-coupling bias = {:.4f} [{}, {}]",
                        ctx.arm or "-", used_split, bias["estimate"],
                        bias.get("ci_lo"), bias.get("ci_hi"))
            if not bias.get("reliable", True):
                logger.warning(
                    "cmi[{}/{}]: that bias is NOT a measurement -- {}. Affected cells: {}/{}.",
                    ctx.arm or "-", used_split, bias.get("reason"),
                    bias.get("n_cells_unreliable"), bias.get("n_cells"),
                )
    return 0


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------
_CONFIG_CAPTION = {
    CONFIG_LATENT: "ground-truth latents (model-free; absolute nats)",
    CONFIG_FEATURE_GT: "scattering features, ground-truth conditioning",
    CONFIG_FEATURE_MODEL: "scattering features, `target_state` conditioning",
}


def _render_cmi_section(ctx: SectionContext) -> List[str]:
    r"""Render the "Neural CMI" section, or ``n/a`` when the stage has not run."""
    lines = ["## Neural CMI (G-G)", ""]
    path = Path(ctx.results_dir) / "cmi.json"
    if not path.is_file():
        lines += ["> n/a — `--stage cmi` has not been run for this split.", ""]
        return lines
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "error" in payload:
        lines += [f"> n/a — {payload['error']}", ""]
        return lines

    configs = payload.get("configs") or []
    ceiling = payload.get("ceiling_nats")
    overall = payload.get("overall") or {}
    rec = payload.get("recovery") or {}

    lines += [
        "An estimate of $I(U_{\\le t}; Y^+_{t,1:H} \\mid c_t)$ that never reads the model's KL. "
        "Residualise $U$ and $Y^+$ on $c_t$, then bound the residual mutual information with a "
        "single InfoNCE critic. Only the `latent` row is an **absolute** nats claim: its variables "
        "are jointly Gaussian by construction, so the residualisation is exact.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| samples | {payload.get('n_samples', 'n/a')} "
        f"(skipped {payload.get('n_samples_skipped', 0)}) |",
        f"| InfoNCE ceiling $\\log K$ | {_fmt(ceiling, '.4f')} nats "
        f"(no absolute claim above {_fmt((ceiling or 0) * (payload.get('ceiling_claim_frac') or 0), '.3f')}) |",
    ]
    if rec.get("available"):
        lines += [
            f"| $\\rho(\\mathrm{{CMI}}_{{\\mathrm{{latent}}}}, \\mathrm{{TE}}_{{\\mathrm{{inj}}}})$ "
            f"| {_fmt(rec.get('spearman_cmi_te_inj'), '.4f')} (N={rec.get('n_cells', '?')}) |",
            f"| $\\max\\lvert \\mathrm{{CMI}} \\rvert$ on the null cells "
            f"| {_fmt(rec.get('max_abs_null_cmi'), '.4f')} |",
            f"| within a factor of 2 of $\\mathrm{{TE}}_{{\\mathrm{{inj}}}}$ "
            f"| {_fmt(rec.get('factor2_pass_frac'), '.0%')} of "
            f"{rec.get('n_absolute_claim_cells', '?')} below-ceiling cells |",
        ]
    bias = overall.get("cmi_bias")
    if bias:
        mark = "" if bias.get("reliable", True) else " — **unreliable**"
        lines.append(
            f"| model-coupling bias $\\mathrm{{CMI}}(\\texttt{{target\\_state}}) - "
            f"\\mathrm{{CMI}}(\\text{{GT}})$ | {_fmt(bias.get('estimate'))} "
            f"[{_fmt(bias.get('ci_lo'), '.3f')}, {_fmt(bias.get('ci_hi'), '.3f')}]{mark} |"
        )
    for name in configs:
        r2 = overall.get(f"cond_r2_{name}")
        if r2:
            lines.append(
                f"| held-out $R^2$ of $c_t$ on $(U, Y^+)$ — {_CONFIG_CAPTION.get(name, name)} "
                f"| {_fmt(r2.get('u'), '.3f')} / {_fmt(r2.get('v'), '.3f')} |"
            )
    lines.append("")
    if bias and not bias.get("reliable", True):
        lines += [
            f"> ⚠ **The bias above must not be read as a measurement.** {bias.get('reason')}. "
            f"Affected: {bias.get('n_cells_unreliable')} of {bias.get('n_cells')} cells.",
            "",
        ]

    rho_rows = [(name, overall.get(f"rho_kbar_cmi_{name}")) for name in configs]
    if any(r for _, r in rho_rows):
        note = " (fewer than 8 cells: reported, not gated)" \
            if overall.get("rho_reported_not_gated") else ""
        lines += [
            f"Rank correlation of the model's KL surrogate against each conditioning{note}. "
            "`target_state` conditioning couples the estimate to the model, so only rank-level "
            "corroboration is claimed for it.",
            "",
            "| conditioning | $\\rho(\\bar K, \\mathrm{CMI})$ | 95% CI | N |",
            "|---|---|---|---|",
        ]
        for name, entry in rho_rows:
            if not entry:
                continue
            ci = entry.get("ci") or [None, None]
            lines.append(
                f"| {_CONFIG_CAPTION.get(name, name)} | {_fmt(entry.get('rho'), '.3f')} | "
                f"[{_fmt(ci[0], '.3f')}, {_fmt(ci[1], '.3f')}] | {entry.get('n_cells', '?')} |"
            )
        lines.append("")

    per_cell = payload.get("per_cell") or {}
    if per_cell:
        head = ["cell", "$\\mathrm{TE}_{\\mathrm{inj}}$", "$D$", "$\\bar K$"]
        head += [f"CMI ({name})" for name in configs]
        if CONFIG_LATENT in configs:
            head.append("exact (Gaussian)")
        if any("bias" in c for c in per_cell.values()):
            head.append("bias")
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for cid in sorted(per_cell, key=lambda k: int(k)):
            cell = per_cell[cid]
            row = [str(cell.get("cell_id")), _fmt(cell.get("te_inj"), ".2f"),
                   str(cell.get("delay")), _fmt(cell.get("kbar"), ".3f")]
            for name in configs:
                block = cell.get(f"cmi_{name}") or {}
                mark = " †" if block.get("near_ceiling") else ""
                row.append(f"{_fmt(block.get('estimate'), '.3f')}{mark}")
            if CONFIG_LATENT in configs:
                row.append(_fmt(cell.get("cmi_latent_gauss_exact"), ".3f"))
            if any("bias" in c for c in per_cell.values()):
                row.append(_fmt((cell.get("bias") or {}).get("estimate"), ".3f"))
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "† within `ceiling_claim_frac` of the InfoNCE ceiling; rank-only.", ""]
    return lines


register_stage(
    StageSpec(
        "cmi", _STAGE_ORDER, False, True, run_cmi_stage,
        fatal=False,
        help="neural CMI in absolute nats vs TE_inj, and the target_state bias (opt-in)",
    )
)
register_section(SectionSpec("Neural CMI", _SECTION_ORDER, _render_cmi_section))
