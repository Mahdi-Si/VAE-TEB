r"""Frozen probes on one checkpoint's latent: is anything about the future readable from $Z$?

Run from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.eval.latent_probes \
        --checkpoint output/<run>/model_checkpoints/<name>.ckpt

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.

**What this pass answers, and why the scoring pass cannot.** A predictive gap says whether the
source-conditioned branch predicts better. It says nothing about whether the latent it predicts
through carries anything: a decoder can improve its score by sharpening an observation variance
while its latent stays inert, and a bounded residual correction can overwrite a coordinate the
target-only prior was using rather than adding to it. Both are invisible to a gap and both are
visible here, because a probe reads the latent alone.

**The target is the future block relative to the anchor's own values**, not the block itself. A
forecast's easiest part is that the future looks like the present, and a probe scored against the
raw block reports that similarity whatever the latent holds. Subtracting the anchor leaves the part
of the horizon a persistence prediction does not already give, which is the quantity design
requires be measured. The subtraction uses the anchor's stored values rather than the decoder's
learned persistence weights **on purpose**: a baseline that moved with the model being probed would
make two arms' figures incomparable, and comparability across arms is the whole use of this pass.

**Six probes rather than one.** The distribution parameters and a draw from them are different
questions -- information that survives only in an effectively deterministic code is real and is
invisible to a probe on samples -- and the prior's and the full distribution's answers are what say
whether the source correction added to the target-only representation or wrote over it. Scales are
probed beside means because a conditional variance that tracks the horizon is itself information.

**Nothing here is fitted on what it is scored on.** Recordings are split by a stable digest of
their identifier, the probe is fitted on one side and scored on the other, and the null it is
scored against is the fit side's own mean -- the prediction a model that had seen only the fitting
recordings would make. An in-sample coefficient of determination on a $64$-dimensional design over
millions of anchors would be a measurement of the fit rather than of the latent.

**Every anchor is used and none is stored.** The probe accumulates the design's second moments
streaming, per split, and solves from those, so the fit is exact over the whole split at a memory
cost that does not grow with it. A subsample would have been the alternative and it would have put
a sampling policy between the split and the number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/eval/latent_probes.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script -- which is what an IDE's Run button does -- this file's own directory goes
# on sys.path instead of the repository root, and every absolute import below fails before
# ``__main__`` is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.eval.numerics import configure_numerics  # noqa: E402
from teb_vae.lag_attn.eval.report import json_safe  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import missing_required, resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_cfs.eval.metrics import batch_guids, model_inputs  # noqa: E402
from teb_vae.lag_attn_cfs.eval.probe import (  # noqa: E402
    load_task,
    read_checkpoint,
    resolve_device,
)
from teb_vae.lag_attn_cfs.eval.run import dump_resolved_config, make_output_dir  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval.binding import (  # noqa: E402
    LAG_RESIDUAL_BINDING,
    MODEL_KIND,
)
from teb_vae.lag_slot_transformer_cfs.eval.run import (  # noqa: E402
    DENSE_ANCHOR_GEOMETRY,
    build_run_config,
)
from train.data_module import GraphDataModule  # noqa: E402

#: The artifact this pass writes.
PROBE_FILENAME = "latent_probes.json"

#: Percentage of recordings the probe is FITTED on; the rest are what it is scored on.
#:
#: Split by recording rather than by anchor, and that is the whole validity of the number: anchors
#: inside one recording share their history and overlap in all but one step of their horizons, so a
#: probe fitted on some anchors of a recording and scored on the rest of the same recording would
#: report how well it interpolated a record it had already seen.
PROBE_FIT_PERCENT = 60

#: Ridge penalty, in **standardised** units: the design's correlation matrix has a unit diagonal,
#: so this number means the same thing whatever the latent's coordinates are scaled like.
#:
#: Small rather than tuned. The probe exists to say whether the future is linearly readable at all,
#: and a penalty chosen by a search over held-out scores would make it a small fitted model whose
#: capacity is one more thing to hold matched between two arms.
RIDGE_ALPHA = 1e-2

#: The six readouts probed, each fitted and scored independently.
#:
#: ``*_mean`` and ``*_scale`` are the two distributions' parameters; ``*_sample`` is one draw from
#: each, under the shared noise the forward already drew. Prior against full is what says whether
#: the source correction added to the target-only representation or overwrote part of it.
FEATURE_SETS: Tuple[str, ...] = (
    "prior_mean",
    "full_mean",
    "prior_scale",
    "full_scale",
    "prior_sample",
    "full_sample",
)

#: Below this many recordings on either side, a split reproduces its own sample rather than
#: estimating anything, and the pass reports the counts and no coefficient.
MIN_RECORDINGS_PER_SIDE = 2


def split_of(guid: str, *, fit_percent: int = PROBE_FIT_PERCENT) -> str:
    """Which side of the probe's own split a recording falls on.

    A digest of the identifier rather than a shuffle: the assignment has to be the same in every
    run of every arm, or two arms' probe scores would be fitted on two different cohorts and their
    difference would carry that.

    Args:
        guid: The recording identifier.
        fit_percent: Percentage of recordings assigned to the fitting side.

    Returns:
        ``'fit'`` or ``'score'``.
    """
    digest = hashlib.sha256(str(guid).encode("utf-8")).hexdigest()
    return "fit" if int(digest[:8], 16) % 100 < int(fit_percent) else "score"


class DesignMoments:
    r"""Streaming second moments of one probe's design against the shared target block.

    Holds $n$, $\sum_i x_i$, $\sum_i x_i x_i^\top$, $\sum_i y_i$, $\sum_i x_i y_i^\top$ and
    $\sum_i y_i \odot y_i$ -- everything a ridge fit and a coefficient of determination need, and
    nothing that grows with the number of rows. Accumulated in FP64: the sums run to millions of
    anchors and the residual sum of squares is a difference of two such totals.

    Args:
        n_features: Latent width $d_z$.
        n_targets: Coefficients in the block, $H \cdot C_Y$.
    """

    def __init__(self, n_features: int, n_targets: int) -> None:
        """Allocate the accumulators at the design's widths.

        Args:
            n_features: Latent width $d_z$.
            n_targets: Coefficients in the block, $H \\cdot C_Y$.
        """
        self.n = 0.0
        self.sum_x = np.zeros(n_features, dtype=np.float64)
        self.sum_xx = np.zeros((n_features, n_features), dtype=np.float64)
        self.sum_y = np.zeros(n_targets, dtype=np.float64)
        self.sum_xy = np.zeros((n_features, n_targets), dtype=np.float64)
        self.sum_yy = np.zeros(n_targets, dtype=np.float64)

    def add(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Accumulate one block of rows.

        Args:
            features: $(n, d_z)$ rows.
            targets: $(n, H C_Y)$ rows.
        """
        if features.shape[0] == 0:
            return
        self.n += float(features.shape[0])
        self.sum_x += features.sum(axis=0)
        self.sum_xx += features.T @ features
        self.sum_y += targets.sum(axis=0)
        self.sum_xy += features.T @ targets
        self.sum_yy += (targets * targets).sum(axis=0)


def ridge_from_moments(
    moments: DesignMoments, *, alpha: float = RIDGE_ALPHA
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Solve the ridge normal equations from accumulated moments.

    On the centred and scaled design,

    $$\hat\beta_{\rm std} = (R + \alpha I)^{-1} D^{-1}\hat C_{xy},
      \qquad R = D^{-1}\hat C_{xx} D^{-1},$$

    with $\hat C$ the fitting split's own covariances and $D$ the diagonal of its standard
    deviations. The intercept is $\bar y - \bar x^\top \beta$, so the model reduces to the fitting
    split's mean when every coefficient is zero -- which is the null the score below is taken
    against.

    A coordinate with no variance on the fitting split gets a unit scale and a zero column: its
    covariance row is already zero, so it contributes nothing, and dividing by its standard
    deviation would be a division by zero in the one case where the answer is known.

    Args:
        moments: The fitting split's moments.
        alpha: Ridge penalty in standardised units.

    Returns:
        ``(coefficients, intercept)``, shaped $(d_z, m)$ and $(m,)$.
    """
    n = max(moments.n, 1.0)
    mean_x = moments.sum_x / n
    mean_y = moments.sum_y / n
    cov_xx = moments.sum_xx / n - np.outer(mean_x, mean_x)
    cov_xy = moments.sum_xy / n - np.outer(mean_x, mean_y)

    scale = np.sqrt(np.clip(np.diag(cov_xx), 0.0, None))
    inert = scale <= 0.0
    scale = np.where(inert, 1.0, scale)
    inverse = np.diag(1.0 / scale)

    correlation = inverse @ cov_xx @ inverse
    standardised = np.linalg.solve(
        correlation + float(alpha) * np.eye(correlation.shape[0]), inverse @ cov_xy
    )
    beta = inverse @ standardised
    beta[inert, :] = 0.0
    return beta, mean_y - beta.T @ mean_x


def explained_variance(
    fit: DesignMoments,
    score: DesignMoments,
    beta: np.ndarray,
    intercept: np.ndarray,
) -> Dict[str, np.ndarray]:
    r"""Out-of-sample residual and total sums of squares, per target coefficient.

    Both are expanded into the score split's own moments rather than computed from stored rows:

    $$\mathrm{SS}_{\rm res} = \sum y^2 - 2a\sum y - 2\beta^\top\!\sum xy + na^2
      + 2a\,\beta^\top\!\sum x + \beta^\top\!\left(\sum xx\right)\!\beta,$$

    and the total is the same expression for the constant model $a = \bar y_{\rm fit}$, which is
    the prediction a probe that had seen only the fitting recordings would make.

    Args:
        fit: The fitting split's moments, for the null.
        score: The scoring split's moments.
        beta: Ridge coefficients.
        intercept: Ridge intercept.

    Returns:
        ``{'ss_res', 'ss_tot'}``, each $(m,)$.
    """
    quadratic = np.einsum("jm,jk,km->m", beta, score.sum_xx, beta)
    cross = np.einsum("jm,jm->m", beta, score.sum_xy)
    linear = beta.T @ score.sum_x
    ss_res = (
        score.sum_yy
        - 2.0 * intercept * score.sum_y
        - 2.0 * cross
        + score.n * intercept * intercept
        + 2.0 * intercept * linear
        + quadratic
    )
    null = fit.sum_y / max(fit.n, 1.0)
    ss_tot = score.sum_yy - 2.0 * null * score.sum_y + score.n * null * null
    return {"ss_res": ss_res, "ss_tot": ss_tot}


def probe_report(
    sums: Mapping[str, np.ndarray], *, horizon: int
) -> Dict[str, Any]:
    r"""Turn one probe's sums of squares into the block a reader reads.

    $$R^2 = 1 - \frac{\sum_c \mathrm{SS}^{\rm res}_c}{\sum_c \mathrm{SS}^{\rm tot}_c}$$

    pooled over the block, and again within each horizon step. Per horizon step rather than per
    channel, because a step is the axis along which "beyond persistence" changes meaning: the first
    step is nearly the anchor and the last is the part of the window a persistence prediction is
    worst at.

    Negative values are reported as they come. A probe fitted on other recordings can predict worse
    than their mean, and clamping that at zero would hide the one outcome that says the latent
    carries nothing linearly readable.

    Args:
        sums: The residual and total sums of squares.
        horizon: Forecast steps $H$, which divides the block.

    Returns:
        The report block.
    """
    ss_res, ss_tot = np.asarray(sums["ss_res"]), np.asarray(sums["ss_tot"])
    per_step_res = ss_res.reshape(int(horizon), -1).sum(axis=1)
    per_step_tot = ss_tot.reshape(int(horizon), -1).sum(axis=1)
    total = float(ss_tot.sum())
    return {
        "r2": float("nan") if total <= 0.0 else 1.0 - float(ss_res.sum()) / total,
        "r2_per_horizon_step": [
            float("nan") if denominator <= 0.0 else 1.0 - float(numerator) / float(denominator)
            for numerator, denominator in zip(per_step_res, per_step_tot)
        ],
    }


def latent_features(outputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    r"""The six probed readouts, each $(B, A, d_z)$.

    Scales are $\sigma = \exp(\lambda/2)$ rather than the log-variance the forward carries, so a
    probe reads the quantity the sampling used.

    Args:
        outputs: The forward's returned tensors.

    Returns:
        ``{name: tensor}`` over :data:`FEATURE_SETS`.
    """
    return {
        "prior_mean": outputs["mu_prior"],
        "full_mean": outputs["mu_post"],
        "prior_scale": torch.exp(0.5 * outputs["logvar_prior"]),
        "full_scale": torch.exp(0.5 * outputs["logvar_post"]),
        "prior_sample": outputs["z_prior"],
        "full_sample": outputs["z_post"],
    }


def probe_batch(
    task: Any, batch: Any, moments: Mapping[str, Mapping[str, DesignMoments]]
) -> Dict[str, Any]:
    """Run one batch and accumulate it into whichever split each recording belongs to.

    Args:
        task: The loaded task.
        batch: A batch already on the model's device.
        moments: ``{split: {feature set: moments}}``, accumulated in place.

    Returns:
        ``{'anchors': {side: count}, 'recordings': {side: set}}``, so the pass can report what it
        was fitted and scored on rather than assert it from the way the split was drawn.
    """
    model = task.orig_model
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)

    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    mask, _coverage = forecast_mask(
        model.scored_weight(weight),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchors,
        anchor_valid=anchor_valid,
    )
    # Complete-coverage anchors only. A partially observed block would leave the probe fitting some
    # coefficients on fewer rows than others, and the pooled figure would then weight the
    # coefficients by how often they happened to be observed.
    complete = mask.sum(dim=-1) >= mask.shape[-1]

    # The anchor's own stored values, gathered by the model's own routine so the probe subtracts
    # exactly the vector the persistence term carries -- the same channels in the same order.
    at_anchor = model._anchor_target_values(torch.cat([y_st, y_ph], dim=-1), anchors)
    residual = (target - at_anchor.unsqueeze(2)).flatten(start_dim=2)

    features = latent_features(outputs)
    guids = batch_guids(batch, int(y_st.shape[0]))
    anchors = {"fit": 0, "score": 0}
    recordings: Dict[str, set] = {"fit": set(), "score": set()}
    for position, guid in enumerate(guids):
        rows = complete[position]
        if not bool(rows.any()):
            continue
        side = split_of(str(guid))
        anchors[side] += int(rows.sum())
        recordings[side].add(str(guid))
        block = residual[position][rows].to(torch.float64).cpu().numpy()
        for name in FEATURE_SETS:
            moments[side][name].add(
                features[name][position][rows].to(torch.float64).cpu().numpy(), block
            )
    return {"anchors": anchors, "recordings": recordings}


def run_pass(
    task: Any, loader: Any, *, max_batches: Optional[int] = None
) -> Dict[str, Any]:
    """Walk the split once, fit each probe on one side and score it on the other.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        max_batches: Stop after this many batches, or ``None`` for the whole split.

    Returns:
        The assembled report.
    """
    model = task.orig_model
    horizon, latent = int(model.horizon), int(model.d_z)
    targets = horizon * int(model.decoder_out_channels)
    moments = {
        side: {name: DesignMoments(latent, targets) for name in FEATURE_SETS}
        for side in ("fit", "score")
    }
    anchors = {"fit": 0, "score": 0}
    recordings: Dict[str, set] = {"fit": set(), "score": set()}

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= int(max_batches):
                break
            batch = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            gained = probe_batch(task, batch, moments)
            for side in ("fit", "score"):
                anchors[side] += int(gained["anchors"][side])
                recordings[side].update(gained["recordings"][side])
            logger.info(f"probed batch {index + 1}")

    probes: Dict[str, Any] = {}
    for name in FEATURE_SETS:
        fit, score = moments["fit"][name], moments["score"][name]
        if fit.n <= 0.0 or score.n <= 0.0:
            probes[name] = {
                "r2": None,
                "r2_per_horizon_step": [],
                "note": (
                    "one side of the split accumulated no complete-coverage anchor, so nothing "
                    "was fitted or scored. Counts, not a coefficient: an absent measurement and a "
                    "measured zero must not read the same."
                ),
            }
            continue
        beta, intercept = ridge_from_moments(fit)
        probes[name] = probe_report(
            explained_variance(fit, score, beta, intercept), horizon=horizon
        )
    return {
        "probes": probes,
        "counts": {
            "fit_anchors": int(anchors["fit"]),
            "score_anchors": int(anchors["score"]),
            "fit_recordings": len(recordings["fit"]),
            "score_recordings": len(recordings["score"]),
            "block_coefficients": targets,
            "latent_width": latent,
            # The two sides are disjoint by construction -- one digest of one identifier decides
            # which side a recording falls on -- and the count is reported so a reader can see
            # that a side is populated rather than take the construction on trust.
            "sides_are_disjoint": not (recordings["fit"] & recordings["score"]),
            "below_minimum_recordings": bool(
                min(len(recordings["fit"]), len(recordings["score"]))
                < MIN_RECORDINGS_PER_SIDE
            ),
        },
        "settings": {
            "fit_percent": PROBE_FIT_PERCENT,
            "ridge_alpha": RIDGE_ALPHA,
            "split_rule": "sha256 of the recording identifier, first 32 bits, modulo 100",
            "target": (
                "the forecast block less the anchor's own stored values, so a probe scores what a "
                "persistence prediction does not already give"
            ),
            "null_model": "the fitting split's own per-coefficient mean",
            "anchors": "complete-coverage anchors only",
        },
        "qualification": (
            "A coefficient of determination here says how much of the anchor-relative future block "
            "is LINEARLY readable from this readout on held-out recordings. It is a lower bound on "
            "what the latent holds, not a measurement of what the decoder uses, and a difference "
            "between the prior's and the full distribution's figures is not a measurement of "
            "source information: the two are read by one probe family on one cohort, and the "
            "predictive score is what says whether a difference between them helped."
        ),
    }


def main(
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    overrides: Optional[str] = None,
    sources: Optional[Mapping[str, str]] = None,
) -> int:
    """Probe one checkpoint's latent and write :data:`PROBE_FILENAME`.

    Args:
        checkpoint: The ``.ckpt`` to probe. Required, enforced here rather than by argparse.
        output_dir: An explicit run directory, or ``None`` for a timestamped one.
        device: Device string, or ``None`` to choose automatically.
        max_batches: Stop after this many batches, or ``None`` for the whole split.
        overrides: An override delta path, or ``None`` for this package's committed one.
        sources: Where each launch value came from, recorded beside the results.

    Returns:
        The process exit code: ``0`` on success, ``2`` on a refusal.
    """
    refusal = missing_required({"checkpoint": checkpoint}, ("checkpoint",))
    if refusal is not None:
        logger.error(refusal)
        return 2

    config = build_run_config(checkpoint, overrides)
    eval_config = config["eval_config"]
    configure_numerics(int(eval_config["seed"]))
    resolved_device = resolve_device(device)

    results_dir = make_output_dir(config, output_dir, binding=LAG_RESIDUAL_BINDING)
    logger.info(f"writing results to {results_dir}")
    dump_resolved_config(config, results_dir)

    blob = read_checkpoint(checkpoint)
    task = load_task(checkpoint, resolved_device, blob=blob, binding=LAG_RESIDUAL_BINDING)
    loader = GraphDataModule(config).test_dataloader()

    results = run_pass(task, loader, max_batches=max_batches)
    results["run"] = {
        "checkpoint": str(checkpoint),
        "model_kind": MODEL_KIND,
        "device": str(resolved_device),
        # The same seed the scoring pass uses, and it matters here for one readout: the sampled
        # probes read the draw the forward made, so an unseeded pass would probe a different code.
        "seed": int(eval_config["seed"]),
        "training_seed": int(config.get("general_config", {}).get("seed", -1)),
        "training_tag": str(config.get("general_config", {}).get("tag", "")),
        "argument_sources": dict(sources or {}),
    }

    path = results_dir / PROBE_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(results), handle, indent=2)
    logger.info(f"wrote {path}")
    return 0


#: Values used when the module is launched with no command line -- i.e. an IDE's Run button. Keyed
#: by argparse ``dest``. A flag on the command line always wins over the entry here, **per key**.
#:
#: ``checkpoint`` MUST be filled in for this file to run at all; everything else may stay ``None``.
#:
#: Nothing that shapes what the probe measures is here. The split rule, the penalty and the target
#: are module constants recorded in the output, because a probe whose cohort or capacity could be
#: set at launch would not be comparable between two arms -- which is the only use it has.
RUN_ARGS: Dict[str, Any] = {
    # REQUIRED. Path to the .ckpt to probe, repo-root-relative or absolute.
    "checkpoint": None,
    # An explicit run directory, or None for a timestamped one under the config's out_dir_base.
    "output_dir": None,
    # 'cuda:0', 'cpu', or None to choose automatically.
    "device": None,
    # Stop after this many batches. None probes the whole split; a small number is a smoke run.
    "max_batches": None,
    # An override delta path, or None for this package's committed eval_overrides.yaml.
    "overrides": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Build this entry point's own parser.

    No ``required=True`` and no non-``None`` default, for the reasons the scoring pass's parser
    records: the first fires before the launch dict is read, and the second makes that key's entry
    unreachable while the operator edits it.

    Returns:
        The parser, whose ``dest`` set is also the valid key set for :data:`RUN_ARGS`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.eval.latent_probes",
        description="Fit frozen probes on a lag-residual checkpoint's latent.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to the .ckpt to probe.")
    parser.add_argument("--output-dir", default=None, help="Explicit run directory.")
    parser.add_argument("--device", default=None, help="'cuda:0', 'cpu', or omit to choose.")
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Stop after this many batches."
    )
    parser.add_argument("--overrides", default=None, help="Override delta path.")
    return parser


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, merge with :data:`RUN_ARGS`, and run.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    # The paths inside a config are repo-root-relative, and under an IDE Run button the working
    # directory is whatever the IDE chose.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(f"argument sources: {sources}")
    return main(**values, sources=sources)


if __name__ == "__main__":
    sys.exit(_cli())
