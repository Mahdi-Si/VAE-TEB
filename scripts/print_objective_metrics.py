r"""Print every objective metric of the six shipped forecasters, at a fixed seed.

The four two-sided models share one objective, one pair of masks and one input adapter, and the
two causal cells reach the same objective through their tasks. Whenever those
shared modules gain a capability a new target domain needs, the models that already exist must
come out **bitwise where they were** -- and "bitwise" is not something a reader can judge from a
test report. This prints the evidence: the complete metric dictionary of every model, under both
likelihoods and both guard states, at full float precision.

The reference is the script's own output on the tree before the change; the check is a byte-level
diff against its output after. Committed decimal constants would not do, because they do not
survive the move between the development box and the production one -- both sides have to be
computed in one run of one process, which is what a diff of two runs of this script gives.

Run from the repository root:

    .venv/Scripts/python.exe scripts/print_objective_metrics.py > before.txt

There is nothing to configure: the geometries, the batch and the coefficients are all fixed here,
so the script takes no arguments and runs unchanged from an IDE's Run button.

The two causal cells are printed through their **tasks** rather than their nets, on both stages:
a tiled ``train`` step at the configured stride and the per-segment phase the task derives, and a
dense ``val`` step, which additionally runs the permutation control and the source-null readout.
The phases and the anchor index are printed beside the metrics, so a change that moved the tiling
without moving a metric is visible too.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import torch

# Run from the repository root; this makes the script work when invoked as `python scripts/...`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from teb_vae.lag_attn_cfs.tests.conftest import (  # noqa: E402
    make_stub_batch as make_causal_stub_batch,
    make_task as make_cfs_task,
)
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs  # noqa: E402
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws  # noqa: E402
from teb_vae.lag_attn_rws.tests.conftest import (  # noqa: E402
    TINY_KWARGS,
    make_stub_batch,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs  # noqa: E402
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (  # noqa: E402
    make_task as make_trf_cfs_task,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import (  # noqa: E402
    TINY_KWARGS as TRF_TINY_KWARGS,
)

#: Objective weights every term is exercised at. Mutually distinct and none of them a default: at
#: equal weights a term swapped for another would print the same numbers, and at a zero weight the
#: term is not computed at all, so its metric would be an exact zero that proves nothing.
COEFFICIENTS: Dict[str, float] = dict(
    beta=0.7,
    beta_prior=0.11,
    lambda_full=1.0,
    lambda_base=0.3,
    free_bits=0.05,
    lambda_ms=0.13,
    lambda_deriv=0.17,
    lambda_boundary=0.19,
)

#: The same weights with the three shape terms off. The feature-domain models ship them at zero --
#: their block's last axis counts channels, which have no order for a pooled trajectory or a first
#: difference to read -- so those two are printed at this set instead.
FEATURE_COEFFICIENTS: Dict[str, float] = {
    key: value
    for key, value in COEFFICIENTS.items()
    if key not in ("lambda_ms", "lambda_deriv", "lambda_boundary")
}

#: Seed of the posterior perturbation, and its scale. Without it the delta heads are still at their
#: zero initialisation, the posterior *is* the prior, and every KL-derived number below would print
#: an exact zero on a correct model and on a broken one alike.
PERTURB_SEED = 3
PERTURB_SCALE = 0.1


#: ``(label, class, ungated kwargs, gated kwargs, target domain)`` for each shipped forecaster.
#: The target domain decides which tensor is handed to ``compute_loss``: the raw trace for the two
#: raw-signal models, the concatenated feature stream for the two feature-domain ones.
#:
#: Both guard states are printed because they exercise different code: without a guard the input
#: adapter builds no availability buffer and no mask projection at all, so a change to either
#: would leave the ungated half of this output untouched.
MODELS: Tuple[Tuple[str, Any, Dict[str, Any], Dict[str, Any], str], ...] = (
    (
        "SeqVaeLagAttnRws",
        SeqVaeLagAttnRws,
        dict(TINY_KWARGS),
        tiny_gated_kwargs(),
        "raw",
    ),
    (
        "SeqVaeLagAttnTrfRws",
        SeqVaeLagAttnTrfRws,
        dict(TRF_TINY_KWARGS),
        tiny_gated_kwargs(TRF_TINY_KWARGS),
        "raw",
    ),
    (
        "SeqVaeLagAttnFs",
        SeqVaeLagAttnFs,
        dict(TINY_KWARGS),
        tiny_gated_kwargs(),
        "feature",
    ),
    (
        "SeqVaeLagAttnTrfFs",
        SeqVaeLagAttnTrfFs,
        dict(TRF_TINY_KWARGS),
        tiny_gated_kwargs(TRF_TINY_KWARGS),
        "feature",
    ),
)


#: ``(label, task factory)`` for the two causal cells. Each factory builds the guarded tiny model
#: at the tiling stride, wrapped in the cell's own task with the production loss hparams.
CAUSAL_TASKS: Tuple[Tuple[str, Any], ...] = (
    ("SeqVaeLagAttnCfs", make_cfs_task),
    ("SeqVaeLagAttnTrfCfs", make_trf_cfs_task),
)

#: The two stages a causal task resolves its anchor geometry from: a tiled train step and a dense
#: validation step.
CAUSAL_STAGES: Tuple[str, ...] = ("train", "val")


def _perturb_posterior(model: torch.nn.Module) -> None:
    """Break the zero-initialised posterior deterministically, so the KL readouts are non-vacuous.

    Args:
        model: The net, before any forward.
    """
    generator = torch.Generator().manual_seed(PERTURB_SEED)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * PERTURB_SCALE)


def metrics_of(
    model_cls: Any, kwargs: Dict[str, Any], domain: str, likelihood: str
) -> Dict[str, torch.Tensor]:
    """Build one model, run one forward, and score it.

    Every seed is set immediately before the operation that consumes it -- construction, then the
    forward's reparameterisation draw -- so the numbers do not depend on how many models ran first.

    Args:
        model_cls: The net class.
        kwargs: Its constructor keywords.
        domain: ``'raw'`` or ``'feature'``; decides the target tensor.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        The objective's metric dictionary.
    """
    batch = make_stub_batch()
    torch.manual_seed(0)
    model = model_cls(**kwargs).eval()
    _perturb_posterior(model)

    torch.manual_seed(0)
    with torch.no_grad():
        outputs = model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )
        target = (
            batch.fhr
            if domain == "raw"
            else torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
        )
        coefficients = COEFFICIENTS if domain == "raw" else FEATURE_COEFFICIENTS
        result = model.compute_loss(
            outputs, target, weight=batch.weight, likelihood=likelihood, **coefficients
        )
    return result["metrics"]


def causal_metrics_of(
    make_task: Any, stage: str
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build one causal task, run one step on the given stage, and return what it scored.

    Args:
        make_task: The cell's task factory.
        stage: ``'train'`` (tiled at the configured stride) or ``'val'`` (dense).

    Returns:
        ``(metrics, phase, anchor_index)``: the step's metric dictionary, the $(B,)$ phase the
        stage resolved and the $(B, A)$ anchor index the forward decoded at.
    """
    batch = make_causal_stub_batch()
    torch.manual_seed(0)
    task = make_task().eval()
    model = task.orig_model
    _perturb_posterior(model)

    phase, stride = task.resolve_anchor_geometry(stage, batch)
    anchors, _valid = model._build_anchor_index(
        batch.weight.shape[0], batch.weight.device, phase, stride
    )
    torch.manual_seed(0)
    with torch.no_grad():
        _loss, metrics = task.compute_loss_and_metrics(batch, 0, stage)
    return metrics, torch.as_tensor(phase).reshape(-1), anchors


def _print_metrics(metrics: Dict[str, Any]) -> None:
    """Print a metric dictionary sorted by name, one full-precision float per line.

    Args:
        metrics: Name to scalar tensor.
    """
    for name in sorted(metrics):
        # repr of a Python float round-trips exactly, so two runs that print the same text held
        # the same bits.
        print(f"  {name:<28} {float(metrics[name])!r}")


def main() -> int:
    """Print every metric of every model, in a stable order.

    Returns:
        The process exit code, always ``0``: the script reports, it does not judge.
    """
    for label, model_cls, ungated, gated, domain in MODELS:
        for guard, kwargs in (("ungated", ungated), ("gated", gated)):
            for likelihood in ("gaussian_nll", "mse"):
                print(f"{label} guard={guard} likelihood={likelihood}")
                _print_metrics(metrics_of(model_cls, kwargs, domain, likelihood))
    for label, make_task in CAUSAL_TASKS:
        for stage in CAUSAL_STAGES:
            print(f"{label} stage={stage}")
            metrics, phase, anchors = causal_metrics_of(make_task, stage)
            print(f"  {'anchor_phase':<28} {phase.tolist()!r}")
            print(f"  {'anchor_index':<28} {anchors.tolist()!r}")
            _print_metrics(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
