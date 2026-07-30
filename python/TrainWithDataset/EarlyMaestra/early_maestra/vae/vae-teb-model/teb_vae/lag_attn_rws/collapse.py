r"""The numeric collapse criterion for a completed training run.

A *completed* run is **collapsed** when either

1. ``val/source_conditioned_kl_raw`` is below :data:`KL_COLLAPSE_THRESHOLD_NATS` at every one of
   its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs, or
2. its final ``val/kld_active_frac`` is below :data:`KL_COLLAPSE_MIN_ACTIVE_DIMS` $/\, d_z$.

The two clauses are one statement at the same threshold: the latent finished carrying less than
two dimensions' worth of source information, in total nats per anchor (clause 1) or in
active-dimension count (clause 2). The criterion reads the *tail* of the run, never an early
window: the KL starts at exactly $0$ by construction (the zero-initialised posterior residual)
and the $\beta$ warm-up holds it there deliberately, so an any-window reading would classify
every healthy run as collapsed. It presumes the run trained at least the patience length, which
every sweep arm's stated minimum epoch count exceeds.

**Why this lives at the package root rather than under ``eval/``, and why the threshold is a
literal.** Two consumers need it and neither may reach the other: the sweep-config lint is a
model test that must not import the evaluation package, and the evaluation's offline acceptance
gate must stay importable on a box with no ``torch`` installed. So this module depends on
nothing -- not ``torch``, not ``nets`` -- and
:data:`KL_COLLAPSE_THRESHOLD_NATS` restates the product
``KL_COLLAPSE_MIN_ACTIVE_DIMS * KLD_ACTIVE_EPS`` rather than importing
:data:`~teb_vae.lag_attn_rws.nets.losses.KLD_ACTIVE_EPS` to compute it. The coupling survives as
a *checked invariant*: ``tests/test_eval_collapse.py`` pins the literal equal to that product, so
a change to the per-dimension activity epsilon fails a test instead of silently drifting.
"""
from __future__ import annotations

from typing import Sequence

#: Consecutive final epochs the raw KL must stay below the threshold for clause 1 to fire.
KL_COLLAPSE_PATIENCE_EPOCHS = 5

#: The active-dimension floor of clause 2: fewer than this many active dimensions at the end of
#: the run is a collapsed latent, whatever the total KL reads.
KL_COLLAPSE_MIN_ACTIVE_DIMS = 2

#: Clause 1's threshold in nats per anchor: the total KL of a latent whose information sits
#: entirely in :data:`KL_COLLAPSE_MIN_ACTIVE_DIMS` dimensions, each barely clearing the
#: per-dimension activity epsilon the training metric ``kld_active_frac`` counts against
#: (``nets/losses.py::KLD_ACTIVE_EPS = 1e-2``). Written out rather than computed; see the module
#: docstring for why, and ``tests/test_eval_collapse.py`` for the pin that keeps the two equal.
KL_COLLAPSE_THRESHOLD_NATS = 0.02


def is_collapsed(
    kl_raw_per_epoch: Sequence[float],
    kld_active_frac_per_epoch: Sequence[float],
    d_z: int,
) -> bool:
    r"""Apply the collapse criterion to a completed run's per-epoch validation series.

    Args:
        kl_raw_per_epoch: The ``val/source_conditioned_kl_raw`` column of the run's metrics CSV,
            in epoch order. Nats per anchor, summed over $d_z$.
        kld_active_frac_per_epoch: The ``val/kld_active_frac`` column, in epoch order.
        d_z: The arm's latent width, from its resolved configuration.

    Returns:
        Whether the run is collapsed under either clause.
    """
    kl_tail = list(kl_raw_per_epoch)[-KL_COLLAPSE_PATIENCE_EPOCHS:]
    # A run shorter than the patience length cannot satisfy clause 1: the tail would be the whole
    # series, which always includes the deliberate zero-KL opening.
    kl_dead = len(kl_tail) == KL_COLLAPSE_PATIENCE_EPOCHS and all(
        value < KL_COLLAPSE_THRESHOLD_NATS for value in kl_tail
    )

    active_frac = list(kld_active_frac_per_epoch)
    dims_dead = bool(active_frac) and (
        active_frac[-1] < KL_COLLAPSE_MIN_ACTIVE_DIMS / float(d_z)
    )
    return kl_dead or dims_dead
