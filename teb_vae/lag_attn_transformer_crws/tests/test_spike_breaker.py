r"""The two loss-scale constants this encoder re-measured, bracketed against what the run recorded.

``main_loss`` here is a learned-variance Gaussian NLL summed over the $H \cdot R$ raw block and
averaged over the anchors the tiling decoded. The shipped breaker disables its relative test with a
floor far above any reachable loss and carries finite blow-up detection in ``additive_margin``, which
compares against the *raw* EMA and keeps working at a negative baseline. The breaker's mechanics and
the floor are the conv-LSTM cell of this row's -- the whole ``spike_breaker`` block is held equal to
that cell's by ``tests/test_config_load.py`` -- and are tested there.

What this encoder adds is a measurement. The block and the anchor count do not change across the
encoder edge, so both constants *could* have transferred, and the only way to know whether they do
is to run this encoder and look. The recorded distribution lives here, and each shipped constant is
bracketed against it, so a value edited outside the bracket fails rather than moving the goalposts
with itself.
"""
from __future__ import annotations

from pathlib import Path

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: The run every number below comes from: ``configs/smoke_causal.yaml`` at the shipped widths over
#: the committed causal fixture, with the clip parked so nothing rescaled the steps the norms were
#: drawn from and the step-granular ramp shortened so most of the run sits at the shipped learning
#: rate. Two regimes were recorded -- the whole committed shard in one batch, the closest reachable
#: analogue of a large-batch production step, and one sample per batch, the noisiest regime the
#: fixture can produce. Neither run skipped a batch or bound the clip, so both records describe the
#: objective rather than the guards that were watching it.
#:
#: Every constant here is **provisional**: four in-sample windows are a thinner tail than a
#: production run's, so the distribution describes a memorised window rather than the production
#: objective.

#: The largest excursion of ``main_loss`` above the EMA the breaker carried into the step, after the
#: priming window, in the **noisiest** regime -- which is the one the margin has to survive.
MEASURED_EXCURSION_MAX = 1598.1

#: The range ``main_loss`` covered over the two instrumented regimes. The margin has to sit under
#: it for the additive test to be able to fire at all.
MEASURED_LOSS_SPAN = 3.3e3

#: Pre-clip ``train/grad_norm`` over the same run, **whole-shard batch**: the regime the shipped
#: batch is the analogue of, and therefore the one the clip is set from.
MEASURED_GRAD_Q99 = 11078.3
MEASURED_GRAD_MAX = 13180.0


def _shipped() -> dict:
    return load_config(str(_CONFIG))


def test_the_margin_clears_the_worst_measured_excursion_and_can_still_fire() -> None:
    r"""Both sides of the bracket. Below the worst excursion the instrumented run measured, the
    breaker skips ordinary batches -- and a run that skipped ordinary batches reads in a log exactly
    like one that keeps blowing up. Above the loss span the objective covered, nothing short of a
    genuine divergence exceeds $\mathrm{EMA} + \mathrm{margin}$ and the breaker degenerates to its
    non-finite guard alone, with nothing in the log saying so."""
    margin = float(_shipped()["advanced_config"]["spike_breaker"]["additive_margin"])

    assert MEASURED_EXCURSION_MAX < margin < MEASURED_LOSS_SPAN


def test_the_clip_clears_the_gradient_distribution_the_instrumented_run_measured() -> None:
    """``gradient_clip_val`` is the smallest round value above the pre-clip norm's measured q99, so
    the clip bites on the tail rather than on the body of the distribution -- a threshold below the
    body rescales most steps and turns the optimizer into a sign-descent method with nothing in the
    log saying so."""
    clip = float(_shipped()["advanced_config"]["trainer"]["gradient_clip_val"])

    assert MEASURED_GRAD_Q99 < clip <= MEASURED_GRAD_MAX
