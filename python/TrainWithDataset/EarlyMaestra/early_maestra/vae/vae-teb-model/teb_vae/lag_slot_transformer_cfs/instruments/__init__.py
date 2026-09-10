r"""Synthetic instruments: generators whose answer is known, and the criteria declared before them.

Every number this architecture reports on real recordings is a measurement whose truth is unknown.
An instrument is the opposite arrangement -- a process whose source-to-target dependence is put
there on purpose, so that what the model and its readouts say can be scored against what is
actually true. Two things then become measurable that a real split cannot supply:

**Power.** On a generator that really does carry source information, how often does the readout
find it? A readout that misses a planted dependence at a signal strength the physiology plausibly
has is not evidence of absence when it reports nothing on real data.

**False-positive rate.** On a generator where the source carries **no** conditional information --
because it is a deterministic function of available target history, or a constant, or absent
entirely -- how often does the readout claim it does? This is the failure the evidence behind this
architecture actually exhibits: a source-conditioned branch can appear to help by correcting a
baseline that was weak for reasons having nothing to do with the source.

**What makes these instruments rather than demonstrations.** Every criterion is declared with its
generator, *before* any fit, and the campaign reads it from that declaration rather than from the
run. Every generator is run at several seeds and the rates are reported over all of them, including
the ones that failed. A displayed example that worked is not a measurement.

**What they cannot tell you.** They exercise a reduced fit -- this package's real forward, real
objective and real readouts under a plain optimizer, with no scheduler, no divergence ramp beyond
a linear one and no early stopping. That is deliberate and it is a limitation: a rate measured here
is a rate for that fit. What it does establish is whether a readout can find a planted dependence
at all, and whether it claims one where none exists, which is the question a production run cannot
answer about itself.

```
lean-limit: the instrument campaign fits under a plain optimizer rather than the production
schedule, so its power and false-positive rates describe that fit; reconsider when a rate here
disagrees with a production arm, likely option driving the campaign through the trainer with a
generated shard once the generators have earned that cost.
```
"""
from __future__ import annotations

__all__: list = []
