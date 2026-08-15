r"""The one import site for the shared, model-free evaluation primitives.

``teb_vae/lag_attn/eval/`` already owns a tested set of pieces that say nothing about any model in
the family: the pinned numeric environment, the rank statistics, the clinical labelling, the
``summary.json`` assembly and fail-soft step wrapper, the figure primitives, and the stratified
subsampler. Reimplementing them here would produce two implementations of the same arithmetic
with nothing keeping them equal -- and reuse across packages is this module's established
convention, not a new one: ``nets/controls.py`` already states it outright for the derangement
machinery.

**This package is a fork of** ``teb_vae/lag_attn_rws/eval``, and these primitives are deliberately
**not** part of that fork. They are bound, not copied, and the reason is sharper than "less code":
forking them would fork the definition of a cohort. ``labels.CLASS_NAMES`` and
``labels.CANONICAL_SUBGROUPS`` decide which recordings are compared against which, and two
packages carrying two copies of that decision would produce two summaries that read as though
they described the same population while describing different ones. The same argument applies to
the bootstrap, to Holm's correction and to the rank tests: a fork of an *interval* is a fork of
what "significant" means.

So they are bound here rather than forked, and this module is the only place the shared evaluation
package is named. ``tests/test_eval_self_contained.py`` enforces exactly that: the modules listed
in its allow-list may be imported, every other module under ``teb_vae.lag_attn.eval`` may not, and
a reach into that package's ``runner``, ``metrics`` or ``analyses`` is reported wherever it
appears. That test additionally forbids ``teb_vae.lag_attn_rws.eval`` outright -- the pipeline this
one was forked from is not reachable sideways, because a half-fork is worse than either whole.

Two allowed modules are deliberately **not** bound here:

* ``config_schema``. Its two validators are imported directly by this package's own
  :mod:`~teb_vae.lag_attn_cfs.eval.config_schema`, which must stay a stdlib-only import: it
  validates a run's settings *before* a model, a loader or an output directory exists, and
  importing it should not cost matplotlib.
* ``collectors``. It imports the shared package's ``runner``, which imports that package's own
  network module, so binding it would make every import of this seam pull in a model this package
  does not evaluate. What ``CollectionPlan`` actually owns is the seeded draw, and that is
  :func:`subsample_indices`, which is bound below and is the same arithmetic.

``band_partition`` is bound for the same reason the rest are, and the reason is worth stating
because it is the one whose *subject* differs: it reads a shard's own ``sel_*`` provenance and
turns it into a channel-to-band map, and its arithmetic -- the harmonic grid, the descending
filter bank, the clinical band edges -- is a property of the dataset pipeline rather than of any
model. It takes the phase block it partitions as an argument, so the same function describes this
model's target stream and its source stream; what is written here is only which two blocks to ask
for and how to lay them out side by side.

Importing this module sets matplotlib's ``Agg`` backend, because :mod:`figures` does. It does
**not** restyle anything: ``figures.configure_figure_style()`` is called once at run start.
"""
from __future__ import annotations

from teb_vae.lag_attn.eval import band_partition, figures, labels, masks, numerics, report, stats
from teb_vae.lag_attn.eval.masks import subsample_indices
from teb_vae.lag_attn.eval.numerics import configure_numerics

__all__ = [
    # Modules, bound whole because every one of them is used by name in more than one place.
    "band_partition",
    "figures",
    "labels",
    "masks",
    "numerics",
    "report",
    "stats",
    # The two single-symbol bindings: nothing else in ``numerics`` or ``masks`` is used here.
    "configure_numerics",
    "subsample_indices",
]
