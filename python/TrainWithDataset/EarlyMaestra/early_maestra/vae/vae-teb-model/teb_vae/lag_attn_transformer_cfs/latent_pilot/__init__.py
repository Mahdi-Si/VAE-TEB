r"""Outcome-separation pilot: a tiny supervised adaptation of one pretrained CFS checkpoint.

The question this package exists to answer, on **one** fold, **one** checkpoint and **one** seed:
can a small supervised change to the pretrained posterior mean improve held-out separation of
healthy against adverse-outcome recordings near delivery, while retaining useful variation over the
preceding three hours? It fits a frozen linear baseline on pooled ``mu_post`` summaries, then
fine-tunes ``posterior_head.delta_mu_head`` together with the same kind of classifier, and compares
the two on recordings withheld from fitting and selection.

**Scope boundary, and it is a hard one.** Everything this pilot adds lives under this directory --
source, configuration, tests, fixture generators and generated run artifacts. Nothing outside it is
edited: the pretrained checkpoint, its saved configuration, the shards and the statistics file are
referenced by path and never rewritten, and no training default of the surrounding package is
touched. The surrounding evaluation pipeline is *imported through small local adapters* rather than
copied, because a copy of a mask rule or a cohort definition is a second definition free to drift
from the one the pretrained model was measured under.

Package layout, one responsibility each:

``config.py``
    Strict pilot settings, path resolution against the repository root, and the one resolver every
    launch mode shares.
``data.py``
    GUID/label/provenance manifest, anchor timestamps, support masks, late bags and time bins.
``model.py``
    Strict checkpoint reconstruction, the classifier wrapper, and the trainable-parameter allowlist.
``extract.py``
    Keyed latent extraction and the frozen training-only scaler.
``train.py``
    Frozen baseline, the tiny adaptation, the shuffled-label control and validation selection.
``evaluate.py``
    Preservation gates, held-out metrics and the GUID bootstrap.
``analyze.py``
    Temporal summaries, full-space geometry and the single shared training PCA.
``report.py``
    Figure and report generators.
``run.py``
    The editable ``RUN_ARGS`` dictionary, ``main()``, the CLI equivalents and the stage dispatcher.

Stages, in the order ``all`` runs them::

    tests -> smoke -> preflight -> extract(train/validation)
          -> baseline -> finetune -> control
          -> freeze selection/settings -> evaluate(test + analyses) -> report

**Runtime inputs the execution machine supplies**, none of which this package discovers on its own
and none of which is a prerequisite for the code to be finished:

===============================================  =========================================
Input                                            How it is handled
===============================================  =========================================
Pretrained checkpoint path                       Strict rebuild from its own stamps; the
                                                 file digest and ``model_class`` recorded
                                                 in the run's protocol.
Saved config / ``model_kwargs``                  The authority on architecture, channel
                                                 contract, clock and trim. Read from the
                                                 checkpoint and the resolved config beside
                                                 it, never re-derived from today's YAML.
Fold-1 train/validation/test shard lists         Fixed splits; a GUID manifest is built and
                                                 disjointness asserted across all three.
Matching statistics file and its provenance      Carried into the protocol record; a
                                                 fitting population that cannot be
                                                 established stays *unknown*.
Pretraining / selection GUID provenance          Optional. Known overlap disables any
                                                 clean-holdout claim; absent provenance is
                                                 recorded as unknown, never as "no overlap".
Patient/delivery grouping beyond GUID            Optional. Absent, GUID-only grouping is
                                                 disclosed and the bootstrap resamples GUIDs.
Device                                           Configurable; one device is enough.
===============================================  =========================================

Every geometry number -- $d_z$, the horizon, the anchor stride, the trim, the channel widths and
the per-channel forecast shift -- is read from the loaded checkpoint's own metadata at runtime.
None of them is written into this package as a constant: the promoted representation and the older
ratio-power configurations disagree about the channel contract, and a hard-coded width would
silently describe a model that was never trained.

**Environment.** The repository pins nothing on purpose: there is no dependency manifest, no lock
file and no interpreter pin, and the root ``setup.py`` declares no ``install_requires``. So the
environment is whichever one the operator already runs this repository under, and this pilot adds
**no** dependency to it and creates no root dependency file for itself.

What it needs is what its own import chain reaches, all of it already used by the surrounding
package: ``torch`` and ``lightning`` for the checkpoint and its task, ``numpy`` and ``pandas`` for
every table, ``pyarrow`` for the parquet indices, ``h5py`` for the shards, ``pyyaml`` for the
configs, ``loguru`` for the logs, ``matplotlib`` for the figures, and ``scipy`` / ``scikit-learn``
by way of the pipeline this package imports rather than through any call of its own. ``pytest`` is
needed to run the tests and nothing else. No version is stated here because none is required: a
number written into a docstring is a pin the repository deliberately does not have, and it would go
stale on the first upgrade while still reading as authoritative.

On the execution machine: check out this revision into an environment where those packages import,
make the checkpoint, its resolved config, the statistics file and the fold-1 shards readable, then
launch ``run.py``. The minimal synthetic logic subset under ``tests/logic`` needs none of those
files and no GPU.

Nothing is re-exported here and nothing runs on import: importing any module of this package must
not read a dataset, build a model, create a directory or parse a command line.
"""
from __future__ import annotations

__all__: list[str] = []
