r"""Raw-signal gap/sentinel convention for the v4 masking (S1-T07).

These constants are **seeded from the write pipeline**
``hdf5_dataset/new_pipeline/create_new_pipeline.py`` (verified 2026-07-12):

- ``interpolate_bad_values`` linearly interpolates NaN/Inf away per row; a fully-bad row is set to
  ``0.0``.
- ``_sanitize_signals`` clips FHR to ``[0, 500]`` (and UP to ``[-50, 500]``), then flushes
  denormals to ``0``.

So stored raw ``fhr`` carries dropouts/gaps as **``0.0`` bpm** (non-physiologic), essentially never
as ``NaN``. The masking (Sprint 3) upsamples the decimated ``weight`` by $16$ and, in
``sentinel_refine`` mode, additionally marks samples equal to :data:`SENTINEL` (and any
non-finite) invalid.

Status: seeded. **Live confirmation on a real production fold is deferred** -- run
``python -m model.vae_teb_prediction.model.model_raw.tools.inspect_raw_gaps --fold <path>`` on the
prod box; if it reports a different convention, update the two constants below.
"""
from __future__ import annotations

#: How gaps are encoded in the stored raw signal. One of ``{"nan", "zero", "held"}``.
GAP_ENCODING: str = "zero"

#: The gap marker value in raw ``fhr``/``up`` (``None`` => ``weight_only`` masking, no value refine).
SENTINEL: float | None = 0.0
