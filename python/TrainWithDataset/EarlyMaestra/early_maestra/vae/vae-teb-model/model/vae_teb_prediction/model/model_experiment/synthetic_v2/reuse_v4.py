r"""S0-T02: the single reuse surface for ``synthetic_v4``.

``synthetic_v4`` reuses two already-built, already-green halves **byte-unchanged**:

* the RAW model stack under ``model_raw/`` (:class:`SeqVaeRawV4`, its Lightning wrapper and
  trainer, geometry, masks, targets, front end, and the ``TestRunner``); and
* the model-free data half of ``synthetic_v2`` (:func:`generate_cell_raw`, the analytic-TE
  inverter/probe, the raw-TE band-pass probe, and the coupling/seed/cache helpers), plus the
  schema-agnostic pipeline helpers (:func:`load_config`, :func:`resolve_arm`, ``_deep_merge``).

Every name below is a **re-export** (the identical object as its origin-module attribute), so a
single import point (``from ... import reuse_v4``) gives the whole surface and the S0-T02 identity
test can prove nothing was accidentally re-implemented. No logic lives here.
"""

from __future__ import annotations

# --- RAW model stack (model_raw/, reused byte-unchanged) ---------------------------------
from model.vae_teb_prediction.model.model_raw import geometry
from model.vae_teb_prediction.model.model_raw.geometry import (
    derive_geometry,
    future_block_start,
    n_raw,
    valid_anchor_range,
)
from model.vae_teb_prediction.model.model_raw.raw_frontend import (
    CausalRawFrontend,
    assert_no_time_pooling_norm,
)
from model.vae_teb_prediction.model.model_raw.raw_masks import frontend_mask
from model.vae_teb_prediction.model.model_raw.raw_targets import build_future_target
from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    make_tiny_raw_model,
    tiny_raw_kwargs,
)
from model.vae_teb_prediction.model.model_raw.trainer_raw_v4 import (
    GraphModelVaeTebRawV4Trainer,
    SeqVaeRawV4Pl,
)
from model.vae_teb_prediction.model.model_raw.vae_teb_raw_v4 import SeqVaeRawV4

# --- synthetic_v2 model-free data half (imported unchanged) ------------------------------
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.analytic_te import (
    B_y_for_mean_te_block_state_space,
    realizable_te_block_from_arrays,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
    cell_seed,
    resolve_cache_dir,
    solve_cell_coupling,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import measure_te_raw
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (
    generate_cell_raw,
)

# --- schema-agnostic pipeline helpers (fork-shared machinery) ----------------------------
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
    _deep_merge,
    load_config,
    resolve_arm,
)

__all__ = [
    # model_raw
    "geometry",
    "n_raw",
    "future_block_start",
    "valid_anchor_range",
    "derive_geometry",
    "frontend_mask",
    "build_future_target",
    "CausalRawFrontend",
    "assert_no_time_pooling_norm",
    "TestRunner",
    "SeqVaeRawV4",
    "SeqVaeRawV4Pl",
    "GraphModelVaeTebRawV4Trainer",
    "make_tiny_raw_model",
    "tiny_raw_kwargs",
    # synthetic data half
    "generate_cell_raw",
    "B_y_for_mean_te_block_state_space",
    "realizable_te_block_from_arrays",
    "measure_te_raw",
    "solve_cell_coupling",
    "cell_seed",
    "resolve_cache_dir",
    # pipeline helpers
    "load_config",
    "resolve_arm",
    "_deep_merge",
]
