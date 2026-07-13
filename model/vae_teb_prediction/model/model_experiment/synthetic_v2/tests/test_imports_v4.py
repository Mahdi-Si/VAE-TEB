r"""S0-T02: assert ``reuse_v4`` re-exports the identical objects as their origin modules.

If any name here is a re-implementation (a copy) rather than a re-export, the ``is`` identity
check fails -- the guard that keeps ``synthetic_v4`` honestly *reusing* the green model_raw and
data-half surfaces instead of forking them.
"""

from __future__ import annotations

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import reuse_v4

pytestmark = pytest.mark.v4


def test_model_raw_reexports_are_identical() -> None:
    r"""The RAW model-stack re-exports are the same objects as in ``model_raw``."""
    from model.vae_teb_prediction.model.model_raw import geometry as g
    from model.vae_teb_prediction.model.model_raw.raw_frontend import (
        CausalRawFrontend,
        assert_no_time_pooling_norm,
    )
    from model.vae_teb_prediction.model.model_raw.raw_masks import frontend_mask
    from model.vae_teb_prediction.model.model_raw.raw_targets import build_future_target
    from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
    from model.vae_teb_prediction.model.model_raw.trainer_raw_v4 import (
        GraphModelVaeTebRawV4Trainer,
        SeqVaeRawV4Pl,
    )
    from model.vae_teb_prediction.model.model_raw.vae_teb_raw_v4 import SeqVaeRawV4

    assert reuse_v4.geometry is g
    assert reuse_v4.n_raw is g.n_raw
    assert reuse_v4.future_block_start is g.future_block_start
    assert reuse_v4.valid_anchor_range is g.valid_anchor_range
    assert reuse_v4.derive_geometry is g.derive_geometry
    assert reuse_v4.frontend_mask is frontend_mask
    assert reuse_v4.build_future_target is build_future_target
    assert reuse_v4.CausalRawFrontend is CausalRawFrontend
    assert reuse_v4.assert_no_time_pooling_norm is assert_no_time_pooling_norm
    assert reuse_v4.TestRunner is TestRunner
    assert reuse_v4.SeqVaeRawV4 is SeqVaeRawV4
    assert reuse_v4.SeqVaeRawV4Pl is SeqVaeRawV4Pl
    assert reuse_v4.GraphModelVaeTebRawV4Trainer is GraphModelVaeTebRawV4Trainer


def test_data_half_reexports_are_identical() -> None:
    r"""The model-free data-half re-exports match their origin modules exactly."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        analytic_te,
        build_dataset_v2,
        eval_v2,
        raw_generators,
    )

    assert reuse_v4.generate_cell_raw is raw_generators.generate_cell_raw
    assert (reuse_v4.B_y_for_mean_te_block_state_space
            is analytic_te.B_y_for_mean_te_block_state_space)
    assert (reuse_v4.realizable_te_block_from_arrays
            is analytic_te.realizable_te_block_from_arrays)
    assert reuse_v4.measure_te_raw is eval_v2.measure_te_raw
    assert reuse_v4.solve_cell_coupling is build_dataset_v2.solve_cell_coupling
    assert reuse_v4.cell_seed is build_dataset_v2.cell_seed
    assert reuse_v4.resolve_cache_dir is build_dataset_v2.resolve_cache_dir


def test_pipeline_helper_reexports_are_identical() -> None:
    r"""The schema-agnostic pipeline helpers are re-exported, not re-implemented."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2

    assert reuse_v4.load_config is run_pipeline_v2.load_config
    assert reuse_v4.resolve_arm is run_pipeline_v2.resolve_arm
    assert reuse_v4._deep_merge is run_pipeline_v2._deep_merge


def test_tiny_model_helper_reexports_are_identical() -> None:
    r"""The ``model_raw`` tiny-model helpers behind ``tiny_checkpoint`` are re-exported."""
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        make_tiny_raw_model,
        tiny_raw_kwargs,
    )

    assert reuse_v4.make_tiny_raw_model is make_tiny_raw_model
    assert reuse_v4.tiny_raw_kwargs is tiny_raw_kwargs


def test_all_exports_present_on_module() -> None:
    r"""Every name in ``reuse_v4.__all__`` is actually bound on the module."""
    for name in reuse_v4.__all__:
        assert hasattr(reuse_v4, name), f"reuse_v4 missing exported name {name!r}"
