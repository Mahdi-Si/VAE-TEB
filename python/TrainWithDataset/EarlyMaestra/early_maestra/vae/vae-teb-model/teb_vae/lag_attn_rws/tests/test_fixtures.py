"""The shared fixtures are themselves load-bearing, so they get their own tests.

Two failure modes here would make the rest of the suite lie silently. A stub batch whose
``weight`` stopped planting its gap would leave every mask test green whether or not the masks
work. And ``perturb_posterior`` is the only thing standing between a KL assertion and vacuous
truth: the posterior delta heads are zero-initialised, so an unperturbed model passes every KL
test no matter how wrong it is.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from teb_vae.lag_attn_rws.tests.conftest import (
    BATCH,
    SEQ_LEN,
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TINY_KWARGS,
    make_stub_batch,
)


def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble derives the repo root from this file's own depth; a wrong
    depth would resolve some unrelated directory without ever raising."""
    repo_root = Path(__file__).resolve().parents[3]
    for package in ("teb_vae", "train", "utils"):
        assert (repo_root / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {repo_root}"
        )


def test_the_stub_batch_carries_raw_signals_at_sixteen_fold_length():
    batch = make_stub_batch()
    assert batch.fhr.shape == (BATCH, 16 * SEQ_LEN)
    assert batch.up.shape == (BATCH, 16 * SEQ_LEN)
    assert batch.fhr.shape[-1] == 16 * batch.fhr_st.shape[-2]


def test_the_stub_batch_plants_its_gap():
    """A silently gap-free fixture would leave every mask test vacuous."""
    batch = make_stub_batch()
    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert (batch.weight == 0.0).any()
    # The gap sits inside the tiny trained-anchor range, where every mask can see it.
    assert TINY_KWARGS["warmup_period"] <= STUB_GAP_STEP
    assert STUB_GAP_STEP < SEQ_LEN - TINY_KWARGS["horizon"]


def test_the_stub_batch_matches_the_feature_contract():
    batch = make_stub_batch()
    assert batch.fhr_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.fhr_ph.shape == (BATCH, SEQ_LEN, 66)
    assert batch.up_st.shape == (BATCH, SEQ_LEN, 43)
    assert batch.up_ph.shape == (BATCH, SEQ_LEN, 15)
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == TINY_KWARGS["c_y"]
    assert batch.up_st.shape[-1] + batch.up_ph.shape[-1] == TINY_KWARGS["c_u"]


def test_tiny_dropout_is_off():
    """Nonzero dropout would make every seeded bitwise comparison in the suite flaky."""
    assert TINY_KWARGS["dropout"] == 0.0


def test_the_kwargs_sets_satisfy_the_constructor_invariants():
    for kwargs in (TINY_KWARGS, SHIPPED_KWARGS):
        assert kwargs["num_heads"] * kwargs["d_head"] == kwargs["d_model"]
        assert kwargs["d_z"] % kwargs["num_heads"] == 0
        assert kwargs["warmup_period"] < kwargs["sequence_length"] - kwargs["horizon"]
        assert kwargs["raw_per_step"] == 16


def test_shipped_kwargs_is_the_production_geometry_not_a_miniature():
    assert SHIPPED_KWARGS["sequence_length"] == 300
    assert SHIPPED_KWARGS["d_z"] == 48
    assert SHIPPED_KWARGS["max_lag"] == 90
    assert SHIPPED_KWARGS["causal_norm"] is True


def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use."""
    markers = request.config.getini("markers")
    assert any(str(marker).startswith("slow") for marker in markers)


def test_no_fixture_files_live_in_this_module():
    """The committed shard and stats are lag_attn's; this module references them by path."""
    assert not (Path(__file__).resolve().parent / "fixtures").exists()
    shared = Path(__file__).resolve().parents[2] / "lag_attn" / "tests" / "fixtures"
    assert (shared / "tiny_shard.hdf5").is_file()
    assert (shared / "tiny_stats.hdf5").is_file()


def test_perturb_posterior_actually_changes_posterior_parameters(perturb_posterior):
    """The imported fixture must land on this model's ``posterior_head`` attribute."""

    class _StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.posterior_head = nn.Linear(4, 4)
            self.other_head = nn.Linear(4, 4)

    model = _StubModel()
    before = {name: parameter.clone() for name, parameter in model.named_parameters()}

    perturb_posterior(model)

    assert not torch.equal(model.posterior_head.weight, before["posterior_head.weight"])
    # Scoped to the posterior: perturbing the whole model would change what the KL tests mean.
    assert torch.equal(model.other_head.weight, before["other_head.weight"])
