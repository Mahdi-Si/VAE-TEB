r"""Tests for the loader-iteration collectors.

The load-bearing assertion in this file is
:func:`test_a_capped_draw_reaches_both_files`. The eval loader is built ``shuffle=False`` over
eight concatenated per-subgroup shards, so a cap implemented as a prefix draws file $0$ alone --
one subgroup and one clinical class. That is the predecessor pipeline's worst bug arriving by a
second route, and it is invisible in every output except the per-file composition. The synthetic
two-file loader here is the smallest thing that can fail under prefix truncation.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval import collectors
from teb_vae.lag_attn.eval.runner import EvalRunner
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.tests.conftest import SEQ_LEN, TINY_KWARGS


def _two_file_loader(per_file: int = 10, batch_size: int = 5) -> List[Any]:
    """Build batches drawn from two shards, concatenated exactly as the real loader concatenates.

    Args:
        per_file: Samples contributed by each file.
        batch_size: Samples per batch.

    Returns:
        A list of batches, ``file_a`` first then ``file_b``, matching ``shuffle=False``.
    """
    sources = ["file_a.hdf5"] * per_file + ["file_b.hdf5"] * per_file
    generator = torch.Generator().manual_seed(0)
    batches = []
    for start in range(0, len(sources), batch_size):
        window = sources[start : start + batch_size]
        size = len(window)
        batches.append(
            types.SimpleNamespace(
                fhr_st=torch.randn(size, SEQ_LEN, 43, generator=generator),
                fhr_ph=torch.randn(size, SEQ_LEN, 66, generator=generator),
                up_st=torch.randn(size, SEQ_LEN, 43, generator=generator),
                up_ph=torch.randn(size, SEQ_LEN, 15, generator=generator),
                weight=torch.ones(size, SEQ_LEN),
                guid=[f"guid-{start + offset:03d}" for offset in range(size)],
                source_file_basename=list(window),
            )
        )
    return batches


@pytest.fixture
def runner(tmp_path) -> EvalRunner:
    """A runner around an untouched tiny model -- enough to iterate and forward."""
    from teb_vae.lag_attn.eval.runner import Objective

    torch.manual_seed(0)
    model = SeqVaeLagAttn(**TINY_KWARGS)
    model.eval()
    return EvalRunner(
        model=model,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        objective=Objective(
            likelihood="mse", sigma_obs=1.0, free_bits=0.0, detach_baseline_in_full=False,
            lambda_full=1.0, lambda_base=0.5, lambda_lag=0.0, beta_schedule=None, kld_beta=0.01,
        ),
        checkpoint_path=tmp_path / "none.ckpt",
    )


def _mean_energy(_: EvalRunner, batch: Any) -> Dict[str, Any]:
    """A trivial per-batch computation, so the tests exercise the plumbing not the metric."""
    return {"energy": batch.fhr_st.pow(2).mean(dim=(1, 2))}


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def test_no_cap_retains_everything() -> None:
    plan = collectors.CollectionPlan.build(20, None, seed=0)
    assert plan.retained is None
    assert all(plan.keeps(index) for index in range(20))
    assert plan.describe()["capped"] is False


def test_a_capped_plan_is_seeded_and_reproducible() -> None:
    """A rerun with the same seed must retain the same samples."""
    first = collectors.CollectionPlan.build(50, 10, seed=3)
    second = collectors.CollectionPlan.build(50, 10, seed=3)
    assert first.retained == second.retained
    assert first.describe()["n_retained"] == 10


# ---------------------------------------------------------------------------
# Caps are subsamples, not prefixes
# ---------------------------------------------------------------------------
def test_a_capped_draw_reaches_both_files(runner: EvalRunner) -> None:
    """The assertion that would fail under prefix truncation.

    A cap of half the dataset over two equally sized concatenated files must draw from both. A
    prefix cap draws file A alone, which in the real eight-shard split is one subgroup and one
    clinical class.
    """
    loader = _two_file_loader(per_file=10)
    plan = collectors.CollectionPlan.build(20, 10, seed=1)

    collected = collectors.collect_metrics(runner, loader, _mean_energy, plan=plan)

    assert len(collected.frame) == 10
    assert set(collected.composition) == {"file_a.hdf5", "file_b.hdf5"}, (
        f"a capped draw covered only {set(collected.composition)}; this is prefix truncation"
    )


def test_a_stratified_plan_covers_both_files_even_when_they_are_unbalanced(
    runner: EvalRunner,
) -> None:
    """Stratification is what makes coverage a guarantee rather than a probability.

    The real shards are wildly unbalanced -- ``hie_cs`` is a fraction of ``healthy_no_bg_no_cs``
    -- so an unstratified draw can plausibly miss the small ones entirely.
    """
    sources = ["file_a.hdf5"] * 38 + ["file_b.hdf5"] * 2
    loader = _two_file_loader(per_file=1)  # replaced below; only the shape is reused
    loader = []
    generator = torch.Generator().manual_seed(0)
    for start in range(0, len(sources), 5):
        window = sources[start : start + 5]
        loader.append(
            types.SimpleNamespace(
                fhr_st=torch.randn(len(window), SEQ_LEN, 43, generator=generator),
                fhr_ph=torch.randn(len(window), SEQ_LEN, 66, generator=generator),
                up_st=torch.randn(len(window), SEQ_LEN, 43, generator=generator),
                up_ph=torch.randn(len(window), SEQ_LEN, 15, generator=generator),
                weight=torch.ones(len(window), SEQ_LEN),
                guid=[f"g{start + i}" for i in range(len(window))],
                source_file_basename=list(window),
            )
        )

    plan = collectors.CollectionPlan.build(40, 8, seed=0, groups=sources)
    collected = collectors.collect_metrics(runner, loader, _mean_energy, plan=plan)
    assert set(collected.composition) == {"file_a.hdf5", "file_b.hdf5"}


def test_the_realised_composition_is_recorded_so_a_skewed_draw_is_visible(
    runner: EvalRunner,
) -> None:
    """A skewed draw must be visible in the run's output rather than invisible."""
    loader = _two_file_loader(per_file=10)
    plan = collectors.CollectionPlan.build(20, 6, seed=4)
    collected = collectors.collect_metrics(runner, loader, _mean_energy, plan=plan)

    summary = collected.summary()
    assert sum(summary["composition"].values()) == len(collected.frame)
    assert summary["n_seen"] == 20
    assert summary["plan"]["cap"] == 6


# ---------------------------------------------------------------------------
# Frame contract
# ---------------------------------------------------------------------------
def test_every_row_carries_its_guid_and_source_file(runner: EvalRunner) -> None:
    """``guid`` survives collation as a ``list[str]`` and must come back as strings.

    Per-recording aggregation is then a ``groupby`` rather than a second loader and a second pass.
    """
    collected = collectors.collect_metrics(runner, _two_file_loader(), _mean_energy)
    # Membership rather than position: the collector also attaches the class and subgroup
    # columns, which sit between the identity columns and the per-batch function's own. Pinning
    # positions would make this test fail whenever a column is added, which is not what it is for.
    assert {"sample_index", "guid", "source_file", "energy"} <= set(collected.frame.columns)
    assert list(collected.frame.columns[:3]) == ["sample_index", "guid", "source_file"]
    assert collected.frame["guid"].map(type).eq(str).all()
    assert collected.frame["guid"].iloc[0] == "guid-000"
    assert collected.frame["sample_index"].tolist() == list(range(20))


def test_a_column_of_the_wrong_length_raises_rather_than_misaligning(
    runner: EvalRunner,
) -> None:
    """A silent misalignment makes every row after the first short batch describe another sample."""

    def _too_short(_: EvalRunner, batch: Any) -> Dict[str, Any]:
        return {"bad": torch.zeros(int(batch.fhr_st.shape[0]) - 1)}

    with pytest.raises(ValueError, match="one value per sample"):
        collectors.collect_metrics(runner, _two_file_loader(), _too_short)


def test_a_scalar_column_is_broadcast_to_the_batch(runner: EvalRunner) -> None:
    """A per-batch constant -- a flag, a threshold -- is a legitimate column."""
    collected = collectors.collect_metrics(
        runner, _two_file_loader(), lambda _r, _b: {"flag": np.float64(1.5)}
    )
    assert collected.frame["flag"].eq(1.5).all()


def test_max_samples_stops_iteration_early(runner: EvalRunner) -> None:
    """A prefix cap, appropriate only for a smoke run -- and it caps by sample, not by batch."""
    collected = collectors.collect_metrics(
        runner, _two_file_loader(per_file=10, batch_size=5), _mean_energy, max_samples=7
    )
    # Does not split a batch, so it overshoots to the batch boundary: 10, not 7.
    assert collected.n_seen == 10


# ---------------------------------------------------------------------------
# Heavy collectors
# ---------------------------------------------------------------------------
def test_collect_predictions_retains_the_planned_samples_only(runner: EvalRunner) -> None:
    """The averaged $(T, c_y)$ rendering, which is what the heatmap figures consume."""
    plan = collectors.CollectionPlan.build(20, 6, seed=2)
    collected = collectors.collect_predictions(runner, _two_file_loader(), plan=plan)

    assert collected.arrays["forecast"].shape == (6, SEQ_LEN, int(runner.model.c_y))
    assert collected.arrays["target"].shape == (6, SEQ_LEN, int(runner.model.c_y))
    assert len(collected.frame) == 6
    assert set(collected.composition) == {"file_a.hdf5", "file_b.hdf5"}


def test_collect_attention_retains_weights_in_lag_order(runner: EvalRunner) -> None:
    """$(N, T, M, L)$ with index $0$ the current step, so a lag needs no reindexing."""
    plan = collectors.CollectionPlan.build(20, 4, seed=6)
    collected = collectors.collect_attention(runner, _two_file_loader(), plan=plan)

    weights = collected.arrays["attn_weights"]
    assert weights.shape == (4, SEQ_LEN, runner.num_heads, runner.num_lags)
    # Under eval() the rows are a genuine distribution over the causally valid lags.
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-4)


def test_a_collector_leaves_the_model_in_the_mode_it_found_it(runner: EvalRunner) -> None:
    """``inference_mode`` restores the prior training flag, so one analysis cannot alter the next."""
    runner.model.train()
    collectors.collect_metrics(runner, _two_file_loader(), _mean_energy)
    assert runner.model.training is True
