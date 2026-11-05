r"""The within-recording time-shift control (``analyses/time_shift.py``).

Three properties carry it. The pairing never hands a segment a partner that shares stored samples
with it (that partner would carry part of the aligned source). The shifted arm differs from the
aligned one by the source and nothing else, which is checked by giving a segment its OWN source as
the "partner": the two arms must then agree exactly. And the verdict reads the lower end of the
interval of $\Delta K$, with too few recordings being INCONCLUSIVE rather than a pass or a fail.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import time_shift

from .conftest import make_stub_batch


def _source_sensitive_task(task_factory):
    """A task whose posterior actually reads the source: the delta heads are zero at init, which
    would make both arms equal to the prior whatever source they were given."""
    module = task_factory()
    module.eval()
    torch.manual_seed(0)
    for parameter in module.orig_model.posterior_head.delta_mu_head.parameters():
        parameter.data.normal_(0.0, 0.2)
    return module


def test_the_partner_is_the_nearest_segment_of_the_recording_that_shares_no_sample() -> None:
    rows = pd.DataFrame({
        "guid": ["A", "A", "A", "A", "B"],
        "epoch": [0.0, 800.0, 1600.0, 3000.0, 0.0],
        "dataset_index": [10, 11, 12, 13, 20],
    })

    paired = time_shift.choose_partners(rows, min_separation_s=1320.0).set_index("epoch")

    # 0 s: 800 overlaps, 1600 is the nearest clear one; 800 s: only 3000 clears; 1600 s: 3000 is
    # nearer than 0; 3000 s: 1600. B has one segment, so it has no partner and is not paired.
    assert paired["partner_epoch"].to_dict() == {0.0: 1600.0, 800.0: 3000.0, 1600.0: 3000.0, 3000.0: 1600.0}
    assert (paired["separation_s"] >= 1320.0).all()
    assert set(paired["guid"]) == {"A"}


def test_a_tie_goes_to_the_earlier_segment() -> None:
    rows = pd.DataFrame({"guid": ["A"] * 3, "epoch": [-2000.0, 0.0, 2000.0], "dataset_index": [0, 1, 2]})

    paired = time_shift.choose_partners(rows, min_separation_s=1320.0).set_index("epoch")

    assert paired.loc[0.0, "partner_epoch"] == -2000.0


def test_the_own_source_as_partner_reproduces_the_aligned_arm_exactly(task) -> None:
    module = _source_sensitive_task(task)
    batch = module.transfer_batch_to_device(make_stub_batch(seed=1), module.device, dataloader_idx=0)

    scores = time_shift.score_pairs(module, batch, batch)

    np.testing.assert_allclose(scores["kld_shifted"], scores["kld_aligned"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(scores["gap_shifted"], scores["gap_aligned"], rtol=1e-6, atol=1e-7)
    # Non-vacuity: the posterior does read the source here, so the equality above is a statement
    # about the two code paths and not two zeros.
    assert float(np.abs(scores["kld_aligned"]).max()) > 0.0


def test_another_source_moves_the_shifted_arm_and_leaves_the_aligned_one(task) -> None:
    module = _source_sensitive_task(task)
    own = module.transfer_batch_to_device(make_stub_batch(seed=1), module.device, dataloader_idx=0)
    other = module.transfer_batch_to_device(make_stub_batch(seed=2), module.device, dataloader_idx=0)

    reference = time_shift.score_pairs(module, own, own)
    shifted = time_shift.score_pairs(module, own, other)

    np.testing.assert_allclose(shifted["kld_aligned"], reference["kld_aligned"], rtol=1e-6)
    assert not np.allclose(shifted["kld_shifted"], reference["kld_shifted"])


@pytest.mark.parametrize(
    ("lower", "status"), [(0.05, "PASS"), (-0.01, "FAIL"), (float("nan"), "INCONCLUSIVE")]
)
def test_the_verdict_reads_the_lower_end_of_the_kl_interval(lower, status) -> None:
    summary = pd.DataFrame([
        {"quantity": "delta_kld", "unit": "nats per step", "value": 0.1, "ci_lo": lower, "ci_hi": 0.2, "n": 12},
        {"quantity": "delta_gap", "unit": "nats per anchor", "value": 0.3, "ci_lo": -0.1, "ci_hi": 0.7, "n": 12},
    ])

    verdict = time_shift.verdict_record(summary)

    assert verdict["name"] == time_shift.VERDICT_NAME
    assert verdict["status"] == status
    assert verdict["values"]["delta_gap_nats"] == 0.3


def test_an_offline_rerun_records_a_skip() -> None:
    context = AnalysisContext(collection=types.SimpleNamespace(), config={}, task=None, loader=None)

    result = time_shift.run_time_shift_analysis(context, eval_config={"seed": 0}, output_dir=".")

    assert result["skipped"] is True
