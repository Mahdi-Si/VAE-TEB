"""The loader probe reports what the loader actually yielded, and raises when a shard is empty.

Nothing else in a run reports per-file coverage, so a shard that silently contributes nothing
is invisible in every other output. That is the predecessor's hardest bug, and this is the only
artifact that can see it.
"""
from __future__ import annotations

import numpy as np
import torch

import json

import numpy as np
import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval.analyses import probe as probe_analysis
from teb_vae.lag_attn.eval.runner import EvalRunner
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG, TINY_SHARD
from train.data_module import GraphDataModule


@pytest.fixture
def config(repo_root):
    return load_config(str(repo_root / EVAL_TINY_CONFIG))


@pytest.fixture
def runner(tiny_checkpoint, tmp_path) -> EvalRunner:
    return EvalRunner.from_checkpoint(tiny_checkpoint, tmp_path / "run", device="cpu")


@pytest.fixture
def loader(config, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    return GraphDataModule(config).test_dataloader()


def test_probe_records_coverage_and_caches_the_latent(runner, loader, config, tmp_path):
    record = probe_analysis.run_probe(
        runner,
        loader,
        configured_files=config["dataset_config"]["vae_test_datasets"],
        output_dir=tmp_path,
    )

    assert record["n_samples"] == 4, "the committed shard holds four samples"
    assert record["n_batches"] == 2, "at batch_size.test = 2"
    assert record["per_file"] == {"tiny_shard.hdf5": 4}
    assert len(record["guids"]) == 4
    assert len(record["source_files"]) == 4

    # The probe does no forward of its own. An earlier form cached a per-sample ``z_mean``
    # through ``encode_only`` so the latent analyses could skip a pass, but nothing ever read it
    # -- ``latent`` takes its own pass and needs the per-step posterior, not a support-averaged
    # coordinate -- so it cost an encode over the whole split every run and saved nothing.
    assert "z_mean" not in record


def test_probe_records_labels_and_the_target_class_histogram(runner, loader, tmp_path):
    record = probe_analysis.run_probe(runner, loader, output_dir=tmp_path)
    assert sum(record["per_cs_label"].values()) == 4
    assert sum(record["per_bg_label"].values()) == 4
    # The first-nonzero target value identifies the class without assuming a one-hot layout.
    assert sum(record["per_target_class"].values()) == 4


def test_probe_answers_whether_weight_is_ever_fractional(runner, loader, tmp_path):
    """An open question the pipeline is meant to settle on first contact with real data."""
    record = probe_analysis.run_probe(runner, loader, output_dir=tmp_path)
    assert "binary" in record["weight"]
    assert isinstance(record["weight"]["binary"], bool)
    assert 0.0 <= record["weight"]["zero_frac"] <= 1.0


def test_probe_writes_a_json_that_omits_the_latent_cache(runner, loader, tmp_path):
    probe_analysis.run_probe(runner, loader, output_dir=tmp_path)
    written = json.loads((tmp_path / probe_analysis.PROBE_FILENAME).read_text(encoding="utf-8"))

    assert written["n_samples"] == 4
    assert written["per_file"] == {"tiny_shard.hdf5": 4}
    # Per-sample vectors have no business in a record meant to be read at a glance -- and
    # ``guids`` would repeat every GUID in the split, in a file the summary also copies.
    for key in probe_analysis.IN_MEMORY_KEYS:
        assert key not in written


def test_a_configured_file_yielding_nothing_raises(runner, loader, tmp_path):
    """A deliberate departure from the predecessor, which only logged and was ignored."""
    with pytest.raises(RuntimeError, match="yielded zero samples"):
        probe_analysis.run_probe(
            runner,
            loader,
            configured_files=[TINY_SHARD, "some/other/absent_subgroup.hdf5"],
            output_dir=tmp_path,
        )


def test_a_capped_pass_warns_rather_than_raising_on_a_missing_file(runner, loader, tmp_path):
    """A prefix cap over concatenated shards reaches only the first ones -- expected, not a bug."""
    record = probe_analysis.run_probe(
        runner,
        loader,
        configured_files=[TINY_SHARD, "some/other/absent_subgroup.hdf5"],
        max_samples=2,
        output_dir=tmp_path,
    )
    assert record["n_samples"] == 2


def test_an_empty_split_raises(runner, tmp_path):
    with pytest.raises(RuntimeError, match="no samples at all"):
        probe_analysis.run_probe(runner, [], output_dir=tmp_path)


def test_probe_makes_exactly_one_pass(runner, loader, tmp_path):
    """One pass: the probe is pure bookkeeping over the loader and does no forward of its own."""
    passes = {"count": 0}

    class CountingLoader:
        def __iter__(self):
            passes["count"] += 1
            return iter(loader)

    probe_analysis.run_probe(runner, CountingLoader(), output_dir=tmp_path)
    assert passes["count"] == 1


# ---------------------------------------------------------------------------
# Raw target values
# ---------------------------------------------------------------------------
def test_the_raw_target_record_counts_every_step_not_one_per_recording(runner, loader, tmp_path):
    """``per_target_class`` samples one step per recording; the truncation check needs them all."""
    record = probe_analysis.run_probe(runner, loader, output_dir=tmp_path)
    values = record["target_values"]
    assert values["n_values"] > record["n_samples"], (
        "the raw record must span every step, not one value per recording"
    )
    for key in ("n_nonzero", "n_fractional", "n_non_finite", "any_fractional"):
        assert key in values


def test_a_fractional_target_is_counted_even_when_the_first_nonzero_step_is_whole():
    """The exact case that made the class histogram the wrong input for the truncation check."""
    target = torch.zeros(2, 10)
    # Both recordings start their nonzero run at full weight and taper at the boundary.
    target[0, 2:8] = 1.0
    target[0, 8] = 0.5
    target[1, 1:6] = 2.0
    target[1, 6] = 1.25

    summary = probe_analysis._target_value_summary(target)
    assert summary["n_fractional"] == 2
    assert summary["n_non_finite"] == 0

    # ... while the class histogram resolves the same rows to whole class codes, which is why it
    # cannot answer this question and _target_value_summary has to.
    weight = torch.ones_like(target)
    codes = [
        labels.clinical_class_code(np.asarray(row), np.asarray(weight_row))
        for row, weight_row in zip(target.numpy(), weight.numpy())
    ]
    assert codes == [1, 2]


def test_the_class_histogram_divides_the_weight_out_rather_than_counting_scaled_values():
    """``target`` is the class code *scaled by* the per-step weight, so it must be divided back.

    Keying the histogram on the raw stored value produced entries like ``'0.75'`` for a recording
    whose first valid step was only partially valid. ``report.check_classes_present`` counts any
    key that is not ``'None'``/``'0.0'``/``'0'``, so a single such recording made a genuinely
    single-class split report two classes -- permanently defeating the coverage check that this
    histogram exists to feed.
    """
    # One acidosis recording (code 2) whose valid steps are all partially weighted: every stored
    # target value is fractional, and none of them equals the class code.
    target = torch.zeros(1, 8)
    weight = torch.zeros(1, 8)
    target[0, 2:6] = 2.0 * 0.5
    weight[0, 2:6] = 0.5

    code = labels.clinical_class_code(np.asarray(target[0]), np.asarray(weight[0]))
    assert code == 2, "the weight was not divided back out of the scaled target"
    assert labels.class_name(code) == "acidosis"
    # The raw value that the old keying would have used is not a class code at all.
    assert float(target[0, 2]) == 1.0


def test_a_non_finite_target_is_counted_rather_than_read_as_fractional():
    """NaN != round(NaN), so an uncounted NaN would masquerade as a fractional value."""
    target = torch.tensor([[1.0, float("nan"), 0.0]])
    summary = probe_analysis._target_value_summary(target)
    assert summary["n_non_finite"] == 1
    assert summary["n_fractional"] == 0


def test_a_split_with_no_target_field_records_an_empty_raw_summary():
    assert probe_analysis._merge_target_summaries([]) == {}
