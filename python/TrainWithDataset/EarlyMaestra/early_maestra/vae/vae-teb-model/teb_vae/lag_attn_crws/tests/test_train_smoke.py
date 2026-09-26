r"""One real fit, through the real entry point, against the committed causal shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU. It is the only place the failures that live *between* the pieces can surface
-- a config key that reaches nothing, a tracked metric no run writes, a validation-only arm that
raises on the first validation epoch -- and the only place the checkpoint a real launch writes is
reloaded and a resume's tile grid is reproduced from it.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the shared
pre-flight guards, this driver's own, the temporary resolved-config file and the resolved-config
write beside the checkpoints hang off the entry point and are reached no other way. Two epochs
rather than the config's one, so the scheduler steps and the tile phase rotates with the epoch.

Determinism is asserted as two separate CPU processes under different ``PYTHONHASHSEED`` values
producing an identical metric row set.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_crws import trainer as trainer_module
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
    absolutize_dataset_paths,
)

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. See the module docstring for why it is not the config's one.
SMOKE_EPOCHS = 2

#: The run seed ``configs/default.yaml`` ships and ``tiny.yaml`` inherits. One of the four halves of
#: the tile-phase key, which is why it has to reach the checkpoint.
SHIPPED_SEED = 42

#: Raw samples a horizon token emits. The decoder's width, and the *only* thing that decides it:
#: unlike the causal-feature cells, no budget and no gate can move it.
RAW_PER_STEP = 16

#: The anchor counts the two stages must produce, derived here the way the model derives them so a
#: geometry change re-derives them rather than failing a literal.
DENSE_ANCHORS = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD
TILE_COUNT = -(-DENSE_ANCHORS // SHIPPED_HORIZON)


def _run_fit(tmp_path):
    """Run one real fit through the entry point and return the driver and its fitted trainer.

    Args:
        tmp_path: Directory the run writes into.

    Returns:
        ``(driver, trainer)``.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original_train_model = LagAttnCrwsTrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnCrwsTrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnCrwsTrainer.train_model

    return captured["driver"], captured["trainer"]


def _metrics(driver) -> pd.DataFrame:
    return pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")


def _run_fit_in_subprocess(tmp_path, hash_seed: str) -> pd.DataFrame:
    """Run one fit in a **separate process, on the CPU**, and return its metrics history.

    Four deliberate choices, each removing a source of variation that has nothing to do with what
    this package decides:

    * a separate process, because "two runs of one config" is literally two processes -- and because
      it is the only way to vary ``PYTHONHASHSEED``, whose per-process salt is what would break a
      tile phase derived from Python's own ``hash()`` with nothing raising;
    * the CPU, because the shipped config chooses cuDNN autotuning, which selects convolution
      algorithms by timing them and therefore accumulates the same sums in different orders. No seed
      controls that. Forcing ``deterministic: true`` instead does not work from inside a test
      session: it needs ``CUBLAS_WORKSPACE_CONFIG`` set before CUDA is initialised, and by then
      another test has initialised it;
    * **one intra-op thread**, because the CPU pool is the same hazard in another spelling: a
      reduction split across $n$ workers accumulates in the order they finish, and the split itself
      moves with how loaded the machine is. Measured rather than assumed -- twenty runs of this
      config, four of them under this pin: unpinned, ``train/source_conditioned_kl_raw`` at epoch
      $1$ takes two values a part in $10^{5}$ apart, and it takes the larger of them in every pinned
      run and in every run that does less work between epochs. The diagnostic page is what made it
      visible here (it is the heaviest thing this run does outside the fit), which is exactly why
      the pin belongs to the harness rather than to the callback: what is under test is what the
      *training path* decides, and a figure must not be able to move it;
    * ``num_workers: 0`` is already the tiny config's, so no loader worker seeding enters either.

    Args:
        tmp_path: Directory the run writes into.
        hash_seed: The ``PYTHONHASHSEED`` the subprocess runs under.

    Returns:
        The run's ``metrics_history.csv``.
    """
    import os
    import subprocess
    import sys

    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    config["advanced_config"]["trainer"]["profiler"] = None
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from teb_vae.lag_attn_crws.trainer import main; main(sys.argv[2])",
            str(repo_root),
            str(config_path),
        ],
        env={
            **dict(os.environ),
            "PYTHONHASHSEED": hash_seed,
            "CUDA_VISIBLE_DEVICES": "",
            # Set in the environment rather than through ``torch.set_num_threads``: the pool is
            # sized when torch is imported, and the entry point below imports it before any line
            # of ours runs.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        cwd=str(repo_root),
        capture_output=True,
        check=True,
        text=True,
    )
    written = list(tmp_path.rglob("metrics_history.csv"))
    assert len(written) == 1, written
    return pd.read_csv(written[0])


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """One real fit. Module-scoped: this is the expensive test in the suite, and every assertion
    below is a different question about the same run."""
    return _run_fit(tmp_path_factory.mktemp("smoke"))


# --------------------------------------------------------------------------------------
# The fit itself
# --------------------------------------------------------------------------------------
def test_the_fit_completes_with_every_logged_metric_finite(fit):
    """Including ``train/grad_norm``: a non-finite norm would zero the clip coefficient and the run
    would train nothing while completing normally."""
    _, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished
    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_spike_breaker_never_latched(fit):
    """A shipped ``additive_margin`` that skipped ordinary batches would read in a log exactly like a
    model that keeps blowing up, and only this column separates them."""
    driver, _ = fit
    frame = _metrics(driver)

    assert float(frame["train/spike_skipped"].max()) == 0.0


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it.

    The two forecast branches are **not** bitwise equal here and that is the shipped
    ``base_decode: mean``: the base branch decodes at $\mu^p$ while the full branch still carries the
    posterior's own draw. The KL is exactly zero regardless, because it depends on the two
    distributions rather than on the samples taken from them."""
    driver, _ = fit
    model = SeqVaeLagAttnCrws(**driver._build_model_kwargs()).eval()
    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, model.sequence_length
    outputs = model(
        torch.randn(batch_size, seq_len, CAUSAL_ST_WIDTH, generator=generator),
        torch.randn(batch_size, seq_len, CAUSAL_PH_WIDTH, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
        torch.zeros(batch_size, dtype=torch.long),
    )

    assert float(outputs["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(outputs["z_prior"], outputs["mu_prior"])
    assert outputs["mu_base"].shape == (batch_size, TILE_COUNT, SHIPPED_HORIZON, RAW_PER_STEP)
    assert not torch.equal(outputs["mu_base"], outputs["mu_full"])


# --------------------------------------------------------------------------------------
# What the run records
# --------------------------------------------------------------------------------------
def test_the_metrics_csv_carries_every_tracked_key_and_no_all_nan_column(fit):
    """Both halves of the tracked list's contract, on a real run: a name the framework never emits
    is a column that is NaN in every row of every run, and a tracked name that produced no column at
    all is a readout nothing ever recorded."""
    driver, _ = fit
    frame = _metrics(driver)

    missing = [name for name in LagAttnCrwsTrainer.TRACKED_METRICS if name not in frame.columns]
    assert missing == [], f"tracked but never written to the CSV: {missing}"
    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_anchor_count_sits_at_its_geometry_derived_value_on_both_stages(fit):
    """The geometry guard, and the one column of the three that is not a result. Training tiles, so
    the count is one of the two tile counts the phase can produce; validation decodes every valid
    anchor, so it is exactly the dense range's length. A value off either band means the tiling is
    not the one the configuration states."""
    driver, _ = fit
    frame = _metrics(driver)

    train = frame["train/anchors_per_sample"].dropna()
    val = frame["val/anchors_per_sample"].dropna()

    assert not train.empty and not val.empty
    assert bool(((train >= TILE_COUNT - 1) & (train <= TILE_COUNT)).all()), list(train)
    assert (val == float(DENSE_ANCHORS)).all(), list(val)


def test_the_checkpoint_carries_its_contract_and_reloads_with_no_shard_present(fit, tmp_path,
                                                                               monkeypatch):
    """The end of the road: a blob that describes itself and rebuilds without a config file **or a
    shard**.

    The stamped keep-index carries which channels the encoders read and the stamped warm-up vectors
    carry the availability terms, and the budget that resolved both is a property of the *data* --
    so a blob recording only the threshold could not be rebuilt anywhere the shards are not.

    Driven from a directory containing no HDF5 at all, with the working directory moved there, so a
    rebuild that reached for a shard by a relative path fails rather than quietly finding one."""
    driver, _ = fit
    kept = driver.pytorch_model.target_adapter.linear.in_features

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    assert blob["model_class"] == "SeqVaeLagAttnCrws"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    assert len(blob["model_kwargs"]["target_keep_index"]) == kept
    assert "decoder_out_channels" not in blob["model_kwargs"]
    check_model_class(blob, "SeqVaeLagAttnCrws")
    assert not list(empty.glob("*.hdf5"))
    rebuilt = SeqVaeLagAttnCrws(**blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == RAW_PER_STEP
    assert rebuilt.target_adapter.linear.in_features == kept
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )
    # The run seed reaches the blob too, which is what lets a resumed run reproduce its tile grid.
    assert blob["hyper_parameters"]["seed"] == SHIPPED_SEED


def test_a_resume_from_the_written_checkpoint_reproduces_the_same_tile_grid(fit):
    r"""The property the determinism requirement rests on and nothing else tests: the per-segment
    phase is a stable hash of the recording identifier, the segment's own start time, the training
    epoch and the **seed**, and a resumed run learns that seed only from the checkpoint's
    hyperparameters -- the framework reads ``general_config.seed`` once, into its own determinism
    setup, and hands it to no task.

    Reconstructed the way a resume would: a task rebuilt from the blob's recorded seed must resolve
    the same $\varphi$, and therefore the same ``anchor_index``, as one carrying the seed the run was
    launched with, for the same ``(guid, domain_start, epoch)``. A resume that lost it would re-tile
    every segment from the epoch it resumed at -- and $A_{\max}$ is a geometry constant either way,
    so no shape, no count and no metric would differ.

    The negative control is the second half rather than a separate test: without it every assertion
    here would also hold for a phase that ignored the seed entirely."""
    driver, _ = fit
    from .conftest import make_stub_batch

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    def _task(seed):
        """A trainer-less task at one seed. ``current_epoch`` is $0$ on all three, which is what
        holds the epoch half of the key fixed while the seed half is varied."""
        return type(driver.pl_model)(
            SeqVaeLagAttnCrws(**blob["model_kwargs"]),
            lr=1e-3,
            model_kwargs=blob["model_kwargs"],
            seed=seed,
        )

    batch = make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH)
    resumed = _task(blob["hyper_parameters"]["seed"])
    as_launched = _task(SHIPPED_SEED)
    reseeded = _task(SHIPPED_SEED + 1)

    resumed_phase = resumed.anchor_phase(batch)
    assert torch.equal(resumed_phase, as_launched.anchor_phase(batch))
    assert not torch.equal(resumed_phase, reseeded.anchor_phase(batch))

    # And the phase is only the key: what a run is reproducing is the decoded anchor set itself.
    device = torch.device("cpu")
    resumed_index, resumed_valid = resumed.orig_model._build_anchor_index(
        batch=2, device=device, anchor_phase=resumed_phase
    )
    launched_index, launched_valid = as_launched.orig_model._build_anchor_index(
        batch=2, device=device, anchor_phase=as_launched.anchor_phase(batch)
    )
    other_index, _ = reseeded.orig_model._build_anchor_index(
        batch=2, device=device, anchor_phase=reseeded.anchor_phase(batch)
    )

    assert torch.equal(resumed_index, launched_index)
    assert torch.equal(resumed_valid, launched_valid)
    assert resumed_index.shape == other_index.shape  # A_max is a geometry constant either way
    assert not torch.equal(resumed_index, other_index)


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def test_two_runs_of_one_config_produce_an_identical_metric_row_set(tmp_path_factory):
    r"""Determinism as the requirement states it: an identical **metric row set**, not merely
    identical anchor indices.

    Identical anchors are necessary and nowhere near sufficient -- the reparameterisation $\epsilon$
    and the permutation generator both move too -- and this cell adds one more thing that could
    break it and would not raise: the tile phase. Derived from a stable hash of the segment, the
    epoch and the seed, it survives a re-run; drawn from the global RNG it would not, and the only
    symptom would be two runs of one config disagreeing.

    Driven as two separate CPU processes under **different** ``PYTHONHASHSEED`` values; see
    :func:`_run_fit_in_subprocess` for why each of those three choices is there rather than an
    in-process GPU pair. The differing salt is not incidental: a phase derived from Python's own
    ``hash()`` would move with it, and nothing would raise -- $A_{\max}$ is a geometry constant
    either way.
    """
    first = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_a"), hash_seed="0")
    second = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_b"), hash_seed="12345")

    assert list(first.columns) == list(second.columns)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
