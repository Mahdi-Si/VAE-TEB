r"""The committed shard really produces the batch the model's forward signature expects.

The model takes ``y_st (B,T,43)``, ``y_ph (B,T,66)`` and a source stream assembled from ``up_st``
and ``up_ph``. Between the HDF5 file and those tensors sit a channel-first on-disk layout, a
per-sample transpose, a symmetric trim, and a normalization step that reads a separate stats file --
and almost every failure in that chain is silent. A missing index field yields "No samples match the
specified filters"; a stats-schema mismatch disables normalization with a warning and hands back
correctly-shaped, wrongly-scaled tensors. So the contract is asserted here rather than discovered
three steps into a training run.

The causal shard is held to a stronger contract than the two-sided one, because it makes a
stronger claim. Its coefficient blocks are not synthesised: they are the real one-sided transform
of its own stored ``fhr``/``up``, which is what lets two assertions here mean "measurably causal"
rather than "declares itself causal". The binding half is the load-bearing one -- a re-run of the
transform proves the *code* is one-sided and would pass even if the committed blocks were noise,
which is exactly the state the two-sided fixture is in.

Regenerate the fixtures with ``python scripts/make_tiny_shard.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest
import torch

from teb_vae.lag_attn.config import load_config
from train.data_module import GraphDataModule

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: What the trim arithmetic must produce from the committed shard: 330 - 2*(240//16) = 300.
_EXPECTED_T = 300
#: 16x decimation: 5280 - 2*240 = 4800.
_EXPECTED_RAW = 4800

#: Every committed fixture file, both variants. Written out rather than globbed: a fixture that
#: exists locally and was never staged would pass a glob and fail on a clean checkout.
_FIXTURE_FILES = (
    "tiny_shard.hdf5",
    "tiny_stats.hdf5",
    "tiny_shard_causal.hdf5",
    "tiny_stats_causal.hdf5",
)

#: The causal shard's stored widths, and the rebased warm-up range each block must reproduce at
#: ``trim_minutes: 1.0``. Hand-written from the documented drop rule: seven scattering channels
#: per block outrun the stored segment and are dropped at write time, leaving $36$ of $43$; both
#: phase blocks keep their full width, their $0.008$ Hz band floor excluding those filters
#: entirely. The ranges are what the whole warm-up budget is resolved against.
_CAUSAL_BLOCKS: Dict[str, Dict[str, Any]] = {
    "fhr_st": {"width": 36, "rebased": (0, 278)},
    "fhr_ph": {"width": 66, "rebased": (0, 134)},
    "up_st": {"width": 36, "rebased": (0, 278)},
    "up_ph": {"width": 15, "rebased": (41, 134)},
}

#: Decimated steps ``trim_minutes: 1.0`` removes from each end, and the decimation itself.
_TRIM_STEPS = 15
_DECIMATION = 16


@pytest.fixture(scope="module")
def config():
    """The resolved tiny config, with dataset paths made absolute.

    The config's paths are repo-root-relative because entry points run from the repo root; pytest
    may not.
    """
    resolved = load_config(str(_TINY_CONFIG))
    dataset = resolved["dataset_config"]
    for key in ("vae_train_datasets", "vae_test_datasets"):
        dataset[key] = [str(_REPO_ROOT / path) for path in dataset[key]]
    dataset["stat_path"] = str(_REPO_ROOT / dataset["stat_path"])
    return resolved


@pytest.fixture(scope="module")
def batch(config):
    return next(iter(GraphDataModule(config).train_dataloader()))


def test_the_fixtures_are_committed():
    """A silently-absent shard would make every test below fail with an unhelpful message."""
    for name in _FIXTURE_FILES:
        assert (_FIXTURES / name).is_file(), name


def test_the_fixtures_are_small_enough_to_live_in_the_repo():
    """Both variants against one budget. The causal pair adds roughly $1.0$ MB to the two-sided
    $1.5$ MB, which is what a shard whose blocks are a real transform of real signal costs."""
    total = sum((_FIXTURES / name).stat().st_size for name in _FIXTURE_FILES)
    assert total < 4 * 1024 * 1024, f"fixtures grew to {total / 1e6:.1f} MB"


def test_the_batch_carries_the_model_input_contract(batch):
    """Channel counts are the contract; the task checks them against every batch."""
    assert batch.fhr_st.shape == (2, _EXPECTED_T, 43)
    assert batch.fhr_ph.shape == (2, _EXPECTED_T, 66)
    assert batch.up_st.shape == (2, _EXPECTED_T, 43)
    assert batch.up_ph.shape == (2, _EXPECTED_T, 15)
    assert batch.weight.shape == (2, _EXPECTED_T)


def test_the_feature_fields_arrive_time_major(batch):
    """The on-disk layout is (N, C, T) and the dataset transposes on read.

    A model that permuted again would silently train on a (channels, time) tensor of the right
    rank. The assertion that catches it is that the last axis is the channel count -- and 43 != 300
    is the only reason it can be caught at all.
    """
    assert batch.fhr_st.shape[-1] == 43 != batch.fhr_st.shape[-2]


def test_the_source_stream_concatenates_to_the_configured_width(batch, config):
    """What the task builds and hands to the model as ``u_stream``."""
    u_stream = torch.cat([batch.up_st, batch.up_ph], dim=-1)
    assert u_stream.shape[-1] == config["model_config"]["VAE_model"]["c_u"] == 58


def test_the_target_stream_concatenates_to_the_configured_width(batch, config):
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == config["model_config"]["VAE_model"]["c_y"]


def test_the_configured_widths_match_the_committed_shard(batch, config):
    r"""The check the net's constructor used to make against a constant, made against data.

    This is the replacement for the old ``c_u == (101 if use_up_st else 58)`` assertion in
    ``test_config_load.py``: same intent, but it reads the widths off a real HDF5 instead of
    off a second copy of the config, so it cannot go stale the way that one did.

    Limitation worth naming: it pins the config against the *tiny* shard, while ``default.yaml``
    points at production HDF5. It is therefore only as good as ``scripts/make_tiny_shard.py``'s
    ``CHANNELS`` tracking ``hdf5_dataset/new_pipeline/create_new_pipeline.py`` -- which is why
    that dict carries a comment pointing back at it.
    """
    vae = config["model_config"]["VAE_model"]
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == vae["c_y"]
    expected_c_u = (
        batch.up_st.shape[-1] + batch.up_ph.shape[-1]
        if vae["use_up_st"]
        else batch.up_ph.shape[-1]
    )
    assert expected_c_u == vae["c_u"]


def test_the_raw_signals_keep_the_sixteen_fold_decimation_ratio(batch):
    """The plots put the raw trace and the feature grid on one time axis."""
    assert batch.fhr.shape == (2, _EXPECTED_RAW)
    assert batch.up.shape == (2, _EXPECTED_RAW)
    assert batch.fhr.shape[-1] == 16 * batch.fhr_st.shape[-2]


def test_the_trim_removes_the_configured_minutes_from_each_end(batch, config):
    """300 on-disk-330 is not an arbitrary fixture size; it is what trim_minutes: 1.0 leaves.

    A stats file built at a different trim only warns, so the geometry is pinned here instead.
    """
    trim_minutes = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    trim_decimated = int(4 * 60 * trim_minutes) // 16
    assert batch.fhr_st.shape[1] == 330 - 2 * trim_decimated == _EXPECTED_T


def test_guid_is_a_list_of_strings_not_a_tensor(batch):
    """So the usual ``{k: v.to(device) for k, v in batch.items()}`` would crash here."""
    assert isinstance(batch.guid, list)
    assert all(isinstance(guid, str) for guid in batch.guid)


def test_the_batch_supports_both_attribute_and_item_access(batch):
    assert torch.equal(batch.fhr_st, batch["fhr_st"])


def test_every_model_input_is_finite(batch):
    """Normalization log-transforms 42 of the 43 scattering channels; a non-positive sample there
    would survive as a finite but absurd value, and a NaN would poison the first backward."""
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        assert torch.isfinite(batch[field]).all(), f"{field} carries a non-finite value"


def test_normalization_actually_happened(batch):
    """The silent failure this fixture exists to rule out.

    Every path into the stats reader is wrapped in a warn-and-continue, so a schema mismatch leaves
    the batch correctly shaped and entirely unnormalized. The shard is written with FHR around 140
    bpm; a mean anywhere near that means the stats file was ignored.
    """
    assert abs(float(batch.fhr.mean())) < 5.0, (
        "fhr looks unnormalized -- the stats file was probably rejected and normalization "
        "silently disabled"
    )


def test_the_val_loader_reads_the_held_out_list(config):
    """`val` and `test` both read `vae_test_datasets`; there is no in-process split."""
    data_module = GraphDataModule(config)
    assert next(iter(data_module.val_dataloader())).fhr_st.shape[1] == _EXPECTED_T


# ---------------------------------------------------------------------------------------
# The causal shard
#
# It carries a claim the two-sided one does not: a coefficient at step t is a function of the
# raw signal up to t and of nothing else, valid only past a per-channel warm-up. That claim is
# checked twice below -- once by rebuilding the blocks from the shard's own raw signals, and once
# by perturbing a raw sample and watching nothing before it move.
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def causal_shard() -> Path:
    """Path to the committed causal shard."""
    return _FIXTURES / "tiny_shard_causal.hdf5"


def test_the_causal_shard_describes_its_own_filter_bank(causal_shard):
    """The root constants the warm-up vectors were measured under.

    Without them the stored boundary is recoverable only from the code that wrote it, and a shard
    rebuilt at another quantile keeps a different channel set with nothing saying so.
    """
    with h5py.File(causal_shard, "r") as handle:
        attrs = dict(handle.attrs)
        assert attrs["transform"] == "causal"
        assert int(attrs["causal_kernel_taps"]) > 0
        assert int(attrs["gammatone_order"]) > 0
        assert 0.0 < float(attrs["causal_warmup_quantile"]) < 1.0
        # The cross-phase block mixes both signals into one number and the causal variant does
        # not produce it; creating it would leave it empty for the whole build.
        assert "fhr_up_ph" not in handle


def test_the_causal_blocks_are_stored_at_the_widths_the_drop_rule_leaves(causal_shard):
    with h5py.File(causal_shard, "r") as handle:
        for name, expected in _CAUSAL_BLOCKS.items():
            assert handle[name].shape[1] == expected["width"], name


def test_the_rebased_warmup_reproduces_the_documented_ranges(causal_shard):
    r"""$W' = \max(W - 15, 0)$ at ``trim_minutes: 1.0``, per block.

    These four ranges are what the whole warm-up budget is chosen against: $134$ is where
    ``fhr_ph`` tops out, and it is the smallest budget that keeps every phase channel.
    """
    with h5py.File(causal_shard, "r") as handle:
        for name, expected in _CAUSAL_BLOCKS.items():
            stored = np.asarray(handle[name].attrs["causal_warmup_steps"], dtype=np.int64)
            rebased = np.maximum(stored - _TRIM_STEPS, 0)
            assert (int(rebased.min()), int(rebased.max())) == expected["rebased"], name


def test_every_causal_channel_records_the_delay_it_is_stale_by(causal_shard):
    """Stored, never compensated -- and never absent either. One-sidedness and zero latency are
    different properties, and this shard buys only the first."""
    with h5py.File(causal_shard, "r") as handle:
        for name in _CAUSAL_BLOCKS:
            delay = np.asarray(handle[name].attrs["causal_delay_s"], dtype=np.float64)
            assert delay.shape == (handle[name].shape[1],), name
            assert float(delay.min()) > 0.0, name


def test_the_loader_reports_the_causal_boundary(causal_shard):
    """Through the real dataset class, at the trim the model runs at."""
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    dataset = CombinedHDF5Dataset(paths=[str(causal_shard)], trim_minutes=1.0)

    assert dataset.transform == "causal"
    warmup = dataset.causal_warmup_steps
    assert warmup is not None
    for name, expected in _CAUSAL_BLOCKS.items():
        vector = warmup[name]
        assert vector.shape == (expected["width"],), name
        assert (int(vector.min()), int(vector.max())) == expected["rebased"], name

    mask = dataset.channel_valid_mask("fhr_st")
    assert mask.shape == (_EXPECTED_T, _CAUSAL_BLOCKS["fhr_st"]["width"])
    assert mask.dtype == torch.bool


@pytest.fixture(scope="module")
def causal_transform() -> Dict[str, Any]:
    """The production causal bank, its channel plan and its phase pairs, on the CPU.

    Built through the writer's own resolver rather than restated, so what the tests below compare
    the shard against is what a production run would have produced -- not a second construction
    that happens to agree today. Module-scoped: the bank is the expensive part and both tests
    below want it.
    """
    from hdf5_dataset.hdf5_dataset import resolve_leg_alignment
    from hdf5_dataset.smoke_check_channel_selection import _import_pipeline

    pipeline = _import_pipeline()
    with h5py.File(_FIXTURES / "tiny_shard_causal.hdf5", "r") as handle:
        signal_len = int(handle["fhr"].shape[1])
        # Read off the shard rather than named here. The two phase-harmonic operators produce
        # identical widths, warm-ups and stored delays and differ only in the coefficients
        # themselves, so a mode written out in this file would silently rebuild the other variant
        # and fail the binding test below with a numeric difference and no reason.
        leg_alignment = resolve_leg_alignment(handle.attrs)
    masks = pipeline.compute_scattering_masks(
        signal_len,
        scattering_T=_DECIMATION,
        device=torch.device("cpu"),
        transform="causal",
        leg_alignment=leg_alignment,
    )
    return {
        "pipeline": pipeline,
        "masks": masks,
        "signal_len": signal_len,
        "leg_alignment": leg_alignment,
        "target_pairs": pipeline._selection_pairs(masks["fhr_ph_selection"]),
        "source_pairs": pipeline._selection_pairs(masks["up_ph_selection"]),
    }


@pytest.mark.slow
def test_the_stored_blocks_are_the_transform_of_the_shards_own_raw_signals(
    causal_shard, causal_transform
):
    """The binding claim, and the load-bearing half of "measurably causal".

    One-sidedness is a property of the transform, so a probe of the transform alone would pass
    against a shard whose committed blocks were noise -- which is precisely the state the
    two-sided fixture is in. This is what ties the stored coefficients to the stored signal.
    """
    from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy

    with h5py.File(causal_shard, "r") as handle:
        fhr = np.asarray(handle["fhr"][:], dtype=np.float32)
        up = np.asarray(handle["up"][:], dtype=np.float32)
        stored = {name: np.asarray(handle[name][:]) for name in _CAUSAL_BLOCKS}

    rebuilt = transform_batch_numpy(
        CausalTorchBank(
            causal_transform["masks"]["causal_bank"],
            torch.device("cpu"),
            n_signal=causal_transform["signal_len"],
        ),
        fhr,
        up,
        causal_transform["target_pairs"],
        causal_transform["source_pairs"],
        plan=causal_transform["masks"]["channel_plan"],
        leg_alignment=causal_transform["leg_alignment"],
    )

    for name, expected in stored.items():
        produced = np.asarray(rebuilt[name], dtype=np.float64)
        reference = expected.astype(np.float64)
        scale = max(float(np.abs(reference).max()), 1e-30)
        assert float(np.abs(produced - reference).max()) / scale < 1e-6, name


@pytest.mark.slow
def test_no_coefficient_depends_on_a_raw_sample_that_comes_after_it(causal_transform):
    r"""Perturb raw sample $n$: no coefficient at any step $s$ with $16 s < n$ may move.

    Run in memory at ``complex128``, because FFT convolution is causal only in exact arithmetic
    and the ``float32`` ulp at FHR scale is already $\approx 1.5 \times 10^{-5}$ -- a tight
    criterion in the production ``complex64`` path is unachievable rather than merely strict. The
    tolerance below is relative to each block's own scale, at which the measured leakage is
    round-off by three orders of magnitude.

    The overall movement is asserted too, so the probe cannot pass on a transform that responds
    to nothing at all.
    """
    from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy

    with h5py.File(_FIXTURES / "tiny_shard_causal.hdf5", "r") as handle:
        fhr = np.asarray(handle["fhr"][:1], dtype=np.float64)
        up = np.asarray(handle["up"][:1], dtype=np.float64)

    bank = CausalTorchBank(
        causal_transform["masks"]["causal_bank"],
        torch.device("cpu"),
        n_signal=causal_transform["signal_len"],
        dtype=torch.complex128,
    )

    def _transform(fhr_in, up_in):
        return transform_batch_numpy(
            bank,
            fhr_in,
            up_in,
            causal_transform["target_pairs"],
            causal_transform["source_pairs"],
            plan=causal_transform["masks"]["channel_plan"],
            leg_alignment=causal_transform["leg_alignment"],
        )

    # Well inside the segment, and not on a decimation boundary: a perturbation at 16*s exactly
    # would sit at the last sample step s is allowed to see, which is the one index the criterion
    # says nothing about.
    sample_index = 200 * _DECIMATION + 7
    perturbed_fhr, perturbed_up = fhr.copy(), up.copy()
    perturbed_fhr[:, sample_index] += 50.0
    perturbed_up[:, sample_index] += 50.0

    base = _transform(fhr, up)
    moved = _transform(perturbed_fhr, perturbed_up)

    # Steps strictly before the perturbation: 16*s < n.
    first_affected = -(-sample_index // _DECIMATION)
    for name in _CAUSAL_BLOCKS:
        difference = np.abs(base[name] - moved[name])
        scale = max(float(np.abs(base[name]).max()), 1e-30)
        assert float(difference[:, :, :first_affected].max()) / scale < 1e-9, name
        assert float(difference.max()) / scale > 1e-4, f"{name} responded to nothing"


def test_a_null_stat_path_disables_normalization_rather_than_raising(config):
    """Why the entry point guards `stat_path` itself.

    The loader passes ``None`` straight through and the dataset merely skips normalization, so a
    typo'd key (the config key is ``stat_path``; the loader parameter is ``stats_path``) trains a
    model on raw-scale inputs and reports nothing. Nothing below the entry point will catch it.
    """
    unguarded = dict(config, dataset_config=dict(config["dataset_config"], stat_path=None))

    batch = next(iter(GraphDataModule(unguarded).train_dataloader()))

    assert abs(float(batch.fhr.mean())) > 100.0  # raw bpm scale: normalization never ran
