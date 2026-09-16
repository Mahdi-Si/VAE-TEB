r"""Shared pytest configuration and fixtures for the lag-residual causal-feature forecaster.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and pins the real ``utils`` package the same way the
sibling suites do.

**The fixtures here are deliberately self-contained**, and that is a decision rather than an
oversight. The sibling suites import their tiny geometry from the causal cell's conftest, which
carries the *legacy* phase operator's channel widths; this architecture's first task is the integer
operator's, and a suite that closed over the legacy widths would test a geometry no configuration
of this package describes. The modules under test here are geometry-agnostic -- they take widths as
arguments and hold no opinion about the dataset -- so seeded tensors at a small declared geometry
are the whole of what they need. The shard-backed fixtures arrive with the model, which is the first
thing in this package that has an opinion about the data.

Every constant below is small enough to keep a whole suite under a few seconds and large enough to
be structurally faithful: more than one anchor, more than one lag reaching before the record
begins, and a warm-up staircase whose slowest channel is still cold at the first anchor -- which is
the case that separates index support from feature warm-up.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pytest
import torch

# teb_vae/lag_slot_transformer_cfs/tests/conftest.py -> parents[0]=tests,
# [1]=lag_slot_transformer_cfs, [2]=teb_vae, [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repository: the real one at the root and a near-empty one
# deeper in the tree. On a repository-wide run another conftest can put the latter's parent first on
# ``sys.path``, shadowing the real one. Binding the root package now -- while the repo root is still
# first -- pins its ``__path__`` for every later ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

#: Samples per batch in every fixture here.
TINY_BATCH = 3

#: Stored steps $T$. Large enough that the earliest anchor still has lags reaching before the
#: record begins, which is the out-of-range branch of the gather.
TINY_SEQ_LEN = 24

#: Source channels $C_U$.
TINY_C_U = 6

#: Candidate lags $L$, so lags run over $0, \ldots, L-1$.
TINY_N_LAGS = 5

#: Target state width $d_h$ the proposal head conditions on.
TINY_D_MODEL = 16

#: Latent width $d_z$.
TINY_D_Z = 4

#: First eligible anchor $F$. Below the slowest channel's warm-up on purpose: at this anchor the
#: lag index is perfectly in range while that channel has not warmed up, which is exactly the pair
#: of conditions the availability rules must keep apart.
TINY_FLOOR = 6

#: $W'_j$ per source channel: a staircase from immediately warm to still cold at the first anchor.
TINY_SOURCE_WARMUP: Tuple[int, ...] = (0, 1, 3, 5, 8, 11)


#: Declared target channels of the integer-operator arm: the first stored block, then the second.
DECLARED_ST = 36
DECLARED_TARGET_PH = 44
DECLARED_C_Y = DECLARED_ST + DECLARED_TARGET_PH

#: Declared source channels of the same arm.
DECLARED_SOURCE_PH = 10
DECLARED_C_U = DECLARED_ST + DECLARED_SOURCE_PH

#: The tiny model's horizon and anchor stride. The stride must not exceed the horizon, and it is
#: above one on purpose: at stride one the tiling has no phase, and every padding assertion in the
#: suite would pass without a padded slot ever existing.
TINY_MODEL_HORIZON = 4
TINY_MODEL_STRIDE = 4

#: Surviving target channels: the last four of the first stored block are dropped, mirroring the
#: shipped budget, so the keep-index is non-contiguous and a positional block split would land in
#: the wrong place.
TINY_TARGET_KEEP = tuple(range(DECLARED_ST - 4)) + tuple(range(DECLARED_ST, DECLARED_C_Y))

#: $W'_c$ per surviving target channel. Bounded by the anchor floor, which the constructor refuses
#: to place below the slowest kept target channel's own warm-up.
TINY_TARGET_WARMUP = tuple(
    min(index // 16, TINY_FLOOR) for index in range(len(TINY_TARGET_KEEP))
)

#: Surviving source channels: all of them, as the shipped budget keeps them.
TINY_SOURCE_KEEP = tuple(range(DECLARED_C_U))

#: $W'_j$ per surviving source channel, rising past the anchor floor on purpose: the slowest source
#: channels are still cold at the first anchors, which is the condition the availability rules must
#: keep separate from index support.
TINY_MODEL_SOURCE_WARMUP = tuple(min(index // 4, 11) for index in range(DECLARED_C_U))


def tiny_model_kwargs(**overrides):
    """Constructor keywords for a structurally faithful miniature of the shipped model.

    Small in every width and every count, and faithful in the four things the suite asserts
    against: a non-contiguous target keep-index, a source warm-up that outlasts the anchor floor,
    a stride above one so the tiling has phases and padding, and an even model width so the
    metadata clock's frequency pairs are complete.

    Args:
        **overrides: Keywords to replace.

    Returns:
        The keyword dict.
    """
    kwargs = dict(
        sequence_length=TINY_SEQ_LEN,
        d_model=TINY_D_MODEL,
        d_z=TINY_D_Z,
        horizon=TINY_MODEL_HORIZON,
        raw_per_step=16,
        warmup_period=TINY_FLOOR,
        c_y=DECLARED_C_Y,
        c_u=DECLARED_C_U,
        use_up_st=True,
        max_lag=TINY_N_LAGS - 1,
        dropout=0.0,
        decoder_hidden=16,
        horizon_depth=2,
        horizon_kernel=3,
        horizon_film=True,
        horizon_attention_blocks=1,
        encoder_conv_kernels=(3, 3),
        encoder_conv_dilations=(1, 2),
        encoder_num_heads=4,
        encoder_d_ff=32,
        target_attention_blocks=2,
        anchor_stride=TINY_MODEL_STRIDE,
        target_keep_index=TINY_TARGET_KEEP,
        target_warmup_steps=TINY_TARGET_WARMUP,
        source_keep_index=TINY_SOURCE_KEEP,
        source_warmup_steps=TINY_MODEL_SOURCE_WARMUP,
        persistence_residual=True,
    )
    kwargs.update(overrides)
    return kwargs


def build_tiny_model(*, seed: int = 20260909, **overrides):
    """Construct the tiny model **deterministically**, so two builds are comparable.

    The generic initialisation pass draws from the global generator, so two constructions in one
    process produce two different models unless the generator is seeded first. Every comparison in
    this suite that builds a model twice -- chunked against unchunked, one share against another,
    an arm against its reference -- depends on that, and without the seed such a test does not fail:
    it passes or fails on whatever random state the tests before it happened to leave, which is the
    worst of both.

    Args:
        seed: Seed applied immediately before construction.
        **overrides: Constructor keywords to replace.

    Returns:
        The model, freshly constructed and in training mode.
    """
    from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs

    torch.manual_seed(seed)
    return SeqVaeLagResidualTrfCfs(**tiny_model_kwargs(**overrides))


def tiny_streams(batch: int = TINY_BATCH, seed: int = 0):
    """Seeded model inputs at the tiny declared widths.

    Args:
        batch: Samples in the batch.
        seed: Seed, so a batch is reproducible.

    Returns:
        ``(y_st, y_ph, u_stream)``.
    """
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, TINY_SEQ_LEN, DECLARED_ST, generator=generator),
        torch.randn(batch, TINY_SEQ_LEN, DECLARED_TARGET_PH, generator=generator),
        torch.randn(batch, TINY_SEQ_LEN, DECLARED_C_U, generator=generator),
    )


@pytest.fixture
def source_stream() -> torch.Tensor:
    """A seeded gated source stream at the tiny declared geometry.

    Returns:
        A $(B, T, C_U)$ float tensor.
    """
    generator = torch.Generator().manual_seed(20260909)
    return torch.randn(TINY_BATCH, TINY_SEQ_LEN, TINY_C_U, generator=generator)


@pytest.fixture
def anchors() -> torch.Tensor:
    """A dense anchor set from the tiny floor to the end of the record.

    Every sample gets the same anchors, which is what makes an assertion about one row an
    assertion about the geometry rather than about a draw.

    Returns:
        A $(B, A)$ ``long`` tensor.
    """
    row = torch.arange(TINY_FLOOR, TINY_SEQ_LEN, dtype=torch.long)
    return row[None, :].expand(TINY_BATCH, -1).contiguous()


@pytest.fixture
def target_state() -> torch.Tensor:
    """A seeded target state at the tiny anchor set, as the proposal head conditions on it.

    Returns:
        A $(B, A, d_h)$ float tensor, with $A$ matching the :func:`anchors` fixture.
    """
    generator = torch.Generator().manual_seed(20260910)
    n_anchors = TINY_SEQ_LEN - TINY_FLOOR
    return torch.randn(TINY_BATCH, n_anchors, TINY_D_MODEL, generator=generator)


# =================================================================================================
# The generated integer-operator cohort fixture, and the run built on it
#
# The committed ``tiny_shard_causal_int.hdf5`` is one file whose ``target`` is all zeros, so every
# class-, subgroup- and cohort-aware path of the family's evaluation self-skips against it. These
# fixtures generate the multi-cohort set instead -- eight subgroup shards, real class codes, the
# clinical fields -- through ``scripts/make_tiny_shard.py``'s ``causal_cohort`` mode, written
# under the INTEGER phase operator this architecture's first experiment is defined on. The
# lag-attentive cells' cohort fixture is the same mode under the legacy operator, and a shard
# written under one operator cannot be read by a model built under the other.
#
# The same rule binds here as in the causal cell's suite: eight real raw segments re-used under
# distinct identities, scored by a tiny model trained for two epochs, are evidence about SCHEMA,
# SHAPE, FINITENESS, DENOMINATORS, COHORT MEMBERSHIP, COUNTS, IDENTITIES and REFUSALS, and about
# nothing else. No test may assert the sign, magnitude, direction or significance of any clinical
# or statistical effect on them.
# =================================================================================================
#: Epochs the cohort fit runs. Two rather than one: the second is what steps the scheduler and
#: rotates the tile phase, and a checkpoint from a model that never left its initialisation would
#: make every source margin identically zero for a reason about the fit.
COHORT_FIT_EPOCHS = 2

#: Segments the single-lag predictive profile is scored on in the cohort run. Fewer than the
#: fixture holds, so the cap is exercised as a cap rather than as a whole-split pass.
COHORT_PROFILE_SEGMENTS = 4


def write_int_cohort_shards(directory: Path) -> Sequence[str]:
    """Generate the eight causal subgroup shards under the integer operator, and their statistics.

    Driven through ``scripts/make_tiny_shard.py``'s own entry point, exactly as the causal cell's
    suite drives the legacy-operator set, with one flag more: the phase operator.

    Args:
        directory: Destination directory.

    Returns:
        The eight shard paths, in canonical subgroup order.
    """
    from scripts.make_tiny_shard import COHORT_SUBGROUPS, PHASE_OPERATOR_INTEGER
    from scripts.make_tiny_shard import _cli as make_tiny_shard_cli
    from teb_vae.lag_attn_cfs.tests.conftest import COHORT_SEQ_LEN

    exit_code = make_tiny_shard_cli(
        [
            "--variants", "causal_cohort",
            "--out-dir", str(directory),
            "--seq-len", str(COHORT_SEQ_LEN),
            "--phase-operator", PHASE_OPERATOR_INTEGER,
        ]
    )
    assert exit_code == 0
    return [str(Path(directory) / f"{subgroup}.hdf5") for subgroup in COHORT_SUBGROUPS]


@pytest.fixture(scope="session")
def int_cohort_shards(tmp_path_factory) -> Sequence[str]:
    """Paths to the eight generated integer-operator subgroup shards. Session-scoped; read-only."""
    return write_int_cohort_shards(tmp_path_factory.mktemp("causal_cohort_int"))


@pytest.fixture(scope="session")
def int_cohort_stats(int_cohort_shards) -> str:
    """The statistics file the generator wrote from those same shards, over the whole set."""
    from teb_vae.lag_attn_cfs.tests.conftest import COHORT_STATS_FILENAME

    return str(Path(int_cohort_shards[0]).parent / COHORT_STATS_FILENAME)


#: The two smoke training profiles a cohort fit may run: the shipped smoke configuration, and its
#: short-bank twin, which is the fixture-scale profile the short-window evaluation delta is
#: declared for.
SMOKE_CONFIG = "tiny.yaml"
SHORT_SMOKE_CONFIG = "tiny_lag25.yaml"

#: The short-window evaluation delta, declared for the window the short-bank profile builds.
SHORT_EVAL_OVERRIDES = "lag25_eval_overrides.yaml"


def fit_cohort(
    config_name: str,
    shards: Sequence[str],
    stats: str,
    run_root: Path,
    *,
    model_overrides: Optional[Mapping[str, Any]] = None,
) -> Path:
    """One real fit of a smoke profile against the generated cohort shards; the run directory.

    Driven through ``trainer.main`` rather than by assembling a checkpoint by hand, because what
    an evaluation reads out of this directory is precisely what the driver puts there: the
    warm-up tuples the budget resolved against these shards, the representation stamp, and
    ``resolved_config.yaml``.

    Args:
        config_name: The smoke configuration filename to fit.
        shards: The cohort shards, used for training and testing alike.
        stats: The statistics file written from those shards.
        run_root: Where the run writes.
        model_overrides: ``model_config.VAE_model`` leaves to set before the fit, for a profile
            layered on the smoke configuration rather than shipped as a file.

    Returns:
        The run directory holding ``model_checkpoints``.
    """
    import yaml

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.tests.conftest import absolutize_dataset_paths
    from teb_vae.lag_slot_transformer_cfs import trainer as trainer_module

    source = Path(_REPO_ROOT) / "teb_vae" / "lag_slot_transformer_cfs" / "configs" / config_name
    config = absolutize_dataset_paths(load_config(str(source)))
    dataset = config["dataset_config"]
    dataset["vae_train_datasets"] = list(shards)
    dataset["vae_test_datasets"] = list(shards)
    dataset["stat_path"] = stats
    config["general_config"]["folders_config"]["out_dir_base"] = str(run_root)
    config["general_config"]["epochs"] = COHORT_FIT_EPOCHS
    config["advanced_config"]["trainer"]["profiler"] = None
    config["model_config"]["VAE_model"].update(dict(model_overrides or {}))

    config_path = run_root / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer_module.main(str(config_path))

    checkpoint_dirs = sorted(run_root.rglob("model_checkpoints"))
    assert len(checkpoint_dirs) == 1, checkpoint_dirs
    return checkpoint_dirs[0].parent


def repoint_overrides(
    overrides_path: Path,
    shards: Sequence[str],
    stats: str,
    destination: Path,
    *,
    occlusion_bands: Optional[Mapping[str, Sequence[int]]] = None,
) -> Path:
    """A shipped evaluation delta with its placeholder leaves repointed at the fixture cohort.

    Exactly the edit an operator makes before a real run: a delta carrying only the shard paths
    would replace the committed one, and with it the clinical ``load_fields`` every cohort-aware
    readout is asked in. The single-lag profile cap is lowered so it is exercised as a cap.

    Args:
        overrides_path: The shipped delta.
        shards: The cohort shards.
        stats: The statistics file written from them.
        destination: Where to write the repointed delta.
        occlusion_bands: Bands to replace the shipped ones with, or ``None`` to keep the delta's
            own -- which is right when the delta is declared for the window the fit built.

    Returns:
        The repointed delta's path.
    """
    import yaml

    from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides

    overrides = load_eval_overrides(overrides_path)
    overrides["dataset_config"]["vae_test_datasets"] = list(shards)
    overrides["dataset_config"]["stat_path"] = stats
    overrides["general_config"]["batch_size"]["test"] = 4
    if occlusion_bands is not None:
        overrides["eval_config"]["occlusion_bands"] = dict(occlusion_bands)
    overrides["eval_config"]["caps"]["lag_profile"] = COHORT_PROFILE_SEGMENTS
    # The schema's floor: the fixture holds a handful of recordings and a wider interval is not
    # made narrower by resampling it more.
    overrides["eval_config"]["bootstrap_resamples"] = 100
    destination.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return destination


def collect_run(run_dir: Path, overrides_path: Path, output_dir: Path) -> Dict[str, Any]:
    """One real evaluation of a fitted run through the family's runner; every artifact it left.

    Two Monte Carlo draws rather than the shipped count: these tests are about the plumbing, and
    each draw decodes every arm over every anchor.

    Args:
        run_dir: The training run directory holding ``model_checkpoints``.
        overrides_path: The repointed evaluation delta.
        output_dir: Where the evaluation writes.

    Returns:
        The checkpoint, the exit code, the summary path, its text, the parsed summary and the
        results directory.
    """
    import json

    from teb_vae.lag_slot_transformer_cfs.eval import run as run_module

    checkpoint = sorted((Path(run_dir) / "model_checkpoints").glob("*.ckpt"))[0]
    exit_code = run_module.main(
        checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        overrides=str(overrides_path),
        device="cpu",
        num_samples=2,
        argument_sources={"checkpoint": "cli", "overrides": "cli"},
    )
    results_dir = output_dir / run_module.RESULTS_DIRNAME
    summary_path = results_dir / run_module.SUMMARY_FILENAME
    text = summary_path.read_text(encoding="utf-8")
    return {
        "checkpoint": checkpoint,
        "exit_code": exit_code,
        "summary_path": summary_path,
        "text": text,
        "summary": json.loads(text),
        "results_dir": results_dir,
    }


@pytest.fixture(scope="session")
def slot_cohort_run(int_cohort_shards, int_cohort_stats, tmp_path_factory) -> Path:
    """One real fit of this cell against the generated cohort shards; returns the run directory.

    Marked ``slow`` at every consumer rather than here, and nothing in the fast subset may depend
    on it.
    """
    return fit_cohort(
        SMOKE_CONFIG, int_cohort_shards, int_cohort_stats,
        Path(tmp_path_factory.mktemp("slot_cohort_run")),
    )


@pytest.fixture(scope="session")
def slot_cohort_overrides(int_cohort_shards, int_cohort_stats, tmp_path_factory) -> Path:
    """This package's committed evaluation delta with its placeholder leaves repointed.

    The lag bands are rescaled to the tiny window through the causal cell's helper, because the
    committed partition is declared for the production window and the smoke fit builds a
    smaller one.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.tests.conftest import _tiny_occlusion_bands
    from teb_vae.lag_slot_transformer_cfs.eval.binding import LAG_RESIDUAL_BINDING

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_slot_transformer_cfs" / "configs" / SMOKE_CONFIG
    bands = _tiny_occlusion_bands(
        int(load_config(str(tiny))["model_config"]["VAE_model"]["max_lag"])
    )
    return repoint_overrides(
        LAG_RESIDUAL_BINDING.overrides_path, int_cohort_shards, int_cohort_stats,
        Path(tmp_path_factory.mktemp("slot_eval_overrides")) / "eval_overrides_repointed.yaml",
        occlusion_bands=bands,
    )


@pytest.fixture(scope="session")
def slot_collected_run(slot_cohort_run, slot_cohort_overrides, tmp_path_factory) -> Dict[str, Any]:
    """One real evaluation of this cell through the family's runner; every artifact it left.

    Marked ``slow`` at every consumer rather than here.
    """
    return collect_run(slot_cohort_run, slot_cohort_overrides, Path(tmp_path_factory.mktemp("slot_eval")))


# =================================================================================================
# The short-bank profile under the target input ablation, end to end
#
# A second fit and a second evaluation, of the short-bank smoke twin with the FHR order-zero
# coefficient ablated at the input, scored under the evaluation delta declared for that window.
# The two profiles are new and each has behaviour no other fixture exercises: a lag axis of
# another length under the same figures, six declared bands two of which overlap the partition,
# an ablated coordinate that every baseline, control and attribution must not read, and an
# acceptance pass that reads the run under its own window's family.
# =================================================================================================
@pytest.fixture(scope="session")
def slot_short_ablated_run(int_cohort_shards, int_cohort_stats, tmp_path_factory) -> Path:
    """One real fit of the short-bank smoke twin with the target ablation on; the run directory."""
    return fit_cohort(
        SHORT_SMOKE_CONFIG, int_cohort_shards, int_cohort_stats,
        Path(tmp_path_factory.mktemp("slot_short_run")),
        model_overrides={"zero_fhr_scattering_s0": True},
    )


@pytest.fixture(scope="session")
def slot_short_overrides(int_cohort_shards, int_cohort_stats, tmp_path_factory) -> Path:
    """The short-window evaluation delta repointed at the cohort, its own bands kept.

    The bands are not rescaled: the delta is declared for exactly the window the short-bank
    smoke twin builds, which is the point of scoring that fit under it.
    """
    from teb_vae.lag_slot_transformer_cfs.eval.binding import LAG_RESIDUAL_BINDING

    return repoint_overrides(
        LAG_RESIDUAL_BINDING.overrides_path.parent / SHORT_EVAL_OVERRIDES,
        int_cohort_shards, int_cohort_stats,
        Path(tmp_path_factory.mktemp("slot_short_overrides")) / "short_overrides_repointed.yaml",
    )


@pytest.fixture(scope="session")
def slot_short_collected_run(
    slot_short_ablated_run, slot_short_overrides, tmp_path_factory
) -> Dict[str, Any]:
    """One real evaluation of the short-bank ablated fit under the short-window delta."""
    return collect_run(
        slot_short_ablated_run, slot_short_overrides, Path(tmp_path_factory.mktemp("slot_short_eval"))
    )
