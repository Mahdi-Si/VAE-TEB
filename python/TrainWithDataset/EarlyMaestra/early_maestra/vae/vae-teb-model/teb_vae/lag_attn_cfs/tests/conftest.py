r"""Shared pytest configuration for the causal-feature-domain lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` and ``hdf5_dataset.*``
imports resolve no matter which directory pytest is invoked from, and exposes the committed causal
fixture plus the configuration builder every refusal test starts from. Mirrors
``teb_vae/lag_attn_fs/tests/conftest.py``, including its ``utils`` pre-import pin.

**The config is built, not loaded from a file.** This package ships no YAML yet, and a builder is
what the refusal tests need anyway: nearly every one of them is "the shipped configuration with one
leaf moved", and a helper that takes overrides states the delta at the call site instead of
repeating three nested dictionaries per test.

The numbers below are the shipped geometry, and the ones a resolved budget is checked against are
**derived in the tests from the fixture's own attributes** rather than restated here. A hand-copied
expectation would pass against a fixture rebuilt at another quantile, which is the exact failure
the resolver exists to refuse.
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pytest
import torch

# teb_vae/lag_attn_cfs/tests/conftest.py -> parents[0]=tests, [1]=lag_attn_cfs, [2]=teb_vae,
# [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a
# near-empty one at model/vae_teb_prediction/utils. On a repository-wide run another conftest can
# put the latter's parent first on ``sys.path``, shadowing the real one. Binding the repo-root
# package now -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for every later
# ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

# Importing a fixture binds it in this conftest's namespace, which is all pytest needs to serve it
# to the tests in this directory. The delta heads are zero-initialised in every model of the
# family, so at initialisation every KL assertion passes vacuously in all of them, and the escape
# from that trap is one shared fixture rather than a copy per package.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401

#: The committed causal shard, and the two-sided one beside it. Both belong to ``lag_attn``: every
#: model in the family reads the same fixtures through the same loader, and the two-sided file is
#: what makes "a causal shard is required" a comparison rather than an assertion about one file.
FIXTURES = Path(_REPO_ROOT) / "teb_vae" / "lag_attn" / "tests" / "fixtures"
CAUSAL_SHARD = FIXTURES / "tiny_shard_causal.hdf5"
TWO_SIDED_SHARD = FIXTURES / "tiny_shard.hdf5"

#: The shipped warm-up threshold, in decimated steps of the trimmed window. $134$ is where
#: ``fhr_ph`` tops out, so every phase channel survives it, and the ``fhr_st`` staircase has a
#: channel at exactly $134$ with the next at $162$ -- the boundary lands on a frequency edge the
#: pipeline independently believes in rather than on an arbitrary cut.
SHIPPED_BUDGET_STEPS = 134

#: The shipped anchor floor. $F = B - 1$ is the smallest value the pairing admits: a forecast at
#: anchor $t$ reads target time $t + 1$ at the earliest, so $t + 1 \ge W'_c$ for every kept channel
#: needs $F \ge B - 1$ and nothing more.
SHIPPED_WARMUP_PERIOD = 133

#: The shipped forecast horizon (one minute) and the sequence length the loader's trim produces.
SHIPPED_HORIZON = 15
SHIPPED_SEQUENCE_LENGTH = 300

#: The trim the shipped configuration reads the shards at, which is also what the committed
#: statistics were accumulated with.
SHIPPED_TRIM_MINUTES = 1.0

#: The causal stream widths: $36 + 66$ target and $36 + 15$ source, against the two-sided $109$ and
#: $58$. Seven scattering channels per block were dropped at write time because their warm-up
#: outruns the stored segment at every trim; both phase blocks keep their full width.
CAUSAL_ST_WIDTH = 36
CAUSAL_PH_WIDTH = 66
CAUSAL_C_Y = CAUSAL_ST_WIDTH + CAUSAL_PH_WIDTH
CAUSAL_C_U = CAUSAL_ST_WIDTH + 15


def causal_config(
    *,
    paths: Optional[Sequence[Path]] = None,
    trim_minutes: Optional[float] = SHIPPED_TRIM_MINUTES,
    **vae_overrides: Any,
) -> Dict[str, Any]:
    """Build a configuration the warm-up resolver accepts, with the named leaves moved.

    Args:
        paths: Shards for both splits. Defaults to the committed causal fixture alone. A two-entry
            sequence is read as ``(train, test)``, which is how the disagreeing-shard tests plant
            their second file.
        trim_minutes: The loader's symmetric trim.
        **vae_overrides: Leaves to replace inside ``model_config.VAE_model``. A value of ``None``
            is applied rather than skipped, so a test can remove a key the resolver requires.

    Returns:
        A fresh config dict, safe to mutate.
    """
    shards = [CAUSAL_SHARD] if paths is None else list(paths)
    train = [str(shards[0])]
    test = [str(shards[-1])]

    vae: Dict[str, Any] = {
        "causal_warmup_budget_steps": SHIPPED_BUDGET_STEPS,
        "causal_reach_budget_s": None,
        "sequence_length": SHIPPED_SEQUENCE_LENGTH,
        "horizon": SHIPPED_HORIZON,
        "warmup_period": SHIPPED_WARMUP_PERIOD,
        # The shipped tiling, stated rather than defaulted: consecutive forecast windows partition
        # the timeline instead of overlapping, so no target coefficient is scored twice in a step.
        "anchor_stride": SHIPPED_HORIZON,
        "c_y": CAUSAL_C_Y,
        "c_u": CAUSAL_C_U,
        "use_up_st": True,
    }
    vae.update(vae_overrides)
    return {
        "model_config": {"VAE_model": vae},
        "dataset_config": {
            "vae_train_datasets": train,
            "vae_test_datasets": test,
            "dataloader_config": {"dataset_kwargs": {"trim_minutes": trim_minutes}},
        },
    }


def stored_warmup(path: Path = CAUSAL_SHARD) -> Dict[str, Any]:
    """Read one shard's stored (untrimmed) warm-up vectors straight off its attributes.

    The tests' expectations are derived from this rather than written out, so a fixture rebuilt at
    another ``causal_warmup_quantile`` fails them instead of passing against stale constants.

    Args:
        path: The shard to read.

    Returns:
        ``{block: (C,) int64}`` in untrimmed step coordinates.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        return {
            name: np.asarray(handle[name].attrs["causal_warmup_steps"], dtype=np.int64)
            for name in ("fhr_st", "fhr_ph", "up_st", "up_ph")
        }


def write_variant(source: Path, destination: Path, mutate) -> Path:
    """Copy a shard and apply one mutation to the open copy.

    Args:
        source: The shard to copy.
        destination: Where to write it.
        mutate: Callable taking the open ``h5py.File`` in append mode.

    Returns:
        ``destination``.
    """
    import shutil

    import h5py

    shutil.copyfile(source, destination)
    with h5py.File(destination, "a") as handle:
        mutate(handle)
    return destination


def without_key(config: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """A deep copy of ``config`` with one ``VAE_model`` key removed entirely.

    Distinct from setting it to ``None``: an absent key and a null one must both be refused, and a
    ``get`` that treats them alike is exactly the sort of thing a test should not assume.

    Args:
        config: The config to copy.
        key: The ``model_config.VAE_model`` key to drop.

    Returns:
        The copy.
    """
    copied = copy.deepcopy(dict(config))
    copied["model_config"]["VAE_model"].pop(key, None)
    return copied


# =================================================================================================
# The model fixtures
# =================================================================================================
#: Decimated steps, samples and the tiny model's geometry. Larger than the siblings' $16$-step
#: fixture on purpose: a tiling needs a floor, a stride and room for more than one tile, and at
#: $T = 16$ with any usable floor there is only one anchor per phase -- which would leave every
#: padding assertion below passing without a padded slot ever existing.
BATCH = 2
TINY_SEQ_LEN = 24
TINY_HORIZON = 4
TINY_WARMUP_PERIOD = 5
TINY_STRIDE = 4

#: The tiny warm-up staircase, per **declared** channel. Hand-built rather than read from the
#: committed shard: the real vectors reach $W' = 278$ against a $300$-step window, and no tiny
#: geometry can carry them. What is reproduced is the shape that matters -- most channels honest
#: immediately, a slow tail that the budget cuts -- so the gather, the mask and the announcement
#: are all exercised.
TINY_BUDGET_STEPS = 6
TINY_TARGET_WARMUP_FULL = tuple(min(index // 8, 9) for index in range(CAUSAL_C_Y))
TINY_SOURCE_WARMUP_FULL = tuple(min(index // 4, 12) for index in range(CAUSAL_C_U))

#: The resolved form of the two vectors above: the surviving target channels and their waits, and
#: the source's identity keep-index, exactly as ``resolve_warmup_budget`` produces them.
TINY_TARGET_KEEP_INDEX = tuple(
    index for index, step in enumerate(TINY_TARGET_WARMUP_FULL) if step <= TINY_BUDGET_STEPS
)
TINY_TARGET_WARMUP_STEPS = tuple(
    TINY_TARGET_WARMUP_FULL[index] for index in TINY_TARGET_KEEP_INDEX
)
TINY_SOURCE_KEEP_INDEX = tuple(range(CAUSAL_C_U))
TINY_SOURCE_WARMUP_STEPS = TINY_SOURCE_WARMUP_FULL

#: The tiny constructor keyword set, **ungated**: no budget, no gate, no warm-up mask. The guarded
#: variants are built from it by :func:`tiny_warmup_kwargs`, so the two differ in the guard alone.
TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=TINY_SEQ_LEN,
    d_model=32,
    d_z=8,
    horizon=TINY_HORIZON,
    raw_per_step=16,
    warmup_period=TINY_WARMUP_PERIOD,
    c_y=CAUSAL_C_Y,
    c_u=CAUSAL_C_U,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    lstm_layers=1,
    horizon_film=True,
    dropout=0.0,
)


def tiny_warmup_kwargs(kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    """Add the tiny warm-up guard to a constructor keyword set.

    Args:
        kwargs: The keyword set to guard. Defaults to :data:`TINY_KWARGS`.
        **overrides: Applied after the guard, so a test can move one leaf of it.

    Returns:
        A fresh dict carrying the guard on both streams.
    """
    guarded = dict(
        TINY_KWARGS if kwargs is None else kwargs,
        target_keep_index=TINY_TARGET_KEEP_INDEX,
        target_warmup_steps=TINY_TARGET_WARMUP_STEPS,
        source_keep_index=TINY_SOURCE_KEEP_INDEX,
        source_warmup_steps=TINY_SOURCE_WARMUP_STEPS,
    )
    guarded.update(overrides)
    return guarded


def shipped_warmup_kwargs(**overrides: Any) -> Dict[str, Any]:
    """The production constructor call: the shipped geometry plus the budget the shards resolve to.

    Built through :func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget` against the
    committed causal fixture rather than written out, so the surviving-channel count comes from the
    data every time and a rebuilt fixture moves the model instead of failing an unrelated literal.

    Args:
        **overrides: Applied last.

    Returns:
        Constructor kwargs. No ``decoder_out_channels``: the width is the model's to resolve.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
    from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    kwargs = dict(
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        d_model=128,
        d_z=64,
        horizon=SHIPPED_HORIZON,
        raw_per_step=16,
        warmup_period=SHIPPED_WARMUP_PERIOD,
        c_y=CAUSAL_C_Y,
        c_u=CAUSAL_C_U,
        use_up_st=True,
        max_lag=90,
        num_heads=4,
        d_head=32,
        lstm_layers=2,
        dropout=0.1,
        decoder_hidden=256,
        horizon_depth=4,
        horizon_kernel=3,
        horizon_film=True,
        horizon_attention_blocks=2,
        horizon_embed_std=0.8,
        head_init_calibration=True,
        a_head_gain=2.0,
        encoder_extra_dilations=(8, 16),
        encoder_extra_kernel=15,
        logvar_clamp=(-5.0, 3.0),
        mu_scale=5.0,
        delta_mu_scale=3.0,
        delta_logvar_scale=2.0,
        use_entmax=True,
        lag_bias_init="alibi_decay",
        causal_norm=True,
        coverage_floor=0.9,
        anchor_stride=SHIPPED_HORIZON,
        **warmup_model_kwargs(resolved, SeqVaeLagAttnCfs),
    )
    # Updated rather than splatted into the literal above: an override naming a key the literal
    # already sets -- ``horizon`` is the one every arm moves -- is a TypeError there and a
    # replacement here, which is what "applied last" has to mean to be usable at all.
    kwargs.update(overrides)
    return kwargs


def make_streams(
    kwargs: Mapping[str, Any], batch: int = BATCH, seed: int = 0
):
    """Seeded ``(y_st, y_ph, u_stream)`` at the declared widths of a keyword set.

    The target blocks are $36 + 66$ and the source $36 + 15$: the causal channel counts, which is
    what makes a stream built here refusable by a model declaring the two-sided widths.

    Args:
        kwargs: A constructor keyword set, read for its sequence length and source width.
        batch: Samples in the batch.
        seed: Seed, so a batch is reproducible.

    Returns:
        The three input tensors.
    """
    generator = torch.Generator().manual_seed(seed)
    length = int(kwargs["sequence_length"])
    width = int(kwargs["c_u"])
    return (
        torch.randn(batch, length, CAUSAL_ST_WIDTH, generator=generator),
        torch.randn(batch, length, CAUSAL_PH_WIDTH, generator=generator),
        torch.randn(batch, length, width, generator=generator),
    )


# =================================================================================================
# The batch fixtures, and the task the driver wraps the net in
# =================================================================================================
#: The decimated step index of the deliberate gap every stub batch carries. Inside the tiny trained
#: anchor range $[F, T - H) = [5, 20)$, so the gap is visible to every mask -- a uniformly valid
#: weight would leave every mask assertion green whether or not the masks work.
STUB_GAP_STEP = 12

#: The loss hyperparameters the shipped config sets, as the task's constructor takes them.
#: ``beta_schedule=None`` means the constant ``kld_beta`` applies, which keeps the schedule out of
#: the way of tests that are not about it.
TASK_HPARAMS: Dict[str, Any] = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


def make_stub_batch(
    batch: int = BATCH, seq_len: int = TINY_SEQ_LEN, seed: int = 0, guid_prefix: str = "SEG"
):
    """Build a batch exposing every field the task reads, at the **causal** channel widths.

    A ``SimpleNamespace`` rather than the real batch type: the task reads batch fields as
    attributes, and standing up an HDF5 loader to test a loss would couple every task test to the
    data layer. The real batch contract is asserted against the committed shard elsewhere.

    ``guid`` and ``epoch`` are here rather than optional, and they are the one thing this batch
    carries that the two-sided siblings' does not need: the anchor tiling's phase is keyed on the
    pair, and a batch without them would make every phase assertion below a test of the refusal
    rather than of the derivation. ``epoch`` is ``domain_start`` in seconds and is **per segment**,
    which is exactly why segments of one recording get different tile grids.

    Args:
        batch: Samples in the batch. At least $2$ to be derangeable.
        seq_len: Decimated sequence length; the raw signals are ``16 * seq_len`` long.
        seed: Seed, so a batch is reproducible.
        guid_prefix: Prefix of the per-sample recording identifiers.

    Returns:
        An object with ``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``, ``fhr``, ``up``, ``weight``,
        ``guid`` and ``epoch``.
    """
    import types

    generator = torch.Generator().manual_seed(seed)
    weight = torch.ones(batch, seq_len)
    weight[:, STUB_GAP_STEP] = 0.0
    return types.SimpleNamespace(
        fhr_st=torch.randn(batch, seq_len, CAUSAL_ST_WIDTH, generator=generator),
        fhr_ph=torch.randn(batch, seq_len, CAUSAL_PH_WIDTH, generator=generator),
        up_st=torch.randn(batch, seq_len, CAUSAL_ST_WIDTH, generator=generator),
        up_ph=torch.randn(batch, seq_len, CAUSAL_C_U - CAUSAL_ST_WIDTH, generator=generator),
        fhr=torch.randn(batch, 16 * seq_len, generator=generator),
        up=torch.randn(batch, 16 * seq_len, generator=generator),
        weight=weight,
        guid=[f"{guid_prefix}{index:03d}" for index in range(batch)],
        # Negative seconds before delivery, distinct per sample and 20 minutes apart -- the stride
        # between stored segments, so this reads as consecutive segments of one recording would.
        epoch=torch.tensor([-36000.0 + 1200.0 * index for index in range(batch)]),
    )


def make_task(model_kwargs: Optional[Mapping[str, Any]] = None,
              hparams: Optional[Mapping[str, Any]] = None,
              **task_kwargs):
    """Build this model wrapped in its task, with the production loss hparams applied.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to the guarded tiny set, because the guard is
            what this package is about: an ungated model has no warm-up, no tertiles and no source
            warmth to report, so an unguarded default would leave most of the metric surface
            untested.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnCfsTask`` with ``setup()`` already called, so the permutation generator
        exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask

    # The tiling stride, not the constructor's inert 1: a task built at stride 1 would resolve the
    # same anchor set on every stage and every phase assertion below it would hold vacuously.
    kwargs = dict(
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if model_kwargs is None else model_kwargs
    )
    model = build(kwargs)
    task = SeqVaeLagAttnCfsTask(
        model,
        lr=1e-3,
        model_kwargs=kwargs,
        **dict(TASK_HPARAMS, **(hparams or {})),
        **task_kwargs,
    )
    task.setup("fit")  # seeds the permutation generator; Lightning would call this itself
    return task


#: Global-seeding calls no shipped module may make. The **call**, not the name: the task's own
#: docstring has to name the framework's seeding route to explain where the run seed comes from,
#: and a bare substring scan would read that sentence as a hand seed.
HAND_SEEDING_CALLS: Sequence[str] = (
    "torch.manual_seed",
    "seed_everything",
    "np.random.seed",
)

#: The one context manager that makes a global seed harmless, because it restores the stream it
#: found on the way out. ``eval/oracle.py`` seeds inside it and has to: the probe's parameters are
#: initialised by ``nn.init``, which takes no generator, so there is no way to make a probe
#: reproducible without touching the global RNG -- and no analysis that runs afterwards may see a
#: stream it would not otherwise have seen.
SEEDING_FORK = "torch.random.fork_rng"


def hand_seeding_offenders(package_dir: Path) -> list:
    """Return every shipped module that seeds a global generator outside a forked RNG.

    ``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route this package has; a stray global seed would silently override it while looking like
    diligence -- and here it would additionally move every tile phase, since the seed is one of
    the four halves of the phase key.

    A call **lexically inside** a ``with torch.random.fork_rng(...)`` block is exempt, and the
    exemption is a property rather than a name: that block restores the global state on exit, so a
    seed inside it provably cannot move the stream anything downstream draws from. Walked as an
    AST rather than searched as text, because a substring scan can see neither the enclosing
    ``with`` nor the difference between a call and a docstring mentioning one.

    Args:
        package_dir: The package root. ``tests`` is skipped: a test seeding itself for
            reproducibility is doing the right thing.

    Returns:
        One ``"<file>: <call>"`` per offending call site.
    """
    import ast

    def _dotted(node: Any) -> str:
        """Return the dotted name of a call target, or ``''`` for anything else."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        parts.append(node.id)
        return ".".join(reversed(parts))

    offenders = []
    for path in sorted(package_dir.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forked = {
            id(statement)
            for node in ast.walk(tree)
            if isinstance(node, (ast.With, ast.AsyncWith))
            and any(_dotted(item.context_expr.func) == SEEDING_FORK
                    for item in node.items
                    if isinstance(item.context_expr, ast.Call))
            for statement in ast.walk(node)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            if name in HAND_SEEDING_CALLS and id(node) not in forked:
                offenders.append(f"{path.name}: {name}(")
    return offenders


def absolutize_dataset_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a config's shard and statistics paths to absolute, in place.

    The shipped paths are repo-root-relative because the entry points run from the repo root; a test
    that drives the loader from pytest's working directory needs them absolute. Shared rather than
    repeated per test file: a miss surfaces as the loader's opaque "No samples match the specified
    filters" rather than as a path error.

    Args:
        config: A loaded config dict.

    Returns:
        The same dict.
    """
    dataset = config["dataset_config"]
    for key in ("vae_train_datasets", "vae_test_datasets"):
        dataset[key] = [str(Path(_REPO_ROOT) / path) for path in dataset[key]]
    dataset["stat_path"] = str(Path(_REPO_ROOT) / dataset["stat_path"])
    return config


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


# =================================================================================================
# The generated causal cohort fixture, and the run built on it
#
# The committed ``tiny_shard_causal.hdf5`` is one file whose ``target`` is all zeros and whose
# ``weight`` is all ones, so every class-, subgroup- and cohort-aware path in the evaluation
# pipeline self-skips against it and only the fallback branches are ever exercised. These fixtures
# generate the multi-cohort set instead -- eight subgroup shards, real class codes, the clinical
# fields -- through ``scripts/make_tiny_shard.py``'s ``causal_cohort`` mode.
#
# **What may be asserted on them, and what may not.** The shards are eight real raw segments from a
# single production shard (``hie_cs.hdf5``), re-used under distinct identities across eight cohort
# shards, scored -- where a model is involved at all -- by a tiny model trained for a handful of
# steps. They are therefore evidence about SCHEMA, SHAPE, FINITENESS, DENOMINATORS, COHORT
# MEMBERSHIP, COUNTS, IDENTITIES and REFUSALS, and about nothing else. No test may assert the sign,
# magnitude, direction or significance of any clinical or statistical effect on them: not that a
# forecast gap is positive, not that one cohort differs from another, not that the coupling exceeds
# the availability clock, not that a lag peak lands anywhere in particular. Every one of those is a
# finding about a model and a population, and this fixture is neither. Where a test needs a
# direction to be non-vacuous it must CONSTRUCT the condition rather than hope the fixture supplies
# it.
#
# They are generated into ``tmp_path_factory`` and never committed: regenerating the family's
# committed binaries to add cohort fields would perturb every number the existing suites are pinned
# against.
# =================================================================================================
#: What the ``causal_cohort`` mode names its statistics file, and the on-disk length it writes.
#: Restated here rather than imported so a fixture cannot silently follow an edit to the script's
#: ``RUN_ARGS``; the generator is driven with both stated explicitly.
COHORT_STATS_FILENAME = "tiny_stats_causal_cohort.hdf5"
COHORT_SEQ_LEN = 330


def write_cohort_shards(directory: Path) -> Sequence[str]:
    """Generate the eight causal subgroup shards and their statistics file into ``directory``.

    Driven through ``scripts/make_tiny_shard.py``'s own entry point rather than by calling its
    writers directly, so the script stays load-bearing: a change to how a causal shard is written
    reaches this suite without anything here being updated. ``--seq-len`` is passed explicitly
    because the merge lets that script's ``RUN_ARGS`` supply it otherwise, and a fixture whose
    geometry depended on a hand-edited dict would move under the tests silently.

    Args:
        directory: Destination directory.

    Returns:
        The eight shard paths, in canonical subgroup order.
    """
    from scripts.make_tiny_shard import COHORT_SUBGROUPS, main as make_tiny_shard_main

    exit_code = make_tiny_shard_main(
        [
            "--variants", "causal_cohort",
            "--out-dir", str(directory),
            "--seq-len", str(COHORT_SEQ_LEN),
        ]
    )
    assert exit_code == 0
    return [str(Path(directory) / f"{subgroup}.hdf5") for subgroup in COHORT_SUBGROUPS]


@pytest.fixture(scope="session")
def cohort_shards(tmp_path_factory) -> Sequence[str]:
    """Paths to the eight generated causal subgroup shards. Session-scoped; treat as read-only."""
    return write_cohort_shards(tmp_path_factory.mktemp("causal_cohort"))


@pytest.fixture(scope="session")
def cohort_stats(cohort_shards) -> str:
    """The statistics file the generator wrote from those same shards.

    One file over the whole set, produced by the real calculator: the dataset reader silently
    disables normalization on any stats-schema mismatch, so a hand-rolled stats file is the one
    shortcut here with a genuinely bad failure mode -- every shape stays right and every number
    becomes wrong. On the causal variant the calculator additionally excludes each channel's
    warm-up region, which is what makes zero the channel mean over the region the model reads.
    """
    return str(Path(cohort_shards[0]).parent / COHORT_STATS_FILENAME)


@pytest.fixture(scope="session")
def cohort_config(cohort_shards, cohort_stats) -> Dict[str, Any]:
    """The tiny training config with the committed evaluation delta merged over it, repointed.

    Built exactly as a run is -- the resolved config first, then
    :func:`~teb_vae.lag_attn_cfs.eval.config_schema.merge_eval_overrides`, then the repoint that
    stands in for editing the ``REPOINT_ME`` placeholders -- so the committed overrides file is
    load-bearing in this suite rather than merely parsed by one test.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.eval.config_schema import (
        force_single_process_loader,
        merge_eval_overrides,
        validate_eval_config,
    )

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_cfs" / "configs" / "tiny.yaml"
    config = copy.deepcopy(
        merge_eval_overrides(absolutize_dataset_paths(load_config(str(tiny))))
    )
    dataset = config["dataset_config"]
    dataset["vae_test_datasets"] = list(cohort_shards)
    dataset["stat_path"] = cohort_stats
    # Four per batch: enough batches that a per-batch bug cannot hide in a single pass.
    config["general_config"]["batch_size"]["test"] = 4
    force_single_process_loader(config)
    config["eval_config"] = validate_eval_config(config)
    return config


@pytest.fixture(scope="session")
def cohort_loader(cohort_config):
    """A real test dataloader over the generated cohort shards."""
    from train.data_module import GraphDataModule

    return GraphDataModule(cohort_config).test_dataloader()


@pytest.fixture(scope="session")
def cohort_run(cohort_shards, cohort_stats, tmp_path_factory) -> Path:
    r"""One real fit against the generated cohort shards; returns the run directory.

    Marked ``slow`` at every consumer rather than here -- a fixture carries no marker of its own --
    and nothing in the fast subset may depend on it.

    Driven through ``trainer.main`` rather than by assembling a checkpoint by hand, because what an
    evaluation reads out of this directory is precisely what the driver puts there and nothing else
    can: ``model_kwargs`` carrying the four warm-up tuples the budget resolved against these shards,
    and ``resolved_config.yaml`` recording the configuration that produced them. A blob saved from a
    freshly constructed model would carry the same keys and would not prove the binding.

    The base is the shipped ``tiny.yaml``, with exactly three leaves moved: both shard lists and
    the statistics path, which is the repoint; and ``out_dir_base``, because the shipped value is a
    path inside the repository and a test must not write there.
    """
    import yaml

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs import trainer as trainer_module

    run_root = tmp_path_factory.mktemp("cohort_run")
    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_cfs" / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    dataset = config["dataset_config"]
    dataset["vae_train_datasets"] = list(cohort_shards)
    dataset["vae_test_datasets"] = list(cohort_shards)
    dataset["stat_path"] = cohort_stats
    config["general_config"]["folders_config"]["out_dir_base"] = str(run_root)

    config_path = run_root / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer_module.main(str(config_path))

    # The driver names its own run directory from the tag and a timestamp, so it is found rather
    # than predicted. Exactly one, or the fit wrote somewhere this fixture does not know about.
    checkpoint_dirs = sorted(run_root.rglob("model_checkpoints"))
    assert len(checkpoint_dirs) == 1, checkpoint_dirs
    return checkpoint_dirs[0].parent


@pytest.fixture(scope="session")
def cohort_overrides(cohort_shards, cohort_stats, tmp_path_factory) -> Path:
    """The committed evaluation delta with its two placeholder leaves repointed.

    Exactly the edit an operator makes to that file before a real run, rather than a bespoke
    overrides file: a delta carrying only the shard paths would **replace** the committed one, and
    with it the clinical ``load_fields`` every cohort-aware readout is asked in -- the loader skips
    a field it was not asked for, silently.

    Session-scoped and written once; treat as read-only.
    """
    import yaml

    from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides

    overrides = load_eval_overrides()
    overrides["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    overrides["dataset_config"]["stat_path"] = cohort_stats
    path = tmp_path_factory.mktemp("eval_overrides") / "eval_overrides_repointed.yaml"
    path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def collected_run(cohort_run, cohort_overrides, tmp_path_factory) -> Dict[str, Any]:
    r"""One real collection pass over the generated cohort shards; every artifact it left behind.

    Marked ``slow`` at every consumer rather than here -- a fixture carries no marker of its own --
    and nothing in the fast subset may depend on it. Session-scoped because the pass decodes four
    branches over every anchor of every generated segment and several suites ask about the *same*
    run; collecting once is what keeps that cost paid once.

    Two Monte Carlo draws rather than the shipped eight: these tests are about the plumbing, and
    each draw decodes every branch over every anchor.

    ``main`` returns the process **exit code**, not a path: an analysis failing must be visible to
    a shell. The results directory is therefore assembled from the directory this fixture named,
    which is what a caller with an explicit ``--output-dir`` does anyway.
    """
    import json

    from teb_vae.lag_attn_cfs.eval import run as run_module

    output_dir = tmp_path_factory.mktemp("eval")
    checkpoint = sorted((Path(cohort_run) / "model_checkpoints").glob("*.ckpt"))[0]
    exit_code = run_module.main(
        checkpoint,
        output_dir,
        overrides=cohort_overrides,
        device="cpu",
        num_samples=2,
    )
    results_dir = Path(output_dir) / run_module.RESULTS_DIRNAME
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


@pytest.fixture
def config() -> Dict[str, Any]:
    """A fresh configuration at the shipped causal geometry (safe to mutate)."""
    return causal_config()


@pytest.fixture
def budget():
    """The resolved warm-up budget at the shipped threshold, against the committed fixture."""
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    return resolved


@pytest.fixture
def tiny_kwargs() -> Dict[str, Any]:
    """A fresh copy of the ungated tiny constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def tiny_warmup() -> Dict[str, Any]:
    """A fresh copy of the tiny kwargs carrying the tiny warm-up guard (safe to mutate)."""
    return tiny_warmup_kwargs()


@pytest.fixture
def streams():
    """Seeded ``(y_st, y_ph, u_stream)`` at the tiny geometry and the causal widths."""
    return make_streams(TINY_KWARGS)


@pytest.fixture
def stub_batch():
    """A two-sample stub batch at the tiny geometry, with the deliberate weight gap."""
    return make_stub_batch()


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


@pytest.fixture
def task():
    """Factory fixture: ``task(model_kwargs=None, hparams=None, **task_kwargs)``."""
    return make_task


def build(kwargs: Mapping[str, Any], seed: int = 0):
    """Construct the model under a fixed seed, so two builds are the same weights.

    Args:
        kwargs: Constructor keyword set.
        seed: Global seed applied immediately before construction.

    Returns:
        The model, in whatever mode ``nn.Module`` defaults to.
    """
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    torch.manual_seed(seed)
    return SeqVaeLagAttnCfs(**dict(kwargs))
