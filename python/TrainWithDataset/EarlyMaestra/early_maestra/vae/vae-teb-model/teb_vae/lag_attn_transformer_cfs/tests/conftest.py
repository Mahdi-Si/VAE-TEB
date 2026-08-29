r"""Shared pytest configuration for the causal-feature conv-Transformer lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` and ``hdf5_dataset.*``
imports resolve no matter which directory pytest is invoked from, and exposes the fixtures the suite
is built on. Mirrors ``teb_vae/lag_attn_transformer_rws/tests/conftest.py``, including its ``utils``
pre-import pin.

**This conftest is spliced from two siblings, and which half comes from which is not
interchangeable.**

*The constructor keyword sets are written here, at the conv-Transformer keyword schema.* That
architecture has no ``lstm_layers``, ``encoder_extra_dilations``, ``encoder_extra_kernel``,
``conv_norm_groups`` or ``causal_norm``, and it has seven encoder keys the conv-LSTM causal cell has
never heard of. Taking the causal suite's sets instead is the specific mistake worth naming, because
it fails *asymmetrically*: its ``TINY_KWARGS`` carries only one of the five absent keys, so the tiny
path would fail on that one keyword and the shipped path on four more, and every failure would name
a keyword rather than the conftest.

*The data half comes from the causal sibling.* The committed causal shard, the configuration
builder every refusal test starts from, the tiny warm-up staircase, the budget resolver, the stub
batch carrying ``guid`` and ``epoch`` -- the two fields the anchor tiling's phase is keyed on -- and
the seeded input streams at the one-sided channel widths are all imported rather than restated.
They describe the *dataset* and the *target domain*, neither of which is a property of an encoder,
and a second copy of any of them would be free to describe a boundary the data no longer has.

The two halves meet at the geometry keys :data:`SHARED_GEOMETRY_KEYS` names. The imported batch
machinery, the imported budget resolution and every anchor expectation in this suite close over
them while the models here are built from the sets below, so the splice is sound only while the two
agree. ``test_fixtures.py`` asserts that they do, which is what turns the paragraph above into a
measurement.

**The evaluation fixtures at the bottom follow the same seam.** The generated cohort shards and
their statistics file are imported from the causal sibling -- they describe the *dataset* -- while
the fit, the repointed override delta and the evaluation run are local, because each is downstream
of the model class. ``test_eval_fixtures.py`` is what proves that half.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest
import torch

# teb_vae/lag_attn_transformer_cfs/tests/conftest.py -> parents[0]=tests,
# [1]=lag_attn_transformer_cfs, [2]=teb_vae, [3]=repo root.
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

# ---------------------------------------------------------------------------------------
# The architecture half: the causality probe and its two tolerances.
#
# Every invariant in this suite is measured the same way -- resample the strict future, require
# bit-stability at the cut *and* visible movement at the end -- and the second half is the negative
# control without which a dead layer passes every causality test in the package. Importing a fixture
# binds it in this conftest's namespace, which is all pytest needs to serve it to the tests here.
# ---------------------------------------------------------------------------------------
from teb_vae.lag_attn_transformer_rws.tests.conftest import (  # noqa: E402,F401
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    assert_token_causal,
    build_stream_encoder,
    relative_change,
    resample_after,
)

# ---------------------------------------------------------------------------------------
# The target half: the committed causal shard, the config builder, the tiny warm-up staircase and
# the stub batch that carries the two phase-key fields.
#
# ``perturb_posterior`` originates in ``lag_attn``; it is taken through the causal suite because
# that is where the rest of this half comes from, and because the delta heads are zero-initialised
# in every model of the family -- so at initialisation every KL assertion in this suite would pass
# vacuously without it.
# ---------------------------------------------------------------------------------------
from teb_vae.lag_attn_cfs.tests.conftest import (  # noqa: E402,F401
    BATCH,
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_SHARD,
    CAUSAL_ST_WIDTH,
    COHORT_STATS_FILENAME,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_TRIM_MINUTES,
    SHIPPED_WARMUP_PERIOD,
    STUB_GAP_STEP,
    TASK_HPARAMS,
    TINY_HORIZON,
    TINY_SEQ_LEN,
    TINY_ALIGNED_WARMUP_PERIOD,
    TINY_SOURCE_ALIGN_DELAYS,
    TINY_SOURCE_KEEP_INDEX,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_STRIDE,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    TINY_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    absolutize_dataset_paths,
    causal_config,
    cohort_shards,
    cohort_stats,
    make_stub_batch,
    make_streams,
    perturb_posterior,
    stored_warmup,
    without_key,
    write_cohort_shards,
    write_variant,
)

#: The geometry keys the imported target half closes over. They are read off the *causal* suite's
#: sets -- by its budget resolution, its stub batch and every anchor count this suite asserts --
#: while every model here is built from the sets below, so the splice is sound only while the two
#: agree. A disagreement builds a model neither parent's suite tests: no shape would differ, because
#: $A_{\max}$ and the block width are geometry constants either way, and the numbers would simply be
#: another model's.
SHARED_GEOMETRY_KEYS = (
    "sequence_length",
    "c_y",
    "c_u",
    "horizon",
    "warmup_period",
    "raw_per_step",
    "use_up_st",
    "anchor_stride",
    "max_lag",
)

#: The five conv-LSTM-only constructor keywords. Absent from every keyword set and every config this
#: package builds: there is no recurrent branch, no extra dilation schedule and no time-pooling
#: normaliser left to causalise, so each of them would reach nothing -- and each raises ``TypeError``
#: at the constructor rather than being ignored, which is why the assertion is worth making once
#: here and reading everywhere.
CONV_LSTM_ONLY_KEYS = (
    "lstm_layers",
    "encoder_extra_dilations",
    "encoder_extra_kernel",
    "conv_norm_groups",
    "causal_norm",
)

# Tiny but structurally faithful, and larger than the conv-Transformer sibling's $16$-step fixture
# for the causal cell's reason: a tiling needs a floor, a stride and room for more than one tile,
# and at $T = 16$ with any usable floor there is only one anchor per phase -- which would leave
# every padding assertion in the suite passing without a padded slot ever existing.
#
# The encoder geometry is shrunk with one property preserved that the shipped one has and a naive
# miniature would lose: the source encoder's receptive-field bound stays strictly inside the
# sequence. Stem reach is $1 + (3-1)\cdot 1 + (3-1)\cdot 2 = 7$ steps, so the source bound is
# $7 + 2(4-1) = 13 < 24$. A bound that clamped at $T$ would make the measured-bound probe vacuous.
#
# ``horizon_film`` is on, matching the shipped config: per-block FiLM is hardcoded in the net, so
# the horizon core is built with ``film=horizon_film`` and ``film_per_block=True``, and
# ``horizon_film=false`` would fail fast at construction.
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
    horizon_film=True,
    dropout=0.0,
    encoder_conv_kernels=(3, 3),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=64,
    target_attention_blocks=2,
    source_attention_blocks=2,
    source_attention_window=4,
)

# What configs/default.yaml sets, at full production geometry: the causal window and channel widths
# -- $300$ steps, $c_y = 102$, $c_u = 51$, a one-minute horizon tiled at $S = H$ from a floor of
# $133$ -- over the conv-Transformer encoders. Construction-time invariants are checked against the
# model that actually trains, not a miniature of it; forward passes in this suite stay on
# ``TINY_KWARGS`` for speed.
#
# Written out rather than derived from the causal suite's set: the two are the two halves of the
# splice, and a set derived from the other could not disagree with it, which would make
# ``test_fixtures.py``'s agreement check a tautology.
SHIPPED_KWARGS: Dict[str, Any] = dict(
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
    dropout=0.1,
    decoder_hidden=256,
    horizon_depth=4,
    horizon_kernel=3,
    horizon_film=True,
    horizon_attention_blocks=2,
    horizon_embed_std=0.8,
    head_init_calibration=True,
    a_head_gain=2.0,
    logvar_clamp=(-5.0, 3.0),
    mu_scale=5.0,
    delta_mu_scale=3.0,
    delta_logvar_scale=2.0,
    use_entmax=True,
    attention_grad_checkpoint=False,
    lag_bias_init="alibi_decay",
    query_uses_logvar=False,
    coverage_floor=0.9,
    anchor_stride=SHIPPED_HORIZON,
    encoder_conv_kernels=(5, 9),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=512,
    target_attention_blocks=6,
    source_attention_blocks=3,
    source_attention_window=16,
)


def tiny_warmup_kwargs(
    kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """Add the tiny warm-up guard to a constructor keyword set.

    The causal suite's identical helper is deliberately not imported: it defaults to *its own*
    conv-LSTM ``TINY_KWARGS``, so a call here that omitted the first argument would silently build
    the wrong architecture's keyword set and fail at ``lstm_layers`` rather than saying so.

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


def tiny_align_kwargs(
    kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """Add the tiny warm-up guard **and** the tiny alignment to a constructor keyword set.

    The shift vectors are the causal-feature suite's own, bound above rather than rebuilt: every
    cell of this grid gates the same two stored streams at the same tiny widths, and a second
    definition of the same vectors is one that can drift. The anchor floor moves with the shift
    because the two are one decision -- a floor left at the unaligned value is refused by the
    constructor, which is itself asserted rather than worked around.

    Args:
        kwargs: The keyword set to guard. Defaults to :data:`TINY_KWARGS`.
        **overrides: Applied after the alignment, so a test can move one leaf of it.

    Returns:
        A fresh dict carrying the guard and the shift on both streams.
    """
    aligned = tiny_warmup_kwargs(
        kwargs,
        target_align_delays=TINY_TARGET_ALIGN_DELAYS,
        source_align_delays=TINY_SOURCE_ALIGN_DELAYS,
        warmup_period=TINY_ALIGNED_WARMUP_PERIOD,
    )
    aligned.update(overrides)
    return aligned


def shipped_warmup_kwargs(*, align: bool = True, **overrides: Any) -> Dict[str, Any]:
    """The production constructor call: this architecture's set plus the budget the shards resolve.

    Resolved through :func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget` against the
    committed causal fixture rather than written out, so the surviving-channel count comes from the
    data every time and a rebuilt fixture moves the model instead of failing an unrelated literal.

    The model class handed to :func:`~teb_vae.lag_attn_cfs.model_kwargs.warmup_model_kwargs` is
    **this** package's, which is what makes the "architecture cannot mask its inputs" refusal a
    check of this constructor rather than of the conv-LSTM cell's.

    The ``align`` flag names the **unaligned comparison arm**: the same geometry resolved with
    ``causal_align_reference`` removed, which is the model that shipped before the common clock and
    is what every "what did the alignment cost" assertion is stated against. It is a flag rather
    than a second builder because the two must not be allowed to drift in anything else.

    Args:
        align: ``False`` resolves the budget with no alignment reference, so the two shift
            vectors are absent and every source channel survives.
        **overrides: Applied last.

    Returns:
        Constructor kwargs. No ``decoder_out_channels``: it is not a keyword of this constructor.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
    from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs
    from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

    resolved = resolve_warmup_budget(
        causal_config() if align else causal_config(causal_align_reference=None)
    )
    assert resolved is not None
    kwargs = dict(SHIPPED_KWARGS, **warmup_model_kwargs(resolved, SeqVaeLagAttnTrfCfs))
    # Updated rather than splatted into the call above: an override naming a key the set already
    # carries -- ``horizon`` is the one every arm moves -- is a TypeError there and a replacement
    # here, which is what "applied last" has to mean to be usable at all.
    kwargs.update(overrides)
    return kwargs


def build(kwargs: Mapping[str, Any], seed: int = 0):
    """Construct the model under a fixed seed, so two builds are the same weights.

    Args:
        kwargs: Constructor keyword set.
        seed: Global seed applied immediately before construction.

    Returns:
        The model, in whatever mode ``nn.Module`` defaults to.
    """
    from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

    torch.manual_seed(seed)
    return SeqVaeLagAttnTrfCfs(**dict(kwargs))


def make_task(
    model_kwargs: Optional[Mapping[str, Any]] = None,
    hparams: Optional[Mapping[str, Any]] = None,
    **task_kwargs,
):
    """Build this model wrapped in this package's task, with the production loss hparams applied.

    This package's own task rather than either parent's: what the diamond adds is the tiling phase,
    the five-argument forward and the source-null readout on one side and the step-granular
    learning-rate ramp on the other, and every contract asserted through this factory is a contract
    of the pair a real run assembles. Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to the guarded tiny set at the tiling stride,
            because the guard is what this package is about: an ungated model has no warm-up, no
            tertiles and no source warmth to report, and a task built at stride $1$ would resolve
            the same anchor set on every stage and leave every phase assertion holding vacuously.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnTrfCfsTask`` with ``setup()`` already called, so the permutation generator
        exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask

    kwargs = dict(
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if model_kwargs is None else model_kwargs
    )
    model = build(kwargs)
    task = SeqVaeLagAttnTrfCfsTask(
        model,
        lr=1e-3,
        model_kwargs=kwargs,
        **dict(TASK_HPARAMS, **(hparams or {})),
        **task_kwargs,
    )
    task.setup("fit")  # seeds the permutation generator; Lightning would call this itself
    return task


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


# =================================================================================================
# The evaluation fixtures
#
# The splice again, and along the same seam. The **shards and their statistics file** are the causal
# cell's, imported above: eight generated subgroup shards carrying ``transform: 'causal'``, the
# per-block warm-up attributes, the ``sel_*`` provenance and the five clinical fields. They describe
# the dataset and the target domain, neither of which is a property of an encoder, and a second
# generator here would be free to describe a boundary the data no longer has.
#
# What is local is everything downstream of the model class: the fit is a conv-Transformer, and the
# repointed delta is **this package's** committed one rather than the causal cell's -- the two ship
# two files, and evaluating this model against the other one's settings is exactly the silent
# divergence the binding test's key-for-key comparison exists to catch.
#
# **What may be asserted on the run these produce.** The rule the causal suite's fixtures carry
# binds here unchanged: eight real raw segments re-used under distinct identities, scored by a tiny
# model trained for one epoch, are evidence about SCHEMA, SHAPE, FINITENESS, DENOMINATORS, COHORT
# MEMBERSHIP, COUNTS, IDENTITIES and REFUSALS and about nothing else. No test may assert the sign,
# magnitude, direction or significance of any clinical or statistical effect on them -- and in
# particular no test may compare this cell's numbers against the causal cell's and read the
# difference as an encoder finding.
# =================================================================================================
@pytest.fixture(scope="session")
def trf_cohort_run(cohort_shards, cohort_stats, tmp_path_factory) -> Path:
    r"""One real fit of this cell against the generated cohort shards; returns the run directory.

    Marked ``slow`` at every consumer rather than here -- a fixture carries no marker of its own --
    and nothing in the fast subset may depend on it.

    Driven through ``trainer.main`` rather than by assembling a checkpoint by hand, because what an
    evaluation reads out of this directory is precisely what the driver puts there and nothing else
    can: ``model_kwargs`` carrying the four warm-up tuples the budget resolved against these shards,
    ``model_class`` stamping which architecture produced the run -- which is what the cross-cell
    table keys its rows on -- and ``resolved_config.yaml`` recording the configuration behind them.
    A blob saved from a freshly constructed model would carry the same keys and would prove none of
    it.

    The base is **this package's** shipped ``tiny.yaml``, with exactly three leaves moved: both
    shard lists and the statistics path, which is the repoint; and ``out_dir_base``, because the
    shipped value is a path inside the repository and a test must not write there.
    """
    import yaml

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_transformer_cfs import trainer as trainer_module

    run_root = tmp_path_factory.mktemp("trf_cohort_run")
    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_transformer_cfs" / "configs" / "tiny.yaml"
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
def trf_cohort_overrides(cohort_shards, cohort_stats, tmp_path_factory) -> Path:
    """**This package's** committed evaluation delta with its two placeholder leaves repointed.

    Exactly the edit an operator makes to that file before a real run, rather than a bespoke
    overrides file: a delta carrying only the shard paths would **replace** the committed one, and
    with it the clinical ``load_fields`` every cohort-aware readout is asked in -- the loader skips
    a field it was not asked for, silently.

    This package's file rather than the causal cell's, even though a test asserts the two are equal
    key for key: that equality is a property this suite checks, not one it may assume, and reading
    the other cell's file here would make the check unable to fail.

    Session-scoped and written once; treat as read-only.

    Three leaves are repointed rather than two, and the third is not a placeholder. The occlusion
    bands are stated in **production lag indices** and the schema refuses a band reaching past the
    model's own ``max_lag`` -- correctly, because a band whose top the attention cannot read would
    score nothing there while its name claimed otherwise. The tiny geometry these fixtures run at
    shrinks that window, so the bands are rescaled to it rather than the refusal being loosened.
    Rescaled through the causal cell's own helper, because the two cells' deltas are asserted equal
    key for key and a second rescaling rule here could make them disagree.
    """
    import yaml

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides
    from teb_vae.lag_attn_cfs.tests.conftest import _tiny_occlusion_bands
    from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

    overrides = load_eval_overrides(TRF_CFS_BINDING.overrides_path)
    overrides["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    overrides["dataset_config"]["stat_path"] = cohort_stats
    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_transformer_cfs" / "configs" / "tiny.yaml"
    overrides["eval_config"]["occlusion_bands"] = _tiny_occlusion_bands(
        int(load_config(str(tiny))["model_config"]["VAE_model"]["max_lag"])
    )
    path = tmp_path_factory.mktemp("trf_eval_overrides") / "eval_overrides_repointed.yaml"
    path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def trf_collected_run(trf_cohort_run, trf_cohort_overrides, tmp_path_factory) -> Dict[str, Any]:
    r"""One real evaluation of this cell through the causal pipeline; every artifact it left.

    Marked ``slow`` at every consumer rather than here, and nothing in the fast subset may depend
    on it. Session-scoped because the pass decodes four branches over every anchor of every
    generated segment and several tests ask about the *same* run; collecting once is what keeps
    that cost paid once.

    Two Monte Carlo draws rather than the shipped eight: these tests are about the plumbing, and
    each draw decodes every branch over every anchor.

    ``main`` returns the process **exit code**, not a path: an analysis failing must be visible to
    a shell. The results directory is therefore assembled from the directory this fixture named,
    which is what a caller with an explicit ``--output-dir`` does anyway.
    """
    import json

    from teb_vae.lag_attn_transformer_cfs.eval import run as run_module

    output_dir = tmp_path_factory.mktemp("trf_eval")
    checkpoint = sorted((Path(trf_cohort_run) / "model_checkpoints").glob("*.ckpt"))[0]
    exit_code = run_module.main(
        checkpoint,
        output_dir,
        overrides=trf_cohort_overrides,
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
    """The resolved warm-up budget at the shipped threshold, against the committed fixture.

    Shipped means **aligned**: both streams carry a shift, the source is 47 channels wide, and
    ``reference_delay_s`` is set. :func:`unaligned_budget` is the comparison arm.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    return resolved


@pytest.fixture
def unaligned_budget():
    """The same budget with the alignment off: no shift, no reference, every source channel kept.

    The comparison arm that stays reachable at one key, and what every assertion phrased as "the
    warm-up budget alone decides this" is stated against.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    resolved = resolve_warmup_budget(causal_config(causal_align_reference=None))
    assert resolved is not None
    return resolved


@pytest.fixture
def tiny_kwargs() -> Dict[str, Any]:
    """A fresh copy of the ungated tiny constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def shipped_kwargs() -> Dict[str, Any]:
    """A fresh copy of the ungated production constructor kwargs (safe to mutate)."""
    return dict(SHIPPED_KWARGS)


@pytest.fixture
def tiny_warmup() -> Dict[str, Any]:
    """A fresh copy of the tiny kwargs carrying the tiny warm-up guard (safe to mutate)."""
    return tiny_warmup_kwargs()


@pytest.fixture
def tiny_align() -> Dict[str, Any]:
    """A fresh copy of the tiny kwargs carrying the guard and the alignment (safe to mutate)."""
    return tiny_align_kwargs()


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
