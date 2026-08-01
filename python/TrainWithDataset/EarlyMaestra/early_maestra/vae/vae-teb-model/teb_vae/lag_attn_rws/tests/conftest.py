"""Shared pytest configuration for the raw-signal lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the tiny-model fixtures the suite is built
on. Mirrors ``teb_vae/lag_attn/tests/conftest.py``, including its ``utils`` pre-import pin.

The fixtures are deliberately small. A structurally faithful model at $d_{model} = 32$ and
$T = 16$ exercises every code path a production-scale one does, in milliseconds and on a CPU.
"""
from __future__ import annotations

import copy
import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import yaml

# teb_vae/lag_attn_rws/tests/conftest.py -> parents[0]=tests, [1]=lag_attn_rws, [2]=teb_vae,
# [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a
# near-empty one at model/vae_teb_prediction/utils. On a repository-wide run another conftest
# can put the latter's parent first on ``sys.path``, shadowing the real one. Binding the
# repo-root package now -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for
# every later ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

# The perturbation fixture is shared with the sibling model: the posterior delta heads are
# zero-initialised in both, so at init every KL assertion passes vacuously, and both suites
# need the same "perturb first, then assert" escape from that trap. Importing the fixture
# binds it in this conftest's namespace, which is all pytest needs to serve it here.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401

# Tiny but structurally faithful: num_heads * d_head == d_model, d_z % num_heads == 0 (the
# posterior is head-structured), and warmup < T - horizon, so every invariant the constructor
# enforces is satisfied. Raw length is sequence_length * raw_per_step = 256.
#
# horizon_film is on, matching the shipped config: per-block FiLM is hardcoded in the net, so the
# horizon core is built with film=horizon_film and film_per_block=True, and horizon_film=false
# would fail fast at construction. Keeping it on here is what makes every contract test exercise
# the per-block-FiLM decoder the production model actually runs.
TINY_KWARGS = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    raw_per_step=16,
    warmup_period=2,
    c_y=109,
    c_u=58,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    horizon_film=True,
    dropout=0.0,
)

BATCH = 2
SEQ_LEN = int(TINY_KWARGS["sequence_length"])

# What configs/default.yaml sets, at full production geometry. Unlike the tiny set this builds
# the real thing -- 300 steps, d_z = 48, entmax attention, causal norms, the extra encoder
# dilations -- so construction-time invariants are checked against the model that actually
# trains, not a miniature of it. Forward passes stay on TINY_KWARGS for speed.
SHIPPED_KWARGS = dict(
    sequence_length=300,
    d_model=128,
    d_z=48,
    horizon=30,
    raw_per_step=16,
    warmup_period=30,
    c_y=109,
    c_u=58,
    use_up_st=True,
    max_lag=90,
    num_heads=4,
    d_head=32,
    lstm_layers=2,
    dropout=0.1,
    decoder_hidden=128,
    horizon_depth=3,
    horizon_kernel=3,
    horizon_film=True,
    horizon_embed_std=0.8,
    head_init_calibration=True,
    a_head_gain=2.0,
    encoder_extra_dilations=(8, 16),
    encoder_extra_kernel=15,
    conv_norm_groups=None,
    logvar_clamp=(-5.0, 3.0),
    mu_scale=5.0,
    delta_mu_scale=3.0,
    delta_logvar_scale=2.0,
    use_entmax=True,
    attention_grad_checkpoint=False,
    lag_bias_init="alibi_decay",
    query_uses_logvar=False,
    causal_norm=True,
    coverage_floor=0.9,
)

#: The decimated step index of the deliberate gap every stub batch carries. Chosen inside the
#: tiny trained-anchor range [warmup, T - H) = [2, 12), so the gap is visible to every mask.
STUB_GAP_STEP = 10


def absolutize_dataset_paths(config: dict) -> dict:
    """Rewrite the tiny config's shard and statistics paths to absolute, in place.

    The shipped paths are repo-root-relative because the entry points run from the repo root;
    a test that drives the loader from pytest's working directory needs them absolute. Shared
    rather than repeated per test file: a renamed dataset key would otherwise have to be fixed in
    every copy, and a miss surfaces as the loader's opaque "No samples match the specified
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


#: Passes over the fit half, and held-out curve points, the whole suite runs the oracle probe at.
#:
#: The sufficiency analysis is the one step of the evaluation whose cost is a *training loop*, and
#: the shipped budget is sized for a production split. Left alone, every fixture that drives a full
#: run would pay for it twice -- once at the comparison width and once at the capacity-check width
#: -- for a probe that cannot converge on twenty-four generated segments anyway.
#:
#: So the suite shrinks the budget and the analysis reports itself un-converged, which is exactly
#: what it is meant to do on a population this size. The tests that question the *fit* set their
#: own budget explicitly rather than inheriting this one.
SUITE_ORACLE_EPOCHS = 4
SUITE_ORACLE_CURVE_POINTS = 4


@pytest.fixture(scope="session", autouse=True)
def suite_oracle_budget():
    """Shrink the oracle probe's fit budget for the whole session, and put it back afterwards."""
    from teb_vae.lag_attn_rws.eval import oracle

    original = (oracle.DEFAULT_FIT_EPOCHS, oracle.DEFAULT_CURVE_POINTS)
    oracle.DEFAULT_FIT_EPOCHS = SUITE_ORACLE_EPOCHS
    oracle.DEFAULT_CURVE_POINTS = SUITE_ORACLE_CURVE_POINTS
    yield
    oracle.DEFAULT_FIT_EPOCHS, oracle.DEFAULT_CURVE_POINTS = original


#: The shipped page counts are sized for a reviewer looking at a production split: ten stratified
#: pages and ten per extreme tail, over three metrics, is seventy renders of a seven-row figure.
#: Every fixture that drives a full run pays for all of them, and a page is a *picture* -- no
#: number in any table depends on how many were drawn, so the suite gains nothing from the volume
#: and the gate roughly doubles because of it.
#:
#: So the suite renders a few of each. The tests that question the page *selection* -- the
#: stratification, the extremes, the disjointness of the two tails -- pass their own counts
#: explicitly rather than inheriting these.
SUITE_STRATIFIED_PAGES = 2
SUITE_EXTREME_PAGES_PER_TAIL = 1


@pytest.fixture(scope="session", autouse=True)
def suite_page_budget():
    """Shrink the diagnostic-page counts for the whole session, and put them back afterwards."""
    from teb_vae.lag_attn_rws.eval.analyses import samples

    original = (samples.DEFAULT_STRATIFIED_PAGES, samples.EXTREME_PAGES_PER_TAIL)
    samples.DEFAULT_STRATIFIED_PAGES = SUITE_STRATIFIED_PAGES
    samples.EXTREME_PAGES_PER_TAIL = SUITE_EXTREME_PAGES_PER_TAIL
    yield
    samples.DEFAULT_STRATIFIED_PAGES, samples.EXTREME_PAGES_PER_TAIL = original


def make_stub_batch(batch_size: int = BATCH, seq_len: int = SEQ_LEN, seed: int = 0):
    """Build a batch exposing the fields the task reads, raw target included.

    A ``SimpleNamespace`` rather than the real ``AttributeDict``: the task reads batch fields
    as attributes, and standing up an HDF5 loader to test a loss would couple every task test
    to the data layer. The real batch contract is asserted against the committed shard in
    ``test_data_contract.py``.

    The ``weight`` carries a deliberate gap at :data:`STUB_GAP_STEP`. That gap is load-bearing:
    a uniformly valid weight would leave every mask test green whether or not the masks work.

    Args:
        batch_size: Samples in the batch. Must be at least 2 to be derangeable.
        seq_len: Decimated sequence length; the raw signals are ``16 * seq_len`` long.
        seed: Seed, so a batch is reproducible.

    Returns:
        An object with ``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``, ``fhr``, ``up`` and
        ``weight``.
    """
    generator = torch.Generator().manual_seed(seed)
    weight = torch.ones(batch_size, seq_len)
    weight[:, STUB_GAP_STEP] = 0.0
    return types.SimpleNamespace(
        fhr_st=torch.randn(batch_size, seq_len, 43, generator=generator),
        fhr_ph=torch.randn(batch_size, seq_len, 66, generator=generator),
        up_st=torch.randn(batch_size, seq_len, 43, generator=generator),
        up_ph=torch.randn(batch_size, seq_len, 15, generator=generator),
        fhr=torch.randn(batch_size, 16 * seq_len, generator=generator),
        up=torch.randn(batch_size, 16 * seq_len, generator=generator),
        weight=weight,
    )


@pytest.fixture
def tiny_kwargs() -> dict:
    """A fresh copy of the tiny-model constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def shipped_kwargs() -> dict:
    """A fresh copy of the production constructor kwargs (safe to mutate)."""
    return dict(SHIPPED_KWARGS)


@pytest.fixture
def stub_batch():
    """A two-sample stub batch with the deliberate weight gap."""
    return make_stub_batch()


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


# The loss hyperparameters the shipped config sets, as the task's constructor takes them.
# `beta_schedule=None` means the constant `kld_beta` applies, which keeps beta out of the way of
# tests that are not about the schedule. `free_bits` is genuinely 0.0 in the shipped config;
# tests about the raw/train KL split override it per-test.
TASK_HPARAMS = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


def _make_task(model_kwargs: dict | None = None, hparams: dict | None = None, **task_kwargs):
    """Build a model wrapped in its task, with the production loss hparams applied.

    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to ``TINY_KWARGS``.
        hparams: Loss hparam overrides on top of ``TASK_HPARAMS``.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnRwsTask`` with ``setup()`` already called, so the permutation
        generator exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**kwargs)
    task = SeqVaeLagAttnRwsTask(
        model,
        lr=1e-3,
        model_kwargs=kwargs,
        **dict(TASK_HPARAMS, **(hparams or {})),
        **task_kwargs,
    )
    task.setup("fit")  # seeds the permutation generator; Lightning would call this itself
    return task


@pytest.fixture
def task():
    """Factory fixture: ``task(model_kwargs=None, hparams=None, **task_kwargs)``."""
    return _make_task


# ---------------------------------------------------------------------------------------
# A multi-subgroup, multi-class shard set for the evaluation suite
#
# The committed ``tiny_shard.hdf5`` is one file whose ``target`` is all zeros, so every
# class-aware path self-skips against it and only the fallback branches are ever exercised. This
# generator writes a small set of shards named after real subgroups, carrying real class codes
# and the five fields the clinical questions are asked in.
#
# Written into ``tmp_path_factory``, never committed: ``test_fixtures.py`` asserts that no
# ``fixtures/`` directory exists under this module, and regenerating the sibling's committed
# binaries to add fields would perturb every number the existing suites are pinned against.
# ---------------------------------------------------------------------------------------

#: Subgroup shard -> the clinical class code it carries. Four shards over three classes: the
#: class axis needs all three codes, and the subgroup axis needs more shards than classes so the
#: two groupings cannot accidentally coincide. Every name is one of the canonical eight.
MULTI_CLASS_SUBGROUPS: Dict[str, int] = {
    "healthy_no_bg_no_cs": 1,
    "healthy_bg_cs": 1,
    "acidosis_cs": 2,
    "hie_no_cs": 3,
}

#: Recordings per shard, and segments each recording contributes. More than one of each is what
#: makes the per-GUID aggregation rule non-vacuous: a GUID with one segment aggregates to itself.
#:
#: Three recordings rather than two, because the cohort statistics need a *testable* cohort: the
#: shared rank tests exclude any group with fewer than ``stats.MIN_GROUP_SIZE = 3`` finite values,
#: so at two recordings per shard every cohort is excluded and the by-subgroup and by-class tests
#: could only ever be exercised as skips.
MULTI_CLASS_GUIDS_PER_SHARD = 3
MULTI_CLASS_SEGMENTS_PER_GUID = 2

#: On-disk decimated length, and the decimated steps ``trim_minutes: 1.0`` removes from each end
#: ($4\,\mathrm{Hz} \times 60\,\mathrm{s} / 16$). The full stored geometry is written because the
#: raw-index geometry is only valid on the trimmed grid.
MULTI_CLASS_SEQ_LEN = 330
MULTI_CLASS_TRIM_STEPS = 15

#: Steps at each end of the *trimmed* window carrying fractional validity, and the length of the
#: deliberate all-zero gap in the middle. The fractional steps are placed inside the trimmed
#: window on purpose: at the stored edges they would be trimmed away and the class-recovery test
#: they exist for would pass on uniformly valid data.
MULTI_CLASS_FRACTIONAL_STEPS = 4
MULTI_CLASS_GAP_STEPS = 2

#: The fractional validity itself. At code 2 this stores ``target = 1.0``, which is exactly what
#: a fully valid healthy step stores -- the case that makes reading ``target`` directly wrong.
MULTI_CLASS_EDGE_WEIGHT = 0.5

# ---------------------------------------------------------------------------------------
# The forecastable variant
#
# The plain shards are white noise: `fhr` is 140 +- 10 bpm of independent draws and the feature
# channels are independent of it, so nothing about the future is predictable from the past and
# the best possible forecast IS climatology. That is fine for testing plumbing and fatal for
# testing skill -- any "the model beats climatology" criterion is unreachable against data whose
# optimum is climatology, and a fit against it can only overfit.
#
# `forecastable=True` puts a signal in and tells the model where it is: a slowly drifting level
# carries most of the variance of `fhr`, and channel 0 of `fhr_st` -- the one channel the loader
# does NOT log-transform -- carries an affine encoding of that level at every step. A model that
# learns to read it forecasts far better than any of the three trivial baselines; one that does
# not, cannot. The exact affine constants below do not matter, and are deliberately not tuned to
# the statistics file: what the model has to learn is the map, not the offset.
# ---------------------------------------------------------------------------------------

#: Spread of the per-segment starting level and of its drift across the segment, in bpm, and the
#: observation noise around it. The level dominates: a predictor that knows it beats one that does
#: not by roughly the ratio of the two variances, which is what makes the skill margin wide enough
#: to assert on rather than a coin flip.
FORECASTABLE_LEVEL_BPM = 15.0
FORECASTABLE_DRIFT_BPM = 10.0
FORECASTABLE_NOISE_BPM = 2.0

#: The scale the level is divided by before it is written into the feature channel, so the encoded
#: value is order one rather than order ten.
FORECASTABLE_FEATURE_SCALE = 12.0

# ---------------------------------------------------------------------------------------
# The event variant
#
# `with_events=True` injects contractions into `up` and a deceleration after each one into `fhr`,
# at indices this module can state exactly, so the detectors have a known answer rather than a
# plausible one. Three things about it are deliberate:
#
#  * The shapes are piecewise **linear** ramps rather than Gaussians. A ramp's onset is a single
#    index rather than a tail asymptoting into noise, so "the recovered onset is within N seconds
#    of the injected one" is a statement about the detector rather than about where one chooses to
#    call a Gaussian's foot.
#  * The noise is small. Both detectors threshold on prominence, and a fixture whose events only
#    just clear the threshold tests the noise draw rather than the detector.
#  * `fhr` is written as **0.0 bpm inside the weight gap**, which is what the real pipeline stores
#    for a missing sample. After z-scoring that is roughly $-11\sigma$ -- the deepest
#    "deceleration" in the recording -- so a detector masking by value rather than by `weight`
#    finds it. One injected deceleration is placed on the gap for exactly that reason: it must not
#    be recovered.
# ---------------------------------------------------------------------------------------

#: Deceleration nadirs, in seconds on the **trimmed** grid the model sees. The third sits on the
#: deliberate weight gap (600.0-608.0 s), so it is the one that must not be recovered.
EVENT_NADIRS_S = (120.0, 360.0, 604.0, 840.0, 1080.0)

#: Seconds from a contraction onset to the deceleration nadir it triggers, and the ramp times of
#: the contraction itself. The onset is what the detectors' walk-back has to find.
EVENT_RESPONSE_LAG_S = 60.0
EVENT_CONTRACTION_RISE_S = 40.0
EVENT_CONTRACTION_FALL_S = 40.0

#: Amplitudes, in each signal's own stored units, and the noise the events sit in.
EVENT_CONTRACTION_AMPLITUDE = 25.0
EVENT_DECELERATION_DEPTH_BPM = 25.0
EVENT_DECELERATION_HALF_WIDTH_S = 15.0
EVENT_UP_NOISE = 1.0
EVENT_FHR_NOISE_BPM = 2.0


def injected_event_indices(*, fs: float = 4.0) -> Dict[str, np.ndarray]:
    """Return the injected event indices on the trimmed raw grid.

    Args:
        fs: Raw sampling rate, in Hz.

    Returns:
        ``contraction_onset``, ``contraction_peak`` and ``deceleration_nadir`` as ``int64``
        indices into the trimmed raw trace the loader yields -- which is the frame every detector
        assertion is made in, and is offset from the stored trace by
        :data:`MULTI_CLASS_TRIM_STEPS` decimated steps.
    """
    nadir = np.round(np.asarray(EVENT_NADIRS_S, dtype=np.float64) * fs).astype(np.int64)
    onset = nadir - int(round(EVENT_RESPONSE_LAG_S * fs))
    peak = onset + int(round(EVENT_CONTRACTION_RISE_S * fs))
    return {
        "contraction_onset": onset,
        "contraction_peak": peak,
        "deceleration_nadir": nadir,
    }


def _triangle(length: int, centre: int, rise: int, fall: int) -> np.ndarray:
    """A piecewise-linear bump: zero, up over ``rise``, down over ``fall``, zero.

    Args:
        length: Samples in the output.
        centre: Index of the apex.
        rise: Samples the leading edge takes to reach the apex.
        fall: Samples the trailing edge takes to return to zero.

    Returns:
        The unit-amplitude bump $(length,)$.
    """
    positions = np.arange(length, dtype=np.float64)
    up = 1.0 - (centre - positions) / float(max(rise, 1))
    down = 1.0 - (positions - centre) / float(max(fall, 1))
    return np.clip(np.minimum(up, down), 0.0, 1.0)


def inject_events(
    fhr: np.ndarray, up: np.ndarray, *, trim_raw: int, fs: float = 4.0
) -> None:
    """Add the injected contractions and decelerations to a shard's raw signals, in place.

    Args:
        fhr: Stored raw FHR $(n, L)$ in bpm.
        up: Stored raw UP $(n, L)$.
        trim_raw: Raw samples the loader trims from the **start**, which is what maps an index on
            the trimmed grid the assertions use onto the stored grid written here.
        fs: Raw sampling rate, in Hz.
    """
    indices = injected_event_indices(fs=fs)
    length = int(fhr.shape[1])
    rise = int(round(EVENT_CONTRACTION_RISE_S * fs))
    fall = int(round(EVENT_CONTRACTION_FALL_S * fs))
    half = int(round(EVENT_DECELERATION_HALF_WIDTH_S * fs))
    for peak in indices["contraction_peak"]:
        up += EVENT_CONTRACTION_AMPLITUDE * _triangle(
            length, int(peak) + trim_raw, rise, fall
        )[None, :]
    for nadir in indices["deceleration_nadir"]:
        fhr -= EVENT_DECELERATION_DEPTH_BPM * _triangle(
            length, int(nadir) + trim_raw, half, half
        )[None, :]


def subgroup_labels(subgroup: str) -> tuple:
    """Return ``(cs_label, bg_label)`` for a canonical subgroup name.

    Substring tests are the trap here, and the obvious ones are both wrong:
    ``'healthy_no_bg_no_cs'.endswith('_cs')`` is ``True`` and ``'_bg_' in
    'healthy_no_bg_no_cs'`` is ``True``, so a fixture built on them labels the doubly negative
    subgroup positive on both axes -- and every by-label table then has one group.

    Args:
        subgroup: One of the canonical subgroup stems.

    Returns:
        The two labels, as the ``0``/``1`` codes the shard stores.
    """
    cs = 0 if subgroup.endswith("_no_cs") else 1
    bg = 1 if ("_bg_" in subgroup and "_no_bg_" not in subgroup) else 0
    return cs, bg


def forecastable_level(rng, n_samples: int, seq_len: int) -> np.ndarray:
    """Draw one slowly drifting level per sample, on the stored decimated grid.

    Args:
        rng: The generator, so a shard set is reproducible from its seed.
        n_samples: Samples in the shard.
        seq_len: Stored decimated length.

    Returns:
        The level in bpm, $(n, T)$: a per-sample offset plus a linear drift across the segment.
        Linear rather than constant on purpose -- a constant level makes the segment mean an
        exact oracle, and a baseline that cannot be beaten is a baseline that measures nothing.
    """
    ramp = np.linspace(0.0, 1.0, seq_len)[None, :]
    offset = rng.uniform(-FORECASTABLE_LEVEL_BPM, FORECASTABLE_LEVEL_BPM, size=(n_samples, 1))
    drift = rng.uniform(-FORECASTABLE_DRIFT_BPM, FORECASTABLE_DRIFT_BPM, size=(n_samples, 1))
    return offset + drift * ramp


def write_multi_class_shards(
    directory: Path,
    *,
    seed: int = 11,
    forecastable: bool = False,
    with_events: bool = False,
    guid_prefix: str = "",
) -> List[str]:
    """Write one shard per entry of :data:`MULTI_CLASS_SUBGROUPS`, at the stored geometry.

    Mirrors what ``create_new_pipeline.py`` writes -- the same field names, channel counts,
    on-disk lengths and per-channel ``sel_*`` provenance -- so the shards load through the real
    ``GraphDataModule`` rather than through a stub, and the input channel map they describe is
    the production one. ``fhr_up_ph`` is deliberately not written: this model never loads it.

    The selection attributes are the **measured** production ones, taken from the sibling
    package's pinned record of what the real selector returns at the production geometry rather
    than synthesised here. That is what makes the channel-map assertions exact: the fourteen
    order-1 filters no selected pair references, and therefore the fourteen scattering channels
    with no recoverable centre frequency, are a property of the real selection and not of a
    fixture's arithmetic.

    Args:
        directory: Destination directory, created if absent.
        seed: Seed, so the fixture is reproducible.
        forecastable: Give ``fhr`` a slowly drifting level and encode that level in the one
            feature channel the loader does not log-transform, so a model can learn to forecast
            it. Off by default: the plain shards are white noise, which is what every plumbing
            test wants and what no skill test can use.
        with_events: Inject the contractions and decelerations of :func:`inject_events` at known
            indices, quieten the noise so they clear the detectors' prominence thresholds by a
            margin, and write ``fhr`` as $0.0$ bpm inside the weight gap exactly as the real
            pipeline stores a missing sample. Off by default, and mutually independent of
            ``forecastable``: the plain shards stay bit-identical to what every existing suite is
            pinned against.
        guid_prefix: Prepended to every recording identifier. Two shard sets drawn from different
            seeds are different recordings, and this is what makes that legible rather than
            leaving two disjoint populations sharing one set of names.

    Returns:
        The written shard paths, in :data:`MULTI_CLASS_SUBGROUPS` order.
    """
    import h5py

    from teb_vae.lag_attn.eval.tests import real_selection

    channels = {"fhr_st": 43, "fhr_ph": 66, "up_st": 43, "up_ph": 15}
    log_fields = ("fhr_st", "up_st")
    per_shard = MULTI_CLASS_GUIDS_PER_SHARD * MULTI_CLASS_SEGMENTS_PER_GUID
    seq_len = MULTI_CLASS_SEQ_LEN
    signal_len = seq_len * 16
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # One weight profile shared by every sample: fractional at both ends of the trimmed window,
    # zero in a short middle gap, fully valid elsewhere.
    weight_row = np.ones(seq_len, dtype="f4")
    low = MULTI_CLASS_TRIM_STEPS
    high = seq_len - MULTI_CLASS_TRIM_STEPS
    weight_row[low : low + MULTI_CLASS_FRACTIONAL_STEPS] = MULTI_CLASS_EDGE_WEIGHT
    weight_row[high - MULTI_CLASS_FRACTIONAL_STEPS : high] = MULTI_CLASS_EDGE_WEIGHT
    middle = seq_len // 2
    weight_row[middle : middle + MULTI_CLASS_GAP_STEPS] = 0.0

    sample_counter = 0
    for offset, (subgroup, code) in enumerate(MULTI_CLASS_SUBGROUPS.items()):
        rng = np.random.default_rng(seed + offset)
        path = directory / f"{subgroup}.hdf5"
        weight = np.tile(weight_row, (per_shard, 1))

        guids = [
            f"{guid_prefix}{subgroup.upper()}_{index // MULTI_CLASS_SEGMENTS_PER_GUID:03d}"
            for index in range(per_shard)
        ]
        # Distinct across the whole set and spanning several hours before delivery, so a
        # trajectory binned by hour has more than one bin to put things in. `epoch` is negative
        # seconds and the dataset floor is -44640, so every value stays inside the shipped
        # `epoch_min: -48000` filter.
        epochs = np.array(
            [-36000.0 + 1200.0 * (sample_counter + index) for index in range(per_shard)],
            dtype="f4",
        )
        sample_counter += per_shard
        # NaN where the recording is absent from the labour-onset table -- preserved, never
        # dropped, so the analyses that read it have to handle it.
        onset = epochs + 5400.0
        if subgroup == "hie_no_cs":
            onset[MULTI_CLASS_SEGMENTS_PER_GUID:] = np.nan

        # The drifting level, on the decimated grid, and the raw signal built around it. Off by
        # default: `level` is then identically zero and `fhr` is the white noise every plumbing
        # test has always seen.
        level = (
            forecastable_level(rng, per_shard, seq_len)
            if forecastable
            else np.zeros((per_shard, seq_len))
        )
        noise_bpm = 10.0
        if forecastable:
            noise_bpm = FORECASTABLE_NOISE_BPM
        elif with_events:
            noise_bpm = EVENT_FHR_NOISE_BPM
        fhr = (
            140.0
            + np.repeat(level, 16, axis=1)
            + noise_bpm * rng.standard_normal((per_shard, signal_len))
        )
        up_noise = EVENT_UP_NOISE if with_events else 10.0
        up = 30.0 + up_noise * rng.standard_normal((per_shard, signal_len))
        if with_events:
            inject_events(fhr, up, trim_raw=MULTI_CLASS_TRIM_STEPS * 16)
            # What the real pipeline stores for a missing sample: 0.0 bpm, which after z-scoring
            # is roughly -11 sigma. It is not a sentinel a detector can recognise by value, which
            # is the whole reason validity travels separately as `weight`.
            fhr[:, np.repeat(weight_row, 16) <= 0.0] = 0.0

        with h5py.File(str(path), "w", libver="latest") as handle:
            handle.create_dataset("fhr", data=fhr.astype("f4"))
            handle.create_dataset("up", data=up.astype("f4"))
            for field, width in channels.items():
                values = rng.standard_normal((per_shard, width, seq_len))
                if field in log_fields:
                    values = np.abs(values) + 0.1
                if forecastable and field == "fhr_st":
                    # Channel 0 is the one the loader leaves untransformed (the others are
                    # log-normalised, which a signed level would not survive), so it is the only
                    # place an affine encoding of the level arrives at the model intact.
                    values[:, 0, :] = 1.0 + level / FORECASTABLE_FEATURE_SCALE
                node = handle.create_dataset(field, data=values.astype("f4"))
                # Per-channel provenance, on the two phase blocks the writer stamps it onto. The
                # channel map is read off these rather than re-derived, so a shard without them
                # is what a recorded skip looks like -- and one with them is what the exact
                # channel counts are asserted against.
                if field.endswith("_ph"):
                    for key, value in real_selection.selection_attrs(field).items():
                        node.attrs[key] = value

            handle.create_dataset("weight", data=weight)
            # The class code scaled by validity, exactly as the real pipeline stores it.
            handle.create_dataset("target", data=(float(code) * weight).astype("f4"))
            handle.create_dataset("epoch", data=epochs)
            handle.create_dataset("time_from_labor_onset", data=onset.astype("f4"))
            cs_label, bg_label = subgroup_labels(subgroup)
            handle.create_dataset("cs_label", data=np.full((per_shard,), cs_label, dtype="u1"))
            handle.create_dataset("bg_label", data=np.full((per_shard,), bg_label, dtype="u1"))
            handle.create_dataset(
                "guid", data=guids, dtype=h5py.string_dtype(encoding="utf-8")
            )
        written.append(str(path))
    return written


@pytest.fixture(scope="session")
def multi_class_shards(tmp_path_factory) -> List[str]:
    """Paths to the generated multi-subgroup shards. Session-scoped; treat as read-only."""
    return write_multi_class_shards(tmp_path_factory.mktemp("multi_class"))


@pytest.fixture(scope="session")
def event_shards(tmp_path_factory) -> List[str]:
    """Shards carrying the injected contractions and decelerations. Treat as read-only."""
    return write_multi_class_shards(
        tmp_path_factory.mktemp("events"), seed=31, with_events=True, guid_prefix="EVENT_"
    )


@pytest.fixture(scope="session")
def event_overrides(event_shards, tmp_path_factory) -> Path:
    """The committed override delta, repointed at the event shards and asking for retention.

    Two caps rather than none, and both are what the event analyses are gated on: ``waveforms``
    because a detector needs an actual forecast block and the ordinary run retains nothing, and
    ``pages`` because the per-sample pages draw one figure per retained row.

    ``pages`` is the shard count exactly, so the stratified draw's coverage guarantee is exercised
    at the boundary where it has to hold. ``waveforms`` is eight because the triggered average's
    own guard needs twenty contractions and the fixture injects four usable ones per segment: at
    four segments the guard fires and the readout could only ever be tested as a skip.
    """
    import yaml as _yaml

    path = write_repointed_overrides(tmp_path_factory.mktemp("event_overrides"), event_shards)
    overrides = _yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides["eval_config"]["caps"] = {"waveforms": 8, "pages": len(event_shards)}
    # Enough for an interval to exist; the numbers this fixture is asserted on are counts.
    overrides["eval_config"]["bootstrap_resamples"] = 200
    path.write_text(_yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def event_evaluated(trained_run, event_overrides, tmp_path_factory) -> Dict[str, Any]:
    """One real evaluation against the event shards, with the waveform and page caps set."""
    import json

    from teb_vae.lag_attn_rws.eval import run as run_module

    output_dir = tmp_path_factory.mktemp("event_eval")
    exit_code = run_module.main(
        trained_run, output_dir, overrides=event_overrides, device="cpu", num_samples=2
    )
    results_dir = Path(output_dir) / run_module.RESULTS_DIRNAME
    return {
        "exit_code": exit_code,
        "summary": json.loads(
            (results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8")
        ),
        "results_dir": results_dir,
    }


@pytest.fixture(scope="session")
def event_loader(event_shards, multi_class_config):
    """A real test dataloader over the event shards, **un-normalised**.

    The deceleration detector works in bpm, so the loader that feeds a detector test must not
    z-score ``fhr``: the assertions are about a $25\\,$bpm dip against a $10\\,$bpm threshold, and
    the conversion back is a separate, separately tested step. ``up`` is left un-normalised for
    the same reason, though the contraction detector is $\\sigma$-relative and does not care.
    """
    import copy as _copy

    from train.data_module import GraphDataModule

    config = _copy.deepcopy(multi_class_config)
    config["dataset_config"]["vae_test_datasets"] = list(event_shards)
    config["dataset_config"]["dataloader_config"]["normalize_fields"] = [
        field
        for field in config["dataset_config"]["dataloader_config"]["normalize_fields"]
        if field not in ("fhr", "up")
    ]
    return GraphDataModule(config).test_dataloader()


def write_repointed_overrides(directory: Path, shards: List[str]) -> Path:
    """Write the committed override delta with its placeholder shard paths replaced.

    Repointing the placeholders is exactly what an operator does before a real run, so a test
    driving the pipeline end to end should do the same thing rather than assemble a delta of its
    own: the committed file stays load-bearing, and a key added to it reaches the run under test
    without anything here being updated.

    Args:
        directory: Where to write the repointed delta.
        shards: The shard paths to evaluate.

    Returns:
        Path to the written delta.
    """
    import yaml

    from teb_vae.lag_attn_rws.eval.config_schema import load_eval_overrides

    overrides = load_eval_overrides()
    overrides["dataset_config"]["vae_test_datasets"] = list(shards)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_overrides_repointed.yaml"
    path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def repointed_overrides(multi_class_shards, tmp_path_factory) -> Path:
    """The committed override delta, repointed at the generated shards. Treat as read-only."""
    return write_repointed_overrides(
        tmp_path_factory.mktemp("eval_overrides"), multi_class_shards
    )


@pytest.fixture(scope="session")
def multi_class_config(multi_class_shards) -> Dict[str, Any]:
    """The tiny training config with the real evaluation override delta merged over it.

    Built exactly as a run is: the resolved config first, then
    :func:`~teb_vae.lag_attn_rws.eval.config_schema.merge_eval_overrides`, then the shard repoint
    that stands in for editing the REPOINT_ME placeholders. So the committed overrides file is
    load-bearing in this suite rather than merely parsed by one test.

    The committed ``tiny_stats.hdf5`` is reused: the shards are written at the same geometry with
    the same field widths, and a stats file describes the *channel layout*, not the recordings.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_rws.eval.config_schema import (
        force_single_process_loader,
        merge_eval_overrides,
        validate_eval_config,
    )

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"
    config = merge_eval_overrides(absolutize_dataset_paths(load_config(str(tiny))))
    config = copy.deepcopy(config)
    config["dataset_config"]["vae_test_datasets"] = list(multi_class_shards)
    # Four samples per shard: enough batches that a per-batch bug cannot hide in a single pass.
    config["general_config"]["batch_size"]["test"] = 4
    force_single_process_loader(config)
    config["eval_config"] = validate_eval_config(config)
    return config


@pytest.fixture(scope="session")
def multi_class_loader(multi_class_config):
    """A real test dataloader over the generated multi-subgroup, multi-class shards."""
    from train.data_module import GraphDataModule

    return GraphDataModule(multi_class_config).test_dataloader()


# ---------------------------------------------------------------------------------------
# One evaluated checkpoint, shared by every test file that asks a question about a real run
#
# Session-scoped rather than per file: the fit is skipped, but the evaluation itself decodes four
# branches over 300 anchors at K Monte Carlo draws for every sample in the generated shards, and
# the run-level, preflight-level and (later) analysis-level suites all ask about the same run.
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def trained_run(tmp_path_factory) -> Path:
    """A checkpoint written into a run-shaped directory, with its resolved config beside it.

    Mirrors what the training entry point leaves behind -- ``model_checkpoints/`` holding the blob
    and the resolved config -- without spending a fit to produce it.

    The posterior perturbation is load-bearing rather than cosmetic. The delta heads are
    zero-initialised, so an unperturbed checkpoint is indistinguishable *in weight space* from one
    that never loaded, and every KL-shaped assertion would hold on a model whose weights were
    discarded.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME, LagAttnRwsTrainer

    run_dir = tmp_path_factory.mktemp("run")
    checkpoint_dir = run_dir / "model_checkpoints"
    checkpoint_dir.mkdir()

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    driver = LagAttnRwsTrainer(config_file_path=str(config_path))
    model_kwargs = driver._build_model_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**model_kwargs)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)

    task = SeqVaeLagAttnRwsTask(
        model, lr=1e-3, model_kwargs=model_kwargs,
        **dict(TASK_HPARAMS, likelihood=config["model_config"]["VAE_model"]["likelihood"]),
    )
    blob = {"state_dict": task.state_dict(), "epoch": 0, "global_step": 0,
            "hyper_parameters": dict(task.hparams)}
    task.on_save_checkpoint(blob)
    torch.save(blob, checkpoint_dir / "lag-attn-rws-epoch=00.ckpt")

    (checkpoint_dir / RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_dir / "lag-attn-rws-epoch=00.ckpt"


@pytest.fixture(scope="session")
def evaluated(trained_run, repointed_overrides, tmp_path_factory) -> Dict[str, Any]:
    """One real evaluation run; every assertion built on this fixture questions the same run.

    Driven through the committed override delta with its placeholder shards repointed, which is
    what an operator does before a real run -- so the merge, the preflight guards it satisfies and
    the generated multi-class shards are all exercised by the same pass.

    Two Monte Carlo draws rather than the shipped eight: these tests are about the plumbing, and
    each draw decodes every branch over every anchor.

    ``main`` returns the process **exit code**, not the summary path: an analysis failing must be
    visible to a shell, and a run without an explicit output directory has no path to hand back
    before it has chosen one. The path is therefore assembled from the directory this fixture
    named, which is what a caller with an explicit ``--output-dir`` does anyway.
    """
    import json

    from teb_vae.lag_attn_rws.eval import run as run_module

    output_dir = tmp_path_factory.mktemp("eval")
    exit_code = run_module.main(
        trained_run,
        output_dir,
        overrides=repointed_overrides,
        device="cpu",
        num_samples=2,
    )
    results_dir = Path(output_dir) / run_module.RESULTS_DIRNAME
    summary_path = results_dir / run_module.SUMMARY_FILENAME
    text = summary_path.read_text(encoding="utf-8")
    return {
        "exit_code": exit_code,
        "summary_path": summary_path,
        "text": text,
        "summary": json.loads(text),
        "results_dir": results_dir,
    }


# ---------------------------------------------------------------------------------------
# A checkpoint that actually forecasts
#
# `trained_run` is a *perturbed random init*, not a model that learned, so every acceptance
# criterion of the form "better than climatology" is unreachable against it -- and would be
# unreachable against a fit on the plain shards too, whose optimum IS climatology. So this fixture
# pairs the forecastable shards with a real, if short, optimisation: a few hundred Adam steps at
# `gaussian_nll`, which is the likelihood the whole calibration path is gated on and which the
# shipped `tiny.yaml` deliberately does not use.
#
# It is fitted on one draw of the forecastable generator and evaluated on another. Held out
# matters here rather than being a formality: the fit population is a few dozen segments against
# far more parameters than segments, so the model memorises, and a skill score read off the data
# it was fitted to measures that instead. Stated as a ratio rather than as two counts, because
# both move -- the segment count with `MULTI_CLASS_GUIDS_PER_SHARD`, the parameter count with
# `tiny.yaml` -- and a written-down number goes stale silently.
# ---------------------------------------------------------------------------------------

#: Optimizer steps, the learning rate they run at, and the batch they run over. Enough that the
#: model learns to read the encoded level and clears every baseline by a wide margin; small enough
#: that the fixture costs a few minutes on a CPU. Every test that reaches it is `slow`.
FITTED_STEPS = 160
FITTED_LR = 1e-3
FITTED_BATCH_SIZE = 4


@pytest.fixture(scope="session")
def forecastable_shards(tmp_path_factory) -> List[str]:
    """Held-out forecastable shards -- a different draw from the ones the fit sees."""
    return write_multi_class_shards(
        tmp_path_factory.mktemp("forecastable"),
        seed=21,
        forecastable=True,
        guid_prefix="HOLDOUT_",
    )


@pytest.fixture(scope="session")
def forecastable_overrides(forecastable_shards, tmp_path_factory) -> Path:
    """The committed override delta, repointed at the held-out forecastable shards."""
    return write_repointed_overrides(
        tmp_path_factory.mktemp("forecastable_overrides"), forecastable_shards
    )


@pytest.fixture(scope="session")
def fit_shards(tmp_path_factory) -> List[str]:
    """The forecastable shards the checkpoint is optimised on. A different draw, different GUIDs.

    Separate from :func:`fitted_run` so the two draws can be compared without paying for the fit:
    "held out" is a claim about the recordings, and a test of that claim should not need a model.
    """
    return write_multi_class_shards(
        tmp_path_factory.mktemp("fit_shards"),
        seed=101,
        forecastable=True,
        guid_prefix="FIT_",
    )


@pytest.fixture(scope="session")
def fitted_run(fit_shards, tmp_path_factory) -> Path:
    """A briefly-fit checkpoint in a run-shaped directory, at ``gaussian_nll``.

    Structurally identical to what :func:`trained_run` leaves behind -- ``model_checkpoints/``
    holding the blob and the resolved config -- so the evaluation entry point cannot tell the two
    apart. The difference is inside the weights.

    The loop calls ``compute_loss_and_metrics`` directly rather than standing up a Lightning
    trainer: the fixture needs gradient steps, not callbacks, checkpointing, logging or a
    strategy, and every one of those is exercised by ``test_train_smoke.py`` already.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME, LagAttnRwsTrainer
    from train.data_module import GraphDataModule

    run_dir = tmp_path_factory.mktemp("fitted_run")
    checkpoint_dir = run_dir / "model_checkpoints"
    checkpoint_dir.mkdir()
    shards = list(fit_shards)

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    # gaussian_nll, not the tiny config's mse: the decoder's learned log-variance heads are the
    # observation model, and an mse checkpoint makes every calibration path a permanent skip.
    config["model_config"]["VAE_model"]["likelihood"] = "gaussian_nll"
    config["dataset_config"]["vae_train_datasets"] = list(shards)
    config["dataset_config"]["vae_test_datasets"] = list(shards)
    config["dataset_config"]["dataloader_config"]["num_workers"] = 0
    config["general_config"]["batch_size"] = {
        "train": FITTED_BATCH_SIZE, "test": FITTED_BATCH_SIZE
    }
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    driver = LagAttnRwsTrainer(config_file_path=str(config_path))
    model_kwargs = driver._build_model_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**model_kwargs)
    task = SeqVaeLagAttnRwsTask(
        model, lr=FITTED_LR, model_kwargs=model_kwargs,
        **dict(TASK_HPARAMS, likelihood="gaussian_nll"),
    )
    task.setup("fit")
    task.train()

    loader = GraphDataModule(config).train_dataloader()
    optimizer = torch.optim.Adam(task.parameters(), lr=FITTED_LR)
    step = 0
    while step < FITTED_STEPS:
        for index, batch in enumerate(loader):
            loss, _metrics = task.compute_loss_and_metrics(batch, index, "train")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= FITTED_STEPS:
                break
    task.eval()

    blob = {"state_dict": task.state_dict(), "epoch": 0, "global_step": step,
            "hyper_parameters": dict(task.hparams)}
    task.on_save_checkpoint(blob)
    torch.save(blob, checkpoint_dir / "lag-attn-rws-epoch=00.ckpt")
    (checkpoint_dir / RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_dir / "lag-attn-rws-epoch=00.ckpt"


@pytest.fixture(scope="session")
def fitted_evaluated(fitted_run, forecastable_overrides, tmp_path_factory) -> Dict[str, Any]:
    """One evaluation of the fitted checkpoint against the held-out forecastable shards."""
    import json

    from teb_vae.lag_attn_rws.eval import run as run_module

    output_dir = tmp_path_factory.mktemp("fitted_eval")
    exit_code = run_module.main(
        fitted_run,
        output_dir,
        overrides=forecastable_overrides,
        device="cpu",
        num_samples=2,
    )
    results_dir = Path(output_dir) / run_module.RESULTS_DIRNAME
    summary_path = results_dir / run_module.SUMMARY_FILENAME
    return {
        "exit_code": exit_code,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "results_dir": results_dir,
    }


@pytest.fixture
def inputs():
    """Seeded ``(y_st, y_ph, u_stream)`` tensors matching the tiny geometry.

    The channel counts are the dataset's, and are independent of model size: $43$ FHR
    scattering, $66$ FHR phase-harmonic, and $58$ for the concatenated UP stream
    ``[up_st(43), up_ph(15)]``. They track
    ``hdf5_dataset/new_pipeline/create_new_pipeline.py``; when its phase-harmonic selection
    changes, these move with it.
    """
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(BATCH, SEQ_LEN, 43, generator=generator)
    y_ph = torch.randn(BATCH, SEQ_LEN, 66, generator=generator)
    u_stream = torch.randn(BATCH, SEQ_LEN, 58, generator=generator)
    return y_st, y_ph, u_stream
