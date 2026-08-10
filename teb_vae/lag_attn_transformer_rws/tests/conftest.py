r"""Shared pytest configuration for the causal conv-Transformer lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the fixtures the suite is built on. Mirrors
``teb_vae/lag_attn_rws/tests/conftest.py``, including its ``utils`` pre-import pin.

The data fixtures are imported from the sibling suite rather than restated: ``perturb_posterior``,
``make_stub_batch``, ``absolutize_dataset_paths``, the multi-class shard writer and its event and
level generators, and the two session-wide budget shrinkers. They describe the *data* and the
*trap*, both of which are shared -- the batch contract is the same one, the shards describe the
dataset rather than either model, and the posterior delta heads are zero-initialised in both
models, so at initialisation every KL assertion passes vacuously in both. The sibling's own
conftest already establishes the convention by importing a perturbation fixture from *its*
sibling.

The constructor keyword sets are defined fresh, because they are the one thing that is not shared:
this model has no ``lstm_layers``, ``encoder_extra_dilations``, ``encoder_extra_kernel``,
``conv_norm_groups`` or ``causal_norm``, and it has seven encoder keys the sibling has never heard
of. A copy of the sibling's set would not construct.

The causality probe lives here too. Every invariant in this suite is measured the same way -- resample
the strict future, require bit-stability at the cut *and* visible movement at the end -- and the second
half is the negative control without which a dead layer passes every causality test in the package.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch

# teb_vae/lag_attn_transformer_rws/tests/conftest.py -> parents[0]=tests,
# [1]=lag_attn_transformer_rws, [2]=teb_vae, [3]=repo root.
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

# Importing the fixture binds it in this conftest's namespace, which is all pytest needs to serve
# it to the tests in this directory.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401
from teb_vae.lag_attn_rws.tests.conftest import (  # noqa: E402,F401
    BATCH,
    MULTI_CLASS_GUIDS_PER_SHARD,
    MULTI_CLASS_SEGMENTS_PER_GUID,
    MULTI_CLASS_SEQ_LEN,
    MULTI_CLASS_SUBGROUPS,
    STUB_GAP_STEP,
    absolutize_dataset_paths,
    forecastable_level,
    inject_events,
    injected_event_indices,
    make_stub_batch,
    subgroup_labels,
    suite_oracle_budget,
    suite_page_budget,
    write_multi_class_shards,
)

# Tiny but structurally faithful: ``num_heads * d_head == d_model``, ``d_z % num_heads == 0`` (the
# posterior is head-structured), ``warmup < T - horizon``, and the encoder head width
# ``d_model // encoder_num_heads`` is even, which rotary position encoding requires. Raw length is
# ``sequence_length * raw_per_step = 256``, matching the stub batch.
#
# The encoder geometry is shrunk with one property preserved that the shipped one has and a naive
# miniature would lose: the source encoder's receptive-field bound stays strictly inside the
# sequence. Stem reach is $1 + (3-1)\cdot 1 + (3-1)\cdot 2 = 7$ steps, so the source bound is
# $7 + 2(4-1) = 13 < 16$. A bound that clamped at $T$ would make the measured-bound probe vacuous.
#
# ``horizon_film`` is on, matching the shipped config: per-block FiLM is hardcoded in the net, so
# the horizon core is built with ``film=horizon_film`` and ``film_per_block=True``, and
# ``horizon_film=false`` would fail fast at construction.
SEQ_LEN = 16

TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=SEQ_LEN,
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
    encoder_conv_kernels=(3, 3),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=64,
    target_attention_blocks=2,
    source_attention_blocks=2,
    source_attention_window=4,
)

# What configs/default.yaml sets, at full production geometry. Unlike the tiny set this describes
# the real thing -- 300 steps, $d_z = 64$, entmax attention, the $(5, 9)$ stem, six full-context
# target blocks and three source blocks at a 16-step window, a $256$-wide decoder core four refine
# blocks deep with two horizon-attention blocks on top -- so construction-time invariants are
# checked against the model that actually trains, not a miniature of it. Forward passes in this
# suite stay on TINY_KWARGS for speed.
#
# Five of the sibling's keys are deliberately absent: ``lstm_layers``,
# ``encoder_extra_dilations``, ``encoder_extra_kernel``, ``conv_norm_groups`` and ``causal_norm``.
# There is no recurrent branch, no extra dilation schedule and no time-pooling normaliser left to
# causalise, so each of them would reach nothing.
SHIPPED_KWARGS: Dict[str, Any] = dict(
    sequence_length=300,
    d_model=128,
    d_z=64,
    horizon=30,
    raw_per_step=16,
    warmup_period=30,
    c_y=109,
    c_u=58,
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
    encoder_conv_kernels=(5, 9),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=512,
    target_attention_blocks=6,
    source_attention_blocks=3,
    source_attention_window=16,
)

#: Relative movement below which two float32 activations are the same computation run twice.
#: float32 round-off on $O(1)$ activations through a handful of layers.
CAUSALITY_TOL = 1e-5

#: Relative movement a probe must *exceed* to count as having reached the module under test. The
#: paired half of every causality assertion: without it a module returning zeros, or a residual
#: branch that was never connected, passes every bit-stability check in this suite.
MOVEMENT_TOL = 1e-3


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


def resample_after(x: torch.Tensor, cut: int, *, seed: int = 0) -> torch.Tensor:
    """Return a copy of ``x`` whose steps strictly after ``cut`` are drawn afresh.

    A random resample rather than a constant offset, deliberately. The input adapter upstream of
    both encoders ends in a ``LayerNorm``, which removes a uniform channel shift outright, so a
    constant-offset probe would report causality that was never tested. ``RMSNorm`` inside the
    encoder does not centre, but the probe runs through both.

    Args:
        x: Sequence-major tensor $(B, T, C)$.
        cut: Last index left untouched. Everything at ``cut + 1`` and beyond is replaced.
        seed: Seed for the replacement draw, so a probe is reproducible.

    Returns:
        A new tensor shaped like ``x``, identical up to and including ``cut``.
    """
    generator = torch.Generator().manual_seed(seed)
    perturbed = x.clone()
    tail_shape = perturbed[:, cut + 1 :].shape
    perturbed[:, cut + 1 :] = torch.randn(tail_shape, generator=generator, dtype=x.dtype)
    return perturbed


def relative_change(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Return $\\lVert a - b \\rVert / \\lVert a \\rVert$, or the absolute norm if ``a`` is zero.

    Args:
        reference: The unperturbed tensor.
        other: The tensor computed from the perturbed input.

    Returns:
        The relative movement as a Python float.
    """
    denominator = float(reference.norm())
    difference = float((reference - other).norm())
    return difference / denominator if denominator > 0.0 else difference


def assert_token_causal(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    cut: int,
    *,
    label: str = "module",
    seed: int = 0,
) -> None:
    """Assert $H_t = f(X_{\\le t})$ by a paired future-perturbation probe.

    Resamples the input strictly after ``cut`` and requires two things at once: the output at
    ``cut`` must be unmoved, and the output at the last step must have moved. The second half is
    the negative control. This architecture has no time-pooling normaliser to flip -- the trick the
    sibling's leak test uses -- so the control is positional instead, and needs no switch in
    production code that exists only for tests.

    Args:
        forward: Maps $(B, T, C)$ to a sequence-major tensor.
        x: The unperturbed input.
        cut: The cutoff $t_0$. Must leave at least one step after it.
        label: Named in the failure message, so a parametrised run says which block failed.
        seed: Seed for the resample.
    """
    assert cut < x.shape[1] - 1, f"cut {cut} leaves no future to perturb in {x.shape[1]} steps"

    reference = forward(x)
    perturbed = forward(resample_after(x, cut, seed=seed))

    at_cut = relative_change(reference[:, cut], perturbed[:, cut])
    at_end = relative_change(reference[:, -1], perturbed[:, -1])
    assert at_cut < CAUSALITY_TOL, (
        f"{label}: output at t={cut} moved by {at_cut:.3e} when only the strict future changed"
    )
    assert at_end > MOVEMENT_TOL, (
        f"{label}: output at the last step moved by only {at_end:.3e} -- the perturbation never "
        f"reached the module, so the bit-stability above proves nothing"
    )


def build_stream_encoder(stream: str, kwargs: Optional[dict] = None, **overrides):
    """Build one stream's encoder from a constructor keyword set, the way the model will.

    The seven encoder keys map onto the two encoders in exactly one way, and it is the same map in
    every structural test file, so it is written here once. The target takes the full causal prefix
    and the source a bounded window; that asymmetry is the architecture, not a setting these tests
    are free to choose.

    Args:
        stream: ``"target"`` or ``"source"``.
        kwargs: A constructor keyword set. Defaults to :data:`TINY_KWARGS`.
        **overrides: Passed through to the encoder, overriding anything derived above. Dropout
            defaults to $0$ here rather than to the keyword set's value, because every structural
            probe in this suite compares two forward passes.

    Returns:
        A ``CausalConvTransformerEncoder`` in eval mode.

    Raises:
        ValueError: If ``stream`` is neither of the two.
    """
    from teb_vae.lag_attn_transformer_rws.nets.encoders import CausalConvTransformerEncoder

    if stream not in ("target", "source"):
        raise ValueError(f"stream must be 'target' or 'source', got {stream!r}")
    source = dict(TINY_KWARGS if kwargs is None else kwargs)
    settings = dict(
        d_model=int(source["d_model"]),
        sequence_length=int(source["sequence_length"]),
        conv_kernels=source["encoder_conv_kernels"],
        conv_dilations=source["encoder_conv_dilations"],
        num_attention_blocks=int(source[f"{stream}_attention_blocks"]),
        num_heads=int(source["encoder_num_heads"]),
        d_ff=int(source["encoder_d_ff"]),
        attention_window=(
            None if stream == "target" else int(source["source_attention_window"])
        ),
        dropout=0.0,
    )
    settings.update(overrides)
    return CausalConvTransformerEncoder(**settings).eval()


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
    """A two-sample stub batch at the tiny geometry, with the deliberate weight gap."""
    return make_stub_batch(BATCH, SEQ_LEN)


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


# The loss hyperparameters the shipped config sets, as the task's constructor takes them.
# ``beta_schedule=None`` means the constant ``kld_beta`` applies, which keeps beta out of the way of
# tests that are not about the schedule. ``free_bits`` is genuinely 0.0 in the shipped config; tests
# about the raw/train KL split override it per-test.
TASK_HPARAMS: Dict[str, Any] = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


def _make_task(model_kwargs: Optional[dict] = None, hparams: Optional[dict] = None, **task_kwargs):
    """Build a model wrapped in its task, with the production loss hparams applied.

    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to :data:`TINY_KWARGS`.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnTrfRwsTask`` with ``setup()`` already called, so the permutation
        generator exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
    from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**kwargs)
    task = SeqVaeLagAttnTrfRwsTask(
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
# The evaluation fixtures
#
# The shards, the shard writer and the two budget shrinkers are the sibling's, imported above: they
# describe the dataset and the suite's own cost, neither of which is a property of either model.
# What is local is everything downstream of the model class -- the repointed delta reads THIS
# package's committed overrides, and the checkpoint is a conv-Transformer.
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def multi_class_shards(tmp_path_factory) -> List[str]:
    """One shard per canonical subgroup, three recordings each, three clinical classes."""
    return write_multi_class_shards(tmp_path_factory.mktemp("multi_class"), seed=11)


def write_repointed_overrides(directory: Path, shards: List[str]) -> Path:
    """Write **this package's** committed delta with its placeholder shard paths replaced.

    Repointing the placeholders is exactly what an operator does before a real run, so a test
    driving the pipeline end to end does the same thing rather than assembling a delta of its own:
    the committed file stays load-bearing, and a key added to it reaches the run under test
    without anything here being updated.

    Not the sibling's helper, which resolves the sibling's delta: the two packages ship two
    committed files, and evaluating this model against the other one's settings is exactly the
    silent divergence the parity test exists to catch.

    Args:
        directory: Where to write the repointed delta.
        shards: The shard paths to evaluate.

    Returns:
        Path to the written delta.
    """
    import yaml

    from teb_vae.lag_attn_rws.eval.config_schema import load_eval_overrides
    from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

    overrides = load_eval_overrides(TRF_BINDING.overrides_path)
    overrides["dataset_config"]["vae_test_datasets"] = list(shards)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_overrides_repointed.yaml"
    path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def repointed_overrides(multi_class_shards, tmp_path_factory) -> Path:
    """This package's committed delta, repointed at the generated shards. Treat as read-only."""
    return write_repointed_overrides(
        tmp_path_factory.mktemp("eval_overrides"), multi_class_shards
    )


#: Optimizer steps the throwaway checkpoint runs, and the rate they run at.
#:
#: Few, deliberately. The checkpoint's job is to be a *loadable* one: preflight verifies a load in
#: weight space by requiring the zero-initialised delta heads or FiLM generators to have moved, and
#: a handful of Adam steps moves them. It is not a fit -- the generated shards are white noise, so
#: no number of steps would make this model forecast them -- and every extra step is time every
#: evaluation test in this package pays once.
TRAINED_STEPS = 8
TRAINED_LR = 1e-3


@pytest.fixture(scope="session")
def trained_run(multi_class_shards, tmp_path_factory) -> Path:
    """A briefly-trained conv-Transformer checkpoint in a run-shaped directory.

    Mirrors what the training entry point leaves behind -- ``model_checkpoints/`` holding the blob
    and the resolved config beside it -- so the evaluation entry point reaches it exactly as it
    would a production run, through ``resolved_config_for``.

    ``gaussian_nll`` rather than the tiny config's ``mse``: the decoder's learned log-variance
    heads are the observation model, and an ``mse`` checkpoint makes every calibration path a
    permanent skip -- which would leave the smoke run asserting that an analysis skipped rather
    than that it ran.

    The training is what moves the posterior delta heads off zero. Those heads are
    zero-initialised, so an untrained checkpoint is indistinguishable *in weight space* from one
    that never loaded, and preflight would refuse it -- a fixture that fails preflight is a fixture
    that tests nothing.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
    from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer
    from train.data_module import GraphDataModule

    import yaml

    run_dir = tmp_path_factory.mktemp("trf_run")
    checkpoint_dir = run_dir / "model_checkpoints"
    checkpoint_dir.mkdir()

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    config["model_config"]["VAE_model"]["likelihood"] = "gaussian_nll"
    config["dataset_config"]["vae_train_datasets"] = list(multi_class_shards)
    config["dataset_config"]["vae_test_datasets"] = list(multi_class_shards)
    config["dataset_config"]["dataloader_config"]["num_workers"] = 0
    config["general_config"]["batch_size"] = {"train": 4, "test": 4}
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    driver = LagAttnTrfRwsTrainer(config_file_path=str(config_path))
    model_kwargs = driver._build_model_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**model_kwargs)
    task = SeqVaeLagAttnTrfRwsTask(
        model, lr=TRAINED_LR, model_kwargs=model_kwargs,
        **dict(TASK_HPARAMS, likelihood="gaussian_nll"),
    )
    task.setup("fit")
    task.train()

    loader = GraphDataModule(config).train_dataloader()
    optimizer = torch.optim.Adam(task.parameters(), lr=TRAINED_LR)
    step = 0
    while step < TRAINED_STEPS:
        for index, batch in enumerate(loader):
            loss, _metrics = task.compute_loss_and_metrics(batch, index, "train")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= TRAINED_STEPS:
                break
    task.eval()

    blob = {"state_dict": task.state_dict(), "epoch": 0, "global_step": step,
            "hyper_parameters": dict(task.hparams)}
    task.on_save_checkpoint(blob)
    checkpoint = checkpoint_dir / f"{LagAttnTrfRwsTrainer.CHECKPOINT_STEM}-epoch=00.ckpt"
    torch.save(blob, checkpoint)
    (checkpoint_dir / RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint


#: Retention for the shared evaluation fixture below. Only this model's own cap is set, and it is
#: set rather than left absent for one reason: the artifact scan that reads this run has to reach
#: the tables and figures **this package** writes, and an absent cap makes ``encoder_attention``
#: record a skip and emit nothing. Eight is what the stratified draw needs to reach all three
#: clinical classes across the eight subgroup shards. The other four caps stay absent, so the run
#: costs no retention it does not need.
EVALUATED_CAPS = {"encoder_attention": 8}


@pytest.fixture(scope="session")
def evaluated(trained_run, multi_class_shards, tmp_path_factory) -> Dict[str, Any]:
    """One real evaluation run of this model; every assertion built on it questions the same run.

    Driven through this package's committed override delta with its placeholder shards repointed,
    which is what an operator does before a real run -- so the merge, the preflight guards it
    satisfies and the generated multi-class shards are all exercised by the same pass.

    Two Monte Carlo draws rather than the shipped eight: the tests reading this fixture are about
    the artifacts rather than the numbers, and each draw decodes every branch over every anchor.

    ``main`` returns the process **exit code**, not the summary path: an analysis failing must be
    visible to a shell. The path is therefore assembled from the directory this fixture named,
    which is what a caller with an explicit ``--output-dir`` does anyway.
    """
    import json

    import yaml

    from teb_vae.lag_attn_transformer_rws.eval import run as trf_run

    overrides = write_repointed_overrides(
        tmp_path_factory.mktemp("evaluated_overrides"), multi_class_shards
    )
    delta = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    delta["eval_config"]["caps"] = dict(EVALUATED_CAPS)
    overrides.write_text(yaml.safe_dump(delta, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path_factory.mktemp("trf_eval")
    exit_code = trf_run.main(
        trained_run, output_dir, overrides=overrides, device="cpu", num_samples=2
    )
    results_dir = Path(output_dir) / trf_run.RESULTS_DIRNAME
    summary_path = results_dir / trf_run.SUMMARY_FILENAME
    text = summary_path.read_text(encoding="utf-8")
    return {
        "exit_code": exit_code,
        "summary_path": summary_path,
        "text": text,
        "summary": json.loads(text),
        "results_dir": results_dir,
    }


@pytest.fixture
def inputs():
    """Seeded ``(y_st, y_ph, u_stream)`` tensors matching the tiny geometry.

    The channel counts are the dataset's and are independent of model size: $43$ FHR scattering,
    $66$ FHR phase-harmonic, and $58$ for the concatenated UP stream. They track
    ``hdf5_dataset/new_pipeline/create_new_pipeline.py``; when its phase-harmonic selection
    changes, these move with it.
    """
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(BATCH, SEQ_LEN, 43, generator=generator)
    y_ph = torch.randn(BATCH, SEQ_LEN, 66, generator=generator)
    u_stream = torch.randn(BATCH, SEQ_LEN, 58, generator=generator)
    return y_st, y_ph, u_stream
