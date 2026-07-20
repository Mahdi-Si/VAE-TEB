"""Shared pytest configuration for the evaluation-pipeline tests.

Two jobs. First, the ``sys.path`` pin and the ``utils`` shadow guard, mirroring
``teb_vae/lag_attn/tests/conftest.py`` one directory deeper. Second, the fixtures.

The model suite's fixtures are *re-exported*, not copied. ``eval/tests/`` is a **sibling** of
``teb_vae/lag_attn/tests/``, not a child, so pytest never collects that conftest's fixtures
here; importing them by name into this module's namespace registers them, and leaves exactly
one definition of each in the tree.

Two fixtures are new, and the second is the load-bearing one. ``perturb_posterior`` breaks the
zero-init of the posterior delta heads so the KL terms become nonzero -- without it every KL
assertion passes on an untouched model, including on a model that is entirely wrong. But
``_zero_init_delta_heads`` zeroes ``residual_decoder.mean_head`` too, so on a model perturbed
only through its posterior ``delta_mu_src`` is identically zero *regardless of* $z$ -- which
makes ``residual_ratio``, the mean-space uplift and every lag-band difference exactly zero,
and every assertion about the forecast pathway vacuous in a second, less obvious way.
``perturb_full_pathway`` perturbs both.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

# teb_vae/lag_attn/eval/tests/conftest.py -> [0]=tests, [1]=eval, [2]=lag_attn, [3]=teb_vae,
# [4]=repo root. One deeper than the model suite's parents[3].
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
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

# Re-exported, not redefined. Flake8 would call these unused; they are the fixtures.
from teb_vae.lag_attn.tests.conftest import (  # noqa: E402,F401
    BATCH,
    PROD_HPARAMS,
    PROD_KWARGS,
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    inputs,
    make_stub_batch,
    make_stub_batch_fn,
    perturb_posterior,
    prod_kwargs,
    shipped_kwargs,
    stub_batch,
    task,
    tiny_kwargs,
)

#: Repo-root-relative path to the eval config the suite runs against.
EVAL_TINY_CONFIG = "teb_vae/lag_attn/eval/tests/fixtures/eval_tiny.yaml"

#: The committed shard every data-touching test reads: 4 samples, $T = 330$ on disk becoming
#: $300$ after ``trim_minutes: 1.0``.
TINY_SHARD = "teb_vae/lag_attn/tests/fixtures/tiny_shard.hdf5"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, so a test can resolve the repo-root-relative config paths."""
    return Path(_REPO_ROOT)


def apply_full_pathway_perturbation(model, seed: int = 3, scale: float = 0.1) -> None:
    """Break the zero-init of *both* delta pathways, in place.

    The posterior delta heads and the residual decoder's mean head are zeroed together by
    ``_zero_init_delta_heads``, and they gate two different things: the first makes $K_t
    \\equiv 0$, the second makes ``delta_mu_src`` identically zero whatever $z$ is. A model
    perturbed only through the first still has a completely dead forecast residual.

    Args:
        model: The model to perturb, mutated in place.
        seed: Seed for the perturbation, so a fixture is reproducible.
        scale: Standard deviation of the added noise.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        parameters = list(model.posterior_head.parameters())
        parameters += list(model.residual_decoder.mean_head.parameters())
        for parameter in parameters:
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)


@pytest.fixture
def perturb_full_pathway():
    """Factory fixture perturbing the posterior head **and** ``residual_decoder.mean_head``.

    Required by any test that asserts on an uplift, a residual ratio, a lag-band difference or
    a load health probe. See the module docstring for why ``perturb_posterior`` alone is not
    enough.
    """
    return apply_full_pathway_perturbation


def build_tiny_checkpoint_blob(
    model_kwargs: Dict[str, Any] | None = None,
    hparams: Dict[str, Any] | None = None,
    *,
    perturb: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build a checkpoint blob shaped exactly as a real training run writes one.

    The blob goes through ``SeqVaeLagAttnTask.on_save_checkpoint``, which calls its base
    first, so it carries both ``model_class`` (the base's stamp) and ``model_kwargs`` (the
    task's). Together those make it self-describing: the architecture rebuilds with no config
    file, and a blob written by a different model is refused before that rebuild is attempted.

    The model is perturbed before saving by default, and that is not cosmetic. A checkpoint of
    a freshly constructed model has zero delta heads, which is *indistinguishable in weight
    space* from a checkpoint that never loaded -- so an unperturbed fixture would fail the
    load verification it is supposed to demonstrate passing.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to ``SHIPPED_KWARGS`` -- the flag set
            the production config actually ships, at the tiny geometry.
        hparams: Task loss hparams. Defaults to ``PROD_HPARAMS``.
        perturb: Whether to break the delta-head zero-init before saving.
        seed: Seed for construction and perturbation.

    Returns:
        The checkpoint dict, ready for ``torch.save``.
    """
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
    from teb_vae.lag_attn.task import SeqVaeLagAttnTask

    kwargs: Dict[str, Any] = dict(SHIPPED_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(seed)
    model = SeqVaeLagAttn(**kwargs)
    if perturb:
        apply_full_pathway_perturbation(model, seed=seed + 3)

    loss_hparams: Dict[str, Any] = dict(PROD_HPARAMS, **(hparams or {}))
    task = SeqVaeLagAttnTask(model, lr=1e-3, model_kwargs=kwargs, **loss_hparams)
    blob: Dict[str, Any] = {
        "state_dict": task.state_dict(),
        "hyper_parameters": dict(task.hparams),
        "epoch": 0,
        "global_step": 0,
    }
    task.on_save_checkpoint(blob)
    return blob


#: The $\beta$ schedule the production config actually ships
#: (``teb_vae/lag_attn/configs/default.yaml``), beside the ``kld_beta`` it documents as the
#: *fallback for* ``kind == constant``. Reproduced here because ``PROD_HPARAMS`` pins
#: ``beta_schedule=None`` -- which keeps $\beta$ out of the way of tests that are not about the
#: schedule, but also means every other fixture in this suite reports the fallback constant and
#: the shipped ramp is exercised nowhere.
SHIPPED_BETA_SCHEDULE: Dict[str, Any] = {
    "kind": "linear_warmup",
    "start": 1.0e-4,
    "end": 0.1,
    "warmup_epochs": 50,
}

#: The shipped fallback constant, two orders of magnitude below the schedule's ``end``. That gap
#: is what makes a checkpoint built with both a usable probe: a pipeline reporting the constant
#: instead of the schedule is off by $100\times$ in $\beta$, not by a rounding error.
SHIPPED_FALLBACK_KLD_BETA: float = 0.001


@pytest.fixture
def warmup_checkpoint(tmp_path):
    r"""Factory writing a checkpoint whose objective carries a *scheduled* $\beta$.

    ``build_tiny_checkpoint_blob`` stamps ``epoch: 0``, which is the one epoch at which a
    ``linear_warmup`` schedule and its ``start`` value coincide -- so a fixture that left it
    there would be nearly as blind as one pinning ``beta_schedule=None``. The epoch is therefore
    a required argument, and it is written onto the blob's own top-level ``epoch`` key: that is
    where Lightning records it and where
    :meth:`~teb_vae.lag_attn.eval.runner.Objective.from_checkpoint` reads it from.

    Returns:
        ``_make(epoch, schedule=None, **hparams) -> Path``, where ``schedule`` defaults to
        :data:`SHIPPED_BETA_SCHEDULE` and ``hparams`` overrides any loss hparam.
    """

    def _make(epoch: int, schedule: Dict[str, Any] | None = None, **hparams) -> Path:
        overrides: Dict[str, Any] = {
            "beta_schedule": SHIPPED_BETA_SCHEDULE if schedule is None else schedule,
            "kld_beta": SHIPPED_FALLBACK_KLD_BETA,
        }
        overrides.update(hparams)
        blob = build_tiny_checkpoint_blob(hparams=overrides)
        blob["epoch"] = int(epoch)
        path = Path(tmp_path) / f"lag-attn-epoch={int(epoch):03d}.ckpt"
        torch.save(blob, path)
        return path

    return _make


@pytest.fixture(scope="session")
def tiny_eval_config() -> Dict[str, Any]:
    """The merged, validated config the analysis tests run against.

    Session-scoped because it is pure parsing and every analysis test wants the same one.
    """
    import os

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn.eval.config_schema import validate_eval_config
    from teb_vae.lag_attn.eval.run import force_single_process_loader

    previous = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        merged = load_config(str(Path(_REPO_ROOT) / EVAL_TINY_CONFIG))
    finally:
        os.chdir(previous)
    force_single_process_loader(merged)
    merged["eval_config"] = validate_eval_config(merged)
    return merged


@pytest.fixture(scope="session")
def tiny_loader(tiny_eval_config):
    """A real test dataloader over the committed 4-sample shard.

    A real loader rather than a list of stub batches, because these tests are about analyses
    that must survive the actual batch contract -- ``guid`` as a ``list[str]``,
    ``source_file_basename``, a ``weight`` field, and a trimmed $T = 300$.

    Session-scoped and iterated many times, which is exactly the multi-pass usage that motivates
    forcing ``num_workers`` to 0.
    """
    import os

    from train.data_module import GraphDataModule

    previous = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        return GraphDataModule(tiny_eval_config).test_dataloader()
    finally:
        os.chdir(previous)


@pytest.fixture
def make_eval_runner(tmp_path):
    """Factory building an :class:`EvalRunner` around a model, for the analysis tests.

    ``perturb`` defaults to the *full pathway*, not the posterior alone. That is the load-bearing
    default: ``_zero_init_delta_heads`` zeroes ``residual_decoder.mean_head`` too, so a model
    perturbed only through its posterior has ``delta_mu_src`` identically zero whatever $z$ is --
    every uplift, residual ratio and lag-band difference is then exactly zero, and a test
    asserting on them passes while proving nothing.
    """

    def _make(
        model_kwargs: Dict[str, Any] | None = None,
        hparams: Dict[str, Any] | None = None,
        *,
        perturb: bool = True,
        seed: int = 0,
        output_dir: Path | None = None,
    ):
        from teb_vae.lag_attn.eval.runner import EvalRunner, Objective
        from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

        kwargs = dict(SHIPPED_KWARGS if model_kwargs is None else model_kwargs)
        torch.manual_seed(seed)
        model = SeqVaeLagAttn(**kwargs)
        if perturb:
            apply_full_pathway_perturbation(model, seed=seed + 3)
        model.eval()

        loss_hparams = dict(PROD_HPARAMS, **(hparams or {}))
        objective = Objective(
            likelihood=loss_hparams["likelihood"],
            sigma_obs=loss_hparams["sigma_obs"],
            free_bits=loss_hparams["free_bits"],
            detach_baseline_in_full=loss_hparams["detach_baseline_in_full"],
            lambda_full=loss_hparams["lambda_full"],
            lambda_base=loss_hparams["lambda_base"],
            lambda_lag=loss_hparams["lambda_lag"],
            beta_schedule=loss_hparams["beta_schedule"],
            kld_beta=loss_hparams["kld_beta"],
        )
        directory = Path(output_dir) if output_dir is not None else tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        return EvalRunner(
            model=model,
            device=torch.device("cpu"),
            output_dir=directory,
            objective=objective,
            checkpoint_path=directory / "synthetic.ckpt",
            model_kwargs=kwargs,
        )

    return _make


# ---------------------------------------------------------------------------
# A multi-subgroup, multi-class shard set
#
# The committed ``tiny_shard.hdf5`` is a single file whose ``target`` is all zeros, so every
# class-aware path self-skips against it and only the fallback branches are ever tested. This
# generator writes a small set of shards named after real subgroups and carrying real class
# codes, into ``tmp_path``.
#
# Written, never committed: the existing binary fixtures are what the whole suite's numbers are
# pinned against, and regenerating them to add two fields would perturb every one of those tests
# for the benefit of these. This is a test helper and lives with the tests, not in the package.
# ---------------------------------------------------------------------------
#: Subgroups the generator writes, with the clinical class code each carries. Two classes over
#: three shards, which is the smallest set that exercises a grouped emission on both axes -- the
#: class axis needs two distinct classes, the subgroup axis needs more shards than classes so the
#: two groupings cannot accidentally coincide.
MULTI_CLASS_SUBGROUPS: Dict[str, int] = {
    "healthy_no_bg_no_cs": 1,
    "healthy_bg_cs": 1,
    "acidosis_cs": 2,
}


def write_multi_class_shards(
    directory: Path, *, n_samples: int = 2, seq_len: int = 330, seed: int = 11
) -> list:
    """Write one shard per entry of :data:`MULTI_CLASS_SUBGROUPS`, at the committed geometry.

    Mirrors ``scripts/make_tiny_shard.py`` -- the same field names, channel counts and on-disk
    length -- so the shards load through the real ``GraphDataModule`` rather than through a stub.
    Two things differ, and they are the point of the fixture: the files are named after canonical
    subgroups, and ``target`` carries a real class code rather than zero.

    ``weight`` is deliberately **fractional at the segment edges**. That is the case the dataset's
    own ``label`` filter would silently drop, and the case
    :func:`~teb_vae.lag_attn.eval.labels.clinical_class_code` exists to handle: at
    ``weight = 0.5`` an acidosis step stores ``target = 1.0``, which is exactly what a fully-valid
    healthy step stores.

    Args:
        directory: Destination directory, created if absent.
        n_samples: Samples per shard.
        seq_len: On-disk feature length before trimming.
        seed: Seed, so the fixture is reproducible.

    Returns:
        The written shard paths, as strings, in :data:`MULTI_CLASS_SUBGROUPS` order.
    """
    import h5py

    channels = {"fhr_st": 43, "fhr_ph": 66, "up_st": 43, "up_ph": 15, "fhr_up_ph": 79}
    log_fields = ("fhr_st", "up_st")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    for offset, (subgroup, code) in enumerate(MULTI_CLASS_SUBGROUPS.items()):
        rng = np.random.default_rng(seed + offset)
        path = directory / f"{subgroup}.hdf5"
        signal_len = seq_len * 16

        weight = np.ones((n_samples, seq_len), dtype="f4")
        # The fractional boundary the exact-equality filter would drop.
        weight[:, :4] = 0.5
        weight[:, -4:] = 0.5

        with h5py.File(str(path), "w", libver="latest") as handle:
            handle.create_dataset(
                "fhr", data=(140.0 + 10.0 * rng.standard_normal((n_samples, signal_len))).astype("f4")
            )
            handle.create_dataset(
                "up", data=(30.0 + 10.0 * rng.standard_normal((n_samples, signal_len))).astype("f4")
            )
            for field, width in channels.items():
                values = rng.standard_normal((n_samples, width, seq_len))
                if field in log_fields:
                    values = np.abs(values) + 0.1
                handle.create_dataset(field, data=values.astype("f4"))

            handle.create_dataset("weight", data=weight)
            # The class code scaled by validity, exactly as the real pipeline stores it.
            handle.create_dataset("target", data=(float(code) * weight).astype("f4"))
            handle.create_dataset(
                "epoch", data=np.full((n_samples,), -20000.0, dtype="f4")
            )
            handle.create_dataset(
                "cs_label",
                data=np.full((n_samples,), 1 if subgroup.endswith("_cs") else 0, dtype="u1"),
            )
            handle.create_dataset(
                "bg_label", data=np.full((n_samples,), 1 if "_bg_" in subgroup else 0, dtype="u1")
            )
            handle.create_dataset(
                "guid",
                data=[f"{subgroup.upper()}_{index:03d}" for index in range(n_samples)],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        written.append(str(path))
    return written


@pytest.fixture(scope="session")
def multi_class_shards(tmp_path_factory) -> list:
    """Paths to the generated multi-subgroup shards. Session-scoped; treat as read-only."""
    return write_multi_class_shards(tmp_path_factory.mktemp("multi_class"))


@pytest.fixture(scope="session")
def multi_class_config(tiny_eval_config, multi_class_shards) -> Dict[str, Any]:
    """The tiny eval config repointed at the multi-class shards.

    Reuses the committed ``tiny_stats.hdf5``: the shards are written at the same geometry with the
    same field widths, and the stats file describes the *channel layout*, not the recordings.
    """
    import copy

    config = copy.deepcopy(dict(tiny_eval_config))
    config["dataset_config"] = dict(config.get("dataset_config") or {})
    config["dataset_config"]["vae_test_datasets"] = list(multi_class_shards)
    return config


@pytest.fixture(scope="session")
def multi_class_loader(multi_class_config):
    """A real test dataloader over the generated multi-subgroup, multi-class shards."""
    import os

    from train.data_module import GraphDataModule

    previous = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        return GraphDataModule(multi_class_config).test_dataloader()
    finally:
        os.chdir(previous)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A saved checkpoint at the tiny geometry, self-describing and load-verifiable.

    Session-scoped: it is read by most of the suite and building it costs a model
    construction. Tests must treat the file as read-only.
    """
    blob = build_tiny_checkpoint_blob()
    path = tmp_path_factory.mktemp("checkpoint") / "lag-attn-epoch=00.ckpt"
    torch.save(blob, path)
    return path
