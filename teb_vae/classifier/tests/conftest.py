"""Shared fixtures for the classifier tests. Imports stay inside fixtures, so a half-written module
elsewhere in the package never breaks collection of an unrelated test file."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CONFIG = REPO_ROOT / "teb_vae" / "classifier" / "configs" / "smoke.yaml"
#: The fixture's two covariates (§7.3), as a ``classifier.context.covariates.variables`` override value.
COVARIATES = ("[{name: parity, kind: categorical, available_at: prospective}, "
              "{name: temp_c, kind: numeric, available_at: prospective}]")
#: The tiny ``lag_attn_transformer_cfs`` VAE of :func:`vae_checkpoint`: d_z, lag-attention heads, lag bins (max_lag 8
#: -> L = 9 lags), and the source keys of :func:`vae_overrides`.
D_Z, HEADS, BINS = 8, 4, [[0, 2], [3, 5], [6, 8]]
KEYS = ("classifier.source.vae.keys=[{name: mu_prior}, {name: delta_mu}, {name: kld_per_dim}, "
        "{name: attn_summary}, {name: kld_per_t, transform: log1p, role: attention}]")


def pytest_configure(config: pytest.Config) -> None:
    """Register ``slow``; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line("markers", "slow: trains networks (seconds each on CPU); deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def fixture_tree(tmp_path_factory) -> Dict[str, Any]:
    """The §15 k-fold fixture tree (3 folds, 4 subgroups, 60 GUIDs) and its stats file."""
    from teb_vae.classifier.tests.fixtures.make_fixture import generate

    return generate(tmp_path_factory.mktemp("classifier_fixture"))


@pytest.fixture(scope="session")
def smoke_overrides(fixture_tree) -> List[str]:
    """``--set`` strings pointing ``smoke.yaml`` at the session's fixture tree."""
    return [f"classifier.data.kfold_root={fixture_tree['root']}",
            f"classifier.source.hdf5.stats_path={fixture_tree['stats_path']}",
            f"classifier.source.cache_root={fixture_tree['root']}/cache"]


@pytest.fixture(scope="session")
def covariate_overrides(smoke_overrides, fixture_tree) -> List[str]:
    """``smoke_overrides`` plus the fixture's static and timed covariate tables and their :data:`COVARIATES`."""
    return smoke_overrides + [f"classifier.context.covariates.static_csv={fixture_tree['static_csv']}",
                              f"classifier.context.covariates.timed_csv={fixture_tree['timed_csv']}",
                              f"classifier.context.covariates.variables={COVARIATES}"]


@pytest.fixture(scope="session")
def smoke_config(smoke_overrides):
    """``smoke.yaml`` loaded against the fixture tree, all three folds."""
    from teb_vae.classifier.config import load

    return load(SMOKE_CONFIG, smoke_overrides + ["classifier.run.folds=[1,2,3]"]).classifier


@pytest.fixture(scope="session")
def fixture_segments(smoke_config):
    """``(segments, info)`` of the three-fold fixture, before GUID-level columns. Treat as read-only."""
    from teb_vae.classifier import cohort

    data = smoke_config.data
    shards = {(k, s): p for k in (1, 2, 3) for s, p in cohort.fold_shards(data, k).items()}
    return cohort.segment_table(shards, trim_minutes=1.0, stride_s=data.stride_s,
                                epoch_min_s=data.epoch_min_s, min_valid_frac=data.min_valid_frac)


@pytest.fixture(scope="session")
def fixture_guids(fixture_segments, smoke_config):
    """``(segments, guids)`` after :func:`guid_table` under the smoke labels. Treat as read-only."""
    from teb_vae.classifier import cohort

    return cohort.guid_table(fixture_segments[0], smoke_config.labels, smoke_config.cohort)


def save_checkpoint(root: Path, model, kwargs, stats_path) -> Path:
    """``root/model_checkpoints/best.ckpt`` plus the ``resolved_config.yaml`` VAE training writes."""
    import torch
    import yaml

    directory = root / "model_checkpoints"
    directory.mkdir(parents=True)
    torch.save({"model_class": type(model).__name__, "model_kwargs": kwargs,
                "hyper_parameters": {"likelihood": "gaussian_nll"},
                "state_dict": model.state_dict()}, directory / "best.ckpt")
    streams = ["fhr", "up", "fhr_st", "fhr_ph", "up_st", "up_ph"]
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump({"dataset_config": {
        "stat_path": stats_path, "vae_train_datasets": [], "vae_test_datasets": [],
        "dataloader_config": {"normalize_fields": streams, "dataset_kwargs": {
            "load_fields": streams + ["weight", "guid", "epoch"], "epoch_min": -48000,
            "trim_minutes": 1.0, "cache_size": 0, "pin_memory": False}}}}))
    return directory / "best.ckpt"


@pytest.fixture(scope="session")
def vae_checkpoint(fixture_tree, tmp_path_factory) -> Path:
    """A tiny ``lag_attn_transformer_cfs`` VAE (``make_task`` on ``TINY_KWARGS``) at the fixture's geometry
    (T = 300, c_y = 102, c_u = 51). Its posterior delta heads are perturbed: zero-initialised, ``delta_mu`` and the KL
    would otherwise be exactly zero."""
    import torch
    from teb_vae.lag_attn_transformer_cfs.tests.conftest import TINY_KWARGS, make_task

    kwargs = dict(TINY_KWARGS, sequence_length=300, horizon=30, warmup_period=134, anchor_stride=30)
    model = make_task(model_kwargs=kwargs).orig_model
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    return save_checkpoint(tmp_path_factory.mktemp("tiny_vae"), model, kwargs, fixture_tree["stats_path"])


@pytest.fixture(scope="session")
def vae_overrides(smoke_overrides, vae_checkpoint) -> List[str]:
    """``smoke_overrides`` with ``source.kind: vae`` on :func:`vae_checkpoint` and the :data:`KEYS`."""
    return list(smoke_overrides) + ["classifier.source.kind=vae", KEYS,
                                    f"classifier.source.vae.checkpoint={vae_checkpoint}",
                                    f"classifier.source.vae.lag_bins={BINS}"]
