r"""Shared pytest configuration for the ``synthetic_v2`` test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_SV2 = _TESTS_DIR.parent
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config) -> None:
    r"""Register custom markers so unknown-mark warnings do not fire.

    ``slow`` tags the heavy end-to-end integration test (S6-T05), which runs the real
    scattering transform on a tiny grid. Deselect it with ``-m "not slow"``.
    """
    config.addinivalue_line(
        "markers", "slow: heavy integration test (real transform); skip with -m 'not slow'"
    )
    # synthetic_v4 (raw-model) suite marker (S0-T06); select with -m v4, exclude with -m "not v4".
    config.addinivalue_line(
        "markers", "v4: synthetic_v4 raw-model validation test; select with -m v4"
    )


# ---------------------------------------------------------------------------
# synthetic_v4 fabricated fixtures (S0-T06): defined in ``conftest_v4.py`` and re-exported here
# so they are visible suite-wide. Heavy imports inside those fixtures stay lazy, so this star
# import is cheap at collection and leaves the v2/v3 suites untouched.
# ---------------------------------------------------------------------------
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from conftest_v4 import *  # noqa: E402,F401,F403


# ---------------------------------------------------------------------------
# Shared real-cache discovery (Sprint 1: parity / forward-contract / zero-KL).
# ---------------------------------------------------------------------------
def find_cache_dir(benchmark: str = "G1_raw"):
    r"""Discover a usable cache dir under ``synthetic_v2/data/<benchmark>/`` or ``None``.

    Prefers the ``data_tag`` resolved from ``config_synth_v3.yaml`` (the prod
    ``G1_raw_v2_notch`` cache when present); otherwise falls back to ANY sibling cache dir
    that carries all three splits + ``meta.json`` (e.g. a reduced local pilot cache built
    under a distinct ``data_tag`` per S0-T00). Returns the first complete cache, or ``None``
    when none has been built yet.
    """
    candidates = []
    try:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E501
            load_config,
        )
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E501
            resolve_cache_dir,
        )

        cfg = load_config(_SV2 / "config_synth_v3.yaml")
        candidates.append(resolve_cache_dir(cfg, benchmark=benchmark))
    except Exception:  # noqa: BLE001 - config optional; fall back to a filesystem scan
        pass

    data_root = _SV2 / "data" / benchmark
    if data_root.is_dir():
        candidates.extend(sorted(p for p in data_root.iterdir() if p.is_dir()))

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if (all((cand / f"{s}.npz").is_file() for s in ("train", "val", "test"))
                and (cand / "meta.json").is_file()):
            return cand
    return None


@pytest.fixture(scope="session")
def shared_cache_dir():
    r"""The discovered real-cache dir, or ``None`` when no cache has been built."""
    return find_cache_dir()


@pytest.fixture(scope="session")
def cache_batch(shared_cache_dir):
    r"""A deterministic real-cache batch as ``(config, batch, cache_dir)``, or ``None``.

    Loads ``config_synth_v3.yaml`` and one **ordered** (val) batch from the discovered cache
    via :class:`SyntheticTEDataModuleV2` with an explicit ``cache_dir`` override, so the
    Sprint 1 parity / forward-contract / zero-KL tests all read the same real scattering
    features. Returns ``None`` when no cache exists (each test then fails loudly, unless it
    is a ``-k fallback`` variant).
    """
    if shared_cache_dir is None:
        return None
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E501
        load_config,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v2 import (  # noqa: E501
        SyntheticTEDataModuleV2,
    )

    cfg = load_config(_SV2 / "config_synth_v3.yaml")
    # Single-process DataLoader: multi-worker spawn is fragile under pytest on Windows
    # (workers re-import the session and can exit unexpectedly). A test batch is tiny, so
    # in-process loading is both robust and fast.
    cfg["dataset"] = {**(cfg.get("dataset") or {}), "num_workers": 0,
                      "persistent_workers": False, "pin_memory": False}
    dm = SyntheticTEDataModuleV2(cfg, batch_size=4, cache_dir=shared_cache_dir)
    dm.setup("fit")
    torch.manual_seed(0)
    # Prefer the ordered val loader for a deterministic batch; fall back to train.
    loader = dm.val_dataloader() or dm.train_dataloader()
    batch = next(iter(loader))
    return {"config": cfg, "batch": batch, "cache_dir": shared_cache_dir}
