r"""Shared fixtures for the execution-machine tests, and nothing the logic subset can trip over.

**Import isolation is the constraint this file exists under.** ``pytest`` loads a ``conftest.py``
for every directory below it, so this one is loaded when ``tests/logic`` runs alone -- and that
subset must run on a machine with no fixtures generated, no checkpoint and no GPU. So the module
level here imports the standard library and ``pytest``, and every heavy import lives inside the
fixture that needs it. A top-level import of the fixture generator, the model or the pipeline would
make the minimal subset depend on the whole repository to collect.

The fixtures below are for the tests that need real files. Each of them is skipped -- not failed --
when the fixtures have not been generated, because "you have not run the generator yet" is a
different message from "the pipeline is broken", and only one of them is worth a traceback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

#: The pilot package root, from this file rather than from a working directory.
PILOT_ROOT = Path(__file__).resolve().parents[1]

#: The smoke configuration, which every fixture-backed test runs under. Never the production one:
#: a test must not be able to read production shards or write into a production run directory.
SMOKE_CONFIG = PILOT_ROOT / "configs" / "smoke.yaml"

#: How to produce the fixtures when they are missing, quoted verbatim in the skip message.
GENERATE_COMMAND = (
    "python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate"
)


@pytest.fixture(scope="session")
def smoke_settings() -> Dict[str, Any]:
    """The resolved smoke settings, with no fixture files required.

    Resolving a configuration reads no dataset, so this fixture works before anything is generated
    and is what the tests that only inspect the configuration depend on.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import config as pilot_config

    return pilot_config.resolve_settings(SMOKE_CONFIG)


@pytest.fixture(scope="session")
def smoke_fixtures(smoke_settings) -> Dict[str, Any]:
    """The generated fixture manifest, or a skip naming the command that writes it.

    Session-scoped and read-only: the smoke scenario runs several stages against these files and
    regenerating them between stages would compare two cohorts.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures import generate as fixtures

    paths = dict(smoke_settings["paths"])
    required: List[Path] = [Path(paths["checkpoint"]), Path(paths["statistics"])]
    for key in ("train_shards", "val_shards", "test_shards"):
        required.extend(Path(path) for path in paths[key])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip(
            f"{len(missing)} smoke fixture file(s) are absent, e.g. {missing[0]}. Generate them "
            f"once with: {GENERATE_COMMAND}"
        )

    return {
        "root": str(fixtures.GENERATED_ROOT),
        "checkpoint": paths["checkpoint"],
        "statistics": paths["statistics"],
        "splits": {
            split: list(paths[f"{key}_shards"])
            for split, key in (("train", "train"), ("val", "val"), ("test", "test"))
        },
        "clinical": False,
    }
