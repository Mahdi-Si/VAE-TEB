"""train.py and the unit config: framework integration (SPEC §10.10, §13.2, §15 T-F1, T-F2, T-F3, T-F5), the
optimiser/scheduler overrides, the §10.2 prior correction and §10.8 calibration, on fold 1 of the fixture tree."""
from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from teb_vae.classifier import config
from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SHIPPED = sorted(config.DEFAULT_CONFIG.parent.glob("*.yaml"))
SCOPES = ("sequence", "segment")
SEED = {"sequence": 42, "segment": 7}  # one run dir, two units
#: The early-stopping monitor per scope; best.ckpt must follow it (§10.10.2 #6).
MONITOR = {"sequence": ("val/guid_logloss", "min"), "segment": ("val/guid_auroc", "max")}
EPOCHS = 2


def _restore_loguru() -> None:
    """run.main / train_unit replace loguru's sinks with files under tmp."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)


@pytest.fixture(scope="module")
def env(smoke_overrides, tmp_path_factory):
    """cohort -> extract (fold 1) into a private tmp tree: ``(overrides, run_dir, manifest)``."""
    from teb_vae.classifier import run

    tmp = tmp_path_factory.mktemp("train")
    overrides = list(smoke_overrides) + ["classifier.run.folds=[1]", f"classifier.source.cache_root={tmp / 'cache'}",
                                         f"classifier.train.max_epochs={EPOCHS}"]
    try:
        for stage in ("cohort", "extract"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp / "run"))
    finally:
        _restore_loguru()
    return overrides, tmp / "run", json.loads((tmp / "run" / "manifest.json").read_text())


def _cfg(env, *sets):
    return config.load(SMOKE_CONFIG, env[0] + list(sets))


@pytest.fixture(scope="module")
def trained(env):
    """T-F2: one 2-epoch CPU unit per scope, MLflow off: ``{scope: (cfg, record, unit)}``."""
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.train import train_unit

    out = {}
    try:
        for scope in SCOPES:
            monitor, mode = MONITOR[scope]
            es = "advanced_config.callbacks.early_stopping"
            cfg = _cfg(env, f"classifier.model.scope={scope}", f"{es}.monitor={monitor}", f"{es}.mode={mode}")
            unit = build_unit(cfg, env[1], env[2]["source"], 1)
            out[scope] = cfg, train_unit(cfg, env[1], env[2], fold=1, seed=SEED[scope], kind="model", unit=unit), unit
    finally:
        _restore_loguru()
    return out


# ---- the task: F11, parameter groups, optimiser, schedule, prior correction ----------------------------------------
def _task(**labels):
    from teb_vae.classifier.model import ClassifierNet
    from teb_vae.classifier.train import ClassifierTask

    c = config.load(SMOKE_CONFIG).classifier  # labels bypass the root validator (the head / loss / λ3 checks)
    kwargs = dict(n_values=3, n_attn=1, n_ctx=2, model_cfg=c.model.model_dump(mode="json"),
                  labels_cfg=c.labels.model_copy(update=labels).model_dump(mode="json"), priors=None)
    return ClassifierTask(ClassifierNet(**kwargs), lr=1e-3, weight_decay=0.01, classifier_kwargs=kwargs,
                          train_cfg=c.train.model_dump(mode="json"), class_weights=[1.0] * 3, prior_offset=[0.0]), c


def test_prior_correction_known_answers():
    from teb_vae.classifier.train import prior_correction

    unit = SimpleNamespace(class_counts=[30, 10], priors={"main": [0.75, 0.25]})

    def offset(*sets):
        return prior_correction(config.load(config.DEFAULT_CONFIG, list(sets)).classifier, unit)

    assert offset() == [0.0]
    assert offset("classifier.train.loss.name=weighted_bce", "classifier.train.loss.weighting=inverse") == \
        pytest.approx([math.log(3)])  # s - log(w1/w0), w = 1/n
    assert offset("classifier.train.loss.name=logit_adjusted") == pytest.approx([math.log(3)])  # s + tau log(pi1/pi0)
    assert offset("classifier.train.loss.name=weighted_ce", "classifier.labels.head=multiclass",
                  "classifier.train.loss.weighting=inverse") == pytest.approx([0.0, math.log(3)])
    # class_balanced trains under a uniform prior: the inverse-weighting offset log(n0/n1), added to any weighting's
    assert offset("classifier.train.sampler=class_balanced") == pytest.approx([math.log(3)])
    assert offset("classifier.train.sampler=class_balanced", "classifier.train.loss.name=weighted_bce",
                  "classifier.train.loss.weighting=inverse") == pytest.approx([2 * math.log(3)])
    assert offset("classifier.train.sampler=class_balanced", "classifier.labels.head=multiclass",
                  "classifier.train.loss.name=ce") == pytest.approx([0.0, math.log(3)])


# ---- T-F3: callback order and the trainer kwargs ------------------------------------------------------------------
# ---- T-F2 / T-F3: the train smoke, both scopes ---------------------------------------------------------------------
# ---- T-F5: teardown ------------------------------------------------------------------------------------------------
# ---- calibration (§10.8) -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 2.0, 20000)
    return z, (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(int)


def test_temperature_recovers_a_known_temperature(synthetic):
    from teb_vae.classifier.train import apply_calibration, fit_calibration

    z, y = synthetic
    cal = fit_calibration(2.5 * z, y, "temperature")  # over-confident by T = 2.5
    assert cal["temperature"] == pytest.approx(2.5, rel=0.05)
    np.testing.assert_allclose(apply_calibration(2.5 * z, cal), z * 2.5 / cal["temperature"])
    assert fit_calibration(z, y, "temperature")["temperature"] == pytest.approx(1.0, rel=0.05)
