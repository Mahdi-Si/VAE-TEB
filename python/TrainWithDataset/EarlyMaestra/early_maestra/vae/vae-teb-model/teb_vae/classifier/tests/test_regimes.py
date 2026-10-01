"""P7 adaptation regimes (SPEC §10.1, §8.2, §16 P7 accept): the trainable VaeSource contract on the tiny trf_cfs
checkpoint (gradient reach inside ``train.unfreeze``, none outside, frozen modules in eval whatever ``train()`` does),
the backbone's optimiser group, the LPFT stage transition, the regime schema, and one online run through predict."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hdf5_dataset.hdf5_dataset import attribute_dict_collate
from teb_vae.classifier import cohort, config, sources
from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

ALLOW = "posterior_head"  # the pilot's allowlist family: the posterior's delta heads


def _restore_loguru() -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)


def _cfg(vae_overrides, *extra):
    return config.load(SMOKE_CONFIG, list(vae_overrides) + list(extra)).classifier


def _backbone(c, unfreeze=(ALLOW,), cotrain=None):
    """``(OnlineFeatures, raw rows)``: the tiny VAE behind an identity scaler, and 4 collated fold-1 train rows."""
    source = sources.VaeSource(c.source, unfreeze=unfreeze)
    dataset = source.dataset(cohort.fold_shards(c.data, 1)["train"])
    rows = attribute_dict_collate([dataset[i] for i in range(4)])
    probe = source(rows)
    channels = probe.channels
    scaler = sources.Scaler(channels=channels, center=np.zeros(len(channels)), scale=np.ones(len(channels)),
                            keep=np.ones(len(channels), bool), record={})
    n_values = probe.values.shape[-1]
    return sources.OnlineFeatures(source, scaler, n_values=n_values, chunk=3, cotrain=cotrain), rows, probe


def _batch(rows, probe):
    """A segment-scope batch around ``rows``: cached-looking ``x``/``attn`` (zeros) and the source's step mask."""
    return {"vae": rows, "x": torch.zeros_like(probe.values), "attn": torch.zeros_like(probe.attn),
            "step_mask": probe.step_mask.clone()}


# ---- the trainable-source contract -------------------------------------------------------------------------------
def test_gradient_reaches_the_allowlist_only_and_frozen_modules_stay_in_eval(vae_overrides):
    features, rows, probe = _backbone(_cfg(vae_overrides))
    source = features.source
    allowed = {n for n, _ in source.allowlist()}
    assert allowed and all(n.startswith(ALLOW + ".") for n in allowed)
    assert {n for n, p in source.model.named_parameters() if p.requires_grad} == allowed

    features.train()
    source.check_frozen()  # every module outside the allowlist is in eval, whatever train() did
    modes = {n: m.training for n, m in source.model.named_modules()}
    assert all(modes[n] == (n == ALLOW or n.startswith(ALLOW + ".")) for n in modes)
    out = features(_batch(rows, probe))
    x, mask = out["x"], probe.step_mask
    assert x.requires_grad and (x[~mask] == 0).all()
    features.eval()  # dropout off: the online features are the source's own (chunked in 3 + 1)
    torch.testing.assert_close(features(_batch(rows, probe))["x"][mask], probe.values[mask])

    features.train()
    (features(_batch(rows, probe))["x"] ** 2).sum().backward()
    grads = {n: p.grad for n, p in source.model.named_parameters()}
    assert all(grads[n] is None for n in grads if n not in allowed)  # no gradient outside the allowlist
    assert sum(float(grads[n].abs().sum()) for n in allowed if grads[n] is not None) > 0
    assert any(grads[n] is not None and grads[n].abs().sum() > 0 for n in allowed if ".delta_mu_head." in n)

    stray = next(p for n, p in source.model.named_parameters() if n not in allowed)
    stray.requires_grad_(True)
    with pytest.raises(RuntimeError, match="requiring grad"):
        source.check_frozen()
    stray.requires_grad_(False)
    source.model.target_encoder.train()
    with pytest.raises(RuntimeError, match="training mode"):
        source.check_frozen()


def _task(c, features):
    from teb_vae.classifier.model import ClassifierNet
    from teb_vae.classifier.train import ClassifierTask

    kwargs = dict(n_values=features.n_values, n_attn=1, n_ctx=0, model_cfg=c.model.model_dump(mode="json"),
                  labels_cfg=c.labels.model_dump(mode="json"), priors=None)
    task = ClassifierTask(ClassifierNet(**kwargs), lr=1e-3, weight_decay=0.01, classifier_kwargs=kwargs,
                          train_cfg=c.train.model_dump(mode="json"), class_weights=[1.0, 1.0], prior_offset=[0.0])
    task.backbone = features
    return task


# ---- one online run --------------------------------------------------------------------------------------------------
# ---- co-training (§10.6) and the preservation gate (§10.10.2 #12) ------------------------------------------------------
COTRAIN = ("classifier.train.regime=cotrain", f"classifier.train.unfreeze=[{ALLOW}]")


def test_preservation_gate_rejects_a_degraded_vae(vae_overrides, tmp_path):
    from teb_vae.classifier.train import PreservationGateCallback

    c = _cfg(vae_overrides, *COTRAIN)
    features, rows, _ = _backbone(c, cotrain=c.train.cotrain)
    task = _task(c, features)
    logged = {}
    task.log = lambda name, value, **kw: logged.__setitem__(name, value)
    task.val_metrics = {"val/guid_logloss": 0.5}
    trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False, current_epoch=0)
    gate = PreservationGateCallback(rows, tolerance=0.10, monitor="val/guid_logloss", mode="min", chunk=3,
                                    output_dir=tmp_path)
    gate.on_fit_start(trainer, task)
    gate.on_validation_epoch_end(trainer, task)
    assert logged["val/forecast_mse_rel"] == pytest.approx(0.0, abs=1e-6)  # mean decode: deterministic
    assert (logged["val/gate_ok"], logged["val/guid_logloss_gated"]) == (1.0, 0.5)
    assert all(np.isfinite(logged[f"val/{k}"]) for k in ("kld_active_frac", "logvar_prior_floor_frac",
                                                          "delta_mu_sat_frac"))
    with torch.no_grad():  # a synthetic degradation of the forecast
        for name, p in features.vae.decoder.named_parameters():
            if name.endswith("bias"):
                p.add_(3.0)
    trainer.current_epoch = 1
    gate.on_validation_epoch_end(trainer, task)
    assert logged["val/forecast_mse_rel"] > 0.10
    assert (logged["val/gate_ok"], logged["val/guid_logloss_gated"]) == (0.0, float("inf"))
    lines = [json.loads(line) for line in (tmp_path / "preservation.jsonl").read_text().splitlines()]
    assert [line["epoch"] for line in lines] == ["frozen", 0, 1]
