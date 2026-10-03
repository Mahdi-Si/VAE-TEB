"""P5-02: ``VaeSource`` loads a ``lag_attn_transformer_patch`` checkpoint with no classifier change.

The checkpoint is the tiny patch model (``TINY_KWARGS``: W = 30, H = 30, T = 300) in the trainer's blob layout,
next to the ``resolved_config.yaml`` of ``configs/tiny.yaml``, read on the committed 4-sample shard. Its posterior
heads are perturbed: zero-initialised, ``delta_mu`` and the KL would otherwise be exactly zero.
"""
from __future__ import annotations

import torch
import yaml

from hdf5_dataset.hdf5_dataset import attribute_dict_collate
from teb_vae.classifier import sources
from teb_vae.classifier.config import load
from teb_vae.classifier.tests.conftest import REPO_ROOT, SMOKE_CONFIG
from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_patch.tests.conftest import TINY_KWARGS, absolutize_dataset_paths, build


def test_vae_source_loads_patch_checkpoint(tmp_path):
    model = build()
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    torch.save({"model_class": type(model).__name__, "model_kwargs": TINY_KWARGS,
                "hyper_parameters": {"seed": 0}, "state_dict": model.state_dict()}, tmp_path / "best.ckpt")
    package = REPO_ROOT / "teb_vae" / "lag_attn_transformer_patch"
    resolved = absolutize_dataset_paths(load_config(str(package / "configs" / "tiny.yaml")))
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(resolved))

    cfg = load(SMOKE_CONFIG, ["classifier.source.kind=vae", "classifier.source.vae.package=lag_attn_transformer_patch",
                              f"classifier.source.vae.checkpoint={tmp_path / 'best.ckpt'}",
                              "classifier.source.vae.step_support=supervised"]).classifier.source
    source = sources.VaeSource(cfg)  # default keys: mu_prior, delta_mu, kld_per_t (log1p, attention)
    dataset = source.dataset(resolved["dataset_config"]["vae_train_datasets"])
    batch = attribute_dict_collate([dataset[i] for i in range(len(dataset))])
    feats = source(batch)

    d_z, t = TINY_KWARGS["d_z"], torch.arange(300)
    assert (source.warmup, source.ceiling) == (30, 270)
    assert feats.values.shape == (4, 300, 2 * d_z) and feats.attn.shape == (4, 300, 1)
    assert feats.channels[-1] == "kld_per_t"
    mask = feats.step_mask
    assert torch.equal(mask, (batch["weight"] > 0) & (t >= 30) & (t < 270))
    kept = mask.any(0).nonzero().ravel()
    assert (kept[0].item(), kept[-1].item()) == (30, 269)
    assert torch.isfinite(feats.values).all() and torch.isfinite(feats.attn).all()
    assert feats.values[..., d_z:][mask].abs().max() > 0 and feats.attn[mask].max() > 0  # the perturbation took
