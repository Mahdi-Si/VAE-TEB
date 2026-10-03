r"""Planted-delay check (plan P5-01): fit ``planted.yaml``, read the KL lag map back.

The planted shard couples raw UP to raw FHR at $\delta$ steps (root attribute
``planted_delay_steps``). Anchor $t$ forecasts $t+1 \ldots t+H$, so the informative lags are
$[\delta - H, \delta - 1]$. The band, its share and the per-lag support correction are imported
from the CFS check and the CFS evaluation: the lag map and its lag axis are the same here, only the
inputs differ. The CFS evaluation itself is feature-domain specific, so the reading is done here.

Prints numbers only; the pass/fail reading belongs to the user. Run from the repository root::

    PYTHONPATH=. python teb_vae/lag_attn_transformer_patch/lag_recovery_check.py
    PYTHONPATH=. python teb_vae/lag_attn_transformer_patch/lag_recovery_check.py \
        --override general_config.epochs=5
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

#: Repository root: ``teb_vae/lag_attn_transformer_patch/lag_recovery_check.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button puts this directory, not the repo root, on sys.path.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval.metrics import lag_profiles  # noqa: E402
from teb_vae.lag_attn_cfs.lag_recovery_check import (  # noqa: E402
    band_share,
    planted_band,
    write_override_config,
)
from teb_vae.lag_attn_rws.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402
from teb_vae.lag_attn_transformer_patch.trainer import LagAttnTrfPatchTrainer  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402


@torch.no_grad()
def val_lag_profiles(task: Any, loader: Any) -> Dict[str, torch.Tensor]:
    """Mean per-segment KL lag profiles over the dense val anchors.

    Support: steps in ``[warmup_period, anchor_ceiling)`` whose ``weight`` is positive. ``raw``
    sums over lags to the mean ``kld_per_t``; ``corrected`` divides each lag by the anchors at
    which it exists (the CFS evaluation's support correction).
    """
    model = task.orig_model.eval()
    task._stage = "val"  # dense geometry (the class default; stated, not relied on)
    floor, ceiling = int(model.warmup_period), int(model.anchor_ceiling)
    raws, corrected, anchors = [], [], 0
    for batch in loader:
        batch = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        lag_map = model(*task._build_forward_inputs(batch))["source_kl_lag_map"]
        weight = batch.weight
        support = torch.zeros_like(weight)
        support[:, floor:ceiling] = (weight[:, floor:ceiling] > 0).to(weight.dtype)
        raw, corr, _ = lag_profiles(
            lag_map, support, model.build_lag_mask(weight.shape[1], weight.device)
        )
        raws.append(raw)
        corrected.append(corr)
        anchors += int(support.sum())
    return {
        "raw": torch.cat(raws).mean(0).cpu(),
        "corrected": torch.cat(corrected).mean(0).cpu(),
        "n_segments": sum(r.shape[0] for r in raws),
        "n_anchors": anchors,
    }


def main(*, config: str, override: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Fit the planted config (with optional ``dotted.key=value`` overrides) and read the band.

    Returns:
        The printed numbers, keyed by name.
    """
    config_path = config if os.path.isabs(config) else os.path.join(_REPO_ROOT, config)
    with tempfile.TemporaryDirectory(prefix="patch_lag_recovery_") as staging:
        merged_path = write_override_config(config_path, override or (), Path(staging))
        merged = load_config(str(merged_path))
        driver = run_training(str(merged_path), trainer_cls=LagAttnTrfPatchTrainer)

    shard = merged["dataset_config"]["vae_test_datasets"][0]
    with h5py.File(shard, "r") as handle:
        delta = int(handle.attrs["planted_delay_steps"])

    task = driver.pl_model
    epochs = int(task.current_epoch)  # read before .to(): the trainer is still attached
    task.to("cuda" if torch.cuda.is_available() else "cpu")
    model = task.orig_model
    horizon, max_lag = int(model.horizon), int(model.lag_attn.L) - 1
    band = planted_band(delta, horizon)
    profiles = val_lag_profiles(task, GraphDataModule(merged).val_dataloader())

    record: Dict[str, Any] = {
        "delta": delta, "horizon": horizon, "max_lag": max_lag, "band": band,
        "epochs": epochs, "n_segments": profiles["n_segments"],
        "n_anchors": profiles["n_anchors"],
        "flat_band_share": (band[1] - band[0] + 1) / (max_lag + 1),
    }
    for name in ("raw", "corrected"):
        profile = profiles[name].tolist()
        record[f"{name}_profile"] = profile
        record[f"{name}_band_share"] = band_share(profile, band)
        record[f"{name}_argmax"] = int(profiles[name].argmax())
    raw_total = float(profiles["raw"].sum())

    print(
        f"planted delta = {delta} steps, H = {horizon}, max_lag = {max_lag} -> informative band "
        f"[{band[0]}, {band[1]}]; a flat profile puts {record['flat_band_share']:.3f} in it\n"
        f"epochs trained: {epochs} (last-epoch weights); {record['n_segments']} val segments, "
        f"{record['n_anchors']} anchors\n"
        f"raw profile (sums to mean kld_per_t = {raw_total:.4g} nats): band "
        f"{record['raw_band_share']:.3f}, out-of-band {1 - record['raw_band_share']:.3f}, "
        f"argmax lag {record['raw_argmax']}\n"
        f"support-corrected profile: band {record['corrected_band_share']:.3f}, out-of-band "
        f"{1 - record['corrected_band_share']:.3f}, argmax lag {record['corrected_argmax']}\n"
        "support-corrected profile by lag (nats): "
        + " ".join(f"{v:.3g}" for v in record["corrected_profile"])
    )
    return record


#: Values for arguments absent from the command line (an IDE's Run button), keyed by ``dest``.
RUN_ARGS: Dict[str, Any] = {
    "config": "teb_vae/lag_attn_transformer_patch/configs/planted.yaml",
    # dotted.key=value deltas, e.g. ['general_config.epochs=5'].
    "override": None,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit planted.yaml and read the lag band back.")
    parser.add_argument("--config", help="A planted config; repo-root relative or absolute.")
    parser.add_argument("--override", nargs="+", help="dotted.key=value config deltas.")
    values, _ = resolve_launch_args(parser, RUN_ARGS)
    # Shard paths inside a config are repo-root relative.
    os.chdir(_REPO_ROOT)
    main(config=values["config"], override=values["override"])
