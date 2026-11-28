r"""Calibrate ``irf_gate_k`` of an IRF checkpoint against another recording's UA.

The IRF gate of a channel is open where $z = \text{slope} / \text{se} > k$ (DESIGN.md §6.5). This
tool runs the head on shuffled batches twice, with the own UA and with the UA of another recording
of the batch (a derangement across GUIDs, so no row reads its own recording's UA; a batch of one
recording is skipped), and prints the $k$ at which
the wrong UA opens the gate at a chosen share of the cells. A cell is a scored anchor with UA
cover on one of the head's scored channels, as in ``irf_gate_open``. Only the head runs; the FHR
path is not needed. ``irf_gate_k`` stays a fixed hyperparameter: put the printed value in the
config. $z$ depends on the head's two penalties, so recalibrate after they have been trained.

From the repo root::

    python -m teb_vae.lag_attn_transformer_cfs.eval.calibrate_irf_gate --checkpoint <ckpt> \
        [--config <resolved_config.yaml>] [--shards a.hdf5 b.hdf5 ...] [--stats stats.hdf5]

From an IDE's Run button, :data:`RUN_ARGS` is used; a flag on the command line overrides its entry.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval.probe import load_task, resolve_device, resolved_config_for  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_rws.nets.controls import groups_can_derange, make_derangement  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING  # noqa: E402


@torch.no_grad()
def collect(task: Any, loader: Any, device: torch.device, batches: int) -> Tuple[np.ndarray, np.ndarray]:
    """The gate statistic $z$ of every cell, with the own UA and with another recording's.

    Returns:
        ``(z_own, z_wrong)``, 1-D arrays over cells.
    """
    model = task.orig_model
    generator = torch.Generator().manual_seed(0)
    channels = model.irf_channel_index
    cell_mask = model.forecast_likelihood_kwargs()["cell_mask"]
    if cell_mask is not None:
        channels = channels[(cell_mask.sum(dim=0) > 0)[channels]]
    z_own, z_wrong = [], []
    for number, batch in enumerate(loader):
        if number == batches:
            break
        batch = task.transfer_batch_to_device(batch, device, 0)
        features, weight = task._build_raw_target(batch)
        ua, ua_ok, step_ok = task._build_irf_inputs(batch)
        anchors, anchor_valid = model._build_anchor_index(len(ua), ua.device, 0, 1)  # dense
        mask, _coverage = forecast_mask(
            model.scored_weight(task._target_quality_weight(batch, weight)), model.geometry,
            coverage_floor=model.coverage_floor, anchors=anchors, anchor_valid=anchor_valid,
        )
        scored = contributing_anchors(mask) > 0
        guids = [str(guid) for guid in batch.guid]
        own = model.irf_source(features, ua, ua_ok, step_ok, anchors)
        z_own.append(own["z"][..., channels][scored & own["ua_ok"]].flatten().cpu())
        if not groups_can_derange(guids):
            continue  # one recording: there is no other recording's UA to hand any row
        donor = make_derangement(len(guids), generator=generator, device=ua.device, groups=guids)
        wrong = model.irf_source(features, ua[donor], ua_ok[donor], step_ok, anchors)
        z_wrong.append(wrong["z"][..., channels][scored & wrong["ua_ok"]].flatten().cpu())
    if not z_wrong:
        raise ValueError("no batch held two recordings, so no wrong-UA cell was drawn; raise batch_size")
    return torch.cat(z_own).numpy(), torch.cat(z_wrong).numpy()


def main(checkpoint: str, config: Optional[str] = None, shards: Optional[Sequence[str]] = None,
         stats: Optional[str] = None, batches: Optional[int] = None, batch_size: Optional[int] = None,
         false_open: Optional[float] = None, device: Optional[str] = None) -> float:
    """Print and return the calibrated ``irf_gate_k``.

    Args:
        checkpoint: A checkpoint with an IRF head (``source_path='irf'`` or ``irf_readout``).
        config: The run's resolved config, read for the loader settings. ``None`` finds it beside
            the checkpoint.
        shards: The shards to read. ``None`` means the config's training shards.
        stats: The statistics file. ``None`` means the config's.
        batches: Batches to run. ``None`` means 30.
        batch_size: Rows per batch. ``None`` means 16.
        false_open: Share of the cells at which a wrong UA may open the gate. ``None`` means 0.05.
        device: Torch device. ``None`` picks ``cuda:0`` when available.
    """
    from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

    batches, batch_size = 30 if batches is None else int(batches), 16 if batch_size is None else int(batch_size)
    false_open = 0.05 if false_open is None else float(false_open)
    torch_device = resolve_device(device)
    task = load_task(Path(checkpoint), torch_device, binding=TRF_CFS_BINDING)
    model = task.orig_model
    if getattr(model, "irf_head", None) is None:
        raise ValueError(f"{checkpoint} has no IRF head (neither source_path='irf' nor irf_readout)")
    dataset = load_config(str(resolved_config_for(Path(checkpoint)) if config is None else config))["dataset_config"]
    kwargs = copy.deepcopy(dataset["dataloader_config"]["dataset_kwargs"])
    kwargs["cache_size"] = 0
    torch.manual_seed(0)  # shuffled, so that the rows of a batch are different recordings
    loader = create_optimized_dataloader(
        hdf5_files=list(shards or dataset["vae_train_datasets"]), batch_size=batch_size, num_workers=0,
        shuffle=True, stats_path=str(stats or dataset["stat_path"]),
        normalize_fields=dataset["dataloader_config"]["normalize_fields"], prefetch_factor=2, **kwargs,
    )
    z_own, z_wrong = collect(task, loader, torch_device, batches)
    k = float(np.quantile(z_wrong, 1.0 - false_open))
    now = float(model.irf_head.gate_k)
    print(f"cells: {z_own.size} own UA, {z_wrong.size} wrong UA, on {model.irf_channel_index.numel()} channels")
    print(f"at the checkpoint's irf_gate_k = {now:g}: gate open at {np.mean(z_own > now):.3f} of the "
          f"cells with the own UA, {np.mean(z_wrong > now):.3f} with a wrong UA")
    print(f"irf_gate_k = {k:.2f} opens the gate at {false_open:g} of the cells with a wrong UA, "
          f"and at {np.mean(z_own > k):.3f} with the own UA")
    return k


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; every ``dest`` is also a :data:`RUN_ARGS` key."""
    p = argparse.ArgumentParser(prog="python -m teb_vae.lag_attn_transformer_cfs.eval.calibrate_irf_gate",
                                description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint")
    p.add_argument("--config")
    p.add_argument("--shards", nargs="+")
    p.add_argument("--stats")
    p.add_argument("--batches", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--false-open", dest="false_open", type=float)
    p.add_argument("--device")
    return p


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,   # str -- the irf checkpoint (required)
    "config": None,       # str | None -- resolved config  (None -> found beside the checkpoint)
    "shards": None,       # list[str] | None  (None -> the config's training shards)
    "stats": None,        # str | None  (None -> the config's stat_path)
    "batches": None,      # int | None  (None -> 30)
    "batch_size": None,   # int | None  (None -> 16)
    "false_open": None,   # float | None  (None -> 0.05)
    "device": None,       # str | None  (None -> cuda:0 when available)
}


if __name__ == "__main__":
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, None)
    refusal = missing_required(values, ("checkpoint",))
    if refusal:
        raise SystemExit(refusal)
    os.chdir(_REPO_ROOT)  # the shard paths inside a resolved config may be repo-root-relative
    main(**values)
