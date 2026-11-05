r"""Shared paths, the recording split and the loader helpers of every probe and surgery script.

Every script in this package reads the same checkpoint run, the same evaluation shard and the same
stand-in normalisation statistics, and splits recordings the same way. This module is the single
place where those choices are written down.

**The recording split.** A recording is assigned by $h = \operatorname{crc32}(\mathrm{guid}) \bmod 4$:
$h = 0$ is the *fit* set (closed-form solves), $h = 2$ is the *select* set (hyperparameters, edit
acceptance) and $h \in \{1, 3\}$ is the *test* set, which no edit was fitted or selected on.
Python's own ``hash`` is salted per process, so it is not used.

**The stand-in statistics.** The training ``stats.hdf5`` is not available locally, so the loader
normalises with statistics computed over the evaluation shard itself
(``assets/stats_standin_acidosis.hdf5``). Absolute numbers are therefore in stand-in units, and a
gain measured here is a gain on this shard under these statistics.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides,
)
from teb_vae.lag_attn_cfs.eval.probe import load_task, read_checkpoint  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402

RUN_DIR = REPO / "tmp" / "latest-results-2026-11-04"
CHECKPOINT = RUN_DIR / "lag-attn-trf-cfs-epoch=524.ckpt"
CONFIG = RUN_DIR / "resolved_config.yaml"
HISTORY = RUN_DIR / "metrics_history.csv"
DATA = RUN_DIR / "acidosis_no_cs.hdf5"
STATS = PKG / "assets" / "stats_standin_acidosis.hdf5"
OVERRIDE_TEMPLATE = PKG / "assets" / "eval_overrides_template.yaml"
RESULTS = PKG / "results"
GENERATED = RESULTS / "generated"

#: Number of kept ``fhr_st`` channels; they come first in the gathered target block.
N_ST = 32


def checkpoint_variant(tag: str) -> Path:
    """Path of an edited checkpoint written next to the original.

    Args:
        tag: The variant suffix, for example ``"surgery"`` or ``"v4"``.

    Returns:
        ``<run dir>/lag-attn-trf-cfs-epoch=524-<tag>.ckpt``.
    """
    return RUN_DIR / f"lag-attn-trf-cfs-epoch=524-{tag}.ckpt"


def guid_hash(guid: Any) -> int:
    r"""Split code of one recording, $\operatorname{crc32}(\mathrm{guid}) \bmod 4$.

    Args:
        guid: The recording GUID as the loader yields it (``str`` or ``bytes``).

    Returns:
        An integer in $\{0, 1, 2, 3\}$: 0 fit, 2 select, 1 and 3 test.
    """
    text = guid.decode() if isinstance(guid, bytes) else str(guid)
    return zlib.crc32(text.encode()) % 4


def write_overrides(name: str = "eval_overrides_local.yaml",
                    allowed_guids: Optional[Sequence[str]] = None,
                    data: Optional[Path] = None) -> Path:
    """Fill the override template with this machine's data and statistics paths.

    Args:
        name: File name inside ``results/generated``.
        allowed_guids: Optional GUID allow-list for the dataset filter; ``None`` keeps every
            recording.
        data: Shard to evaluate; ``None`` uses :data:`DATA`, the acidosis shard. Every shard is
            normalised with the same stand-in statistics, so features stay comparable across
            shards.

    Returns:
        Path of the written override file.
    """
    GENERATED.mkdir(parents=True, exist_ok=True)
    data = DATA if data is None else Path(data)
    text = OVERRIDE_TEMPLATE.read_text().replace("<DATA>", data.as_posix()).replace("<STATS>", STATS.as_posix())
    overrides = yaml.safe_load(text)
    if allowed_guids is not None:
        overrides["dataset_config"]["dataloader_config"]["dataset_kwargs"]["allowed_guids"] = list(allowed_guids)
    path = GENERATED / name
    path.write_text(yaml.safe_dump(overrides, sort_keys=False))
    return path


def setup(checkpoint: Path = CHECKPOINT, config: Path = CONFIG,
          overrides: Optional[Path] = None, device: str = "cuda:0") -> Tuple[torch.device, dict, Any]:
    """Load the configuration and the checkpoint through the evaluation pipeline's own loader.

    The loader is :func:`teb_vae.lag_attn_cfs.eval.probe.load_task`, which builds the model from
    the checkpoint's ``model_kwargs`` and loads the state dict strictly through
    ``train/graph_models_utils.py``.

    Args:
        checkpoint: Checkpoint to load.
        config: The run's resolved configuration.
        overrides: Evaluation override file; ``None`` writes and uses the default local one.
        device: Torch device string.

    Returns:
        ``(device, config, task)``. ``task.orig_model`` is the network.
    """
    dev = torch.device(device)
    overrides = write_overrides() if overrides is None else overrides
    cfg = merge_eval_overrides(load_config(str(config)), str(overrides))
    force_single_process_loader(cfg)
    task = load_task(str(checkpoint), dev, blob=read_checkpoint(str(checkpoint)), binding=TRF_CFS_BINDING)
    task.orig_model.eval()
    return dev, cfg, task


def batches(task: Any, cfg: dict, dev: torch.device, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Iterate the evaluation shard's batches on the device, in loader order.

    Args:
        task: The loaded task (for its batch transfer).
        cfg: The merged configuration.
        dev: Target device.
        limit: Stop after this many batches; ``None`` reads the whole shard.

    Yields:
        One batch dictionary per step.
    """
    for index, batch in enumerate(GraphDataModule(cfg).test_dataloader()):
        if limit is not None and index >= limit:
            break
        yield task.transfer_batch_to_device(batch, dev, 0)


def forward_parts(task: Any, model: Any, batch: Dict[str, Any]):
    """Run one forward and build the scored target and step mask exactly as training does.

    Args:
        task: The loaded task (input builder, target builder, quality weight).
        model: The network to run; usually ``task.orig_model``, possibly edited.
        batch: One batch on the model's device.

    Returns:
        ``(inputs, outputs, target, step_mask, anchors, anchor_valid)`` where ``target`` is
        $(B, A, H, C)$, ``step_mask`` is $(B, A, H)$ and ``anchors`` is $(B, A)$.
    """
    inputs = task._build_forward_inputs(batch)
    target_features, weight = task._build_raw_target(batch)
    weight = task._target_quality_weight(batch, weight)
    outputs = model(*inputs)
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    step_mask, _ = forecast_mask(model.scored_weight(weight), model.geometry,
                                 coverage_floor=model.coverage_floor, anchors=anchors,
                                 anchor_valid=anchor_valid)
    return inputs, outputs, target, step_mask.to(target.dtype), anchors, anchor_valid


def split_codes(batch: Dict[str, Any], dev: torch.device) -> torch.Tensor:
    """Split code per sample of a batch.

    Args:
        batch: One batch with a ``guid`` entry.
        dev: Device of the returned tensor.

    Returns:
        A ``(B,)`` long tensor of :func:`guid_hash` values.
    """
    return torch.tensor([guid_hash(g) for g in batch["guid"]], device=dev)
