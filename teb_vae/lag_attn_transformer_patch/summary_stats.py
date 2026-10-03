"""Print the target standardization constants for the per-patch ``[level, variability]`` summaries.

Prints ``target_summary_loc``, ``target_summary_scale`` and ``variability_eps`` as YAML, ready to
paste under ``model_config.VAE_model``. Run it on the production training shards, through the same
config (so the same loader, stats file, ``normalize_fields`` and ``trim_minutes``) as training.

Recordings are weighted equally: a long recording contributes many segments, and pooling tokens
would let a few long recordings set the scale every recording is scored in.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

#: Repository root: ``teb_vae/lag_attn_transformer_patch/summary_stats.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Launched as a script, Python puts this directory on sys.path instead of the repository root.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_transformer_patch.nets.patching import (  # noqa: E402
    patch_summaries,
    patchify,
)
from train.data_module import GraphDataModule  # noqa: E402

#: Monitor resolution of the FHR trace, bpm.
FHR_RESOLUTION_BPM = 0.25

def build_config(
    config: str, train_datasets: Optional[List[str]], stat_path: Optional[str]
) -> Dict[str, Any]:
    """Load the training config and apply the shard/stats overrides.

    Args:
        config: Training config path (its ``base:`` chain is resolved).
        train_datasets: Replaces ``dataset_config.vae_train_datasets`` when given.
        stat_path: Replaces ``dataset_config.stat_path`` when given.

    Returns:
        A config mapping :class:`GraphDataModule` accepts.
    """
    cfg = load_config(config)
    dataset_config = cfg.setdefault("dataset_config", {})
    if train_datasets:
        dataset_config["vae_train_datasets"] = list(train_datasets)
    if stat_path:
        dataset_config["stat_path"] = stat_path
    return cfg


@torch.no_grad()
def summary_stats(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Recording-weighted location and scale of the valid-token summaries.

    Args:
        cfg: Config mapping with a ``dataset_config`` block.

    Returns:
        ``target_summary_loc``, ``target_summary_scale``, ``variability_eps`` and the counts.

    Raises:
        ValueError: If the loader does not z-score ``fhr``, or no token is valid.
    """
    loader = GraphDataModule(cfg).train_dataloader()
    dataset = loader.dataset
    if not dataset.normalization_enabled or (
        dataset.normalize_fields is not None and "fhr" not in dataset.normalize_fields
    ):
        raise ValueError("fhr is not normalized by this loader; check stat_path / normalize_fields")
    # Epsilon first: the variability summary uses it, in z-scored units.
    eps = FHR_RESOLUTION_BPM / float(dataset.normalization_stats["fhr"]["std"])

    per_recording: Dict[str, torch.Tensor] = {}  # guid -> [n, sum(2), sum_sq(2)]
    for batch in loader:
        fhr, weight = batch["fhr"], batch["weight"]
        r = fhr.shape[-1] // weight.shape[-1]
        x = patchify(fhr, weight, raw_per_step=r, validity="fhr_weight")
        valid = x[..., -1] + 1.0
        s = patch_summaries(x[..., :r], valid, eps=eps)
        m = (valid > 0.5).double().unsqueeze(-1)
        s = s.double() * m
        rows = torch.cat((m.sum(1), s.sum(1), s.square().sum(1)), dim=-1)
        for guid, row in zip(batch["guid"], rows):
            per_recording[guid] = per_recording.get(guid, 0) + row

    rows = [row for row in per_recording.values() if row[0] > 0]
    if not rows:
        raise ValueError("no valid token in the training shards")
    acc = torch.stack(rows)
    n = acc[:, :1]
    loc = (acc[:, 1:3] / n).mean(0)
    scale = ((acc[:, 3:5] / n).mean(0) - loc.square()).sqrt()
    return {
        "target_summary_loc": loc.tolist(),
        "target_summary_scale": scale.tolist(),
        "variability_eps": eps,
        "n_recordings": int(acc.shape[0]),
        "n_tokens": int(n.sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    """The command line, every default ``None`` so :data:`RUN_ARGS` stays reachable."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="Training config whose loader settings are used.")
    parser.add_argument("--train-datasets", nargs="+", help="Override vae_train_datasets.")
    parser.add_argument("--stat-path", help="Override dataset_config.stat_path.")
    return parser


#: Values for arguments absent from the command line (an IDE's Run button), keyed by argparse dest.
RUN_ARGS: Dict[str, Any] = {
    "config": "teb_vae/lag_attn_transformer_patch/configs/default.yaml",
    "train_datasets": None,
    "stat_path": None,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS`, then print the YAML block."""
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    os.chdir(_REPO_ROOT)  # config and shard paths are repo-root-relative
    stats = summary_stats(build_config(**values))
    counts = {k: stats.pop(k) for k in ("n_recordings", "n_tokens")}
    print(f"# {counts['n_recordings']} recordings, {counts['n_tokens']} valid tokens")
    print(yaml.safe_dump(stats, default_flow_style=None, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
