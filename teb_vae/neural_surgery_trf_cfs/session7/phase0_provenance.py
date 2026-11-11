r"""Phase 0: strict load, training membership and the learned soft-mask parameters.

Three checks, written to ``out/phase0_provenance.json``:

1. **Strict load.** The checkpoint must align into the model that the resolved configuration
   builds. The script compares the checkpoint's ``model_kwargs`` with the ``VAE_model`` block key
   by key, then loads the state dict strictly.
2. **Membership.** Every local recording gets its cohort, its split code and its training
   membership label (see :mod:`s7common`).
3. **Soft-mask parameters.** The s6 rung added three zero-initialised parameters: the per-head
   key bias $b_m$ (``source_invalid_key_bias``), the gate weight $w_\rho$ on the invalid attention
   mass (``source_invalid_gate``) and the adapter's per-channel quality input $W_q$
   (``source_adapter.quality_proj``). The script reports their size against matching trained
   tensors. A value near zero means the soft mask did not learn.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass ``--device``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0"}


def compare_kwargs(blob: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the checkpoint's model kwargs with the configuration's ``VAE_model`` block.

    Args:
        blob: The checkpoint dictionary.
        config: The resolved configuration.

    Returns:
        ``{'n_shared', 'mismatch'}`` where ``mismatch`` maps a key to ``[checkpoint, config]``.
    """
    kwargs = blob["model_kwargs"]
    block = config["model_config"]["VAE_model"]
    shared = sorted(set(kwargs) & set(block))
    mismatch = {}
    for key in shared:
        left, right = kwargs[key], block[key]
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            same = list(left or []) == list(right or [])
        else:
            same = left == right
        if not same:
            mismatch[key] = [left, right]
    return {"n_shared": len(shared), "mismatch": mismatch}


def soft_mask_report(model: torch.nn.Module) -> Dict[str, Any]:
    r"""Summarise the three soft-mask parameters against trained reference tensors.

    The reference for $W_q$ is the adapter's value projection: both map the same $C_u$ channels
    to $d_{\mathrm{model}}$, so the ratio of their Frobenius norms says how strongly the flags
    drive the adapter relative to the values.

    Args:
        model: The loaded network.

    Returns:
        A dictionary of norms, per-head values and per-channel column norms.
    """
    adapter = model.source_adapter
    w_q = adapter.quality_proj.weight.detach().float().cpu()
    report: Dict[str, Any] = {
        "source_invalid_key_bias": model.source_invalid_key_bias.detach().cpu().tolist(),
        "source_invalid_gate_weight": model.source_invalid_gate.weight.detach().cpu().flatten().tolist(),
        "quality_proj_fro": float(w_q.norm()),
        "quality_proj_col_norm": w_q.norm(dim=0).tolist(),
        "quality_proj_abs_max": float(w_q.abs().max()),
    }
    for name, param in adapter.named_parameters():
        if param.dim() == 2 and tuple(param.shape) == tuple(w_q.shape) and "quality" not in name:
            report[f"reference_{name}_fro"] = float(param.detach().norm())
            report[f"reference_{name}_col_norm"] = param.detach().float().cpu().norm(dim=0).tolist()
    gate_head = getattr(model.posterior_head, "event_gate_head", None)
    if gate_head is not None:
        report["event_gate_head"] = {
            name: {"shape": list(p.shape), "fro": float(p.detach().norm()),
                   "values": p.detach().cpu().flatten().tolist() if p.numel() <= 16 else None}
            for name, p in gate_head.named_parameters()
        }
    return report


def main(device: str) -> Dict[str, Any]:
    """Run the three checks and write the report.

    Args:
        device: Torch device string.

    Returns:
        The report dictionary.
    """
    from teb_vae.lag_attn_cfs.eval.probe import read_checkpoint

    blob = read_checkpoint(str(S.CHECKPOINT))
    config = yaml.safe_load(S.CONFIG.read_text())
    report: Dict[str, Any] = {
        "checkpoint": str(S.CHECKPOINT),
        "checkpoint_keys": sorted(k for k in blob if k != "state_dict"),
        "epoch": blob.get("epoch"), "global_step": blob.get("global_step"),
        "kwargs_vs_config": compare_kwargs(blob, config),
    }
    _, _, task = S.load_model(device=device)       # strict: raises when the state dict misaligns
    model = task.orig_model
    report["strict_load"] = True
    report["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    report["soft_mask"] = soft_mask_report(model)
    report["logvar_clamp"] = list(blob["model_kwargs"].get("logvar_clamp", []))

    cohort = S.cohort_of_guids()
    rows = [{"guid": g, "cohort": c, "split_code": S.guid_hash(g), "role": S.split_role(S.guid_hash(g)),
             "membership": S.MEMBERSHIP[c]} for g, c in sorted(cohort.items())]
    counts: Dict[str, int] = {}
    for row in rows:
        key = f"{row['cohort']}/{row['role']}/{row['membership']}"
        counts[key] = counts.get(key, 0) + 1
    report["membership_counts"] = counts
    S.OUT.mkdir(parents=True, exist_ok=True)
    (S.OUT / "membership.json").write_text(json.dumps(rows, indent=1))
    (S.OUT / "phase0_provenance.json").write_text(json.dumps(report, indent=1, default=str))

    sm = report["soft_mask"]
    print("strict load: OK;", report["n_parameters"], "parameters; epoch", report["epoch"])
    print("kwargs mismatch:", report["kwargs_vs_config"]["mismatch"] or "none",
          f"({report['kwargs_vs_config']['n_shared']} shared keys)")
    print("key bias per head:", np.round(sm["source_invalid_key_bias"], 4))
    print("gate weight per head:", np.round(sm["source_invalid_gate_weight"], 4))
    print("quality_proj Frobenius:", round(sm["quality_proj_fro"], 4), "abs max", round(sm["quality_proj_abs_max"], 4))
    for key, value in sm.items():
        if key.startswith("reference_") and key.endswith("_fro"):
            print(key, round(value, 4))
    print("membership:", counts)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    main(parser.parse_args().device)
