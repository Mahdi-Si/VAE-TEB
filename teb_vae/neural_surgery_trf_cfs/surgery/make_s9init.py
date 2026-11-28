r"""Build the warm starts of the IRF rungs from a lag-attention checkpoint (rung 8).

The IRF head (``teb_vae/lag_attn_cfs/nets/irf_source.py``, method in
``teb_vae/coupling_irf/DESIGN.md``) has three parameters a rung-8 checkpoint does not hold. Three
modes, one per config:

``readout`` (``configs/sweep_next_s9_irf_readout.yaml``, writes ``<stem>-s9readout-init.ckpt``)
    The head beside the lag attention. Only adds the head's parameters at their initial values.
``s10`` (``configs/sweep_next_s10_specific_source.yaml``, writes ``<stem>-s10init.ckpt``)
    The same, plus the event gate's two evidence weights at zero.
``source`` (``configs/sweep_alt_irf_source_only.yaml``, writes ``<stem>-s9init.ckpt``)
    The head in place of the lag attention: drops the keys of the query projection, the lag
    attention, the posterior head and the soft UA mask, and adds the head's parameters.

Every mode sets the model keywords and the task weights of its config (:data:`MODES`), reloads
the result strictly into the new network, and checks on a batch with the same inputs and the same
noise that the forecasts are the source network's, bit for bit: the base and the full forecast in
the first two modes, the base forecast in the third. Nothing is fitted and no data is read. The
optimizer state is dropped; the config's ``core_model_checkpoint`` loads weights only.

From the repo root::

    python -m teb_vae.neural_surgery_trf_cfs.surgery.make_s9init --mode readout --source <run>/lag-attn-trf-cfs-epoch=369.ckpt

``--output`` is optional and defaults to the mode's file name beside the source. From an IDE's Run
button, with no command line, :data:`RUN_ARGS` is used instead; a flag given on the command line
overrides its entry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict, network_state_dict  # noqa: E402

#: The head's keywords, as in the three configs: the calibrated gate, the FHR level and the fast
#: scattering channels.
_HEAD: Dict[str, Any] = dict(irf_gate_k=4.45, irf_context_steps=0, irf_channels=list(range(21)))
#: The head's task weights, as in the three configs.
_HEAD_TASK: Dict[str, Any] = dict(lambda_source_expl=1.0, lambda_source_wrong=1.0, irf_lr_scale=30.0)
#: Per mode: the output suffix, the key prefixes dropped, the model keywords and the task weights
#: set, and the forward outputs that must not move.
MODES: Dict[str, Dict[str, Any]] = {
    "readout": dict(
        suffix="-s9readout-init",
        dropped=(),
        model_kwargs=dict(_HEAD, irf_readout=True),
        task_hparams=dict(_HEAD_TASK, lambda_source_irf=1.0),
        unchanged=("mu_base", "logvar_base", "mu_full", "logvar_full"),
    ),
    "s10": dict(
        suffix="-s10init",
        dropped=(),
        model_kwargs=dict(_HEAD, irf_readout=True, irf_gate_evidence=True),
        task_hparams=dict(_HEAD_TASK, lambda_source_irf=1.0, lambda_full_wrong=1.0,
                          event_weight_floor=None, event_weight_ua_valid=False,
                          event_weight_ua_tolerance_s=None),
        unchanged=("mu_base", "logvar_base", "mu_full", "logvar_full"),
    ),
    "source": dict(
        suffix="-s9init",
        dropped=("query_proj.", "lag_attn.", "posterior_head.", "source_invalid_key_bias",
                 "source_invalid_gate.", "source_adapter.quality_proj."),
        model_kwargs=dict(_HEAD, source_path="irf", posterior_event_gate=False,
                          source_quality_mask=False, source_mask_mode="hard",
                          source_running_gap_keys="keep"),
        task_hparams=dict(_HEAD_TASK, lambda_event_gate=0.0, lambda_source_null=0.0,
                          event_weight_floor=None, event_weight_ua_valid=False,
                          event_weight_ua_tolerance_s=None, source_valid_tolerance_s=None),
        unchanged=("mu_base", "logvar_base"),
    ),
}
#: The IDE entry: set ``source`` and press Run. ``output`` None writes the mode's file name.
RUN_ARGS: Dict[str, Any] = dict(
    source="SET_ME/lag-attn-trf-cfs-epoch=369.ckpt",
    output=None,
    mode="readout",
)


def _forecast_difference(old: SeqVaeLagAttnTrfCfs, new: SeqVaeLagAttnTrfCfs, keys) -> float:
    """Largest difference of the two networks' forward outputs ``keys`` on one seeded batch, dense anchors."""
    batch, steps, split = 2, int(old.sequence_length), int(old.TARGET_BLOCK_SPLIT)
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(batch, steps, split, generator=generator)
    y_ph = torch.randn(batch, steps, int(old.c_y) - split, generator=generator)
    u = torch.randn(batch, steps, int(old.c_u), generator=generator)
    valid = torch.ones(batch, steps, dtype=torch.bool)
    head = dict(ua_level=torch.randn(batch, steps, generator=generator), ua_valid=valid, target_step_valid=valid)

    def masks(model):  # the two mask inputs a model built with the source-quality mask takes
        if not model.source_quality_mask:
            return ()
        removed = torch.zeros(batch, steps, int(model.max_lag) + 1, dtype=torch.bool)
        return (valid[..., None].expand(-1, -1, int(model.c_u)),) + (
            (removed,) if model.source_running_gap_keys == "remove" else ())

    old.eval(), new.eval()
    with torch.no_grad():
        torch.manual_seed(0)  # the same draw of the latent noise in both forwards
        out_old = old(y_st, y_ph, u, 0, 1, *masks(old))
        torch.manual_seed(0)
        out_new = new(y_st, y_ph, u, 0, 1, *masks(new), **head)
    return max(float((out_old[key] - out_new[key]).abs().max()) for key in keys)


def build(source: Path, output: Path, mode: str) -> Path:
    """Write the warm start of ``mode``, reload it strictly, and check the forecasts are unchanged."""
    spec = MODES[mode]
    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    check_model_class(blob, SeqVaeLagAttnTrfCfs.__name__)
    stored = blob["model_kwargs"]
    if stored.get("source_path", "lag_attn") != "lag_attn" or stored.get("irf_readout", False):
        raise ValueError(f"{source} already holds an IRF head; give the rung-8 checkpoint")
    # One copy of the network, whatever the wrapper (model., model._orig_mod., _orig_model.).
    old_state = dict(network_state_dict(blob))
    old = load_checkpoint_strict(model=SeqVaeLagAttnTrfCfs(**stored), checkpoint=old_state)
    if old is None:
        raise RuntimeError(f"{source} does not load strictly into the network of its own model_kwargs")

    kwargs = dict(stored, **spec["model_kwargs"])
    dropped = sorted(k for k in old_state if spec["dropped"] and k.startswith(spec["dropped"]))
    state = {k: v for k, v in old_state.items() if k not in dropped}
    # What the new network holds beyond that, at its initial values: the head, and in mode s10 the
    # zero evidence weights.
    added = {k: v.detach().clone() for k, v in SeqVaeLagAttnTrfCfs(**kwargs).state_dict().items()
             if k not in state}
    state.update(added)
    # Carried so a task can be rebuilt from the warm start; the training driver reads only the weights.
    hparams = dict(blob.get("hyper_parameters") or {}, **spec["task_hparams"])
    if "model_kwargs" in hparams:
        hparams["model_kwargs"] = dict(kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()},
                "model_class": blob.get("model_class"), "model_kwargs": kwargs,
                "hyper_parameters": hparams, "causal_representation": blob.get("causal_representation"),
                "s9_init": {"source": str(source), "epoch": blob.get("epoch"), "mode": mode,
                            "dropped": dropped, "added": sorted(added)}},
               str(output))

    by_module: Dict[str, int] = {}
    for key in dropped:
        prefix = next(p for p in spec["dropped"] if key.startswith(p)).rstrip(".")
        by_module[prefix] = by_module.get(prefix, 0) + 1
    print(f"mode {mode}: dropped {len(dropped)} keys" + (": " if dropped else "")
          + ", ".join(f"{name} ({n})" for name, n in by_module.items()))
    print("added: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in added.items()))
    print("model kwargs set: " + ", ".join(f"{k}={v!r}" for k, v in spec["model_kwargs"].items()))
    print("task weights set: " + ", ".join(f"{k}={v!r}" for k, v in spec["task_hparams"].items()))

    # The driver's strict load, then exactness of what the mode must not move.
    new = load_checkpoint_strict(model=SeqVaeLagAttnTrfCfs(**kwargs), checkpoint=str(output))
    if new is None:
        raise RuntimeError(f"{output} does not load strictly into the new network")
    if set(new.state_dict()) != set(state):
        raise RuntimeError("the new network and the warm start hold different keys")
    difference = _forecast_difference(old, new, spec["unchanged"])
    if difference != 0.0:
        raise RuntimeError(f"{spec['unchanged']} moved by {difference}; refusing the warm start")
    print(f"strict load passed ({len(state)} keys); {', '.join(spec['unchanged'])} identical to the "
          f"source (max |difference| = {difference})")
    return output


def main(args: Dict[str, Any]) -> Path:
    if str(args["source"]).startswith("SET_ME"):
        raise ValueError("set --source on the command line, or source in RUN_ARGS")
    source = Path(args["source"])
    suffix = MODES[args["mode"]]["suffix"]
    output = Path(args["output"]) if args["output"] else source.with_name(f"{source.stem}{suffix}.ckpt")
    path = build(source, output, args["mode"])
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=RUN_ARGS["source"],
                   help="the trained lag-attention checkpoint (e.g. lag-attn-trf-cfs-epoch=369.ckpt)")
    p.add_argument("--output", default=RUN_ARGS["output"],
                   help="where to write the warm start [default: the mode's file name beside the source]")
    p.add_argument("--mode", default=RUN_ARGS["mode"], choices=sorted(MODES),
                   help="which rung's warm start to build [readout]")
    main(vars(p.parse_args()))
