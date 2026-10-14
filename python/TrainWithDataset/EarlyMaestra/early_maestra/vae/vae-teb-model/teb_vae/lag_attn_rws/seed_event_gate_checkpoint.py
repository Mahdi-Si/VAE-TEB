r"""Seed the posterior event gate into a checkpoint trained without it, for a warm start.

A model built with ``posterior_event_gate: true`` carries two tensors a pre-gate checkpoint lacks,
the gate's linear weight $(1, M d_h)$ and bias $(1,)$, and the driver's warm start goes through
:func:`~train.graph_models_utils.load_checkpoint_strict`, which refuses a checkpoint with any
missing key. This script writes a copy of the checkpoint with the two tensors added exactly as the
constructor seeds them -- zero weight, bias :data:`~teb_vae.lag_attn.nets.heads.EVENT_GATE_BIAS_INIT`
-- so the gate starts open and input-independent at $g_t = \sigma(b) \approx 0.88$, and the loaded
residual is the old model's scaled by that constant on the first steps. Nothing else in the
checkpoint moves. The stamped ``model_kwargs`` gain ``posterior_event_gate: True`` so the copy
describes the model it now fits.

**The copy is a warm-start file, not a resume file.** The driver's ``core_model_checkpoint`` loads
weights only, which is all the copy changes. A Lightning ``ckpt_path`` resume would also restore the
optimiser states, whose parameter groups still count the old parameters and hold no state for the
two new tensors, and that load would raise. Resume is not a path this family's driver offers.

The gate's input width is read off the checkpoint's own tensors rather than off any config: the
head-structured posterior normalises each of $M$ head summaries of width $d_h$, so $d_h$ is the
width of ``posterior_head.a_head_norm.weight`` and $M$ the number of ``delta_mu_head`` entries.
The key prefix the checkpoint uses (``model.``, or none) is read off those same keys, so the new
tensors land beside them whatever the wrapper that saved the file.

Run from an IDE's Run button by editing :data:`RUN_ARGS`, or from the command line::

    python -m teb_vae.lag_attn_rws.seed_event_gate_checkpoint --checkpoint <run>/model_checkpoints/<name>.ckpt
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from teb_vae.lag_attn.nets.heads import EVENT_GATE_BIAS_INIT

#: Where the gate lives, relative to the model's root, as the head names its submodules.
GATE_WEIGHT_KEY = "posterior_head.event_gate_head.weight"
GATE_BIAS_KEY = "posterior_head.event_gate_head.bias"
_HEAD_NORM_KEY = "posterior_head.a_head_norm.weight"
_FLAT_NORM_KEY = "posterior_head.a_norm.weight"
_DELTA_MU_PATTERN = re.compile(r"posterior_head\.delta_mu_head\.(\d+)\.weight$")

#: The IDE entry: edit these and press Run. ``output`` ``None`` writes ``<stem>.event_gate.ckpt``
#: beside the input; ``bias`` ``None`` uses the constructor's seed.
RUN_ARGS: Dict[str, Any] = dict(
    checkpoint="REPOINT_ME/model_checkpoints/last.ckpt",
    output=None,
    bias=None,
)


def _state_dict_of(blob: Any) -> Dict[str, torch.Tensor]:
    """Return the blob's state dict: the ``state_dict`` entry of a Lightning file, else the blob."""
    if isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
        return blob["state_dict"]
    if isinstance(blob, dict) and blob and all(isinstance(v, torch.Tensor) for v in blob.values()):
        return blob
    raise ValueError("the checkpoint carries no state_dict and is not a raw state dict itself")


def seed_event_gate(blob: Any, *, bias: Optional[float] = None) -> Any:
    r"""Return a copy of ``blob`` whose state dict carries the seeded event gate.

    Args:
        blob: A loaded checkpoint -- a Lightning dict with a ``state_dict`` entry, or a raw state
            dict -- of a model whose posterior head has no event gate.
        bias: The gate bias $b$ to seed; ``None`` uses the constructor's
            :data:`~teb_vae.lag_attn.nets.heads.EVENT_GATE_BIAS_INIT`.

    Returns:
        A shallow copy of the blob with the two gate tensors added under the checkpoint's own key
        prefix and, when ``model_kwargs`` is stamped, ``posterior_event_gate`` set to ``True``.

    Raises:
        ValueError: If the state dict has no posterior head, already carries the gate, or holds a
            head-structured posterior whose head count cannot be read.
    """
    state = _state_dict_of(blob)
    bias_value = EVENT_GATE_BIAS_INIT if bias is None else float(bias)

    prefix: Optional[str] = None
    gate_in: Optional[int] = None
    reference: Optional[torch.Tensor] = None
    for key, tensor in state.items():
        if key.endswith(_HEAD_NORM_KEY):
            prefix = key[: -len(_HEAD_NORM_KEY)]
            d_head = int(tensor.shape[0])
            heads = {
                int(match.group(1))
                for name in state
                if name.startswith(prefix) and (match := _DELTA_MU_PATTERN.search(name))
            }
            if not heads:
                raise ValueError("head-structured posterior without delta_mu_head entries")
            gate_in = d_head * len(heads)
            reference = tensor
            break
        if key.endswith(_FLAT_NORM_KEY):
            prefix = key[: -len(_FLAT_NORM_KEY)]
            gate_in = int(tensor.shape[0])
            reference = tensor
            break
    if prefix is None or gate_in is None or reference is None:
        raise ValueError(
            "no posterior head found in the state dict; expected a key ending in "
            f"'{_HEAD_NORM_KEY}' or '{_FLAT_NORM_KEY}'"
        )
    weight_key, bias_key = prefix + GATE_WEIGHT_KEY, prefix + GATE_BIAS_KEY
    if weight_key in state or bias_key in state:
        raise ValueError("the checkpoint already carries the event gate; nothing to seed")

    seeded_state = OrderedDict(state)
    seeded_state[weight_key] = torch.zeros((1, gate_in), dtype=reference.dtype)
    seeded_state[bias_key] = torch.full((1,), bias_value, dtype=reference.dtype)

    if isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
        seeded = dict(blob)
        seeded["state_dict"] = seeded_state
        if isinstance(seeded.get("model_kwargs"), dict):
            seeded["model_kwargs"] = dict(seeded["model_kwargs"], posterior_event_gate=True)
        return seeded
    return seeded_state


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Command-line entry; with no arguments, :data:`RUN_ARGS` is the configuration.

    Args:
        argv: Command-line arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        The path the seeded copy was written to.

    Raises:
        SystemExit: If the checkpoint does not exist or the output would overwrite it.
        ValueError: From :func:`seed_event_gate`, on a checkpoint without a posterior head or
            with the gate already present.
    """
    parser = argparse.ArgumentParser(
        description="Seed the posterior event gate into a checkpoint trained without it."
    )
    parser.add_argument("--checkpoint", default=RUN_ARGS["checkpoint"])
    parser.add_argument("--output", default=RUN_ARGS["output"])
    parser.add_argument("--bias", type=float, default=RUN_ARGS["bias"])
    args = parser.parse_args(argv)

    source = Path(args.checkpoint)
    if not source.is_file():
        raise SystemExit(f"checkpoint not found: {source} (edit RUN_ARGS or pass --checkpoint)")
    output = Path(args.output) if args.output else source.with_name(f"{source.stem}.event_gate{source.suffix}")
    if output.resolve() == source.resolve():
        raise SystemExit("refusing to overwrite the source checkpoint; pass a different --output")

    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    seeded = seed_event_gate(blob, bias=args.bias)
    torch.save(seeded, str(output))
    state = _state_dict_of(seeded)
    key = next(k for k in state if k.endswith(GATE_WEIGHT_KEY))
    print(
        f"seeded {key} {tuple(state[key].shape)} and its bias at "
        f"{EVENT_GATE_BIAS_INIT if args.bias is None else args.bias}; wrote {output}"
    )
    return output


if __name__ == "__main__":
    main(sys.argv[1:])
