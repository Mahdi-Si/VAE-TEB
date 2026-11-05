r"""Rebuild the rung-6 warm-start checkpoint from the epoch-524 checkpoint, with no data access.

:mod:`~teb_vae.neural_surgery_trf_cfs.surgery.init_s6` fits the warm start on local shards. A
training machine that holds only the epoch-524 checkpoint cannot rerun that fit, and a refit on
other data would give other numbers. This script needs no data. Only six tensors of the warm start
are not epoch-524 tensors, and the fit's exact values for them are committed in
``assets/s6init_fitted.npz``:

* ``decoder.persistence_lag_weight`` $(H, 5, C)$, the fitted extra AR-lag weights;
* ``target_ar_logit`` $(C,)$, the chosen $a_c$ with $\phi_c = 0.95\tanh(a_c)$;
* ``decoder.logvar_head.bias`` $(C,)$, the epoch-524 bias plus the chosen per-channel shift;
* ``source_adapter.quality_proj.weight``, ``source_invalid_key_bias`` and
  ``source_invalid_gate.weight``, the soft UA mask's terms, all zero.

Every other tensor is copied from the epoch-524 checkpoint unchanged. The model keywords are the
epoch-524 ones with the rung-6 overrides the fit stamped, which the asset also stores.

**Exactness is checked, not assumed.** The asset holds two SHA-256 digests over every tensor's
name, dtype, shape and bytes: one of the epoch-524 network state the fit started from, and one of
the warm-start state the fit wrote. The build refuses an input whose digest differs, and it
refuses to write an output whose digest differs. A file this script writes is therefore the
warm start the held-out numbers in the report describe, tensor for tensor. A final strict load
into the rung-6 network runs the training driver's own two checks.

Run from an IDE's Run button by editing :data:`RUN_ARGS`, or from the command line::

    python -m teb_vae.neural_surgery_trf_cfs.surgery.make_s6init

``--export`` writes the asset from a fitted warm start and is needed only where the fit ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: The committed fit: the six non-epoch-524 tensors, the two digests and the fit's report.
ASSET = _REPO / "teb_vae" / "neural_surgery_trf_cfs" / "assets" / "s6init_fitted.npz"
#: The run directory both checkpoints live in, relative to the repository root.
RUN_DIR = _REPO / "tmp" / "latest-results-2026-11-04"
#: The state-dict prefix the epoch-524 file and the warm start both use.
PREFIX = "model."
#: The tensors the warm start does not take from epoch 524.
FITTED_KEYS = (
    "decoder.persistence_lag_weight",
    "target_ar_logit",
    "decoder.logvar_head.bias",
    "source_adapter.quality_proj.weight",
    "source_invalid_key_bias",
    "source_invalid_gate.weight",
)

#: The IDE entry: edit these and press Run. ``export`` is ``False`` on the training machine.
RUN_ARGS: Dict[str, Any] = dict(
    source=str(RUN_DIR / "lag-attn-trf-cfs-epoch=524.ckpt"),
    output=str(RUN_DIR / "lag-attn-trf-cfs-epoch=524-s6init.ckpt"),
    export=False,
    fitted=str(RUN_DIR / "lag-attn-trf-cfs-epoch=524-s6init.ckpt"),
)


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over every tensor's name, dtype, shape and bytes, in sorted key order.

    Args:
        state: A state dict without prefix.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def network_state(blob: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Return a checkpoint's network state without the ``model.`` prefix.

    Args:
        blob: A loaded checkpoint holding ``state_dict``.

    Returns:
        The state dict of the network.

    Raises:
        RuntimeError: If no key carries the prefix.
    """
    state = {key[len(PREFIX):]: value for key, value in blob["state_dict"].items()
             if key.startswith(PREFIX)}
    if not state:
        raise RuntimeError(f"no state-dict key starts with {PREFIX!r}")
    return state


def export(source: Path, fitted: Path, asset: Path = ASSET) -> Path:
    """Write the asset from the epoch-524 checkpoint and a warm start ``init_s6`` fitted.

    Args:
        source: The epoch-524 checkpoint.
        fitted: The warm start written by :mod:`init_s6`.
        asset: Where to write the asset.

    Returns:
        The asset path.

    Raises:
        RuntimeError: If the warm start differs from epoch 524 anywhere outside
            :data:`FITTED_KEYS`, which would leave the asset unable to rebuild it.
    """
    old = network_state(torch.load(str(source), map_location="cpu", weights_only=False))
    blob = torch.load(str(fitted), map_location="cpu", weights_only=False)
    new = network_state(blob)
    outside = sorted(set(new) - set(old) - set(FITTED_KEYS))
    changed = sorted(key for key in old if key not in FITTED_KEYS
                     and (key not in new or not torch.equal(new[key], old[key])))
    if outside or changed:
        raise RuntimeError(f"the warm start differs outside the fitted keys: {outside + changed}")
    source_kwargs = torch.load(str(source), map_location="cpu", weights_only=False)["model_kwargs"]
    overrides = {key: value for key, value in blob["model_kwargs"].items()
                 if source_kwargs.get(key, "<absent>") != value}
    asset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        asset,
        **{key: new[key].numpy() for key in FITTED_KEYS},
        source_digest=np.array(state_digest(old)),
        output_digest=np.array(state_digest(new)),
        report=np.array(json.dumps(blob.get("s6_init_report", {}))),
        overrides=np.array(json.dumps(overrides)),
    )
    return asset


def build(source: Path, output: Path, asset: Path = ASSET) -> Path:
    """Rebuild the warm start from the epoch-524 checkpoint and the asset, and verify it.

    Args:
        source: The epoch-524 checkpoint on this machine.
        output: Where to write the warm start.
        asset: The committed fit.

    Returns:
        The written path.

    Raises:
        RuntimeError: If the source is not the epoch-524 state the fit started from, if the
            rebuilt state differs from the fitted one, or if the strict load fails.
    """
    with np.load(asset, allow_pickle=False) as stored:
        fitted = {key: torch.from_numpy(stored[key].copy()) for key in FITTED_KEYS}
        source_digest = str(stored["source_digest"])
        output_digest = str(stored["output_digest"])
        report = json.loads(str(stored["report"]))
        overrides = json.loads(str(stored["overrides"]))

    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    old = network_state(blob)
    if state_digest(old) != source_digest:
        raise RuntimeError(
            f"{source} is not the epoch-524 checkpoint the warm start was fitted from (its "
            f"network digest differs from the one in {asset.name}); refusing to build"
        )

    state = dict(old)
    state.update(fitted)
    if state_digest(state) != output_digest:
        raise RuntimeError("the rebuilt warm start differs from the fitted one; refusing to write")

    kwargs = dict(blob["model_kwargs"], **overrides)
    out = {
        "state_dict": {f"{PREFIX}{key}": value for key, value in state.items()},
        "model_class": blob.get("model_class"),
        "model_kwargs": kwargs,
        "causal_representation": blob.get("causal_representation"),
        "s6_init_report": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))

    # The training driver's own two checks, against the file as written.
    written = torch.load(str(output), map_location="cpu", weights_only=False)
    check_model_class(written, SeqVaeLagAttnTrfCfs.__name__)
    if state_digest(network_state(written)) != output_digest:
        raise RuntimeError(f"{output} does not read back as the fitted warm start")
    if load_checkpoint_strict(model=SeqVaeLagAttnTrfCfs(**kwargs), checkpoint=str(output)) is None:
        raise RuntimeError(f"{output} does not load strictly into the rung-6 network")
    return output


def main(args: Dict[str, Any]) -> Path:
    """Export the asset or build the warm start, as ``args['export']`` says.

    Args:
        args: :data:`RUN_ARGS` or the command-line equivalent.

    Returns:
        The written path.
    """
    if args["export"]:
        path = export(Path(args["source"]), Path(args["fitted"]))
        print(f"wrote {path}")
        return path
    path = build(Path(args["source"]), Path(args["output"]))
    print(f"wrote {path}; digest verified against {ASSET.name}; strict load into "
          f"{SeqVaeLagAttnTrfCfs.__name__} passed")
    return path


def _parse(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Read the command line into a copy of :data:`RUN_ARGS`.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        The argument dictionary.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=RUN_ARGS["source"])
    parser.add_argument("--output", default=RUN_ARGS["output"])
    parser.add_argument("--export", action="store_true", default=RUN_ARGS["export"])
    parser.add_argument("--fitted", default=RUN_ARGS["fitted"])
    return dict(RUN_ARGS, **vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(_parse() if len(sys.argv) > 1 else dict(RUN_ARGS))
