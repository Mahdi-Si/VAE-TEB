r"""Build the rung-8 warm start: a rung-7 checkpoint with the AR skip widened from 6 to 38 lags.

The decoder's persistence skip adds $\sum_p w_{\tau,p,c}\, y_{t-p,c}$ per channel and horizon
step. Rung 7 reads 6 lags (20 s). The s7 audit (``tmp/audit_s7/REPORT.md``, test split of
``split_last4h_gap10``) found the scattering mean forecast is the weak point, and an own-channel
direct AR fitted on the train split improves with history: scattering MSE 0.604 at 6 lags, 0.575
at 16, 0.547 at 38 (148 s), against 0.567 for the epoch-443 VAE on the same cells. Phase does not
move (0.664 -> 0.663).

The extra weights ``decoder.persistence_lag_weight[:, 5:, :]`` are written as zeros, so the warm
start is the rung-7 network exactly: same forecast, same NLL, same KL at step 0. Nothing is fitted
and no data is read. The optimizer state is dropped (one tensor changed shape); the config's
``core_model_checkpoint`` loads weights only.

From the repo root::

    python -m teb_vae.neural_surgery_trf_cfs.surgery.make_s8init --source <run>/lag-attn-trf-cfs-epoch=443.ckpt

``--output`` is optional and defaults to ``<source stem>-s8init.ckpt`` beside the source. From an
IDE's Run button, with no command line, :data:`RUN_ARGS` is used instead; a flag given on the
command line overrides its entry.
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
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

KEY = "decoder.persistence_lag_weight"
#: The IDE entry: set ``source`` and press Run. ``output`` None writes ``<source stem>-s8init.ckpt``.
RUN_ARGS: Dict[str, Any] = dict(
    source="SET_ME/lag-attn-trf-cfs-epoch=443.ckpt",
    output=None,
    lags=38,
)


def build(source: Path, output: Path, lags: int) -> Path:
    """Zero-pad the AR skip to ``lags`` lags, write the warm start, and check it is exact."""
    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    check_model_class(blob, SeqVaeLagAttnTrfCfs.__name__)
    old_lags = int(blob["model_kwargs"]["persistence_lags"])
    if lags <= old_lags:
        raise ValueError(f"lags={lags} must exceed the checkpoint's persistence_lags={old_lags}")
    # Lightning's checkpoint holds the network twice (model. and the compiled _orig_model.); keep one.
    state = {k: v for k, v in blob["state_dict"].items() if k.startswith("model.")}
    w = state[f"model.{KEY}"]
    state[f"model.{KEY}"] = torch.cat([w, w.new_zeros(w.shape[0], lags - old_lags, w.shape[2])], 1)
    kwargs = dict(blob["model_kwargs"], persistence_lags=lags)
    # Carried so the evaluation can score the warm start; the training driver reads only the weights.
    hparams = dict(blob.get("hyper_parameters") or {})
    if "model_kwargs" in hparams:
        hparams["model_kwargs"] = dict(hparams["model_kwargs"], persistence_lags=lags)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "model_class": blob.get("model_class"), "model_kwargs": kwargs,
                "hyper_parameters": hparams, "causal_representation": blob.get("causal_representation"),
                "s8_init": {"source": str(source), "epoch": blob.get("epoch"), "lags": [old_lags, lags]}},
               str(output))

    # The driver's strict load, then exactness: the extra lags contribute nothing at step 0.
    model = load_checkpoint_strict(model=SeqVaeLagAttnTrfCfs(**kwargs), checkpoint=str(output))
    if model is None:
        raise RuntimeError(f"{output} does not load strictly into the rung-8 network")
    model.eval()
    c = w.shape[2]
    torch.manual_seed(0)
    z = torch.randn(2, 3, int(kwargs["d_z"]))
    pers = torch.randn(2, 3, lags * c)
    cut = pers.clone()
    cut[..., old_lags * c:] = 0
    with torch.no_grad():
        mu, lv = model.decoder(z, persistence=pers)
        mu0, lv0 = model.decoder(z, persistence=cut)
    if not (torch.equal(mu, mu0) and torch.equal(lv, lv0)):
        raise RuntimeError("the padded lags change the decoder output; refusing the warm start")
    return output


def main(args: Dict[str, Any]) -> Path:
    if str(args["source"]).startswith("SET_ME"):
        raise ValueError("set --source on the command line, or source in RUN_ARGS")
    source = Path(args["source"])
    output = Path(args["output"]) if args["output"] else source.with_name(f"{source.stem}-s8init.ckpt")
    path = build(source, output, int(args["lags"]))
    print(f"wrote {path}; strict load passed; decoder output unchanged by the {args['lags']}-lag padding")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=RUN_ARGS["source"],
                   help="the trained rung-7 checkpoint to widen (e.g. lag-attn-trf-cfs-epoch=443.ckpt)")
    p.add_argument("--output", default=RUN_ARGS["output"],
                   help="where to write the warm start [default: <source stem>-s8init.ckpt beside the source]")
    p.add_argument("--lags", type=int, default=RUN_ARGS["lags"],
                   help="persistence lags of the warm start; must match persistence_lags in the s8 config [38]")
    main(vars(p.parse_args()))
