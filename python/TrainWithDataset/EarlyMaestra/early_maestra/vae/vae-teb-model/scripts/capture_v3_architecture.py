"""Serialise the architecture of ``SeqVaeLagAttnV3`` to a JSON fixture.

``teb_vae.lag_attn`` is a rebuild of the model at
``model/vae_teb_prediction/model/vae_teb_lag_attn_v3.py``, flattened out of a three-level
inheritance chain. The rebuild is not checkpoint-compatible and is not asserted to be
bit-identical, so the thing that has to be proved is *structural*: the same modules, the same
shapes, the same parameter count, the same forward and loss contracts.

This script captures that structure from the original. ``teb_vae/lag_attn/tests/
test_architecture_snapshot.py`` asserts the rebuild against it through an explicit map of the
intentional deviations, which is what turns "the flatten changed nothing it should not have" from
a claim into a test.

What is recorded:

* ``total_params`` / ``trainable_params`` -- ``numel()`` sums. Catches a dropped head, a wrong
  channel count, a missed projection.
* ``state_dict`` -- every key mapped to its shape. Catches a renamed or restructured module.
* ``forward_keys`` -- the forward contract, each key mapped to its shape (``null`` for a key whose
  value is not a tensor).
* ``loss_keys`` -- the loss contract.
* ``n_causalized_norms`` -- how many ``GroupNorm`` modules the causal rewrite replaced.

What it cannot catch: a transposed weight or a reordered residual. Both preserve every shape and
count. That is an accepted limit -- the model is being retrained, not weight-transferred.

This script is kept rather than deleted after use. The original is forked, not moved, so it stays
importable indefinitely; keeping the capture reproducible is what leaves the door open to a
stronger golden-tensor comparison later, should a retrain ever diverge from a published result.

Run from the repository root::

    .venv/Scripts/python.exe scripts/capture_v3_architecture.py \\
        --out teb_vae/lag_attn/tests/fixtures/v3_architecture.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# scripts/capture_v3_architecture.py -> parents[0]=scripts, [1]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (  # noqa: E402
    CausalGroupNorm,
    SeqVaeLagAttnV3,
)

# The construction kwargs are duplicated here rather than imported from the original test suite on
# purpose: this script must keep capturing the *original* signature even after the rebuilt model
# drops `posterior_logvar` and `logvar_bound` as constructor arguments. Sharing the fixture would
# couple the frozen record to a moving definition.
TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    warmup_period=2,
    c_y=87,
    c_u=101,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    dropout=0.0,
)

# The flag set the original's own test suite called "prod". `posterior_logvar` and
# `logvar_bound` are pinned to the values the shipped config runs; in the rebuild they are the
# only behaviour and stop being constructor arguments at all.
#
# Note what this does NOT turn on -- see SHIPPED_KWARGS below. It is a misnomer inherited from
# the original suite, kept under its original name because that is what the fixture records.
PROD_KWARGS: Dict[str, Any] = dict(
    TINY_KWARGS,
    causal_norm=True,
    posterior_logvar="residual",
    logvar_bound="smooth",
    kld_support="anchor",
    lag_bias_init="alibi_decay",
    lambda_perm=0.1,
    perm_every_n_batches=2,
    freeze_unused_attn_proj=True,
)

# What config_lag_attn_v3.yaml actually ships, at the tiny geometry.
#
# PROD_KWARGS leaves four shipped flags at their constructor defaults, and each one gates real
# modules: `head_structured_latent` swaps the flat posterior for a per-head one (an extra norm
# plus two ModuleLists of per-group heads), `horizon_film` adds the FiLM generator,
# `horizon_depth: 3` adds a refine block, and `encoder_extra_dilations` adds two conv blocks and
# their skip norms to each encoder. A snapshot captured without them therefore records none of
# the structure production runs, and the equivalence assertion silently covers a model nobody
# trains. `freeze_unused_attn_proj` also only bites here, since it requires head structure.
SHIPPED_KWARGS: Dict[str, Any] = dict(
    PROD_KWARGS,
    use_entmax=True,
    head_structured_latent=True,
    horizon_depth=3,
    horizon_film=True,
    encoder_extra_dilations=(8, 16),
)

VARIANTS: Dict[str, Dict[str, Any]] = {"prod": PROD_KWARGS, "shipped": SHIPPED_KWARGS}

BATCH = 2
SEQ_LEN = TINY_KWARGS["sequence_length"]

# Channel counts of the model's input contract: FHR scattering, FHR phase-harmonic, and the
# concatenated UP stream [up_st(43), up_ph(58)].
C_Y_ST = 43
C_Y_PH = 44
C_U = 101

_SEED = 0


def _shape_of(value: Any) -> Any:
    """Return a JSON-serialisable shape for a forward output.

    Args:
        value: A forward-output value, which may not be a tensor.

    Returns:
        The shape as a list of ints, or ``None`` when the value is not a tensor.
    """
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def capture(variant: str) -> Dict[str, Any]:
    """Build the original model at one variant's geometry and record its structure.

    The seed is set twice: once before construction, because weight init consumes the RNG, and
    once before the forward, because the VAE samples its latent inside ``forward``. Only shapes
    and counts are recorded, so the seed does not affect the result -- it is set to keep the
    capture reproducible if the recorded content is ever widened to values.

    Args:
        variant: Which entry of :data:`VARIANTS` to capture.

    Returns:
        The architecture record, ready to serialise.
    """
    kwargs = VARIANTS[variant]
    torch.manual_seed(_SEED)
    model = SeqVaeLagAttnV3(**kwargs).eval()

    state_dict = {key: list(tensor.shape) for key, tensor in sorted(model.state_dict().items())}
    n_causalized_norms = sum(
        1 for _, module in model.named_modules() if isinstance(module, CausalGroupNorm)
    )

    generator = torch.Generator().manual_seed(_SEED)
    y_st = torch.randn(BATCH, SEQ_LEN, C_Y_ST, generator=generator)
    y_ph = torch.randn(BATCH, SEQ_LEN, C_Y_PH, generator=generator)
    u_stream = torch.randn(BATCH, SEQ_LEN, C_U, generator=generator)

    torch.manual_seed(_SEED)
    with torch.no_grad():
        forward_outputs = model(y_st, y_ph, u_stream)
        loss_outputs = model.compute_loss(forward_outputs, y_st, y_ph)

    return {
        "source": "model/vae_teb_prediction/model/vae_teb_lag_attn_v3.py::SeqVaeLagAttnV3",
        "variant": variant,
        "kwargs": kwargs,
        "input_geometry": {"batch": BATCH, "seq_len": SEQ_LEN, "c_y_st": C_Y_ST,
                           "c_y_ph": C_Y_PH, "c_u": C_U},
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_dict": state_dict,
        "forward_keys": {key: _shape_of(value) for key, value in sorted(forward_outputs.items())},
        "loss_keys": sorted(loss_outputs.keys()),
        "n_causalized_norms": n_causalized_norms,
    }


def main() -> None:
    """Parse ``--out`` / ``--variant`` and write the architecture record as JSON."""
    parser = argparse.ArgumentParser(
        description="Serialise the architecture of SeqVaeLagAttnV3 to a JSON fixture."
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write the JSON architecture record to.",
    )
    parser.add_argument(
        "--variant",
        default="prod",
        choices=sorted(VARIANTS),
        help="Which geometry to capture: 'prod' is the original suite's flag set, 'shipped' "
        "adds the four flags config_lag_attn_v3.yaml actually sets.",
    )
    args = parser.parse_args()

    record = capture(args.variant)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.out}  (variant={args.variant})")
    print(f"  total_params        = {record['total_params']}")
    print(f"  trainable_params    = {record['trainable_params']}")
    print(f"  state_dict keys     = {len(record['state_dict'])}")
    print(f"  forward keys        = {len(record['forward_keys'])}")
    print(f"  loss keys           = {len(record['loss_keys'])}")
    print(f"  n_causalized_norms  = {record['n_causalized_norms']}")


if __name__ == "__main__":
    main()
