"""Per-position output equals the prefix-sweep reference under causal AR.

With ``apply_segment_dropout`` disabled and a fixed model, the per-position
output at position ``n`` must equal the model's output at the *last* visible
position when the batch is truncated to a prefix of length ``n+1``. This
proves the new design is mathematically equivalent to the old prefix-sweep
inference (and hence that eval CSV semantics are unchanged).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (  # noqa: E402
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.tests.synthetic_cache import (  # noqa: E402
    D_MODEL,
    D_Z,
    L,
    T,
)


def _make_batch(B: int, N: int):
    """Construct a synthetic forward-ready batch (no dataloader needed)."""
    return {
        "h_y": torch.randn(B, N, T, D_MODEL),
        "mu_prior_norm": torch.randn(B, N, T, D_Z),
        "mu_post_norm": torch.randn(B, N, T, D_Z),
        "kld_per_t": torch.rand(B, N, T),
        "mean_alpha": torch.softmax(torch.randn(B, N, T, L), dim=-1),
        "hat_w": torch.ones(B, N, T),
        "weight": torch.ones(B, N, T),
        "c_meta": torch.randn(B, N, 5),
        "segment_mask": torch.ones(B, N, dtype=torch.bool),
        "rel_bucket_idx": torch.zeros(B, N, N, dtype=torch.long),
        "num_segments": torch.full((B,), N, dtype=torch.long),
    }


def test_per_position_matches_prefix_truncation() -> None:
    """``prob_3[:, n-1, :]`` from a single forward equals the prefix-``n``
    forward's last-position output for every ``n`` in ``1..N``.
    """
    cfg = GuidClassifierConfig(
        d_model_vae=D_MODEL, d_z=D_Z, n_layers=1, dropout=0.0
    )
    model = GuidOutcomeClassifier(cfg).eval()

    B, N = 2, 5
    batch = _make_batch(B, N)

    with torch.no_grad():
        out_full = model(batch)

    for n in range(1, N + 1):
        sliced = {**batch}
        # Truncate the visible prefix to length n.
        sm = batch["segment_mask"].clone()
        sm[:, n:] = False
        sliced["segment_mask"] = sm
        sliced["num_segments"] = torch.full((B,), n, dtype=torch.long)

        with torch.no_grad():
            out_sliced = model(sliced)

        # The prefix-n output at position (n-1) must match the full-forward
        # output at position (n-1) — proves causal independence of future
        # tokens.
        full_at_n = out_full["prob_3"][:, n - 1, :]
        sliced_at_n = out_sliced["prob_3"][:, n - 1, :]
        assert torch.allclose(full_at_n, sliced_at_n, atol=1e-5), (
            f"position {n - 1} differs between full and prefix-{n} forwards: "
            f"max abs diff = {(full_at_n - sliced_at_n).abs().max().item()}"
        )

        full_bin = out_full["prob_bin"][:, n - 1]
        sliced_bin = out_sliced["prob_bin"][:, n - 1]
        assert torch.allclose(full_bin, sliced_bin, atol=1e-5)
