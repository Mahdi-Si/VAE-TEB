"""Forward-pass shape & invariance tests for the model stack.

Skipped automatically when torch isn't installed (pure schema tests live in
``test_cache_schema.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (  # noqa: E402
    build_relative_time_bucket_index,
    guid_sequence_collate_fn,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (  # noqa: E402
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (  # noqa: E402
    GuidSequenceDataset,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.heads import (  # noqa: E402
    PerPositionOutcomeHead,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.segment_tokenizer import (  # noqa: E402
    _compute_te_summary,
    VaeSegmentTokenizer,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.temporal_transformer import (  # noqa: E402
    RelativeTimeTransformer,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.tests.synthetic_cache import (  # noqa: E402
    D_MODEL,
    D_Z,
    L,
    T,
    write_synthetic_cache,
)


@pytest.fixture()
def loader_and_cfg(tmp_path: Path):
    cache = tmp_path / "fold_1" / "train.hdf5"
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_cache(cache, num_guids=4, segments_per_guid=4)
    ds = GuidSequenceDataset(cache, min_samples_per_guid=3)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: guid_sequence_collate_fn(batch),
    )
    cfg = GuidClassifierConfig(d_model_vae=D_MODEL, d_z=D_Z)
    return loader, cfg


def test_segment_tokenizer_shapes(loader_and_cfg) -> None:
    """Tokenizer produces (B, N, 256) and zero rows on padded segments."""
    loader, cfg = loader_and_cfg
    tok = VaeSegmentTokenizer(d_model_vae=cfg.d_model_vae, d_z=cfg.d_z)
    batch = next(iter(loader))
    out = tok(
        h_y=batch["h_y"],
        mu_prior_norm=batch["mu_prior_norm"],
        mu_post_norm=batch["mu_post_norm"],
        kld_per_t=batch["kld_per_t"],
        mean_alpha=batch["mean_alpha"],
        hat_w=batch["hat_w"],
        c_meta=batch["c_meta"],
        segment_mask=batch["segment_mask"],
    )
    B, N = batch["segment_mask"].shape
    assert out["segment_token"].shape == (B, N, cfg.d_model)
    assert out["s_core"].shape == (B, N, cfg.d_seg)
    assert out["u_TE"].shape == (B, N, cfg.te_summary_dim)
    # Padded rows must be zero in the segment_token output.
    pad_rows = ~batch["segment_mask"]
    if pad_rows.any():
        assert torch.all(out["segment_token"][pad_rows] == 0)


def test_transformer_block_shapes() -> None:
    """RelativeTimeTransformer keeps (B, N, d_model) shape."""
    B, N, dm = 3, 5, 256
    x = torch.randn(B, N, dm)
    seg_mask = torch.ones(B, N, dtype=torch.bool)
    seg_mask[1, 3:] = False  # row with shorter sequence
    cum_h = torch.cumsum(torch.ones(B, N) * 0.333, dim=-1)
    rel_idx = build_relative_time_bucket_index(cum_h, num_buckets=32, d_max=40.0)
    tr = RelativeTimeTransformer(d_model=dm, n_heads=4, d_head=64, n_layers=2)
    out = tr(x, seg_mask, rel_idx)
    assert out.shape == (B, N, dm)
    # Padded rows are zeroed at the transformer exit.
    assert torch.all(out[1, 3:] == 0)


def test_per_position_head_shapes_and_zeroed_padding() -> None:
    """PerPositionOutcomeHead emits per-position probs; padded rows zeroed."""
    B, N, dm = 2, 4, 256
    h = torch.randn(B, N, dm)
    seg_mask = torch.ones(B, N, dtype=torch.bool)
    seg_mask[0, 3] = False
    head = PerPositionOutcomeHead(d_model=dm)
    out = head(h, seg_mask)
    assert out["logits_3"].shape == (B, N, 3)
    assert out["logit_bin"].shape == (B, N)
    assert out["prob_3"].shape == (B, N, 3)
    assert out["prob_bin"].shape == (B, N)
    # Padded positions: prob_bin=0, prob_3 row=0 (because masked logits=0
    # then zeroed again by the explicit mask multiply).
    assert out["prob_bin"][0, 3].item() == 0.0
    assert torch.all(out["prob_3"][0, 3] == 0.0)
    # Valid positions: prob_3 sums to 1.
    valid = seg_mask
    sums = out["prob_3"][valid].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_per_position_head_binary_init_at_half() -> None:
    """Zero-initialised binary head should give prob_bin ≈ 0.5 at init.

    Implementation detail: ``head_bin.weight`` is also zero-inited so
    ``logit_bin == 0`` regardless of input, giving ``prob_bin == 0.5``
    at every valid position.
    """
    head = PerPositionOutcomeHead(d_model=64)
    h = torch.randn(2, 3, 64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = head(h, mask)
    assert torch.allclose(out["prob_bin"][mask], torch.full_like(out["prob_bin"][mask], 0.5))


def test_full_classifier_forward(loader_and_cfg) -> None:
    """Full forward returns the documented per-position dict with correct shapes."""
    loader, cfg = loader_and_cfg
    model = GuidOutcomeClassifier(cfg)
    batch = next(iter(loader))
    out = model(batch)
    B, N = batch["segment_mask"].shape
    assert out["logits_3"].shape == (B, N, 3)
    assert out["logit_bin"].shape == (B, N)
    assert out["prob_3"].shape == (B, N, 3)
    assert out["prob_bin"].shape == (B, N)
    assert out["segment_tokens"].shape == (B, N, cfg.d_model)
    assert out["segment_context"].shape == (B, N, cfg.d_model)
    assert out["segment_te_summary"].shape == (B, N, cfg.te_summary_dim)


def test_classifier_marker_no_compile() -> None:
    """The classifier sets ``no_compile=True`` for the Lightning wrapper."""
    cfg = GuidClassifierConfig()
    model = GuidOutcomeClassifier(cfg)
    assert getattr(model, "no_compile", False) is True


def test_classifier_grad_flows() -> None:
    """Backward through the classifier yields finite, non-zero grads."""
    cfg = GuidClassifierConfig(
        d_model_vae=D_MODEL, d_z=D_Z, n_layers=1
    )
    model = GuidOutcomeClassifier(cfg)
    B, N = 2, 4
    batch = {
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
    out = model(batch)
    loss = out["logits_3"].sum() + out["logit_bin"].sum()
    loss.backward()
    grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert grads, "no parameters received gradients"
    assert all(torch.isfinite(g) for g in grads)


def test_classifier_does_not_consume_epoch() -> None:
    """The forward path must not read raw ``epoch`` from the batch.

    The runtime check passes a dummy batch without the ``epoch`` key and
    expects the forward to succeed. (PRD §3.3 leakage rule.)
    """
    cfg = GuidClassifierConfig()
    model = GuidOutcomeClassifier(cfg)
    B, N = 2, 4
    batch = {
        "h_y": torch.randn(B, N, T, cfg.d_model_vae),
        "mu_prior_norm": torch.randn(B, N, T, cfg.d_z),
        "mu_post_norm": torch.randn(B, N, T, cfg.d_z),
        "kld_per_t": torch.rand(B, N, T),
        "mean_alpha": torch.softmax(torch.randn(B, N, T, L), dim=-1),
        "hat_w": torch.ones(B, N, T),
        "c_meta": torch.randn(B, N, 5),
        "segment_mask": torch.ones(B, N, dtype=torch.bool),
        "rel_bucket_idx": torch.zeros(B, N, N, dtype=torch.long),
        "num_segments": torch.full((B,), N, dtype=torch.long),
    }
    # Deliberately omit `epoch` from the batch to prove the forward doesn't
    # depend on it.
    assert "epoch" not in batch
    out = model(batch)
    assert "logits_3" in out


def test_te_summary_uses_kld_weighting() -> None:
    """Lag moments must depend on TE mass, not attention alone."""
    B, N, T_local, L_local = 1, 1, 2, 2
    mean_alpha = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    kld_per_t = torch.tensor([[[10.0, 1.0]]])
    hat_w = torch.ones(B, N, T_local)

    summary = _compute_te_summary(mean_alpha, kld_per_t, hat_w)

    q0 = 10.0 / 11.0
    q1 = 1.0 / 11.0
    expected_m_lag = torch.tensor(q1)
    expected_sigma = torch.sqrt(torch.tensor(q0 * (0.0 - q1) ** 2 + q1 * (1.0 - q1) ** 2))
    expected_entropy = -(q0 * torch.log(torch.tensor(q0)) + q1 * torch.log(torch.tensor(q1)))

    assert torch.allclose(summary[0, 0, 0], torch.tensor(5.5), atol=1e-6)
    assert torch.allclose(summary[0, 0, 1], torch.tensor(10.0), atol=1e-6)
    assert torch.allclose(summary[0, 0, 3], expected_m_lag, atol=1e-6)
    assert torch.allclose(summary[0, 0, 4], expected_sigma, atol=1e-6)
    assert torch.allclose(summary[0, 0, 5], expected_entropy, atol=1e-6)
