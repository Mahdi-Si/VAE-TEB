"""Per-position loss math tests (causal-AR design)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (  # noqa: E402
    GuidClassifierLoss,
    LossWeights,
    estimate_inverse_frequency_class_weights_3,
    estimate_inverse_frequency_class_weights_bin,
)


def _dummy_outputs(B: int, N: int, num_classes: int = 3):
    """Per-position outputs matching the new classifier API."""
    return {
        "logits_3": torch.randn(B, N, num_classes, requires_grad=True),
        "logit_bin": torch.randn(B, N, requires_grad=True),
    }


def test_inverse_frequency_class_weights_3_sums_to_3() -> None:
    w = estimate_inverse_frequency_class_weights_3([0, 0, 1, 1, 2, 2])
    assert pytest.approx(float(w.sum()), rel=1e-6) == 3.0


def test_inverse_frequency_class_weights_3_handles_missing_class() -> None:
    """A class absent from the labels gets weight 1.0 and the sum stays >0."""
    w = estimate_inverse_frequency_class_weights_3([0, 0, 1, 1])  # no class 2
    assert w[2].item() == pytest.approx(1.0)
    assert float(w.sum()) > 0


def test_inverse_frequency_class_weights_bin() -> None:
    w = estimate_inverse_frequency_class_weights_bin([0, 0, 0, 1])
    assert w.shape == (2,)
    assert w[0].item() < w[1].item()  # rarer class gets larger weight


def test_combined_loss_runs_and_is_positive() -> None:
    B, N = 4, 5
    outputs = _dummy_outputs(B, N)
    batch = {
        "label_3": torch.tensor([0, 1, 2, 1]),
        "label_bin": torch.tensor([0.0, 1.0, 1.0, 1.0]),
        "segment_mask": torch.ones(B, N, dtype=torch.bool),
    }
    weights = LossWeights()
    loss_module = GuidClassifierLoss(weights=weights)
    components = loss_module(outputs=outputs, batch=batch)
    for k in ("total_loss", "ce_3", "bce_bin"):
        assert k in components
        assert torch.isfinite(components[k])
    assert components["total_loss"] > 0
    components["total_loss"].backward()


def test_two_level_reduction_matches_hand_rolled() -> None:
    """Loss = (1/B) Σ_b ((Σ_n m_{b,n}·CE_{b,n}) / Σ_n m_{b,n}).

    Verified against a hand-rolled reference. No class weights here so the
    reduction is simply masked-mean per row, then mean over rows.
    """
    B, N, C = 3, 4, 3
    torch.manual_seed(0)
    logits_3 = torch.randn(B, N, C)
    logit_bin = torch.randn(B, N)
    label_3 = torch.tensor([0, 1, 2])
    label_bin = torch.tensor([0.0, 1.0, 1.0])
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    # Hand-rolled per-position CE → masked mean per row → mean over rows.
    target_pos = label_3.unsqueeze(1).expand(B, N).reshape(-1)
    per_pos = F.cross_entropy(logits_3.reshape(-1, C), target_pos, reduction="none").reshape(B, N)
    expected_ce = ((per_pos * mask.float()).sum(-1) / mask.float().sum(-1)).mean()

    target_bin_pos = label_bin.unsqueeze(1).expand(B, N)
    per_pos_bin = F.binary_cross_entropy_with_logits(
        logit_bin, target_bin_pos, reduction="none"
    )
    expected_bce = ((per_pos_bin * mask.float()).sum(-1) / mask.float().sum(-1)).mean()

    loss_module = GuidClassifierLoss(weights=LossWeights(lambda_3=1.0, lambda_2=0.0))
    components = loss_module(
        outputs={"logits_3": logits_3, "logit_bin": logit_bin},
        batch={"label_3": label_3, "label_bin": label_bin, "segment_mask": mask},
    )
    assert torch.allclose(components["ce_3"], expected_ce, atol=1e-6)
    assert torch.allclose(components["bce_bin"], expected_bce, atol=1e-6)


def test_segment_mask_zero_yields_zero_loss() -> None:
    """When all segments are masked out, the per-position loss must be 0."""
    B, N = 2, 3
    outputs = _dummy_outputs(B, N)
    batch = {
        "label_3": torch.tensor([0, 1]),
        "label_bin": torch.tensor([0.0, 1.0]),
        "segment_mask": torch.zeros(B, N, dtype=torch.bool),
    }
    loss_module = GuidClassifierLoss(weights=LossWeights())
    components = loss_module(outputs=outputs, batch=batch)
    assert components["ce_3"].item() == 0.0
    assert components["bce_bin"].item() == 0.0
    assert components["total_loss"].item() == 0.0


def test_class_weights_change_loss_value() -> None:
    """Equal-frequency labels with skewed weights should change the CE."""
    B, N = 4, 3
    outputs = _dummy_outputs(B, N)
    batch = {
        "label_3": torch.tensor([0, 1, 2, 0]),
        "label_bin": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "segment_mask": torch.ones(B, N, dtype=torch.bool),
    }
    base = GuidClassifierLoss(weights=LossWeights())
    weighted = GuidClassifierLoss(
        weights=LossWeights(),
        class_weights_3=torch.tensor([0.1, 1.0, 10.0]),
        class_weights_bin=torch.tensor([0.5, 5.0]),
    )
    base_loss = base(outputs=outputs, batch=batch)["total_loss"]
    weighted_loss = weighted(outputs=outputs, batch=batch)["total_loss"]
    assert not torch.isclose(base_loss, weighted_loss)
