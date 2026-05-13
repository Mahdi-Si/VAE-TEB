"""Per-head enable/disable tests for ``guid_cls_v1``.

Covers the head-toggle feature added in plan
``read-model-dataset-explained-research-md-idempotent-sedgewick``:

  * ``PerPositionOutcomeHead`` instantiates only the enabled head
    linears, exposes only the enabled keys in its forward dict, and
    refuses construction when both heads are disabled.
  * ``GuidClassifierConfig`` propagates the flags into the head and
    raises when both are disabled.
  * ``GuidClassifierLoss`` skips the disabled term and the returned
    dict omits the corresponding key.
  * ``validate_predictions_df`` (head-aware schema checker) accepts a
    3-class-only DataFrame.

All tests are skipped when torch isn't installed, mirroring
``test_model_shapes.py`` / ``test_losses.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (  # noqa: E402
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.heads import (  # noqa: E402
    PerPositionOutcomeHead,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (  # noqa: E402
    GuidClassifierLoss,
    LossWeights,
)


# ---------------------------------------------------------------------------
# PerPositionOutcomeHead
# ---------------------------------------------------------------------------


def test_head_default_carries_both_branches() -> None:
    """Default construction wires up both head linears."""
    head = PerPositionOutcomeHead(d_model=64)
    assert head.head_3 is not None
    assert head.head_bin is not None
    out = head(torch.randn(2, 3, 64), torch.ones(2, 3, dtype=torch.bool))
    for key in ("logits_3", "prob_3", "logit_bin", "prob_bin"):
        assert key in out


def test_head_binary_only_omits_3class_keys() -> None:
    """``enable_three_class=False`` drops the 3-class linear and outputs."""
    head = PerPositionOutcomeHead(
        d_model=64, enable_three_class=False, enable_binary=True
    )
    assert head.head_3 is None
    assert head.head_bin is not None
    # Disabled head holds no parameters — none of head_3.* shows up in
    # named_parameters().
    param_names = {n for n, _ in head.named_parameters()}
    assert not any(n.startswith("head_3") for n in param_names)
    out = head(torch.randn(2, 3, 64), torch.ones(2, 3, dtype=torch.bool))
    assert "logit_bin" in out
    assert "prob_bin" in out
    assert "logits_3" not in out
    assert "prob_3" not in out


def test_head_three_class_only_omits_binary_keys() -> None:
    """``enable_binary=False`` drops the binary linear and outputs."""
    head = PerPositionOutcomeHead(
        d_model=64, enable_three_class=True, enable_binary=False
    )
    assert head.head_3 is not None
    assert head.head_bin is None
    param_names = {n for n, _ in head.named_parameters()}
    assert not any(n.startswith("head_bin") for n in param_names)
    out = head(torch.randn(2, 3, 64), torch.ones(2, 3, dtype=torch.bool))
    assert "logits_3" in out
    assert "prob_3" in out
    assert "logit_bin" not in out
    assert "prob_bin" not in out


def test_head_refuses_both_disabled() -> None:
    """Constructor raises when both heads are disabled."""
    with pytest.raises(ValueError):
        PerPositionOutcomeHead(
            d_model=64, enable_three_class=False, enable_binary=False
        )


def test_head_prior_init_skips_disabled_branch() -> None:
    """``init_class_bias_from_prior`` ignores priors for disabled heads."""
    head = PerPositionOutcomeHead(
        d_model=32, enable_three_class=False, enable_binary=True
    )
    # Passing a 3-class prior into a binary-only head is a silent no-op
    # (the 3-class head doesn't exist).
    head.init_class_bias_from_prior(
        prior_3=torch.tensor([0.7, 0.2, 0.1]),
        prior_bin=torch.tensor(0.05),
    )
    # Sanity: the binary bias matches logit(0.05).
    import math

    expected = math.log(0.05) - math.log(1.0 - 0.05)
    assert head.head_bin is not None
    assert pytest.approx(head.head_bin.bias.item(), rel=1e-5) == expected


# ---------------------------------------------------------------------------
# GuidClassifierConfig + GuidOutcomeClassifier wiring
# ---------------------------------------------------------------------------


def test_classifier_config_propagates_flags() -> None:
    """The dataclass plumbs flags into ``PerPositionOutcomeHead``."""
    cfg = GuidClassifierConfig(
        enable_three_class_head=False,
        enable_binary_head=True,
    )
    model = GuidOutcomeClassifier(cfg)
    assert model.enable_three_class_head is False
    assert model.enable_binary_head is True
    assert model.outcome_head.head_3 is None
    assert model.outcome_head.head_bin is not None


def test_classifier_config_rejects_both_disabled() -> None:
    """``GuidClassifierConfig`` raises when both flags are False."""
    with pytest.raises(ValueError):
        GuidClassifierConfig(
            enable_three_class_head=False,
            enable_binary_head=False,
        )


# ---------------------------------------------------------------------------
# GuidClassifierLoss
# ---------------------------------------------------------------------------


def _seg_batch(B: int, N: int) -> dict:
    return {
        "label_3": torch.tensor([c % 3 for c in range(B)], dtype=torch.long),
        "label_bin": torch.tensor(
            [float(c % 2) for c in range(B)], dtype=torch.float32
        ),
        "segment_mask": torch.ones(B, N, dtype=torch.bool),
    }


def test_loss_binary_only_omits_ce3() -> None:
    """``enable_three_class=False`` -> ``ce_3`` not in components, total = λ₂·BCE."""
    B, N = 4, 5
    outputs = {"logit_bin": torch.randn(B, N, requires_grad=True)}
    loss = GuidClassifierLoss(
        LossWeights(
            lambda_3=1.0,
            lambda_2=0.5,
            enable_three_class=False,
            enable_binary=True,
        )
    )
    components = loss(outputs=outputs, batch=_seg_batch(B, N))
    assert "ce_3" not in components
    assert "bce_bin" in components
    expected = 0.5 * components["bce_bin"]
    assert torch.allclose(components["total_loss"], expected)


def test_loss_three_class_only_omits_bce() -> None:
    """``enable_binary=False`` -> ``bce_bin`` not in components, total = λ₃·CE₃."""
    B, N = 3, 4
    outputs = {"logits_3": torch.randn(B, N, 3, requires_grad=True)}
    loss = GuidClassifierLoss(
        LossWeights(
            lambda_3=1.0,
            lambda_2=0.5,
            enable_three_class=True,
            enable_binary=False,
        )
    )
    components = loss(outputs=outputs, batch=_seg_batch(B, N))
    assert "bce_bin" not in components
    assert "ce_3" in components
    expected = 1.0 * components["ce_3"]
    assert torch.allclose(components["total_loss"], expected)


def test_loss_raises_when_logits_missing() -> None:
    """Mismatch between flag and outputs dict -> KeyError with a helpful msg."""
    B, N = 2, 3
    # Loss says 3-class enabled but outputs dict has only the binary key.
    outputs = {"logit_bin": torch.randn(B, N)}
    loss = GuidClassifierLoss(LossWeights(enable_three_class=True, enable_binary=False))
    with pytest.raises(KeyError, match="logits_3"):
        loss(outputs=outputs, batch=_seg_batch(B, N))


def test_loss_gradient_flows_with_single_head() -> None:
    """``loss.backward()`` produces non-zero grads on the enabled-head logits only."""
    B, N = 3, 4
    logits_3 = torch.randn(B, N, 3, requires_grad=True)
    outputs = {"logits_3": logits_3}
    loss = GuidClassifierLoss(
        LossWeights(enable_three_class=True, enable_binary=False)
    )
    components = loss(outputs=outputs, batch=_seg_batch(B, N))
    components["total_loss"].backward()
    assert logits_3.grad is not None
    assert torch.isfinite(logits_3.grad).all()
    assert logits_3.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# validate_predictions_df (head-aware schema)
# ---------------------------------------------------------------------------


def test_validate_predictions_df_accepts_3class_only_schema() -> None:
    """Schema check passes when binary columns are intentionally absent."""
    pd = pytest.importorskip("pandas")
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: E402
        validate_predictions_df,
    )

    df = pd.DataFrame(
        {
            "guid": ["a", "a", "b"],
            "epoch": [-3600.0, -1800.0, -3600.0],
            "prob_healthy": [0.7, 0.6, 0.5],
            "prob_acidosis": [0.2, 0.3, 0.4],
            "prob_hie": [0.1, 0.1, 0.1],
            "predicted_class_3": [0, 0, 1],
            "guid_class_3_target": [0, 0, 1],
        }
    )
    # Should not raise — binary block is correctly absent.
    validate_predictions_df(df, "binary-disabled")


def test_validate_predictions_df_still_checks_binary_range() -> None:
    """When binary columns are present, value-range checks still fire."""
    pd = pytest.importorskip("pandas")
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: E402
        validate_predictions_df,
    )

    df = pd.DataFrame(
        {
            "guid": ["a"],
            "epoch": [-3600.0],
            "binary_target": [0],
            "prob_class_1": [1.5],  # out-of-range
        }
    )
    with pytest.raises(ValueError, match="prob_class_1"):
        validate_predictions_df(df, "bad-binary")
