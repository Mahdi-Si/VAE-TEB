"""Position-weighted per-position loss tests (Phase B of the plateau fix).

The classifier's per-position loss broadcasts the GUID label to every
visible segment position. ``position_weight_alpha`` up-weights late
positions (which carry more history under the causal mask) so the
optimizer escapes the class-prior solution. These tests pin:

* ``alpha == 0`` is identical to the prior uniform reduction.
* ``alpha > 0`` produces a strictly increasing weight schedule that
  late positions dominate.
* Padded positions never contribute, regardless of ``alpha``.
* Class-weighted CE composes correctly with the position weighting.
* Fully-masked rows return zero loss without NaN.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (  # noqa: E402
    GuidClassifierLoss,
    LossWeights,
)


# ---------------------------------------------------------------------------
# _position_weights_from_mask
# ---------------------------------------------------------------------------


def test_position_weights_alpha_zero_recovers_mask() -> None:
    """``alpha = 0`` must return ``mask`` cast to the requested dtype."""
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    w = GuidClassifierLoss._position_weights_from_mask(mask, 0.0, torch.float32)
    expected = mask.float()
    assert torch.equal(w, expected)


def test_position_weights_late_bias_is_monotone_increasing() -> None:
    """For valid positions inside one row, weights must rise with rank."""
    mask = torch.ones(1, 10, dtype=torch.bool)
    w = GuidClassifierLoss._position_weights_from_mask(mask, 1.5, torch.float32)
    diffs = w[0, 1:] - w[0, :-1]
    assert (diffs > 0).all(), f"weights not monotone: {w[0]}"
    assert torch.isclose(w[0, -1], torch.tensor(1.0))


def test_position_weights_late_bias_dominates_for_typical_n() -> None:
    """Per the plan, the top half of positions should carry ~80% of the mass
    when ``alpha = 1.5`` and ``N_g = 10``."""
    mask = torch.ones(1, 10, dtype=torch.bool)
    w = GuidClassifierLoss._position_weights_from_mask(mask, 1.5, torch.float32)
    total = w[0].sum()
    top_half = w[0, 5:].sum()
    assert (top_half / total).item() > 0.75


def test_position_weights_zero_at_padded_positions() -> None:
    """Padded slots must always receive zero weight — no rank carry-over."""
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, True, True],  # interior padding
            [True, True, True, True, True],
        ]
    )
    w = GuidClassifierLoss._position_weights_from_mask(mask, 1.5, torch.float32)
    # Padded positions must be exactly 0.
    assert (w[~mask] == 0.0).all()
    # Valid positions must be strictly positive.
    assert (w[mask] > 0.0).all()


def test_position_weights_use_rank_not_absolute_index() -> None:
    """A row with one mid-row padded position must still have its last valid
    position weighted at 1.0 — proving rank is computed within valid slots,
    not over the absolute column index."""
    mask = torch.tensor([[True, True, False, True, True]])
    w = GuidClassifierLoss._position_weights_from_mask(mask, 1.5, torch.float32)
    # n_valid = 4; ranks at valid positions are 1, 2, 3, 4 → last is 1.0.
    assert torch.isclose(w[0, -1], torch.tensor(1.0))


# ---------------------------------------------------------------------------
# _two_level_mean
# ---------------------------------------------------------------------------


def test_two_level_mean_alpha_zero_matches_prior_uniform_reduction() -> None:
    """``alpha = 0`` must reproduce the uniform per-row mean."""
    torch.manual_seed(0)
    B, N = 3, 4
    per_step = torch.rand(B, N)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    expected = (
        ((per_step * mask.float()).sum(-1) / mask.float().sum(-1)).mean()
    )
    got = GuidClassifierLoss._two_level_mean(
        per_step, mask, position_weight_alpha=0.0
    )
    assert torch.allclose(got, expected, atol=1e-7)


def test_two_level_mean_alpha_pos_changes_value() -> None:
    """``alpha > 0`` must produce a different scalar than ``alpha = 0`` when
    the per-step values are not constant."""
    torch.manual_seed(0)
    B, N = 2, 8
    per_step = torch.rand(B, N) * 5.0
    mask = torch.ones(B, N, dtype=torch.bool)
    a0 = GuidClassifierLoss._two_level_mean(
        per_step, mask, position_weight_alpha=0.0
    )
    a15 = GuidClassifierLoss._two_level_mean(
        per_step, mask, position_weight_alpha=1.5
    )
    assert not torch.isclose(a0, a15)


def test_two_level_mean_constant_per_step_invariant_to_alpha() -> None:
    """If ``per_step`` is constant across positions, the weighted mean
    equals that constant for any ``alpha`` (sanity)."""
    B, N = 2, 6
    per_step = torch.full((B, N), 1.234)
    mask = torch.ones(B, N, dtype=torch.bool)
    for alpha in (0.0, 0.5, 1.5, 3.0):
        got = GuidClassifierLoss._two_level_mean(
            per_step, mask, position_weight_alpha=alpha
        )
        assert torch.isclose(got, torch.tensor(1.234))


def test_two_level_mean_fully_masked_row_excluded() -> None:
    """Rows with no valid positions must not contribute."""
    B, N = 3, 4
    per_step = torch.full((B, N), 1.0)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    got = GuidClassifierLoss._two_level_mean(
        per_step, mask, position_weight_alpha=1.5
    )
    # Only rows 0 and 2 contribute, each with constant per_step=1 → mean=1.
    assert torch.isclose(got, torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Loss-level integration (CE / BCE)
# ---------------------------------------------------------------------------


def test_ce_3_per_pos_matches_hand_rolled_with_alpha_15() -> None:
    """Hand-rolled position-weighted CE matches the loss module."""
    B, N, C = 3, 6, 3
    torch.manual_seed(123)
    logits_3 = torch.randn(B, N, C)
    logit_bin = torch.randn(B, N)  # unused here
    label_3 = torch.tensor([0, 1, 2])
    label_bin = torch.tensor([0.0, 1.0, 1.0])
    mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )

    target_pos = label_3.unsqueeze(1).expand(B, N).reshape(-1)
    per_pos = F.cross_entropy(
        logits_3.reshape(-1, C), target_pos, reduction="none"
    ).reshape(B, N)
    weights = GuidClassifierLoss._position_weights_from_mask(
        mask, 1.5, per_pos.dtype
    )
    expected = (
        (per_pos * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)
    ).mean()

    loss_module = GuidClassifierLoss(
        weights=LossWeights(lambda_3=1.0, lambda_2=0.0, position_weight_alpha=1.5)
    )
    components = loss_module(
        outputs={"logits_3": logits_3, "logit_bin": logit_bin},
        batch={"label_3": label_3, "label_bin": label_bin, "segment_mask": mask},
    )
    assert torch.allclose(components["ce_3"], expected, atol=1e-6)


def test_alpha_15_default_in_lossweights_is_zero() -> None:
    """Backwards-compatible default: ``LossWeights()`` keeps the prior
    uniform reduction unless callers opt in."""
    w = LossWeights()
    assert w.position_weight_alpha == 0.0


def test_alpha_15_recovers_uniform_when_class_weighted() -> None:
    """Composition with class weights: alpha=0 must still match the
    pre-existing class-weighted CE."""
    B, N, C = 2, 4, 3
    torch.manual_seed(7)
    logits_3 = torch.randn(B, N, C)
    logit_bin = torch.randn(B, N)
    label_3 = torch.tensor([0, 2])
    label_bin = torch.tensor([0.0, 1.0])
    mask = torch.ones(B, N, dtype=torch.bool)
    cls_w = torch.tensor([0.1, 1.0, 10.0])

    base = GuidClassifierLoss(
        weights=LossWeights(lambda_3=1.0, lambda_2=0.0),
        class_weights_3=cls_w,
    )
    new = GuidClassifierLoss(
        weights=LossWeights(
            lambda_3=1.0, lambda_2=0.0, position_weight_alpha=0.0
        ),
        class_weights_3=cls_w,
    )
    base_loss = base(
        outputs={"logits_3": logits_3, "logit_bin": logit_bin},
        batch={"label_3": label_3, "label_bin": label_bin, "segment_mask": mask},
    )["ce_3"]
    new_loss = new(
        outputs={"logits_3": logits_3, "logit_bin": logit_bin},
        batch={"label_3": label_3, "label_bin": label_bin, "segment_mask": mask},
    )["ce_3"]
    assert torch.allclose(base_loss, new_loss, atol=1e-7)


def test_alpha_15_changes_loss_under_skewed_per_position_signal() -> None:
    """Construct a synthetic case where positions 0..K-1 have wrong
    predictions and positions K..N-1 are correct. With ``alpha = 1.5`` the
    loss is materially smaller than with ``alpha = 0`` because the late
    correct positions carry more weight."""
    B, N, C = 1, 10, 3
    torch.manual_seed(42)
    label_3 = torch.tensor([1])
    label_bin = torch.tensor([1.0])
    mask = torch.ones(B, N, dtype=torch.bool)

    # Build logits so the first 5 positions strongly predict the wrong
    # class (0) and the last 5 strongly predict the correct class (1).
    logits_3 = torch.zeros(B, N, C)
    logits_3[0, :5, 0] = 5.0       # very confident wrong
    logits_3[0, 5:, 1] = 5.0       # very confident correct
    logit_bin = torch.zeros(B, N)

    base = GuidClassifierLoss(
        weights=LossWeights(lambda_3=1.0, lambda_2=0.0, position_weight_alpha=0.0)
    )
    weighted = GuidClassifierLoss(
        weights=LossWeights(lambda_3=1.0, lambda_2=0.0, position_weight_alpha=1.5)
    )
    base_loss = base(
        outputs={"logits_3": logits_3, "logit_bin": logit_bin},
        batch={"label_3": label_3, "label_bin": label_bin, "segment_mask": mask},
    )["ce_3"]
    w_loss = weighted(
        outputs={"logits_3": logits_3, "logit_bin": logit_bin},
        batch={"label_3": label_3, "label_bin": label_bin, "segment_mask": mask},
    )["ce_3"]
    assert (w_loss < base_loss).item(), (
        f"position-weighted loss should be smaller "
        f"(late positions are correct): {float(w_loss)} vs {float(base_loss)}"
    )
