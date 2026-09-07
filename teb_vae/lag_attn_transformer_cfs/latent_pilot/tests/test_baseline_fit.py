r"""The frozen baseline's fit: the optimizer step, the budget, and what it cannot reach.

**Execution-machine tests.** They construct a classifier and take real gradient steps, which the
synthetic logic subset excludes; they need no checkpoint file, no clinical data and no GPU::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_baseline_fit.py -q

What they establish: the fit separates a separable split and stops when it stops improving; the
selected classifier is the selected step's rather than the last one's; the budget is finite and
honoured; a fit round-trips through a run directory with its scaler and threshold intact; and --
the property the whole "frozen baseline" claim rests on -- no VAE parameter is reachable from this
function at all.

The bag reduction, the metrics, the threshold and the selection rule are hand-checked without an
optimizer in ``tests/logic/test_baseline.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate, extract, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: A latent narrow enough to read and wide enough that the fit has something to choose.
D_Z = 4

#: The declared baseline budget, shrunk so a test is a test and not a training run.
LR = 0.1
MAX_STEPS = 200
PATIENCE = 20


def _scaler(d_z: int = D_Z):
    """An identity scaler, so a logit is a function of the bag and nothing else."""
    center = np.zeros(d_z, dtype=np.float64)
    scale = np.ones(d_z, dtype=np.float64)
    center.setflags(write=False)
    scale.setflags(write=False)
    return extract.LatentScaler(center=center, scale=scale, record={"population": "train"})


def _bags(split: str, *, healthy: int = 6, adverse: int = 6, separation: float = 3.0, seed: int = 0):
    """Separable bags: the adverse class sits ``separation`` away along the first coordinate."""
    generator = np.random.default_rng(seed)
    labels = np.array([0] * healthy + [1] * adverse, dtype=np.int64)
    values = generator.normal(scale=0.25, size=(labels.size, D_Z))
    values[:, 0] += separation * labels
    frame = pd.DataFrame([
        {
            data.GUID_COLUMN: f"{split.upper()}-{index:02d}",
            data.SPLIT_COLUMN: split,
            data.OUTCOME_COLUMN: int(label),
            "n_segments": 1,
            "n_anchors": 3,
            data.HOURS_COLUMN: 0.5,
            data.ROW_COLUMN: index,
        }
        for index, label in enumerate(labels.tolist())
    ])
    return train.Bags(frame=frame, values=values, split=split, record={"split": split})


def _fit(**overrides):
    """Fit the baseline on separable train and validation bags."""
    arguments = {
        "scaler": _scaler(),
        "lr": LR,
        "max_steps": MAX_STEPS,
        "patience": PATIENCE,
        "weight_decay": 1e-4,
        "seed": 42,
    }
    arguments.update(overrides)
    train_bags = arguments.pop("train_bags", _bags("train", seed=0))
    val_bags = arguments.pop("val_bags", _bags("val", seed=1))
    return train.fit_baseline(train_bags, val_bags, **arguments)


# =============================================================================
# The fit
# =============================================================================
def test_the_fit_separates_a_separable_split_and_is_reproducible():
    first = _fit()
    second = _fit()

    assert first.record["selected_val_auroc"] == pytest.approx(1.0)
    assert first.record["selected_step"] == second.record["selected_step"]
    assert np.allclose(
        first.logits(_bags("val", seed=1).values), second.logits(_bags("val", seed=1).values)
    )


def test_the_returned_classifier_is_the_selected_step_and_not_the_last_one():
    """Early stopping keeps running for ``patience`` steps, and those steps move the weights."""
    fit = _fit()

    assert fit.record["selected_step"] <= fit.record["n_steps_run"]
    val_bags = _bags("val", seed=1)
    scored = evaluate.auroc(val_bags.labels, fit.logits(val_bags.values))
    assert scored == pytest.approx(fit.record["selected_val_auroc"])


def test_the_budget_is_finite_and_reported():
    fit = _fit(max_steps=5, patience=5)

    assert fit.record["n_steps_run"] == 5
    assert fit.record["stopped_because"] == "budget_exhausted"
    assert len(fit.history) == 5


def test_the_fit_stops_once_the_validation_auroc_stops_improving():
    """A twelve-sigma separation is learned well inside the budget, so patience ends the run.

    And it must be patience on the **AUROC**: the cross-entropy keeps falling after the ranking is
    perfect, so a fit that counted patience on the selection rule would spend its whole budget
    chasing a tie-break.
    """
    fit = _fit(patience=3)

    assert fit.record["stopped_because"] == "patience_exhausted"
    assert fit.record["n_steps_run"] < MAX_STEPS
    assert fit.record["selected_val_auroc"] == pytest.approx(1.0)


def test_the_history_records_every_step_it_was_selected_against():
    fit = _fit()

    assert list(fit.history.columns) == ["step", "train_loss", "val_auroc", "val_bce"]
    assert fit.history["step"].tolist() == list(range(1, len(fit.history) + 1))
    assert np.isfinite(fit.history["train_loss"]).all()


def test_the_threshold_is_chosen_on_validation_and_recorded_with_its_rule():
    fit = _fit()
    val_bags = _bags("val", seed=1)

    assert np.isfinite(fit.threshold)
    assert fit.record["threshold"]["population"] == "validation only"
    assert evaluate.balanced_accuracy(
        val_bags.labels, fit.logits(val_bags.values), threshold=fit.threshold
    ) == pytest.approx(fit.record["threshold"]["balanced_accuracy"])


def test_the_classifier_carries_the_scaler_it_was_fitted_with():
    """Buffers, not neighbours: a bag can never be scored under different constants."""
    scaler = extract.LatentScaler(
        center=np.arange(D_Z, dtype=np.float64),
        scale=np.full(D_Z, 2.0),
        record={"population": "train"},
    )
    fit = _fit(scaler=scaler)

    assert np.allclose(fit.classifier.center.numpy(), np.arange(D_Z))
    assert np.allclose(fit.classifier.scale.numpy(), 2.0)


# =============================================================================
# What the fit cannot reach
# =============================================================================
def test_no_vae_parameter_moves_during_a_baseline_fit():
    """The frozen baseline is frozen structurally: the fit is handed no model to move."""
    from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
        TINY_STRIDE,
        make_task,
        tiny_warmup_kwargs,
    )

    task = make_task(model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE))
    model = task.orig_model
    pilot_model.freeze_for_pilot(model)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    _fit()

    after = model.state_dict()
    assert set(before) == set(after)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name


def test_the_test_split_cannot_be_fitted_or_selected_on():
    with pytest.raises(PilotConfigError, match="test split is opened once"):
        _fit(train_bags=_bags("test", seed=0))
    with pytest.raises(PilotConfigError, match="test split is opened once"):
        _fit(val_bags=_bags("test", seed=1))


def test_a_split_carrying_one_class_cannot_be_fitted():
    with pytest.raises(PilotConfigError, match="one binary class"):
        _fit(train_bags=_bags("train", healthy=8, adverse=0))


def test_bags_and_scaler_of_different_widths_are_refused():
    with pytest.raises(PilotConfigError, match="same latent"):
        _fit(scaler=_scaler(D_Z + 1))


def test_the_fit_does_not_advance_another_stage_s_random_stream():
    """The seed is restored, so a fit cannot silently change what a later draw produces."""
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    _fit()
    assert torch.equal(torch.randn(4), expected)


# =============================================================================
# The permuted-label control's path
# =============================================================================
def test_the_control_fits_the_same_path_under_a_permutation():
    """Its own fit, from its own labels -- never initialised from a true-label one.

    The permutation here is by row parity rather than random: it is a fixed rearrangement of the
    same six-and-six label vector, so the comparison below is a statement about the labels the fit
    read and not about which draw a generator happened to produce.
    """
    train_bags, val_bags = _bags("train", seed=0), _bags("val", seed=1)
    permuted = {
        bags.split: {guid: index % 2 for index, guid in enumerate(bags.guids)}
        for bags in (train_bags, val_bags)
    }
    arguments = dict(
        scaler=_scaler(), lr=LR, max_steps=50, patience=PATIENCE, weight_decay=1e-4, seed=42
    )

    truth = train.fit_baseline(train_bags, val_bags, **arguments)
    control = train.fit_baseline(train_bags, val_bags, labels=permuted, **arguments)

    assert control.record["labels_permuted"] is True
    assert truth.record["labels_permuted"] is False
    # Same initialisation, same data, different labels: the two fits cannot land in one place.
    assert not torch.equal(
        control.classifier.linear.weight, truth.classifier.linear.weight
    )


# =============================================================================
# Persistence
# =============================================================================
def test_a_fit_round_trips_through_a_run_directory(tmp_path):
    fit = _fit()
    train.save_fit(fit, tmp_path)

    reloaded = train.load_fit(tmp_path)
    values = _bags("val", seed=1).values

    assert np.allclose(reloaded.logits(values), fit.logits(values))
    assert reloaded.threshold == pytest.approx(fit.threshold)
    assert reloaded.record["selected_step"] == fit.record["selected_step"]
    assert len(reloaded.history) == len(fit.history)


def test_the_saved_classifier_is_not_a_model_checkpoint(tmp_path):
    """Its keys are the classifier's own and never travel through the base model's strict load."""
    fit = _fit()
    train.save_fit(fit, tmp_path)

    blob = torch.load(tmp_path / f"{train.BASELINE_NAME}_{train.CLASSIFIER_FILENAME}",
                      map_location="cpu", weights_only=False)

    assert set(blob["state_dict"]) == {"linear.weight", "linear.bias", "center", "scale"}
    assert "model_kwargs" not in blob and "model_class" not in blob


def test_a_missing_fit_names_the_stage_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fitted before the adaptation"):
        train.load_fit(tmp_path)
