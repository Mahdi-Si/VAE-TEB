r"""The shuffled-label control's own fit, and the frozen ``mu_prior`` probe.

**Execution-machine tests.** They construct classifiers and take real gradient steps, which the
synthetic logic subset excludes; they need no checkpoint file, no clinical data and no GPU::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_controls.py -q

What they establish: the control fits its **own** permuted-label classifier rather than inheriting
one already fitted to the association it exists to test the absence of; a fit under permuted labels
does not carry the true-label signal to a held-out split; the prior probe reads ``mu_prior`` through
the same code and under its own scaler; and switching the probe off withdraws the claim it would
have supported instead of leaving it implied.

The permutation itself, the metrics and the bootstrap are hand-checked without a fit in
``tests/logic/test_metrics.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate, extract, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

D_Z = 4

#: A shrunken but structurally complete settings block: only the leaves these two fits read.
SETTINGS = {
    "seed": 42,
    "windows": {"supervised_hours": 1.0},
    "bag": {"halflife_hours": 0.5},
    "baseline": {"lr": 0.1, "max_steps": 100, "patience": 20},
    "optim": {"weight_decay": 1e-4},
    "prior_probe": True,
}


class _Extraction:
    """The attributes the bag and scaler paths read off a real extraction."""

    def __init__(self, frame, arrays):
        self.frame, self.arrays = frame, arrays

    @property
    def retained(self):
        return data.retained(self.frame)


def _extraction(split, *, n_healthy=8, n_adverse=8, seed=0, separation=3.0):
    """One anchor per recording, separable in ``mu_post`` and **not** in ``mu_prior``.

    The prior is deliberately label-free here: the probe must be able to report that the target-only
    latent does not carry the signal, which is a result and not a failure.
    """
    generator = np.random.default_rng(seed)
    outcomes = [0] * n_healthy + [1] * n_adverse
    rows = []
    for index, outcome in enumerate(outcomes):
        rows.append({
            data.SPLIT_COLUMN: split,
            data.GUID_COLUMN: f"{split.upper()}-{index:02d}",
            data.EPOCH_COLUMN: -3000.0,
            data.ANCHOR_COLUMN: 0,
            data.HOURS_COLUMN: 0.5,
            data.EXCLUSION_COLUMN: "",
            data.ROW_COLUMN: index,
        })
    post = generator.normal(scale=0.25, size=(len(outcomes), D_Z))
    post[:, 0] += separation * np.asarray(outcomes, dtype=np.float64)
    prior = generator.normal(scale=1.0, size=(len(outcomes), D_Z))
    return _Extraction(pd.DataFrame(rows), {"mu_post": post, "mu_prior": prior})


def _recordings(splits=("train", "val"), *, n_healthy=8, n_adverse=8):
    """The recording table matching the extractions above."""
    rows = []
    for split in splits:
        for index, outcome in enumerate([0] * n_healthy + [1] * n_adverse):
            rows.append({
                data.GUID_COLUMN: f"{split.upper()}-{index:02d}",
                data.SPLIT_COLUMN: split,
                data.OUTCOME_COLUMN: outcome,
                data.EXCLUSION_COLUMN: "",
                "eligible": True,
            })
    return pd.DataFrame(rows)


def _scaler(d_z=D_Z):
    """An identity scaler, so a logit is a function of the bag and nothing else."""
    center, scale = np.zeros(d_z), np.ones(d_z)
    center.setflags(write=False)
    scale.setflags(write=False)
    return extract.LatentScaler(center=center, scale=scale, record={"population": "train"})


@pytest.fixture
def extractions():
    """A training and a validation extraction, drawn from different streams."""
    return _extraction("train", seed=0), _extraction("val", seed=1)


# =============================================================================
# The shuffled-label control
# =============================================================================
def _true_label_baseline(train_extraction, val_extraction, recordings):
    """The ordinary baseline fit, for comparison against the control's own."""
    bags = {
        split: train.build_bags(
            extraction, recordings, split=split, supervised_hours=1.0, halflife_hours=0.5
        )
        for split, extraction in (("train", train_extraction), ("val", val_extraction))
    }
    return train.fit_baseline(
        bags["train"],
        bags["val"],
        scaler=_scaler(),
        lr=SETTINGS["baseline"]["lr"],
        max_steps=SETTINGS["baseline"]["max_steps"],
        patience=SETTINGS["baseline"]["patience"],
        weight_decay=SETTINGS["optim"]["weight_decay"],
        seed=SETTINGS["seed"],
    )


def test_the_control_fits_its_own_classifier_rather_than_inheriting_one(extractions):
    """Inheriting the true-label head would hand it the association it exists to test."""
    train_extraction, val_extraction = extractions
    recordings = _recordings()
    permuted, record = train.control_recordings(recordings, seed=7)

    truth = _true_label_baseline(train_extraction, val_extraction, recordings)
    control = train.fit_control_baseline(
        train_extraction,
        val_extraction,
        permuted,
        permutation=record,
        scaler=_scaler(),
        settings=SETTINGS,
    )

    assert control.record["labels_permuted"] is True
    assert control.record["permutation"]["seed"] == 7
    assert "never a true-label one" in control.record["initialised_from"]
    assert truth.record["labels_permuted"] is False
    assert not torch.equal(
        control.classifier.linear.weight, truth.classifier.linear.weight
    )


def test_a_control_fitted_under_the_null_does_not_carry_the_true_signal(extractions):
    """The whole point of the control: it must not reproduce the real fit's held-out result."""
    train_extraction, val_extraction = extractions
    recordings = _recordings()
    permuted, _record = train.control_recordings(recordings, seed=7)
    held_out = _extraction("test", seed=2)
    truth_bags = train.build_bags(
        held_out,
        _recordings(("test",)),
        split="test",
        supervised_hours=1.0,
        halflife_hours=0.5,
    )

    truth = _true_label_baseline(train_extraction, val_extraction, recordings)
    control = train.fit_control_baseline(
        train_extraction,
        val_extraction,
        permuted,
        permutation=_record,
        scaler=_scaler(),
        settings=SETTINGS,
    )

    # True held-out labels for both, which is how the control is read.
    real = evaluate.auroc(truth_bags.labels, truth.logits(truth_bags.values))
    null = evaluate.auroc(truth_bags.labels, control.logits(truth_bags.values))
    assert real == pytest.approx(1.0)
    assert null < real


def test_the_control_reads_its_labels_through_the_ordinary_bag_path(extractions):
    """Permuting the table rather than threading a mapping is what keeps the paths identical."""
    train_extraction, _val = extractions
    permuted, record = train.control_recordings(_recordings(), seed=7)

    bags = train.build_bags(
        train_extraction, permuted, split="train", supervised_hours=1.0, halflife_hours=0.5
    )

    assert sorted(bags.labels.tolist()) == [0] * 8 + [1] * 8
    assert record["per_split"]["train"]["n_changed"] > 0


def test_the_control_cannot_be_fitted_on_a_permuted_held_out_split():
    """The refusal is on the *request*, not on the table.

    A production recording table always carries test rows -- it is the cohort -- so refusing one
    would refuse every real control run. What must be impossible is asking for the held-out labels
    to be shuffled, and what must hold for the ordinary call is that those labels come back
    untouched.
    """
    with pytest.raises(PilotConfigError, match="never permuted"):
        train.permute_outcomes(
            _recordings(("train", "test")), seed=7, splits=("train", "test")
        )

    frame = _recordings(("train", "val", "test"))
    permuted, record = train.control_recordings(frame, seed=7)
    held_out = frame[data.SPLIT_COLUMN].astype(str) == "test"
    assert (
        permuted.loc[held_out, data.OUTCOME_COLUMN].tolist()
        == frame.loc[held_out, data.OUTCOME_COLUMN].tolist()
    )
    assert "test" not in record["per_split"]
    assert record["test_split_permuted"] is False


# =============================================================================
# The frozen prior probe
# =============================================================================
def test_the_prior_probe_reads_mu_prior_under_its_own_scaler(extractions):
    train_extraction, val_extraction = extractions

    fit, record = train.fit_prior_probe(
        train_extraction, val_extraction, _recordings(), settings=SETTINGS
    )

    assert record["enabled"] is True
    assert record["key"] == "mu_prior"
    assert fit.record["train"]["key"] == "mu_prior"
    # Its constants come from mu_prior's own training anchors, not from mu_post's.
    assert record["scaler"]["key"] == "mu_prior"
    assert not np.allclose(fit.classifier.scale.numpy(), 1.0)


def test_the_prior_probe_reports_what_the_target_only_latent_does_not_carry(extractions):
    """A probe finding nothing is a result: it is the comparison the mu_post claim rests on."""
    train_extraction, val_extraction = extractions

    fit, _record = train.fit_prior_probe(
        train_extraction, val_extraction, _recordings(), settings=SETTINGS
    )

    # The fixture's prior is label-free by construction, so this must not look like the posterior.
    assert fit.record["selected_val_auroc"] < 1.0


def test_switching_the_probe_off_withdraws_the_combined_branch_claim(extractions):
    train_extraction, val_extraction = extractions

    fit, record = train.fit_prior_probe(
        train_extraction,
        val_extraction,
        _recordings(),
        settings=dict(SETTINGS, prior_probe=False),
    )

    assert fit is None
    assert record["enabled"] is False
    assert record["combined_branch_claim_supported"] is False
    assert "makes no claim" in record["note"]


def test_the_disclosure_matches_whether_the_probe_actually_ran(extractions):
    """The report's claim and the run's controls are read from one record, not two."""
    train_extraction, val_extraction = extractions
    fit, record = train.fit_prior_probe(
        train_extraction, val_extraction, _recordings(), settings=SETTINGS
    )

    disclosure = evaluate.control_disclosure(
        n_control_fits=1, prior_probe=record["enabled"]
    )

    assert fit is not None
    assert disclosure["combined_branch_claim_supported"] is record[
        "combined_branch_claim_supported"
    ]
    assert disclosure["permutation_p_value"] is False
