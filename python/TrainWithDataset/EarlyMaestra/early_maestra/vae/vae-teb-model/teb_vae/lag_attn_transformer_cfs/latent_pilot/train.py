r"""The frozen baseline, the tiny adaptation, the shuffled-label control, and selection.

The frozen baseline (LP-08) and the tiny adaptation (LP-09) are implemented below; the
shuffled-label control is the control half of LP-11 and is still described here rather than
written.

Objective
---------

With $v_i(\theta)$ the recency-weighted final-hour bag from ``data.py`` and $S$ the frozen scaler
from ``extract.py``, one linear logit and a class-balanced binary cross-entropy:

$$\ell_i = w^\top S(v_i(\theta)) + b, \qquad
\mathcal L_{\rm cls} = \frac12 \sum_{c \in \{0,1\}} \frac{1}{N_c}
\sum_{i : y_i = c} \operatorname{BCEWithLogits}(\ell_i, y_i).$$

Balance is implemented **either** as balanced sampling of recordings **or** as recording-level class
weights, never both -- applying both reweights the minority class twice. Sampling is uniform within
each binary class; the natural acidosis/HIE mixture inside the positive class is retained and
disclosed. Validation and test keep natural prevalence. A class-balanced logit is an association
score, not a calibrated clinical risk.

The pretrained model is a fixed, detached teacher over every valid anchor in the last three hours:

$$\mathcal L_{\rm keep} = \operatorname{mean}_i \operatorname{mean}_{s \in i}
\operatorname{mean}_{a \in A^{3h}_{is}} \frac{1}{d_z}
\left\| \frac{\mu^q_{\theta,isa} - \mu^q_{0,isa}}{s} \right\|_2^2, \qquad
\mathcal L = \mathcal L_{\rm cls} + 0.1\, \mathcal L_{\rm keep}.$$

Preservation is an engineering regularizer against gratuitous movement, including in the earlier
hours that carry no outcome label. It is not a smoothness assumption, not a guarantee that forecasts
survive, and not a label: earlier anchors can still move, because the same parameters serve every
time point.

The original forecast loss is deliberately **outside** this objective. Including it would pull in the
decoder, the stochastic likelihood scaling and the pretraining schedule; forecast preservation is a
validation gate instead (see ``evaluate.py``). Nothing here inherits the pretraining optimizer, its
$\beta$ warm-up or its 2000-step learning-rate ramp -- re-entering either through the existing
trainer defaults would restart a schedule this fit is not running.

Why a linear classification loss first: its gradient with respect to the standardized bag is
$(\operatorname{sigmoid}(\ell) - y)\, w$, which pushes discrimination along one latent direction while
the preservation term limits movement elsewhere. It does not promise compact Euclidean clusters, and
the figures are read with that distinction in mind. Supervised contrastive learning is a later
experiment, not this one.

Fitting order and selection
---------------------------

The frozen baseline is fitted first, on the same training bags and under the same validation rule,
against cached pretrained vectors with an explicit finite budget that is logged. The adaptation then
starts from the original mean-head weights and **from that baseline classifier**, so the comparison
isolates the representation change rather than a differently initialised head.

Optimizer: AdamW with two parameter groups -- mean heads at $10^{-4}$, classifier at $10^{-3}$ --
weight decay $10^{-4}$, gradient-norm clip $1.0$, float32, fixed seeds. Batches are 8 distinct
recordings, 4 per binary class, with every available late segment in a bag; a class with fewer than
four training recordings reduces the batch and accumulates instead. Repeated segments are never
presented as distinct patients.

At most ``optim.max_epochs`` epochs, stopping after ``optim.patience`` without an improved
eligible validation AUROC. A candidate
is eligible only if it **passes the preservation gates first**; selection is then the highest
recording-level validation AUROC, ties broken by lower validation BCE and then by the earlier epoch.
The pretrained model is kept as candidate epoch zero: if no adapted candidate improves selection, the
frozen model is retained and reported as the outcome. Rejection reasons and the full history are
logged.

Shuffled-label control
----------------------

One run, starting from the **original pretrained model**, with a fixed GUID-level permutation of the
training labels -- every segment keeps its own recording's assigned label -- and its **own**
permuted-label baseline classifier. It is never initialised from a true-label fit. Validation labels
are permuted independently and drive its selection; true test labels are used once, at final
evaluation. A single control fit is a leakage and overfitting sanity check. It is not a permutation
p-value, and calling it one would require many full refits including selection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate, extract
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: The name the frozen baseline's artifacts are filed under. The adaptation and the shuffled-label
#: control write beside it under names of their own, so no stage can overwrite another's fit.
BASELINE_NAME = "baseline"

#: On-disk names inside a run directory, prefixed by the fit's name.
CLASSIFIER_FILENAME = "classifier.pt"
FIT_FILENAME = "fit.json"

#: The splits a classifier may be fitted or selected on. The test split is absent by construction:
#: it is read once, from the evaluation stage, after everything here is locked.
FITTING_SPLITS: Tuple[str, ...] = ("train", "val")


@dataclass(frozen=True)
class Bags:
    """One split's supervised bags: one vector per eligible recording, and its outcome.

    Attributes:
        frame: One row per recording -- GUID, binary outcome, the segment and anchor counts behind
            its bag, its median time before delivery, and ``row`` indexing ``values``.
        values: ``(N, d_z)`` pooled posterior means, **unstandardized**. The scaler lives inside the
            classifier and is applied there, so no call site can pool one way and standardize
            another.
        split: Which split these came from.
        record: Counts, prevalence and the window settings the bags were built under.
    """

    frame: pd.DataFrame
    values: np.ndarray
    split: str
    record: Dict[str, Any]

    @property
    def labels(self) -> np.ndarray:
        """The binary outcomes, aligned with :attr:`values`."""
        return np.asarray(self.frame[data.OUTCOME_COLUMN], dtype=np.int64)

    @property
    def guids(self) -> List[str]:
        """The recordings, in row order."""
        return [str(value) for value in self.frame[data.GUID_COLUMN].tolist()]


def build_bags(
    extraction: Any,
    recordings: pd.DataFrame,
    *,
    split: str,
    supervised_hours: float,
    halflife_hours: float,
    key: str = "mu_post",
    require_both: bool = True,
) -> Bags:
    r"""Pool one split's final-hour anchors into one supervised vector per eligible recording.

    Anchors are averaged within segments and segments are then pooled with the recency weight
    $\omega_{is} = 2^{-\widetilde r_{is}/h}$, which is ``data.recording_bags`` -- the same reduction
    the adaptation's forward path uses, so the frozen baseline and the fine-tuned model are fitted
    on bags built by one function rather than by two that agree today.

    Only recordings that passed **every** rule enter: a labelling exclusion, or a late-coverage
    exclusion, keeps a recording out of the fit and out of every count reported from it. The
    eligibility rules were fixed before any outcome comparison and are applied identically to both
    classes.

    Args:
        extraction: The split's :class:`~latent_pilot.extract.LatentExtraction`.
        recordings: The recording table, with outcomes and eligibility attached.
        split: The split being built. Checked against the extraction, because a bag table labelled
            with the wrong split would be fitted on or evaluated against the wrong population.
        supervised_hours: The supervised window's upper edge.
        halflife_hours: The recency half-life inside it.
        key: Which latent quantity to pool. ``mu_post`` is the pilot's subject; ``mu_prior`` is the
            conditional FHR-only probe, which reads the same bags through the same code.
        require_both: Refuse a split carrying one binary class. True for the two fitting splits,
            where discrimination is otherwise undefined.

    Returns:
        The bags.

    Raises:
        PilotConfigError: If the extraction is not the named split's, if no eligible recording
            survives, or if ``require_both`` and one class is missing.
    """
    frame = extraction.retained
    present = sorted({str(value) for value in frame[data.SPLIT_COLUMN].tolist()})
    if present != [str(split)]:
        raise PilotConfigError(
            f"the extraction carries split(s) {present} but bags were requested for {split!r}. "
            f"A bag table labelled with another split's name would be fitted on, or scored "
            f"against, the wrong population."
        )

    eligible, outcomes = data.eligible_anchors(frame, recordings, split=split)
    if eligible.empty:
        raise PilotConfigError(
            f"no eligible {split!r} recording contributed a retained anchor. Either the cohort's "
            f"labelling excluded every recording, or the late-coverage rules did; the coverage "
            f"table names which, and the pilot cannot be fitted on an empty split."
        )

    # The extraction's **full** array: ``row`` indexes it, and every reduction below reads that
    # column, so a gathered matrix would be addressed by the ungathered positions.
    bag_frame, values = data.recording_bags(
        eligible,
        extraction.arrays[key],
        supervised_hours=supervised_hours,
        halflife_hours=halflife_hours,
    )
    if bag_frame.empty:
        raise PilotConfigError(
            f"no eligible {split!r} recording has a retained anchor inside the supervised window "
            f"(0, {supervised_hours}] h, so no recording can supply a bag."
        )
    bag_frame = bag_frame.copy()
    bag_frame[data.SPLIT_COLUMN] = str(split)
    bag_frame[data.OUTCOME_COLUMN] = [
        int(outcomes[str(guid)]) for guid in bag_frame[data.GUID_COLUMN].tolist()
    ]

    labels = np.asarray(bag_frame[data.OUTCOME_COLUMN], dtype=np.int64)
    counts = {int(label): int((labels == label).sum()) for label in (0, 1)}
    if require_both and (counts[0] == 0 or counts[1] == 0):
        raise PilotConfigError(
            f"the {split!r} bags carry {counts} recording(s) by binary outcome, so discrimination "
            f"cannot be estimated on them. Report that the pilot cannot run on this fold rather "
            f"than choosing another one, whose results would then have been chosen by looking at "
            f"them."
        )

    record = {
        "split": str(split),
        "key": key,
        "n_recordings": int(len(bag_frame)),
        "n_healthy": counts[0],
        "n_adverse": counts[1],
        "prevalence": float(labels.mean()) if labels.size else float("nan"),
        "n_segments": int(bag_frame["n_segments"].sum()),
        "n_anchors": int(bag_frame["n_anchors"].sum()),
        "supervised_hours": float(supervised_hours),
        "halflife_hours": float(halflife_hours),
        "d_z": int(values.shape[1]) if values.size else 0,
        "note": (
            "one bag per eligible recording; segment means first, then the recency weighting, so "
            "duplicating a segment's anchors cannot multiply that recording's supervised weight"
        ),
    }
    logger.info(
        f"{split} bags: {record['n_recordings']} recording(s) "
        f"({counts[0]} healthy, {counts[1]} adverse, prevalence {record['prevalence']:.3f}) from "
        f"{record['n_segments']} segment(s) and {record['n_anchors']} anchor(s)"
    )
    return Bags(frame=bag_frame, values=values, split=str(split), record=record)


# =============================================================================
# The objective and the selection rule
# =============================================================================
def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""The class-balanced binary cross-entropy of §5.2.

    $$\mathcal L_{\rm cls} = \frac12 \sum_{c \in \{0, 1\}} \frac{1}{N_c}
    \sum_{i : y_i = c} \operatorname{BCEWithLogits}(\ell_i, y_i).$$

    **This is the class-weighting arm of the protocol's either/or**, and the baseline fit uses it
    because a full batch of cached vectors has nothing to sample: every training recording is
    present at every step, so balance can only come from the reduction. A fit that draws
    class-balanced batches must therefore use the plain mean instead -- applying both would reweight
    the minority class twice. The natural acidosis/HIE mixture inside the positive class is
    untouched: it is one class here, and its composition is disclosed rather than balanced.

    A class absent from the batch contributes no term rather than a zero, so the reduction stays
    the mean over the classes that are present.

    Args:
        logits: One logit per recording.
        labels: Their binary outcomes, as floats.

    Returns:
        A scalar loss.
    """
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    terms = [
        per_example[labels == value].mean()
        for value in (0.0, 1.0)
        if bool((labels == value).any())
    ]
    return torch.stack(terms).mean()


def is_better(candidate: Mapping[str, Any], best: Optional[Mapping[str, Any]]) -> bool:
    """Whether one step's validation result improves on the best so far.

    The protocol's rule, written once and read by the baseline fit and by the adaptation's
    epoch selection: **highest validation AUROC**, ties broken by lower validation BCE, then by the
    earlier step. Because "earlier" wins a full tie, this returns ``False`` on an exact tie -- the
    incumbent was reached first and keeps its place.

    A non-finite AUROC never improves on anything: it means a split carried one class, which is a
    reason to stop rather than a score to compare.

    Args:
        candidate: A record carrying ``val_auroc`` and ``val_bce``.
        best: The incumbent, or ``None`` when there is none yet.

    Returns:
        Whether the candidate replaces the incumbent.
    """
    auroc = float(candidate.get("val_auroc", float("nan")))
    if not np.isfinite(auroc):
        return False
    if best is None:
        return True
    incumbent = float(best.get("val_auroc", float("nan")))
    if not np.isfinite(incumbent):
        return True
    if auroc != incumbent:
        return auroc > incumbent
    return float(candidate.get("val_bce", float("inf"))) < float(
        best.get("val_bce", float("inf"))
    )


# =============================================================================
# The frozen baseline
# =============================================================================
@dataclass(frozen=True)
class BaselineFit:
    """A fitted linear classifier, its validation threshold, and how it got there.

    Attributes:
        classifier: The fitted :class:`~latent_pilot.model.LatentClassifier`, carrying the frozen
            scaler as buffers so it can never be scored under different constants.
        threshold: The decision threshold, chosen on validation and never on test.
        history: One row per optimization step: the training loss and the validation AUROC and BCE
            it was selected against.
        record: The selected step, the budget it was allowed, the counts behind it and the
            threshold's own record.
    """

    classifier: Any
    threshold: float
    history: pd.DataFrame
    record: Dict[str, Any]

    def logits(self, values: Any) -> np.ndarray:
        """Score a bag matrix under this classifier.

        Args:
            values: ``(N, d_z)`` pooled latent vectors, unstandardized.

        Returns:
            One logit per row. Logits, not probabilities: the fit is class-balanced, so the sigmoid
            of this is an association score and not a calibrated clinical risk.
        """
        self.classifier.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
            return self.classifier(tensor).detach().cpu().numpy()


def fit_baseline(
    train_bags: Bags,
    val_bags: Bags,
    *,
    scaler: Any,
    lr: float,
    max_steps: int,
    patience: int,
    weight_decay: float,
    seed: int,
    # The pipeline's shuffled-label control does NOT come through here: `fit_control_baseline`
    # permutes the recording TABLE and lets `build_bags` carry the permuted outcomes, so the
    # control walks the same code path as the run it controls. This argument is the direct-mapping
    # equivalent, used by the contract tests, and it sets the same `labels_permuted` flag.
    labels: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> BaselineFit:
    r"""Fit the frozen baseline's linear classifier on cached pretrained bags.

    **No model is involved.** The bags are pooled posterior means already extracted from the
    pretrained checkpoint, so this function receives no net, holds no gradient that could reach one,
    and cannot move a VAE weight -- which is what makes "the baseline is the frozen model" a
    property of the code rather than a claim about it. The comparison this baseline anchors is
    therefore between two representations, one of which was not touched.

    Full batch, because the whole training split is a few hundred vectors: there is no sampling
    here at all, and the class balance is entirely :func:`balanced_bce`'s. Validation is scored at
    every step -- also cheap on cached vectors -- and the budget is finite and logged rather than
    "until it stops improving" with no bound.

    Selection is :func:`is_better` applied to the validation split at natural prevalence, and the
    returned classifier is the **selected** step's, not the last one's.

    Args:
        train_bags: The training bags.
        val_bags: The validation bags.
        scaler: The frozen training scaler; its constants are copied into the classifier.
        lr: Learning rate for AdamW.
        max_steps: The finite budget.
        patience: Steps without an improved validation AUROC before stopping.
        weight_decay: AdamW's decay.
        seed: Seeds the classifier's initialization. The global RNG state is restored afterwards,
            so a fit cannot silently advance another stage's stream.
        labels: Optional GUID -> label override, used by the shuffled-label control to fit the same
            path under a permutation. Keyed by split name, then by GUID.

    Returns:
        The fit.

    Raises:
        PilotConfigError: If the two bag tables are not train and validation, if their widths
            disagree with the scaler, or if either carries one class only.
    """
    for bags, expected in ((train_bags, "train"), (val_bags, "val")):
        if bags.split != expected:
            raise PilotConfigError(
                f"the {expected!r} argument was given {bags.split!r} bags. Fitting and selection "
                f"read train and validation only; the test split is opened once, from the "
                f"evaluation stage, after everything here is locked."
            )
    d_z = int(train_bags.values.shape[1])
    if int(val_bags.values.shape[1]) != d_z or int(scaler.center.size) != d_z:
        raise PilotConfigError(
            f"the bags are {d_z}- and {int(val_bags.values.shape[1])}-dimensional and the scaler "
            f"is {int(scaler.center.size)}-dimensional; all three describe the same latent and a "
            f"mismatch means two of them came from different extractions."
        )

    train_y = _labels_for(train_bags, labels)
    val_y = _labels_for(val_bags, labels)
    for name, values in (("train", train_y), ("val", val_y)):
        if len(set(values.tolist())) < 2:
            raise PilotConfigError(
                f"the {name!r} bags carry one binary class after labelling, so the objective "
                f"cannot balance and the selection metric is undefined."
            )

    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        classifier = pilot_model.LatentClassifier(
            d_z, center=scaler.center, scale=scaler.scale
        )
    finally:
        torch.random.set_rng_state(state)

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    train_x = torch.as_tensor(np.asarray(train_bags.values, dtype=np.float32))
    val_x = torch.as_tensor(np.asarray(val_bags.values, dtype=np.float32))
    train_labels = torch.as_tensor(train_y.astype(np.float32))

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    # Stopping and selection are two rules, and they are deliberately not the same one. Patience
    # counts steps without an improved **AUROC**, which is what the protocol stops on; selection is
    # AUROC then BCE then the earlier step, which can keep moving while AUROC stands still. Folded
    # into one rule, a fit whose AUROC had saturated would run its whole budget chasing a
    # cross-entropy that is only ever a tie-break.
    best_auroc = -float("inf")
    since_improvement = 0
    stopped = "budget_exhausted"

    # The budget is finite and the bar states how much of it is left; early stopping ends it sooner
    # and the bar closes where it stopped, which is itself the useful fact.
    for step in tqdm(range(1, int(max_steps) + 1), desc="baseline", unit="step"):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_bce(classifier(train_x), train_labels)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_x).detach().cpu().numpy()
        entry = {
            "step": step,
            "train_loss": float(loss.detach().item()),
            # Natural prevalence on validation, as the protocol keeps it: the balanced reduction is
            # the fitting objective's, and reporting it here would grade the fit on its own terms.
            "val_auroc": evaluate.auroc(val_y, val_logits),
            "val_bce": evaluate.binary_cross_entropy(val_y, val_logits),
        }
        history.append(entry)

        if is_better(entry, best):
            best = dict(entry)
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in classifier.state_dict().items()
            }
        # Checked after selection, so a step that ties the best AUROC with a lower BCE is still
        # taken on the step that ends the run.
        if np.isfinite(entry["val_auroc"]) and entry["val_auroc"] > best_auroc:
            best_auroc = float(entry["val_auroc"])
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= int(patience):
                stopped = "patience_exhausted"
                break

    if best is None or best_state is None:
        raise PilotConfigError(
            f"no step of the baseline fit produced a finite validation AUROC over "
            f"{len(history)} step(s), so there is nothing to select. This happens when validation "
            f"carries one binary class, which is checked before fitting and reported there."
        )

    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_x).detach().cpu().numpy()
    threshold = evaluate.select_threshold(val_y, val_logits)

    record = {
        "name": BASELINE_NAME,
        "selected_step": int(best["step"]),
        "selected_val_auroc": float(best["val_auroc"]),
        "selected_val_bce": float(best["val_bce"]),
        "selected_train_loss": float(best["train_loss"]),
        "n_steps_run": len(history),
        "max_steps": int(max_steps),
        "patience": int(patience),
        "stopped_because": stopped,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "d_z": d_z,
        "balance": "recording-level class weights in the loss; no sampling anywhere in this fit",
        "selection": "highest validation AUROC, then lower validation BCE, then the earlier step",
        "stopping": "steps since the validation AUROC last improved, which is not the tie-break",
        "labels_permuted": labels is not None,
        "train": dict(train_bags.record),
        "val": dict(val_bags.record),
        "threshold": threshold,
        "scaler": dict(getattr(scaler, "record", {}) or {}),
        "note": (
            "fitted on cached pretrained bags: no VAE parameter is reachable from this function, "
            "and the test split is not among its inputs"
        ),
    }
    logger.info(
        f"baseline: selected step {record['selected_step']} of {record['n_steps_run']} "
        f"({stopped}); validation AUROC {record['selected_val_auroc']:.4f}, BCE "
        f"{record['selected_val_bce']:.4f}, threshold {threshold['threshold']:.4f} at balanced "
        f"accuracy {threshold['balanced_accuracy']:.4f}"
    )
    return BaselineFit(
        classifier=classifier,
        threshold=float(threshold["threshold"]),
        history=pd.DataFrame(history),
        record=record,
    )


def _labels_for(
    bags: Bags, labels: Optional[Mapping[str, Mapping[str, int]]]
) -> np.ndarray:
    """The outcomes a fit uses for one split, permuted where a control supplies a mapping.

    The permutation is applied **per GUID**, so every segment of a recording keeps that recording's
    assigned label and the control tests the association rather than a within-recording shuffle
    that would not exist in the data.

    Args:
        bags: The split's bags.
        labels: GUID -> label, keyed by split, or ``None`` for the true outcomes.

    Returns:
        The labels, aligned with the bag rows.

    Raises:
        PilotConfigError: If a mapping is supplied for this split but omits one of its recordings.
    """
    if labels is None or bags.split not in labels:
        return bags.labels
    mapping = labels[bags.split]
    missing = [guid for guid in bags.guids if guid not in mapping]
    if missing:
        raise PilotConfigError(
            f"the label mapping for split {bags.split!r} omits {len(missing)} recording(s), e.g. "
            f"{missing[:5]}. A permutation must cover every recording it is applied to, or the "
            f"control would be fitted on a different population from the run it controls."
        )
    return np.asarray([int(mapping[guid]) for guid in bags.guids], dtype=np.int64)


# =============================================================================
# Persistence
# =============================================================================
def save_fit(fit: BaselineFit, directory: Any, *, name: str = BASELINE_NAME) -> Path:
    """Write a fitted classifier and its record into a run directory.

    **Separately from the base model, always.** The classifier is not part of the checkpoint's
    architecture, and pushing its keys through the strict loader that rebuilds the net would either
    be refused or -- worse -- accepted by a loader that tolerated extras. The pretrained checkpoint
    is never written to at all.

    Args:
        fit: The fit.
        directory: The run directory. Created if absent.
        name: The prefix its two files are filed under.

    Returns:
        The directory they were written into.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": fit.classifier.state_dict(),
            "d_z": int(fit.classifier.d_z),
            "threshold": float(fit.threshold),
            "record": fit.record,
        },
        path / f"{name}_{CLASSIFIER_FILENAME}",
    )
    (path / f"{name}_{FIT_FILENAME}").write_text(
        json.dumps(
            {"record": fit.record, "history": fit.history.to_dict(orient="records")},
            indent=2, sort_keys=True, default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"wrote the {name!r} fit ({len(fit.history)} step(s)) to {path}")
    return path


def load_fit(directory: Any, *, name: str = BASELINE_NAME) -> BaselineFit:
    """Read a fitted classifier back, with its scaler and threshold.

    Args:
        directory: The run directory.
        name: The prefix used when it was written.

    Returns:
        The fit. Its history comes from the JSON companion, so a reader that only wants the
        classifier still gets the budget it was fitted under.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    path = Path(directory)
    classifier_path = path / f"{name}_{CLASSIFIER_FILENAME}"
    fit_path = path / f"{name}_{FIT_FILENAME}"
    for candidate in (classifier_path, fit_path):
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{candidate} is missing, so the {name!r} fit cannot be read back. The frozen "
                f"baseline is fitted before the adaptation, which initializes its classifier from "
                f"this file."
            )
    blob = torch.load(classifier_path, map_location="cpu", weights_only=False)
    classifier = pilot_model.LatentClassifier(int(blob["d_z"]))
    classifier.load_state_dict(blob["state_dict"])
    classifier.eval()
    payload = json.loads(fit_path.read_text(encoding="utf-8"))
    return BaselineFit(
        classifier=classifier,
        threshold=float(blob["threshold"]),
        history=pd.DataFrame(payload.get("history") or []),
        record=dict(payload.get("record") or blob.get("record") or {}),
    )



# =============================================================================
# The recording source: one recording's segments at a time
# =============================================================================
@dataclass(frozen=True)
class RecordingSource:
    """Random access to a split's segments, grouped by recording.

    The adaptation's unit of supervision is a **recording**, not a segment: one bag, one
    classification loss, and a preservation term over that recording's whole three-hour support.
    The ordinary dataloader yields segments in shard order and cannot be asked for a particular
    recording, so this holds the dataset itself and an index into it.

    Segments are fetched and collated directly rather than through a ``DataLoader``: the pilot's
    loader is single-process anyway, the batch is one recording's few segments, and going through
    the dataset gives the caller the one thing a sampler cannot -- the rows come back in the order
    the plan asks for, so a bag's segment weights cannot be applied to the wrong rows.

    Attributes:
        dataset: The ``CombinedHDF5Dataset`` over this split's shards.
        index: ``(guid, rounded epoch)`` -> dataset index.
        record: How the index was built, including any segment key seen twice.
    """

    dataset: Any
    index: Dict[Tuple[str, float], int]
    record: Dict[str, Any]

    @staticmethod
    def _key(guid: Any, epoch: Any) -> Tuple[str, float]:
        """The index key. Rounded, because one side of it comes back through a float32 tensor."""
        return (str(guid), round(float(epoch), 3))

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, shards: Sequence[str]
    ) -> "RecordingSource":
        """Build the source from the pilot's own loader configuration.

        The dataset is constructed with exactly the keyword arguments the split's loader uses, so
        the trim, the normalisation fields and every filter are the ones the extraction ran under
        -- which is what makes the cached teacher below the same quantity as a live forward.

        Args:
            config: The pilot loader configuration from ``data.pilot_loader_config``.
            shards: This split's shards.

        Returns:
            The source.

        Raises:
            PilotConfigError: If the dataset holds no sample at all.
        """
        from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

        dataset_config = dict(config.get("dataset_config") or {})
        loader_config = dict(dataset_config.get("dataloader_config") or {})
        dataset = CombinedHDF5Dataset(
            paths=list(shards),
            stats_path=dataset_config.get("stat_path"),
            normalize_fields=loader_config.get("normalize_fields"),
            **dict(loader_config.get("dataset_kwargs") or {}),
        )
        # File-grouped and sample-ascending, which is the order ``index_map`` is built in, so the
        # position in these lists is the dataset index.
        guids, epochs, _targets = dataset.get_the_lists()
        index: Dict[Tuple[str, float], int] = {}
        n_repeated = 0
        for position, (guid, epoch) in enumerate(zip(guids, epochs)):
            key = cls._key(guid, epoch)
            if key in index:
                n_repeated += 1
                continue
            index[key] = position
        if not index:
            raise PilotConfigError(
                f"the shards {list(shards)!r} hold no segment after the loader's filters, so no "
                f"recording can be presented to the adaptation."
            )
        record = {
            "n_samples": len(guids),
            "n_segments_indexed": len(index),
            "n_repeated_segment_keys": n_repeated,
            "n_recordings": len({guid for guid, _epoch in index}),
            "shards": [str(shard) for shard in shards],
        }
        if n_repeated:
            logger.warning(
                f"{n_repeated} segment key(s) appear more than once in these shards; the first is "
                f"indexed and the rest are not presented, so one instant cannot enter a bag twice"
            )
        return cls(dataset=dataset, index=index, record=record)

    def batch(self, guid: str, epochs: Sequence[float]) -> Any:
        """Collate one recording's segments, in the order asked for.

        Args:
            guid: The recording.
            epochs: Its segment starts, in the plan's order.

        Returns:
            The collated batch, on the CPU. Row ``i`` is ``epochs[i]``; the move to the model's
            device happens at the forward, through the task's own transfer hook.

        Raises:
            PilotConfigError: If a segment is not in this split's shards.
        """
        from hdf5_dataset.hdf5_dataset import attribute_dict_collate

        positions: List[int] = []
        for epoch in epochs:
            key = self._key(guid, epoch)
            if key not in self.index:
                raise PilotConfigError(
                    f"segment {key} is in the extraction but not in this split's shards. The plan "
                    f"and the source must be built from the same shard list, or a bag would be "
                    f"pooled over segments the model was never shown."
                )
            positions.append(self.index[key])
        return attribute_dict_collate([self.dataset[position] for position in positions])


# =============================================================================
# The plan: which anchors, which segments, and what the teacher said there
# =============================================================================
@dataclass(frozen=True)
class SegmentSupport:
    """One segment's retained anchors and the pretrained means at them.

    Attributes:
        epoch: The segment start, which is how the source finds it.
        anchors: The retained anchor indices, ascending. They are **decimated step indices**, so
            they index the latent's own time axis directly and carry no assumption about how the
            forward laid out its anchor axis.
        hours: Their times before delivery.
        late: Which of them fall inside the supervised window.
        teacher: The pretrained ``mu_post`` at those anchors, ``(n_anchors, d_z)``.
        median_late_hours: The median time of the late anchors, which is what the recency weight
            is computed from -- not the segment's start, and not a nominal window's midpoint.
    """

    epoch: float
    anchors: np.ndarray
    hours: np.ndarray
    late: np.ndarray
    teacher: np.ndarray
    median_late_hours: float

    @property
    def has_late(self) -> bool:
        """Whether this segment contributes to the supervised bag."""
        return bool(self.late.any())


@dataclass(frozen=True)
class RecordingPlan:
    """Everything the adaptation needs about one recording, precomputed once.

    Attributes:
        guid: The recording.
        outcome: Its binary outcome.
        segments: Its segments carrying at least one retained anchor.
        weights: The recency weights of the segments that reach the supervised window, already
            normalised and aligned with :attr:`late_segments`.
        late_segments: Indices into :attr:`segments` of the ones that contribute to the bag.
    """

    guid: str
    outcome: int
    segments: Tuple[SegmentSupport, ...]
    weights: np.ndarray
    late_segments: Tuple[int, ...]

    @property
    def epochs(self) -> List[float]:
        """The segment starts, in the order the source must return them."""
        return [segment.epoch for segment in self.segments]

    @property
    def n_anchors(self) -> int:
        """How many retained anchors this recording contributes to the preservation term."""
        return int(sum(segment.anchors.size for segment in self.segments))


def build_plans(
    extraction: Any,
    recordings: pd.DataFrame,
    *,
    split: str,
    supervised_hours: float,
    halflife_hours: float,
) -> Dict[str, RecordingPlan]:
    r"""Precompute every recording's support, weights and teacher values from one extraction.

    **The support is not recomputed during training.** It is read off the extraction, which already
    applied the forecast-contributing rule, the window, the post-delivery endpoint check and both
    duplication rules -- so the anchors the student is supervised and held at are, by construction,
    the anchors every table and figure in this run is built from. Recomputing it per step would be
    a second definition, and would put a pandas deduplication inside the inner loop of a fit.

    **The teacher is cached, not re-forwarded.** $\mu^q_0$ is a deterministic function of the
    inputs, and the pretrained extraction is exactly that function evaluated at exactly these
    anchors under exactly this loader contract; a second forward through a frozen copy would
    recompute numbers this run already has. The identity is verified rather than assumed: the
    student's own first forward, before any update, must reproduce these values -- see
    :func:`fit_adaptation`.

    Args:
        extraction: The **pretrained** extraction of the split.
        recordings: The recording table, with outcomes and eligibility attached.
        split: The split being planned, checked against the extraction.
        supervised_hours: The supervised window's upper edge.
        halflife_hours: The recency half-life inside it.

    Returns:
        GUID -> plan, for the eligible recordings that reach the supervised window.

    Raises:
        PilotConfigError: If the extraction is another split's, or if no recording qualifies.
    """
    frame = extraction.retained
    present = sorted({str(value) for value in frame[data.SPLIT_COLUMN].tolist()})
    if present != [str(split)]:
        raise PilotConfigError(
            f"the extraction carries split(s) {present} but plans were requested for {split!r}."
        )

    _eligible, outcomes = data.eligible_anchors(frame, recordings, split=split)

    teacher = np.asarray(extraction.arrays["mu_post"], dtype=np.float32)
    plans: Dict[str, RecordingPlan] = {}
    for guid, group in frame.groupby(data.GUID_COLUMN, sort=True):
        guid = str(guid)
        if guid not in outcomes:
            continue
        segments: List[SegmentSupport] = []
        for epoch, rows in group.groupby(data.EPOCH_COLUMN, sort=True):
            positions = np.asarray(rows[data.ROW_COLUMN], dtype=np.int64)
            anchors = np.asarray(rows[data.ANCHOR_COLUMN], dtype=np.int64)
            hours = np.asarray(rows[data.HOURS_COLUMN], dtype=np.float64)
            order = np.argsort(anchors, kind="stable")
            positions, anchors, hours = positions[order], anchors[order], hours[order]
            late = data.in_window(hours, 0.0, float(supervised_hours))
            segments.append(SegmentSupport(
                epoch=float(epoch),
                anchors=anchors,
                hours=hours,
                late=late,
                teacher=teacher[positions],
                median_late_hours=(
                    float(np.median(hours[late])) if late.any() else float("nan")
                ),
            ))
        late_segments = tuple(
            index for index, segment in enumerate(segments) if segment.has_late
        )
        if not late_segments:
            continue
        weights = data.recency_weights(
            [segments[index].median_late_hours for index in late_segments],
            halflife_hours=halflife_hours,
        )
        plans[guid] = RecordingPlan(
            guid=guid,
            outcome=int(outcomes[guid]),
            segments=tuple(segments),
            weights=np.asarray(weights / weights.sum(), dtype=np.float64),
            late_segments=late_segments,
        )

    if not plans:
        raise PilotConfigError(
            f"no eligible {split!r} recording reaches the supervised window "
            f"(0, {supervised_hours}] h, so the adaptation has nothing to supervise."
        )
    counts = {
        outcome: sum(1 for plan in plans.values() if plan.outcome == outcome)
        for outcome in (0, 1)
    }
    logger.info(
        f"{split} plans: {len(plans)} recording(s) ({counts[0]} healthy, {counts[1]} adverse), "
        f"{sum(len(plan.segments) for plan in plans.values())} segment(s), "
        f"{sum(plan.n_anchors for plan in plans.values())} preserved anchor(s)"
    )
    return plans


def epoch_batches(
    plans: Mapping[str, RecordingPlan], *, per_class: int, seed: int, epoch: int
) -> List[List[str]]:
    """Order one epoch's recordings into class-balanced batches of distinct recordings.

    **This is the sampling arm of the protocol's either/or**, and it is why the classification loss
    below is a plain mean rather than :func:`balanced_bce`: the batch is already balanced, and
    weighting it again would reweight the minority class twice.

    Each class is drawn in **whole blocks cut from a fresh permutation of its own pool**, which is
    what makes the recordings in one batch distinct: a stream built by concatenating permutations
    and then slicing it could straddle two of them and present one recording twice in a batch of
    eight. A permutation's trailing remainder -- the recordings left when the pool is not a multiple
    of the block size -- is dropped from that cycle and picked up by the next reshuffle, so coverage
    within a single epoch is exact only when the block divides the pool. That is the price of
    distinctness, and it is the right way round: an epoch is a bookkeeping unit, while a batch
    holding one recording twice would weight it twice in a loss that is defined per recording.

    When the smaller class has fewer recordings than ``per_class`` the **batch shrinks** to that
    class's size rather than repeating a recording inside it -- the protocol's rule, and the reason
    its counterpart is gradient accumulation rather than a bigger batch. The number of batches is
    one pass over the **larger** class; the smaller one is cycled through reshuffled permutations.

    Args:
        plans: The split's plans.
        per_class: Recordings per binary class per batch; the batch is twice this by construction.
        seed: The run's seed.
        epoch: The epoch index, mixed into the seed so two epochs differ.

    Returns:
        A list of batches, each a list of GUIDs -- the healthy ones first, then the adverse.

    Raises:
        PilotConfigError: If either class is empty, where a balanced batch cannot be formed.
    """
    pools = {
        outcome: sorted(guid for guid, plan in plans.items() if plan.outcome == outcome)
        for outcome in (0, 1)
    }
    if not pools[0] or not pools[1]:
        raise PilotConfigError(
            f"the plans carry {{0: {len(pools[0])}, 1: {len(pools[1])}}} recording(s) by binary "
            f"outcome; a class-balanced batch needs both."
        )

    size = min(int(per_class), len(pools[0]), len(pools[1]))
    n_batches = int(np.ceil(max(len(pools[0]), len(pools[1])) / size))
    generator = np.random.default_rng([int(seed), int(epoch)])

    blocks: Dict[int, List[List[str]]] = {}
    for outcome, pool in pools.items():
        drawn: List[List[str]] = []
        while len(drawn) < n_batches:
            order = [pool[index] for index in generator.permutation(len(pool))]
            # Whole blocks only: the remainder of this permutation is left for the next one, so
            # every block is a set of distinct recordings rather than a slice across two shuffles.
            for start in range(0, len(order) - size + 1, size):
                drawn.append(order[start:start + size])
        blocks[outcome] = drawn

    return [
        [*blocks[0][index], *blocks[1][index]] for index in range(n_batches)
    ]


# =============================================================================
# The differentiable per-recording terms
# =============================================================================
def recording_terms(
    task: Any,
    plan: RecordingPlan,
    batch: Any,
    *,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""One recording's supervised bag and its preservation term, with gradients.

    $$e_{is} = \frac{1}{|A^{\rm late}_{is}|} \sum_{a} \mu^q_{isa}, \qquad
    v_i = \frac{\sum_s \omega_{is} e_{is}}{\sum_s \omega_{is}},$$

    $$\mathcal L^{(i)}_{\rm keep} = \operatorname{mean}_{s \in i} \operatorname{mean}_{a}
    \frac{1}{d_z} \left\| \frac{\mu^q_{\theta,isa} - \mu^q_{0,isa}}{s} \right\|_2^2 .$$

    The latent is gathered at the anchors' own **step indices**, straight out of $\mu^q$'s time
    axis, so nothing here depends on how the forward laid out its anchor axis. The teacher enters
    as a constant: it is stored data, carries no graph, and cannot be moved by the update it is
    the reference for.

    The preservation term runs over the **whole** preserved window, including the two earlier hours
    that carry no outcome label. Those anchors receive preservation supervision only: they are not
    relabelled healthy, and nothing here asks them to change monotonically or at all.

    Args:
        task: The loaded task, whose seam assembles the forward's arguments.
        plan: The recording's plan.
        batch: Its segments, collated in ``plan.epochs`` order.
        scale: The frozen scaler's per-coordinate scales, ``(d_z,)``, on the model's device.

    Returns:
        ``(bag, keep, teacher_gap)``: the pooled bag ``(d_z,)`` with gradient, the recording's
        preservation term, and the largest absolute student-minus-teacher difference on this
        recording -- zero before the first update, and the check that the cached teacher is the
        live forward's own value.
    """
    batch = pilot_model.to_device(task, batch)
    outputs = task.orig_model(*pilot_model.forward_inputs(task, batch))
    mu = outputs["mu_post"]
    device, dtype = mu.device, mu.dtype

    bag_parts: List[torch.Tensor] = []
    keep_parts: List[torch.Tensor] = []
    gaps: List[torch.Tensor] = []
    d_z = float(mu.shape[-1])
    for row, segment in enumerate(plan.segments):
        anchors = torch.as_tensor(segment.anchors, dtype=torch.long, device=device)
        gathered = mu[row].index_select(0, anchors)
        teacher = torch.as_tensor(segment.teacher, dtype=dtype, device=device)
        residual = (gathered - teacher) / scale
        keep_parts.append(residual.pow(2).sum(dim=-1).div(d_z).mean())
        gaps.append((gathered - teacher).abs().max().detach())
        if segment.has_late:
            late = torch.as_tensor(
                np.flatnonzero(segment.late), dtype=torch.long, device=device
            )
            bag_parts.append(gathered.index_select(0, late).mean(dim=0))

    weights = torch.as_tensor(plan.weights, dtype=dtype, device=device)
    bag = (weights[:, None] * torch.stack(bag_parts)).sum(dim=0)
    return bag, torch.stack(keep_parts).mean(), torch.stack(gaps).max()


# =============================================================================
# The adaptation
# =============================================================================
@dataclass(frozen=True)
class AdaptationFit:
    """The selected candidate, the history that chose it, and what it was chosen against.

    Attributes:
        mean_head_state: The selected epoch's ``delta_mu_head`` weights. Only those: everything
            else in the net is the pretrained checkpoint's and is never written out as if it had
            been trained.
        classifier: The selected epoch's classifier.
        threshold: Its validation threshold.
        selected_epoch: Which epoch was selected. ``0`` is the frozen model, retained when no
            adapted candidate improved eligible validation AUROC.
        history: One row per epoch, including the ones the gates rejected and why.
        record: The selection, the budget, the objective's weights and the counts behind them.
    """

    mean_head_state: Dict[str, torch.Tensor]
    classifier: Any
    threshold: float
    selected_epoch: int
    history: pd.DataFrame
    record: Dict[str, Any]

    @property
    def adapted(self) -> bool:
        """Whether an adapted candidate was selected at all."""
        return self.selected_epoch > 0


def _mean_head_state(model: Any) -> Dict[str, torch.Tensor]:
    """Snapshot the trainable mean-output weights, detached from the graph."""
    return {
        name: parameter.detach().clone()
        for name, parameter in pilot_model.mean_head_parameters(model)
    }


def _load_mean_head_state(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    """Write a snapshot back into the model's mean-output weights."""
    parameters = dict(pilot_model.mean_head_parameters(model))
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor)


def fit_adaptation(
    loaded: Any,
    *,
    source: RecordingSource,
    plans: Mapping[str, RecordingPlan],
    val_loader: Any,
    recordings: pd.DataFrame,
    scaler: Any,
    baseline: BaselineFit,
    gate_guids: Sequence[str],
    gate_baseline: Mapping[str, Any],
    outcomes: Mapping[str, Optional[int]],
    settings: Mapping[str, Any],
    teacher_tolerance: float = 1e-4,
) -> AdaptationFit:
    r"""Fine-tune the posterior mean-output layers and the classifier, and select an epoch.

    $$\mathcal L = \lambda_{\rm cls} \operatorname{mean}_i
    \operatorname{BCEWithLogits}(\ell_i, y_i) + \lambda_{\rm keep} \operatorname{mean}_i
    \mathcal L^{(i)}_{\rm keep},$$

    with the classification mean taken over a **class-balanced batch**, which is why it is a plain
    mean: :func:`epoch_batches` is the sampling arm of the protocol's either/or, and
    :func:`balanced_bce` -- the weighting arm -- is deliberately not applied on top of it.

    Nothing from the pretraining run is inherited. The optimizer is built here, from
    ``pilot_model.parameter_groups`` -- an explicit allowlist rather than a ``requires_grad``
    filter -- with two learning rates and no scheduler at all, so neither the $\beta$ warm-up nor
    the 2000-step learning-rate ramp can re-enter through a trainer default. The backbone stays in
    evaluation mode throughout, which is checked before **every** step rather than once at the
    start: a stray ``train()`` would put dropout back on and make the student's forward and its
    teacher incomparable without changing a single shape.

    Each recording's loss is backpropagated as it is computed and the step is taken once the batch
    is complete, so memory is a function of one recording rather than of the batch size. That is
    the protocol's accumulation, and it is what makes a batch of eight recordings affordable when
    one of them carries a dozen segments.

    Selection runs **gates first**: a candidate that fails preservation is not eligible however
    good its validation AUROC is, and its rejection reason is recorded rather than dropped. Among
    eligible candidates the rule is :func:`is_better`. Epoch zero is the frozen model with the
    baseline classifier and is always eligible -- it is the reference the gates are measured
    against -- so a run in which no adapted candidate improves selection retains and reports the
    pretrained model rather than an arbitrary changed one.

    Args:
        loaded: The loaded checkpoint bundle, already frozen by ``model.freeze_for_pilot``.
        source: Random access to the training split's segments.
        plans: The training split's plans.
        val_loader: The validation dataloader, re-iterated once per epoch for the bags and once for
            the gate.
        recordings: The recording table, for the validation bags.
        scaler: The frozen training scaler.
        baseline: The frozen baseline's fit; its classifier initialises this one.
        gate_guids: The fixed preservation subset.
        gate_baseline: The frozen model's preservation record, which the gates compare against.
        outcomes: GUID -> binary outcome, for the gate's healthy-only readout.
        settings: The resolved pilot settings; ``windows``, ``optim`` and ``gates`` are read.
        teacher_tolerance: How far the student's first forward may sit from the cached teacher
            before the cache is refused.

    Returns:
        The fit, with the model and classifier left holding the selected epoch's weights.

    Raises:
        PilotConfigError: If the freeze has drifted, or if the cached teacher does not reproduce
            the student's own first forward.
    """
    model, task = loaded.model, loaded.task
    device = next(model.parameters()).device
    windows, optim, gates = settings["windows"], settings["optim"], settings["gates"]

    pilot_model.check_pilot_mode(model)
    original_state = _mean_head_state(model)
    classifier = pilot_model.LatentClassifier(
        int(scaler.center.size), center=scaler.center, scale=scaler.scale
    ).to(device)
    classifier.load_state_dict(baseline.classifier.state_dict())
    scale = torch.as_tensor(
        np.asarray(scaler.scale, dtype=np.float32), device=device
    )

    optimizer = torch.optim.AdamW(
        pilot_model.parameter_groups(
            model,
            classifier,
            mean_head_lr=float(optim["mean_head_lr"]),
            classifier_lr=float(optim["classifier_lr"]),
            weight_decay=float(optim["weight_decay"]),
        )
    )
    trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    def _validate(epoch: int) -> Tuple[Dict[str, Any], Any, Any]:
        """Score this epoch's model on validation, and put it through the gates."""
        extraction = extract.extract_split(
            loaded,
            val_loader,
            split="val",
            preservation_hours=float(windows["preservation_hours"]),
            bin_hours=float(windows["bin_hours"]),
        )
        bags = build_bags(
            extraction,
            recordings,
            split="val",
            supervised_hours=float(windows["supervised_hours"]),
            halflife_hours=float(settings["bag"]["halflife_hours"]),
        )
        classifier.eval()
        with torch.no_grad():
            logits = classifier(
                torch.as_tensor(np.asarray(bags.values, dtype=np.float32), device=device)
            ).detach().cpu().numpy()
        if epoch == 0:
            # The caller measured this exact reading to obtain `gate_baseline`: same bundle, same
            # loader, same gate subset, same window, and no optimizer step has run yet -- only the
            # classifier has been loaded, which the preservation pass does not touch. Taking it
            # again would spend one full validation forward to recompute a number already in hand.
            record = dict(gate_baseline)
            reading = None
            gate = _reference_gate(record)
        else:
            reading = evaluate.preservation_pass(
                loaded,
                val_loader,
                guids=gate_guids,
                outcomes=outcomes,
                preservation_hours=float(windows["preservation_hours"]),
            )
            record = reading.record
            gate = evaluate.gate_decision(
                gate_baseline,
                record,
                forecast_mse_max_increase=float(gates["forecast_mse_max_increase"]),
                saturation_max_increase_pp=float(gates["saturation_max_increase_pp"]),
            )
        entry = {
            "epoch": epoch,
            "val_auroc": evaluate.auroc(bags.labels, logits),
            "val_bce": evaluate.binary_cross_entropy(bags.labels, logits),
            "gate_passed": bool(gate.passed),
            "gate_reasons": "; ".join(gate.reasons),
            "gate_warnings": "; ".join(gate.warnings),
            "mse_full": record["mse_full"],
            "delta_mu_sat_pp": record["delta_mu_sat_pp"],
            "n_val_recordings": int(len(bags.frame)),
        }
        return entry, bags, (logits, gate, reading)

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_classifier: Optional[Dict[str, torch.Tensor]] = None
    best_logits: Optional[np.ndarray] = None
    best_labels: Optional[np.ndarray] = None
    best_auroc = -float("inf")
    since_improvement = 0
    stopped = "budget_exhausted"

    # Epoch zero: the frozen model and the baseline classifier, kept as a candidate so a run in
    # which nothing improves has something honest to report.
    entry, bags, (logits, _gate, _reading) = _validate(0)  # _reading is None here by construction
    entry["train_loss"] = float("nan")
    entry["cls_loss"] = float("nan")
    entry["keep_loss"] = float("nan")
    history.append(entry)
    best, best_epoch = dict(entry), 0
    best_state = {name: tensor.clone() for name, tensor in original_state.items()}
    best_classifier = {
        name: tensor.detach().clone() for name, tensor in classifier.state_dict().items()
    }
    best_logits, best_labels = logits, bags.labels
    best_auroc = float(entry["val_auroc"]) if np.isfinite(entry["val_auroc"]) else -float("inf")

    verified = False
    # Two bars, because the two questions are different: the outer one is how much of the budget is
    # spent, and it carries the numbers that decide the run -- validation AUROC, the epoch selected
    # so far -- so the answer to "where are we" and the answer to "is it working" are on one line.
    # The inner one is within-epoch and does not survive the epoch.
    epochs = tqdm(
        range(1, int(optim["max_epochs"]) + 1), desc="adaptation", unit="epoch",
    )
    for epoch in epochs:
        batches = epoch_batches(
            plans,
            per_class=int(optim["recordings_per_class"]),
            seed=int(settings["seed"]),
            epoch=epoch,
        )
        totals = {"loss": 0.0, "cls": 0.0, "keep": 0.0}
        for guids in tqdm(
            batches, desc=f"epoch {epoch}", unit="batch", leave=False
        ):
            pilot_model.check_pilot_mode(model)
            optimizer.zero_grad(set_to_none=True)
            classifier.train()
            for guid in guids:
                plan = plans[guid]
                bag, keep, gap = recording_terms(
                    task, plan, source.batch(guid, plan.epochs), scale=scale
                )
                if not verified:
                    _verify_teacher(float(gap), teacher_tolerance, guid)
                    verified = True
                logit = classifier(bag[None, :]).squeeze(0)
                label = torch.as_tensor(float(plan.outcome), device=device)
                # A plain mean: the batch is already class-balanced by construction.
                cls = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
                loss = (
                    float(optim["classification_weight"]) * cls
                    + float(optim["preservation_weight"]) * keep
                ) / len(guids)
                loss.backward()
                totals["loss"] += float(loss.detach().item())
                totals["cls"] += float(cls.detach().item()) / len(guids)
                totals["keep"] += float(keep.detach().item()) / len(guids)
            torch.nn.utils.clip_grad_norm_(trainable, float(optim["grad_clip"]))
            optimizer.step()

        entry, bags, (logits, _gate, _reading) = _validate(epoch)
        entry["train_loss"] = totals["loss"] / max(len(batches), 1)
        entry["cls_loss"] = totals["cls"] / max(len(batches), 1)
        entry["keep_loss"] = totals["keep"] / max(len(batches), 1)
        history.append(entry)
        logger.info(
            f"epoch {epoch}: loss {entry['train_loss']:.4f} (cls {entry['cls_loss']:.4f}, keep "
            f"{entry['keep_loss']:.4f}); validation AUROC {entry['val_auroc']:.4f}, BCE "
            f"{entry['val_bce']:.4f}; gate "
            f"{'passed' if entry['gate_passed'] else 'FAILED: ' + entry['gate_reasons']}"
        )

        epochs.set_postfix(
            auroc=f"{entry['val_auroc']:.3f}",
            loss=f"{entry['train_loss']:.3f}",
            gate="ok" if entry["gate_passed"] else "FAIL",
            best=best_epoch,
        )

        eligible = bool(entry["gate_passed"])
        if eligible and is_better(entry, best):
            best, best_epoch = dict(entry), epoch
            best_state = _mean_head_state(model)
            best_classifier = {
                name: tensor.detach().clone()
                for name, tensor in classifier.state_dict().items()
            }
            best_logits, best_labels = logits, bags.labels
        if eligible and np.isfinite(entry["val_auroc"]) and entry["val_auroc"] > best_auroc:
            best_auroc = float(entry["val_auroc"])
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= int(optim["patience"]):
                stopped = "patience_exhausted"
                break

    # Closed explicitly: the loop above leaves by ``break`` whenever patience runs out, and an
    # unclosed bar keeps the cursor on its own line for everything logged afterwards.
    epochs.close()

    _load_mean_head_state(model, best_state)
    classifier.load_state_dict(best_classifier)
    classifier.eval()
    threshold = evaluate.select_threshold(best_labels, best_logits)

    rejected = [
        {"epoch": int(row["epoch"]), "reasons": row["gate_reasons"]}
        for row in history if not row["gate_passed"]
    ]
    record = {
        "name": "finetuned",
        "selected_epoch": best_epoch,
        "adapted": best_epoch > 0,
        "selected_val_auroc": float(best["val_auroc"]),
        "selected_val_bce": float(best["val_bce"]),
        "n_epochs_run": len(history) - 1,
        "max_epochs": int(optim["max_epochs"]),
        "patience": int(optim["patience"]),
        "stopped_because": stopped,
        "n_gate_rejections": len(rejected),
        "gate_rejections": rejected,
        "mean_head_lr": float(optim["mean_head_lr"]),
        "classifier_lr": float(optim["classifier_lr"]),
        "weight_decay": float(optim["weight_decay"]),
        "grad_clip": float(optim["grad_clip"]),
        "recordings_per_class": int(optim["recordings_per_class"]),
        "classification_weight": float(optim["classification_weight"]),
        "preservation_weight": float(optim["preservation_weight"]),
        "balance": "class-balanced sampling of recordings; the loss is a plain mean, not weighted",
        "selection": "gates first, then highest validation AUROC, lower BCE, earlier epoch",
        "stopping": "epochs since an eligible candidate last improved the validation AUROC",
        "teacher": "the pretrained extraction's own mu_post, verified against the first forward",
        "n_train_recordings": len(plans),
        "n_train_anchors": int(sum(plan.n_anchors for plan in plans.values())),
        "threshold": threshold,
        "trainable": pilot_model.describe_trainable(model, classifier),
        "source": dict(source.record),
        "note": (
            "the frozen model is candidate epoch 0; a run selecting it reports that this "
            "adaptation added no demonstrated benefit rather than an arbitrary changed model"
        ),
    }
    logger.info(
        f"adaptation: selected epoch {best_epoch} of {record['n_epochs_run']} ({stopped}); "
        f"validation AUROC {record['selected_val_auroc']:.4f}; "
        f"{record['n_gate_rejections']} candidate(s) rejected by the gates"
    )
    return AdaptationFit(
        mean_head_state={name: tensor.cpu() for name, tensor in best_state.items()},
        classifier=classifier,
        threshold=float(threshold["threshold"]),
        selected_epoch=best_epoch,
        history=pd.DataFrame(history),
        record=record,
    )


def _reference_gate(record: Mapping[str, Any]) -> Any:
    """The frozen model's own gate result: it is the reference, so it passes by definition.

    Stated rather than computed, because comparing the baseline against itself would produce a
    relative increase of zero for a reason that says nothing -- and would fail on a zero baseline
    for a reason that says less.

    Args:
        record: The frozen model's preservation record.

    Returns:
        A passing :class:`~latent_pilot.evaluate.GateResult`.
    """
    return evaluate.GateResult(
        passed=True,
        reasons=[],
        warnings=[],
        record={
            "passed": True,
            "rule": "reference",
            "baseline_mse_full": record.get("mse_full"),
            "candidate_mse_full": record.get("mse_full"),
            "support_digest": record.get("support_digest"),
            "note": "the frozen model is what the gates are measured against",
        },
    )


def _verify_teacher(gap: float, tolerance: float, guid: str) -> None:
    """Refuse a cached teacher that the student's own first forward does not reproduce.

    Before the first update the student **is** the pretrained model, so its ``mu_post`` at these
    anchors must equal the cached values. A difference means the extraction and this fit are not
    reading the same inputs -- a different shard list, a different statistics file, a different
    trim -- and every preservation term afterwards would be measured against the wrong reference.

    Args:
        gap: The largest absolute student-minus-teacher difference on the first recording.
        tolerance: How far apart they may be. Zero on CPU, where the two paths are bit-identical;
            a small positive value covers GPU kernels whose reductions are not.
        guid: The recording, for the message.

    Raises:
        PilotConfigError: If they disagree.
    """
    if not (gap <= float(tolerance)):
        raise PilotConfigError(
            f"the cached teacher does not reproduce the student's first forward on recording "
            f"{guid!r}: they differ by {gap:.3g}, above the {float(tolerance):.3g} tolerance. "
            f"Before the first update the two are the same model, so a difference means the "
            f"extraction and this fit are not reading the same inputs -- check that the plans, "
            f"the source and the extraction were built from one shard list, one statistics file "
            f"and one checkpoint."
        )



# =============================================================================
# The pilot checkpoint, and the base-model export beside it
# =============================================================================
#: The run's own checkpoint: the adapted weights, the classifier, and enough identity to refuse to
#: be applied to the wrong model. Named as §8 names it.
PILOT_CHECKPOINT_FILENAME = "pilot_checkpoint.pt"

#: The base-model-compatible export, in a subdirectory of its own so the resolved configuration can
#: sit beside it where ``eval.probe.resolved_config_for`` looks.
ADAPTED_EXPORT_DIRNAME = "adapted_model"
ADAPTED_CHECKPOINT_FILENAME = "adapted_model.ckpt"

#: Bumped when the pilot checkpoint's own layout changes. A reader that does not recognise the
#: version refuses rather than guessing which keys it holds.
PILOT_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class PilotCheckpoint:
    """A saved adaptation: what was changed, what it was changed from, and under what.

    Attributes:
        mean_head_state: The adapted ``delta_mu_head`` weights. **Only** those -- the rest of the
            net is the pretrained checkpoint's, and writing it out as though it had been trained
            would misdescribe a change confined to ``posterior_head.delta_mu_head`` -- whose size
            this run derives and records in ``record["trainable"]["mean_head"]["n_parameters"]``
            -- as a whole model.
        classifier_state: The classifier's state dict, its frozen scaler travelling inside it as
            buffers.
        threshold: The decision threshold, chosen on validation.
        selected_epoch: Which candidate was selected; ``0`` is the frozen model.
        source: The pretrained checkpoint's identity -- path, digest, class and geometry.
        fingerprint: The support fingerprint the latents behind this fit were read under.
        record: The fit's own record.
    """

    mean_head_state: Dict[str, torch.Tensor]
    classifier_state: Dict[str, torch.Tensor]
    threshold: float
    selected_epoch: int
    source: Dict[str, Any]
    fingerprint: Dict[str, Any]
    record: Dict[str, Any]

    def classifier(self) -> Any:
        """Rebuild the classifier this fit selected.

        Returns:
            A :class:`~latent_pilot.model.LatentClassifier` carrying the saved weights and the
            frozen scaler they were fitted with.
        """
        built = pilot_model.LatentClassifier(int(self.classifier_state["linear.weight"].shape[1]))
        built.load_state_dict(self.classifier_state)
        built.eval()
        return built


def _source_record(loaded: Any) -> Dict[str, Any]:
    """The pretrained checkpoint's identity, as a saved artifact must carry it."""
    return {
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_digest": loaded.digest,
        "model_class": dict(loaded.geometry).get("model_class"),
        "d_z": dict(loaded.geometry).get("d_z"),
        "geometry": dict(loaded.geometry),
    }


def save_adapted(
    fit: AdaptationFit,
    loaded: Any,
    directory: Any,
    *,
    fingerprint: Mapping[str, Any],
    name: str = PILOT_CHECKPOINT_FILENAME,
) -> Path:
    """Write the pilot's own checkpoint into a run directory.

    **Standalone, and never through the base model's loader.** The classifier is not part of the
    checkpoint's architecture; sending its keys through the strict load that rebuilds the net would
    either be refused or, worse, accepted by a loader that tolerated extras. So the two travel in
    one file of this package's own shape, and the base-compatible export -- which carries no
    classifier key at all -- is written separately by :func:`export_base_checkpoint`.

    **The pretrained checkpoint is never written to.** This writes inside the run directory, and
    the source's path and digest are recorded here so a later reader can say which file the
    adaptation started from.

    Args:
        fit: The selected fit.
        loaded: The loaded checkpoint bundle, for the source identity.
        directory: The run directory. Created if absent.
        fingerprint: The support fingerprint the fit's latents were read under.
        name: The filename. The pipeline always writes the default: the shuffled-label control is
            a linear probe on the frozen pretrained latents and produces no adapted checkpoint to
            file beside this one. The parameter exists for a control adaptation that this pilot
            deliberately does not run -- see :func:`~latent_pilot.evaluate.control_disclosure`.

    Returns:
        The written path.

    Raises:
        PilotConfigError: If the destination would be the source checkpoint itself.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / name
    _refuse_source_path(target, loaded)
    torch.save(
        {
            "version": PILOT_CHECKPOINT_VERSION,
            "mean_head_state": {
                key: tensor.detach().cpu() for key, tensor in fit.mean_head_state.items()
            },
            "classifier_state": {
                key: tensor.detach().cpu()
                for key, tensor in fit.classifier.state_dict().items()
            },
            "threshold": float(fit.threshold),
            "selected_epoch": int(fit.selected_epoch),
            "source": _source_record(loaded),
            "fingerprint": dict(fingerprint),
            "record": fit.record,
        },
        target,
    )
    logger.info(
        f"wrote the pilot checkpoint (epoch {fit.selected_epoch}, "
        f"{sum(tensor.numel() for tensor in fit.mean_head_state.values())} adapted parameter(s)) "
        f"to {target}"
    )
    return target


def load_adapted(directory: Any, *, name: str = PILOT_CHECKPOINT_FILENAME) -> PilotCheckpoint:
    """Read a pilot checkpoint back.

    Args:
        directory: The run directory.
        name: The filename it was written under.

    Returns:
        The checkpoint.

    Raises:
        FileNotFoundError: If it was never written.
        PilotConfigError: If it was written by a layout this reader does not recognise.
    """
    target = Path(directory) / name
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} is missing, so the adapted weights cannot be read back. The finetune stage "
            f"writes it once its selection is made."
        )
    blob = torch.load(target, map_location="cpu", weights_only=False)
    version = int(blob.get("version", 0))
    if version != PILOT_CHECKPOINT_VERSION:
        raise PilotConfigError(
            f"{target} carries pilot-checkpoint version {version}, and this reader understands "
            f"{PILOT_CHECKPOINT_VERSION}. The layout changed; re-run the fit rather than reading "
            f"keys that may no longer mean what they did."
        )
    return PilotCheckpoint(
        mean_head_state=dict(blob["mean_head_state"]),
        classifier_state=dict(blob["classifier_state"]),
        threshold=float(blob["threshold"]),
        selected_epoch=int(blob["selected_epoch"]),
        source=dict(blob.get("source") or {}),
        fingerprint=dict(blob.get("fingerprint") or {}),
        record=dict(blob.get("record") or {}),
    )


def apply_adapted(checkpoint: PilotCheckpoint, loaded: Any) -> Dict[str, Any]:
    """Write a saved adaptation's mean-head weights into a freshly loaded model.

    Checked before it is applied, on the two things that would make it silently wrong: the source
    checkpoint's digest, so an adaptation of one pretrained model cannot be dropped into another,
    and the parameter names, so a geometry change surfaces here rather than as a shape error deep
    in a forward.

    Args:
        checkpoint: The saved adaptation.
        loaded: A freshly loaded bundle of the **same** pretrained checkpoint.

    Returns:
        What was applied: the names, the count and the source it came from.

    Raises:
        PilotConfigError: On a digest or parameter-name mismatch.
    """
    stored = str(checkpoint.source.get("checkpoint_digest") or "")
    if stored and stored != str(loaded.digest):
        raise PilotConfigError(
            f"this adaptation was fitted on checkpoint digest {stored} and is being applied to "
            f"{loaded.digest}. The adapted weights are a delta on one pretrained model's posterior "
            f"head; applied to another they would be a different model's numbers under this run's "
            f"name."
        )
    parameters = dict(pilot_model.mean_head_parameters(loaded.model))
    missing = sorted(set(checkpoint.mean_head_state) - set(parameters))
    extra = sorted(set(parameters) - set(checkpoint.mean_head_state))
    if missing or extra:
        raise PilotConfigError(
            f"the saved adaptation names mean-head parameters this model does not have ({missing}) "
            f"or omits ones it does ({extra}). The two geometries differ, and applying the "
            f"intersection would leave part of the head pretrained and part adapted."
        )
    _load_mean_head_state(loaded.model, checkpoint.mean_head_state)
    record = {
        "names": sorted(checkpoint.mean_head_state),
        "n_parameters": int(
            sum(tensor.numel() for tensor in checkpoint.mean_head_state.values())
        ),
        "selected_epoch": checkpoint.selected_epoch,
        "source_digest": stored,
    }
    logger.info(
        f"applied the adaptation from epoch {checkpoint.selected_epoch}: "
        f"{record['n_parameters']} parameter(s) in {len(record['names'])} tensor(s)"
    )
    return record


def _refuse_source_path(target: Path, loaded: Any) -> None:
    """Refuse a write that would land on the pretrained checkpoint or its configuration.

    Args:
        target: The path about to be written.
        loaded: The loaded bundle, which knows where it came from.

    Raises:
        PilotConfigError: If the two resolve to the same file.
    """
    for protected in (Path(loaded.checkpoint_path), Path(loaded.config_path)):
        try:
            same = target.resolve() == protected.resolve()
        except OSError:  # a path that cannot be resolved is not the source
            same = False
        if same:
            raise PilotConfigError(
                f"refusing to write {target}: that is the pretrained checkpoint's own file. The "
                f"source is opened read-only for the whole of this pilot, and every artifact this "
                f"run produces belongs inside its own run directory."
            )


def export_base_checkpoint(
    loaded: Any, directory: Any, *, selected_epoch: Optional[int] = None
) -> Path:
    """Export the adapted net as a checkpoint the ordinary evaluator can load.

    The model's **own** state dict, with the adapted mean heads already in it and no classifier key
    anywhere, stamped with the source's ``model_class``, ``model_kwargs`` and ``hyper_parameters``
    so ``check_model_class`` and the strict load both find what they expect. The resolved
    configuration is copied in beside it, because ``resolved_config_for`` looks next to a
    checkpoint and a file moved away from its own run directory has lost the record of what it was
    trained on.

    Args:
        loaded: The bundle, holding the model at the weights to export.
        directory: The run directory; the export lands in a subdirectory of it.
        selected_epoch: Which candidate the fit selected, stamped into the payload. ``0`` means
            the frozen model was retained, in which case this file holds the PRETRAINED weights
            under an "adapted" name -- a reader who has only the file needs to be told that.
            ``None`` records the selection as unknown rather than asserting one.

    Returns:
        The exported checkpoint's path.

    Raises:
        PilotConfigError: If the export would overwrite the pretrained checkpoint.
    """
    import shutil

    path = Path(directory) / ADAPTED_EXPORT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    target = path / ADAPTED_CHECKPOINT_FILENAME
    _refuse_source_path(target, loaded)

    blob = dict(loaded.blob)
    payload = {
        "state_dict": {
            key: tensor.detach().cpu() for key, tensor in loaded.model.state_dict().items()
        },
        "model_class": blob.get("model_class"),
        "model_kwargs": blob.get("model_kwargs"),
        "hyper_parameters": blob.get("hyper_parameters"),
        "epoch": blob.get("epoch"),
        "pilot": {
            "source_checkpoint": str(loaded.checkpoint_path),
            "source_digest": loaded.digest,
            # Named for what it holds: the module path whose weights were replaced. The fit
            # record's own "adapted" key is a boolean, and two keys of one name holding two types
            # is how a reader ends up trusting the wrong one.
            "adapted_module": pilot_model.MEAN_HEAD_PATH,
            "selected_epoch": None if selected_epoch is None else int(selected_epoch),
            "adapted": None if selected_epoch is None else bool(int(selected_epoch) > 0),
            "note": (
                "the pretrained net with this pilot's posterior mean-output weights in place; "
                "no classifier key is present, and the classifier lives in the pilot checkpoint. "
                "selected_epoch 0 means the frozen model was retained, so these weights are the "
                "pretrained ones and nothing here was adapted"
            ),
        },
    }
    torch.save(payload, target)
    shutil.copyfile(Path(loaded.config_path), path / Path(loaded.config_path).name)
    logger.info(f"exported a base-model-compatible checkpoint to {target}")
    return target



# =============================================================================
# The shuffled-label control
# =============================================================================
#: The name the control's artifacts are filed under, beside the baseline's and the adaptation's.
CONTROL_NAME = "control"

#: The name the frozen prior probe's artifacts are filed under.
PRIOR_PROBE_NAME = "prior_probe"


def permute_outcomes(
    recordings: pd.DataFrame,
    *,
    seed: int,
    splits: Sequence[str] = ("train", "val"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Permute the outcomes **within each split, independently**, at the recording level.

    Three properties the protocol asks for, and each is a line here rather than a convention:

    * **At GUID level.** The outcome is a property of a recording, so permuting the recording table
      is what makes every segment keep its own recording's assigned label. A shuffle applied per
      segment would test something that does not exist in the data.
    * **Within the fitted population.** Only the recordings that are eligible on that split are
      shuffled among themselves; excluded and ineligible recordings keep their outcome. Shuffling
      across that boundary would move adverse labels into and out of the eligible subset, so the
      control would be fitted and selected at a different class prevalence from the run it
      controls -- and where eligibility correlates with outcome, which is exactly the coverage
      confound the controls exist to describe, the draw can leave a split with one class and abort
      the stage after the real fit has already spent its whole epoch budget.
    * **Independently per split.** Validation's labels are permuted on their own draw, so the
      control's selection is driven by a null that the training permutation does not determine.
    * **The test split is untouched.** True held-out labels are used once, at the final evaluation,
      against a model fitted and selected entirely under the null.

    Args:
        recordings: The recording table.
        seed: The run's seed. Each split's draw is derived from it and from the split's name, so
            two splits differ and the whole control is reproducible from the record.
        splits: Which splits to permute. The test split is deliberately not among the defaults.

    Returns:
        ``(frame, record)``: a copy with the outcomes permuted, and what was drawn -- including how
        many labels actually moved, because a permutation that happened to be the identity is a
        control that controls nothing.

    Raises:
        PilotConfigError: If the test split is asked for. A permuted held-out label would make the
            control unreadable against the thing it controls.
    """
    if "test" in splits:
        raise PilotConfigError(
            "the test split's labels are never permuted: the control is fitted and selected under "
            "the null and then evaluated once against the true held-out outcomes, which is what "
            "makes its held-out result comparable with the real run's."
        )
    out = recordings.copy()
    outcomes = out[data.OUTCOME_COLUMN].to_numpy(dtype=object).copy()
    guids = out[data.GUID_COLUMN].astype(str).to_numpy(dtype=object)
    per_split: Dict[str, Any] = {}
    for split in splits:
        # The eligible recordings of this split, which is the population the real fit sees. If
        # eligibility has not been attached yet the mapping is empty and every labelled row of the
        # split is shuffled, which is the old behaviour and the right one when there is no
        # eligibility to respect.
        fitted = set(data.eligible_outcomes(out, split=split))
        in_split = (out[data.SPLIT_COLUMN].astype(str) == str(split)).to_numpy(dtype=bool)
        labelled = np.array(
            [value is not None and not pd.isna(value) for value in outcomes], dtype=bool
        )
        if fitted:
            in_split = in_split & np.array([guid in fitted for guid in guids], dtype=bool)
        rows = np.flatnonzero(in_split & labelled)
        if rows.size == 0:
            per_split[split] = {"n_recordings": 0, "n_changed": 0}
            continue
        generator = np.random.default_rng(
            [int(seed), int.from_bytes(str(split).encode("utf-8"), "big") % (2**31)]
        )
        original = outcomes[rows].astype(np.int64)
        permuted = original[generator.permutation(rows.size)]
        outcomes[rows] = permuted
        per_split[split] = {
            # The population that was shuffled, which is the population the fit sees -- not the
            # split's total. `n_adverse` is therefore invariant under the permutation, which is
            # the property that makes the control the same experiment under the null.
            "n_recordings": int(rows.size),
            "n_changed": int((original != permuted).sum()),
            "n_adverse": int((original == 1).sum()),
            "population": "eligible" if fitted else "all labelled (eligibility not attached)",
        }
        if per_split[split]["n_changed"] == 0:
            logger.warning(
                f"the {split!r} permutation left every label where it was, so the control on that "
                f"split is not a control; this is possible only on a very small split"
            )
    out[data.OUTCOME_COLUMN] = outcomes
    record = {
        "seed": int(seed),
        "splits": list(splits),
        "level": "recording (guid)",
        "per_split": per_split,
        "test_split_permuted": False,
        "note": (
            "one fixed permutation per split, drawn independently, confined to the eligible "
            "recordings the real fit sees so the class marginal is unchanged; a single control "
            "fit is a leakage and overfitting check and is never a permutation p-value"
        ),
    }
    logger.info(f"shuffled-label control: permuted outcomes {per_split}")
    return out, record


def control_recordings(
    recordings: pd.DataFrame, *, seed: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The recording table the whole control runs against.

    Permuting the **table** rather than threading a label mapping through every stage is what makes
    the control the same code path as the real run: ``build_bags``, ``build_plans`` and the
    validation scoring all read outcomes from here, so none of them needs a control-aware branch
    that could drift from the branch it controls.

    Args:
        recordings: The recording table.
        seed: The run's seed.

    Returns:
        ``(frame, record)`` from :func:`permute_outcomes`.
    """
    return permute_outcomes(recordings, seed=seed)


def fit_control_baseline(
    train_extraction: Any,
    val_extraction: Any,
    permuted: pd.DataFrame,
    *,
    permutation: Mapping[str, Any],
    scaler: Any,
    settings: Mapping[str, Any],
) -> BaselineFit:
    """Fit the control's **own** permuted-label baseline classifier.

    Its own, and this is the point of the function existing at all: initialising the control from
    the true-label baseline would hand it a head already fitted to the association it is supposed
    to be testing the absence of, and any held-out result it then produced would be that head's
    rather than the null's.

    Args:
        train_extraction: The pretrained training extraction.
        val_extraction: The pretrained validation extraction.
        permuted: The recording table with permuted outcomes.
        permutation: The record :func:`control_recordings` returned with it. Required rather than
            inferred: this function cannot look at a table and tell whether it was shuffled, and a
            fit that *claimed* permuted labels without having been given the draw that produced
            them would be an unverifiable line in a report.
        scaler: The frozen training scaler -- the same one, because the standardizing constants are
            a property of the representation and not of the labels.
        settings: The resolved pilot settings.

    Returns:
        The control's baseline fit.
    """
    windows, baseline = settings["windows"], settings["baseline"]
    bags = {
        split: build_bags(
            extraction,
            permuted,
            split=split,
            supervised_hours=float(windows["supervised_hours"]),
            halflife_hours=float(settings["bag"]["halflife_hours"]),
        )
        for split, extraction in (("train", train_extraction), ("val", val_extraction))
    }
    fit = fit_baseline(
        bags["train"],
        bags["val"],
        scaler=scaler,
        lr=float(baseline["lr"]),
        max_steps=int(baseline["max_steps"]),
        patience=int(baseline["patience"]),
        weight_decay=float(settings["optim"]["weight_decay"]),
        seed=int(settings["seed"]),
    )
    fit.record["name"] = CONTROL_NAME
    fit.record["labels_permuted"] = True
    fit.record["permutation"] = dict(permutation)
    fit.record["initialised_from"] = "its own permuted-label fit, never a true-label one"
    return fit


# =============================================================================
# The conditional frozen prior probe
# =============================================================================
def fit_prior_probe(
    train_extraction: Any,
    val_extraction: Any,
    recordings: pd.DataFrame,
    *,
    settings: Mapping[str, Any],
) -> Tuple[Optional[BaselineFit], Dict[str, Any]]:
    r"""A frozen linear probe on pooled ``mu_prior``, under the same split and protocol.

    Required before any claim that the combined branch helps. ``mu_prior`` is the target-only
    latent and is **unchanged** by this adaptation, so the probe is two things at once: the
    FHR-only comparison the claim rests on, and an invariant -- its bags are identical before and
    after by construction.

    Its scaler is fitted on ``mu_prior``'s own training anchors rather than reused from
    ``mu_post``: the two latents are different quantities and standardizing one by the other's
    spread would make the comparison a comparison of scalings.

    Switching it off through ``prior_probe: false`` is supported and has a consequence the report
    states rather than omits: with nothing to compare against, this run makes no claim that the
    combined branch helps.

    Args:
        train_extraction: The pretrained training extraction.
        val_extraction: The pretrained validation extraction.
        recordings: The recording table, with true outcomes.
        settings: The resolved pilot settings; ``prior_probe`` switches this on.

    Returns:
        ``(fit, record)``. The fit is ``None`` when the probe is disabled, and the record says so
        and withdraws the claim it would have supported.
    """
    if not bool(settings.get("prior_probe", True)):
        return None, {
            "enabled": False,
            "combined_branch_claim_supported": False,
            "note": (
                "the frozen mu_prior probe was switched off, so this run makes no claim that the "
                "combined branch helps: there is nothing to compare mu_post's discrimination "
                "against"
            ),
        }

    windows, baseline = settings["windows"], settings["baseline"]
    bags = {
        split: build_bags(
            extraction,
            recordings,
            split=split,
            supervised_hours=float(windows["supervised_hours"]),
            halflife_hours=float(settings["bag"]["halflife_hours"]),
            key="mu_prior",
        )
        for split, extraction in (("train", train_extraction), ("val", val_extraction))
    }
    retained = train_extraction.retained
    scaler = extract.fit_scaler(
        retained, train_extraction.arrays["mu_prior"], key="mu_prior"
    )
    fit = fit_baseline(
        bags["train"],
        bags["val"],
        scaler=scaler,
        lr=float(baseline["lr"]),
        max_steps=int(baseline["max_steps"]),
        patience=int(baseline["patience"]),
        weight_decay=float(settings["optim"]["weight_decay"]),
        seed=int(settings["seed"]),
    )
    fit.record["name"] = PRIOR_PROBE_NAME
    fit.record["key"] = "mu_prior"
    record = {
        "enabled": True,
        "key": "mu_prior",
        "combined_branch_claim_supported": True,
        "selected_val_auroc": fit.record["selected_val_auroc"],
        "scaler": dict(scaler.record),
        "note": (
            "mu_prior is unchanged by this adaptation, so this probe is both the FHR-only "
            "comparison and an invariant; a better mu_post result than this one still does not "
            "establish UP-specific information on its own"
        ),
    }
    logger.info(
        f"frozen mu_prior probe: validation AUROC {fit.record['selected_val_auroc']:.4f}"
    )
    return fit, record


__all__ = [
    "BASELINE_NAME",
    "CLASSIFIER_FILENAME",
    "FITTING_SPLITS",
    "FIT_FILENAME",
    "ADAPTED_CHECKPOINT_FILENAME",
    "ADAPTED_EXPORT_DIRNAME",
    "AdaptationFit",
    "Bags",
    "BaselineFit",
    "CONTROL_NAME",
    "PILOT_CHECKPOINT_FILENAME",
    "PILOT_CHECKPOINT_VERSION",
    "PRIOR_PROBE_NAME",
    "PilotCheckpoint",
    "RecordingPlan",
    "RecordingSource",
    "SegmentSupport",
    "balanced_bce",
    "build_bags",
    "build_plans",
    "apply_adapted",
    "epoch_batches",
    "export_base_checkpoint",
    "fit_adaptation",
    "fit_baseline",
    "fit_control_baseline",
    "fit_prior_probe",
    "is_better",
    "load_adapted",
    "load_fit",
    "permute_outcomes",
    "recording_terms",
    "save_adapted",
    "save_fit",
]
