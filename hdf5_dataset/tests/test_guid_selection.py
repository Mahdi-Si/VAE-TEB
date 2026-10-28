r"""Step 2 of ``create_new_pipeline.py``: the classification pool, the test set and pretraining.

The checks run on a synthetic screening table with the shape of the production one: a healthy
population that is mostly no-BG, few unhealthy GUIDs, and some ineligible and unreadable rows.

What is pinned:

* each healthy subgroup's share of one fold's healthy test equals its share $p_g$ of the
  eligible healthy population, up to rounding;
* the pool takes the requested fraction of each healthy subgroup;
* the pool, the test set and the pretraining set are disjoint, and pretraining holds every other
  eligible healthy GUID;
* ``no_bg_cls_fraction=None`` keeps the original deficit rule;
* a no-BG fraction draws at random, so the pool keeps the population TLO rate;
* ``healthy_eligibility='eligible_2h'`` admits 2-3 h healthy GUIDs to classification, while
  pretraining keeps the 3 h rule;
* ``selection_only`` writes the fold pickle and no HDF5 file.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

#: Eligible GUIDs per subgroup; the healthy ones are mostly no-BG, as in production.
ELIGIBLE = {
    "acidosis_cs": 120, "acidosis_no_cs": 240, "hie_cs": 30, "hie_no_cs": 30,
    "healthy_bg_cs": 800, "healthy_bg_no_cs": 1400,
    "healthy_no_bg_cs": 900, "healthy_no_bg_no_cs": 8000,
}
#: Rows that no rule may select: ineligible, or unreadable, per subgroup.
INELIGIBLE, UNREADABLE = 20, 5
HEALTHY = ("healthy_bg_cs", "healthy_bg_no_cs", "healthy_no_bg_cs", "healthy_no_bg_no_cs")
K = 10


#: Share of GUIDs with a known labour onset in the synthetic table: far from the 75 % quota of
#: the deficit rule, so a quota shows.
TLO_SHARE = 0.6
#: Healthy rows with 2-3 h of valid signal, per subgroup, in the ``screening_2h`` table.
BAND_2H = 200


def _screening(band_2h: int) -> pd.DataFrame:
    """Eligible (3 h), 2-3 h (healthy only), ineligible and unreadable rows per subgroup."""
    rng = np.random.default_rng(0)
    rows = []
    for sg, n in ELIGIBLE.items():
        band = band_2h if sg in HEALTHY else 0
        for i in range(n + band + INELIGIBLE + UNREADABLE):
            has_tlo = bool(rng.random() < TLO_SHARE)
            rows.append({
                "record_path": f"/data/{sg}/{sg}_{i:05d}.mat", "subgroup": sg,
                "error": i >= n + band + INELIGIBLE, "eligible_2h": i < n + band,
                "eligible_3h": i < n,
                "has_tlo": has_tlo, "tlo_hours": float(rng.uniform(2, 20)) if has_tlo else np.nan,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def screening() -> pd.DataFrame:
    """Every eligible GUID has 3 h, so both healthy rules select the same GUIDs."""
    return _screening(0)


@pytest.fixture(scope="module")
def screening_2h() -> pd.DataFrame:
    """Each healthy subgroup also has :data:`BAND_2H` GUIDs with 2-3 h of valid signal."""
    return _screening(BAND_2H)


def _population_share() -> Dict[str, float]:
    total = sum(ELIGIBLE[sg] for sg in HEALTHY)
    return {sg: ELIGIBLE[sg] / total for sg in HEALTHY}


def _healthy_test_per_fold(result: Dict[str, Any], test_mode: str) -> Dict[str, float]:
    """Healthy GUIDs per subgroup in one fold's test partition."""
    if test_mode == "holdout":
        return {sg: float(len(result["test"].get(sg, []))) for sg in HEALTHY}
    return {sg: len(result["trainval"].get(sg, [])) / K + len(result["test"].get(sg, []))
            for sg in HEALTHY}


def _assert_population_mix(result: Dict[str, Any], test_mode: str) -> None:
    per_fold = _healthy_test_per_fold(result, test_mode)
    total = sum(per_fold.values())
    for sg, share in _population_share().items():
        # Each subgroup is rounded once, so it may miss its target by half a GUID, plus the
        # fractional pool share of the augmented mode.
        assert abs(per_fold[sg] - total * share) <= 1.0, (sg, per_fold, total * share)


def _assert_disjoint_and_complete(result: Dict[str, Any], screening: pd.DataFrame) -> None:
    pool = {p for paths in result["trainval"].values() for p in paths}
    test = {p for paths in result["test"].values() for p in paths}
    pretraining = {p for paths in result["pretraining"].values() for p in paths}
    assert not pool & test and not pool & pretraining and not test & pretraining
    eligible = screening[~screening["error"] & screening["eligible_3h"]
                         & screening["subgroup"].isin(HEALTHY)]
    assert pretraining == set(eligible["record_path"]) - pool - test
    for sg in HEALTHY:
        assert all(f"/{sg}/" in p for p in result["pretraining"][sg])


@pytest.mark.parametrize("bg_fraction, no_bg_fraction", [(0.3, 0.05), (0.2, 0.2), (0.1, 0.3)])
def test_augmented_test_keeps_the_population_mix(
    pipeline: Any, screening: pd.DataFrame, bg_fraction: float, no_bg_fraction: float
) -> None:
    """Lower, equal and higher no-BG fractions: the extras go wherever the mix needs them."""
    result = pipeline.select_classification_guids(
        screening, test_mode="augmented", bg_cls_fraction=bg_fraction,
        no_bg_cls_fraction=no_bg_fraction, verbose=False)

    for sg in HEALTHY:
        fraction = bg_fraction if "_no_bg_" not in sg else no_bg_fraction
        assert len(result["trainval"][sg]) == round(ELIGIBLE[sg] * fraction), sg
    for sg in ("acidosis_cs", "acidosis_no_cs", "hie_cs", "hie_no_cs"):
        assert len(result["trainval"][sg]) == ELIGIBLE[sg]
    _assert_population_mix(result, "augmented")
    _assert_disjoint_and_complete(result, screening)

    extra = result["stats"]["n_extra_test"]
    if no_bg_fraction < bg_fraction:
        assert extra["healthy_no_bg_no_cs"] > 0 and extra["healthy_bg_no_cs"] <= 1
    elif no_bg_fraction > bg_fraction:
        assert extra["healthy_bg_no_cs"] > 0 and extra["healthy_no_bg_no_cs"] <= 1


def test_holdout_test_keeps_the_population_mix(pipeline: Any, screening: pd.DataFrame) -> None:
    """The fixed test set: the BG holdout plus the extra GUIDs."""
    result = pipeline.select_classification_guids(
        screening, test_mode="holdout", bg_cls_fraction=0.3, no_bg_cls_fraction=0.1,
        verbose=False)

    _assert_population_mix(result, "holdout")
    _assert_disjoint_and_complete(result, screening)


def test_none_keeps_the_deficit_rule(pipeline: Any, screening: pd.DataFrame) -> None:
    """``no_bg_cls_fraction=None``: no-BG fills train/val up to healthy = unhealthy."""
    result = pipeline.select_classification_guids(
        screening, test_mode="augmented", bg_cls_fraction=0.1, no_bg_cls_fraction=None,
        verbose=False)

    n_unhealthy = sum(len(result["trainval"][sg])
                      for sg in ("acidosis_cs", "acidosis_no_cs", "hie_cs", "hie_no_cs"))
    n_healthy = sum(len(result["trainval"][sg]) for sg in HEALTHY)
    assert n_healthy == n_unhealthy
    _assert_population_mix(result, "augmented")
    _assert_disjoint_and_complete(result, screening)


def test_a_no_bg_fraction_keeps_the_population_tlo_rate(
    pipeline: Any, screening: pd.DataFrame
) -> None:
    """Random draws: the no-BG pool sits near 60 % TLO, not at the deficit rule's 75 %."""
    result = pipeline.select_classification_guids(
        screening, bg_cls_fraction=0.3, no_bg_cls_fraction=0.3, verbose=False)
    tlo = screening.set_index("record_path")["has_tlo"]
    for sg in ("healthy_no_bg_cs", "healthy_no_bg_no_cs"):
        population = tlo[screening.loc[(screening["subgroup"] == sg)
                                       & screening["eligible_3h"], "record_path"]].mean()
        pool = tlo[result["trainval"][sg]].mean()
        assert abs(pool - population) < 0.06, (sg, pool, population)
    assert result["stats"]["no_bg_sampling"] == "random"


def test_two_hour_rule_admits_the_band_and_pretraining_keeps_three_hours(
    pipeline: Any, screening_2h: pd.DataFrame
) -> None:
    result = pipeline.select_classification_guids(
        screening_2h, bg_cls_fraction=0.2, no_bg_cls_fraction=0.2, verbose=False,
        healthy_eligibility="eligible_2h")

    eligible_2h = screening_2h[~screening_2h["error"] & screening_2h["eligible_2h"]]
    band = set(eligible_2h.loc[~eligible_2h["eligible_3h"], "record_path"])
    pool = {p for paths in result["trainval"].values() for p in paths}
    test = {p for paths in result["test"].values() for p in paths}
    pretraining = {p for paths in result["pretraining"].values() for p in paths}
    for sg in HEALTHY:
        assert len(result["trainval"][sg]) == round((ELIGIBLE[sg] + BAND_2H) * 0.2), sg
    assert pool & band, "a 2-3 h healthy GUID must be able to enter the pool"
    assert not pretraining & band, "pretraining keeps the 3 h rule"
    three_h = screening_2h[~screening_2h["error"] & screening_2h["eligible_3h"]
                           & screening_2h["subgroup"].isin(HEALTHY)]
    assert pretraining == set(three_h["record_path"]) - pool - test

    per_fold = _healthy_test_per_fold(result, "augmented")
    total = sum(per_fold.values())
    n_eligible = sum(ELIGIBLE[sg] + BAND_2H for sg in HEALTHY)
    for sg in HEALTHY:
        share = (ELIGIBLE[sg] + BAND_2H) / n_eligible
        assert abs(per_fold[sg] - total * share) <= 1.0, sg


def test_an_unknown_healthy_rule_is_refused(pipeline: Any, screening: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="healthy_eligibility"):
        pipeline.select_classification_guids(screening, verbose=False,
                                             healthy_eligibility="eligible_1h")


def test_selection_only_writes_the_pickle_and_no_hdf5(
    pipeline: Any, screening: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steps 1-3 only: the fold pickle and the summary, no masks and no HDF5 file."""
    def no_masks(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a selection-only run must not compute the masks")

    monkeypatch.setattr(pipeline, "setup_verbosity", lambda verbose: None)
    monkeypatch.setattr(pipeline, "compute_scattering_masks", no_masks)
    csv_path = tmp_path / "guid_screening_results.csv"
    screening.to_csv(csv_path, index=False)
    out = tmp_path / "out"

    pipeline.create_new_pipeline(
        records_base_path=str(tmp_path), output_base_path=str(out),
        tlo_csv_path=str(tmp_path / "unused.csv"), screening_csv_path=str(csv_path),
        bg_cls_fraction=0.2, no_bg_cls_fraction=0.2, selection_only=True)

    with open(out / "classification_dataset_records.pickle", "rb") as f:
        cv_result = pickle.load(f)
    assert len(cv_result["folds"]) == K
    assert (out / "classification_guid_selection_summary.json").exists()
    assert not list(out.rglob("*.hdf5"))

    # A second run into the same directory would replace the pickle: refused.
    with pytest.raises(FileExistsError, match="NEW output_base_path"):
        pipeline.create_new_pipeline(
            records_base_path=str(tmp_path), output_base_path=str(out),
            tlo_csv_path=str(tmp_path / "unused.csv"), screening_csv_path=str(csv_path),
            selection_only=True)


def test_selection_only_refuses_a_pickle(pipeline: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection_only"):
        pipeline.create_new_pipeline(
            records_base_path=str(tmp_path), output_base_path=str(tmp_path / "out"),
            tlo_csv_path=str(tmp_path / "unused.csv"),
            classification_pickle_path=str(tmp_path / "old.pickle"), selection_only=True)


@pytest.mark.parametrize("kwargs", [{"bg_cls_fraction": 0.0}, {"bg_cls_fraction": 1.5},
                                    {"no_bg_cls_fraction": -0.1}])
def test_a_fraction_out_of_range_is_refused(
    pipeline: Any, screening: pd.DataFrame, kwargs: Dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="cls_fraction"):
        pipeline.select_classification_guids(screening, verbose=False, **kwargs)
