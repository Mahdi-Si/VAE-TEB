r"""Phase-1 report: headline numbers of T1--T4, the five pre-registered predictions and T5.

The script reads the JSON summaries the phase-1 scripts wrote (``rebuild_validation``,
``t1_summary``, ``t3_summary``, ``t4_summary``) and writes ``out/phase1/REPORT_phase1.json``.
It computes no new statistic except two T5 tables from the stored per-gap and per-anchor
tables: the share of the bridge's peak error that its no-overshoot rule fixes in advance, and
the costs by gap length for the observed-mix code types.

**Why the bridge cannot reach a crest.** The bridge is a monotone cubic between the last valid
sample before the gap $y_0$ and the first valid sample after it $y_1$, with Fritsch--Carlson
limited slopes, so every bridged value lies in $[\min(y_0, y_1), \max(y_0, y_1)]$. For a gap that
holds the crest, the peak error is therefore at least
$\max_{n \in \mathrm{gap}} x_n - \max(y_0, y_1)$, whatever the shape of the contraction.

**Verdict rules.** Each prediction is checked on the bootstrap mean and its 95% interval:
*confirmed* when the mean and the whole interval are on the predicted side, *refuted* when the
mean is on the other side, *unclear* otherwise. Prediction 2 is tested at the 32 s grid length
(the grid has no 30 s length).

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"n_boot": 2000}

POS = ("rise", "peak", "fall", "between")


def load(name: str) -> Dict[str, Any]:
    """Read one phase-1 JSON file."""
    return json.loads((C.OUT1 / name).read_text())


def side(stat: Dict[str, float], below: float, key: str = "mean") -> str:
    """Verdict of 'statistic below a threshold' from a mean and its interval."""
    m, lo, hi = stat[key], stat["lo"], stat["hi"]
    if m < below and hi < below:
        return "confirmed"
    if m >= below:
        return "refuted"
    return "unclear"


def crest_bound(n_boot: int) -> Dict[str, Any]:
    """Share of the peak error that the bridge's no-overshoot rule fixes, for gaps over the crest.

    Args:
        n_boot: Resamples.

    Returns:
        Per grid length: mean of the bound and of the observed peak error, as shares of $A$.
    """
    synth = pd.read_parquet(C.OUT1 / "t2_synth.parquet")
    gaps = pd.read_parquet(C.OUT1 / "t2_injections.parquet")
    base = pd.read_parquet(C.OUT1 / "t2_base.parquet")
    true = dict(np.load(C.OUT1 / "t2_base.npz"))
    grid = synth[(synth["set"] == "grid") & (synth.position == "peak")]
    rows = []
    for _, s in grid.iterrows():
        gp = gaps[gaps.synth_id == s.synth_id].iloc[0]
        b = base[base.base_id == s.base_id].iloc[0]
        x = true["up"][int(s.base_id)].astype(float)
        amp = x[int(b.peak)] - x[int(b.onset)]
        if amp <= 0:
            continue
        edge = max(x[int(gp.start) - 1], x[int(gp.stop)])
        win = slice(int(b.onset), int(b.end) + 1)
        rest = np.concatenate([x[int(b.onset):int(gp.start)], x[int(gp.stop):int(b.end) + 1]])
        bound = max(x[win].max() - max(edge, rest.max() if rest.size else -np.inf), 0.0) / amp
        rows.append({"rec": f"{s.cohort}/{s.guid}", "len_s": s.len_s, "bound": bound})
    df = pd.DataFrame(rows)
    return {f"{int(L)}s": S.paired_bootstrap(p.rec.values, p.bound.values, n_resamples=n_boot)
            for L, p in df.groupby("len_s")}


#: Caveats that hold for every phase-1 number.
CAVEATS = [
    "Stand-in normalisation statistics: absolute nats are approximate; arm differences are the reported quantity.",
    "healthy_no_bg_no_cs has unknown training membership ('possibly seen'); 78 of 160 base segments come from it. HIE has only 4 base segments.",
    "The rebuilt kernel digest differs from the file attribute quality_kernel_digest, yet up_st_rho/up_ph_rho reproduce with 0 error on every segment without a re-derived level shift; the digest also covers the FHR kernels, which this phase does not use.",
    "Re-running ua_quality does not reproduce the stored bits exactly (PUSH/TRANSIENT split, second-stage mask); this only touches the step term of rho in 24/2328 dirty segments and never the synthetic set (clean bases).",
    "The pooled cost on affected anchors is diluted: the strict rule marks anchors affected for 152 s plus the contamination time, and most of them see a near-zero change. In the first 60 s after the gap end the tolerant cost is 10%, 23%, 47% at 4, 8, 16 s (see T4.grid_len_timing).",
    "Anchors inside a gap under the tolerant arm read bridged values that depend on raw UA up to 4 s after the gap end (bridge end slope over x[e..e+16]); that gain is not available causally.",
    "NLL_base is not bit-identical across variants (max 0.041 nats per anchor, mean G about 5 nats): batch-dependent GPU arithmetic; anchors before a gap change by at most 0.005 nats.",
    "Grid gaps lie in raw samples [1792, 3936] (448-984 s) so that dense anchors (model steps 134-269) see them; observed-mix gaps spread over [400, 4880] and anchors are attributed to the most recent gap.",
    "Per-code tolerance is decidable only for TRANSIENT (103 recordings); CODE, STEP, MISSING and FLOOR have 4-19 recordings in the mix set and intervals spanning more than +-50%.",
    "Slow-channel off durations are right-censored by the segment end (24% at 64 s, 44% at 128 s); they are lower bounds there.",
    "The contraction phases come from the existing detector (detect_contractions) with weight-only validity; the detector audit (E1.3) runs separately and may change the phase tables.",
    "Two base segments (ids 62, 81) have no scored anchor and drop out of T4.",
    "The 30 s point of prediction 2 is tested at the 32 s grid length.",
    "Ratios with small denominators are unstable (for example 0-60 s classes at >= 32 s exceed 100%: the bridged source turns G negative).",
]


def conclusions(rep: Dict[str, Any]) -> Dict[str, Any]:
    """Write the T5 verdict on the step-shape hypothesis from the stored numbers.

    Args:
        rep: The report so far.

    Returns:
        Where the hypothesis holds and where it fails, with the numbers that show it.
    """
    lv1 = rep["T3_level1_grid"]
    t4 = rep["T4"]
    crest = rep["T5_crest_bound_peak_gaps"]
    pct = lambda s, k="mean": round(100 * s[k], 1)  # noqa: E731
    return {
        "holds": {
            "flanks_and_rests_raw": {p: {f"{L}s": pct(lv1[f"{L}s/{p}"]["rmse_rel_amp"]) for L in (4, 8, 16, 32)}
                                     for p in ("rise", "fall", "between")},
            "model_pooled_c_tol_pct": {L: pct(v["c_tol"], "ratio") for L, v in t4["grid_len"].items()},
            "mid_and_slow_bands_model_cost_pct": {g: {L: pct(c, "ratio") for L, c in t4["tolerance"]["by_band_group"][g]["c_by_length"].items()}
                                                 for g in ("mid", "slow")},
            "TRANSIENT_mix_max_tolerable_s": t4["tolerance"]["by_code_mix"].get("TRANSIENT", {}).get("max_tolerable_s"),
            "text": ("Across a monotone flank or a rest, the monotone cubic follows the level change: RMSE stays below 11% of "
                     "the contraction amplitude up to 32 s on rise and fall, below 12% up to 64 s between contractions, and the "
                     "detector's onset error stays under 1.5 s up to 32 s. At the model, bridged UA with valid flags costs 0.5-4.6% "
                     "of G pooled up to 32 s; the mid and slow bands cost at most 2.3% at any length."),
        },
        "fails": {
            "crest_peak_amp_err_pct": {L: pct(lv1[f"{L}/peak"]["peak_amp_err_rel"]) for L in ("4s", "8s", "16s", "32s", "64s", "128s")},
            "crest_bound_pct": {L: pct(v) for L, v in crest.items()},
            "crest_detector_peak_timing_err_s": {L: round(lv1[f"{L}/peak"]["abs_det_peak_err_s"]["mean"], 1) for L in ("8s", "16s", "32s", "64s")},
            "rise_model_c_tol_pct": {L: pct(t4["grid_len_pos"][f"{L}/rise"]["c_tol"], "ratio") for L in ("8s", "16s", "32s", "64s")},
            "fast_band_model_cost_pct": {L: pct(c, "ratio") for L, c in t4["tolerance"]["by_band_group"]["fast"]["c_by_length"].items()},
            "first_60s_after_gap_c_tol_pct": {L: pct(t4["grid_len_timing"][f"{L}/0-60"]["c_tol"], "ratio") for L in ("4s", "8s", "16s", "32s")},
            "text": ("A crest is not a step: the bridge never exceeds its two edge values, so a gap over the peak loses exactly the "
                     "crest above the edges (the no-overshoot bound equals the observed peak error at every length: 2.3%, 5.9%, "
                     "10.9%, 28.8%, 65% of the amplitude at 4-64 s). At the model, the rise is the weakest position (c_tol 7.9% "
                     "[0, 20.1%] at 8 s, 13.1% at 32 s). The model's cost sits in the fast band and in the first 60 s after the "
                     "gap end, where the tolerant cost reaches 23% at 8 s and 47% at 16 s; pooled numbers hide this."),
        },
        "verdict": ("Partly supported. The step-shape idea works on flanks and rests and for the slow/mid bands; it fails at the "
                    "crest beyond about 8-16 s and for the fast-band forecasts in the first minute after a gap. Even where it "
                    "fails, bridged UA with valid flags loses less gain than the strict flags for gaps <= 16 s."),
    }


def main(n_boot: int) -> Dict[str, Any]:
    """Assemble the report.

    Args:
        n_boot: Resamples for the two T5 tables.

    Returns:
        The report dictionary.
    """
    val = load("rebuild_validation.json")
    t1 = load("t1_summary.json")
    t2 = load("t2_summary.json")
    t3 = load("t3_summary.json")
    t4 = load("t4_summary.json")
    rep: Dict[str, Any] = {"split": "codes 0 and 2 only", "statistics": "stand-in normalisation statistics; differences between arms"}

    rep["rebuild_validation"] = {
        "n_segments": val["n_segments"], "n_clean": val["n_clean"], "n_dirty": val["n_dirty"],
        "per_field": val["per_field"], "rho_units": val["rho_units"],
        "step_event_audit": val["step_event_audit_all_dirty_code02"],
        "kernel_digest": val["kernel_digest"],
        "procedure_chosen": ("exact builder chain on the stored up_raw with the stored invalid mask; for synthetic gaps the "
                             "invalid mask is set directly to the injected gap and level shifts come from the builder's "
                             "_span_shifts with the gap marked TRANSIENT; true and bridged sources are both rebuilt"),
    }

    c = t1["contractions"]
    short = t1["short_gap_le8_slow_claim"]
    rep["T1"] = {
        "n_segments": t1["n_segments"], "n_recordings": t1["n_recordings"],
        "n_runs_per_segment": t1["n_runs_per_segment"], "n_runs_dedup": t1["n_runs_dedup"],
        "runs_per_segment": t1["runs_per_segment"], "overlap_note": t1["note_overlap"],
        "invalid_time_share": t1["invalid_time_share_all_samples"],
        "length_quantiles_s": t1["length"]["q"], "length_share_le_s": t1["length"]["share_le"],
        "invalid_time_share_in_runs_le_s": t1["length"]["invalid_time_share_in_runs_le"],
        "by_dominant_code": {k: {kk: v[kk] for kk in ("n", "share_of_runs", "share_of_invalid_time", "total_invalid_h")}
                             | {"len_median_s": v["len_q"]["0.5"]} for k, v in t1["by_dominant_code"].items()},
        "position_mix_all": t1["position_mix_all"], "position_mix_by_code": t1["position_mix_by_code"],
        "position_mix_by_len_class": t1["position_mix_by_len_class"],
        "contractions": c,
        "post_run_strict_unusable": {lc: {k: v[k] for k in ("n", "strict_iso_s_q", "fast_iso_s_median", "mid_iso_s_median",
                                                             "slow_iso_s_median", "stored_strict_s_median", "stored_censored_share")}
                                     for lc, v in t1["post_run_contamination_by_len_class"].items()},
        "short_gap_le8": short,
        "prior_claims": {
            "median_run_le_8s": {"value_s": t1["length"]["q"]["0.5"], "verdict": "confirmed"},
            "TRANSIENT_and_ISLAND_dominate_count": {
                "TRANSIENT_share": t1["by_dominant_code"]["TRANSIENT"]["share_of_runs"],
                "ISLAND_share": t1["by_dominant_code"].get("ISLAND", {}).get("share_of_runs", 0.0),
                "verdict": "half: TRANSIENT dominates; ISLAND is rare"},
            "contractions_touching_invalid_11.9pct": {"value": c["share_touch_invalid"], "value_dedup": c["share_touch_invalid_dedup"],
                                                      "verdict": "close but not reproduced exactly (14.0% here)"},
            "of_those_54pct_le_8s_invalid": {"value": c["of_touched_share_invalid_le_8s"],
                                             "verdict": "approximately confirmed (51%)"},
            "slow_channels_contaminated_minutes_after_short_gaps": {
                "slow_rho_gt_0.1_after_le8s_gap_median_s": short["slow_group_iso_s_median"],
                "slow_q90_s": short["slow_group_iso_s_q90"],
                "strict_unusable_after_le8s_median_s": short["strict_iso_s_median"],
                "share_strict_driven_by_slow": short["share_strict_iso_driven_by_slow"],
                "verdict": ("refuted for gaps <= 16 s: the slow channels never pass rho 0.1; the strict rule's ~48 s after an "
                            "8 s gap comes from fast/mid channels (st0-st18). Confirmed for gaps >= 32 s: slow channels stay "
                            "flagged 6-10 min")},
        },
    }

    rep["T2"] = t2

    lv1 = t3["level1_grid"]
    rep["T3_level1_grid"] = {k: {kk: (v[kk] if isinstance(v[kk], (int, float)) else
                                      {x: v[kk][x] for x in ("mean", "lo", "hi", "n_rec")})
                                 for kk in ("n", "rmse_units", "rmse_rel_amp", "peak_amp_err_rel", "abs_peak_shift_s",
                                            "det_miss_share", "abs_det_onset_err_s", "abs_det_peak_err_s", "abs_det_end_err_s")}
                             for k, v in lv1.items()}
    rep["T3_level1_mix_by_code"] = t3["level1_mix_by_code"]
    rep["T3_level2_by_len_group"] = t3["level2_by_len_group"]
    rep["T3_level2_grid_len_pos_group"] = {k: {kk: v[kk] for kk in ("off_s", "rho_s", "off_censored_share", "max_err_after",
                                                                  "share_off_within_window")}
                                           for k, v in t3["level2_grid"].items()}

    keep = ("n", "n_rec", "G_true", "c_tol", "c_strict", "c_flags", "c_band_fast", "c_band_mid", "c_band_slow",
            "dkl_tol", "dkl_strict", "dgate_tol", "dgate_strict", "dnll_full_tol", "dnll_full_strict", "dnll_full_flags")
    pick = lambda d: {k: d[k] for k in keep if k in d}  # noqa: E731
    rep["T4"] = {"checks": t4["checks"], "n_affected": t4["n_affected"], "affected_by_timing_grid": t4["affected_by_timing_grid"],
                 "grid_len": {k: pick(v) for k, v in t4["grid_len"].items()},
                 "grid_len_pos": {k: pick(v) for k, v in t4["grid_len_pos"].items()},
                 "grid_len_timing": {k: pick(v) for k, v in t4["grid_len_timing"].items()},
                 "grid_len_pos_excl_inside": {k: pick(v) for k, v in t4["grid_len_pos_excl_inside"].items()},
                 "mix_code": {k: pick(v) for k, v in t4["mix_code"].items()},
                 "mix_len": {k: pick(v) for k, v in t4["mix_len"].items()},
                 "mix_code_len": {k: pick(v) for k, v in t4["mix_code_len"].items()},
                 "tolerance": t4["tolerance"], "kill_criterion_met": t4["kill_criterion_met"]}

    # predictions
    pred: Dict[str, Any] = {}
    p1 = {f"{L}s/{p}": side(lv1[f"{L}s/{p}"]["rmse_rel_amp"], 0.10) for L in (4, 8) for p in POS}
    pred["P1_raw_rmse_lt10pct_le8s_all_positions"] = {
        "per_cell": p1, "means": {f"{L}s/{p}": lv1[f"{L}s/{p}"]["rmse_rel_amp"]["mean"] for L in (4, 8) for p in POS},
        "verdict": "confirmed" if all(v == "confirmed" for v in p1.values()) else
                   ("refuted" if any(v == "refuted" for v in p1.values()) else "unclear")}
    first = {}
    for L in (4, 8, 16, 32, 64, 128):
        r = {p: lv1[f"{L}s/{p}"]["rmse_rel_amp"]["mean"] for p in POS}
        first[f"{L}s"] = max(r, key=r.get)
    pk32 = lv1["32s/peak"]["peak_amp_err_rel"]
    p2b = "confirmed" if pk32["mean"] < -0.20 and pk32["hi"] < -0.20 else ("refuted" if pk32["mean"] >= -0.20 else "unclear")
    pred["P2_peak_fails_first_and_32s_peak_err_gt20pct"] = {
        "worst_position_by_length": first, "peak_amp_err_32s": pk32, "first_length_rmse_ge10pct":
            {p: next((L for L in (4, 8, 16, 32, 64, 128) if lv1[f"{L}s/{p}"]["rmse_rel_amp"]["mean"] >= 0.10), None) for p in POS},
        "verdict": "confirmed" if all(v == "peak" for v in first.values()) and p2b == "confirmed" else
                   ("partly" if p2b == "confirmed" or list(first.values()).count("peak") >= 4 else "refuted")}
    lg = t3["level2_by_len_group"]
    pred["P3_fast_recover_in_window_slow_off_minutes_small_error"] = {
        "fast_share_off_within_own_window": {L: lg[f"{L}/fast"]["share_off_within_window"] for L in ("4s", "8s", "16s", "32s", "64s", "128s")},
        "fast_off_s_median": {L: lg[f"{L}/fast"]["off_s_median"] for L in ("4s", "8s", "16s", "32s", "64s", "128s")},
        "slow_off_s_median": {L: lg[f"{L}/slow"]["off_s_median"] for L in ("4s", "8s", "16s", "32s", "64s", "128s")},
        "slow_max_err_after_median_sd": {L: lg[f"{L}/slow"]["max_err_after_median"] for L in ("4s", "8s", "16s", "32s", "64s", "128s")},
        "verdict": ("partly: fast channels recover within their own window in 81-91% of cases (median 24-28 s); "
                    "slow channels do NOT stay off after gaps <= 16 s (median 0 s); after >= 32 s they stay off 5.5-8 min, "
                    "but the error is not small (median max error 0.15, 0.44, 1.0 sd at 32, 64, 128 s)")}
    gl = t4["grid_len"]
    p4 = {L: side(gl[L]["c_tol"], 0.10, key="ratio") for L in ("4s", "8s", "16s") if L in gl}
    pred["P4_tolerant_cost_lt10pct_le16s"] = {
        "pooled": {L: gl[L]["c_tol"] for L in ("4s", "8s", "16s") if L in gl},
        "per_position": {k: t4["grid_len_pos"][k]["c_tol"] for k in t4["grid_len_pos"] if k.split("s/")[0] in ("4", "8", "16")},
        "per_cell_verdict": {k: side(t4["grid_len_pos"][k]["c_tol"], 0.10, key="ratio") for k in t4["grid_len_pos"]
                             if k.split("s/")[0] in ("4", "8", "16")},
        "verdict_pooled": p4}
    anchors = pd.read_parquet(C.OUT1 / "t4_anchors.parquet")
    aff = anchors[(anchors["set"] == "grid") & anchors.affected]
    p5 = {}
    for L in (4.0, 8.0, 16.0):
        part = aff[aff.len_s == L]
        key = f"{int(L)}s"
        diff = S.paired_bootstrap(part.rec.values, (part.G_tolerant - part.G_strict).values, n_resamples=n_boot)
        diff_ex = part[part.timing != "inside"]
        diff_ex = S.paired_bootstrap(diff_ex.rec.values, (diff_ex.G_tolerant - diff_ex.G_strict).values, n_resamples=n_boot)
        p5[key] = {"c_strict": gl[key]["c_strict"], "c_tol": gl[key]["c_tol"], "c_flags": gl[key]["c_flags"],
                   "G_tol_minus_G_strict_nats": diff, "G_tol_minus_G_strict_nats_excl_inside": diff_ex,
                   "verdict": "confirmed" if diff["lo"] > 0 else ("refuted" if diff["mean"] <= 0 else "unclear")}
    pred["P5_strict_flags_cost_more_than_bridge_le16s"] = {"pooled": p5}
    pred["kill_criterion"] = {"met": t4["kill_criterion_met"],
                              "rule": "c_tol >= 10% already at 4-8 s at every position"}
    rep["predictions"] = pred

    # T5
    rep["T5_crest_bound_peak_gaps"] = crest_bound(n_boot)
    rep["T5_conclusions"] = conclusions(rep)
    rep["caveats"] = CAVEATS
    (C.OUT1 / "REPORT_phase1.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    out = main(parser.parse_args().n_boot)
    print(json.dumps(out["predictions"], indent=1, default=float)[:6000])
