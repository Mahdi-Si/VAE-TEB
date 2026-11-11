r"""Phase 1: assemble the detector audit into ``out/detectors/REPORT_detectors.json``.

Reads ``contractions.json`` (:mod:`p1_detector_contractions`), ``decels.json``
(:mod:`p1_detector_decels`) and ``improved.json`` (:mod:`p1_detector_improved`), pulls the
headline numbers, and applies the pre-registered verdict rule to each detector:

* **WEAK** if recall $< 0.80$, or the onset-error IQR $> 20\,$s, or the overlap agreement
  $< 0.85$;
* **FINE** if recall $\ge 0.90$, the onset-error IQR $\le 12\,$s and the agreement $\ge 0.90$;
* **MIXED** otherwise.

The rule reads the point estimates. The interval of each number is copied next to it.

The *as-trained* recall counts every reference contraction whose peak lies off the $30\,$s edges,
including those whose span touches an invalid UA sample, because the training reward misses
those too. The *core* recall counts only reference contractions with a fully valid span, which
isolates the peak finder from the validity rule. The agreement is the lowest re-cut agreement
with a $60\,$s margin.

Edit :data:`RUN_ARGS` and run this file from the IDE; it takes no arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import p1_detector_io as IO  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"out": str(IO.OUT_DIR / "REPORT_detectors.json")}


#: What the panel figures show, written after reading them (codes 0 and 2 only).
VISUAL_NOTES: Dict[str, Any] = {
    "contractions_shipped": [
        "Clear contractions are dropped when a short UA-invalid strip (mostly TRANSIENT) sits on the apex or in "
        "the rise: 279191C8 (1 of 8 detected), B555FDA6 (3 of 8), BB02F8B2 (2 of 9), FF117DE1, A5668EBD, 3CF78167.",
        "False positives on 5 to 10 unit bumps in quiet or noisy segments, because 0.5 sigma of a flat segment is "
        "tiny: 24B36776, BCD598A8, E8EA1E04, 51569450, 24DC8F56.",
        "A large step in uterine tone inflates sigma and clear 25-unit contractions are missed: B60C0F79 (0 vs 10).",
        "Multi-peak stretches are merged into long spans: 76E3985C, 04FF0413.",
        "Onsets are mostly at the foot of the rise; no systematic mid-flank onsets were seen.",
        "Regular 10 to 17 unit bumps in the latent phase (F7B014CF, 264F4259) are ambiguous: the shipped detector "
        "calls them contractions, the theta=20 reference does not.",
    ],
    "decelerations_shipped": [
        "Most events are 10-bpm troughs of normal variability or the return to baseline after an acceleration "
        "(prominence is measured from the acceleration apex): 700C98FA, 2AC3BCF5, 7AFC78FE.",
        "Clear NICHD decelerations are found (high recall), but spans are often long and merged (279191C8).",
    ],
    "contractions_v7_code2": [
        "Recovers apex-flagged contractions (29CD237E at 2.3 min) and drops flat-trace bumps.",
        "Misses every bump on low-gain recordings whose whole trace spans about 10 units (59167804): an absolute "
        "threshold cannot see them; the shipped detector called 4 there.",
        "Still misses spike-topped contractions (3A82A528 at 2.5 h: 10.3, 15.6, 19.1 min): the 10-s Savitzky-Golay "
        "trace keeps a 5-9 s apex spike, find_peaks' distance rule keeps the spike and the width rule then rejects "
        "it. Testing width before spacing gained only 0.4 recall points on code 0. Some of these may be artifacts "
        "that the 0.04 Hz reference smears into contraction-like bumps, so the 'width' miss class is ambiguous.",
        "One new false positive on a noisy flat trace (3DCDCB03 at 3.0 min).",
    ],
    "decelerations_v7_code2": [
        "Reports only NICHD-sized dips, but misses deep decelerations whose span touches an invalid FHR sample "
        "(5E139862 at 15-16.5 and 19.3-21 min, F4C311D0 at 10.6 min): it inherits the strict any-invalid-sample "
        "drop rule. Not clearly better than the shipped detector as an event source.",
    ],
}

#: Caveats a reader needs before using any number here.
CAVEATS = [
    "No ground truth: recall and precision are against a detector-free reference whose threshold (20 toco units) "
    "was set by a two-component mixture on code-0 bump amplitudes; at theta=12 or 30 the shipped precision moves "
    "from 0.92 to 0.61 (code 0).",
    "v7 measures its rise against the same kind of rolling low-quantile baseline as the reference, so part of its "
    "precision gain is built in. The recall gain on reference events with at most 8 s invalid and the overlap "
    "and panel checks do not share that construction.",
    "The shipped share of peaks inside the 60-s reward window (100%) is guaranteed by the 60-s onset walk bound; "
    "12% of shipped onsets and 32% of ends sit on that bound (code 0).",
    "v7 thresholds are in stored toco units; the training batch's up is z-scored, so v7 must be called with "
    "scale = the up std of the loader's statistics file (19.94 in the local stand-in statistics; the training "
    "statistics are not local).",
    "The native neighbour overlap is only 120 s, so the 60-s-margin overlap test runs on re-cut windows of "
    "stitched runs; with fixed absolute validity it mostly measures the sigma and edge dependence.",
    "v7 durations above 180 s (4-5%) come from the longer 120-s walk on merged or plateau contractions.",
]


def verdict(recall: float, iqr: float, agreement: float) -> str:
    """Apply the pre-registered rule; see the module docstring."""
    if recall < 0.80 or iqr > 20.0 or agreement < 0.85:
        return "WEAK"
    if recall >= 0.90 and iqr <= 12.0 and agreement >= 0.90:
        return "FINE"
    return "MIXED"


def ci(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Keep the mean and the interval of a bootstrap result."""
    if not isinstance(d, dict) or "mean" not in d:
        return None
    return {"mean": d["mean"], "lo": d["lo"], "hi": d["hi"], "n_rec": d.get("n_rec")}


def min_agreement(block: Dict[str, Any]) -> Dict[str, Any]:
    """The lowest re-cut agreement with a 60 s margin, and its key."""
    keys = [k for k in block if k.startswith("offset_")]
    worst = min(keys, key=lambda k: block[k]["agreement"]["mean"])
    return {"key": worst, **ci(block[worst]["agreement"]), "onset_diff_iqr": ci(block[worst]["onset_diff_iqr"])}


def contraction_section(c: Dict[str, Any], im: Dict[str, Any]) -> Dict[str, Any]:
    """Headline numbers and verdicts of the shipped contraction detector."""
    out: Dict[str, Any] = {}
    agree = min_agreement(c["overlap_shipped"])
    out["overlap_recut"] = agree
    out["overlap_native_120s_margin30"] = {k: ci(v) for k, v in c["overlap_shipped"]["native_120s_margin30"].items()
                                           if isinstance(v, dict)}
    for code in ("code0", "code2", "code0+2"):
        b = c[code]
        r = b["reference_theta20"]
        trained = r["recall_all_offedge"]["mean"]
        core = r["recall_eligible"]["mean"]
        iqr = r["onset_error_iqr"]["mean"]
        out[code] = {
            "verdict_as_trained": verdict(trained, iqr, agree["mean"]),
            "verdict_core_peak_finder": verdict(core, iqr, agree["mean"]),
            "recall_as_trained": ci(r["recall_all_offedge"]),
            "recall_core_valid_span": ci(r["recall_eligible"]),
            "precision_vs_ref20": ci(r["precision"]),
            "recall_theta12": ci(b["reference_theta12"]["recall_eligible"]),
            "precision_theta12": ci(b["reference_theta12"]["precision"]),
            "recall_theta30": ci(b["reference_theta30"]["recall_eligible"]),
            "precision_theta30": ci(b["reference_theta30"]["precision"]),
            "onset_error_median_s": ci(r["onset_error_median"]),
            "onset_error_iqr_s": ci(r["onset_error_iqr"]),
            "end_error_iqr_s": ci(r["end_error_iqr"]),
            "miss_reasons_core": r.get("miss_reasons"),
            "unmatched_shipped_rise_pct": r.get("unmatched_shipped_rise_pct"),
            "rate_per10min": ci(b["rates_shipped"]["all"]),
            "rate_per10min_clean_segments": ci(b["rates_shipped_clean_segments"]["all"]),
            "rate_reference_per10min": ci(b["rates_reference"]["all"]),
            "rate_by_hours": {k: ci(v) for k, v in b["rates_shipped"].items() if k.startswith("hours=")},
            "rate_reference_by_hours": {k: ci(v) for k, v in b["rates_reference"].items() if k.startswith("hours=")},
            "rate_by_cohort": {k: ci(v) for k, v in b["rates_shipped"].items() if k.startswith("cohort=")},
            "share_segments_gt5_per10min": ci(b["rates_shipped"]["share_segments_rate_gt5"]),
            "duration_s": b["shapes_shipped"]["duration_s"],
            "onset_to_peak_s": b["shapes_shipped"]["onset_to_peak_s"],
            "peak_interval_s": b["shapes_shipped"]["peak_interval_s"],
            "share_duration_lt20": ci(b["shapes_shipped"]["share_duration_lt20"]),
            "share_duration_gt180": ci(b["shapes_shipped"]["share_duration_gt180"]),
            "share_interval_lt60": ci(b["shapes_shipped"]["share_interval_lt60"]),
            "validity_rule": {
                "share_dropped": ci(b["validity_rule"]["share_predrop_dropped_by_ua_rule"]),
                "share_of_dropped_with_le8s_invalid": ci(b["validity_rule"]["share_dropped_with_le8s_invalid"]),
                "dropped_invalid_seconds_pct": b["validity_rule"]["dropped_invalid_seconds_pct"],
                "invalid_bits_in_dropped_spans": b["validity_rule"]["invalid_bits_in_dropped_spans"],
                "reward_mass_removed": ci(b["validity_rule"]["reward_mass_removed_by_drop_only"]),
            },
            "invalid_rate_by_phase": {k: ci(v) for k, v in b.get("invalid_by_phase", {}).items() if isinstance(v, dict) and "mean" in v},
            "apex_over_between_invalid_rate": b.get("invalid_by_phase", {}).get("apex_over_between_rate"),
            "apex_minus_between_invalid_rate": ci(b.get("invalid_by_phase", {}).get("apex_minus_between_rate")),
            "reward_window": {k: ci(v) for k, v in b["reward_window"].items()},
        }
        if code in im:
            out[code]["walk_bound"] = {k: ci(v) for k, v in im[code]["walk_bound_shipped"].items()}
    vr = c["code0+2"]["validity_rule"]
    drop, le8 = vr["share_predrop_dropped_by_ua_rule"], vr["share_dropped_with_le8s_invalid"]
    out["prior_claim_2d"] = {
        "claim": "11.9% of contractions dropped; 54% of those with <= 8 s invalid",
        "measured_share_dropped_code0+2": ci(drop),
        "measured_share_le8s_code0+2": ci(le8),
        "share_dropped_claim_inside_ci": drop["lo"] <= 0.119 <= drop["hi"],
        "share_le8s_claim_inside_ci": le8["lo"] <= 0.54 <= le8["hi"],
    }
    return out


def decel_section(d: Dict[str, Any]) -> Dict[str, Any]:
    """Headline numbers and verdict of the shipped deceleration detector."""
    out: Dict[str, Any] = {"overlap_recut": min_agreement(d["overlap_shipped"])}
    for code in ("code0", "code2"):
        b = d[code]
        r = b["reference"]
        out[code] = {
            "verdict": verdict(r["recall"]["mean"], r["onset_error_iqr"]["mean"], out["overlap_recut"]["mean"]),
            "recall_vs_nichd_reference": ci(r["recall"]),
            "precision_vs_nichd_reference": ci(r["precision_vs_reference"]),
            "onset_error_median_s": ci(r["onset_error_median"]),
            "onset_error_iqr_s": ci(r["onset_error_iqr"]),
            "rate_per10min": ci(b["rates_shipped"]["all"]),
            "rate_nichd_qualified_per10min": ci(b["rates_shipped_nichd_iter10"]["all"]),
            "rate_reference_per10min": ci(b["rates_reference"]["all"]),
            "rate_nichd_by_hours": {k: ci(v) for k, v in b["rates_shipped_nichd_iter10"].items() if k.startswith("hours=")},
            "rate_nichd_by_cohort": {k: ci(v) for k, v in b["rates_shipped_nichd_iter10"].items() if k.startswith("cohort=")},
            "share_nichd_qualified": {k: ci(b[f"baseline={k}"]["share_nichd"]) for k in ("iter10", "med5", "p60_10")},
            "share_depth_below_baseline_lt15": ci(b["baseline=iter10"]["share_depth_lt15"]),
            "share_prominence_lt15": ci(b["share_prominence_lt15"]),
            "share_duration_lt15": ci(b["share_duration_lt15"]),
            "share_duration_ge120": ci(b["share_duration_ge120"]),
            "prominence_bpm": b["shipped_prominence_bpm"],
            "duration_s": b["shipped_duration_s"],
            "onset_to_nadir_s": b["shipped_onset_to_nadir_s"],
            "baseline_sensitivity": {k: ci(b[k]) for k in b if k.startswith("nichd_share_delta") or k.startswith("nichd_flag_flip")},
            "timing_nichd_vs_shipped_contractions": {k: ci(v) for k, v in b["timing_vs_ctx_ship"]["nichd_qualified"].items()},
            "timing_nichd_vs_reference_contractions": {k: ci(v) for k, v in b["timing_vs_ctx_ref"]["nichd_qualified"].items()},
            "nadir_minus_peak_s": b["timing_vs_ctx_ship"]["nadir_minus_peak_s"],
            "nadir_minus_peak_median": ci(b["timing_vs_ctx_ship"]["nadir_minus_peak_median"]),
        }
    return out


def improved_section(im: Dict[str, Any]) -> Dict[str, Any]:
    """Headline numbers of the v7 detectors and the activity, with the v7 verdict per code."""
    out: Dict[str, Any] = {"v7_params": im["v7_params"], "activity_params": im["activity_params"],
                           "grid_code0": im.get("grid_code0"), "activity_causality": im["activity_causality"]}
    for code in ("code0", "code2"):
        b = im[code]
        agree = min_agreement(b["overlap_v7"])
        rec = b["recall_ref20_offedge_all"]
        iqr = b["v7_vs_ref20"]["onset_error_iqr"]["mean"]
        w = b["weights_and_activity"]
        out[code] = {
            "role": "tune" if code == "code0" else "validate",
            "verdict_v7_as_deployed": verdict(rec["v7"]["mean"], iqr, agree["mean"]),
            "recall_offedge_all": {k: ci(v) if isinstance(v, dict) else v for k, v in rec.items()},
            "recall_offedge_le8s_invalid": {k: ci(v) if isinstance(v, dict) else v for k, v in b["recall_ref20_offedge_le8s_invalid"].items()},
            "recall_valid_span": {k: ci(v) if isinstance(v, dict) else v for k, v in b["recall_ref20_valid_span"].items()},
            "precision_ref20": {k: ci(v) for k, v in b["precision_ref20"].items()},
            "precision_ref12": {k: ci(v) for k, v in b["precision_ref12"].items()},
            "v7_onset_error_median_s": ci(b["v7_vs_ref20"]["onset_error_median"]),
            "v7_onset_error_iqr_s": ci(b["v7_vs_ref20"]["onset_error_iqr"]),
            "v7_overlap_recut": agree,
            "rate_shipped_per10min": ci(b["rates_shipped"]["all"]),
            "rate_v7_per10min": ci(b["rates_v7"]["all"]),
            "rate_v7_by_hours": {k: ci(v) for k, v in b["rates_v7"].items() if k.startswith("hours=")},
            "rate_v7_minus_shipped": ci(b["rate_v7_minus_shipped_per10min"]),
            "v7_duration_s": b["shapes_v7"]["duration_s"],
            "v7_onset_to_peak_s": b["shapes_v7"]["onset_to_peak_s"],
            "v7_share_duration_gt180": ci(b["shapes_v7"]["share_duration_gt180"]),
            "v7_share_interval_lt60": ci(b["shapes_v7"]["share_interval_lt60"]),
            "walk_bound_v7": {k: ci(v) for k, v in b["walk_bound_v7"].items()},
            "v7_validity_drop_share": {k: ci(v) for k, v in b["v7_validity_drop_share"].items()},
            "reward_share_valid_steps_shipped": ci(w["share_valid_steps_weight1_shipped"]),
            "reward_share_valid_steps_v7": ci(w["share_valid_steps_weight1_v7"]),
            "reward_mass_v7_over_shipped": ci(w["reward_mass_v7_over_shipped"]),
            "confusion_share_of_valid_steps": w["confusion_share_of_valid_steps"],
            "activity": {k: ci(v) for k, v in w.items() if k.startswith(("activity_", "mean_activity", "shipped1_given"))},
            "decel_v7": {
                "rate_per10min": ci(b["decel_v7"]["rates_v7"]["all"]),
                "rate_by_hours": {k: ci(v) for k, v in b["decel_v7"]["rates_v7"].items() if k.startswith("hours=")},
                "share_prolonged": ci(b["decel_v7"]["share_prolonged"]),
                "depth_bpm": b["decel_v7"]["depth_bpm"], "duration_s": b["decel_v7"]["duration_s"],
                "timing_classes_vs_v7_contractions": {k: ci(v) for k, v in b["decel_v7"]["timing_classes"].items()},
                "nadir_minus_peak_s": b["decel_v7"]["nadir_minus_peak_s"],
                "overlap_recut": min_agreement(b["decel_v7_overlap"]),
            },
        }
    return out


def main(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble and write the report."""
    c = json.loads((IO.OUT_DIR / "contractions.json").read_text())
    d = json.loads((IO.OUT_DIR / "decels.json").read_text())
    im = json.loads((IO.OUT_DIR / "improved.json").read_text())
    report = {
        "scope": "split codes 0 (tune) and 2 (check) only; recording-level bootstrap, 2000 resamples, 95% percentile CI",
        "verdict_rule": "WEAK if recall<0.80 or onset IQR>20 s or overlap agreement<0.85; FINE if recall>=0.90 and IQR<=12 s and agreement>=0.90; else MIXED",
        "training_call": "lag_attn_cfs.eval.events.detect_contractions on the bridged up (z-scored by the loader; sigma-relative so units do not matter) under valid = (step weight > 0) & (up_quality & 2299 == 0); reward 1 on steps onset..onset+15",
        "segment_geometry": "epoch = segment start; neighbours start 1200 s apart and share 120 s of identical signal; overlap test re-cuts stitched runs at offsets 330 s and 660 s with a 60 s margin",
        "contractions_shipped": contraction_section(c, im),
        "decelerations_shipped": decel_section(d),
        "improved": improved_section(im),
        "visual_notes": VISUAL_NOTES,
        "caveats": CAVEATS,
        "figures": {"contractions_shipped": c.get("panels"), "decels_shipped": d.get("panels"),
                    "contractions_v7_code2": im.get("panels_contractions_v7"),
                    "decels_v7_code2": im.get("panels_decels_v7")},
    }
    out = Path(args.out)
    out.write_text(json.dumps(IO.to_jsonable(report), indent=1))
    print("wrote", out)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line, defaulting to :data:`RUN_ARGS`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=RUN_ARGS["out"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
