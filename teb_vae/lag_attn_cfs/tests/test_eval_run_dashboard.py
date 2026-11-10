r"""The run dashboard: it builds from files alone, it survives a partial directory, and it explains.

The page is a view of a run directory, so its three properties are about files. A directory with
few tables must still give a page, with a card that names each missing file. A directory filled by
several passes must be read as one run. And every plot must carry its three explanations, because
a plot with no explanation is the thing the page exists to replace.
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.nets import lag_report
from teb_vae.lag_attn_cfs import gate_states
from teb_vae.lag_attn_cfs.eval import run_dashboard, run_dashboard_core, run_dashboard_gate
from teb_vae.lag_attn_cfs.eval.analyses import gate_open
from teb_vae.lag_attn_cfs.eval.run_dashboard_core import (
    Geometry,
    MissingInput,
    RunFiles,
    encode_array,
    heatmap,
    json_safe,
)


def _decode(packed: dict) -> np.ndarray:
    """Invert :func:`encode_array` the way the page script does."""
    import base64

    raw = np.frombuffer(base64.b64decode(packed["b"]), dtype="<i2").astype(np.float64)
    values = packed["o"] + packed["s"] * raw
    values[raw == -32768] = np.nan
    return values.reshape(packed["d"])


def _gate_open_directory(path) -> None:
    """Write the ``gate_open`` tables of a small synthetic run into a results directory."""
    # Four recordings of 32 anchors each, so every recording has 20 gate-on anchors in one window.
    pattern = np.tile(np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.1, 0.5]), 4)
    gate = np.tile(pattern, 4)
    kl = np.tile(np.tile(np.arange(8, dtype=float), 4), 4)
    per_anchor = pd.DataFrame({
        "guid": np.repeat(["G0", "G1", "G2", "G3"], 32), "epoch": -3600.0,
        "anchor": np.tile(np.arange(10, 42), 4), "event_gate": gate, "kld_per_t": kl,
        "mc_pred_gap": 2.0 * kl, "argmax_lag": np.where(gate >= 0.2, 2, 0),
        "seconds_since_contraction": np.tile(10.0 * np.arange(32), 4),
    })
    kl_map = kl[:, None] * np.where((gate >= 0.2)[:, None], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0])
    collection = types.SimpleNamespace(
        per_anchor=per_anchor,
        per_sample=pd.DataFrame({"guid": ["G0", "G1", "G2", "G3"],
                                 "clinical_class": ["healthy", "healthy", "acidosis", "acidosis"]}),
        anchor_vectors={"kl_lag_map": kl_map, "attention_lag_map": np.full((128, 3), 1 / 3)},
        results={"lag": {"n_lags": 3, "delay_steps": 0}},
    )
    gate_open.run_gate_open_analysis(
        types.SimpleNamespace(collection=collection), eval_config={"bootstrap_resamples": 20},
        output_dir=path,
    )


def test_an_encoded_array_decodes_to_its_values_and_keeps_a_gap() -> None:
    """Sixteen bits resolve the range to one part in 65534, and a ``NaN`` stays a ``NaN``."""
    values = np.array([[0.0, 1.5, np.nan], [-3.0, 2.25, 10.0]])
    decoded = _decode(encode_array(values))
    assert np.isnan(decoded[0, 2])
    assert np.nanmax(np.abs(decoded - values)) <= 13.0 / 65534.0


def test_a_payload_value_that_is_not_finite_becomes_null() -> None:
    """``json.dumps(allow_nan=False)`` must accept everything a builder can return."""
    payload = json_safe({"a": np.float32("nan"), "b": [np.int64(3), float("inf")], "c": np.array([1.0, np.nan])})
    assert payload == {"a": None, "b": [3, None], "c": [1.0, None]}
    json.dumps(payload, allow_nan=False)


def test_a_logarithmic_heatmap_colours_the_logarithm_and_hovers_the_value() -> None:
    """The colour is $\\log_{10} z$ with the floor applied, and the hover data is $z$ itself."""
    trace = heatmap([0, 1], [0], [[0.1, 0.0]], label="share", log=True, floor=1e-3)
    assert trace["z"] == [[pytest.approx(-1.0), pytest.approx(-3.0)]]
    assert trace["customdata"] == [[0.1, 0.0]]


def test_the_page_restates_no_constant_its_owner_could_change() -> None:
    """The step length is restated to keep the module free of ``torch``, so it is pinned here."""
    assert run_dashboard_core.SECONDS_PER_STEP == lag_report.SECONDS_PER_STEP
    assert run_dashboard_core.GATE_SHUT_BELOW is gate_states.GATE_SHUT_BELOW
    assert run_dashboard_core.GATE_OPEN_FROM is gate_states.GATE_OPEN_FROM
    assert gate_open.SECONDS_PER_STEP == lag_report.SECONDS_PER_STEP


def test_the_lag_axis_and_the_texts_follow_the_run_geometry(tmp_path) -> None:
    """A delay of two steps moves the lag axis by two steps, and the texts quote the run's numbers."""
    summary = {"results": {"lag": {"n_lags": 3, "delay_steps": 2}},
               "collection": {"geometry": {"t": 150, "horizon": 10, "anchors_per_sample": 40}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    geometry = Geometry(RunFiles(tmp_path))
    assert geometry.lag_seconds() == [8.0, 12.0, 16.0]
    assert geometry.band_seconds(0, 1) == "8 to 12 s"
    assert "3 lags" in geometry.lags_text and "10 steps (40 s)" in geometry.horizon_text
    assert geometry.segment_text == "10-minute segment" and "40" in geometry.anchors_text
    # A directory with no summary states no number at all.
    empty = Geometry(RunFiles(tmp_path / "none"))
    assert empty.lag_seconds() == [] and empty.segment_text == "segment"


def test_the_summary_is_merged_from_every_pass_newest_first(tmp_path) -> None:
    """A later partial pass must not hide the result block an earlier pass wrote."""
    older = {"checkpoint": "old",
             "results": {"forecast": {"n": 1}, "headline": {"a": 1.0, "b": 2.0},
                         "sanity": {"checks": {"argmax_lag": {"verdict": "fail"}}}},
             "steps": [{"name": "forecast", "ok": True}, {"name": "attention", "ok": True}]}
    newer = {"checkpoint": "new",
             "results": {"gate_open": {"n": 2}, "headline": {"a": 5.0, "b": None},
                         "sanity": {"checks": {"argmax_lag": {"verdict": "inconclusive"}}}},
             "steps": [{"name": "gate_open", "ok": True}]}
    backup = tmp_path / "summary.bak.2026-01-01--[00-00-00].json"
    backup.write_text(json.dumps(older), encoding="utf-8")
    os.utime(backup, (1_000_000, 1_000_000))
    (tmp_path / "summary.json").write_text(json.dumps(newer), encoding="utf-8")
    run = RunFiles(tmp_path)
    assert run.summary["checkpoint"] == "new"
    assert run.result("forecast") == {"n": 1} and run.result("gate_open") == {"n": 2}
    # The newest value wins, and a null does not hide an older value.
    assert run.headline("a") == 5.0 and run.headline("b") == 2.0
    assert {step["name"] for step in run.summary["steps_merged"]} == {"forecast", "attention", "gate_open"}
    # The consistency checks come from the pass that ran the most steps: the one-analysis pass
    # could not test the attention block and must not replace the check that did.
    assert run.result("sanity")["checks"]["argmax_lag"]["verdict"] == "fail"
    assert run.summary["derived_blocks_from"] == backup.name


def test_a_missing_table_raises_and_names_the_analysis(tmp_path) -> None:
    """The exception carries what the card needs to say: the file and who writes it."""
    with pytest.raises(MissingInput) as caught:
        RunFiles(tmp_path).csv("gate_open/gate_open_summary.csv")
    assert caught.value.path == "gate_open/gate_open_summary.csv"
    assert caught.value.analysis == "gate_open"


def test_an_empty_directory_still_gives_a_page_of_named_gaps(tmp_path) -> None:
    """No table at all is a page whose every data card names its missing file."""
    payload = run_dashboard.build_payload(tmp_path)
    panels = [panel for section in payload["sections"] for panel in section["panels"]]
    assert [section["id"] for section in payload["sections"]][:2] == ["overview", "gate"]
    assert all(panel["kind"] in ("missing", "html") for panel in panels)
    assert any("gate_open_summary.csv" in panel.get("reason", "") for panel in panels)
    page = run_dashboard.render_page(payload)
    assert "__PAYLOAD__" not in page and "__JS__" not in page and "Plotly.react" in page


def test_every_gate_plot_explains_itself_and_follows_the_filter(tmp_path) -> None:
    """Each plot has its three texts, and a filtered plot has one variant per population."""
    _gate_open_directory(tmp_path)
    run = RunFiles(tmp_path)
    builders = [
        run_dashboard_gate.population_summary, run_dashboard_gate.population_contributions,
        run_dashboard_gate.class_trajectory, run_dashboard_gate.gain_vs_contraction,
        run_dashboard_gate.lag_profile_populations, run_dashboard_gate.lag_shares,
        run_dashboard_gate.lag_vs_contraction, run_dashboard_gate.lag_vs_gate,
    ]
    for builder in builders:
        panel = builder(run)
        assert panel["kind"] == "plot" and panel["variants"], builder.__name__
        for key in ("how", "shows", "read"):
            assert len(panel["explain"][key]) > 40, (builder.__name__, key)
    filtered = run_dashboard_gate.gain_vs_contraction(run)
    assert [item["pop"] for item in filtered["variants"]] == ["all", "gate_on", "fully_open", "gate_off"]
    # The gate is on at 5 of the 8 anchors of every recording.
    assert run_dashboard_gate.gate_headline(run)["gate_on_frac"] == pytest.approx(5 / 8)


def test_the_page_is_written_beside_the_tables(tmp_path) -> None:
    """One call writes the page, and the record counts the cards that lack their input."""
    _gate_open_directory(tmp_path)
    record = run_dashboard.write_dashboard(tmp_path)
    assert (tmp_path / run_dashboard.DASHBOARD_FILENAME).is_file()
    assert record["n_panels"] > len(record["missing"]) > 0


def test_the_results_directory_is_found_from_the_run_directory(tmp_path) -> None:
    """A run directory resolves to its ``eval_results`` folder, and an empty one is refused."""
    with pytest.raises(FileNotFoundError):
        run_dashboard.resolve_results_dir(tmp_path)
    results = tmp_path / run_dashboard.RESULTS_DIRNAME
    results.mkdir()
    (results / "summary.json").write_text("{}", encoding="utf-8")
    assert run_dashboard.resolve_results_dir(tmp_path) == results
    assert run_dashboard.resolve_results_dir(results) == results
