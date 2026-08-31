r"""The identifiability instrument, driven as an operator drives it.

The planted fixture is a shard in which the source determines a target feature at a known delay, so
the lags that carry information are known: a band around the plant, and nowhere else. That is what
lets it gate a lag readout, and the check script is what reads the gate.

**Its pass line is a measurement, not a test outcome, and this file does not assert it.** Whether a
given architecture recovers the planted band is the empirical question the instrument exists to
answer, and a suite that required a pass would either be red for a real finding or be quietly
loosened until it was not. What is asserted here is that the *instrument works*: it builds the model
through the driver's own config-to-constructor path, it prints the switch header that says which
configuration produced a number, it reports the KL at initialisation, its manifest describes the
model that was built, and both feature-target cells' planted configs resolve.

The manifest mode is the cheap half and carries most of the properties above -- it builds and prints
and costs no fit, so it runs in the fast subset. The fit is marked ``slow``: it is the mode an
operator runs between architecture changes, and it is here so that the path an outcome block was
written from is one the suite has executed rather than one nobody has run since.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from teb_vae.lag_attn_cfs import lag_recovery_check

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The two planted config deltas, one per feature-target cell. Both are checked because every gate
#: in this family's revision was read per parent: the two encoders answer the same question
#: differently, and a result on one is not a result on the other.
PLANTED_CONFIGS = (
    "teb_vae/lag_attn_cfs/configs/planted.yaml",
    "teb_vae/lag_attn_transformer_cfs/configs/planted.yaml",
)

#: The five switch keys the header prints, so every gate's configuration is on the page of its own
#: output rather than reconstructed from a shell history afterwards.
SWITCH_KEYS = (
    "lag_kv_source",
    "prior_availability_input",
    "persistence_residual",
    "horizon_weight_halflife_steps",
    "alibi_slope_scale",
)

#: Enough epochs that the fit runs its loop more than once and the checkpoint is written from a
#: model that moved, and few enough that the wrapper is minutes rather than the tens the recorded
#: gates were measured over. The number a gate is run at lives in the operator's own launch, not
#: here: this asserts the path, not the outcome.
WRAPPER_EPOCHS = 2


#: A synthetic lag block shaped like the finding this fixture exists to qualify: the matched
#: attribution pinned at the near edge, and a source-null arm that accounts for almost all of it
#: there -- so the CLOCK-EXCESS profile peaks inside the planted band while the pooled one does
#: not. Hand-built rather than fitted, because what is asserted below is that the record reports
#: each readout separately, which is a property of the code and not of any run.
_SYNTHETIC_N_LAGS = 11
_SYNTHETIC_BAND = (3, 6)
_SYNTHETIC_MATCHED = [5.0, 1.0, 0.5, 0.6, 0.7, 0.6, 0.5, 0.2, 0.1, 0.1, 0.1]
_SYNTHETIC_NULL = [4.9, 0.9, 0.4, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]


def _synthetic_results():
    """An ``evaluate``-shaped dict carrying every lag key the record reads."""
    excess = [m - n for m, n in zip(_SYNTHETIC_MATCHED, _SYNTHETIC_NULL)]
    return {
        "lag": {
            "delay_steps": 0,
            "n_lags": _SYNTHETIC_N_LAGS,
            "num_heads": 2,
            "kl_lag_profile": _SYNTHETIC_MATCHED,
            "kl_lag_profile_support_corrected": _SYNTHETIC_MATCHED,
            "kl_lag_profile_untruncated": _SYNTHETIC_MATCHED,
            "kl_argmax_lag_step_support_corrected": 0,
            "kl_lag_profile_null": _SYNTHETIC_NULL,
            "kl_lag_profile_clock_excess": excess,
            "kl_argmax_lag_step_clock_excess": max(
                range(_SYNTHETIC_N_LAGS), key=lambda index: max(excess[index], 0.0)
            ),
            "kl_lag_anchor_counts": [10.0] * _SYNTHETIC_N_LAGS,
            "attention_lag_profile_per_head": [
                [0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02],
                [0.02, 0.03, 0.05, 0.30, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01],
            ],
            "kl_lag_profile_per_head": [
                [2.5, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.2, 0.9, 0.8, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1],
            ],
            "attention_entropy_per_head_nats": [1.2, 2.0],
            "identity_residuals": {},
        },
        "readouts": {
            "source_conditioned_kl_raw": sum(_SYNTHETIC_MATCHED),
            "kld_source_null": sum(_SYNTHETIC_NULL),
            "coupling_minus_clock": sum(_SYNTHETIC_MATCHED) - sum(_SYNTHETIC_NULL),
        },
        "n_samples": 8,
        "n_recordings": 8,
    }


def test_the_record_scores_every_lag_readout_and_not_only_the_gated_one() -> None:
    """The record grew a clock-excess block and a per-head KL column; this asserts they are
    populated and self-consistent.

    **It does not assert which readout recovered the plant.** That is a measurement, exactly as
    this file's header says of the pass line itself -- a suite requiring ``recovered_by`` to hold
    would be red for a real finding or be loosened until it was not.
    """
    record = lag_recovery_check.recovery_record(
        _synthetic_results(),
        band=_SYNTHETIC_BAND,
        delay_steps=7,
        bands={"anchor": (0, 2), "near": (3, 6), "mid": (7, 8), "far": (9, 10)},
    )

    excess = record["clock_excess"]
    assert excess["measured"] is True
    # The identity the whole lag-resolved null exists for, restated at the record's own level: the
    # signed profile sums to the scalar the availability-clock verdict gates.
    assert excess["net_nats"] == pytest.approx(
        _synthetic_results()["readouts"]["coupling_minus_clock"]
    )
    # Shares are taken of the POSITIVE part, so they are shares.
    assert 0.0 <= excess["band_share"] <= 1.0
    assert set(excess["occlusion_band_shares"]) == {"anchor", "near", "mid", "far"}
    assert sum(excess["occlusion_band_shares"].values()) == pytest.approx(1.0)

    # Every head carries both readings, so a reader can tell an in-band attention peak from a head
    # with no KL behind it from one that actually carried the belief movement.
    assert record["heads"]
    for head in record["heads"]:
        assert "kl_argmax_lag_step" in head
        assert "kl_band_share" in head
        assert "kl_degenerate" in head

    assert set(record["recovered_by"]) == {
        "kl_support_corrected",
        "clock_excess",
        "any_head_attention",
        "any_head_kl",
    }


def test_the_pass_line_is_the_one_the_recorded_gates_were_measured_on() -> None:
    """``recovered`` and the exit code stay wired to the pooled support-corrected argmax alone.

    The new readouts are reported beside it, never folded into it: every gate already written into
    ``RESULTS.md``'s identifiability table was read on this criterion, and widening it would make
    those rows incomparable with the next ones rather than extending them.
    """
    record = lag_recovery_check.recovery_record(
        _synthetic_results(), band=_SYNTHETIC_BAND, delay_steps=7
    )

    # The matched profile peaks at lag 0 -- the near-edge pin -- so the gate fails ...
    assert record["argmax_support_corrected"] == 0
    assert record["recovered"] is False
    # ... and it fails even though another readout on the same block did find the band.
    assert record["recovered_by"]["kl_support_corrected"] is False
    assert record["recovered"] is record["recovered_by"]["kl_support_corrected"]

    # With no band partition passed, the planted band's own share is still reported: the partition
    # is a reading convenience, not a precondition.
    assert record["clock_excess"]["occlusion_band_shares"] == {}
    assert record["clock_excess"]["measured"] is True


def test_a_run_predating_the_null_profile_reports_it_absent_rather_than_recovered() -> None:
    """A run directory collected before the source-null arm was lag-resolved is a partial input.
    ``measured`` false and a NaN share, never a zero share -- a zero would read as a clock-excess
    profile that was computed and found to put nothing in the band."""
    import math

    results = _synthetic_results()
    del results["lag"]["kl_lag_profile_clock_excess"]
    del results["lag"]["kl_argmax_lag_step_clock_excess"]

    record = lag_recovery_check.recovery_record(
        results, band=_SYNTHETIC_BAND, delay_steps=7
    )

    assert record["clock_excess"]["measured"] is False
    assert math.isnan(record["clock_excess"]["band_share"])
    assert record["clock_excess"]["degenerate"] is None
    assert record["recovered_by"]["clock_excess"] is False
    # And the rendering says so in words rather than printing a NaN table.
    assert "not measured" in lag_recovery_check.format_recovery(record)


@pytest.mark.parametrize("config", PLANTED_CONFIGS)
def test_both_planted_configs_resolve_and_pin_the_instruments_geometry(config) -> None:
    r"""The instrument's geometry is pinned and must stay pinned, on both cells.

    Three leaves carry the whole of it. ``max_lag`` is the **production** lag window rather than the
    tiny variant's, because the planted delay has to sit strictly inside $(H, L - 1)$ and the tiny
    window makes that interval empty. ``causal_align_reference_source`` is explicitly null even
    though the shipped default sets it, because the shipped source clock shifts source content
    twenty-five steps earlier and would move the plant's readable band off the near edge -- so a
    later default flip cannot move the instrument. And the target reference is pinned for the same
    reason, one clock rather than two.

    Asserted as a *config* property rather than through a run, because this is what makes every
    recorded gate comparable to the ones before it.
    """
    from teb_vae.lag_attn.config import load_config

    resolved = load_config(str(_REPO_ROOT / config))
    vae = resolved["model_config"]["VAE_model"]

    assert vae["max_lag"] == 90
    assert vae["horizon"] == 30
    assert vae["horizon"] < 45 < vae["max_lag"], "the planted delay is not inside the window"
    assert vae["causal_align_reference"] == "target_max"
    assert vae["causal_align_reference_source"] is None
    assert "planted" in resolved["dataset_config"]["vae_train_datasets"][0]


@pytest.mark.parametrize("config", PLANTED_CONFIGS)
def test_the_manifest_mode_builds_the_model_and_prints_its_switches(config, tmp_path, capsys):
    """The cheap half of the instrument: build, print the header and the state dict, no fit.

    This is the mode a bitwise off-state claim is checked with between gates, so what it has to
    produce is a description of the model that was *built* -- keys and shapes, not a parameter
    total. A total alone cannot say two constructions are the same model: a key renamed, a buffer
    that became persistent, or a tensor that changed shape while another compensated would all
    leave it standing.
    """
    code = lag_recovery_check.main(
        config=config, mode="manifest", output_dir=str(tmp_path)
    )
    printed = capsys.readouterr().out

    assert code == 0
    for key in SWITCH_KEYS:
        assert key in printed, key
    # Both clocks and the offset between them, so a run's alignment arm is on its own page.
    assert "causal_align_reference" in printed
    assert "manifest:" in printed and "state-dict entries" in printed


def test_the_manifest_describes_the_model_that_was_built(tmp_path) -> None:
    """The manifest is compared *between* runs, so it has to be a function of the model rather than
    a restatement of the config: the whole point of a bitwise off-state check is that two configs
    claiming the same architecture produce the same keys and shapes.

    Checked against a model built here rather than against a second manifest call, so a manifest
    that had come to describe something else would fail rather than agree with itself.
    """
    from teb_vae.lag_attn_cfs.tests.conftest import build, tiny_warmup_kwargs

    model = build(tiny_warmup_kwargs())
    record = lag_recovery_check.manifest(model)

    assert record["n_parameters"] == sum(p.numel() for p in model.parameters())
    assert record["n_state_dict_entries"] == len(model.state_dict())
    assert set(record["state_dict"]) == set(model.state_dict())
    for name, shape in record["state_dict"].items():
        assert shape == list(model.state_dict()[name].shape), name


@pytest.mark.slow
def test_the_check_runs_end_to_end_and_reports_the_band_it_measured(tmp_path, capsys):
    r"""The instrument's own path, executed: build, fit, save a loadable checkpoint, read the
    support-corrected lag profile through the evaluation's own code, and report.

    **The exit code is not asserted.** It follows the pass criterion -- the profile's argmax inside
    the planted band and not at lag $0$ -- which is the empirical question the instrument exists to
    ask, and on this family's geometry the pooled argmax is pre-registered as the statistic that
    cannot move. A wrapper that required a pass would be red for a finding rather than for a
    defect; a wrapper that required a failure would go red the day the redesign worked. So what is
    asserted is that the report *was produced*, and carries the three things a gate is read on: the
    band it was measured against, the profile's argmax, and the KL at initialisation.

    The KL line is the one worth having in a suite rather than in an outcome block. Exact zero at
    initialisation is what makes every KL number in this family's records comparable across the
    revision, and three of the shipped mechanisms could break it -- so a run that reported anything
    else here would be a defect rather than a measurement.
    """
    code = lag_recovery_check.main(
        config=PLANTED_CONFIGS[0],
        override=[f"general_config.epochs={WRAPPER_EPOCHS}"],
        output_dir=str(tmp_path),
    )
    printed = capsys.readouterr().out

    assert code in (0, 1), "the check neither completed nor failed its own criterion"
    for key in SWITCH_KEYS:
        assert key in printed, key
    assert "band" in printed
    assert "argmax" in printed
    assert "kld_source_null" in printed
    # Exactly zero, and stated as such: the mechanisms are zero-impact at initialisation by
    # construction, so anything else is a construction defect rather than a training outcome.
    assert "0.000e+00" in printed
