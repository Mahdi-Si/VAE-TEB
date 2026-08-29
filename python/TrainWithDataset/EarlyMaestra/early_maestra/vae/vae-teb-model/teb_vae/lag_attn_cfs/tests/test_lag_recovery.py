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
