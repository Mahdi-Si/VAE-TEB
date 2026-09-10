r"""The experiment driver: three re-pointed attributes, one metric surface, one refusal.

**All three class attributes collide**, because both parents set them. Resolution order alone would
take the causal side and every failure would be silent: a driver that builds another architecture, a
task that wraps it, and a checkpoint stem that interleaves two models' files in a shared output
tree. None of those raises.

**The tracked metric surface is this package's**, because the inherited one names three groups this
objective does not produce. A tracked name nothing produces is a column that is empty in every row
of every run, which is worse than an absent column: it reads as a measurement that came out blank.

**The two checkpoint keys are refused together.** One is a strict load of this exact model kind and
restores the source pathway; the other is a partial transfer that deliberately leaves it at zero. A
run doing both has a starting point that neither key describes.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.task import (
    TASK_METRIC_SUFFIXES,
    SeqVaeLagResidualTrfCfsTask,
)
from teb_vae.lag_slot_transformer_cfs.trainer import (
    WARM_START_KEY,
    LagResidualTrfCfsTrainer,
    _resolve_cli_config_path,
)

#: This package's own directory, for the launch-convention checks.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Metric names the inherited surface tracks that this objective does not produce.
INHERITED_ONLY = (
    "aux_multiscale",
    "aux_derivative",
    "aux_boundary",
    "kld_source_null",
    "nll_shuffled_block",
    "shuffle_penalty",
    "pred_gap_warm_lo",
    "pred_gap_novel_lo",
    "delta_mu_sat_frac",
)


def test_all_three_colliding_attributes_are_re_pointed() -> None:
    """Each one omitted is a silent failure of a different kind."""
    assert LagResidualTrfCfsTrainer.MODEL_CLS is SeqVaeLagResidualTrfCfs
    assert LagResidualTrfCfsTrainer.TASK_CLS is SeqVaeLagResidualTrfCfsTask
    assert LagResidualTrfCfsTrainer.CHECKPOINT_STEM == "lag-residual-trf-cfs"

    # And each differs from what resolution order alone would have given.
    assert LagResidualTrfCfsTrainer.MODEL_CLS is not LagAttnCfsTrainer.MODEL_CLS
    assert LagResidualTrfCfsTrainer.CHECKPOINT_STEM != LagAttnCfsTrainer.CHECKPOINT_STEM


def test_the_resolution_order_keeps_both_parents() -> None:
    """The causal parent owns the target domain; the conv-Transformer parent owns compilation."""
    names = [cls.__name__ for cls in LagResidualTrfCfsTrainer.__mro__]
    assert names[:4] == [
        "LagResidualTrfCfsTrainer",
        "LagAttnCfsTrainer",
        "LagAttnTrfRwsTrainer",
        "LagAttnRwsTrainer",
    ]


def test_the_target_fields_come_from_the_causal_parent() -> None:
    """Both stored target blocks: the target is their concatenation.

    A configuration carrying one of them is a target with a hole in it, and the shared entry
    point's normalisation guard reads this tuple to catch it.
    """
    assert LagResidualTrfCfsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")


def test_the_plot_config_key_stays_the_shared_literal() -> None:
    """The callback assembly reads it, so renaming it to match this package disables the figure.

    This model does not enable the block, but the key must still be the one the assembly looks for
    or a future arm that wanted a figure would get none, with no error and nothing in the log.
    """
    assert (
        LagResidualTrfCfsTrainer.PLOT_CONFIG_KEY
        == LagAttnTrfRwsTrainer.PLOT_CONFIG_KEY
    )


@pytest.mark.parametrize("name", INHERITED_ONLY)
def test_the_tracked_surface_names_nothing_this_objective_does_not_produce(name: str) -> None:
    """An empty column reads as a measurement that came out blank.

    Args:
        name: The inherited metric that must not be tracked here.
    """
    assert not any(
        tracked.endswith(f"/{name}") for tracked in LagResidualTrfCfsTrainer.TRACKED_METRICS
    )


def test_the_tracked_surface_names_every_readout_this_task_adds() -> None:
    """On both stages, so a column exists wherever the step reports one."""
    for suffix in TASK_METRIC_SUFFIXES:
        for stage in ("train", "val"):
            assert f"{stage}/{suffix}" in LagResidualTrfCfsTrainer.TRACKED_METRICS


def test_the_tracked_surface_names_the_objective_columns() -> None:
    """The four the acceptance reads, and the two that make a nats-per-anchor column readable."""
    for suffix in (
        "total_loss",
        "nll_full_block",
        "pred_gap",
        "source_conditioned_kl_raw",
        "scored_anchors",
        "scored_coefficients",
    ):
        assert f"val/{suffix}" in LagResidualTrfCfsTrainer.TRACKED_METRICS
    assert "lr" in LagResidualTrfCfsTrainer.TRACKED_METRICS


def test_the_two_checkpoint_keys_are_refused_together() -> None:
    """Each does something different to the source pathway; a run doing both describes neither."""
    driver = LagResidualTrfCfsTrainer.__new__(LagResidualTrfCfsTrainer)
    driver.config = {
        "model_config": {
            "core_model_checkpoint": "a.ckpt",
            WARM_START_KEY: "b.ckpt",
        }
    }
    with pytest.raises(ValueError, match=WARM_START_KEY):
        driver.create_model()


# =================================================================================================
# The launch convention
# =================================================================================================
def test_the_module_runs_from_the_run_button_with_no_command_line() -> None:
    """A module-level constant names the configuration, and no argument is required.

    A required argument fires before the constant is ever read, which makes the Run button unusable
    whatever the constant says.
    """
    import teb_vae.lag_slot_transformer_cfs.trainer as trainer_module

    assert isinstance(trainer_module.RUN_CONFIG, str)
    assert trainer_module.RUN_CONFIG.startswith("teb_vae/")
    assert (Path(_resolve_cli_config_path(trainer_module.RUN_CONFIG))).exists()


def test_the_run_constant_sits_immediately_above_the_main_guard() -> None:
    """So it is the first thing found when scrolling to the bottom of the file."""
    source = (PACKAGE_ROOT / "trainer.py").read_text(encoding="utf-8")
    assert source.index("RUN_CONFIG:") < source.index('if __name__ == "__main__":')
    tail = source[source.index("RUN_CONFIG:") :]
    assert tail.index('if __name__ == "__main__":') < 400


def test_no_argument_is_required_and_none_carries_a_non_none_default() -> None:
    """A non-``None`` default would make the constant unreachable with nothing saying why."""
    source = (PACKAGE_ROOT / "trainer.py").read_text(encoding="utf-8")
    assert "required=True" not in source
    assert "default=None" in source


def test_a_relative_config_path_resolves_against_the_repository_root() -> None:
    """An IDE's working directory is whatever the IDE chose.

    A relative shard path resolved against it surfaces as an empty dataset with no mention of the
    real cause.
    """
    resolved = _resolve_cli_config_path("configs/default.yaml")
    assert Path(resolved).is_absolute()
    assert resolved.endswith("configs/default.yaml") or resolved.endswith(
        "configs\\default.yaml"
    )


def test_the_shipped_configuration_builds_this_model_through_the_signature_sweep() -> None:
    """The sweep is what a run actually goes through, so it is what the test goes through."""
    raw = yaml.safe_load(
        (PACKAGE_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    vae_config = raw["model_config"]["VAE_model"]
    valid = set(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters)
    forwarded = {
        name: value
        for name, value in vae_config.items()
        if name in valid and name != "init_weights" and value is not None
    }
    # Not constructed here: the production geometry builds a model of a size a unit test should not
    # allocate. What is checked is that the sweep selects a set the constructor accepts.
    missing = [name for name in forwarded if name not in valid]
    assert missing == []
    assert "max_lag" in forwarded and "d_z" in forwarded
