"""The metrics-history CSV writer, and the naming rule that decides whether a column has data.

``MetricsLoggingCallback`` reads ``trainer.callback_metrics`` by exact name. What lands in that
dict is decided by ``LightningModelBase._log_metrics``, which prefixes every key that does not
already contain a ``/``. So a tracked name is either prefixed, or it is one of the few keys the
framework logs bare, or it silently never matches -- producing a column that is NaN for every
epoch of every run, with no error anywhere.

That is not a hypothetical: ``kld_beta`` was tracked bare in the consumer this writer came from,
and its column was NaN in every ``metrics_history.csv`` ever produced. The first three tests pin
the rule itself, in the framework, so the fix cannot regress into the same shape twice.
"""
import pandas as pd
import pytest
import torch
from loguru import logger

from train.callbacks import (
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)
from train.test_utils import FakeTrainer, TinyLightningModel


@pytest.fixture
def loguru_messages():
    """Collect loguru output.

    ``caplog`` cannot see it: loguru does not route through the stdlib ``logging`` module, so a
    ``caplog.at_level`` assertion against a loguru warning passes vacuously.
    """
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


class _RecordingModule(TinyLightningModel):
    """A module whose ``log`` records rather than routing to an absent Trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.logged = {}

    def log(self, name, value, **kwargs):  # type: ignore[override]
        self.logged[name] = value


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------
def test_an_unprefixed_metric_name_is_emitted_stage_prefixed():
    """Why a bare tracked name never matches: the framework renamed it on the way out."""
    module = _RecordingModule()

    module._log_metrics({"kld_beta": torch.tensor(0.5)}, stage="train", on_step=False)

    assert "train/kld_beta" in module.logged
    assert "kld_beta" not in module.logged


def test_a_name_containing_a_slash_is_emitted_unchanged():
    """The escape hatch, and the reason a prefixed tracked name works."""
    module = _RecordingModule()

    module._log_metrics({"val/total_loss": torch.tensor(1.0)}, stage="train", on_step=False)

    assert "val/total_loss" in module.logged


def test_lr_is_the_bare_key_the_framework_actually_emits():
    """``lr`` may legitimately be tracked bare -- and ``learning_rate`` may not.

    ``_log_learning_rate`` logs the literal key ``lr``, bypassing ``_log_metrics`` entirely, so
    it is the one name that is correct without a stage prefix.
    """
    module = _RecordingModule()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.007)
    module.optimizers = lambda: optimizer  # type: ignore[assignment]

    module._log_learning_rate()

    assert module.logged["lr"] == 0.007


def test_unreachable_names_are_exactly_the_bare_non_framework_ones():
    names = ("train/total_loss", "val/kld_raw", "lr", "kld_beta", "learning_rate")

    assert _unreachable_metric_names(names) == ("kld_beta", "learning_rate")


# --------------------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------------------
def test_construction_warns_about_names_that_can_never_carry_data(tmp_path, loguru_messages):
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss", "kld_beta"))

    MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    assert any("kld_beta" in message for message in loguru_messages)


def test_construction_is_quiet_for_a_correctly_named_list(tmp_path, loguru_messages):
    """The mirror: the warning above must be capable of not firing."""
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss", "val/total_loss", "lr"))

    MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    assert loguru_messages == []


def test_the_csv_carries_every_tracked_metric_plus_an_epoch_column(tmp_path):
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss", "val/total_loss"))
    source.history["train/total_loss"] = [3.0, 2.0]
    source.history["val/total_loss"] = [3.5, 2.5]
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    callback.on_fit_end(FakeTrainer(), None)

    frame = pd.read_csv(tmp_path / "metrics_history.csv")
    assert list(frame.columns) == ["epoch", "train/total_loss", "val/total_loss"]
    assert list(frame["epoch"]) == [0, 1]
    assert list(frame["train/total_loss"]) == [3.0, 2.0]


def test_the_epoch_column_is_the_real_epoch_not_the_row_index(tmp_path):
    """The collector has no sanity-check guard; this writer does.

    With ``num_sanity_val_steps > 0`` the source's first history row is the sanity pass -- a row
    this writer skipped. Numbering rows positionally would then label the sanity row 'epoch 0' and
    shift every real epoch by one against MLflow, the checkpoint filenames and the loss plots.
    """
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    source.history["train/total_loss"] = [9.9, 3.0, 2.0]  # sanity row, then epochs 0 and 1
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    callback.on_validation_epoch_end(FakeTrainer(current_epoch=1), None)

    frame = pd.read_csv(tmp_path / "metrics_history.csv")
    assert list(frame["epoch"]) == [-1, 0, 1]  # the sanity row is visibly not an epoch


def test_the_epoch_column_counts_from_zero_on_an_ordinary_run(tmp_path):
    """The common case: no sanity pass, so the rows are the epochs."""
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    source.history["train/total_loss"] = [3.0, 2.0]
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    callback.on_validation_epoch_end(FakeTrainer(current_epoch=1), None)

    frame = pd.read_csv(tmp_path / "metrics_history.csv")
    assert list(frame["epoch"]) == [0, 1]


def test_the_csv_is_rewritten_each_validation_epoch(tmp_path):
    """So a killed run leaves the history it reached rather than nothing."""
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))
    trainer = FakeTrainer()

    source.history["train/total_loss"].append(3.0)
    callback.on_validation_epoch_end(trainer, None)
    assert len(pd.read_csv(tmp_path / "metrics_history.csv")) == 1

    source.history["train/total_loss"].append(2.0)
    callback.on_validation_epoch_end(trainer, None)
    assert len(pd.read_csv(tmp_path / "metrics_history.csv")) == 2


def test_nothing_is_written_off_rank_zero(tmp_path):
    """Every rank accumulates history; only rank 0 owns the file."""
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    source.history["train/total_loss"] = [1.0]
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    callback.on_fit_end(FakeTrainer(is_global_zero=False), None)

    assert not (tmp_path / "metrics_history.csv").exists()


def test_the_sanity_check_epoch_does_not_write(tmp_path):
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    source.history["train/total_loss"] = [1.0]
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    callback.on_validation_epoch_end(FakeTrainer(sanity_checking=True), None)

    assert not (tmp_path / "metrics_history.csv").exists()


def test_an_empty_history_writes_no_file(tmp_path):
    """``pd.DataFrame({})`` would otherwise write a headerless, rowless file.

    Note the history cannot be emptied via the constructor: ``tracked_metrics or (defaults)``
    means an empty tuple is falsy and silently yields the default list instead. That quirk is
    not this callback's to fix, so the empty case is reached the only way it can occur -- a
    source that collected nothing.
    """
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    source.history = {}
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(tmp_path))

    assert callback._write() is None
    assert not (tmp_path / "metrics_history.csv").exists()


def test_a_missing_output_directory_is_created(tmp_path):
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss",))
    source.history["train/total_loss"] = [1.0]
    nested = tmp_path / "does" / "not" / "exist"
    callback = MetricsHistoryCsvCallback(source=source, output_dir=str(nested))

    callback.on_fit_end(FakeTrainer(), None)

    assert (nested / "metrics_history.csv").is_file()


def test_a_tracked_metric_absent_from_callback_metrics_records_nan(tmp_path):
    """The failure mode itself, made visible: the column exists and is NaN.

    This is what a bare ``kld_beta`` produced for every run. The writer cannot detect it after
    the fact -- an absent metric and a genuinely-NaN metric look identical here -- which is why
    the warning fires at construction instead.
    """
    source = MetricsLoggingCallback(tracked_metrics=("train/total_loss", "kld_beta"))
    trainer = FakeTrainer(
        callback_metrics={
            "train/total_loss": torch.tensor(2.0),
            # What the framework actually emitted for the metric the consumer called `kld_beta`.
            "train/kld_beta": torch.tensor(0.5),
        }
    )

    source.on_validation_epoch_end(trainer, None)

    assert source.history["train/total_loss"] == [2.0]
    assert len(source.history["kld_beta"]) == 1
    assert source.history["kld_beta"][0] != source.history["kld_beta"][0]  # NaN
