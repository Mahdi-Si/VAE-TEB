r"""The two shared-code seams this package is built on, exercised before it has any code of its own.

Both are additive hooks on the comparison model's task and driver, and both are worth their own
tests here rather than only in the packages they live in, because the property that matters is the
one only a *second* implementation can demonstrate: that a subclass which overrides the hook
actually changes what runs. A hook with one caller is indistinguishable from a hook nobody reads.

So the subclasses below are deliberately minimal and deliberately not this package's real ones --
they wrap the **comparison model**, whose forward signature is known and whose readouts are already
pinned. What they prove is that the seam carries an override, not that any particular architecture
works.

``_build_forward_inputs`` is checked against the one readout that can distinguish "the net was fed
something different" from "the numbers moved": the prior branch never sees the source, so a
deranged source stream must leave ``nll_base_block`` bitwise identical while ``nll_full_block``
moves. ``preflight`` is checked by refusing a launch, which is the only observable a guard has.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_rws.tests.conftest import TINY_KWARGS as SIBLING_TINY_KWARGS
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn.config import load_config

from .conftest import BATCH, SEQ_LEN, absolutize_dataset_paths, make_stub_batch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIBLING_TINY = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"

#: Loss hyperparameters as the task's constructor takes them. ``beta_schedule=None`` means the
#: constant ``kld_beta`` applies, which keeps the schedule out of the way of a test about inputs.
_TASK_HPARAMS = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


class _DerangedSourceTask(SeqVaeLagAttnRwsTask):
    """The comparison model's task with one method overridden, and nothing else.

    The override reverses the source stream along the batch axis, which pairs each sample's target
    history with a stranger's source. That is a change no other seam could produce: the loss, the
    metric assembly, the permutation control and the checkpoint contract are all inherited
    untouched, so any difference in the returned metrics came through the hook.
    """

    def _build_forward_inputs(self, batch):
        """Return the inherited tuple with its source stream deranged.

        Args:
            batch: A batch from the data module.

        Returns:
            ``(y_st, y_ph, u_stream.flip(0))``.
        """
        y_st, y_ph, u_stream = super()._build_forward_inputs(batch)
        return y_st, y_ph, u_stream.flip(0)


def _build_task(task_cls, *, seed: int = 0):
    """Build the comparison model at the tiny geometry, wrapped in ``task_cls``.

    Seeded and rebuilt per call rather than shared, so two tasks of different classes hold weights
    that are equal element for element -- which is what makes a metric comparison between them a
    statement about their inputs.

    Args:
        task_cls: The task class to wrap the net in.
        seed: Seed applied immediately before construction.

    Returns:
        The task, with ``setup()`` already called.
    """
    kwargs: dict = dict(SIBLING_TINY_KWARGS)
    torch.manual_seed(seed)
    model = SeqVaeLagAttnRws(**kwargs)
    task = task_cls(model, lr=1e-3, model_kwargs=kwargs, **_TASK_HPARAMS)
    task.setup("fit")
    return task


# ---------------------------------------------------------------------------------------
# The forward-input seam
# ---------------------------------------------------------------------------------------
def test_the_default_hook_feeds_the_net_what_it_was_always_fed(perturb_posterior):
    """The seam is additive, so the unoverridden path must be the old one exactly. Asserted on the
    builders' own output rather than on a remembered number, which would go stale."""
    task = _build_task(SeqVaeLagAttnRwsTask)
    perturb_posterior(task.orig_model)
    batch = make_stub_batch(BATCH, SEQ_LEN)

    inputs = task._build_forward_inputs(batch)

    y_st, y_ph = task._build_target_streams(batch)
    assert torch.equal(inputs[0], y_st)
    assert torch.equal(inputs[1], y_ph)
    assert torch.equal(inputs[2], task._build_source_stream(batch))


def test_a_subclass_of_the_hook_decides_what_the_net_receives(perturb_posterior):
    """The seam doing its job, on the comparison model, before this package has any code.

    The seed is re-set between the two calls because one ``randn_like`` draw enters both branches
    of the objective; without it ``nll_base_block`` would move for reasons that have nothing to do
    with the hook, and the bitwise half of the assertion could not be made at all.
    """
    batch = make_stub_batch(BATCH, SEQ_LEN)
    default = _build_task(SeqVaeLagAttnRwsTask)
    deranged = _build_task(_DerangedSourceTask)
    for task in (default, deranged):
        perturb_posterior(task.orig_model)

    torch.manual_seed(7)
    _loss, reference = default.compute_loss_and_metrics(batch, 0, "train")
    torch.manual_seed(7)
    _loss, overridden = deranged.compute_loss_and_metrics(batch, 0, "train")

    # The prior branch never sees the source, so it must be untouched...
    assert torch.equal(overridden["nll_base_block"], reference["nll_base_block"])
    # ...and the posterior branch, which does, must have moved.
    assert not torch.equal(overridden["nll_full_block"], reference["nll_full_block"])


def test_the_override_costs_exactly_one_method():
    """What makes the seam worth having: an architecture over a different input representation
    reuses the objective, the metric surface, the permutation control and the checkpoint contract
    without taking any of them back."""
    # Dunders and ``_abc_impl`` are Python's, not the subclass's: ``LightningModule`` is an ABC, so
    # every subclass gets an abstract-method cache whether it defines anything or not.
    defined = {
        name
        for name in vars(_DerangedSourceTask)
        if not name.startswith("__") and name != "_abc_impl"
    }

    assert defined == {"_build_forward_inputs"}


# ---------------------------------------------------------------------------------------
# The pre-flight seam
# ---------------------------------------------------------------------------------------
class _StubDataModule:
    """A data module that hands out loaders nothing iterates."""

    def train_dataloader(self):
        """Return a placeholder the stubbed ``train_model`` never touches."""
        return object()

    def val_dataloader(self):
        """Return a placeholder the stubbed ``train_model`` never touches."""
        return object()


class _RefusingTrainer(LagAttnRwsTrainer):
    """A driver whose ``preflight`` refuses every launch."""

    #: Named in the raised message, so the test asserts on content rather than on a bare type.
    REFUSAL = "this driver refuses every launch"

    @classmethod
    def preflight(cls, config) -> None:
        """Refuse.

        Args:
            config: The resolved config, unused.

        Raises:
            ValueError: Always.
        """
        raise ValueError(cls.REFUSAL)


def _sibling_tiny_config_at(tmp_path: Path) -> str:
    """Write a resolved, path-absolutised copy of the comparison model's tiny config.

    Absolute paths because the four inherited guards read the statistics file and the first shard
    off disk, and pytest's working directory is not something this test may assume.

    Args:
        tmp_path: Directory to write into.

    Returns:
        The written path.
    """
    config = absolutize_dataset_paths(load_config(str(_SIBLING_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def test_a_drivers_own_preflight_is_what_runs_and_it_stops_the_launch(tmp_path, monkeypatch):
    """A guard added by a driver must fail the launch *before* the run directory, the log sinks
    and the MLflow run exist -- which is the entire reason the hook is called where it is rather
    than inside ``create_model``."""
    reached = []
    monkeypatch.setattr(
        LagAttnRwsTrainer, "setup_config", lambda self: reached.append("setup_config")
    )

    with pytest.raises(ValueError, match=_RefusingTrainer.REFUSAL):
        shared_trainer.main(_sibling_tiny_config_at(tmp_path), trainer_cls=_RefusingTrainer)

    assert reached == []


def test_the_same_config_launches_when_the_hook_does_not_refuse(tmp_path, monkeypatch):
    """The paired half. Without it the test above would pass against a config that was doomed for
    some other reason, and the hook would be shown to do nothing at all."""
    reached = []

    def _stub_setup(self) -> None:
        # The real one seeds, opens log sinks and probes MLflow; all this launch needs is the
        # attribute the resolved-config write reads immediately afterwards.
        reached.append("setup_config")
        self.model_checkpoint_dir = str(tmp_path / "checkpoints")

    monkeypatch.setattr(LagAttnRwsTrainer, "setup_config", _stub_setup)
    monkeypatch.setattr(LagAttnRwsTrainer, "create_model", lambda self: None)
    monkeypatch.setattr(LagAttnRwsTrainer, "train_model", lambda self, *args: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: _StubDataModule())
    monkeypatch.setattr(shared_trainer, "_persist_resolved_config", lambda *args: None)

    shared_trainer.main(_sibling_tiny_config_at(tmp_path), trainer_cls=LagAttnRwsTrainer)

    assert reached == ["setup_config"]
