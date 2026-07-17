"""``configure_param_groups`` seam + group-aware parameter overview.

The default seam returns the flat ``requires_grad`` list, so the optimizer is
numerically identical to the historical behaviour; a subclass that returns param-group
dicts gets a multi-group optimizer and a parameter overview that does not crash on the
dict entries.
"""
import torch

from train.test_utils import TinyLightningModel


def test_default_param_groups_match_trainable_list():
    model = TinyLightningModel()
    groups = model.configure_param_groups()
    assert list(groups) == model._trainable_parameters()


def test_default_optimizer_is_single_group():
    model = TinyLightningModel(lr=1e-3, weight_decay=1e-4)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    # The lone group carries every trainable parameter at the base lr.
    n_in_group = sum(p.numel() for p in optimizer.param_groups[0]["params"])
    n_trainable = sum(p.numel() for p in model._trainable_parameters())
    assert n_in_group == n_trainable
    assert optimizer.param_groups[0]["lr"] == 1e-3


class _TwoGroupModel(TinyLightningModel):
    """Splits the two Linear layers into differentially-LR'd groups."""

    def configure_param_groups(self):
        base = self._orig_model
        return [
            {"params": list(base.fc1.parameters()), "lr": 1e-3},
            {"params": list(base.fc2.parameters()), "lr": 1e-4},
        ]


def test_grouped_return_builds_multigroup_optimizer():
    model = _TwoGroupModel()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[1]["lr"] == 1e-4


def test_log_parameter_overview_handles_group_dicts():
    model = _TwoGroupModel()
    # Must not raise on a list of param-group dicts (numel accounting flattens them).
    model._log_parameter_overview(model.configure_param_groups())
    # And still works on a flat parameter list.
    model._log_parameter_overview(model._trainable_parameters())


class _GeneratorGroupModel(TinyLightningModel):
    """Returns a lazy generator, the most natural override form."""

    def configure_param_groups(self):
        return (p for p in self.parameters() if p.requires_grad)


def test_generator_param_groups_not_double_consumed():
    # The overview and the optimizer both consume the return value; a generator must
    # not be exhausted by the first, or the optimizer ends up with no parameters.
    model = _GeneratorGroupModel()
    optimizer = model.configure_optimizers()
    n_in_optimizer = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    assert n_in_optimizer == sum(p.numel() for p in model._trainable_parameters())
    assert n_in_optimizer > 0
