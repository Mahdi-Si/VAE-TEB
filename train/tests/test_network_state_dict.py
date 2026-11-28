"""``network_state_dict`` reduces every Lightning wrapper layout to the bare network keys."""
from collections import OrderedDict

import pytest
import torch

from train.graph_models_utils import network_state_dict
from train.test_utils import TinyModule


@pytest.mark.parametrize("prefix", ["model.", "model._orig_mod."], ids=["eager", "compiled"])
def test_a_lightning_checkpoint_reduces_to_the_bare_network(prefix):
    net = TinyModule()
    bare = net.state_dict()
    # Lightning stores the network under `model.` (compiled: `model._orig_mod.`) and again
    # under `_orig_model.`; both copies must collapse to one set of bare keys.
    state = OrderedDict({f"{prefix}{k}": v for k, v in bare.items()})
    state.update({f"_orig_model.{k}": v for k, v in bare.items()})

    out = network_state_dict({"state_dict": state, "epoch": 3})

    assert list(out) == list(bare)
    assert all(torch.equal(out[k], bare[k]) for k in bare)
    TinyModule().load_state_dict(out, strict=True)


def test_a_checkpoint_without_weights_is_refused():
    with pytest.raises(ValueError):
        network_state_dict({"epoch": 3})
