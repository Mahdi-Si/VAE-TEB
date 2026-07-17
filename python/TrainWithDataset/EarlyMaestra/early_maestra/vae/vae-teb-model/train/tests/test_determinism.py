"""``configure_determinism`` sets backend flags and seeds reproducibly."""
import importlib
import os

import torch

from train.test_utils import make_graph_model


def test_deterministic_true_flips_all_flags(config_path):
    # Pre-set the "fast" flags, then request a deterministic run.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    gm = make_graph_model(config_path, **{"advanced_config.trainer.deterministic": True})
    gm.configure_determinism()

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_deterministic_false_is_fast_path(config_path):
    gm = make_graph_model(config_path)  # shipped config: deterministic=false
    gm.configure_determinism()

    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "high"


def test_seed_workers_and_reproducible(config_path):
    gm = make_graph_model(config_path)
    gm.configure_determinism()
    assert os.environ.get("PL_SEED_WORKERS") == "1"

    first = torch.rand(4)
    gm.configure_determinism()  # re-seed with the same seed
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_import_does_not_mutate_backend_flags():
    # The four backend flags moved out of import scope into configure_determinism;
    # reloading the module must not reset a pre-set sentinel.
    torch.backends.cudnn.benchmark = False
    import train.graph_model_base as gmb

    importlib.reload(gmb)
    assert torch.backends.cudnn.benchmark is False
