r"""Checkpoints are self-describing: they carry ``model_class`` and ``model_kwargs``.

A stock Lightning ``.ckpt`` carries neither. With both, a checkpoint can be rebuilt with no config
file -- which matters because the config that produced a run is a mutable file that may not exist,
or may have moved on, by the time anyone loads the weights -- and ``check_model_class`` can refuse
a blob written by a different model *before* the rebuild is attempted, when the error can still say
what is wrong.

The wrapper registers the same module twice (``self.model`` and ``self._orig_model`` alias one
object), so the Lightning ``state_dict`` holds every tensor under two prefixes. That is inherited
behaviour, the checkpoint loader's prefix stripping absorbs it, and the round-trip below is what
proves it still does.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from train.graph_models_utils import check_model_class, load_checkpoint_strict


def _lightning_style_checkpoint(module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3, "global_step": 42}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def test_the_checkpoint_carries_the_model_class_and_kwargs(task, prod_kwargs):
    checkpoint = _lightning_style_checkpoint(task())

    assert checkpoint["model_class"] == "SeqVaeLagAttn"
    assert checkpoint["model_kwargs"] == prod_kwargs


def test_the_stamp_names_the_eager_class_not_the_wrapper(task):
    """It is the class that would be rebuilt, not the class that saved it.

    Were compilation ever on, this would also have to skip the ``_orig_mod`` wrapper -- which is
    why the base derives it from ``_orig_model`` rather than from ``self``.
    """
    checkpoint = _lightning_style_checkpoint(task())

    assert checkpoint["model_class"] == type(task().orig_model).__name__


def test_the_flags_that_change_the_architecture_survive(task):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then align
    nothing and return ``None`` -- which a caller that does not check reads as success."""
    checkpoint = _lightning_style_checkpoint(task())

    for flag in ("causal_norm", "kld_support", "lag_bias_init", "lambda_perm", "d_model", "d_z"):
        assert flag in checkpoint["model_kwargs"], f"{flag} missing from model_kwargs"


def test_the_base_stamp_survives_the_override(task):
    """``on_save_checkpoint`` here adds a field; it must not replace the base's work.

    An override that skipped ``super()`` would drop ``model_class`` and every guard that reads it
    would silently degrade to its warn-and-continue path.
    """
    checkpoint = _lightning_style_checkpoint(task())

    assert "model_class" in checkpoint
    assert "model_kwargs" in checkpoint
    assert checkpoint["epoch"] == 3  # and it must not have clobbered Lightning's own fields


def test_the_class_guard_accepts_this_model_and_rejects_another(task):
    checkpoint = _lightning_style_checkpoint(task())

    check_model_class(checkpoint, "SeqVaeLagAttn")  # must not raise
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(checkpoint, "SeqVaeRawV4")


def test_a_checkpoint_round_trips_into_a_fresh_model(task, inputs, tmp_path):
    """The whole contract, end to end: save, reload, rebuild from the blob, same forward out."""
    module = task()
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(blob, "SeqVaeLagAttn")
    rebuilt = SeqVaeLagAttn(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    module.orig_model.eval()
    rebuilt.eval()
    # Seeded before each forward: the model samples z, so without this the two differ by noise.
    torch.manual_seed(5)
    reference = module.orig_model(*inputs)
    torch.manual_seed(5)
    got = rebuilt(*inputs)

    for key in ("mu_prior", "logvar_post", "mu_full", "logvar_full", "te_lag_map"):
        assert torch.allclose(reference[key], got[key], atol=1e-6), f"drift on {key}"


def test_the_wrapper_state_dict_holds_both_prefixes(task):
    """Documents the duplicate-prefix quirk the checkpoint loader absorbs."""
    state = task().state_dict()

    assert any(key.startswith("model.") for key in state)
    assert any(key.startswith("_orig_model.") for key in state)
    assert torch.equal(
        state["model.lag_attn.lag_embeddings"], state["_orig_model.lag_attn.lag_embeddings"]
    ), "the two prefixes must alias one module"


def test_a_rebuilt_model_keeps_its_causal_encoders(task):
    """The one flag whose loss would be invisible.

    Every other flag changes a shape and fails loudly. This one leaves the architecture identical
    and only changes whether the prior can see the future -- so a checkpoint that lost it would
    reload, run, and report a KL that is not a transfer entropy.
    """
    checkpoint = _lightning_style_checkpoint(task())

    rebuilt = SeqVaeLagAttn(**checkpoint["model_kwargs"])

    assert rebuilt.causal_norm is True
    assert rebuilt.n_causalized_norms == 10


def test_the_loss_hyperparameters_reach_the_checkpoint(task):
    """So a run's objective is recoverable from its checkpoint, not only from a mutable config."""
    module = task()
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)

    for name in ("likelihood", "sigma_obs", "free_bits", "lambda_full", "beta_schedule"):
        assert name in module.hparams, f"{name} is not in hparams and so will not be checkpointed"
