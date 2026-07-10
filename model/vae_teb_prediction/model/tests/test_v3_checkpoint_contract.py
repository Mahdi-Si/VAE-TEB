r"""S4-T06: v3 checkpoints carry ``model_class`` and ``model_kwargs``.

``TestRunner.from_checkpoint`` prefers the version-agnostic path -- rebuild the architecture
straight from the blob's ``model_kwargs``, no config file needed -- and ``check_model_class``
guards against loading a v1 or v2 checkpoint under the v3 alias. A stock Lightning ``.ckpt``
carries neither field, so ``SeqVaeLagAttnV3Pl.on_save_checkpoint`` stamps both.

The wrapper registers the same module twice (``self.model`` and ``self._orig_model`` alias one
object), so the Lightning ``state_dict`` holds every tensor under two prefixes. This is
pre-existing v1 behaviour; ``_clean_state_dict`` strips both and the duplicates collapse onto
identical values. The round-trip below is what proves it.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import check_model_class
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from train.graph_models_utils import load_checkpoint_strict


def _lightning_style_checkpoint(pl_module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": pl_module.state_dict(), "epoch": 3, "global_step": 42}
    pl_module.on_save_checkpoint(checkpoint)
    return checkpoint


def test_checkpoint_carries_the_model_class_and_kwargs(v3_pl, prod_kwargs):
    pl_module = v3_pl()
    checkpoint = _lightning_style_checkpoint(pl_module)

    assert checkpoint["model_class"] == "SeqVaeLagAttnV3"
    assert checkpoint["model_kwargs"] == prod_kwargs
    # The v3 flags must survive, or the rebuilt model would silently differ.
    for flag in ("causal_norm", "posterior_logvar", "logvar_bound", "kld_support",
                 "lag_bias_init", "lambda_perm"):
        assert flag in checkpoint["model_kwargs"], f"{flag} missing from model_kwargs"


def test_model_class_guard_accepts_v3_and_rejects_v1(v3_pl):
    checkpoint = _lightning_style_checkpoint(v3_pl())

    check_model_class(checkpoint, "SeqVaeLagAttnV3")  # must not raise
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(checkpoint, "SeqVaeLagAttnV1")


def test_checkpoint_round_trips_into_a_fresh_model(v3_pl, prod_kwargs, inputs, tmp_path):
    """The version-agnostic load path the testing pipeline uses."""
    pl_module = v3_pl()
    checkpoint = _lightning_style_checkpoint(pl_module)
    path = tmp_path / "v3.ckpt"
    torch.save(checkpoint, path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(blob, "SeqVaeLagAttnV3")
    rebuilt = SeqVaeLagAttnV3(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    # Same weights in, same forward out.
    pl_module.orig_model.eval()
    rebuilt.eval()
    torch.manual_seed(5)
    reference = pl_module.orig_model(*inputs)
    torch.manual_seed(5)
    got = rebuilt(*inputs)
    for key in ("mu_prior", "logvar_post", "mu_full", "logvar_full", "te_lag_map"):
        assert torch.allclose(reference[key], got[key], atol=1e-6), f"drift on {key}"


def test_wrapper_state_dict_holds_both_prefixes(v3_pl):
    """Documents the duplicate-prefix quirk that ``_clean_state_dict`` absorbs."""
    state = v3_pl().state_dict()
    assert any(k.startswith("model.") for k in state)
    assert any(k.startswith("_orig_model.") for k in state)
    a = state["model.lag_attn.lag_embeddings"]
    b = state["_orig_model.lag_attn.lag_embeddings"]
    assert torch.equal(a, b), "the two prefixes must alias one module"


def test_rebuilt_model_keeps_the_causal_encoders(v3_pl):
    """G0 must survive the checkpoint round-trip, or K would silently stop being a TE."""
    checkpoint = _lightning_style_checkpoint(v3_pl())
    rebuilt = SeqVaeLagAttnV3(**checkpoint["model_kwargs"])
    assert rebuilt.causal_norm is True
    assert rebuilt.n_causalized_norms == 10
