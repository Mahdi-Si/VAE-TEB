r"""S4-T05: warm-start a v3 model from a v1 checkpoint, submodule-by-submodule.

:func:`train.graph_models_utils.load_checkpoint_strict` loads a candidate module only on a
*perfect* key/shape bijection. A v3 model never achieves one against a v1 blob -- it gains
``delta_logvar_head`` and, in residual mode, deletes ``logvar_post_head`` -- so calling it here
would align nothing, log a warning, return ``None``, and leave the model at random
initialisation. Training would then proceed from scratch without anyone noticing. These tests
pin the filtered load that replaces it, and pin that the key delta is exactly the known
v1-vs-v3 architectural difference and nothing else.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.trainer_lag_attn_v3 import warm_start_from_v1
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


def _save_v1_checkpoint(tmp_path, tiny_kwargs, *, trained: bool = False, seed: int = 0):
    """Write a ``{model_class, model_state_dict, model_kwargs}`` blob for a tiny v1 model."""
    torch.manual_seed(seed)
    v1 = SeqVaeLagAttnV1(**tiny_kwargs)
    if trained:
        # Simulate a trained checkpoint: the zero-initialised delta_mu_head has moved.
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for p in v1.posterior_head.delta_mu_head.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
    path = tmp_path / "v1.ckpt"
    torch.save(
        {
            "model_class": "SeqVaeLagAttnV1",
            "model_state_dict": v1.state_dict(),
            "model_kwargs": dict(tiny_kwargs),
        },
        path,
    )
    return path, v1


def test_warm_start_aligns_shared_modules_and_reports_the_delta(
    tmp_path, tiny_kwargs, prod_kwargs
):
    path, v1 = _save_v1_checkpoint(tmp_path, tiny_kwargs)
    v3 = SeqVaeLagAttnV3(**prod_kwargs)

    report = warm_start_from_v1(v3, str(path))

    assert report["loaded"], "warm-start aligned nothing"
    # Exactly the known architectural delta -- nothing else may differ.
    assert all(
        k.startswith("posterior_head.delta_logvar_head.") or k == "lag_attn.lag_score_bias"
        for k in report["missing"]
    ), report["missing"]
    assert all(
        k.startswith("posterior_head.logvar_post_head.") for k in report["unexpected"]
    ), report["unexpected"]
    assert "lag_attn.lag_score_bias" in report["missing"]  # G5's bias is new in v3
    assert report["unexpected"], "v1's independent logvar_post_head should be reported"

    # Every shared tensor really took the v1 value.
    v1_state, v3_state = v1.state_dict(), v3.state_dict()
    for key in (
        "target_encoder.body.lstm.weight_ih_l0",
        "lag_attn.lag_embeddings",
        "posterior_head.delta_mu_head.weight",
    ):
        assert key in v1_state, f"test is stale: {key} no longer exists in v1"
        assert torch.allclose(v3_state[key], v1_state[key]), f"{key} was not warm-started"


def test_causal_group_norm_does_not_break_alignment(tmp_path, tiny_kwargs, prod_kwargs):
    """G0's norm swap must stay parameter-compatible with ``nn.GroupNorm``."""
    path, v1 = _save_v1_checkpoint(tmp_path, tiny_kwargs)
    v3 = SeqVaeLagAttnV3(**prod_kwargs)
    assert v3.causal_norm and v3.n_causalized_norms == 10

    warm_start_from_v1(v3, str(path))

    key = "target_encoder.body.convs.0.pre_norm.weight"
    assert torch.allclose(v3.state_dict()[key], v1.state_dict()[key])


def test_warm_start_from_a_fresh_v1_preserves_zero_kl_init(
    tmp_path, tiny_kwargs, prod_kwargs, inputs
):
    r"""A *fresh* v1 has a zero ``delta_mu_head``, so :math:`q = p` still holds after loading."""
    path, _ = _save_v1_checkpoint(tmp_path, tiny_kwargs)
    v3 = SeqVaeLagAttnV3(**prod_kwargs)
    warm_start_from_v1(v3, str(path))

    v3.eval()
    outs = v3(*inputs)
    assert outs["kld_per_t"].abs().max().item() < 1e-6
    for p in v3.posterior_head.delta_logvar_head.parameters():
        assert torch.count_nonzero(p) == 0, "delta_logvar_head must stay exactly zero"


def test_warm_start_from_a_trained_v1_yields_positive_kl(
    tmp_path, tiny_kwargs, prod_kwargs, inputs
):
    r"""Documented behaviour: a trained ``delta_mu_head`` makes :math:`\mu_q \neq \mu_p`.

    Step-0 :math:`K > 0` after warm-starting from a real checkpoint is expected, not a bug --
    only the *variance* component is reset, by the zero ``delta_logvar_head``.
    """
    path, _ = _save_v1_checkpoint(tmp_path, tiny_kwargs, trained=True)
    v3 = SeqVaeLagAttnV3(**prod_kwargs)
    warm_start_from_v1(v3, str(path))

    v3.eval()
    outs = v3(*inputs)
    assert outs["kld_per_t"].abs().max().item() > 1e-6
    assert torch.allclose(outs["logvar_post"], outs["logvar_prior"], atol=1e-6)


def test_warm_start_rejects_a_non_v1_checkpoint(tmp_path, prod_kwargs):
    path = tmp_path / "v3.ckpt"
    torch.save({"model_class": "SeqVaeLagAttnV3", "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="expects a SeqVaeLagAttnV1"):
        warm_start_from_v1(SeqVaeLagAttnV3(**prod_kwargs), str(path))


def test_warm_start_raises_rather_than_silently_loading_nothing(tmp_path, prod_kwargs):
    """The exact failure mode this function exists to prevent."""
    path = tmp_path / "alien.ckpt"
    torch.save({"model_state_dict": {"totally.unrelated.weight": torch.zeros(3)}}, path)
    with pytest.raises(ValueError, match="aligned ZERO tensors"):
        warm_start_from_v1(SeqVaeLagAttnV3(**prod_kwargs), str(path))


def test_warm_start_flags_an_unexplained_key_difference(tmp_path, tiny_kwargs, prod_kwargs):
    """A missing shared tensor must surface loudly, not be absorbed into ``missing``."""
    torch.manual_seed(0)
    v1 = SeqVaeLagAttnV1(**tiny_kwargs)
    state = v1.state_dict()
    del state["lag_attn.lag_embeddings"]  # an unexplained absence
    path = tmp_path / "v1_broken.ckpt"
    torch.save({"model_class": "SeqVaeLagAttnV1", "model_state_dict": state}, path)

    with pytest.raises(ValueError, match="beyond the known v1-vs-v3 delta"):
        warm_start_from_v1(SeqVaeLagAttnV3(**prod_kwargs), str(path))
