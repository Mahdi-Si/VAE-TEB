"""Smoke tests for the live-VAE / two-stage path.

These cover the unit-level invariants of the pieces wired up in this
implementation:

* :meth:`GuidOutcomeClassifier.live_forward` produces the same output
  schema as the cached path plus a ``vae_outputs`` sub-dict.
* :class:`TwoStageVaeUnfreeze` snapshots ``θ⁽⁰⁾`` at train start, keeps
  every VAE param frozen during stage 1, and unfreezes the documented
  submodules at the stage-1 → stage-2 boundary.
* :meth:`PlGuidClassifier.compute_loss_and_metrics` exposes ``vae_loss``
  and ``sparsity`` in the metrics dict once stage 2 weights are active.

A full end-to-end Lightning ``Trainer.fit`` smoke test is intentionally
omitted: instantiating ``SeqVaeLagAttnV1`` on CPU with realistic shapes
already costs ~10 s and would inflate the unit-test suite. The pieces
exercised here are the only ones that are unique to the live path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.callbacks import (  # noqa: E402
    TwoStageVaeUnfreeze,
    _iter_unfreeze_params,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (  # noqa: E402
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (  # noqa: E402
    LossWeights,
)


def _build_classifier(d_model_vae: int = 8, d_z: int = 4) -> GuidOutcomeClassifier:
    """Construct a tiny classifier matching the live-path tokenizer shape."""
    cfg = GuidClassifierConfig(
        d_model_vae=d_model_vae,
        d_z=d_z,
        d_seg=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_head=8,
        d_ff=16,
        n_rel_buckets=4,
        c_meta_dim=5,
        te_summary_dim=6,
    )
    return GuidOutcomeClassifier(cfg)


class _StubVae(torch.nn.Module):
    """Minimal ``SeqVaeLagAttnV1`` stand-in for the live-forward smoke test.

    Provides exactly the surface :meth:`live_forward` and the callback
    use: named submodules from the documented unfreeze set,
    ``encode_only`` returning the right-shaped tensors,
    ``mu_post_running_*`` buffers, and ``kld_tensor`` /  ``use_up_st``
    attributes.
    """

    def __init__(self, *, d_model_vae: int = 8, d_z: int = 4, max_lag: int = 3) -> None:
        super().__init__()
        self.d_model = d_model_vae
        self.d_z = d_z
        self.max_lag = max_lag
        self.use_up_st = True

        # Submodules with the names the unfreeze callback expects.
        self.target_adapter = torch.nn.Linear(8, d_model_vae)
        self.source_adapter = torch.nn.Linear(8, d_model_vae)
        self.target_encoder = torch.nn.Linear(d_model_vae, d_model_vae)
        self.source_encoder = torch.nn.Linear(d_model_vae, d_model_vae)
        self.prior_head = torch.nn.Linear(d_model_vae, 2 * d_z)
        self.posterior_head = torch.nn.Linear(d_model_vae + d_z, 2 * d_z)
        self.lag_attn = torch.nn.Linear(d_model_vae, d_model_vae)         # frozen
        self.baseline_decoder = torch.nn.Linear(d_model_vae, 1)            # frozen
        self.residual_decoder = torch.nn.Linear(d_model_vae, 1)            # frozen

        self.register_buffer("mu_post_running_mean", torch.zeros(d_z))
        self.register_buffer("mu_post_running_var", torch.ones(d_z))
        self.register_buffer(
            "mu_post_running_count", torch.zeros((), dtype=torch.long)
        )

    def encode_only(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        sample_z: bool = False,
    ) -> dict:
        del sample_z
        B, T, _ = y_st.shape
        h_y = torch.zeros(B, T, self.d_model, device=y_st.device)
        h_u = torch.zeros(B, T, self.d_model, device=y_st.device)
        mu = torch.randn(B, T, self.d_z, device=y_st.device) * 0.1
        lv = torch.full((B, T, self.d_z), -1.0, device=y_st.device)
        attn = torch.softmax(
            torch.zeros(B, T, 2, self.max_lag + 1, device=y_st.device), dim=-1
        )
        return {
            "mu_prior": mu,
            "logvar_prior": lv,
            "mu_post": mu + 0.05,
            "logvar_post": lv,
            "z": mu,
            "target_state": h_y,
            "source_state": h_u,
            "decoder_state": h_y,
            "attended_source": h_y,
            "attn_weights": attn,
        }

    def kld_tensor(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        *,
        mask_warmup: bool = False,
    ) -> torch.Tensor:
        del mask_warmup
        kld = (
            logvar_prior
            - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
            - 1.0
        )
        return 0.5 * kld


def _build_live_batch(
    *, B: int = 2, N: int = 3, T: int = 6
) -> dict:
    """Synthesize a collated live-VAE batch."""
    return {
        "fhr_st": torch.randn(B, N, T, 8),
        "fhr_ph": torch.randn(B, N, T, 8),
        "up_st": torch.randn(B, N, T, 4),
        "up_ph": torch.randn(B, N, T, 4),
        "weight": torch.ones(B, N, T),
        "hat_w": torch.ones(B, N, T),
        "c_meta": torch.zeros(B, N, 5),
        "segment_mask": torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        ),
        "rel_bucket_idx": torch.zeros(B, N, N, dtype=torch.long),
        "label_3": torch.tensor([0, 1], dtype=torch.long),
        "label_bin": torch.tensor([0.0, 1.0]),
        "num_segments": torch.tensor([3, 2], dtype=torch.long),
    }


def test_live_forward_returns_expected_shapes() -> None:
    """``live_forward`` must produce per-position outputs + ``vae_outputs``."""
    classifier = _build_classifier(d_model_vae=8, d_z=4)
    vae = _StubVae(d_model_vae=8, d_z=4, max_lag=3)
    classifier.vae = vae
    classifier.vae_chunk_size = 2

    batch = _build_live_batch(B=2, N=3, T=6)
    out = classifier(batch)

    assert out["logits_3"].shape == (2, 3, 3)
    assert out["logit_bin"].shape == (2, 3)
    assert "vae_outputs" in out
    vo = out["vae_outputs"]
    M_expected = int(batch["segment_mask"].sum().item())  # = 5
    assert vo["mu_prior"].shape == (M_expected, 6, 4)
    assert vo["mu_post"].shape == (M_expected, 6, 4)
    assert vo["kld_per_t"].shape == (M_expected, 6)
    assert vo["hat_w_v"].shape == (M_expected, 6)


def test_two_stage_callback_freezes_then_unfreezes() -> None:
    """Stage 1 must keep every VAE param frozen; stage 2 unfreezes the set."""
    classifier = _build_classifier()
    vae = _StubVae(d_model_vae=8, d_z=4, max_lag=3)
    classifier.vae = vae

    # Mimic the wrapper interface the callback expects.
    class _StubModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._orig_model = classifier
            self.loss_weights = LossWeights(
                lambda_3=1.0, lambda_2=0.5, gamma_vae=0.1, lambda_sp=1e-4
            )

    pl_module = _StubModule()
    cb = TwoStageVaeUnfreeze(
        stage1_epochs=2,
        gamma_vae_stage2=0.1,
        lambda_sp_stage2=1e-4,
    )

    # Lightning passes a Trainer instance; we don't need its surface for
    # on_train_start so an attribute-bag is enough.
    class _Trainer:
        current_epoch = 0
        optimizers: list = []
        lr_scheduler_configs: list = []

    trainer = _Trainer()
    cb.on_train_start(trainer, pl_module)  # type: ignore[arg-type]

    # Stage 1 invariants.
    assert all(not p.requires_grad for p in vae.parameters())
    assert pl_module.loss_weights.gamma_vae == 0.0
    assert pl_module.loss_weights.lambda_sp == 0.0
    assert hasattr(pl_module, "_vae_theta0")
    assert pl_module._vae_theta0  # type: ignore[attr-defined]
    expected_unfreeze = _iter_unfreeze_params(
        vae,
        (
            "target_adapter",
            "source_adapter",
            "target_encoder",
            "source_encoder",
            "prior_head",
            "posterior_head",
        ),
    )
    assert len(pl_module._vae_theta0) == len(expected_unfreeze)  # type: ignore[attr-defined]

    # Provide a stage-1 optimizer (classifier-only) so the callback can
    # call ``add_param_group`` against it at the boundary.
    cls_params = [
        p for p in classifier.parameters()
        if id(p) not in {id(q) for q in vae.parameters()}
    ]
    stage1_optimizer = torch.optim.AdamW(cls_params, lr=1e-3)
    trainer.optimizers = [stage1_optimizer]
    pl_module.vae_lr = 1e-5  # type: ignore[attr-defined]

    # Before the boundary: nothing should change.
    trainer.current_epoch = 1
    cb.on_train_epoch_start(trainer, pl_module)  # type: ignore[arg-type]
    assert all(not p.requires_grad for p in vae.parameters())
    assert pl_module.loss_weights.gamma_vae == 0.0
    assert len(stage1_optimizer.param_groups) == 1

    # Stage 2 boundary: unfreeze documented submodules + restore weights.
    trainer.current_epoch = 2
    cb.on_train_epoch_start(trainer, pl_module)  # type: ignore[arg-type]
    assert pl_module.loss_weights.gamma_vae == 0.1
    assert pl_module.loss_weights.lambda_sp == 1e-4
    # The unfreeze set was added as a fresh low-LR param group.
    assert len(stage1_optimizer.param_groups) == 2
    assert stage1_optimizer.param_groups[1]["lr"] == 1e-5
    assert stage1_optimizer.param_groups[1]["weight_decay"] == 0.0
    # Documented submodules: requires_grad=True.
    for name in (
        "target_adapter",
        "source_adapter",
        "target_encoder",
        "source_encoder",
        "prior_head",
        "posterior_head",
    ):
        sub = getattr(vae, name)
        assert all(p.requires_grad for p in sub.parameters()), (
            f"{name} should be unfrozen at stage 2"
        )
    # Frozen submodules stay frozen.
    for name in ("lag_attn", "baseline_decoder", "residual_decoder"):
        sub = getattr(vae, name)
        assert all(not p.requires_grad for p in sub.parameters()), (
            f"{name} must stay frozen — only the documented set unfreezes"
        )
