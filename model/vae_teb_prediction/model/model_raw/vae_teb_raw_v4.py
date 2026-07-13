r"""``SeqVaeRawV4`` -- the raw-signal fork of the v3 lag-attentive VAE-TEB model.

``SeqVaeRawV4`` is a subclass of :class:`SeqVaeLagAttnV3` that changes **only** the input
representation and the reconstruction target; the entire v3 information architecture and every
scientific-cleanliness guarantee (G0-G11) is inherited unchanged. Concretely:

- the two fixed scattering/phase adapters (``target_adapter`` / ``source_adapter``) are replaced by
  two learned, strictly-causal :class:`CausalRawFrontend` front ends $F_y, F_u$ over the raw $4$ Hz
  FHR/UP signals (§5.3 of the roadmap), and
- the two feature-domain decoders are replaced by :class:`RawBaselineFutureDecoderV4` /
  :class:`RawResidualFutureDecoderV4`, which forecast the future **raw FHR waveform**
  ($H = 30$ low-rate steps $\times R = 16$ raw substeps $= 480$ samples $= 2$ min) instead of the
  $87$-channel feature future (§5.4).

Every inherited method that positionally calls the feature adapters or reads feature-shaped batch
fields is overridden (:meth:`forward`, :meth:`encode_only`, :meth:`measure_transfer_entropy`,
:meth:`permutation_kl`, :meth:`_default_batch_to_inputs`); the domain-agnostic KL machinery, the fused
permutation control (:meth:`perm_kl_from_forward` / :meth:`perm_forward_outputs`), the four v3 heads,
and the shared horizon core are reused verbatim. The forward-dict **key contract is preserved
exactly** (the same 25 keys as v3), so the trainer, testing pipeline, and perm control run unchanged;
the raw-shaped decoder outputs now carry $(B, T, H, R)$ tensors and ``raw_future_pred`` is non-null.

Standardization convention (roadmap §5.5, "Loader normalizes; front-end stats = identity"): the
dataloader z-scores ``fhr``/``up`` with the global scalar stats, so the model works entirely in
normalized space. The front ends are therefore constructed with identity featurize stats
(``mean=0.0, std=1.0``) and ``sentinel=None`` -- a raw-bpm sentinel (e.g. $0$ bpm) does **not** survive
normalization and $0.0$ is itself the normalized mean, so the decimated ``weight`` mask (applied by
:meth:`CausalRawFrontend.featurize` via ``* mask``) is the authoritative gap signal.

The single-phase loss (:meth:`compute_loss`) is added in Sprint 3.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.vae_teb_prediction.model.model_raw.geometry import (
    CROP,
    H,
    WARMUP,
    RawGeometry,
    derive_geometry,
)
from model.vae_teb_prediction.model.model_raw.raw_frontend import (
    CausalRawFrontend,
    assert_no_time_pooling_norm,
)
from model.vae_teb_prediction.model.model_raw.raw_losses import (
    kld_terms,
    lowpass_loss,
    raw_mae,
    raw_nll,
    smooth_loss,
)
from model.vae_teb_prediction.model.model_raw.raw_masks import (
    forecast_mask,
    frontend_mask,
    kl_mask,
    low_rate_mask,
)
from model.vae_teb_prediction.model.model_raw.raw_targets import (
    build_future_index,
    build_future_target,
)
from model.vae_teb_prediction.model.model_raw.reuse import (
    HorizonDecoderCore,
    ResidualMLP,
    SeqVaeLagAttnV3,
    geometric_schedule,
    initialization,
)

# ``_apply_logvar_bound`` is the exact bound v3 applies to every log-variance head (clamp | smooth);
# reuse it so the raw decoders honour the same G2 smooth-bound contract.
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import _apply_logvar_bound

_DECODER_HEAD_CHOICES = ("learned_basis", "linear")


def _smooth_within_block_basis(basis_size: int, r: int) -> torch.Tensor:
    r"""A smooth $(K_b, R)$ within-block basis (DCT-II cosines) for the learned-basis head.

    Row $k$ is the increasing-frequency cosine

    $$
    B_{k, r} = \cos\!\left(\frac{\pi\, k\, (2r + 1)}{2R}\right),
    \qquad k \in [0, K_b),\; r \in [0, R),
    $$

    so a decoder that mixes a handful of these produces smooth $R$-sample forecasts and does not waste
    capacity chasing per-sample jitter (roadmap §5.4). The basis is a learnable :class:`nn.Parameter`
    (this is only its initialisation); the constant $k = 0$ row lets a single coefficient encode the
    block mean.

    Args:
        basis_size: Number of smooth basis functions $K_b$.
        r: Raw substeps per low-rate step $R$.

    Returns:
        A $(K_b, R)$ tensor of smooth basis functions.
    """
    k = torch.arange(basis_size, dtype=torch.float32).unsqueeze(1)  # (K_b, 1)
    n = torch.arange(r, dtype=torch.float32).unsqueeze(0)           # (1, R)
    return torch.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * r))     # (K_b, R)


class _RawFutureDecoderV4(nn.Module):
    r"""Shared raw-block head machinery for the two raw future decoders.

    Both raw decoders map a per-anchor horizon feature $G_{t,\tau} \in \mathbb{R}^{d_{\mathrm{hidden}}}$
    (produced by the shared :class:`HorizonDecoderCore`) to the $R$ raw substeps of that horizon step,
    emitting a mean and a smooth-bounded learned log-variance over $(B, T, H, R)$. The within-block
    head is **construction-gated** on ``decoder_head`` (only the selected branch is built, so no
    parameter is registered-but-unused -- important for DDP ``find_unused_parameters=False``):

    - ``'learned_basis'``: a coefficient head $a_{t,\tau,k} = \mathrm{Linear}(G_{t,\tau})$ then
      $\hat x_{t,\tau,r} = \sum_k a_{t,\tau,k} B_{k,r}$ with the smooth basis of
      :func:`_smooth_within_block_basis`;
    - ``'linear'``: a plain $\mathrm{Linear}(d_{\mathrm{hidden}}, R)$.

    The mean/coefficient projection is exposed as ``self.mean_head`` (a single :class:`nn.Linear`) so
    the inherited :meth:`SeqVaeLagAttnV1._zero_init_delta_heads` -- which zeroes
    ``residual_decoder.mean_head`` -- keeps the warm-start invariant working: with the coefficient head
    zeroed, $\hat x = 0 \cdot B = 0$ regardless of the basis. The log-variance is always a plain
    $\mathrm{Linear}(d_{\mathrm{hidden}}, R)$ (kept linear for stability, matching v3).
    """

    def _build_raw_head(
        self,
        d_hidden: int,
        r: int,
        *,
        decoder_head: str,
        basis_size: int,
        logvar_clamp: Tuple[float, float],
        logvar_bound: str,
    ) -> None:
        """Build the construction-gated within-block mean/logvar head (see the class docstring)."""
        if decoder_head not in _DECODER_HEAD_CHOICES:
            raise ValueError(
                f"decoder_head must be one of {_DECODER_HEAD_CHOICES}, got {decoder_head!r}"
            )
        self.r = int(r)
        self.decoder_head = str(decoder_head)
        self.basis_size = int(basis_size)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.logvar_bound = str(logvar_bound)

        if self.decoder_head == "learned_basis":
            self.mean_head = nn.Linear(d_hidden, self.basis_size)
            self.basis = nn.Parameter(_smooth_within_block_basis(self.basis_size, self.r))
        else:  # "linear"
            self.mean_head = nn.Linear(d_hidden, self.r)
        self.logvar_head = nn.Linear(d_hidden, self.r)

    def _build_proj_and_head(
        self,
        in_dim: int,
        d_hidden: int,
        r: int,
        *,
        decoder_head: str,
        basis_size: int,
        logvar_clamp: Tuple[float, float],
        logvar_bound: str,
        dropout: float,
    ) -> None:
        """Build the shared conditioning ``proj`` (a :class:`ResidualMLP`) + within-block head.

        The two raw decoders are identical here except for ``in_dim`` (``d_model`` for the baseline,
        ``d_model + d_z`` for the residual), so the whole projection+head stack lives on the shared
        base and each subclass only supplies its ``in_dim``.
        """
        self.proj = ResidualMLP(
            input_dim=in_dim,
            hidden_dims=geometric_schedule(in_dim, d_hidden, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self._build_raw_head(
            d_hidden,
            r,
            decoder_head=decoder_head,
            basis_size=basis_size,
            logvar_clamp=logvar_clamp,
            logvar_bound=logvar_bound,
        )

    def _raw_block(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map horizon features $(B, T, H, d_{\\mathrm{hidden}})$ to $(mu, logvar)$ over $(B, T, H, R)$."""
        coef = self.mean_head(feat)
        if self.decoder_head == "learned_basis":
            mu = torch.matmul(coef, self.basis)  # (B, T, H, K_b) @ (K_b, R) -> (B, T, H, R)
        else:
            mu = coef
        logvar = _apply_logvar_bound(self.logvar_head(feat), self.logvar_clamp, self.logvar_bound)
        return mu, logvar


class RawBaselineFutureDecoderV4(_RawFutureDecoderV4):
    r"""FHR-only baseline decoder over the raw future block (§5.4).

    Predicts $\hat X^{\mathrm{base}}_t = D_{\mathrm{base}}(b_t)$ of shape $(B, T, H, R)$ from the
    target-only decoder state $b_t$, reusing the shared :class:`HorizonDecoderCore`. Replaces
    ``self.baseline_decoder`` in place so the inherited losses/controls reach it unchanged.
    """

    def __init__(
        self,
        core: HorizonDecoderCore,
        d_model: int = 128,
        d_hidden: int = 128,
        r: int = 16,
        *,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        logvar_bound: str = "smooth",
        decoder_head: str = "learned_basis",
        basis_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the raw baseline decoder (see :class:`_RawFutureDecoderV4`)."""
        super().__init__()
        self.core = core
        self._build_proj_and_head(
            d_model,
            d_hidden,
            r,
            decoder_head=decoder_head,
            basis_size=basis_size,
            logvar_clamp=logvar_clamp,
            logvar_bound=logvar_bound,
            dropout=dropout,
        )

    def forward(self, decoder_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu_base, logvar_base)`` each of shape ``(B, T, H, R)``."""
        h = self.proj(decoder_state)
        feat = self.core.decode(h)
        return self._raw_block(feat)


class RawResidualFutureDecoderV4(_RawFutureDecoderV4):
    r"""Source-driven residual decoder over the raw future block (§5.4).

    Predicts $\Delta\hat X^{\mathrm{src}}_t = D_{\mathrm{src}}(b_t, z_t)$ of shape $(B, T, H, R)$ via a
    2-argument ``forward(decoder_state, z)`` (matching :class:`ResidualFutureDecoderV3`), reusing the
    shared :class:`HorizonDecoderCore`. The ``mean_head`` (mean/coefficient projection) is zero-inited
    by :meth:`SeqVaeLagAttnV1._zero_init_delta_heads`, so $\Delta\hat X^{\mathrm{src}} \equiv 0$ at init
    and $\hat X^{\mathrm{full}} = \hat X^{\mathrm{base}}$ (the warm-start / zero-KL invariant, G1).
    Replaces ``self.residual_decoder`` in place so the inherited fused perm control reuses it.
    """

    def __init__(
        self,
        core: HorizonDecoderCore,
        d_model: int = 128,
        d_z: int = 24,
        d_hidden: int = 128,
        r: int = 16,
        *,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        logvar_bound: str = "smooth",
        decoder_head: str = "learned_basis",
        basis_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the raw residual decoder (see :class:`_RawFutureDecoderV4`)."""
        super().__init__()
        self.core = core
        self._build_proj_and_head(
            d_model + d_z,
            d_hidden,
            r,
            decoder_head=decoder_head,
            basis_size=basis_size,
            logvar_clamp=logvar_clamp,
            logvar_bound=logvar_bound,
            dropout=dropout,
        )

    def forward(
        self, decoder_state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(delta_mu_src, logvar_full)`` each of shape ``(B, T, H, R)``."""
        h_in = torch.cat([decoder_state, z], dim=-1)
        h = self.proj(h_in)
        feat = self.core.decode(h)
        return self._raw_block(feat)


class SeqVaeRawV4(SeqVaeLagAttnV3):
    r"""Raw-signal lag-attentive VAE-TEB (v4). See the module docstring for the full contract."""

    #: Stored in checkpoints for the version-agnostic testing load-path guard.
    model_class = "SeqVaeRawV4"

    def __init__(
        self,
        *,
        frontend: Optional[Dict] = None,
        raw_len: int = 5280,
        decimation: int = 16,
        disable_source: bool = False,
        fhr_mean: float = 0.0,
        fhr_std: float = 1.0,
        up_mean: float = 0.0,
        up_std: float = 1.0,
        **v3_kwargs,
    ) -> None:
        r"""Initialize ``SeqVaeRawV4``.

        Args:
            frontend: The front-end configuration block (roadmap §5.3). Keys ``decoder_head`` and
                ``basis_size`` drive the raw decoders; every other key
                (``stages``/``channels``/``d_raw``/``antialias``/``antialias_kernel``/``gated``/
                ``norm_kind``/``norm_num_groups``/``first_kernels_fhr``/``first_kernels_up``/
                ``dropout``) is forwarded to :class:`CausalRawFrontend`. ``None`` uses the front-end
                defaults.
            raw_len: Raw samples per segment $L_{\mathrm{raw}}$ (config-driven geometry).
            decimation: Front-end total stride $D$ (also raw substeps per low-rate step $R$).
            disable_source: No-UP ablation. When True the attended source is zeroed before the
                posterior, so $q \approx p$ and $K_t \approx 0$ (roadmap §16, item 7).
            fhr_mean, fhr_std, up_mean, up_std: Fixed featurize z-score stats for $F_y$ / $F_u$.
                Default to identity ($0.0$ / $1.0$) because the loader already normalizes ``fhr``/``up``
                (roadmap §5.5); override only for the model-owned-standardization escape hatch.
            **v3_kwargs: Forwarded verbatim to :class:`SeqVaeLagAttnV3` (backbone widths, the v3
                scientific-cleanliness flags, etc.).
        """
        # -- Front-end config + config-driven geometry, resolved BEFORE super().__init__ so
        #    ``sequence_length`` can be forced to the front-end token count T (the two must match).
        fe = dict(frontend or {})
        decoder_head = fe.pop("decoder_head", "learned_basis")
        basis_size = fe.pop("basis_size", 8)
        # The geometry crop must equal the front-end crop, and horizon/warmup must equal the v3
        # ``horizon``/``warmup_period`` -- derive from the same sources so they cannot drift.
        crop = int(fe.get("crop", CROP))
        horizon = int(v3_kwargs.get("horizon", H))
        warmup = int(v3_kwargs.get("warmup_period", WARMUP))
        geometry = derive_geometry(
            raw_len, decimation, crop=crop, horizon=horizon, warmup=warmup
        )
        # The inherited core (encoders/attention/heads) is sized by ``sequence_length``; it MUST equal
        # the number of tokens the front end emits (T), so override it here rather than trusting the
        # caller to keep two numbers in sync.
        v3_kwargs["sequence_length"] = geometry.t

        super().__init__(**v3_kwargs)

        self.geometry: RawGeometry = geometry
        self.raw_len = int(raw_len)
        self.decimation = int(decimation)
        self.disable_source = bool(disable_source)

        # Cache the geometry-only future-target index grid as a (non-persistent) buffer so it moves
        # with the model and is NOT rebuilt on CPU + re-uploaded every training step (compute_loss
        # threads it into both build_future_target and forecast_mask).
        self.register_buffer(
            "_future_index", build_future_index(self.geometry), persistent=False
        )

        # Remove the feature adapters: they are now replaced by the front ends, and left in place
        # they would be parameters that never receive a gradient (starving DDP with
        # find_unused_parameters=False).
        del self.target_adapter
        del self.source_adapter

        # -- Front ends (distinct objects, never weight-shared: FHR and UP have different morphology).
        # Design A: identity featurize stats + sentinel disabled (0.0 is the normalized mean, a valid
        # value; the weight-derived mask is the authoritative gap signal). Any config `sentinel` is
        # ignored in favour of None to avoid zeroing valid mean-bpm samples.
        fe.pop("sentinel", None)
        # The raw tokens feed straight into the v1 encoders, which assume already-projected
        # (B, T, d_model) input -- v4 has NO re-projection adapter (unlike v3's feature adapters,
        # which structurally mapped any feature width to d_model). So the front-end output width
        # d_raw MUST equal d_model; validate it here and fail loudly at construction rather than
        # with a confusing shape mismatch deep inside the encoder on the first forward. (channels[-1]
        # is irrelevant: CausalRawFrontend forces its last stage output to d_raw.)
        d_model = int(v3_kwargs.get("d_model", 128))
        d_raw = int(fe.get("d_raw", 128))
        if d_raw != d_model:
            raise ValueError(
                f"front-end token width d_raw ({d_raw}) must equal the model width d_model "
                f"({d_model}); the raw tokens feed straight into the v1 encoders (no re-projection "
                "adapter). Set frontend.d_raw == d_model."
            )
        self.frontend_y = CausalRawFrontend(
            stream="y",
            mean=fhr_mean,
            std=fhr_std,
            raw_len=raw_len,
            decimation=decimation,
            sentinel=None,
            **fe,
        )
        self.frontend_u = CausalRawFrontend(
            stream="u",
            mean=up_mean,
            std=up_std,
            raw_len=raw_len,
            decimation=decimation,
            sentinel=None,
            **fe,
        )

        # -- Raw future decoders replace the inherited feature decoders in place (share horizon_core).
        # d_model was resolved (and validated == d_raw) above, before the front ends were built.
        d_z = int(v3_kwargs.get("d_z", 24))
        decoder_hidden = int(v3_kwargs.get("decoder_hidden", 128))
        dropout = float(v3_kwargs.get("dropout", 0.1))
        logvar_clamp = v3_kwargs.get("logvar_clamp", (-5.0, 3.0))
        r = self.geometry.r
        self.baseline_decoder = RawBaselineFutureDecoderV4(
            core=self.horizon_core,
            d_model=d_model,
            d_hidden=decoder_hidden,
            r=r,
            logvar_clamp=logvar_clamp,
            logvar_bound=self.logvar_bound,
            decoder_head=decoder_head,
            basis_size=basis_size,
            dropout=dropout,
        )
        self.residual_decoder = RawResidualFutureDecoderV4(
            core=self.horizon_core,
            d_model=d_model,
            d_z=d_z,
            d_hidden=decoder_hidden,
            r=r,
            logvar_clamp=logvar_clamp,
            logvar_bound=self.logvar_bound,
            decoder_head=decoder_head,
            basis_size=basis_size,
            dropout=dropout,
        )

        # Initialise only the NEW submodules. The front ends are fully new; for the decoders init the
        # proj + heads but NOT ``core`` (the shared horizon core the parent already initialised).
        initialization(self.frontend_y)
        initialization(self.frontend_u)
        for dec in (self.baseline_decoder, self.residual_decoder):
            initialization(dec.proj)
            initialization(dec.mean_head)
            initialization(dec.logvar_head)

        # Re-assert the zero-init delta heads: the parent's call in ``_install_v3_heads`` ran before
        # the raw decoders existed (it zeroed the discarded v3 decoder), so re-run it now to zero the
        # raw residual ``mean_head`` -> delta_mu_src == 0 at init (G1 warm-start invariant).
        self._zero_init_delta_heads()

        # G0-in-the-front-end: fail loudly on any time-pooling / batch-coupling normaliser in the
        # strictly-causal INPUT pathway (front ends + encoders). Scoped to the input pathway on
        # purpose: the shared decoder ``horizon_core`` keeps ``nn.GroupNorm`` (it pools over the
        # forecast-horizon axis within a single anchor, not across input time, and v3 proves it leaks
        # nothing), so a whole-model guard would false-positive on it. The encoders are already
        # causalised by ``causal_norm=True`` (required for the raw TE reading); the front ends also
        # self-guard in their own ``__init__``.
        assert_no_time_pooling_norm(self.frontend_y)
        assert_no_time_pooling_norm(self.frontend_u)
        assert_no_time_pooling_norm(self.target_encoder)
        assert_no_time_pooling_norm(self.source_encoder)

    # ------------------------------------------------------------------
    # Forward / sampling (raw input pathway, v3 key contract preserved)
    # ------------------------------------------------------------------
    def _encode_states(
        self,
        fhr_raw: torch.Tensor,
        up_raw: torch.Tensor,
        mask: torch.Tensor,
        *,
        up_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the two front ends + encoders, returning the target/source states ``(H_y, H_u)``.

        ``up_mask`` defaults to ``mask`` (the shared weight-derived raw validity). Pass a distinct
        ``up_mask`` to mask the **source** stream independently of the target -- e.g. an honest
        UP-only perturbation control that reorders the source *and its validity together* without
        disturbing the FHR path (the target always keeps the original ``mask``). This mirrors
        :meth:`permutation_kl`, which likewise front-ends the permuted source with ``mask[perm]``
        while the target keeps the original mask.
        """
        up_mask = mask if up_mask is None else up_mask
        H_y = self.target_encoder(self.frontend_y(fhr_raw, mask))
        H_u = self.source_encoder(self.frontend_u(up_raw, up_mask))
        return H_y, H_u

    def _zero_disabled_source(
        self, A: torch.Tensor, A_heads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Zero the attended source for the no-UP ablation ($q \approx p$, $K_t \approx 0$), DDP-safely.

        Multiplies by $0.0$ rather than replacing with a fresh ``zeros_like`` tensor so the source
        front end / encoder / attention stay in the autograd graph (they receive a zero gradient,
        not ``None``) -- a disconnected source subtree would be reported as unused parameters under
        DDP ``find_unused_parameters=False`` and crash the backward (S8-T05). Shared by
        :meth:`forward` and :meth:`encode_only` so the invariant lives in one place. (The
        permuted-source control in :meth:`_perm_posterior` zeroes ``h_u_perm`` upstream instead.)
        """
        return A * 0.0, A_heads * 0.0

    def forward(
        self,
        fhr_raw: torch.Tensor,
        up_raw: torch.Tensor,
        mask: torch.Tensor,
        *,
        up_mask: Optional[torch.Tensor] = None,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the full raw pipeline; returns the exact 25-key v3 forward dict.

        Identical to :meth:`SeqVaeLagAttnV3.forward` except the two feature adapters are replaced by
        the raw front ends and the decoder outputs are raw-shaped $(B, T, H, R)$. The only key whose
        semantics change is ``raw_future_pred`` (now the non-null full raw forecast $\hat X^{full}$);
        ``mu_base``/``logvar_base``/``delta_mu_src``/``mu_full``/``logvar_full`` keep their names but
        carry $(B, T, H, R)$.

        Args:
            fhr_raw: Raw FHR signal $(B, L_{\mathrm{raw}})$ (loader-normalized).
            up_raw: Raw UP signal $(B, L_{\mathrm{raw}})$ (loader-normalized).
            mask: Raw validity mask $(B, L_{\mathrm{raw}})$ (1 = valid); masks the target and (by
                default) the source stream.
            up_mask: Optional separate validity mask for the source stream; defaults to ``mask``.
                Used by UP-only perturbation controls that reorder the source and its validity
                together (see :meth:`_encode_states`).
            lag_band_mask: Optional lag keep-mask; ``None`` is a bit-exact no-op (see v3).

        Returns:
            The 25-key forward dict.
        """
        H_y, H_u = self._encode_states(fhr_raw, up_raw, mask, up_mask=up_mask)

        mu_prior, logvar_prior, decoder_state, raw_logvar_prior = self.prior_head(H_y)

        m_lag, dead = self._combined_lag_mask(H_y.size(1), H_y.device, lag_band_mask)
        A, alpha, A_heads = self.lag_attn(H_y, H_u, m_lag)
        A, alpha, A_heads = self._ablate_dead_anchors(A, alpha, A_heads, dead)

        if self.disable_source:
            A, A_heads = self._zero_disabled_source(A, A_heads)

        post_src = A_heads if self.head_structured_latent else A
        mu_post, logvar_post = self.posterior_head(
            H_y, post_src, mu_prior, raw_logvar_prior
        )
        z = self.reparameterize(mu_post, logvar_post)

        with torch.no_grad():
            mu_prior_sat_frac = (
                mu_prior.abs() >= (0.99 * self.mu_scale)
            ).float().mean()
            delta_mu_sat_frac = (
                (mu_post - mu_prior).abs() >= (0.99 * self.delta_mu_scale)
            ).float().mean()

        if self.training:
            self._update_latent_running_stats(mu_post)

        mu_base, logvar_base = self.baseline_decoder(decoder_state)
        delta_mu_src, logvar_full = self.residual_decoder(decoder_state, z)
        mu_full = mu_base + delta_mu_src

        kld_btd = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
            mask_warmup=False,
        )
        kld_per_t, te_lag_map, kld_per_t_per_head = self.te_analysis(
            kld_btd, alpha, head_structured=self.head_structured_latent
        )

        warmup_mask = self._build_warmup_valid_mask(H_y.size(1), device=H_y.device)
        kld_active_frac = self._kld_active_frac(kld_btd)

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "raw_logvar_prior": raw_logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": H_u,
            "decoder_state": decoder_state,
            "attended_source": A,
            "attended_source_heads": A_heads,
            "attn_weights": alpha,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "delta_mu_src": delta_mu_src,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "raw_future_pred": mu_full,
            "kld_per_t": kld_per_t,
            "kld_per_t_per_head": kld_per_t_per_head,
            "te_lag_map": te_lag_map,
            "warmup_mask": warmup_mask,
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
            "kld_active_frac": kld_active_frac,
        }

    def encode_only(
        self,
        fhr_raw: torch.Tensor,
        up_raw: torch.Tensor,
        mask: torch.Tensor,
        sample_z: bool = True,
        *,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run front ends + encoders + posterior only (no decoders). Same 11-key contract as v3.

        Args:
            fhr_raw: Raw FHR signal $(B, L_{\\mathrm{raw}})$.
            up_raw: Raw UP signal $(B, L_{\\mathrm{raw}})$.
            mask: Raw validity mask $(B, L_{\\mathrm{raw}})$.
            sample_z: Reparameterise when True, else return the posterior mean as ``z``.
            lag_band_mask: Optional lag keep-mask; ``None`` is a bit-exact no-op.

        Returns:
            The 11-key encode dict.
        """
        H_y, H_u = self._encode_states(fhr_raw, up_raw, mask)
        mu_prior, logvar_prior, decoder_state, raw_logvar_prior = self.prior_head(H_y)
        m_lag, dead = self._combined_lag_mask(H_y.size(1), H_y.device, lag_band_mask)
        A, alpha, A_heads = self.lag_attn(H_y, H_u, m_lag)
        A, alpha, A_heads = self._ablate_dead_anchors(A, alpha, A_heads, dead)
        if self.disable_source:
            A, A_heads = self._zero_disabled_source(A, A_heads)
        post_src = A_heads if self.head_structured_latent else A
        mu_post, logvar_post = self.posterior_head(
            H_y, post_src, mu_prior, raw_logvar_prior
        )
        z = self.reparameterize(mu_post, logvar_post) if sample_z else mu_post
        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": H_u,
            "decoder_state": decoder_state,
            "attended_source": A,
            "attended_source_heads": A_heads,
            "attn_weights": alpha,
        }

    def measure_transfer_entropy(
        self,
        fhr_raw: torch.Tensor,
        up_raw: torch.Tensor,
        mask: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        """Estimate the TE surrogate $\\mathrm{KL}(q\\,\\|\\,p)$ over the KL support (raw inputs).

        Mirrors :meth:`SeqVaeLagAttnV3.measure_transfer_entropy` but builds the encode states from the
        raw front ends. ``reduce_mean=True`` returns the anchor-support scalar; ``reduce_mean=False``
        returns the $(B, T, d_z)$ per-step tensor with out-of-support steps set to ``NaN``.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                enc = self.encode_only(fhr_raw, up_raw, mask, sample_z=True)
                if reduce_mean:
                    return self._kld_loss(
                        mu_prior=enc["mu_prior"],
                        logvar_prior=enc["logvar_prior"],
                        mu_post=enc["mu_post"],
                        logvar_post=enc["logvar_post"],
                        reduce_mean=True,
                    )
                kld = self.kld_tensor(
                    mu_prior=enc["mu_prior"],
                    logvar_prior=enc["logvar_prior"],
                    mu_post=enc["mu_post"],
                    logvar_post=enc["logvar_post"],
                    mask_warmup=False,
                )
                support = self._kld_support_mask(kld.size(1), device=kld.device) > 0
                kld = kld.clone()
                kld[:, ~support, :] = float("nan")
                return kld
        finally:
            # Restore the caller's mode; a train-then-measure harness must not be left in eval()
            # (dropout off / eval-norm) for its subsequent training minibatches.
            if was_training:
                self.train()

    def permutation_kl(
        self,
        fhr_raw: torch.Tensor,
        up_raw: torch.Tensor,
        mask: torch.Tensor,
        *,
        weight: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        perm_index: Optional[torch.Tensor] = None,
        detach_prior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        r"""Source-permutation control $L_{\mathrm{perm}}$ (G6), re-encoding the raw source.

        The eval-time reference for the fused :meth:`perm_kl_from_forward`: deranges the batch on the
        raw UP stream, re-encodes $\pi(U)$ through $F_u$ + the source encoder, and returns the
        anchor-masked $\mathrm{KL}(q(z\mid Y, \pi(U)) \,\|\, p(z\mid Y))$. Because $F_u$ and the source
        encoder are batch-independent (per-sample causal convs / cumulative norms / LSTM / LayerNorm),
        re-encoding $\pi(U)$ equals permuting the source state along the batch axis -- which is exactly
        what :meth:`perm_kl_from_forward` does; the two agree to numerical precision.

        Args:
            fhr_raw: Raw FHR signal $(B, L_{\mathrm{raw}})$.
            up_raw: Raw UP signal $(B, L_{\mathrm{raw}})$.
            mask: Raw validity mask $(B, L_{\mathrm{raw}})$ (permuted with the source to match).
            weight: Optional per-sample dataset weight broadcastable to $(B, T)$.
            generator: Optional CPU generator seeding the derangement.
            perm_index: Optional precomputed derangement $(B,)$; drawn if omitted.
            detach_prior: See :meth:`SeqVaeLagAttnV3._perm_posterior`.

        Returns:
            ``{"perm_kl", "kld_shuffled", "kld_shuffled_per_t", "perm_index"}``.
        """
        perm_index = self._resolve_perm_index(
            fhr_raw.size(0), perm_index, generator, fhr_raw.device
        )
        h_y = self.target_encoder(self.frontend_y(fhr_raw, mask))
        h_u_perm = self.source_encoder(
            self.frontend_u(up_raw[perm_index], mask[perm_index])
        )
        mu_prior, logvar_prior, _, raw_logvar_prior = self.prior_head(h_y)
        mu_p, logvar_p, mu_q, logvar_q = self._perm_posterior(
            h_y, h_u_perm, mu_prior, logvar_prior, raw_logvar_prior, detach_prior
        )
        return self._perm_kl_result(mu_p, logvar_p, mu_q, logvar_q, perm_index, weight)

    def _perm_posterior(
        self,
        h_y: torch.Tensor,
        h_u_perm: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        raw_logvar_prior: Optional[torch.Tensor],
        detach_prior: bool,
        *,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""v3's permuted-source posterior, with the ``disable_source`` no-UP ablation honoured.

        When ``disable_source`` is set, :meth:`forward` zeroes the attended source so $q \approx p$
        and $K_t \approx 0$; the permutation control must collapse the same way. Both entry points
        -- the :meth:`permutation_kl` override and the inherited fused
        :meth:`SeqVaeLagAttnV3.perm_kl_from_forward` -- delegate here, so zeroing the permuted
        source state at this single point makes ``kld_shuffled`` collapse to $\approx 0$ too (a
        weighted sum over zero-valued attention values is zero). Without it the control re-encodes
        the live permuted UP, so ``kld_shuffled`` is spuriously non-zero against a near-zero
        ``kld_raw`` (blowing up ``kld_shuffled_ratio``), and with ``lambda_perm > 0`` the perm
        gradient would train the posterior on a source the main path structurally ignores. Uses
        ``* 0.0`` (not ``zeros_like``) to keep the source subtree in the autograd graph, matching
        :meth:`forward`.
        """
        if self.disable_source:
            h_u_perm = h_u_perm * 0.0
        return super()._perm_posterior(
            h_y,
            h_u_perm,
            mu_prior,
            logvar_prior,
            raw_logvar_prior,
            detach_prior,
            lag_band_mask=lag_band_mask,
        )

    # ------------------------------------------------------------------
    # Batch adapter (keeps the inherited fit_latent_stats working on raw batches)
    # ------------------------------------------------------------------
    def _default_batch_to_inputs(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(fhr_raw, up_raw, mask)`` from a raw batch (``fhr``/``up``/``weight``).

        The raw validity mask is the decimated ``weight`` nearest-upsampled to the raw grid
        (:func:`frontend_mask`); the full forecast/KL masks are built inside :meth:`compute_loss`.
        """
        mask = frontend_mask(batch.weight, self.raw_len, self.decimation)
        return batch.fhr, batch.up, mask

    # ------------------------------------------------------------------
    # Single-phase raw loss (Sprint 3)
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        fhr_raw: torch.Tensor,
        mask: torch.Tensor,
        *,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
        lambda_full: float = 1.0,
        lambda_base: float = 0.5,
        likelihood: str = "gaussian_nll",
        sigma_obs: str = "learned",
        free_bits: float = 0.0,
        detach_baseline_in_full: bool = True,
        lambda_lp: float = 0.5,
        lambda_smooth: float = 0.1,
        lowpass_scales: Sequence[int] = (4, 16, 32, 60),
        lambda_lag: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        r"""The single-phase raw objective (§10). Does **not** delegate to the feature-domain v3 loss.

        $$
        \mathcal L = \lambda_{\mathrm{full}}\,\mathcal L_{\mathrm{raw}}
        + \lambda_{\mathrm{base}}\,\mathcal L_{\mathrm{base}}
        + \beta\,\mathcal L_{\mathrm{KL}}
        + \lambda_{\mathrm{lp}}\,\mathcal L_{\mathrm{lowpass}}
        + \lambda_{\mathrm{smooth}}\,\mathcal L_{\Delta}
        + \lambda_{\mathrm{lag}}\,\mathcal L_{\mathrm{lag}}.
        $$

        Args:
            forward_outputs: The dict returned by :meth:`forward` (raw-shaped decoder tensors).
            fhr_raw: Raw FHR signal $(B, L_{\mathrm{raw}})$ -- the source of the forecast target
                $X^+$ (gathered crop-aligned).
            mask: Raw validity mask $(B, L_{\mathrm{raw}})$ -- the source of the forecast/KL masks.
            compute_kld_loss: If False, the KL term is $0$ (the prediction-space perm readout path).
            beta: KL weight.
            lambda_full, lambda_base: Weights on the full/baseline raw NLL.
            likelihood: Only ``'gaussian_nll'`` is supported (the raw model is learned-variance NLL).
            sigma_obs: Only ``'learned'`` is supported (the decoder log-variance heads).
            free_bits: Per-dim free-bit floor for the trained KL.
            detach_baseline_in_full: When True (default), the full NLL uses a stop-gradiented baseline
                ($\hat X^{\mathrm{full}} = \mathrm{sg}(\hat X^{\mathrm{base}}) + \Delta\hat X^{\mathrm{src}}$)
                so the residual path cannot improve the baseline.
            lambda_lp, lambda_smooth: Weights on the low-pass and first-difference auxiliary losses.
            lowpass_scales: Block-average scales (seconds) for the low-pass loss.
            lambda_lag: Weight on the inherited lag-embedding smoothness regulariser.

        Returns:
            A dict with ``feat_loss`` (= the full raw NLL, so ``pred_gap``/prog-bar work), ``raw_loss``,
            ``base_loss``, ``kld_loss``, ``kld_raw``, ``kld_train``, ``kld_active_frac``,
            ``lowpass_loss``, ``smooth_loss``, ``lag_smoothness``, ``raw_mae`` (a scale-free reported
            forecast diagnostic), ``mean_logvar_full``, ``mean_logvar_base``, ``total_loss`` and ``beta``.
        """
        if likelihood != "gaussian_nll":
            raise ValueError(
                f"SeqVaeRawV4 supports only likelihood='gaussian_nll', got {likelihood!r}"
            )
        if sigma_obs != "learned":
            raise ValueError(
                f"SeqVaeRawV4 supports only sigma_obs='learned', got {sigma_obs!r}"
            )

        geo = self.geometry
        t_valid = geo.t_valid

        # -- Target + multi-resolution masks (all derived from the single raw validity mask).
        # The static (T_valid, H, R) target-index grid is the cached model buffer (built once in
        # __init__), so build_future_target/forecast_mask reuse it instead of rebuilding + uploading.
        fut_idx = self._future_index
        x_plus = build_future_target(fhr_raw, geo, future_index=fut_idx)   # (B, T_valid, H, R)
        m_low = low_rate_mask(mask, geo)                                   # (B, T)
        f_mask = forecast_mask(mask, m_low, geo, future_index=fut_idx)     # (B, T_valid, H, R)
        kl_weight = kl_mask(m_low, geo)                                    # (B, T)

        # -- Slice predictions to the valid anchor range.
        mu_base = forward_outputs["mu_base"][:, :t_valid]
        logvar_base = forward_outputs["logvar_base"][:, :t_valid]
        logvar_full = forward_outputs["logvar_full"][:, :t_valid]
        if detach_baseline_in_full:
            mu_full = mu_base.detach() + forward_outputs["delta_mu_src"][:, :t_valid]
        else:
            mu_full = forward_outputs["mu_full"][:, :t_valid]

        # -- Raw + baseline NLL (learned variance) + the variance-collapse diagnostics.
        raw_loss, mean_logvar_full = raw_nll(mu_full, logvar_full, x_plus, f_mask)
        base_loss, mean_logvar_base = raw_nll(mu_base, logvar_base, x_plus, f_mask)
        # -- Scale-free forecast diagnostic (reported, not optimised).
        raw_mae_val = raw_mae(mu_full, x_plus, f_mask)

        # -- KL (inherited v3 support + honest reporting).
        kld_train, kld_raw, kld_active_frac = kld_terms(
            self,
            forward_outputs,
            weight=kl_weight,
            free_bits=free_bits,
            compute_kld_loss=compute_kld_loss,
        )

        # -- Auxiliary raw-domain losses (on the full prediction).
        # Raw rate is 4 Hz (f_s), so a scale of q seconds is a block of 4q samples.
        lp_loss = lowpass_loss(mu_full, x_plus, f_mask, scales_sec=lowpass_scales, fs=4)
        sm_loss = smooth_loss(mu_full, x_plus, f_mask)

        # -- Lag-embedding smoothness (inherited pattern).
        if lambda_lag > 0.0:
            r = self.lag_attn.lag_embeddings
            lag_diff = r[1:] - r[:-1]
            lag_smoothness = (lag_diff ** 2).mean()
        else:
            lag_smoothness = raw_loss.new_zeros(())

        total_loss = (
            lambda_full * raw_loss
            + lambda_base * base_loss
            + beta * kld_train
            + lambda_lp * lp_loss
            + lambda_smooth * sm_loss
            + lambda_lag * lag_smoothness
        )

        return {
            "feat_loss": raw_loss,
            "raw_loss": raw_loss,
            "base_loss": base_loss,
            "kld_loss": kld_train,
            "kld_raw": kld_raw,
            "kld_train": kld_train,
            "kld_active_frac": kld_active_frac,
            "lowpass_loss": lp_loss,
            "smooth_loss": sm_loss,
            "lag_smoothness": lag_smoothness,
            "raw_mae": raw_mae_val,
            "mean_logvar_full": mean_logvar_full,
            "mean_logvar_base": mean_logvar_base,
            "total_loss": total_loss,
            "beta": raw_loss.new_tensor(float(beta)),
        }
