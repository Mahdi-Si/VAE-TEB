r"""Future decoders: two forecasts of the same horizon, one blind to the source.

The model forecasts the next $H_d$ steps of target features twice:

* the **baseline** forecast $\hat{Y}^{base}$, from the target's own past only;
* the **full** forecast $\hat{Y}^{base} + \Delta\hat{Y}^{src}$, where the correction is driven by
  the latent -- and therefore by the source.

That pairing is the whole design. The baseline is trained to be a strong forecaster in its own
right, so the correction can only earn its keep by carrying information the target's past did not
already have. Without the baseline term the model could route target information through the
latent and post a good forecast while telling us nothing about the source.

The residual decoder's mean head is zero-initialised, so $\Delta\hat{Y}^{src} \approx 0$ at init
and the full forecast starts equal to the baseline. Divergence is learned, not assumed.

Both decoders share one :class:`HorizonDecoderCore`. The horizon dynamics are the same problem
for both, so learning them twice would cost parameters and, worse, let the two forecasts drift
into different representation spaces -- at which point their difference stops being a correction
and becomes a comparison of strangers.

**The optional persistence residual, and why it opens no bypass.** The baseline decoder can be
built to add $w_{\tau,c}\, y_{t,c}$ to its mean, where $y_t$ is the *target's* own value at the
anchor step. It is target-only by construction, the caller hands the **same** tensor to both
invocations of the shared decoder, and it touches the mean head alone -- so the base-minus-full
gap it is read through stays a pure source readout and the log-variance path is untouched. What it
buys is that the mean head stops having to synthesise the level of a strongly autocorrelated
coefficient out of $z$ before it can say anything about its motion, which is what a horizon-uniform
objective was paying it to do at the near steps.
"""
from __future__ import annotations

from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound

#: What each horizon-attention block's residual gain is set to at construction. Small enough that
#: the stack starts near-identity, nonzero so every projection behind it carries gradient from the
#: first step. Named because it is a *known construction constant* that a trained checkpoint must
#: have moved -- which is what makes the gains usable as a load witness in the evaluation
#: pipeline's preflight.
HORIZON_ATTENTION_GAIN_INIT = 1.0e-2

#: Half-life, in horizon steps, of the persistence residual's **seeded** weight decay:
#: $w_{\tau,c} = 2^{-\tau / \text{halflife}}$ at construction, so $w_{0,c} = 1$ (the near step
#: starts as plain persistence) and $w_{29,c} \approx 0.017$.
#:
#: Short against the $30$-step horizon on purpose. The weight is a raw parameter and is learned, so
#: this is where the search starts rather than where it ends -- but a stored coefficient's own
#: autocorrelation is short, and the fast channels this residual exists for lose their persistence
#: skill within a handful of steps. Seeding a long tail there would start the mean head correcting a
#: term that is mostly wrong at the far steps, which is the opposite of the point.
#:
#: $5$ steps is $20$ s on the family's $4$ s grid. A named construction constant rather than a
#: configuration key: no arm varies it, and what an arm would vary is whether the residual exists.
PERSISTENCE_DECAY_HALFLIFE = 5.0


class _HorizonRefine(nn.Module):
    """Stack of dilated convolutions along the forecast-horizon axis.

    Each block is ``Conv1d -> GroupNorm -> GELU`` with a residual add. Dilations double with
    depth, so the horizon receptive field grows exponentially and a shallow stack still sees the
    whole forecast window.

    Shapes:
        Input:  ``(N, d_hidden, H_d)``
        Output: ``(N, d_hidden, H_d)``
    """

    def __init__(
        self,
        d_hidden: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2),
        film_cond_dim: Optional[int] = None,
    ) -> None:
        """Initialize the refine stack.

        Args:
            d_hidden: Channel width, held constant through the stack.
            kernel_size: Convolution kernel width.
            dilations: Dilation per block.
            film_cond_dim: Width of the per-block FiLM conditioning vector, or ``None`` for no
                per-block FiLM. When set, each block gets its own zero-initialised
                $(\\gamma, \\beta)$ generator, so every block -- not just the top of the stack --
                reads the latent. Zero-init makes the modulation an exact identity at construction,
                so this is a strict capacity add.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            # Symmetric padding: this axis is the forecast horizon, not time. Every step of it
            # is predicted from the same anchor, so there is no future here to leak.
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            d_hidden,
                            d_hidden,
                            kernel_size=kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        "norm": nn.GroupNorm(
                            num_groups=min(8, d_hidden), num_channels=d_hidden
                        ),
                    }
                )
            )

        # Per-block FiLM: one generator per block, zero-initialised so gamma = beta = 0 and the
        # block behaves exactly as it does without FiLM at construction. Left None when the core
        # owns FiLM at the top of the stack (or runs no FiLM at all), so no dead generators exist.
        self.film: Optional[nn.ModuleList]
        if film_cond_dim is not None:
            self.film = nn.ModuleList(
                [nn.Linear(film_cond_dim, 2 * d_hidden) for _ in dilations]
            )
            for layer in self.film:
                layer = cast(nn.Linear, layer)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            self.film = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Refine along the horizon axis, optionally modulating each block by the latent.

        Args:
            x: Input of shape ``(N, d_hidden, H_d)``.
            cond: Per-block FiLM conditioning ``(N, film_cond_dim)``, or ``None``. Ignored unless
                the stack was built with ``film_cond_dim``.

        Returns:
            Output of the same shape.
        """
        for index, block in enumerate(self.blocks):
            block = cast(nn.ModuleDict, block)
            y = block["conv"](x)
            y = F.gelu(block["norm"](y))
            if self.film is not None and cond is not None:
                film = cast(nn.ModuleList, self.film)
                gamma, beta = cast(nn.Linear, film[index])(cond).chunk(2, dim=-1)
                # Broadcast the per-channel modulation over the horizon axis (the last dim of y).
                y = y * (1.0 + gamma[..., None]) + beta[..., None]
            x = x + y
        return x


class _HorizonSelfAttention(nn.Module):
    r"""Pre-norm bidirectional self-attention over the forecast-horizon tokens.

    The dilated convolutions above mix horizon steps through a fixed local window; this mixes all
    $H_d$ of them at once, so a forecast can be shaped as a whole rather than assembled from
    overlapping neighbourhoods. The attention is deliberately **not** masked: this axis is the
    forecast horizon of a single anchor, every step of which is predicted from the same $t$, so
    there is no future here to leak -- the same argument that lets the refine stack pad
    symmetrically.

    Hand-rolled rather than :class:`torch.nn.MultiheadAttention` for two reasons that are
    correctness rather than taste. MHA applies its attention dropout functionally, where the
    dropout-zero scans that guard the twice-invoked decoder cannot see it; and the generic
    :func:`~teb_vae.lag_attn.nets.blocks.initialization` pass would xavier-fill MHA's *packed*
    ``in_proj_weight`` as one $3d \times d$ matrix, giving q, k and v a fan-in three times the one
    they actually have.

    No positional encoding of its own: the core's ``horizon_embedding`` is already what tells the
    $H_d$ tokens apart, and a second one would be a duplicate identity for the same axis.

    Shapes:
        Input:  ``(N, H_d, d_hidden)``
        Output: ``(N, H_d, d_hidden)``
    """

    def __init__(self, d_hidden: int, num_heads: int) -> None:
        """Initialize one attention block.

        Args:
            d_hidden: Token width; must be divisible by ``num_heads``. Validated by the core,
                which is where the two values meet a configuration.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = int(num_heads)
        self.d_head = d_hidden // self.num_heads

        self.norm = nn.LayerNorm(d_hidden)
        # Bias-free: the pre-norm already centres the input, so a bias on the query and key
        # projections only shifts every logit by the same constant.
        self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.k_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.v_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_hidden, bias=False)

        # A bare nn.Parameter, which is exactly what makes this survive: `initialization` fills
        # Linear and Conv1d weights and re-inits LayerNorm, and ignores loose parameters -- so the
        # gain stays where the constructor put it and the stack starts near-identity. Small rather
        # than zero, so every projection above carries gradient from the first step instead of
        # waiting for the gain to leave an exact zero.
        self.residual_gain = nn.Parameter(torch.full((1,), HORIZON_ATTENTION_GAIN_INIT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix the horizon tokens of each row and add the result back.

        Args:
            x: Horizon tokens ``(N, H_d, d_hidden)``. Anchors are folded into ``N`` by the caller,
                so one row is one anchor's forecast and nothing crosses between them.

        Returns:
            The same shape.
        """
        n_rows, horizon, d_hidden = x.shape
        normed = self.norm(x)
        # (N, heads, H_d, d_head): heads next to the batch, so the attention runs over the horizon.
        query = self.q_proj(normed).reshape(n_rows, horizon, self.num_heads, self.d_head)
        key = self.k_proj(normed).reshape(n_rows, horizon, self.num_heads, self.d_head)
        value = self.v_proj(normed).reshape(n_rows, horizon, self.num_heads, self.d_head)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            # No mask (the horizon is not time) and no dropout (this module is invoked twice per
            # forward, and two masks would make the base and full forecasts differ by noise).
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(n_rows, horizon, d_hidden)
        return x + self.residual_gain * self.out_proj(attended)


class HorizonDecoderCore(nn.Module):
    r"""Shared horizon-refinement core, used by both future decoders.

    Holds the forecast-step embedding, optional FiLM modulation, the dilated refine stack, an
    optional stack of self-attention blocks over the horizon tokens, and the output norm. Both
    decoders push their projected ``(B, T, d_hidden)`` state through the same
    :meth:`decode`, so the residual decoder operates as a correction *in the baseline's
    representation space*.

    FiLM modulates the horizon-expanded features with per-channel $(\gamma, \beta)$ generated
    from the projected state. Its generator is zero-initialised, so the core starts as an
    identity.

    Note this core's normalisers are deliberately *not* causalised. They pool over the forecast
    axis of a single anchor, not across input time: every step of that axis is predicted from the
    same $t$, so pooling over it cannot reach information the anchor did not already have. The
    optional self-attention over the horizon tokens is symmetric for the same reason.
    """

    def __init__(
        self,
        d_hidden: int = 128,
        horizon: int = 30,
        kernel_size: int = 3,
        depth: int = 2,
        film: bool = False,
        film_per_block: bool = False,
        attention_blocks: int = 0,
        attention_heads: int = 4,
    ) -> None:
        """Initialize the shared horizon core.

        Args:
            d_hidden: Decoder hidden width.
            horizon: Forecast horizon $H_d$.
            kernel_size: Horizon-convolution kernel width.
            depth: Number of dilated blocks; dilations are $1, 2, 4, \\dots$.
            film: Whether to apply FiLM conditioning per horizon step.
            film_per_block: Move FiLM from a single top-of-stack generator to one generator per
                refine block, so every block reads the latent instead of only the input to the
                stack. Requires ``film``. When set, the single ``film_gen`` is **not** built and the
                per-block generators live inside ``refine``; the two are never built together, so no
                dead generator exists.
            attention_blocks: Number of bidirectional self-attention blocks run over the $H_d$
                horizon tokens after the refine stack. ``0`` (the default) builds no module at all,
                so a core that does not ask for them is parameter-for-parameter and
                forward-identical to one built before they existed.
            attention_heads: Heads per attention block. Must divide ``d_hidden``. Read only when
                ``attention_blocks`` is positive, so a width that no attention will ever see is not
                held to a constraint it does not need.

        Raises:
            ValueError: If ``film_per_block`` is set without ``film`` -- per-block FiLM is a form of
                FiLM, not a separate mechanism, so that combination is a construction error rather
                than a silent no-op. Or if ``attention_heads`` does not divide ``d_hidden`` while
                attention is requested.
        """
        super().__init__()
        self.horizon = int(horizon)
        self.d_hidden = int(d_hidden)
        self.film = bool(film)
        self.film_per_block = bool(film_per_block)
        if self.film_per_block and not self.film:
            raise ValueError(
                "film_per_block=True requires film=True; per-block FiLM is a form of FiLM, not a "
                "separate mechanism"
            )

        # An internal forecast-step embedding: which step of the horizon this is, not what time
        # it was in the recording.
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, d_hidden))
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)

        depth = max(1, int(depth))
        dilations = tuple(2**index for index in range(depth))
        refine_film_dim = d_hidden if (self.film and self.film_per_block) else None
        self.refine = _HorizonRefine(
            d_hidden, kernel_size=kernel_size, dilations=dilations, film_cond_dim=refine_film_dim
        )

        if self.film and not self.film_per_block:
            # Single top-of-stack FiLM. Zero-init gives gamma = beta = 0, i.e. an identity FiLM at
            # construction. Built only when FiLM is on and not per-block, so the per-block core
            # carries no dead generator.
            self.film_gen: Optional[nn.Linear] = nn.Linear(d_hidden, 2 * d_hidden)
            nn.init.zeros_(self.film_gen.weight)
            nn.init.zeros_(self.film_gen.bias)
        else:
            self.film_gen = None

        # Built only when asked for, following the same rule as ``film_gen``: a core at the default
        # holds no attention module, so its state dict, its parameter count and its forward are the
        # ones it had before this knob existed.
        self.attention_blocks = max(0, int(attention_blocks))
        self.attention_heads = int(attention_heads)
        self.attention: Optional[nn.ModuleList]
        if self.attention_blocks > 0:
            if self.attention_heads < 1 or self.d_hidden % self.attention_heads != 0:
                raise ValueError(
                    f"attention_heads={self.attention_heads} must be a positive divisor of "
                    f"d_hidden={self.d_hidden}; the horizon tokens are split evenly across heads "
                    f"and a remainder would silently drop channels"
                )
            self.attention = nn.ModuleList(
                _HorizonSelfAttention(self.d_hidden, self.attention_heads)
                for _ in range(self.attention_blocks)
            )
        else:
            self.attention = None

        self.out_norm = nn.LayerNorm(d_hidden)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Expand a per-step state over the horizon and refine it.

        Args:
            h: Projected decoder state ``(B, T, d_hidden)``.

        Returns:
            Horizon-expanded features ``(B, T, H_d, d_hidden)``.
        """
        batch, seq_len, d_hidden = h.shape
        horizon = self.horizon

        feat = h.unsqueeze(2).expand(-1, -1, horizon, -1)
        feat = feat + self.horizon_embedding[None, None, :, :]
        if self.film_gen is not None:
            # Single top-of-stack FiLM path (only when not per-block).
            gamma, beta = self.film_gen(h).chunk(2, dim=-1)
            feat = feat * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]

        skip = feat
        # Fold (B, T) into the batch: each anchor's horizon is refined independently.
        flat = feat.reshape(batch * seq_len, horizon, d_hidden).transpose(1, 2).contiguous()
        # Per-block FiLM reads the projected state h -- the only latent entry point -- so each
        # block's modulation is a function of z alone, and the decoder still consumes nothing else.
        cond = h.reshape(batch * seq_len, d_hidden) if self.film_per_block else None
        flat = self.refine(flat, cond)

        # Back to channels-last, with the anchors still folded into the batch: the attention mixes
        # the horizon tokens of one anchor and can no more reach across anchors than the
        # convolutions above can. Each block adds its own residual; the skip and the output norm
        # below are untouched, so the attention is a strict addition to the refine path.
        feat = flat.transpose(1, 2)
        if self.attention is not None:
            for block in self.attention:
                feat = block(feat)

        feat = feat.reshape(batch, seq_len, horizon, d_hidden)
        return self.out_norm(feat + skip)


class BaselineFutureDecoder(nn.Module):
    r"""Forecast future target features from the target-only decoder state.

    Produces $\hat{Y}^{base}$ and a heteroscedastic ``logvar_base``. This is the branch that must
    be good on its own: the loss trains it as a full forecaster precisely so the residual branch
    cannot win by re-deriving what it already knows.
    """

    #: Declared so the optional persistence weight types as its own class rather than as the
    #: ``Tensor | Module`` ``nn.Module.__getattr__`` otherwise gives it. ``None`` on a decoder built
    #: without the residual, matching the convention every other optional term in this file uses.
    persistence_weight: Optional[nn.Parameter]

    def __init__(
        self,
        core: HorizonDecoderCore,
        d_model: int = 128,
        out_channels: int = 109,
        d_hidden: int = 128,
        dropout: float = 0.1,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        persistence_residual: bool = False,
    ) -> None:
        r"""Initialize the baseline decoder.

        Args:
            core: The shared horizon core. Passed in rather than constructed so both decoders
                provably hold the same one. Its ``horizon`` is also where the persistence weight's
                first axis comes from, so the weight cannot be sized against a horizon the core
                does not emit.
            d_model: Width of the incoming decoder state.
            out_channels: Number of target feature channels $C$.
            d_hidden: Decoder hidden width.
            dropout: Dropout inside the projection MLP.
            logvar_clamp: ``(lo, hi)`` effective range of the log-variance bound.
            persistence_residual: Build the target-only persistence term, so the mean becomes
                $\mu_{\tau,c} = w_{\tau,c}\, y_{t,c} + f_\theta(\cdot)_{\tau,c}$ and
                :meth:`forward` requires the anchor's target vector. ``False`` -- the default --
                builds **no** parameter at all, so the decoder is bitwise the one built before this
                keyword existed rather than one carrying an inert tensor.
        """
        super().__init__()
        self.core = core
        self.out_channels = int(out_channels)
        self.d_hidden = int(d_hidden)
        self.logvar_clamp = logvar_clamp

        self.proj = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_hidden, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.mean_head = nn.Linear(d_hidden, out_channels)
        self.logvar_head = nn.Linear(d_hidden, out_channels)

        # A raw nn.Parameter, and that is what makes the seeded decay survive: the generic
        # `initialization` pass fills Linear and Conv1d weights and re-inits LayerNorm, and ignores
        # loose parameters -- the same reason the horizon attention's residual gain and the
        # attention's `lag_score_bias` keep the values their constructors gave them. A Linear here
        # would be xavier-refilled after construction and the decay would silently not exist.
        #
        # Per (tau, c) rather than per tau: how long a coefficient persists is a property of the
        # channel's own filter, and the objective's gradient is the cheapest estimate of it.
        if persistence_residual:
            steps = torch.arange(self.core.horizon, dtype=torch.float32)
            decay = torch.pow(0.5, steps / PERSISTENCE_DECAY_HALFLIFE)
            self.persistence_weight = nn.Parameter(
                decay[:, None].repeat(1, self.out_channels)
            )
        else:
            self.persistence_weight = None

    def forward(
        self,
        decoder_state: torch.Tensor,
        persistence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forecast the horizon from the target-only state.

        Args:
            decoder_state: Target-only conditioning ``(B, T, d_model)``.
            persistence: The target's own value at each anchor ``(B, T, C)`` -- required exactly
                when the decoder was built with ``persistence_residual`` and refused otherwise. The
                caller owns what it is; this module owns only the weighting. It enters the **mean**
                head's output and nothing else, so the log-variance path is the one it was without
                the residual.

        Returns:
            ``(mu_base, logvar_base)``, both ``(B, T, H_d, C)``.

        Raises:
            ValueError: If the decoder and the call disagree about whether there is a persistence
                input. Both directions are refused rather than tolerated: a decoder built with the
                weight and called without one would leave that parameter out of the graph on that
                step -- the ``find_unused_parameters=False`` hazard -- and a tensor handed to a
                decoder that cannot use it would be silently discarded while the configuration said
                the residual was on.
        """
        if (self.persistence_weight is None) != (persistence is None):
            built = "with" if self.persistence_weight is not None else "without"
            called = "with" if persistence is not None else "without"
            raise ValueError(
                f"the decoder was built {built} a persistence residual and was called {called} a "
                f"persistence input. The two are one decision: persistence_residual builds the "
                f"weight and the forward that supplies it, and half of that is a model whose mean "
                f"forecast is not the one its configuration describes."
            )
        h = self.proj(decoder_state)
        feat = self.core.decode(h)
        mu_base = self.mean_head(feat)
        if self.persistence_weight is not None and persistence is not None:
            # (B, T, 1, C) against (H_d, C): the anchor's own vector, weighted per horizon step.
            mu_base = mu_base + self.persistence_weight * persistence[..., None, :]
        logvar_base = smooth_bound(self.logvar_head(feat), *self.logvar_clamp)
        return mu_base, logvar_base


class ResidualFutureDecoder(nn.Module):
    r"""Forecast the source-driven correction $\Delta\hat{Y}^{src}$.

    Consumes the target-only decoder state *and* the latent $z$. Its mean head is
    zero-initialised, so at init $\Delta\hat{Y}^{src} \approx 0$ and the full forecast equals the
    baseline exactly. The model then learns to diverge only where the latent carries genuinely
    incremental source information.

    Shares its horizon core with the baseline decoder, so the correction lives in the same
    representation space as the thing it corrects.
    """

    def __init__(
        self,
        core: HorizonDecoderCore,
        d_model: int = 128,
        d_z: int = 24,
        out_channels: int = 109,
        d_hidden: int = 128,
        dropout: float = 0.1,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
    ) -> None:
        """Initialize the residual decoder.

        Args:
            core: The shared horizon core.
            d_model: Width of the incoming decoder state.
            d_z: Latent dimensionality.
            out_channels: Number of target feature channels $C$.
            d_hidden: Decoder hidden width.
            dropout: Dropout inside the projection MLP.
            logvar_clamp: ``(lo, hi)`` effective range of the log-variance bound.
        """
        super().__init__()
        self.core = core
        self.out_channels = int(out_channels)
        self.d_hidden = int(d_hidden)
        self.logvar_clamp = logvar_clamp

        in_dim = d_model + d_z
        self.proj = ResidualMLP(
            input_dim=in_dim,
            hidden_dims=geometric_schedule(in_dim, d_hidden, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.mean_head = nn.Linear(d_hidden, out_channels)
        self.logvar_head = nn.Linear(d_hidden, out_channels)

    def forward(
        self, decoder_state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forecast the correction from the state and the latent.

        Args:
            decoder_state: Target-only conditioning ``(B, T, d_model)``.
            z: Sampled latent ``(B, T, d_z)``.

        Returns:
            ``(delta_mu_src, logvar_full)``, both ``(B, T, H_d, C)``.
        """
        h = self.proj(torch.cat([decoder_state, z], dim=-1))
        feat = self.core.decode(h)
        delta_mu_src = self.mean_head(feat)
        logvar_full = smooth_bound(self.logvar_head(feat), *self.logvar_clamp)
        return delta_mu_src, logvar_full
