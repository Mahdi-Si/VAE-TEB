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
"""
from __future__ import annotations

from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound


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
    ) -> None:
        """Initialize the refine stack.

        Args:
            d_hidden: Channel width, held constant through the stack.
            kernel_size: Convolution kernel width.
            dilations: Dilation per block.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine along the horizon axis.

        Args:
            x: Input of shape ``(N, d_hidden, H_d)``.

        Returns:
            Output of the same shape.
        """
        for block in self.blocks:
            block = cast(nn.ModuleDict, block)
            y = block["conv"](x)
            y = F.gelu(block["norm"](y))
            x = x + y
        return x


class HorizonDecoderCore(nn.Module):
    r"""Shared horizon-refinement core, used by both future decoders.

    Holds the forecast-step embedding, optional FiLM modulation, the dilated refine stack and the
    output norm. Both decoders push their projected ``(B, T, d_hidden)`` state through the same
    :meth:`decode`, so the residual decoder operates as a correction *in the baseline's
    representation space*.

    FiLM modulates the horizon-expanded features with per-channel $(\gamma, \beta)$ generated
    from the projected state. Its generator is zero-initialised, so the core starts as an
    identity.

    Note this core's normalisers are deliberately *not* causalised. They pool over the forecast
    axis of a single anchor, not across input time: every step of that axis is predicted from the
    same $t$, so pooling over it cannot reach information the anchor did not already have.
    """

    def __init__(
        self,
        d_hidden: int = 128,
        horizon: int = 30,
        kernel_size: int = 3,
        depth: int = 2,
        film: bool = False,
    ) -> None:
        """Initialize the shared horizon core.

        Args:
            d_hidden: Decoder hidden width.
            horizon: Forecast horizon $H_d$.
            kernel_size: Horizon-convolution kernel width.
            depth: Number of dilated blocks; dilations are $1, 2, 4, \\dots$.
            film: Whether to apply FiLM conditioning per horizon step.
        """
        super().__init__()
        self.horizon = int(horizon)
        self.d_hidden = int(d_hidden)
        self.film = bool(film)

        # An internal forecast-step embedding: which step of the horizon this is, not what time
        # it was in the recording.
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, d_hidden))
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)

        depth = max(1, int(depth))
        dilations = tuple(2**index for index in range(depth))
        self.refine = _HorizonRefine(d_hidden, kernel_size=kernel_size, dilations=dilations)

        if self.film:
            # Zero-init gives gamma = beta = 0, i.e. an identity FiLM at init.
            self.film_gen: Optional[nn.Linear] = nn.Linear(d_hidden, 2 * d_hidden)
            nn.init.zeros_(self.film_gen.weight)
            nn.init.zeros_(self.film_gen.bias)
        else:
            self.film_gen = None
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
            gamma, beta = self.film_gen(h).chunk(2, dim=-1)
            feat = feat * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]

        skip = feat
        # Fold (B, T) into the batch: each anchor's horizon is refined independently.
        flat = feat.reshape(batch * seq_len, horizon, d_hidden).transpose(1, 2).contiguous()
        flat = self.refine(flat)
        feat = flat.transpose(1, 2).reshape(batch, seq_len, horizon, d_hidden)
        return self.out_norm(feat + skip)


class BaselineFutureDecoder(nn.Module):
    r"""Forecast future target features from the target-only decoder state.

    Produces $\hat{Y}^{base}$ and a heteroscedastic ``logvar_base``. This is the branch that must
    be good on its own: the loss trains it as a full forecaster precisely so the residual branch
    cannot win by re-deriving what it already knows.
    """

    def __init__(
        self,
        core: HorizonDecoderCore,
        d_model: int = 128,
        out_channels: int = 109,
        d_hidden: int = 128,
        dropout: float = 0.1,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
    ) -> None:
        """Initialize the baseline decoder.

        Args:
            core: The shared horizon core. Passed in rather than constructed so both decoders
                provably hold the same one.
            d_model: Width of the incoming decoder state.
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

    def forward(self, decoder_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forecast the horizon from the target-only state.

        Args:
            decoder_state: Target-only conditioning ``(B, T, d_model)``.

        Returns:
            ``(mu_base, logvar_base)``, both ``(B, T, H_d, C)``.
        """
        h = self.proj(decoder_state)
        feat = self.core.decode(h)
        mu_base = self.mean_head(feat)
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
