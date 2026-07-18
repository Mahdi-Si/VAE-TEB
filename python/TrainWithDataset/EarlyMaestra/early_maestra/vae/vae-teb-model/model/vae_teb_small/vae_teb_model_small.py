import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any, Sequence

import math
import copy
from typing import List

from model.model_utils import *

# Logging is configured once per run by GraphModelBase.setup_config(), never at
# import time: setup_logging() reconfigures loguru process-wide, so importing this
# module after setup_config() used to tear down the run's sinks and redirect the
# whole run into a stray my_service.log in the working directory.

from loguru import logger as log
import logging as std_logging

try:
    from torch._dynamo.eval_frame import OptimizedModule as _TorchOptimizedModule
except Exception:
    _TorchOptimizedModule = tuple()



# -----------------------------------------------------------------------------
# Shape conventions used throughout this module
# -----------------------------------------------------------------------------
# B  = batch size
# T  = decimated sequence length (typically 300 for 20 min @ 4 Hz with /16)
# H  = forecasting horizon in decimated steps (e.g., 30 -> 2 min)
# L  = generic latent sequence length (T for reconstruction, H for forecasting)
# C  = feature channels
#   C_st = 43  (FHR scattering channels)
#   C_ph = 44  (FHR phase channels)
#   C_x  = 130 (UP+FHR cross-phase channels)
# D  = latent/hidden dimensionality (typically 16)
# R  = raw length at 4 Hz (4800 for 20 min). Relationship: R = L * s, s=16.
# s  = decimation factor (default 16)
# -----------------------------------------------------------------------------

def geometric_schedule(
    input_size: int,
    output_size: int,
    n_hidden: int,
    *,
    round_fn=round
) -> List[int]:
    steps = n_hidden + 1
    r = (output_size / input_size) ** (1 / steps)

    sizes = [input_size]
    current_r = r
    for _ in range(n_hidden):
        sizes.append(int(round_fn(input_size * current_r)))
        current_r *= r
    sizes.append(output_size)

    return tuple(sizes[1:])

def initialization(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for param_name, param in module.named_parameters():
                if "weight_ih" in param_name or "weight_hh" in param_name:
                    nn.init.orthogonal_(param)
                elif "bias" in param_name:
                    nn.init.zeros_(param)
                    if "bias_hh" in param_name:
                        hidden_size = module.hidden_size
                        param.data[hidden_size : 2 * hidden_size].fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class CausalConv1d(nn.Module):
    """
    1D convolution with explicit left-only padding to enforce causality.

    Shapes
    - Input:  x  (B, Cin, L)
    - Output: y  (B, Cout, Lout)
        Lout = floor((L + left_padding - dilation*(k-1) - 1)/stride + 1)
        With stride=1 and our left-padding choice, Lout == L.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super(CausalConv1d, self).__init__()
        # left padding for causal convolution
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Cin, L)
        Returns:
            (B, Cout, Lout)
        """
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel_size, dilation=dilation)
        self.n1 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.c2 = CausalConv1d(ch, ch, kernel_size, dilation=dilation)
        self.n2 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # (B,C,T)
        y = self.c1(x); y = F.gelu(self.n1(y))
        y = self.c2(y); y = self.n2(y)
        y = self.drop(y)
        return F.gelu(x + y)


class CausalMultiChannelConvBlock(nn.Module):
    """
    Causal version of MultiChannelConvBlock that ensures no future information leaks.
    Uses causal padding instead of reflection padding and supports upsampling.

    Shapes
    - Input:  x  (B, Cin, L)
    - Output: y  (B, Cout, L') where L' depends on stride and optional upsampling.
        If up_sampling is False and stride=1, L' == L.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        groups: int = 1,
        filter_size: int = 3,
        up_sampling: bool = False,
        up_sample_scale: int = 2,
        activation: nn.Module = nn.ReLU,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        super(CausalMultiChannelConvBlock, self).__init__()

        self.up_sampling = up_sampling
        self.up_sample_scale = up_sample_scale
        # Ccausal padding (left padding only)
        self.left_padding = (filter_size - 1) * dilation
        # Pre-normalization
        self.pre_norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        # Main convolution layer
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=filter_size,
            groups=groups,
            bias=bias,
            padding=0,
            dilation=dilation,
            stride=stride
        )
        # Activation function
        self.act_fn = activation()

        # Residual projection: match input channels to output channels if different
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Cin, L)
        Returns:
            (B, Cout, L') with strict causality
        """

        residual = x
        if self.up_sampling:
            x = F.interpolate(
                x,
                scale_factor=self.up_sample_scale,
                mode='linear',
                align_corners=False
            )

            residual = F.interpolate(
                residual,
                scale_factor=self.up_sample_scale,
                mode='linear',
                align_corners=False
            )

        x = self.pre_norm(x)  # (B, Cin, L or upsampled L)
        x = self.act_fn(x)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        output = self.conv(x)  # (B, Cout, L')

        if self.dropout is not None:
            output = self.dropout(output)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return output + residual


class MultiChannelConvBlock(nn.Module):
    """
    Non-causal multi-channel conv block (not used in prediction path).

    Shapes
    - Input:  (B, Cin, L)
    - Output: (B, Cout, L')
    """
    def __init__(
        self, in_channels=1, out_channels=1, groups=1,
        filter_size=3, up_sampling=False, up_sample_scale=2,
        tanh=False):
        super().__init__()
        self.tanh = tanh
        self.up_sampling = up_sampling
        self.up_scale = up_sample_scale
        self.filter_size = filter_size
        self.padding = (filter_size - 1) // 2

        # Pre-normalization for better gradient flow
        self.pre_norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=filter_size,
            groups=groups, bias=False)

    def forward(self, x):
        if self.up_sampling:
            x = F.interpolate(
                x, scale_factor=self.up_scale,
                mode='linear', align_corners=False)

        # Apply pre-normalization
        x = self.pre_norm(x)
        
        # Apply activation before convolution (PreAct)
        x = torch.tanh(x) if self.tanh else F.gelu(x)

        p = self.padding
        if p > 0:
            if x.shape[-1] <= p:
                # too-short fallback: still safe on CUDA
                x = F.pad(x, (p, p), mode='replicate')
            else:
                # manual reflect-pad:
                #   left  = reverse of x[..., 1 : p+1]
                #   right = reverse of x[..., -p-1 : -1]
                left  = x[..., 1 : p+1].flip(dims=[-1])
                right = x[..., -p-1 : -1].flip(dims=[-1])
                x = torch.cat([left, x, right], dim=-1)

        x = self.conv(x)
        return x


class ResidualMLP(nn.Module):
    """
    Residual MLP operating per time step with optional skip connection.

    Shapes
    - Input:  x  (B, L, Cin)
    - Output: y  (B, L, Cout) where Cout = hidden_dims[-1]
    """
    def __init__(
        self,
        input_dim,
        hidden_dims=(72, 68, 64),
        final_activation=True,
        activation=nn.GELU,
        use_skip_connection=True,
        use_input_layer_norm=True,
        dropout=0.1,
    ):
        super().__init__()
        self.final_activation = final_activation
        self.use_skip_connection = use_skip_connection
        self.dropout = dropout
        self.input_norm = nn.LayerNorm(input_dim) if use_input_layer_norm else nn.Identity()
        self.activation_factory = self._build_activation_factory(activation)

        # Sequence of (Linear → LayerNorm → Activation → Dropout) for intermediate layers
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(hidden_dims)):
            is_final_layer = i == len(hidden_dims) - 1
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if not is_final_layer or final_activation:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if not is_final_layer:
                layers.append(self.activation_factory())
                # Add dropout after activation for intermediate layers
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)

        final_dim = hidden_dims[-1]
        if self.use_skip_connection:
            if input_dim != final_dim:
                self.skip_proj = nn.Linear(input_dim, final_dim)
            else:
                self.skip_proj = nn.Identity()
        else:
            self.skip_proj = None

        self.final_act = self.activation_factory() if final_activation else None

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.body(x_norm)  # (B, L, Cout)
        if self.use_skip_connection:
            skip = self.skip_proj(x_norm)  # (B, L, Cout)
            y = y + skip
        if self.final_activation and self.final_act is not None:
            y = self.final_act(y)  # (B, L, Cout)
        return y  # (B, L, Cout)

    @staticmethod
    def _build_activation_factory(activation):
        if isinstance(activation, nn.Module):
            return lambda: copy.deepcopy(activation)
        if isinstance(activation, type) and issubclass(activation, nn.Module):
            return activation
        if callable(activation):
            return activation
        raise TypeError(
            "activation must be an nn.Module instance, nn.Module subclass, or callable returning an nn.Module."
        )



class TargetEncoder(nn.Module):
    """
    Encodes the target signal (y) to produce the parameters of the prior distribution p(z|y).

    Architecture: Parallel Conv-LSTM design for complementary temporal modeling

    The target signal is composed of two parts: scattering transform features (y_st) and
    phase harmonic features (y_ph). These are processed through:

    1. Modality-specific MLPs: Project inputs to latent dimension
    2. Modality-specific convs: Multi-scale local feature extraction (dilations: 1, 2, 4)
    3. PARALLEL temporal branches:
        - Conv branch: Additional dilated convs (dilations: 8, 16) for ~35s receptive field
        - LSTM branch: Bidirectional LSTM for global sequential dependencies (~75s)
    4. Fusion: Concatenate parallel outputs and project to final latent dimension

    This design provides complementary temporal modeling:
        - Conv: Local patterns (accelerations, sharp changes) with exponential receptive field
        - LSTM: Global context (baseline drift, long UC contractions, state transitions)

    Shapes:
    - Inputs:
        y_st: (B, T, 43) - Scattering transform features
        y_ph: (B, T, 44) - Phase harmonic features
    - Outputs:
        mu:             (B, T, D) - Mean of prior distribution
        prior_logvar:   (B, T, D) - Log-variance of prior
        cond_features:  (B, T, D) - Conditioning features for ConditionalEncoder
    """
    def __init__(
        self,
        input_dim_st: int = 43,
        input_dim_ph: int = 44,
        latent_dim: int = 8,
        lstm_hidden_dim: int = 32,
        lstm_num_layers: int = 4,
        use_bidirectional_lstm: bool = False,
        activation: nn.Module = nn.GELU,
        conv_dropout: float = 0.1,
        lstm_dropout: float = 0.1,
    ):
        super(TargetEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_bidirectional = use_bidirectional_lstm
        self.activation = activation

        combined_in = input_dim_st + input_dim_ph
        self.mlp_combined_1 = ResidualMLP(
            input_dim=combined_in,
            hidden_dims=geometric_schedule(combined_in, combined_in, 4),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.GELU
        )
        
        self.mlp_combined_2 = ResidualMLP(
            input_dim=87,
            hidden_dims=geometric_schedule(87, 16, 6),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.GELU
        )
        
        self.mlp_combined_skip_proj_1 = nn.Linear(combined_in, 16)
        self.mlp_combined_skip_proj_2 = nn.Linear(combined_in, 16)

        self.conv_1 = CausalMultiChannelConvBlock(
            in_channels=16, out_channels=16, filter_size=3, dilation=1, dropout=conv_dropout
        )
        self.conv_2 = CausalMultiChannelConvBlock(
            in_channels=16, out_channels=16, filter_size=7, dilation=1, dropout=conv_dropout
        )
        self.conv_3 = CausalMultiChannelConvBlock(
            in_channels=16, out_channels=16, filter_size=11, dilation=1, dropout=conv_dropout
        )
        self.stack_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.stack_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.fused_norm = nn.LayerNorm(16)


        self.lstm_temporal = nn.LSTM(
            input_size=16,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=use_bidirectional_lstm,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_dim * (2 if use_bidirectional_lstm else 1)
        self.lstm_temporal_norm = nn.LayerNorm(lstm_output_dim)

        parallel_concat_dim =  lstm_output_dim + 16

        self.fusion = ResidualMLP(
            input_dim=parallel_concat_dim,
            hidden_dims=geometric_schedule(parallel_concat_dim, 16, 5),
            final_activation=True,
            activation=nn.GELU,
            use_skip_connection=True
        )

        self.mu_layer = ResidualMLP(
            input_dim=16,
            hidden_dims=geometric_schedule(16, latent_dim, 4),
            final_activation=False,
            activation=nn.GELU
        )

        self.prior_logvar_layer = ResidualMLP(
            input_dim=16,
            hidden_dims=geometric_schedule(16, latent_dim, 4),
            final_activation=False,
            activation=nn.GELU
        )

        self.conditioning_layer = ResidualMLP(
            input_dim=16,
            hidden_dims=geometric_schedule(16, latent_dim, 4),
            final_activation=False,
            activation=nn.GELU
        )

    def forward(
        self,
        scattering_input: torch.Tensor,
        phase_harmonic_input: torch.Tensor,
        return_hidden: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Forward pass through the parallel Conv-LSTM encoder.

        Processing stages:
            1. Modality-specific MLPs project inputs to latent dimension
            2. Modality-specific dilated conv stacks (RF ≈ 10.75s)
            3. Concatenate scattering and phase features
            4. PARALLEL temporal branches:
                - Conv branch: Extended dilated convs (RF ≈ 35s)
                - LSTM branch: Bidirectional LSTM (RF = full 75s)
            5. Fusion: Concatenate and project parallel outputs
            6. Output heads: Generate mu, logvar, and conditioning features

        Args:
            scattering_input (torch.Tensor): Scattering transform features, shape $(B, T, 43)$.
            phase_harmonic_input (torch.Tensor): Phase harmonic features, shape $(B, T, 44)$.
            return_hidden (bool, optional): Whether to return intermediate hidden states for
                debugging/analysis. Defaults to False.

        Returns:
            Tuple containing:
                - mu (torch.Tensor): Mean of prior distribution $p(z|y)$, shape $(B, T, D)$.
                - prior_logvar (torch.Tensor): Log-variance of prior, shape $(B, T, D)$.
                - cond_features (torch.Tensor): Conditioning features for ConditionalEncoder, shape $(B, T, D)$.
                - hidden_states (Dict[str, torch.Tensor], optional): Intermediate activations if return_hidden=True.
        """
        hidden_states = {} if return_hidden else None

        combined = torch.cat([scattering_input, phase_harmonic_input], dim=-1)  # (B, T, 87)
        x_linear = self.mlp_combined_1(combined)
        x_linear_ = self.mlp_combined_2(x_linear + combined)  # (B, T, 32)
        x_linear = x_linear_ + self.mlp_combined_skip_proj_1(combined) + self.mlp_combined_skip_proj_2(x_linear)  # (B, T, 32)
        
        if return_hidden:
            hidden_states["combined_mlp"] = x_linear

        # === STAGE 2 (new): Single causal conv stack ===
        x_conv = x_linear.transpose(1, 2)  # (B, 32, T)
        conv_1 = self.conv_1(x_conv)
        conv_2 = self.conv_2(conv_1)
        conv_2 = conv_2 + self.stack_skip_norm_1(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_3 = conv_3 + self.stack_skip_norm_2(conv_2)
        conv_output = self.fused_norm(conv_3.transpose(1, 2).contiguous() + x_linear)  # (B, T, 32)
        if return_hidden:
            hidden_states["conv_stack_output"] = conv_output

        lstm_temporal, (hidden, cell) = self.lstm_temporal(x_linear)
        lstm_temporal = self.lstm_temporal_norm(lstm_temporal)

        if return_hidden:
            hidden_states["lstm_temporal"] = lstm_temporal
            hidden_states["lstm_hidden"] = hidden
            hidden_states["lstm_cell"] = cell


        parallel_concat = torch.cat([conv_output, lstm_temporal], dim=-1)
        fused = self.fusion(parallel_concat)  # (B, T, 32)

        if return_hidden:
            hidden_states["parallel_concat"] = parallel_concat
            hidden_states["fused"] = fused

        mu = self.mu_layer(fused)  # (B, T, D)
        prior_logvar = self.prior_logvar_layer(fused)  # (B, T, D)
        conditioning_features = self.conditioning_layer(fused)  # (B, T, D)

        prior_logvar = torch.clamp(prior_logvar, min=-10, max=10)

        if return_hidden:
            hidden_states["mu"] = mu
            hidden_states["prior_logvar"] = prior_logvar
            hidden_states["conditioning_features"] = conditioning_features
            return mu, prior_logvar, conditioning_features, hidden_states

        return mu, prior_logvar, conditioning_features

    def get_encoder_features(
        self, scattering_input: torch.Tensor, phase_harmonic_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract encoder features without variational sampling.
        Useful for analysis and visualization.
        """
        with torch.no_grad():
            mu, _, _ = self.forward(scattering_input, phase_harmonic_input)
            return mu


class SourceEncoder(nn.Module):
    """
    Encodes the source signal (x) to produce a deterministic latent representation h_x.

    Architecture: Parallel Conv-LSTM design for complementary temporal modeling

    This encoder processes the source signal features (cross-phase UP+FHR) through:

    1. Initial MLP: Project 130-channel input to 32D
    2. Dilated conv stack: Multi-scale local feature extraction (dilations: 1, 2, 4)
    3. PARALLEL temporal branches:
        - Conv branch: Additional dilated convs (dilations: 8, 16) for ~35s receptive field
        - LSTM branch: Unidirectional LSTM for global sequential dependencies (~75s)
    4. Fusion: Concatenate parallel outputs and project to final latent dimension
    5. Output: Deterministic latent vector h_x (used to condition posterior)

    Note: Unlike TargetEncoder, this outputs only a deterministic representation (no logvar).
    The output is named `mu_x` for consistency, but it is NOT a mean of a distribution.

    This design provides complementary temporal modeling:
        - Conv: Local UC patterns (contractions, intensity changes) with exponential receptive field
        - LSTM: Global context (UC history, state transitions, long-term coupling)

    Shapes:
    - Input:  x_ph (B, T, 130) - Cross-phase harmonic features (UP + FHR)
    - Output: mu_x (B, T, D)   - Deterministic source encoding
    """

    def __init__(
        self,
        input_channels: int = 130,
        latent_dim: int = 8,
        lstm_hidden_dim: int = 32,  # Reduced from 128 (parallel design needs less)
        lstm_num_layers: int = 4,   # Reduced from 4 (parallel design needs less)
        conv_dropout: float = 0.1,
        lstm_dropout: float = 0.1,
    ):
        super(SourceEncoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # === STAGE 1: Input MLP ===
        self.mlp = ResidualMLP(
            input_dim=130,
            hidden_dims=geometric_schedule(130, 16, 6),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.GELU
        )

        self.conv_1 = CausalMultiChannelConvBlock(
            in_channels=16, out_channels=16, filter_size=3, dilation=1, dropout=conv_dropout
        )
        self.conv_2 = CausalMultiChannelConvBlock(
            in_channels=16, out_channels=16, filter_size=5, dilation=1, dropout=conv_dropout
        )
        self.conv_3 = CausalMultiChannelConvBlock(
            in_channels=16, out_channels=16, filter_size=11, dilation=1, dropout=conv_dropout
        )

        self.source_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.source_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)

        self.fused_norm = nn.LayerNorm(16)
        self.lstm_temporal = nn.LSTM(
            input_size=16,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,  # Keep unidirectional for source
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_temporal_norm = nn.LayerNorm(lstm_hidden_dim)

        parallel_concat_dim = 16 + lstm_hidden_dim

        self.fusion = ResidualMLP(
            input_dim=parallel_concat_dim,
            hidden_dims=geometric_schedule(parallel_concat_dim, 16, 5),
            final_activation=True,
            activation=nn.GELU,
            use_skip_connection=True
        )

        self.mu_layer = ResidualMLP(
            input_dim=16,
            hidden_dims=geometric_schedule(16, latent_dim, 4),
            final_activation=False,
            activation=nn.GELU
        )

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through the parallel Conv-LSTM encoder.

        Processing stages:
            1. Initial MLP projects 130-channel input to 32D
            2. Dilated conv stack (RF ≈ 10.75s) with dilations [1, 2, 4]
            3. PARALLEL temporal branches:
                - Conv branch: Extended dilated convs (RF ≈ 35s) with dilations [8, 16]
                - LSTM branch: Unidirectional LSTM (RF = full 75s)
            4. Fusion: Concatenate and project parallel outputs
            5. Output head: Generate deterministic mu_x

        Args:
            x (torch.Tensor): Cross-phase harmonic features (UP+FHR), shape $(B, T, 130)$.
            return_intermediate (bool, optional): Whether to return intermediate activations
                for debugging/analysis. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
                - mu_x (torch.Tensor): Deterministic source encoding, shape $(B, T, D)$.
                - intermediates (Dict[str, torch.Tensor], optional): Intermediate activations
                    if return_intermediate=True.
        """
        intermediates = {} if return_intermediate else None
        x_linear = self.mlp(x)  # (B, T, 32)

        if return_intermediate:
            intermediates["mlp_output"] = x_linear
        x_conv = x_linear.transpose(1, 2)  # (B, 32, T)

        conv_1 = self.conv_1(x_conv)
        conv_2 = self.conv_2(conv_1)
        skip_1_norm = self.source_skip_norm_1(conv_1)
        conv_2 = conv_2 + skip_1_norm

        conv_3 = self.conv_3(conv_2)
        skip_2_norm = self.source_skip_norm_2(conv_2)
        conv_3 = conv_3 + skip_2_norm

        conv_out = conv_3.transpose(1, 2).contiguous()  # (B, T, 32)
        conv_out = self.fused_norm(conv_out)

        if return_intermediate:
            intermediates["conv_stack_output"] = conv_out

        lstm_temporal, (hidden, cell) = self.lstm_temporal(x_linear)  # (B, T, 64)
        lstm_temporal = self.lstm_temporal_norm(lstm_temporal)

        if return_intermediate:
            intermediates["lstm_temporal"] = lstm_temporal
            intermediates["lstm_hidden"] = hidden
            intermediates["lstm_cell"] = cell


        parallel_concat = torch.cat([conv_out, lstm_temporal], dim=-1)  # (B, T, 96)
        fused = self.fusion(parallel_concat)  # (B, T, 32)

        if return_intermediate:
            intermediates["parallel_concat"] = parallel_concat
            intermediates["fused"] = fused

        mu_x = self.mu_layer(fused)  # (B, T, D)

        if return_intermediate:
            intermediates["mu_x"] = mu_x
            return mu_x, intermediates

        return mu_x

    def get_sequence_encoding(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Get encoding up to a specific timestep (causal encoding).
        Useful for incremental inference.

        Args:
            x: Input tensor from optimized dataloader (batch_size, seq_len=300, channels=130) - fhr_up_ph cross-phase features
            timestep: Timestep up to which to encode (inclusive)

        Returns:
            Encoding up to the specified timestep
        """
        timestep = min(timestep, x.size(1) - 1)
        mu = self.forward(x)
        return mu[:, : timestep + 1, :]


class ConditionalEncoder(nn.Module):
    """
    Implements the conditional encoder q(z | x, y) for the TEB framework.

    This module models the posterior distribution over the latent variable z, conditioned on both
    the source signal (x) and the target signal (y). It takes the latent representation of the
    source (h_x) and a conditioning feature from the target (c_y) as input. It outputs the
    parameters of the posterior distribution, which is a diagonal Gaussian.

    Shapes
    - Inputs: h_x (B, T, D), h_y (B, T, D)
    - Outputs: mu_post (B, T, D), logvar_post (B, T, D)
    """

    def __init__(self, dim_hx: int, dim_hy: int, dim_z: int):
        """
        Args:
            dim_hx: Dimensionality of the source encoder's output (h_x).
            dim_hy: Dimensionality of the target encoder's output (h_y).
            hidden_dims: Tuple of hidden layer sizes for the MLP.
            dim_z: Dimensionality of the latent variable z.
        """
        super().__init__()
        self.mlp = ResidualMLP(
            input_dim=dim_hx + dim_hy,
            hidden_dims=geometric_schedule(dim_hx + dim_hy, 8, 4),
            final_activation=True,
            use_skip_connection=True, 
            activation=nn.GELU,
        )

        self.fc_mu = ResidualMLP(
            input_dim=8,
            hidden_dims=geometric_schedule(8, 8, 4),
            final_activation=False,
            use_skip_connection=False, 
            activation=nn.GELU,
        )
        self.fc_logvar = ResidualMLP(
            input_dim=8,
            hidden_dims=geometric_schedule(8, 8, 4),
            final_activation=False,
            use_skip_connection=False, 
            activation=nn.GELU,
        )

    def forward(
        self, h_x: torch.Tensor, h_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the conditional encoder.

        Args:
            h_x: Latent representation from the source encoder.
                    Shape: (batch_size, sequence_length, dim_hx)
            h_y: Latent representation from the target encoder.
                    Shape: (batch_size, sequence_length, dim_hy)

        Returns:
            A tuple containing:
            - mu (torch.Tensor): The mean of the posterior distribution.
                                    Shape: (batch_size, sequence_length, dim_z)
            - logvar (torch.Tensor): The log-variance of the posterior distribution.
                                        Shape: (batch_size, sequence_length, dim_z)
        """
        h_combined = torch.cat([h_x, h_y], dim=-1)
        h_merged = self.mlp(h_combined)
        mu = self.fc_mu(h_merged)
        logvar = self.fc_logvar(h_merged)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        sequence_length: int = 300,
        target_length: int = 4800,
    ):
        """
        Args:
            latent_dim: Input latent dimension (default 32 for TEB model)
            sequence_length: Input sequence length (default 300)
            target_length: Target raw signal length (default 4800 = 20min at 4Hz)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.target_length = target_length

        self.feature_expansion = ResidualMLP(
                input_dim=latent_dim,
                hidden_dims=geometric_schedule(latent_dim, 32, 5),
                final_activation=True,
                use_skip_connection=True, 
                activation=nn.GELU,
        )
        
        self.lstm_temporal = ResidualMLP(
            input_dim=latent_dim,
            hidden_dims=geometric_schedule(latent_dim, 32, 5),
            final_activation=True,
            use_skip_connection=True, 
            activation=nn.GELU,
        )
        
        
        self.skip_z_expanded = nn.Linear(latent_dim, 64)
        self.temporal_fusion_layer_norm = nn.LayerNorm(64)
        
        self.st_ph_linear = ResidualMLP(
            input_dim=64,
            hidden_dims=geometric_schedule(64, 87, 5), 
            final_activation=False,
            activation=nn.GELU,
            use_skip_connection=True
        )

        self.upsample_1 = nn.ConvTranspose1d(87, 64, kernel_size=4, stride=2, padding=1)
        self.norm_1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.temporal_refine_1 = nn.Conv1d(64, 64, kernel_size=7, dilation=2, padding=6)

        self.upsample_2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.norm_2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.temporal_refine_2 = nn.Conv1d(32, 32, kernel_size=7, dilation=2, padding=6)

        self.upsample_3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.norm_3 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.temporal_refine_3 = nn.Conv1d(16, 16, kernel_size=7, dilation=2, padding=6)

        self.upsample_4 = nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1)
        self.norm_4 = nn.GroupNorm(num_groups=2, num_channels=8)
        self.temporal_refine_4 = nn.Conv1d(8, 8, kernel_size=7, dilation=2, padding=6)


        self.skip_to_x1 = nn.Linear(latent_dim, 64)  # Will be upsampled to 600
        self.skip_to_x2 = nn.Linear(latent_dim, 32)  # Will be upsampled to 1200
        self.skip_to_x3 = nn.Linear(latent_dim, 16)  # Will be upsampled to 2400


        self.refine_stack = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv1d(8, 8, kernel_size=3, dilation=2, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 4, kernel_size=3, dilation=4, padding=4),
            nn.GELU(),
        )
        self.final_conv = nn.Conv1d(4, 1, kernel_size=5, padding=2)

        self.signal_mu = nn.Conv1d(1, 1, kernel_size=1)
        self.signal_logvar = nn.Conv1d(1, 1, kernel_size=1)

    def forward(
        self,
        latent_z: torch.Tensor,
        first_st_ph_sample: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Progressive upsampling decoder with temporal refinement and multi-scale skip connections.

        Architecture improvements for high-fidelity FHR reconstruction:
        1. **Temporal refinement convolutions**: Dilated convs after each upsampling stage
           smooth transitions and maintain temporal coherence (RF ≈ 7s at each stage)
        2. **Multi-scale skip connections**: Direct paths from latent $z$ to intermediate
           resolutions for better gradient flow and information preservation
        3. **Dilated refinement stack**: Final refinement with dilations [1,2,4] provides
           larger receptive field (RF ≈ 6.5s) for capturing local FHR patterns

        Upsampling path: 300 → 600 → 1200 → 2400 → 4800 (16× total upsampling)

        Args:
            latent_z: Latent variables, shape $(B, L, D)$ where $L=300$, $D=16$

        Returns:
            Tuple containing:
            - linear_output: Auxiliary features, shape $(B, L, 87)$
            - raw_signal_mu: Raw signal reconstruction mean, shape $(B, 16L)$ = $(B, 4800)$
            - raw_signal_logvar: Raw signal reconstruction log-variance, shape $(B, 16L)$ = $(B, 4800)$
        """
        L = latent_z.size(1)
        z_expanded_linear = self.feature_expansion(latent_z)
        # z_expanded_lstm, (_, _) = self.lstm_temporal(latent_z)
        z_expanded_lstm = self.lstm_temporal(latent_z)
        z_expanded = self.skip_z_expanded(latent_z) + torch.cat([z_expanded_linear, z_expanded_lstm], dim=-1)  # (B, L, 87)
        z_expanded = self.temporal_fusion_layer_norm(z_expanded)  # (B, L, 64)
        st_ph_coefs = self.st_ph_linear(z_expanded)

        if first_st_ph_sample is not None:
            st_ph_coefs = torch.cat([first_st_ph_sample.unsqueeze(1), st_ph_coefs], dim=1)   # Force first frame to match input sample
            st_ph_coefs = st_ph_coefs[:, :-1, :]  # Remove last frame to maintain length L
        
        st_ph_coefs = st_ph_coefs.transpose(1, 2)       # (B, 128, L) for conv operations
        x1 = F.gelu(self.norm_1(self.upsample_1(st_ph_coefs)))      # (B, 64, 2L)
        x1 = F.gelu(self.temporal_refine_1(x1))                # (B, 64, 2L)
        skip1 = self.skip_to_x1(latent_z).transpose(1, 2)      # (B, 64, L)
        skip1 = F.interpolate(skip1, size=x1.size(2), mode='linear', align_corners=False)  # (B, 64, 2L)
        x1 = x1 + skip1

        x2 = F.gelu(self.norm_2(self.upsample_2(x1)))          # (B, 32, 4L)
        x2 = F.gelu(self.temporal_refine_2(x2))                # (B, 32, 4L)
        skip2 = self.skip_to_x2(latent_z).transpose(1, 2)      # (B, 32, L)
        skip2 = F.interpolate(skip2, size=x2.size(2), mode='linear', align_corners=False)  # (B, 32, 4L)
        x2 = x2 + skip2

        x3 = F.gelu(self.norm_3(self.upsample_3(x2)))          # (B, 16, 8L)
        x3 = F.gelu(self.temporal_refine_3(x3))                # (B, 16, 8L)
        skip3 = self.skip_to_x3(latent_z).transpose(1, 2)      # (B, 16, L)
        skip3 = F.interpolate(skip3, size=x3.size(2), mode='linear', align_corners=False)  # (B, 16, 8L)
        x3 = x3 + skip3

        x4 = F.gelu(self.norm_4(self.upsample_4(x3)))          # (B, 8, 16L)
        x4 = F.gelu(self.temporal_refine_4(x4))                # (B, 8, 16L)

        refined = self.refine_stack(x4)                         # (B, 4, 16L)
        features = self.final_conv(refined)                     # (B, 1, 16L)

        mu = self.signal_mu(features).squeeze(1)                # (B, 16L)
        logvar = self.signal_logvar(features).squeeze(1)        # (B, 16L)

        logvar = torch.clamp(logvar, min=-10, max=10)
        # st_ph_coefs: (B, L, 87) is the scattering transform coefficients and phase harmonics
        return st_ph_coefs.transpose(1, 2), mu, logvar

    @staticmethod
    def compute_loss(
        linear_output: torch.Tensor,
        raw_mu_predicted: torch.Tensor, 
        raw_logvar_predicted: torch.Tensor,
        target_fhr_st: torch.Tensor,
        target_fhr_ph: torch.Tensor,
        target_raw_signal: torch.Tensor,
        compute_st_mse: bool = True):
        """
        Compute auxiliary reconstruction MSE and raw-signal NLL losses.
        
        Args:
            linear_output: Output from linear layers (B, S, 87)
            raw_mu_predicted: Predicted raw signal mean (B, 4800)
            raw_logvar_predicted: Predicted raw signal log variance (B, 4800)
            target_fhr_st: Target scattering coefficients (B, S, 43)
            target_fhr_ph: Target phase coefficients (B, S, 44)
            target_raw_signal: Target raw signal (B, 4800)
            
        Returns:
            Dictionary containing individual loss components
        """
        device = raw_mu_predicted.device

        if target_raw_signal.dim() == 3 and target_raw_signal.size(-1) == 1:
            target_raw_signal = target_raw_signal.squeeze(-1)  # Remove channel dimension if present

        if (
            compute_st_mse
            and target_fhr_st.dim() == 3
            and target_fhr_ph.dim() == 3
            and target_fhr_st.shape[:2] == linear_output.shape[:2]
            and target_fhr_ph.shape[:2] == linear_output.shape[:2]
            and linear_output.shape[-1] == (target_fhr_st.shape[-1] + target_fhr_ph.shape[-1])
        ):
            stacked_target = torch.cat([target_fhr_st, target_fhr_ph], dim=-1)
            mse_loss = F.mse_loss(linear_output, stacked_target)
        else:
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)

        var = raw_logvar_predicted.exp()
        nll_loss = 0.5 * (raw_logvar_predicted + (target_raw_signal - raw_mu_predicted) ** 2 / var)
        nll_loss = nll_loss.mean()

        return {
            'mse_loss': mse_loss,
            'nll_loss': nll_loss,
            'total_decoder_loss': mse_loss + nll_loss
        }


class SeqVaeCore(nn.Module):
    """
    Core sequential VAE module (encoders + decoder) with reconstruction losses.
    """

    def __init__(
        self,
        sequence_length: int = 300,
        latent_dim_source: int = 8,
        latent_dim_target: int = 8,
        latent_dim_z: int = 8,
        decimation_factor: int = 16,
        warmup_period: int = 30,
        lstm_hidden_dim: int = 32,
        lstm_num_layers: int = 4,
        *,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.latent_dim_source = latent_dim_source
        self.latent_dim_target = latent_dim_target
        self.latent_dim_z = latent_dim_z
        self.decimation_factor = decimation_factor
        self.warmup_period = warmup_period

        self.source_encoder = SourceEncoder(
            latent_dim=latent_dim_source,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        self.target_encoder = TargetEncoder(
            latent_dim=latent_dim_target,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        self.conditional_encoder = ConditionalEncoder(
            dim_hx=latent_dim_source,
            dim_hy=latent_dim_target,
            dim_z=latent_dim_z,
        )
        self.decoder = Decoder(latent_dim=latent_dim_z, sequence_length=sequence_length)

        if init_weights:
            initialization(self)

    def forward(
        self,
        y_st: torch.Tensor,  # (B, T, 43)
        y_ph: torch.Tensor,  # (B, T, 44)
        x_ph: torch.Tensor,  # (B, T, 130)
        prediction_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete VAE (encode + decode).

        Performs the full TEB encoding and reconstruction:
            1. Encode source: $h_x = f_{source}(x_{ph})$
            2. Encode target prior: $\mu_y, \log\sigma^2_y = f_{target}(y_{st}, y_{ph})$
            3. Encode posterior: $\mu_{post}, \log\sigma^2_{post} = f_{cond}(h_x, c_{logvar})$
            4. Add residual: $\mu_{post} += \mu_y$ (TEB residual connection)
            5. Sample latent: $z \sim q(z|x,y)$
            6. Decode: $p(y|z)$ with Gaussian likelihood

        Args:
            prediction_mode:
            y_st (torch.Tensor): Target scattering features, shape $(B, T, 43)$ where
                $B$ is batch size, $T$ is sequence_length (300), and 43 is the number
                of scattering transform channels.
            y_ph (torch.Tensor): Target phase harmonic features, shape $(B, T, 44)$.
            x_ph (torch.Tensor): Source cross-phase harmonic features, shape $(B, T, 130)$.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - z: Sampled posterior latent, shape $(B, T, D_z)$.
                - linear_output: Intermediate decoder features, shape $(B, T, 87)$.
                - mu_pr: Predicted raw signal mean, shape $(B, 4800)$.
                - logvar_pr: Predicted raw signal log-variance, shape $(B, 4800)$.
                - mu_prior: Target prior mean, shape $(B, T, D_{target})$.
                - logvar_prior: Target prior log-variance, shape $(B, T, D_{target})$.
                - mu_post: Posterior mean after residual, shape $(B, T, D_z)$.
                - logvar_post: Posterior log-variance, shape $(B, T, D_z)$.
                - mu_next: None (legacy placeholder).
                - logvar_next: None (legacy placeholder).
                - next_step_indices: None (legacy placeholder).
        """
        mu_x = self.source_encoder(x_ph)
        mu_y, logvar_y_prior, c_logvar = self.target_encoder(y_st, y_ph)

        mu_post, logvar_post = self.conditional_encoder(mu_x, c_logvar)
        mu_post = mu_post + mu_y
        z = self.reparameterize(mu_post, logvar_post)
        if prediction_mode:
            first_coef = torch.cat([y_st, y_ph], -1)[:, 0, :]
            linear_output, mu_pr, logvar_pr = self.decoder(z, first_coef)
        else:
            linear_output, mu_pr, logvar_pr = self.decoder(z)

        return {
            "z": z,  # (B, T, latent_dim_z)
            "linear_output": linear_output,  # (B, T, 87)
            "mu_pr": mu_pr,  # (B, 4800)
            "logvar_pr": logvar_pr,  # (B, 4800)
            "mu_prior": mu_y,  # (B, T, latent_dim_target)
            "logvar_prior": logvar_y_prior,  # (B, T, latent_dim_target)
            "mu_post": mu_post,  # (B, T, latent_dim_z)
            "logvar_post": logvar_post,  # (B, T, latent_dim_z)
        }

    def encode_only(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        sample_z: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mu_x = self.source_encoder(x_ph)
        mu_y, logvar_y_prior, c_logvar = self.target_encoder(y_st, y_ph)
        mu_post, logvar_post = self.conditional_encoder(mu_x, c_logvar)
        mu_post = mu_post + mu_y
        z = self.reparameterize(mu_post, logvar_post) if sample_z else mu_post
        return {
            "mu_prior": mu_y,
            "logvar_prior": logvar_y_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
        }

    @staticmethod
    def measure_transfer_entropy(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        """Estimate transfer entropy via posterior/prior KL using the current core."""
        self.eval()
        with torch.no_grad():
            enc = self.encode_only(y_st=y_st, y_ph=y_ph, x_ph=x_ph, sample_z=True)
            kld = self._kld_loss(
                mu_prior=enc["mu_prior"],
                logvar_prior=enc["logvar_prior"],
                mu_post=enc["mu_post"],
                logvar_post=enc["logvar_post"],
                reduce_mean=reduce_mean,
            )
        return kld

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kld_loss(
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        *,
        reduce_mean: bool = True,
    ) -> torch.Tensor:
        kld = (
            logvar_prior
            - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
            - 1.0
        )
        kld = 0.5 * kld
        return kld.mean() if reduce_mean else kld.sum()

    @staticmethod
    def gaussian_nll(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        var = logvar.exp()
        nll = 0.5 * (logvar + (target - mu) ** 2 / var)
        if mask is not None:
            nll = nll * mask
            denom = mask.sum().clamp_min(1.0)
            return nll.sum() / denom
        return nll.mean()

    def _next_step_indices(self, device: torch.device) -> torch.Tensor:
        steps = max(0, self.sequence_length - 1)
        if steps == 0:
            return torch.zeros(0, device=device, dtype=torch.long)
        base = torch.arange(steps, device=device, dtype=torch.long) + 1
        return self.decimation_factor * base

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],  # Dict from forward()
        y_st: torch.Tensor,  # (B, T, 43)
        y_ph: torch.Tensor,  # (B, T, 44)
        y_raw: torch.Tensor,  # (B, 4800) or (B, 4800, 1)
        *,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction and regularization losses for VAE training.

        Computes the complete VAE objective with three loss components:
            1. MSE loss: Mean squared error between intermediate decoder features and
                concatenated scattering + phase harmonic inputs (auxiliary supervision).
            2. NLL loss: Gaussian negative log-likelihood for raw signal reconstruction
                $p(y_{raw} | z)$, allowing heteroscedastic noise modeling.
            3. KLD loss: Transfer entropy $\text{KL}(q(z|x,y) \| p(z|y))$ measuring
                information flow from source to latent representation.

        Total loss: $\mathcal{L} = \text{MSE} + \text{NLL} + \beta \cdot \text{KLD}$

        Args:
            forward_outputs (Dict[str, torch.Tensor]): Dictionary from forward() containing
                mu_pr, logvar_pr, mu_prior, logvar_prior, mu_post, logvar_post, and
                optionally linear_output.
            y_st (torch.Tensor): Target scattering features, shape $(B, T, 43)$.
            y_ph (torch.Tensor): Target phase harmonic features, shape $(B, T, 44)$.
            y_raw (torch.Tensor): Raw target signal for reconstruction, shape $(B, 4800)$
                or $(B, 4800, 1)$. If 3D with last dim=1, it will be squeezed to $(B, 4800)$.
            compute_kld_loss (bool, optional): Whether to compute and include KL divergence
                term. Set to False for deterministic autoencoders or ablation studies.
                Defaults to True.
            beta (float, optional): Weight for KL divergence term ($\beta$-VAE framework).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing all loss components (all scalars):
                - reconstruction_loss: MSE + NLL.
                - mse_loss: Auxiliary MSE loss on intermediate features.
                - nll_loss: Gaussian NLL for raw signal reconstruction.
                - kld_loss: Transfer entropy $\text{KL}(q(z|x,y) \| p(z|y))$.
                - total_loss: reconstruction_loss + $\beta$ * kld_loss.
                - total_decoder_loss: Alias for reconstruction_loss (backward compatibility).
        """
        if y_raw.dim() == 3 and y_raw.size(-1) == 1:
            y_raw = y_raw.squeeze(-1)

        device = y_raw.device

        mse_loss = torch.tensor(0.0, device=device)
        linear_output = forward_outputs.get("linear_output")
        if (
            linear_output is not None
            and linear_output.dim() == 3
            and y_st.shape[:2] == linear_output.shape[:2]
            and y_ph.shape[:2] == linear_output.shape[:2]
            and linear_output.shape[-1] == (y_st.shape[-1] + y_ph.shape[-1])
        ):
            stacked_target = torch.cat([y_st, y_ph], dim=-1)
            mse_loss = F.mse_loss(linear_output, stacked_target)

        mu_pr = forward_outputs.get("mu_pr")
        logvar_pr = forward_outputs.get("logvar_pr")
        if mu_pr is not None and logvar_pr is not None:
            var = logvar_pr.exp()
            nll_loss = 0.5 * (logvar_pr + (y_raw - mu_pr) ** 2 / var)
            nll_loss = nll_loss.mean()
        else:
            nll_loss = torch.tensor(0.0, device=device)

        kld_loss = torch.tensor(0.0, device=y_raw.device)
        if compute_kld_loss:
            kld_loss = self._kld_loss(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"],
                reduce_mean=True,
            )

        total_decoder = mse_loss + nll_loss
        total_loss = total_decoder + beta * kld_loss
        return {
            "reconstruction_loss": total_decoder,  # scalar: MSE + NLL
            "mse_loss": mse_loss,  # scalar: intermediate features MSE
            "nll_loss": nll_loss,  # scalar: Gaussian NLL for raw signal
            "kld_loss": kld_loss,  # scalar: KL(q(z|x,y) || p(z|y))
            "total_loss": total_loss,  # scalar: reconstruction_loss + β * kld_loss
            "total_decoder_loss": total_decoder,  # scalar: alias for reconstruction_loss
        }



if __name__ == "__main__":

    batch_size = 4
    seq_len = 300
    channels = 76
    prediction_horizon = 30
    warmup_period = 30

    y_st_input = torch.randn(batch_size, seq_len, 43)  # UPDATED: (B, L, C) format from optimized dataloader
    y_ph_input = torch.randn(batch_size, seq_len, 44)
    x_ph_input = torch.randn(batch_size, seq_len, 130)
    y_raw_input = torch.randn(
        batch_size, seq_len * 16
    )

    # target encoder test: -------------------------------------------------------
    # model = TargetEncoder(sequence_length=seq_len)
    
    # mu, logvar, logvar_c = model(scattering_input=y_st_input,
    #                                phase_harmonic_input=y_ph_input)

    # source encoder test: -------------------------------------------------------
    # model = SourceEncoder()
    # mu = model(x_ph_input)

    # conditional encoder test: --------------------------------------------------
    # model = ConditionalEncoder(32, 32, 32)
    # mu, logvar = model(
    #     torch.randn(batch_size, seq_len, 32),
    #     torch.randn(batch_size, seq_len, 32)
    # )

    # decoder test: --------------------------------------------------------------
    # Original decoder test
    # model = Decoder()
    # linear_output, mu, logvar = model(
    #     torch.randn(batch_size, seq_len, 32),
    # )
    # loss_dict = model.compute_loss(
    #     linear_output,
    #     mu, logvar,
    #     torch.randn(batch_size, seq_len, 43),  # target_fhr_st
    #     torch.randn(batch_size, seq_len, 44),  # target_fhr_ph
    #     torch.randn(batch_size, 4800)  # target_raw_signal
    # )
    
    # Core model test: --------------------------------------------------------------
    model = SeqVaeCore()
    outputs = model(
        y_st_input,
        y_ph_input,
        x_ph_input, 
    )
    loss_dict = model.compute_reconstruction_loss(
        forward_outputs=outputs,
        y_st=torch.randn(batch_size, seq_len, 43),
        y_ph=torch.randn(batch_size, seq_len, 44),
        y_raw=torch.randn(batch_size, 4800),
    )
    # ImprovedDecoder test: ------------------------------------------------------
    # model = ImprovedDecoder(latent_dim=32, sequence_length=seq_len, target_length=4800)
    # linear_output, mu, logvar = model(
    #     torch.randn(batch_size, seq_len, 32),
    # )
    # loss_dict = model.compute_loss(
    #     linear_output,
    #     mu, logvar,
    #     torch.randn(batch_size, seq_len, 43),  # target_fhr_st
    #     torch.randn(batch_size, seq_len, 44),  # target_fhr_ph
    #     torch.randn(batch_size, 4800)  # target_raw_signal
    # )
    # print(f"ImprovedDecoder - Linear: {linear_output.shape}, Mu: {mu.shape}, LogVar: {logvar.shape}")
    # print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test VAE model: ------------------------------------------------------------
    # Standard TEB model
    # model = SeqVaeTeb(
    #     input_channels=channels,
    #     sequence_length=seq_len,
    #     decimation_factor=16,
    #     warmup_period=warmup_period,
    #     use_improved_decoder=False
    # )
    
    # TEB model with ImprovedDecoder
    # model = SeqVaeTeb(
    #     input_channels=channels,
    #     sequence_length=seq_len,
    #     decimation_factor=16,
    #     warmup_period=warmup_period,
    #     use_improved_decoder=True  # Use research-backed decoder
    # )
    # forward_outputs = model(
    #     y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input
    # )
    # prd_x_mu = model.get_average_predictions(forward_outputs['mu_pr'])
    # prd_x_logvar = model.get_average_predictions(forward_outputs['logvar_pr'])
    # loss = model.compute_loss(forward_outputs, y_raw_input)
    # print('done')

