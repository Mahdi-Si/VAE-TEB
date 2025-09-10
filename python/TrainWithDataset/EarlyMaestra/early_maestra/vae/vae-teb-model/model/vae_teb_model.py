import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict

import math
from typing import List

from utils.custom_logger import setup_logging

setup_logging(
    log_to_file=True,
    log_to_console=True,
    file_path="my_service.log",
    file_level="DEBUG",
    console_level="INFO",
    rotation="100 MB",
    retention="14 days",
    compression="zip",
    serialize=False,   # True → JSON output
    backtrace=True,    # include full stack backtraces
    diagnose=False,    # include local vars in tracebacks
)

# Now all logging goes through Loguru
from loguru import logger as log
import logging as std_logging

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

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate left padding for causal convolution (more efficient)
        self.left_padding = (kernel_size - 1) * dilation

        # Use grouped convolution for efficiency when possible
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
            x: Input tensor (batch_size, channels, seq_len)
        Returns:
            Causal convolution output
        """
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))

        return self.conv(x)


class CausalMultiChannelConvBlock(nn.Module):
    """
    Causal version of MultiChannelConvBlock that ensures no future information leaks.
    Uses causal padding instead of reflection padding and supports upsampling.
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
        bias: bool = False
    ):
        super(CausalMultiChannelConvBlock, self).__init__()
        
        self.up_sampling = up_sampling
        self.up_sample_scale = up_sample_scale
        self.filter_size = filter_size
        self.activation = activation
        self.dilation = dilation
        self.stride = stride
        
        # Calculate causal padding (left padding only)
        self.left_padding = (filter_size - 1) * dilation
        
        # Pre-normalization (consistent with MultiChannelConvBlock)
        self.pre_norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        
        # Main convolution layer
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=filter_size,
            groups=groups,
            bias=bias,
            padding=0,  # We handle padding manually
            dilation=dilation,
            stride=stride
        )
            
        # Activation function
        self.act_fn = activation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, length)
        Returns:
            Causal convolution output with optional upsampling
        """
        # Apply upsampling first if requested
        if self.up_sampling:
            x = F.interpolate(
                x, 
                scale_factor=self.up_sample_scale, 
                mode='linear', 
                align_corners=False
            )
        
        # Apply pre-normalization (PreNorm architecture - consistent with MultiChannelConvBlock)
        x = self.pre_norm(x)
        
        # Apply activation before convolution (PreAct architecture)
        x = self.act_fn(x)
        
        # Apply causal padding (left padding only)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        
        # Apply convolution
        output = self.conv(x)
        
        return output

class MultiChannelConvBlock(nn.Module):
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
        x = torch.tanh(x) if self.tanh else F.relu(x)

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
    def __init__(
        self, input_dim, hidden_dims=(72, 68, 64), final_activation=True, activation=nn.ReLU, use_skip_connection=True
    ):
        super().__init__()
        # initial layer-norm on raw input
        self.input_norm = nn.LayerNorm(input_dim)
        self.final_activation = final_activation
        self.activation = activation
        self.use_skip_connection = use_skip_connection
        # build the sequence of (Linear → LayerNorm → activation → Dropout)
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(hidden_dims)):
            # For final layer, skip activation and layernorm if final_activation=False
            is_final_layer = (i == len(hidden_dims) - 1)
            if is_final_layer and not final_activation:
                layers += [
                    nn.Linear(dims[i], dims[i + 1]),
                ]
            elif is_final_layer and final_activation:
                layers += [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                ]
            else:
                layers += [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    self.activation(),
                ]
        self.body = nn.Sequential(*layers)

        # if input_dim ≠ final hidden_dims[-1], project it (only if using skip connection)
        final_dim = hidden_dims[-1]
        if self.use_skip_connection:
            if input_dim != final_dim:
                self.skip_proj = nn.Linear(input_dim, final_dim)
            else:
                self.skip_proj = nn.Identity()
        else:
            self.skip_proj = None

        # only norm, no activation after skip connection
        # self.post_norm = nn.LayerNorm(final_dim)
        # final activation applied before skip connection if needed
        self.final_act = self.activation() if final_activation else None

    def forward(self, x):
        # 1) normalize raw input
        x0 = self.input_norm(x)

        # 2) run through MLP body
        y = self.body(x0)

        # 3) apply final activation before skip connection if needed
        if self.final_activation:
            y = self.final_act(y)

        # 4) conditionally add skip connection
        if self.use_skip_connection:
            skip = self.skip_proj(x0)
            z = y + skip
        else:
            z = y

        # 5) only apply normalization, no activation after skip connection
        return z


class TargetEncoder(nn.Module):
    """
    Encodes the target signal (y) to produce the parameters of the prior distribution p(z|y).

    The target signal is composed of two parts: scattering transform features (y_st) and 
    phase harmonic features (y_ph). These are processed through parallel MLP and causal convolution
    stacks, fused, and then passed through an LSTM to produce the final representations.

    The encoder outputs the mean and a composite log-variance vector. This vector is
    later split into the log-variance for the prior and a conditioning
    feature used by the ConditionalEncoder.

    **Mathematical Formulation:**
    """
    def __init__(
        self,
        sequence_length: int = 300,
        latent_dim: int = 16,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 5,
        use_bidirectional_lstm: bool = False,
        activation: nn.Module = nn.GELU,
    ):
        super(TargetEncoder, self).__init__()

        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_bidirectional = use_bidirectional_lstm

        self.activation = activation
        
        self.mlp_scattering = ResidualMLP(
                input_dim=43,
                hidden_dims=geometric_schedule(43, 16, 4),
                final_activation=False,
                use_skip_connection=True,
                activation=nn.ReLU
                )

        self.mlp_phase = ResidualMLP(
            input_dim=44,
            hidden_dims=geometric_schedule(44, 16, 4),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.ReLU
            )

        self.conv_scattering_1 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=3, dilation=1)
        self.conv_scattering_2 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=5, dilation=1)
        self.conv_scattering_3 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=7, dilation=1)
        
        self.scatter_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.scatter_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)

        self.conv_phase_1 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=3, dilation=1)
        self.conv_phase_2 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=5, dilation=1)
        self.conv_phase_3 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=7, dilation=1)
        
        self.phase_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.phase_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        
        self.scatter_fused_norm = nn.LayerNorm(16)
        self.phase_fused_norm = nn.LayerNorm(16)
        
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim * (2 if use_bidirectional_lstm else 1))

        self.cross_modal_fusion = ResidualMLP(
            input_dim=16 * 2,
            hidden_dims=(32, 32, 32),
            final_activation=False,
            activation=nn.ReLU,
            use_skip_connection=True
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=use_bidirectional_lstm,
        )

        lstm_output_dim = lstm_hidden_dim * (2 if use_bidirectional_lstm else 1)

        self.pre_output = ResidualMLP(
            input_dim=lstm_output_dim,
            hidden_dims=geometric_schedule(lstm_output_dim, 32, 5),
            final_activation=True,
            activation=nn.ReLU
        )

        self.mu_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.ReLU
        )
        
        self.prior_logvar_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.ReLU
        )
        
        self.conditioning_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.ReLU
        )

    def forward(
        self,
        scattering_input: torch.Tensor,
        phase_harmonic_input: torch.Tensor,
        return_hidden: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Forward pass through the encoder.

        Args:
            scattering_input: Scattering transform features from optimized dataloader (batch_size, seq_len=300, channels=43)
            phase_harmonic_input: Phase harmonic features from optimized dataloader (batch_size, seq_len=300, channels=44)
            return_hidden: Whether to return intermediate hidden states

        Returns:
            mu: Mean of latent distribution (batch_size, seq_len, latent_dim)
            logvar: Log variance of latent distribution (batch_size, seq_len, 2*latent_dim)
            hidden_states: Dictionary of intermediate states (if return_hidden=True)
        """
        hidden_states = {} if return_hidden else None

        scatter_linear = self.mlp_scattering(scattering_input)
        phase_linear = self.mlp_phase(phase_harmonic_input)
        
        if return_hidden:
            hidden_states["scattering_reduced"] = scatter_linear
            hidden_states["phase_reduced"] = phase_linear

        x_scatter = scatter_linear.transpose(1, 2)  # (B, C, L)
        
        scatter_conv_1 = self.conv_scattering_1(x_scatter)
        scatter_conv_2 = self.conv_scattering_2(scatter_conv_1)
        skip_1_norm = self.scatter_skip_norm_1(scatter_conv_1)
        scatter_conv_2 = scatter_conv_2 + skip_1_norm  # Normalized skip connection
        
        scatter_conv_3 = self.conv_scattering_3(scatter_conv_2)
        skip_2_norm = self.scatter_skip_norm_2(scatter_conv_2)
        scatter_conv_3 = scatter_conv_3 + skip_2_norm  # Normalized skip connection
        
        scatter_conv = scatter_conv_3.transpose(1, 2).contiguous()  # Back to (B, L, C)
        scatter_conv = self.scatter_fused_norm(scatter_conv)
        del scatter_linear, x_scatter, scatter_conv_1, scatter_conv_2, scatter_conv_3, skip_1_norm, skip_2_norm

        x_phase = phase_linear.transpose(1, 2)  # (B, C, L)
        
        phase_conv_1 = self.conv_phase_1(x_phase)
        
        phase_conv_2 = self.conv_phase_2(phase_conv_1)
        skip_1_norm = self.phase_skip_norm_1(phase_conv_1)
        phase_conv_2 = phase_conv_2 + skip_1_norm  # Normalized skip connection
        
        phase_conv_3 = self.conv_phase_3(phase_conv_2)
        skip_2_norm = self.phase_skip_norm_2(phase_conv_2)
        phase_conv_3 = phase_conv_3 + skip_2_norm  # Normalized skip connection
        
        phase_conv = phase_conv_3.transpose(1, 2).contiguous()  # Back to (B, L, C)
        phase_conv = self.phase_fused_norm(phase_conv)
        del phase_linear, x_phase, phase_conv_1, phase_conv_2, phase_conv_3, skip_1_norm, skip_2_norm

        combined = torch.cat([scatter_conv, phase_conv], dim=-1)
        del scatter_conv, phase_conv
        x = self.cross_modal_fusion(combined)
        del combined

        x, (hidden, cell) = self.lstm(x)  # (batch, length, channel)
        x = self.lstm_norm(x)

        if return_hidden:
            hidden_states["lstm_out"] = x
            hidden_states["lstm_hidden"] = hidden
            hidden_states["lstm_cell"] = cell
        else:
            del hidden, cell  # Clean up LSTM states if not needed

        x = self.pre_output(x)

        mu = self.mu_layer(x)
        prior_logvar = self.prior_logvar_layer(x)
        conditioning_features = self.conditioning_layer(x)

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

    This encoder processes the source signal features (x_ph) through an MLP, a stack of causal
    convolutions, and an LSTM to capture the temporal dependencies in the source signal.
    The output is a deterministic vector, which is used to condition the posterior distribution.
    In the code, this output is named `mu_x` for consistency, but it is not the mean of a
    distribution.
    """

    def __init__(
        self,
        input_channels: int = 130,
        sequence_length: int = 300,
        latent_dim: int = 16,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 4,
    ):
        super(SourceEncoder, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.mlp = ResidualMLP(
            input_dim=130,
            hidden_dims=geometric_schedule(130, 32, 5),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.ReLU
            )
        
        self.conv_1 = CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=3, dilation=1)
        self.conv_2 = CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=5, dilation=1)
        self.conv_3 = CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=7, dilation=1)
        
        self.source_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 32), num_channels=32)
        self.source_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 32), num_channels=32)
        
        self.fused_norm = nn.LayerNorm(32)

        self.linear_after_conv = ResidualMLP(
            input_dim=32, 
            hidden_dims=(32, 32),
            final_activation=False,
            activation=nn.ReLU
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=32,  # Updated to match fusion_path output
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.pre_output = ResidualMLP(
            input_dim=lstm_hidden_dim,
            hidden_dims=geometric_schedule(lstm_hidden_dim, 32, 4),
            final_activation=True,
            activation=nn.ReLU
        )

        self.mu_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.ReLU
        )

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor from optimized dataloader (batch_size, seq_len=300, channels=130) - fhr_up_ph cross-phase features
            return_intermediate: Whether to return intermediate activations

        Returns:
            mu: Latent mean representations (batch_size, seq_len, latent_dim)
            intermediates: Dictionary of intermediate activations (if requested)
        """
        intermediates = {} if return_intermediate else None

        if return_intermediate:
            intermediates["input_with_bias"] = x
        x_linear = self.mlp(x)
        
        if return_intermediate:
            intermediates["channel_reduced"] = x_linear
        
        x_conv = x_linear.transpose(1, 2)  # (B, C, L)        
        conv_1 = self.conv_1(x_conv)

        conv_2 = self.conv_2(conv_1)
        skip_1_norm = self.source_skip_norm_1(conv_1)
        conv_2 = conv_2 + skip_1_norm  # Normalized skip connection
        
        conv_3 = self.conv_3(conv_2)
        skip_2_norm = self.source_skip_norm_2(conv_2)
        conv_3 = conv_3 + skip_2_norm  # Normalized skip connection
        
        conv_out = conv_3.transpose(1, 2).contiguous()  # Back to (B, L, C)
        
        if return_intermediate:
            intermediates["conv_path"] = conv_out

        x = self.fused_norm(conv_out)
        x = self.linear_after_conv(x)
        del x_linear, x_conv, conv_1, conv_2, conv_3, conv_out, skip_1_norm, skip_2_norm  # Explicit cleanup

        x, (hidden, cell) = self.lstm(x)
        x = self.lstm_norm(x)

        if return_intermediate:
            intermediates["lstm_output"] = x
            intermediates["lstm_hidden"] = hidden
            intermediates["lstm_cell"] = cell
        else:
            del hidden, cell  # Clean up LSTM states if not needed

        x = self.pre_output(x)

        if return_intermediate:
            intermediates["post_lstm"] = x

        mu = self.mu_layer(x)

        if return_intermediate:
            intermediates["mu"] = mu
            return mu, intermediates

        return mu

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
            hidden_dims=geometric_schedule(dim_hx + dim_hy, 20, 4),
            final_activation=True,
            use_skip_connection=True, 
            activation=nn.ReLU,
        )

        self.fc_mu = ResidualMLP(
            input_dim=20,
            hidden_dims=geometric_schedule(20, 16, 4),
            final_activation=False,
            use_skip_connection=False, 
            activation=nn.ReLU,
        )
        self.fc_logvar = ResidualMLP(
            input_dim=20,
            hidden_dims=geometric_schedule(20, 16, 4),
            final_activation=False,
            use_skip_connection=False, 
            activation=nn.ReLU,
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
        del h_combined  # Clean up concatenated tensor
        mu = self.fc_mu(h_merged)
        logvar = self.fc_logvar(h_merged)
        del h_merged  # Clean up intermediate tensor
        return mu, logvar



class Decoder(nn.Module):
    """
    Research-backed decoder with progressive upsampling for optimal information bottleneck preservation.
    
    This decoder enforces strict latent compression by using progressive temporal upsampling
    through ConvTranspose1d layers, preventing any information shortcuts that would bypass 
    the TEB bottleneck principle. Based on 2024-2025 VAE research findings.
    """

    def __init__(
        self,
        latent_dim: int = 16,
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

        # Stage 1: Feature expansion from latent bottleneck
        # Force all information through 32D latent space - no shortcuts!
        self.feature_expansion = nn.Sequential(
            ResidualMLP(
                input_dim=latent_dim,
                hidden_dims=geometric_schedule(latent_dim, 50, 5),
                final_activation=True,
                use_skip_connection=True, 
                activation=nn.ReLU,
            ),
            
            ResidualMLP(
                input_dim=50,
                hidden_dims=geometric_schedule(50, 87, 5),
                final_activation=True,
                activation=nn.ReLU,
                use_skip_connection=True
            )
        )
        
        self.pre_linear = ResidualMLP(
            input_dim=87,
            hidden_dims=geometric_schedule(87, 128, 3), 
            final_activation=False,
            activation=nn.ReLU,
            use_skip_connection=True
        )
        
        # Stage 2: Progressive temporal upsampling with ConvTranspose1d
        # 300 → 600 → 1200 → 2400 → 4800 (16x total upsampling)
        # Research shows this is optimal for physiological signal reconstruction
        self.upsample_1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm_1 = nn.GroupNorm(num_groups=8, num_channels=64)
        
        self.upsample_2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) 
        self.norm_2 = nn.GroupNorm(num_groups=8, num_channels=32)
        
        self.upsample_3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.norm_3 = nn.GroupNorm(num_groups=4, num_channels=16)
        
        self.upsample_4 = nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1)
        self.norm_4 = nn.GroupNorm(num_groups=2, num_channels=8)
        
        # Stage 3: Multi-scale refinement for physiological signal quality
        self.refine_conv = nn.Conv1d(8, 4, kernel_size=7, padding=3)
        self.final_conv = nn.Conv1d(4, 1, kernel_size=5, padding=2)
        
        # Stage 4: Gaussian parameter prediction for raw FHR signal
        # Separate heads for mean and log-variance of reconstructed signal
        self.signal_mu = nn.Conv1d(1, 1, kernel_size=1)
        self.signal_logvar = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, latent_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Progressive upsampling forward pass with strict information bottleneck preservation.
        
        This implementation forces all reconstruction to flow through the 32D latent bottleneck,
        preventing any information shortcuts. Research-backed architecture ensures optimal
        latent representation learning.
        
        Args:
            latent_z: Latent variables (batch_size, sequence_length=300, latent_dim=32)
        Returns:
            Tuple containing:
            - linear_output: Auxiliary features (batch_size, sequence_length, 87)
            - raw_signal_mu: Raw signal reconstruction mean (batch_size, 4800)
            - raw_signal_logvar: Raw signal reconstruction log variance (batch_size, 4800)
        """
        z_expanded = self.feature_expansion(latent_z)  # (B, 300, 87)
        z_expanded_pre = self.pre_linear(z_expanded)  # (B, 300, 128)
        z_conv = z_expanded_pre.transpose(1, 2)  # (B, 128, 300) for conv operations
        del z_expanded_pre  # Clean up intermediate tensor
        
        # Stage 2: Progressive temporal upsampling
        # Each step doubles the temporal resolution while reducing channels
        # 300 → 600 samples
        x1 = F.gelu(self.norm_1(self.upsample_1(z_conv)))      # (B, 64, 600)
        del z_conv  # Clean up input tensor
        # 600 → 1200 samples  
        x2 = F.gelu(self.norm_2(self.upsample_2(x1)))          # (B, 32, 1200)  
        del x1  # Clean up intermediate tensor
        # 1200 → 2400 samples
        x3 = F.gelu(self.norm_3(self.upsample_3(x2)))          # (B, 16, 2400)
        del x2  # Clean up intermediate tensor
        # 2400 → 4800 samples
        x4 = F.gelu(self.norm_4(self.upsample_4(x3)))          # (B, 8, 4800)
        del x3  # Clean up intermediate tensor
        
        # Stage 3: Multi-scale refinement for physiological signal quality
        # Capture both coarse trends and fine temporal details
        refined = F.gelu(self.refine_conv(x4))                  # (B, 4, 4800)
        del x4  # Clean up intermediate tensor
        features = self.final_conv(refined)                     # (B, 1, 4800)
        del refined  # Clean up intermediate tensor
        
        # Stage 4: Gaussian parameter prediction
        # Separate prediction heads for mean and log-variance
        mu = self.signal_mu(features).squeeze(1)                # (B, 4800)
        logvar = self.signal_logvar(features).squeeze(1)        # (B, 4800)
        del features  # Clean up intermediate tensor
        
        # Clamp log-variance for numerical stability (as in original TEB model)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return z_expanded, mu, logvar
    
    @staticmethod
    def compute_loss(
        linear_output: torch.Tensor,
        raw_mu_predicted: torch.Tensor, 
        raw_logvar_predicted: torch.Tensor,
        target_fhr_st: torch.Tensor,
        target_fhr_ph: torch.Tensor,
        target_raw_signal: torch.Tensor):
        """
        Compute two-part loss: MSE loss for linear output and NLL loss for raw signal reconstruction.
        
        Identical to original Decoder.compute_loss() for compatibility.
        
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
        
        # MSE Loss: Compare linear output with stacked fhr_st and fhr_ph
        if linear_output.shape[-1] == 87 and target_fhr_st.shape[-1] == 43 and target_fhr_ph.shape[-1] == 44:
            # Stack fhr_st and fhr_ph along the last dimension (43 + 44 = 87)
            stacked_target = torch.cat([target_fhr_st, target_fhr_ph], dim=-1)  # (B, S, 87)
            mse_loss = F.mse_loss(linear_output, stacked_target)
        else:
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # NLL Loss: Full raw signal reconstruction (no warmup period)
        # Ensure target_raw_signal is the right shape
        if target_raw_signal.dim() == 3 and target_raw_signal.size(-1) == 1:
            target_raw_signal = target_raw_signal.squeeze(-1)  # Remove channel dimension if present
        
        # Compute Gaussian NLL: 0.5 * (log(var) + (target - mu)^2 / var)
        diff = target_raw_signal - raw_mu_predicted  # (B, 4800)
        var = raw_logvar_predicted.exp()  # (B, 4800)
        nll_loss = 0.5 * (raw_logvar_predicted + diff.pow(2) / var)  # (B, 4800)
        nll_loss = nll_loss.mean()  # Average over all samples and time points
        
        return {
            'mse_loss': mse_loss,
            'nll_loss': nll_loss,
            'total_decoder_loss': mse_loss + nll_loss
        }


class SeqVaeTeb(nn.Module):
    """
    Sequence VAE with Transfer Entropy Bottleneck (TEB).

    This model implements the full SeqVaeTeb framework, which learns a latent representation
    of a target signal (y) that is predictive of the signal's future, while minimizing the
    information it contains about a source signal (x). This is achieved by minimizing the
    KL divergence between a posterior distribution q(z|x,y) and a prior distribution p(z|y).
    """

    def __init__(
        self,
        sequence_length: int = 300,
        latent_dim_source: int = 16,
        latent_dim_target: int = 16,
        latent_dim_z: int = 16,
        decimation_factor: int = 16,
        warmup_period: int = 30,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 5,
    ):
        super().__init__()

        self.latent_dim_source = latent_dim_source
        self.latent_dim_target = latent_dim_target
        self.latent_dim_z = latent_dim_z
        self.decimation_factor = decimation_factor
        self.warmup_period = warmup_period

        self.source_encoder = SourceEncoder(
            sequence_length=sequence_length,
            latent_dim=latent_dim_source,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        self.target_encoder = TargetEncoder(
            sequence_length=sequence_length,
            latent_dim=latent_dim_target,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        self.conditional_encoder = ConditionalEncoder(
            dim_hx=latent_dim_source,
            dim_hy=latent_dim_target,
            dim_z=latent_dim_z,
        )
        
        self.decoder = Decoder(latent_dim=latent_dim_z, sequence_length=sequence_length)  # Original decoder for backward compatibility

        initialization(self)

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of the SeqVaeTeb model.

        Args:
            y_st: Target scattering input from optimized dataloader (Batch, sequence_len=300, channels=43)
            y_ph: Target phase harmonic input from optimized dataloader (Batch, sequence_len=300, channels=44)
            x_ph: Source phase harmonic input from optimized dataloader (Batch, sequence_len=300, channels=130)

        Returns:
            A dictionary containing tensors needed for loss computation.
        """

        # Source encoder for q(h_x|x)
        mu_x = self.source_encoder(x_ph)

        # Target encoder for p(z|y) - now returns separate outputs for TEB compliance
        mu_y, logvar_y_prior, c_logvar = self.target_encoder(y_st, y_ph)

        # Conditional encoder for q(z|x, y)
        mu_post, logvar_post = self.conditional_encoder(mu_x, c_logvar)
        mu_post = mu_post + mu_y

        z = self.reparameterize(mu_post, logvar_post)

        # Decode raw signal predictions from z (no encoder features to preserve information bottleneck)
        linear_output, mu_pr, logvar_pr = self.decoder(z)

        return {
            "z": z,  # (batch, length, channel)
            "linear_output": linear_output,  # (batch, length, 87)
            "mu_pr": mu_pr, # (batch, 4800) - raw signal reconstruction
            "logvar_pr": logvar_pr,  # (batch, 4800) - raw signal reconstruction
            "mu_prior": mu_y,
            "logvar_prior": logvar_y_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
        }

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample from a Gaussian."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Remove in-place operation

    def _kld_loss(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        reduce_mean: bool = True,
    ) -> torch.Tensor:
        """
        Computes the KL divergence between two Gaussian distributions.

        Args:
            mu_prior: Mean of the prior distribution.
            logvar_prior: Log variance of the prior distribution.
            mu_post: Mean of the posterior distribution.
            logvar_post: Log variance of the posterior distribution.
            reduce_mean: If True, returns the mean KLD (scalar). 
                        If False, returns the KLD tensor (batch, seq_len, latent_dim).
        """
        kld = (
                logvar_prior
                - logvar_post
                - 1
                + (logvar_post.exp() + (mu_post - mu_prior).pow(2))
                / logvar_prior.exp()
        )
        kld = 0.5 * kld

        if reduce_mean:
            return kld.sum(dim=-1).mean()
        return kld

    
    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor, 
        y_raw: torch.Tensor,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the total training loss with MSE and NLL components.

        Args:
            forward_outputs: The dictionary returned by the forward pass.
            y_st: Target scattering coefficients from optimized dataloader (B, S=300, channels=43)
            y_ph: Target phase coefficients from optimized dataloader (B, S=300, channels=44)
            y_raw: Ground truth raw signal data from optimized dataloader (B, 4800)
            compute_kld_loss (bool): Whether to compute KLD loss.
            beta (float): Beta weight for KLD loss in VAE training.

        Returns:
            A dictionary of computed losses.
        """
        device = y_raw.device
        kld_loss = torch.tensor(0.0, device=device)

        if y_raw.dim() == 3 and y_raw.size(-1) == 1:
            y_raw = y_raw.squeeze(-1)  # Remove channel dimension if present

        # Decoder losses (MSE + NLL)
        decoder_losses = self.decoder.compute_loss(
            linear_output=forward_outputs['linear_output'],
            raw_mu_predicted=forward_outputs['mu_pr'], 
            raw_logvar_predicted=forward_outputs['logvar_pr'],
            target_fhr_st=y_st,
            target_fhr_ph=y_ph,
            target_raw_signal=y_raw
        )

        # KLD loss
        if compute_kld_loss:
            kld_loss = self._kld_loss(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"],
                reduce_mean=True,  # Ensure scalar loss for training
            )

        # Total loss with beta-weighted KLD
        total_loss = decoder_losses['total_decoder_loss'] + beta * kld_loss

        return {
            "reconstruction_loss": decoder_losses['total_decoder_loss'],  # For backward compatibility
            "mse_loss": decoder_losses['mse_loss'],
            "nll_loss": decoder_losses['nll_loss'], 
            "kld_loss": kld_loss,
            "total_loss": total_loss,
            "classification_loss": None,  # Required by interface
        }

    def measure_transfer_entropy(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        """
        Measures the transfer entropy from source (x) to the latent representation (z).
        This is equivalent to the KL divergence between the posterior q(z|x,y) and the prior p(z|y).

        Args:
            y_st: Target scattering input (Batch, sequence_len, channels)
            y_ph: Target phase harmonic input (Batch, sequence_len, channels)
            x_ph: Source phase harmonic input (Batch, sequence_len, channels)
            reduce_mean: If True, returns the mean KLD (a scalar). 
                        If False, returns the KLD for each latent dim at each timestep.

        Returns:
            The transfer entropy value. A scalar if reduce_mean is True, 
            or a tensor of shape (batch, seq_len, latent_dim) otherwise.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            forward_outputs = self.forward(y_st, y_ph, x_ph)
            transfer_entropy = self._kld_loss(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"],
                reduce_mean=reduce_mean,
            )
        return transfer_entropy

    @staticmethod
    def get_predictions(x, stride=16, new_C=4800):
        """
        x: (B, N, C)
        returns:
            y:      (B, N, new_C)  — with NaNs where no data was placed
            mean:   (B, new_C)      — nan-mean over dim=1
        """
        B, N, C = x.shape
        y = x.new_full((B, N, new_C), float('nan'))
        for i in range(N):
            start = i * stride
            if start >= new_C:
                break
            end = min(start + C, new_C)
            length = end - start
            y[:, i, start:end] = x[:, i, :length]
        mean = torch.nanmean(y, dim=1)  # → shape (B, new_C)
        return y, mean


class SeqVaeTebClassifier(nn.Module):
    """
    Combined model that integrates SeqVaeTeb encoder with FHR Inception Time classifier.
    
    This model can:
    1. Load a pretrained SeqVaeTeb model for feature extraction
    2. Freeze the SeqVaeTeb encoder during classification training
    3. Fine-tune the entire model end-to-end
    4. Perform classification on learned latent representations
    """
    
    def __init__(
        self,
        # Classifier parameters
        num_classes: int = 2,
        classifier_filters: int = 32,
        classifier_depth: int = 6,
        classifier_dropout: float = 0.2,
        use_attention: bool = True,
        # Training parameters
        latent_dim_z: int = 16,
        freeze_vae: bool = True,
        pretrained_vae_path: Optional[str] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.freeze_vae = freeze_vae
        self.num_classes = num_classes
        
        # Initialize SeqVaeTeb encoder
        self.vae_model = SeqVaeTeb()
        
        # Load pretrained VAE if provided
        if pretrained_vae_path is not None:
            self.load_pretrained_vae(pretrained_vae_path)
        
        # Freeze VAE parameters if specified
        if freeze_vae:
            self.freeze_vae_parameters()
        
        # Import FHRInceptionTimeClassifier
        try:
            from model.inception_time import FHRInceptionTimeClassifier
        except ImportError:
            from inception_time import FHRInceptionTimeClassifier
        
        # Initialize classifier
        self.classifier = FHRInceptionTimeClassifier(
            input_size=latent_dim_z,
            num_classes=num_classes,
            filters=classifier_filters,
            depth=classifier_depth,
            dropout=classifier_dropout,
            use_attention=use_attention
        )
        
        self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def load_pretrained_vae(self, pretrained_path: str):
        """Load pretrained SeqVaeTeb weights (handles Lightning checkpoints).

        Notes:
        - Lightning `state_dict` usually prefixes model params with 'model.'.
        - We strip that prefix when loading directly into a bare SeqVaeTeb.
        - torch.compile does not affect state_dict; compile after loading if desired.
        """
        try:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            # Prefer Lightning's 'state_dict' if present; otherwise treat the whole object as a state dict
            sd = ckpt.get("state_dict", ckpt)

            # Strip leading 'model.' if present on keys
            new_sd = {}
            for k, v in sd.items():
                nk = k[6:] if k.startswith("model.") else k
                new_sd[nk] = v

            # Load with strict=False to tolerate non-matching classifier-related keys, etc.
            incompatible = self.vae_model.load_state_dict(new_sd, strict=False)

            # Support both tuple and IncompatibleKeys return types
            try:
                missing_keys = getattr(incompatible, "missing_keys", [])
                unexpected_keys = getattr(incompatible, "unexpected_keys", [])
            except Exception:
                try:
                    missing_keys, unexpected_keys = incompatible
                except Exception:
                    missing_keys, unexpected_keys = [], []

            if missing_keys:
                log.warning(f"[SeqVaeTebClassifier] Missing keys when loading VAE: {missing_keys}")
            if unexpected_keys:
                log.warning(f"[SeqVaeTebClassifier] Unexpected keys when loading VAE: {unexpected_keys}")

            log.info(f"Loaded pretrained VAE weights from {pretrained_path}")

        except Exception as e:
            log.error(f"Error loading pretrained VAE from '{pretrained_path}': {e}")
            log.warning("Continuing with randomly initialized VAE parameters…")
    
    def freeze_vae_parameters(self):
        """Freeze all VAE parameters to prevent updates during classification training."""
        for param in self.vae_model.parameters():
            param.requires_grad = False
        log.info("Frozen VAE parameters for classification training")
    
    def unfreeze_vae_parameters(self):
        """Unfreeze VAE parameters for end-to-end fine-tuning."""
        for param in self.vae_model.parameters():
            param.requires_grad = True
        log.info("Unfrozen VAE parameters for end-to-end training")
    
    def extract_latent_features(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        return_all_outputs: bool = False
    ):
        """
        Extract latent features from the VAE encoder.
        
        Args:
            y_st: Target scattering input (batch, 300, 43)
            y_ph: Target phase harmonic input (batch, 300, 44)
            x_ph: Source phase harmonic input (batch, 300, 130)
            return_all_outputs: Whether to return all VAE outputs or just latent z
            
        Returns:
            latent_z: Latent representations (batch, 300, 32)
            vae_outputs: Full VAE outputs (if return_all_outputs=True)
        """
        # Set VAE to eval mode if frozen
        if self.freeze_vae:
            self.vae_model.eval()
        
        # Forward pass through VAE
        with torch.set_grad_enabled(not self.freeze_vae):
            vae_outputs = self.vae_model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        
        latent_z = vae_outputs['z']  # (batch, 300, 32)
        
        if return_all_outputs:
            return latent_z, vae_outputs
        return latent_z
    
    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ):
        """
        Forward pass for classification.
        
        Args:
            y_st: Target scattering input (batch, 300, 43)
            y_ph: Target phase harmonic input (batch, 300, 44)  
            x_ph: Source phase harmonic input (batch, 300, 130)
            labels: Ground truth labels for loss computation (batch,)
            return_latent: Whether to return latent representations
            
        Returns:
            Dictionary containing classification results and optionally latent features
        """
        # Extract latent features
        latent_z = self.extract_latent_features(y_st, y_ph, x_ph)
        
        # Classification
        logits = self.classifier(latent_z)  # (batch, num_classes)
        
        # Compute loss if labels provided
        classification_loss = None
        if labels is not None:
            classification_loss = self.classification_criterion(logits, labels)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'predictions': torch.argmax(logits, dim=-1),
            'classification_loss': classification_loss,
        }
        
        if return_latent:
            outputs['latent_z'] = latent_z
            
        return outputs
    
    def compute_loss(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        labels: torch.Tensor,
        y_raw: Optional[torch.Tensor] = None,
        compute_vae_loss: bool = False,
        vae_loss_weight: float = 0.1
    ):
        """
        Compute combined loss for classification and optionally VAE reconstruction.
        
        Args:
            y_st, y_ph, x_ph: Input tensors
            labels: Classification labels (batch,)
            y_raw: Raw signal for VAE loss computation (batch, 4800)
            compute_vae_loss: Whether to include VAE reconstruction loss
            vae_loss_weight: Weight for VAE loss component
            
        Returns:
            Dictionary of loss components
        """
        # Get classification outputs
        if compute_vae_loss and y_raw is not None:
            # Need full VAE outputs for reconstruction loss
            latent_z, vae_outputs = self.extract_latent_features(
                y_st, y_ph, x_ph, return_all_outputs=True
            )
            
            # Compute VAE losses
            vae_losses = self.vae_model.compute_loss(
                forward_outputs=vae_outputs,
                y_st=y_st,
                y_ph=y_ph,
                y_raw=y_raw,
                compute_kld_loss=True,
                beta=1.0  # Default beta for classifier training
            )
            vae_total_loss = vae_losses['total_loss']
        else:
            latent_z = self.extract_latent_features(y_st, y_ph, x_ph)
            vae_total_loss = torch.tensor(0.0, device=latent_z.device)
        
        # Classification loss
        logits = self.classifier(latent_z)
        classification_loss = self.classification_criterion(logits, labels)
        
        # Combined loss
        total_loss = classification_loss + vae_loss_weight * vae_total_loss
        
        return {
            'classification_loss': classification_loss,
            'vae_loss': vae_total_loss,
            'total_loss': total_loss,
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'predictions': torch.argmax(logits, dim=-1),
        }
    
    def predict(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        return_probabilities: bool = True
    ):
        """
        Make predictions on input data.
        
        Args:
            y_st, y_ph, x_ph: Input tensors
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions and optionally probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                y_st=y_st, y_ph=y_ph, x_ph=x_ph,
                labels=None, return_latent=False
            )
        
        if return_probabilities:
            return outputs['predictions'], outputs['probabilities']
        return outputs['predictions']


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
    model = SourceEncoder()
    mu = model(x_ph_input)

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
    #
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     exit(1)
    #
    # # --- Test Loss Computation ---
    # try:
    #     loss_dict = model.compute_loss(forward_outputs, y_raw=y_raw_input)
    #
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     exit(1)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # num_epochs = 3
    #
    # for epoch in range(num_epochs):
    #     model.train()  # Set the model to training mode
    #
    #     # 1. Zero the gradients
    #     optimizer.zero_grad()
    #
    #     # 2. Forward pass
    #     forward_outputs = model(
    #         y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input
    #     )
    #
    #     # 3. Compute loss
    #     loss_dict = model.compute_loss(forward_outputs, y_raw=y_raw_input)
    #     total_loss = loss_dict["total_loss"]
    #
    #     # 4. Backward pass
    #     total_loss.backward()
    #
    #     # 5. Update weights
    #     optimizer.step()
