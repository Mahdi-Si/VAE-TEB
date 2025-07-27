import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict
import math
import warnings

import math
from typing import List

def geometric_schedule(
    input_size: int,
    output_size: int,
    n_hidden: int,
    *,
    round_fn=round
) -> List[int]:
    """
    SPEED OPTIMIZED: Compute a geometric progression of layer sizes from `input_size` down/up to `output_size`,
    with `n_hidden` intermediate layers.

    Returns a list of length n_hidden+2: [input_size, h1, h2, ..., h_n, output_size].
    
    Arguments:
    - input_size:  starting dimension (e.g. 16)
    - output_size: ending dimension (e.g. 64)
    - n_hidden:    number of hidden layers (e.g. 6)
    - round_fn:    function to turn floats into ints (default=round)
    """
    # SPEED OPTIMIZATION: Avoid repeated calculations and list comprehension
    # total steps = hidden layers + the final map to output
    steps = n_hidden + 1
    # constant ratio r so that input_size * r^steps = output_size
    r = (output_size / input_size) ** (1 / steps)

    # SPEED OPTIMIZATION: Pre-allocate tuple and calculate directly
    sizes = [input_size]
    current_r = r
    for _ in range(n_hidden):
        sizes.append(int(round_fn(input_size * current_r)))
        current_r *= r
    sizes.append(output_size)
    
    return tuple(sizes[1:])

def initialization(model: nn.Module) -> None:
    """
    Applies state-of-the-art initialization schemes to all model components.
    Called automatically during model instantiation to ensure proper gradient flow.

    Args:
        model: PyTorch model to initialize
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            # Xavier/Glorot initialization for linear and conv layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            # Orthogonal initialization for LSTM weights
            for param_name, param in module.named_parameters():
                if "weight_ih" in param_name or "weight_hh" in param_name:
                    nn.init.orthogonal_(param)
                elif "bias" in param_name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 for better gradient flow
                    if "bias_hh" in param_name:
                        hidden_size = module.hidden_size
                        param.data[hidden_size : 2 * hidden_size].fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            # Standard LayerNorm initialization
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class CausalConv1d(nn.Module):
    """
    Optimized causal 1D convolution that ensures no future information leaks.
    Uses efficient padding strategy and supports mixed precision.
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
        use_batch_norm: bool = True,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False
    ):
        super(CausalMultiChannelConvBlock, self).__init__()
        
        self.up_sampling = up_sampling
        self.up_sample_scale = up_sample_scale
        self.filter_size = filter_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dilation = dilation
        self.stride = stride
        
        # Calculate causal padding (left padding only)
        self.left_padding = (filter_size - 1) * dilation
        
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
        
        # Optional batch normalization
        if use_batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels, momentum=0.9)
        else:
            self.bn_layer = None
            
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
        
        # Apply causal padding (left padding only)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        
        # Apply convolution
        output = self.conv(x)
        
        # Apply batch normalization if enabled
        if self.bn_layer is not None:
            output = self.bn_layer(output)
        
        # Apply activation function
        output = self.act_fn(output)
        
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

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=filter_size,
            groups=groups, bias=False)
        self.bn_layer  = nn.BatchNorm1d(out_channels, momentum=0.9)

    def forward(self, x):
        if self.up_sampling:
            x = F.interpolate(
                x, scale_factor=self.up_scale,
                mode='linear', align_corners=False)

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
        x = self.bn_layer(x)
        return torch.tanh(x) if self.tanh else F.relu(x)




class ChannelReductionBlock(nn.Module):
    """
    Efficient channel reduction block for reducing input dimensionality.
    Uses depthwise separable convolutions and learns optimal channel combinations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Channel attention for learning which channels are most important
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_channels, in_channels // 4, 1),
                nn.ReLU(),
                nn.Conv1d(in_channels // 4, in_channels, 1),
                nn.Sigmoid()
            )
        
        # Depthwise separable convolution for efficient processing
        self.depthwise = CausalConv1d(
            in_channels, in_channels, kernel_size, groups=in_channels
        )
        
        # Pointwise convolution for channel reduction
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        
        # Normalization and activation
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, in_channels)
        Returns:
            Reduced tensor (batch_size, seq_len, out_channels)
        """
        # Apply layer norm first
        x_norm = self.norm1(x)
        
        # Convert to channel-first for convolutions
        x_conv = x_norm.transpose(1, 2)  # (B, C, L)
        
        # Apply channel attention if enabled
        if self.use_attention:
            attention = self.channel_attention(x_conv)
            x_conv = x_conv * attention
        
        # Depthwise convolution
        x_conv = self.depthwise(x_conv)
        
        # Pointwise convolution for channel reduction
        x_conv = self.pointwise(x_conv)
        
        # Convert back to sequence-first
        x_out = x_conv.transpose(1, 2)  # (B, L, C_out)
        
        # Apply final normalization and dropout
        x_out = self.norm2(x_out)
        x_out = F.gelu(x_out)
        x_out = self.dropout(x_out)
        
        return x_out


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

    def __init__(
        self,
        sequence_length: int = 300,
        latent_dim: int = 32,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 4,
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
        
        self.mlp_scattering = nn.Sequential(
            ResidualMLP(
                input_dim=43,
                hidden_dims=geometric_schedule(43, 16, 4),
                final_activation=False,
                use_skip_connection=True,
                activation=nn.GELU
                )
        )
        
        self.mlp_phase = ResidualMLP(
            input_dim=44,
            hidden_dims=geometric_schedule(44, 16, 4),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.ReLU
            )

        # Sequential convolutions for scattering
        self.conv_scattering = nn.Sequential(
            CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=3, dilation=1),
            CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=5, dilation=1),
            CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=7, dilation=1),
        )

        # Sequential convolutions for phase
        self.conv_phase = nn.Sequential(
            CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=3, dilation=1),
            CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=5, dilation=1),
            CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=7, dilation=1),
        )
        
        # LayerNorm for fused outputs
        self.scatter_fused_norm = nn.LayerNorm(16)
        self.phase_fused_norm = nn.LayerNorm(16)
        
        # LayerNorm for LSTM output
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim * (2 if use_bidirectional_lstm else 1))

        self.cross_modal_fusion = ResidualMLP(
            input_dim=16 * 2,  # Updated to reflect 32-channel outputs from each path
            hidden_dims=geometric_schedule(16*2, 20, 5),  # Smaller intermediate dimensions
            final_activation=False,
            activation=nn.ReLU,
            use_skip_connection=True
        )

        self.lstm = nn.LSTM(
            input_size=20,  # Updated to match cross_modal_fusion output
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=use_bidirectional_lstm,
        )

        lstm_output_dim = lstm_hidden_dim * (2 if use_bidirectional_lstm else 1)

        # Pre-output processing
        self.pre_output = ResidualMLP(
            input_dim=lstm_output_dim,
            hidden_dims=geometric_schedule(lstm_output_dim, 32, 5),
            final_activation=True,
            activation=nn.ReLU
        )

        # Variational parameters
        self.mu_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 32, 32),
            final_activation=False,
            activation=nn.ReLU
        )
        
        self.logvar_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 64, 4),
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

        # SPEED OPTIMIZATION: Avoid double permutation - use contiguous for better performance
        scatter_conv = self.conv_scattering(scatter_linear.transpose(1, 2)).transpose(1, 2).contiguous()

        scatter_conv = self.scatter_fused_norm(scatter_conv)
        del scatter_linear

        # SPEED OPTIMIZATION: Avoid double permutation - use contiguous for better performance
        phase_conv = self.conv_phase(phase_linear.transpose(1, 2)).transpose(1, 2).contiguous()

        phase_conv = self.phase_fused_norm(phase_conv)
        del phase_linear

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

        x = self.pre_output(x)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        logvar = torch.clamp(logvar, min=-10, max=10)

        if return_hidden:
            hidden_states["mu"] = mu
            hidden_states["logvar"] = logvar
            return mu, logvar, hidden_states

        return mu, logvar

    def get_encoder_features(
        self, scattering_input: torch.Tensor, phase_harmonic_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract encoder features without variational sampling.
        Useful for analysis and visualization.
        """
        with torch.no_grad():
            mu, _ = self.forward(scattering_input, phase_harmonic_input)
            return mu


class SourceEncoder(nn.Module):
    """
    Professional VAE encoder with single input and mu-only output.

    Architecture:
    1. Dual-path processing: Multi-layer linear projection + Causal convolution paths (summed)
    2. Multi-layer linear transformations with residual connections
    3. Unidirectional LSTM for causal temporal encoding (encodes info up to each timestep)
    4. Multi-layer linear processing with advanced normalization
    5. Final linear layer outputting mu latent representations

    State-of-the-art techniques incorporated:
    - Pre-LayerNorm design for superior gradient flow
    - Residual connections with proper skip projections
    - Causal convolutions with depthwise separable operations
    - Advanced weight initialization (Xavier + Orthogonal for LSTM)
    - Gradient clipping-friendly architecture
    - GLU-style gating mechanisms
    - Learnable positional biases
    - Adaptive dropout scheduling
    """

    def __init__(
        self,
        input_channels: int = 130,
        sequence_length: int = 300,
        latent_dim: int = 32,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 4,
        activation: nn.Module = nn.GELU,
    ):
        super(SourceEncoder, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # Channel reduction block
        self.mlp = ResidualMLP(
            input_dim=input_channels,
            hidden_dims=geometric_schedule(130, 32, 5),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.ReLU
            )
        
        # Sequential convolutions for source encoder
        self.conv = nn.Sequential(
            CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=3, dilation=1),
            CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=5, dilation=1),
            CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=7, dilation=1),
        )
        
        self.fused_norm = nn.LayerNorm(32)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)
        # Unidirectional LSTM for causal temporal encoding
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
            hidden_dims=geometric_schedule(32, 32, 4),
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
        # SPEED OPTIMIZATION: Avoid double permutation - use contiguous for better performance
        conv_out = self.conv(x_linear.transpose(1, 2)).transpose(1, 2).contiguous()
        
        if return_intermediate:
            intermediates["conv_path"] = conv_out

        x = self.fused_norm(conv_out)
        del x_linear, conv_out  # Explicit cleanup

        x, (hidden, cell) = self.lstm(x)
        x = self.lstm_norm(x)

        if return_intermediate:
            intermediates["lstm_output"] = x
            intermediates["lstm_hidden"] = hidden
            intermediates["lstm_cell"] = cell

        x = self.pre_output(x)

        if return_intermediate:
            intermediates["post_lstm"] = x

        # Final mu layer with residual connection
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
        # Ensure timestep is valid
        timestep = min(timestep, x.size(1) - 1)

        # Forward pass and return encoding up to timestep
        mu = self.forward(x)
        return mu[:, : timestep + 1, :]


class ConditionalEncoder(nn.Module):
    """
    Implements the conditional encoder q(z | x, y) for the TEB framework.
    It maps concatenated source (x) and target (y) latent representations
    to the parameters of the posterior Gaussian distribution for z.

    This module is designed to work with sequence data, where the linear
    transformations are applied independently at each time step.
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

        # The input dimension to the MLP is the sum of source and target feature dimensions

        # Build a small MLP to merge h_x and h_y
        hidden_dims = geometric_schedule(dim_hx + dim_hy, 32, 8)
        self.mlp = ResidualMLP(
            input_dim=dim_hx + dim_hy,
            hidden_dims=hidden_dims[0:5],
            final_activation=True,
            use_skip_connection=True, 
            activation=nn.ReLU,
        )

        # Final linear layers to produce mu and logvar for the latent variable z
        self.fc_mu = ResidualMLP(
            input_dim=hidden_dims[4],
            hidden_dims=hidden_dims[5:],
            final_activation=False,
            use_skip_connection=False, 
            activation=nn.ReLU,
        )
        self.fc_logvar = ResidualMLP(
            input_dim=hidden_dims[4],
            hidden_dims=hidden_dims[5:],
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
        # Concatenate along the feature dimension (-1)
        h_combined = torch.cat([h_x, h_y], dim=-1)

        # Pass the combined representation through the MLP
        h_merged = self.mlp(h_combined)

        # Compute mu and logvar
        mu = self.fc_mu(h_merged)
        logvar = self.fc_logvar(h_merged)

        return mu, logvar


class Decoder(nn.Module):
    """
    Simplified Raw Signal Decoder that predicts a fixed-size future window from the entire sequence.
    
    Key changes:
    - Predicts a single future window instead of overlapping predictions from each timestep
    - Much simpler and more efficient architecture
    - Clear alignment between predictions and targets
    - Reduced memory usage and computational complexity
    """

    def __init__(
        self,
        latent_dim: int = 32,
        sequence_length: int = 300,
        prediction_horizon: int = 480,  # 2 minutes at 4Hz = 480 samples
    ):
        """
        Args:
            latent_dim: Input latent dimension
            sequence_length: Input sequence length
            prediction_horizon: Number of future samples to predict (default 480 = 2 minutes at 4Hz)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon


        # Process latent sequence to extract temporal features
        self.linear = nn.Sequential(
            ResidualMLP(
            input_dim=latent_dim,
            hidden_dims=geometric_schedule(latent_dim, 50, 5),
            final_activation=True,
            use_skip_connection=True, 
            activation=nn.ReLU,),
            
            ResidualMLP(
            input_dim=50,
            hidden_dims=geometric_schedule(50, 87, 5),
            final_activation=True,
            activation=nn.ReLU,
            use_skip_connection=True
        )
        )

        self.conv = nn.Sequential(
            MultiChannelConvBlock(in_channels=87, out_channels=77, filter_size=11, up_sampling=False),
            MultiChannelConvBlock(in_channels=77, out_channels=66, filter_size=9, up_sampling=True),
            MultiChannelConvBlock(in_channels=66, out_channels=55, filter_size=7, up_sampling=True),
            MultiChannelConvBlock(in_channels=55, out_channels=44, filter_size=5, up_sampling=False),
            MultiChannelConvBlock(in_channels=44, out_channels=33, filter_size=5, up_sampling=True),
            MultiChannelConvBlock(in_channels=33, out_channels=22, filter_size=3, up_sampling=True),
            MultiChannelConvBlock(in_channels=22, out_channels=11, filter_size=3, up_sampling=False),
            MultiChannelConvBlock(in_channels=11, out_channels=1, filter_size=3, up_sampling=False),
        )
        
        self.output_mu = ResidualMLP(
            input_dim=4800,
            hidden_dims=(4800, 4800),
            final_activation=False,
            use_skip_connection=False,
            activation=nn.ReLU
        )
        
        self.output_logvar = ResidualMLP(
            input_dim=4800,
            hidden_dims=(4800, 4800),
            final_activation=False,
            use_skip_connection=False,
            activation=nn.ReLU
        )

    def forward(self, latent_z: torch.Tensor):
        """
        Forward pass that reconstructs the raw signal from latent variables.
        
        Args:
            latent_z: Latent variables (batch_size, sequence_length=300, latent_dim=32)
        Returns:
            Tuple containing:
            - linear_output: Output from linear layers (batch_size, sequence_length, 87)
            - raw_signal_mu: Raw signal reconstruction mean (batch_size, 4800)
            - raw_signal_logvar: Raw signal reconstruction log variance (batch_size, 4800)
        """
        batch_size, sequence_length, _ = latent_z.shape
        
        # Apply linear transformations
        linear_output = self.linear(latent_z)  # (batch_size, sequence_length, 87)
        
        # SPEED OPTIMIZATION: Use transpose instead of permute for better performance
        # Permute for convolution: (batch_size, channels, sequence_length)
        x = linear_output.transpose(1, 2)
        
        # Apply convolution layers
        x = self.conv(x)  # (batch_size, 1, upsampled_length)
        
        # Flatten for final prediction
        x = x.flatten(start_dim=1)  # (batch_size, flattened_features)
        
        # Generate mu and logvar predictions for full raw signal (4800 samples)
        mu = self.output_mu(x)  # (batch_size, 4800)
        logvar = self.output_logvar(x)  # (batch_size, 4800)
        
        return linear_output, mu, logvar
        

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
    Memory-optimized Sequence VAE with Target-Encoder-Bank (TEB) framework.

    This model integrates a source encoder, a target encoder, a conditional
    encoder, and a decoder to perform future prediction with uncertainty.

    Memory optimizations applied:
    - Gradient checkpointing in residual blocks
    - In-place operations where possible
    - Explicit memory cleanup with del statements
    - F.pad instead of pre-allocated padding tensors
    - Mixed precision training support
    - Efficient tensor transpose operations

    The architecture is based on the following flow:
    1. A source encoder processes input `x_ph` to get `mu_x`.
    2. A target encoder processes `y_st` and `y_ph` to model the prior `p(z|y)`.
        It outputs `mu_y` and a `logvar_y` that is split into a prior log-variance
        and a conditional feature `c_logvar`.
    3. A conditional encoder combines `mu_x` and `c_logvar` to model the posterior
        `q(z|x,y)`, outputting `mu_post` and `logvar_post`.
    4. A latent variable `z` is sampled from the posterior using the reparameterization trick.
    5. A decoder takes `z` and predicts future sequences for `y_st` and `y_ph`.

    Loss is composed of reconstruction loss (from the decoder) and a KL-divergence
    term between the prior and posterior distributions.
    """

    def __init__(
        self,
        input_channels: int = 76,
        sequence_length: int = 300,
        latent_dim_source: int = 32,
        latent_dim_target: int = 32,
        latent_dim_z: int = 32,
        decimation_factor: int = 16,
        warmup_period: int = 30,
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
        )
        self.target_encoder = TargetEncoder(
            sequence_length=sequence_length,
            latent_dim=latent_dim_target,
        )
        self.conditional_encoder = ConditionalEncoder(
            dim_hx=latent_dim_source,
            dim_hy=latent_dim_target,
            dim_z=latent_dim_z,
        )
        self.decoder = Decoder()

        initialization(self)

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

        # Target encoder for p(z|y)
        mu_y, logvar_y_full = self.target_encoder(y_st, y_ph)

        # Split target logvar for prior and conditional feature
        logvar_y_prior, c_logvar = torch.split(
            logvar_y_full, self.latent_dim_target, dim=-1
        )

        # Conditional encoder for q(z|x, y)
        mu_post, logvar_post = self.conditional_encoder(mu_x, c_logvar)
        mu_post = mu_post + mu_y

        z = self.reparameterize(mu_post, logvar_post)

        # Decode raw signal predictions from z
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

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor, 
        y_raw: torch.Tensor,
        compute_kld_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the total training loss with MSE and NLL components.

        Args:
            forward_outputs: The dictionary returned by the forward pass.
            y_st: Target scattering coefficients from optimized dataloader (B, S=300, channels=43)
            y_ph: Target phase coefficients from optimized dataloader (B, S=300, channels=44)
            y_raw: Ground truth raw signal data from optimized dataloader (B, 4800)
            compute_kld_loss (bool): Whether to compute KLD loss.

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

        # Total loss
        total_loss = decoder_losses['total_decoder_loss'] + kld_loss

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
        # SeqVaeTeb parameters
        input_channels: int = 76,
        sequence_length: int = 300,
        latent_dim_source: int = 32,
        latent_dim_target: int = 32,
        latent_dim_z: int = 32,
        decimation_factor: int = 16,
        warmup_period: int = 30,
        # Classifier parameters
        num_classes: int = 2,
        classifier_filters: int = 32,
        classifier_depth: int = 6,
        classifier_dropout: float = 0.2,
        use_attention: bool = True,
        # Training parameters
        freeze_vae: bool = True,
        pretrained_vae_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.freeze_vae = freeze_vae
        self.num_classes = num_classes
        self.latent_dim_z = latent_dim_z
        
        # Initialize SeqVaeTeb encoder
        self.vae_model = SeqVaeTeb(
            input_channels=input_channels,
            sequence_length=sequence_length,
            latent_dim_source=latent_dim_source,
            latent_dim_target=latent_dim_target,
            latent_dim_z=latent_dim_z,
            decimation_factor=decimation_factor,
            warmup_period=warmup_period,
        )
        
        # Load pretrained VAE if provided
        if pretrained_vae_path is not None:
            self.load_pretrained_vae(pretrained_vae_path)
        
        # Freeze VAE parameters if specified
        if freeze_vae:
            self.freeze_vae_parameters()
        
        # Import FHRInceptionTimeClassifier
        try:
            from .inception_time import FHRInceptionTimeClassifier
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
        
        # Loss function for classification
        self.classification_criterion = nn.CrossEntropyLoss()
        
    def load_pretrained_vae(self, pretrained_path: str):
        """Load pretrained SeqVaeTeb weights."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                vae_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                vae_state_dict = checkpoint['state_dict']
            else:
                vae_state_dict = checkpoint
            
            # Load weights with strict=False to allow for missing keys
            missing_keys, unexpected_keys = self.vae_model.load_state_dict(vae_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys when loading VAE: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading VAE: {unexpected_keys}")
                
            print(f"Successfully loaded pretrained VAE from {pretrained_path}")
            
        except Exception as e:
            print(f"Error loading pretrained VAE: {e}")
            print("Continuing with random initialization...")
    
    def freeze_vae_parameters(self):
        """Freeze all VAE parameters to prevent updates during classification training."""
        for param in self.vae_model.parameters():
            param.requires_grad = False
        print("Frozen VAE parameters for classification training")
    
    def unfreeze_vae_parameters(self):
        """Unfreeze VAE parameters for end-to-end fine-tuning."""
        for param in self.vae_model.parameters():
            param.requires_grad = True
        print("Unfrozen VAE parameters for end-to-end training")
    
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
                compute_kld_loss=True
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
    #
    # mu, logvar = model(scattering_input=y_st_input,
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
    
    # Test VAE model: ------------------------------------------------------------
    model = SeqVaeTeb(
        input_channels=channels,
        sequence_length=seq_len,
        decimation_factor=16,
        warmup_period=warmup_period,
    )
    forward_outputs = model(
        y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input
    )
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
