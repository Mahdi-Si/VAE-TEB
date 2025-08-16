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
    """
    Encodes the target signal (y) to produce the parameters of the prior distribution p(z|y).

    The target signal is composed of two parts: scattering transform features (y_st) and 
    phase harmonic features (y_ph). These are processed through parallel MLP and causal convolution
    stacks, fused, and then passed through an LSTM to produce the final representations.

    The encoder outputs the mean (μ_y) and a composite log-variance vector. This vector is
    later split into the log-variance for the prior (log(σ^2_y)) and a conditioning
    feature (c_y) used by the ConditionalEncoder.

    **Mathematical Formulation:**

    The encoder models the prior distribution over the latent variable z, conditioned on y:
    $$ p(\mathbf{z}_t | \mathbf{y}_t) = \mathcal{N}(\mathbf{z}_t | \boldsymbol{\mu}^{y}_t, \text{diag}(\boldsymbol{\sigma}^{2,y}_t)) $$

    The encoder function f_t maps the input features to the parameters of this distribution:
    $$ (\boldsymbol{\mu}^{y}_t, [\log\boldsymbol{\sigma}^{2,y}_t, \mathbf{c}_t]) = f_t(\mathbf{y}^{st}_t, \mathbf{y}^{ph}_t) $$

    where:
    -  **y_st_t**: Scattering transform features of the target signal at time t.
    -  **y_ph_t**: Phase harmonic features of the target signal at time t.
    -  **μ_y_t**: The mean of the prior distribution.
    -  **log(o^2_y_t)**: The log-variance of the prior distribution.
    -  **c_t**: A conditioning feature passed to the ConditionalEncoder.
    """
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

        # Sequential convolutions for scattering with skip connections
        self.conv_scattering_1 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=3, dilation=1)
        self.conv_scattering_2 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=5, dilation=1)
        self.conv_scattering_3 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=7, dilation=1)
        
        # Skip connection normalization for scattering path
        self.scatter_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.scatter_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)

        # Sequential convolutions for phase with skip connections
        self.conv_phase_1 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=3, dilation=1)
        self.conv_phase_2 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=5, dilation=1)
        self.conv_phase_3 = CausalMultiChannelConvBlock(in_channels=16, out_channels=16, filter_size=7, dilation=1)
        
        # Skip connection normalization for phase path
        self.phase_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        self.phase_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 16), num_channels=16)
        
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
        
        # Separate layers for prior and conditioning features (TEB compliance)
        self.prior_logvar_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 32, 4),
            final_activation=False,
            activation=nn.ReLU
        )
        
        self.conditioning_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 32, 4),
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

        # Apply convolutions with skip connections for scattering path
        x_scatter = scatter_linear.transpose(1, 2)  # (B, C, L)
        
        # First conv block
        scatter_conv_1 = self.conv_scattering_1(x_scatter)
        
        # Second conv block with normalized skip connection
        scatter_conv_2 = self.conv_scattering_2(scatter_conv_1)
        skip_1_norm = self.scatter_skip_norm_1(scatter_conv_1)
        scatter_conv_2 = scatter_conv_2 + skip_1_norm  # Normalized skip connection
        
        # Third conv block with normalized skip connection
        scatter_conv_3 = self.conv_scattering_3(scatter_conv_2)
        skip_2_norm = self.scatter_skip_norm_2(scatter_conv_2)
        scatter_conv_3 = scatter_conv_3 + skip_2_norm  # Normalized skip connection
        
        scatter_conv = scatter_conv_3.transpose(1, 2).contiguous()  # Back to (B, L, C)
        scatter_conv = self.scatter_fused_norm(scatter_conv)
        del scatter_linear, x_scatter, scatter_conv_1, scatter_conv_2, scatter_conv_3

        # Apply convolutions with skip connections for phase path
        x_phase = phase_linear.transpose(1, 2)  # (B, C, L)
        
        # First conv block
        phase_conv_1 = self.conv_phase_1(x_phase)
        
        # Second conv block with normalized skip connection
        phase_conv_2 = self.conv_phase_2(phase_conv_1)
        skip_1_norm = self.phase_skip_norm_1(phase_conv_1)
        phase_conv_2 = phase_conv_2 + skip_1_norm  # Normalized skip connection
        
        # Third conv block with normalized skip connection
        phase_conv_3 = self.conv_phase_3(phase_conv_2)
        skip_2_norm = self.phase_skip_norm_2(phase_conv_2)
        phase_conv_3 = phase_conv_3 + skip_2_norm  # Normalized skip connection
        
        phase_conv = phase_conv_3.transpose(1, 2).contiguous()  # Back to (B, L, C)
        phase_conv = self.phase_fused_norm(phase_conv)
        del phase_linear, x_phase, phase_conv_1, phase_conv_2, phase_conv_3

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
        prior_logvar = self.prior_logvar_layer(x)
        conditioning_features = self.conditioning_layer(x)

        # Clamp only the prior logvar for numerical stability
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

    **Mathematical Formulation:**

    The encoder function f_s maps the source features x_t to a deterministic representation h^x_t:
    $$ \mathbf{h}^x_t = f_s(\mathbf{x}_t) $$

    where:
    - **x_t**: Source signal features at time t.
    - **h^x_t**: The deterministic latent representation of the source signal.
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
        
        # Sequential convolutions for source encoder with skip connections
        self.conv_1 = CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=3, dilation=1)
        self.conv_2 = CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=5, dilation=1)
        self.conv_3 = CausalMultiChannelConvBlock(in_channels=32, out_channels=32, filter_size=7, dilation=1)
        
        # Skip connection normalization for source encoder
        self.source_skip_norm_1 = nn.GroupNorm(num_groups=min(8, 32), num_channels=32)
        self.source_skip_norm_2 = nn.GroupNorm(num_groups=min(8, 32), num_channels=32)
        
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
        
        # Apply convolutions with skip connections
        x_conv = x_linear.transpose(1, 2)  # (B, C, L)
        
        # First conv block
        conv_1 = self.conv_1(x_conv)
        
        # Second conv block with normalized skip connection
        conv_2 = self.conv_2(conv_1)
        skip_1_norm = self.source_skip_norm_1(conv_1)
        conv_2 = conv_2 + skip_1_norm  # Normalized skip connection
        
        # Third conv block with normalized skip connection
        conv_3 = self.conv_3(conv_2)
        skip_2_norm = self.source_skip_norm_2(conv_2)
        conv_3 = conv_3 + skip_2_norm  # Normalized skip connection
        
        conv_out = conv_3.transpose(1, 2).contiguous()  # Back to (B, L, C)
        
        if return_intermediate:
            intermediates["conv_path"] = conv_out

        x = self.fused_norm(conv_out)
        del x_linear, x_conv, conv_1, conv_2, conv_3, conv_out  # Explicit cleanup

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

    This module models the posterior distribution over the latent variable z, conditioned on both
    the source signal (x) and the target signal (y). It takes the latent representation of the
    source (h_x) and a conditioning feature from the target (c_y) as input. It outputs the
    parameters of the posterior distribution, which is a diagonal Gaussian.

    **Mathematical Formulation:**

    The posterior distribution is defined as:
    $$ q(\mathbf{z}_t | \mathbf{x}_t, \mathbf{y}_t) = \mathcal{N}(\mathbf{z}_t | \boldsymbol{\mu}^{post}_t, \text{diag}(\boldsymbol{\sigma}^{2,post}_t)) $$

    The encoder function f_c computes the parameters of this distribution from the combined
    latent representations:
    $$ (\tilde{\boldsymbol{\mu}}^{post}_t, \log\boldsymbol{\sigma}^{2,post}_t) = f_c([\mathbf{h}^x_t, \mathbf{c}_t]) $$

    The final posterior mean is shifted by the prior mean to center it:
    $$ \boldsymbol{\mu}^{post}_t = \tilde{\boldsymbol{\mu}}^{post}_t + \boldsymbol{\mu}^{y}_t $$

    where:
    - **h^x_t**: Latent representation from the SourceEncoder.
    - **c_t**: Conditioning feature from the TargetEncoder.
    - **μ^post_t**: The mean of the posterior distribution.
    - **log(σ^2_post_t)**: The log-variance of the posterior distribution.
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
    Reconstructs the raw target signal and auxiliary features from the latent sequence z.

    The decoder takes the full latent sequence z as input and performs two tasks:
    1.  **Auxiliary Feature Reconstruction**: It predicts the concatenated target features 
        (scattering and phase harmonics) at each time step. This is used as an auxiliary
        loss to stabilize training.
    2.  **Raw Signal Reconstruction**: It upsamples the latent sequence and predicts the mean 
        and log-variance of the raw FHR signal over a fixed window.

    **Mathematical Formulation:**

    The decoder models the likelihood of the raw signal given the latent sequence:
    $$ p(\mathbf{r} | \mathbf{z}_{1:T}) = \mathcal{N}(\mathbf{r} | \boldsymbol{\mu}^{raw}, \text{diag}(\boldsymbol{\sigma}^{2,raw})) $$

    The decoder function f_d maps the latent sequence to the reconstruction outputs:
    $$ (\widehat{\mathbf{y}}_{1:T}, \boldsymbol{\mu}^{raw}, \log\boldsymbol{\sigma}^{2,raw}) = f_d(\mathbf{z}_{1:T}) $$

    where:
    - **z_{1:T}**: The full latent sequence.
    - **ŷ_{1:T}**: The reconstructed auxiliary features.
    - **μ^raw**: The mean of the reconstructed raw signal.
    - **log(σ^2_raw)**: The log-variance of the reconstructed raw signal.
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

        # Individual conv blocks for skip connections
        self.conv_1 = MultiChannelConvBlock(in_channels=87, out_channels=77, filter_size=11, up_sampling=False)
        self.conv_2 = MultiChannelConvBlock(in_channels=77, out_channels=66, filter_size=9, up_sampling=True)
        self.conv_3 = MultiChannelConvBlock(in_channels=66, out_channels=55, filter_size=7, up_sampling=True)
        self.conv_4 = MultiChannelConvBlock(in_channels=55, out_channels=44, filter_size=5, up_sampling=False)
        self.conv_5 = MultiChannelConvBlock(in_channels=44, out_channels=33, filter_size=5, up_sampling=True)
        self.conv_6 = MultiChannelConvBlock(in_channels=33, out_channels=22, filter_size=3, up_sampling=True)
        self.conv_7 = MultiChannelConvBlock(in_channels=22, out_channels=11, filter_size=3, up_sampling=False)
        self.conv_8 = MultiChannelConvBlock(in_channels=11, out_channels=1, filter_size=3, up_sampling=False)
        
        # Skip connection projection layers for dimension matching
        self.skip_proj_77_to_66 = nn.Conv1d(77, 66, kernel_size=1)  # For conv_1 -> conv_3
        self.skip_proj_55_to_44 = nn.Conv1d(55, 44, kernel_size=1)  # For conv_3 -> conv_4
        self.skip_proj_33_to_22 = nn.Conv1d(33, 22, kernel_size=1)  # For conv_5 -> conv_6
        
        # Normalization for decoder skip connections
        self.decoder_skip_norm_77 = nn.GroupNorm(num_groups=min(8, 77), num_channels=77)
        self.decoder_skip_norm_55 = nn.GroupNorm(num_groups=min(8, 55), num_channels=55)
        self.decoder_skip_norm_33 = nn.GroupNorm(num_groups=min(8, 33), num_channels=33)
        
        # Note: No encoder-decoder skip connections to preserve TEB information bottleneck
        
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
        Forward pass that reconstructs the raw signal from latent variables only.
        
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
        
        # Apply convolution layers with skip connections (no encoder-decoder skips)
        # Conv block 1: 87 -> 77
        x1 = self.conv_1(x)
        
        # Conv block 2: 77 -> 66 (with upsampling)
        x2 = self.conv_2(x1)
        
        # Conv block 3: 66 -> 55 (with upsampling) + skip from x1
        x3 = self.conv_3(x2)
        if x1.shape[-1] == x3.shape[-1]:  # Check if sequence lengths match after upsampling
            skip_x1_norm = self.decoder_skip_norm_77(x1)
            x3 = x3 + self.skip_proj_77_to_66(skip_x1_norm)  # Normalized skip connection with projection
        
        # Conv block 4: 55 -> 44 + skip from x3
        x4 = self.conv_4(x3)
        if x3.shape[-1] == x4.shape[-1]:  # Check if sequence lengths match
            skip_x3_norm = self.decoder_skip_norm_55(x3)
            x4 = x4 + self.skip_proj_55_to_44(skip_x3_norm)  # Normalized skip connection with projection
        
        # Conv block 5: 44 -> 33 (with upsampling)
        x5 = self.conv_5(x4)
        
        # Conv block 6: 33 -> 22 (with upsampling) + skip from x5
        x6 = self.conv_6(x5)
        if x5.shape[-1] == x6.shape[-1]:  # Check if sequence lengths match after upsampling
            skip_x5_norm = self.decoder_skip_norm_33(x5)
            x6 = x6 + self.skip_proj_33_to_22(skip_x5_norm)  # Normalized skip connection with projection
        
        # Conv block 7: 22 -> 11
        x7 = self.conv_7(x6)
        
        # Conv block 8: 11 -> 1
        x = self.conv_8(x7)  # (batch_size, 1, upsampled_length)
        
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
    Sequence VAE with Transfer Entropy Bottleneck (TEB).

    This model implements the full SeqVaeTeb framework, which learns a latent representation
    of a target signal (y) that is predictive of the signal's future, while minimizing the
    information it contains about a source signal (x). This is achieved by minimizing the
    KL divergence between a posterior distribution q(z|x,y) and a prior distribution p(z|y).

    **Core Components:**
    - **SourceEncoder**: Encodes the source signal `x` into a latent representation `h_x`.
    - **TargetEncoder**: Encodes the target signal `y` into the parameters of the prior `p(z|y)`.
    - **ConditionalEncoder**: Combines `h_x` and a feature from `y` to define the posterior `q(z|x,y)`.
    - **Decoder**: Reconstructs the target signal from samples of the latent variable `z`.

    **Mathematical Formulation:**

    The model is trained by maximizing the Evidence Lower Bound (ELBO), which is equivalent
    to minimizing the following loss function:

    $$ \mathcal{L}_{\text{total}} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x},\mathbf{y})}[-\log p(\mathbf{r}|\mathbf{z})] + \beta \cdot \text{KL}[q(\mathbf{z}|\mathbf{x},\mathbf{y}) || p(\mathbf{z}|\mathbf{y})] $$

    where:
    - The first term is the reconstruction loss (NLL of the raw signal + MSE of auxiliary features).
    - The second term is the KL divergence, which acts as a proxy for transfer entropy.
    - β is a hyperparameter that controls the strength of the information bottleneck.
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
        self.decoder = Decoder()  # No encoder features to preserve information bottleneck

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

    def compute_tc_loss(self, z, mu, logvar, dataset_size):
        """
        Computes the β-TCVAE loss components using Minibatch Weighted Sampling.
        
        Mathematical Foundation:
        - I_q(z;n) = E[log q(z|x,y) - log q(z)]
        - TC(z) = E[log q(z) - log ∏_j q(z_j)]  
        - DW_KL = E[log ∏_j q(z_j) - log p(z)]

        Args:
            z (torch.Tensor): Latent samples from the posterior. Shape: (batch_size, seq_len, latent_dim)
            mu (torch.Tensor): Mean of the posterior. Shape: (batch_size, seq_len, latent_dim)
            logvar (torch.Tensor): Log-variance of the posterior. Shape: (batch_size, seq_len, latent_dim)
            dataset_size (int): The total number of samples in the training dataset.

        Returns:
            dict: A dictionary containing mi_loss, tc_loss, and dw_kl_loss.
        """
        batch_size, seq_len, latent_dim = z.shape
        
        # Reshape for minibatch processing: (batch_size * seq_len, latent_dim)
        z_flat = z.reshape(batch_size * seq_len, latent_dim)
        mu_flat = mu.reshape(batch_size * seq_len, latent_dim)
        logvar_flat = logvar.reshape(batch_size * seq_len, latent_dim)
        
        # Numerical stability: clamp log variance
        logvar_flat = torch.clamp(logvar_flat, min=-10, max=10)
        
        num_samples = batch_size * seq_len
        
        # Log-density of the posterior q(z|x,y) for each sample
        log_q_z_xy = self._gaussian_log_density(z_flat, mu_flat, logvar_flat)

        # MWS: Compute log q(z_i) under all encoders q(z|x_j,y_j) in the minibatch
        # Shape: (num_samples, num_samples, latent_dim)
        z_expanded = z_flat.unsqueeze(1)  # (num_samples, 1, latent_dim)
        mu_expanded = mu_flat.unsqueeze(0)  # (1, num_samples, latent_dim)
        logvar_expanded = logvar_flat.unsqueeze(0)  # (1, num_samples, latent_dim)
        
        # Compute log densities: log q(z_i | x_j, y_j) for all i,j pairs
        _log_q_z = self._gaussian_log_density_broadcast(z_expanded, mu_expanded, logvar_expanded)
        # Shape: (num_samples, num_samples)

        # MWS estimator for log q(z): logsumexp over j, then normalize
        log_q_z = torch.logsumexp(_log_q_z, dim=1) - math.log(dataset_size * num_samples)

        # MWS estimator for log ∏_j q(z_j): sum over dimensions of individual logsumexp
        # For each dimension d: log q(z_d) = logsumexp_j(log q(z_d | x_j, y_j)) - log(N*M)
        # Compute marginal log densities for each dimension separately
        log_q_z_marginals = []
        for d in range(latent_dim):
            z_d = z_flat[:, d:d+1].unsqueeze(1)  # (num_samples, 1, 1)
            mu_d = mu_flat[:, d:d+1].unsqueeze(0)  # (1, num_samples, 1)
            logvar_d = logvar_flat[:, d:d+1].unsqueeze(0)  # (1, num_samples, 1)
            
            log_q_zd = self._gaussian_log_density_broadcast(z_d, mu_d, logvar_d)  # (num_samples, num_samples)
            log_q_zd_marginal = torch.logsumexp(log_q_zd, dim=1) - math.log(dataset_size * num_samples)  # (num_samples,)
            log_q_z_marginals.append(log_q_zd_marginal)
        
        log_q_z_marginals = torch.stack(log_q_z_marginals, dim=1)  # (num_samples, latent_dim)
        log_prod_q_z_j = log_q_z_marginals.sum(dim=1)  # (num_samples,)

        # Log-density of the prior p(z) = N(0, I)
        log_p_z = self._standard_normal_log_density(z_flat)

        # Decomposed loss terms (sign convention: positive = penalty)
        mi_loss = (log_q_z_xy - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z_j).mean()
        dw_kl_loss = (log_prod_q_z_j - log_p_z).mean()

        return {
            'mi_loss': mi_loss,
            'tc_loss': tc_loss,
            'dw_kl_loss': dw_kl_loss
        }

    def _gaussian_log_density(self, samples, mu, logvar):
        """
        Compute log density of samples under Gaussian distribution.
        
        Args:
            samples: (N, D) tensor
            mu: (N, D) tensor
            logvar: (N, D) tensor
        
        Returns:
            log_density: (N,) tensor
        """
        normalization = -0.5 * (math.log(2 * math.pi) + logvar)
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * ((samples - mu) ** 2 * inv_var)
        return log_density.sum(dim=-1)

    def _gaussian_log_density_broadcast(self, samples, mu, logvar):
        """
        Compute log density with broadcasting for MWS.
        
        Args:
            samples: (N, 1, D) tensor
            mu: (1, M, D) tensor  
            logvar: (1, M, D) tensor
        
        Returns:
            log_density: (N, M) tensor
        """
        normalization = -0.5 * (math.log(2 * math.pi) + logvar)
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * ((samples - mu) ** 2 * inv_var)
        return log_density.sum(dim=-1)

    def _standard_normal_log_density(self, samples):
        """
        Compute log density under standard normal N(0, I).
        
        Args:
            samples: (N, D) tensor
        
        Returns:
            log_density: (N,) tensor
        """
        return -0.5 * (math.log(2 * math.pi) + samples.pow(2)).sum(dim=1)

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

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor, 
        y_raw: torch.Tensor,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
        use_tcvae: bool = False,
        alpha: float = 1.0,
        gamma: float = 1.0,
        dataset_size: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the total training loss with MSE and NLL components.
        Supports both standard TEB and β-TCVAE loss computation.

        Args:
            forward_outputs: The dictionary returned by the forward pass.
            y_st: Target scattering coefficients from optimized dataloader (B, S=300, channels=43)
            y_ph: Target phase coefficients from optimized dataloader (B, S=300, channels=44)
            y_raw: Ground truth raw signal data from optimized dataloader (B, 4800)
            compute_kld_loss (bool): Whether to compute KLD loss.
            beta (float): Beta weight for KLD loss in VAE training.
            use_tcvae (bool): Whether to use β-TCVAE decomposed loss instead of standard KLD.
            alpha (float): Weight for Index-Code MI term in β-TCVAE.
            gamma (float): Weight for Dimension-wise KL term in β-TCVAE.
            dataset_size (int): Total dataset size for MWS computation.

        Returns:
            A dictionary of computed losses.
        """
        device = y_raw.device
        kld_loss = torch.tensor(0.0, device=device)
        mi_loss = torch.tensor(0.0, device=device)
        tc_loss = torch.tensor(0.0, device=device)
        dw_kl_loss = torch.tensor(0.0, device=device)

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

        # Choose loss computation method
        if compute_kld_loss:
            if use_tcvae:
                # β-TCVAE decomposed loss
                tc_loss_dict = self.compute_tc_loss(
                    z=forward_outputs['z'],
                    mu=forward_outputs['mu_post'],
                    logvar=forward_outputs['logvar_post'],
                    dataset_size=dataset_size
                )
                
                mi_loss = tc_loss_dict['mi_loss']
                tc_loss = tc_loss_dict['tc_loss']
                dw_kl_loss = tc_loss_dict['dw_kl_loss']
                
                # Total regularization loss
                regularization_loss = alpha * mi_loss + beta * tc_loss + gamma * dw_kl_loss
            else:
                # Standard TEB KLD loss
                kld_loss = self._kld_loss(
                    mu_prior=forward_outputs["mu_prior"],
                    logvar_prior=forward_outputs["logvar_prior"],
                    mu_post=forward_outputs["mu_post"],
                    logvar_post=forward_outputs["logvar_post"],
                    reduce_mean=True,  # Ensure scalar loss for training
                )
                regularization_loss = beta * kld_loss
        else:
            regularization_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = decoder_losses['total_decoder_loss'] + regularization_loss

        return {
            "reconstruction_loss": decoder_losses['total_decoder_loss'],  # For backward compatibility
            "mse_loss": decoder_losses['mse_loss'],
            "nll_loss": decoder_losses['nll_loss'], 
            "kld_loss": kld_loss,
            "mi_loss": mi_loss,
            "tc_loss": tc_loss,
            "dw_kl_loss": dw_kl_loss,
            "regularization_loss": regularization_loss,
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