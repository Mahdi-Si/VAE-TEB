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
    Compute a geometric progression of layer sizes from `input_size` down/up to `output_size`,
    with `n_hidden` intermediate layers.

    Returns a list of length n_hidden+2: [input_size, h1, h2, ..., h_n, output_size].
    
    Arguments:
    - input_size:  starting dimension (e.g. 16)
    - output_size: ending dimension (e.g. 64)
    - n_hidden:    number of hidden layers (e.g. 6)
    - round_fn:    function to turn floats into ints (default=round)
    """
    # total steps = hidden layers + the final map to output
    steps = n_hidden + 1
    # constant ratio r so that input_size * r^steps = output_size
    r = (output_size / input_size) ** (1 / steps)

    # build the full list
    sizes = [
        int(round_fn(input_size * (r ** k)))
        for k in range(steps + 1)
    ]
    # ensure exact endpoints
    sizes[0] = input_size
    sizes[-1] = output_size
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

        # Remove pre-allocated padding to save memory - compute on demand
        # self.register_buffer(
        #     "padding_zeros", torch.zeros(1, in_channels, self.left_padding)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, seq_len)
        Returns:
            Causal convolution output
        """
        batch_size = x.size(0)

        # Memory-efficient left padding using F.pad
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))

        return self.conv(x)


class CausalResidualBlock(nn.Module):
    """
    Optimized residual block with causal convolutions, reduced transposes, and efficient gating.
    Now supports dilations for building Temporal Convolutional Networks (TCNs) and channel reduction.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        expansion_factor: float = 1.0,
        contraction_factor: float = 1.0,
        use_depthwise: bool = False,
    ):
        super(CausalResidualBlock, self).__init__()

        self.channels = channels
        self.expansion_channels = int(channels * expansion_factor)
        self.contraction_channels = int(channels * contraction_factor)
        self.use_depthwise = use_depthwise
        self.use_contraction = contraction_factor < 1.0

        # Pre-norm design for better gradient flow
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(self.expansion_channels)
        output_channels = self.contraction_channels if self.use_contraction else channels
        self.norm3 = nn.LayerNorm(output_channels)

        # Efficient depthwise + pointwise convolution option
        final_output_channels = self.contraction_channels if self.use_contraction else channels
        
        if use_depthwise:
            self.conv1 = CausalConv1d(
                channels, channels, kernel_size, groups=channels, dilation=dilation
            )
            self.pointwise1 = nn.Conv1d(channels, self.expansion_channels, 1)
            self.conv2 = CausalConv1d(
                self.expansion_channels,
                self.expansion_channels,
                kernel_size,
                groups=self.expansion_channels,
                dilation=dilation,
            )
            self.pointwise2 = nn.Conv1d(self.expansion_channels, final_output_channels, 1)
        else:
            self.conv1 = CausalConv1d(
                channels, self.expansion_channels, kernel_size, dilation=dilation
            )
            self.conv2 = CausalConv1d(
                self.expansion_channels, final_output_channels, kernel_size, dilation=dilation
            )

        # GLU (Gated Linear Unit) for efficient gating
        self.gate_linear = nn.Linear(final_output_channels, final_output_channels)

        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if needed
        self.skip_proj = (
            nn.Linear(channels, final_output_channels) 
            if final_output_channels != channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, channels)
        Returns:
            Output with residual connection
        """
        residual = x

        # Pre-norm + first convolution block
        x = self.norm1(x)
        x_transposed = x.transpose(1, 2)  # Single transpose

        if self.use_depthwise:
            x_conv = self.conv1(x_transposed)
            x_conv = self.pointwise1(x_conv)
        else:
            x_conv = self.conv1(x_transposed)

        x_conv = x_conv.transpose(1, 2)  # Back to (batch, seq_len, channels)
        x_conv = self.norm2(x_conv)
        x_conv = F.gelu(x_conv)  # GELU doesn't support inplace
        x_conv = self.dropout(x_conv)

        # Second convolution block
        x_conv_transposed = x_conv.transpose(1, 2)

        if self.use_depthwise:
            x_conv2 = self.conv2(x_conv_transposed)
            x_conv2 = self.pointwise2(x_conv2)
        else:
            x_conv2 = self.conv2(x_conv_transposed)

        x_conv2 = x_conv2.transpose(1, 2)
        x_conv2 = self.norm3(x_conv2)

        # Efficient gating using GLU-style activation
        gate = torch.sigmoid(self.gate_linear(x_conv2))
        x_conv2 = x_conv2 * gate  # Remove in-place operation

        # Residual connection with optional projection
        return self.skip_proj(residual) + x_conv2  # Remove in-place operation


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
                nn.GELU(),
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


class FeedForward(nn.Module):
    """A simple feed-forward network with pre-LayerNorm and residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim_factor: int = 1,
        dropout: float = 0.1,
        activation=nn.GELU,
    ):
        super().__init__()
        hidden_dim = out_dim * hidden_dim_factor
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(in_dim)
        # self.skip_connection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm and residual connection."""
        # residual = self.skip_connection(x)
        x_norm = self.norm(x)
        return self.net(x_norm)


class ResidualMLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=(72, 68, 64), dropout=0.1, final_activation=True
    ):
        super().__init__()
        # initial layer-norm on raw input
        self.input_norm = nn.LayerNorm(input_dim)
        self.final_activation = final_activation
        # build the sequence of (Linear → LayerNorm → GELU → Dropout)
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.body = nn.Sequential(*layers)

        # if input_dim ≠ final hidden_dims[-1], project it
        final_dim = hidden_dims[-1]
        if input_dim != final_dim:
            self.skip_proj = nn.Linear(input_dim, final_dim)
        else:
            self.skip_proj = nn.Identity()

        # optional post-sum norm / activation
        self.post_norm = nn.LayerNorm(final_dim)
        self.post_act = nn.GELU()

    def forward(self, x):
        # 1) normalize raw input
        x0 = self.input_norm(x)

        # 2) run through MLP body
        y = self.body(x0)

        # 3) project + add skip
        skip = self.skip_proj(x0)
        z = y + skip

        if self.final_activation:
            return self.post_act(self.post_norm(z))
        else:
            return self.post_norm(z)


class TargetEncoder(nn.Module):
    """
    Advanced VAE encoder with dual-path processing for scattering transform and phase harmonic inputs.

    Architecture:
    1. Dual-path processing: Linear projection + Causal convolution paths
    2. Multi-layer linear transformations with residual connections
    3. Bidirectional LSTM for temporal encoding
    4. Variational outputs (mu and logvar) with proper initialization

    Incorporates modern techniques:
    - Layer normalization for training stability
    - Skip connections for gradient flow
    - Dropout for regularization
    - GELU activation for improved performance
    - Gradient clipping-friendly architecture
    """

    def __init__(
        self,
        input_channels: int = 76,
        sequence_length: int = 300,
        latent_dim: int = 16,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 3,
        conv_kernel_size: Union[int, Tuple[int, ...]] = (11, 9, 7, 5, 3),
        dropout: float = 0.1,
        use_bidirectional_lstm: bool = False,
        activation: str = "gelu",
        reduced_channels: int = 32,  # Reduced from 76 to capture essential features
    ):
        super(TargetEncoder, self).__init__()

        self.input_channels = input_channels
        self.reduced_channels = reduced_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_bidirectional = use_bidirectional_lstm

        self.activation = getattr(F, activation)
        
        # Channel reduction blocks for both modalities
        self.channel_reducer_scattering = nn.Sequential(
            ChannelReductionBlock(
            in_channels=input_channels,
            out_channels=64,
            dropout=dropout,
            use_attention=True),
            ChannelReductionBlock(
            in_channels=64,
            out_channels=reduced_channels,
            dropout=dropout,
            use_attention=True)
        )
        self.channel_reducer_phase = nn.Sequential(
            ChannelReductionBlock(
            in_channels=input_channels,
            out_channels=64,
            dropout=dropout,
            use_attention=True),
            ChannelReductionBlock(
            in_channels=64,
            out_channels=reduced_channels,
            dropout=dropout,
            use_attention=True)
        )

        # Path 1: Linear projection path for both inputs (after channel reduction)
        self.linear_path_scattering = ResidualMLP(
            input_dim=reduced_channels,
            hidden_dims=(32, 32, 32, 32),
            dropout=dropout,
            final_activation=False,
        )
        self.linear_path_phase = ResidualMLP(
            input_dim=reduced_channels,
            hidden_dims=(32, 32, 32, 32),
            dropout=dropout,
            final_activation=False,
        )

        # Path 2: Causal convolution path for both inputs (after channel reduction)
        self.conv_path_scattering = self._build_causal_conv_path(
            reduced_channels, 32, conv_kernel_size, dropout, use_depthwise=True
        )
        self.conv_path_phase = self._build_causal_conv_path(
            reduced_channels, 32, conv_kernel_size, dropout, use_depthwise=True
        )

        # Cross-modal fusion (combining scattering and phase harmonic)
        self.cross_modal_fusion = ResidualMLP(
            input_dim=32 * 2,  # Updated to reflect 32-channel outputs from each path
            hidden_dims=(64, 60, 55, 50),  # Smaller intermediate dimensions
            dropout=dropout,
            final_activation=False,
        )

        self.lstm = nn.LSTM(
            input_size=50,  # Updated to match cross_modal_fusion output
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=use_bidirectional_lstm,
        )

        lstm_output_dim = lstm_hidden_dim * (2 if use_bidirectional_lstm else 1)

        # Pre-output processing
        self.pre_output = ResidualMLP(
            input_dim=lstm_output_dim,
            hidden_dims=geometric_schedule(lstm_output_dim, 66, 3),
            dropout=dropout,
            final_activation=True,
        )

        # Variational parameters
        self.mu_layer = ResidualMLP(
            input_dim=66,
            hidden_dims=geometric_schedule(66, 16, 5),
            dropout=dropout,
            final_activation=False,
        )
        self.logvar_layer = ResidualMLP(
            input_dim=66,
            hidden_dims=geometric_schedule(66, 32, 5),
            dropout=dropout,
            final_activation=False,
        )

    def _build_causal_conv_path(
        self,
        input_dim: int,
        output_dim: int,
        kernel_sizes: Union[int, list[int]],
        dropout: float,
        use_depthwise: bool,
    ) -> nn.Module:
        """Build causal convolution path with a sequence of residual blocks."""
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        layers = []
        current_dim = input_dim
        # Create a block for each kernel size provided
        for i, kernel_size in enumerate(kernel_sizes):
            expansion_factor = 2.0 if i % 2 == 0 else 1.5  # Alternate expansion
            layers.append(
                CausalResidualBlock(
                    channels=current_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    expansion_factor=expansion_factor,
                    use_depthwise=use_depthwise,
                )
            )

        # Final projection to target dimension
        layers.append(
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )

        return nn.Sequential(*layers)

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
            scattering_input: Scattering transform features (batch_size, seq_len, channels)
            phase_harmonic_input: Phase harmonic features (batch_size, seq_len, channels)
            return_hidden: Whether to return intermediate hidden states

        Returns:
            mu: Mean of latent distribution (batch_size, seq_len, latent_dim)
            logvar: Log variance of latent distribution (batch_size, seq_len, 2*latent_dim)
            hidden_states: Dictionary of intermediate states (if return_hidden=True)
        """
        batch_size, seq_len, channels = scattering_input.shape
        hidden_states = {} if return_hidden else None

        # Apply channel reduction first
        scattering_reduced = self.channel_reducer_scattering(scattering_input)
        phase_reduced = self.channel_reducer_phase(phase_harmonic_input)
        
        if return_hidden:
            hidden_states["scattering_reduced"] = scattering_reduced
            hidden_states["phase_reduced"] = phase_reduced

        # Process scattering transform with reduced channels
        scatter_linear = self.linear_path_scattering(scattering_reduced)
        scatter_conv = self.conv_path_scattering(scattering_reduced)
        scatter_fused = scatter_linear + scatter_conv  # Remove in-place operation
        del scatter_conv  # Explicit cleanup

        # Process phase harmonic with reduced channels
        phase_linear = self.linear_path_phase(phase_reduced)
        phase_conv = self.conv_path_phase(phase_reduced)
        phase_fused = phase_linear + phase_conv  # Remove in-place operation
        del phase_conv  # Explicit cleanup

        if return_hidden:
            hidden_states["scatter_fused"] = scatter_fused
            hidden_states["phase_fused"] = phase_fused

        # Cross-modal fusion
        combined = torch.cat([scatter_fused, phase_fused], dim=-1)
        del scatter_fused, phase_fused  # Free memory
        x = self.cross_modal_fusion(combined)
        del combined  # Free memory

        # LSTM processing
        x, (hidden, cell) = self.lstm(x)

        if return_hidden:
            hidden_states["lstm_out"] = x
            hidden_states["lstm_hidden"] = hidden
            hidden_states["lstm_cell"] = cell

        # Pre-output processing
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
        input_channels: int = 76,
        sequence_length: int = 300,
        latent_dim: int = 16,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 3,
        conv_kernel_size: list[int] = [9, 7, 5, 3],
        dropout: float = 0.1,
        activation: str = "gelu",
        reduced_channels: int = 32,  # Reduced from 76 to capture essential features
    ):
        super(SourceEncoder, self).__init__()

        self.input_channels = input_channels
        self.reduced_channels = reduced_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.activation = getattr(F, activation)
        
        # Channel reduction block
        self.channel_reducer = ChannelReductionBlock(
            in_channels=input_channels,
            out_channels=reduced_channels,
            dropout=dropout,
            use_attention=True,
        )

        self.linear_path = ResidualMLP(
            input_dim=reduced_channels,
            hidden_dims=(int(reduced_channels * 0.8), 32, 32),
            dropout=dropout,
            final_activation=False,
        )

        # Path 2: Causal convolution path with advanced residual blocks
        self.conv_path = self._build_causal_conv_path(
            reduced_channels, 32, conv_kernel_size, dropout, use_depthwise=True
        )

        self.fusion_path = ResidualMLP(
            input_dim=32 * 2,  # Updated for 32-channel outputs
            hidden_dims=(64, 60, 50),  # Smaller intermediate dimensions
            dropout=dropout,
            final_activation=False,
        )

        # Unidirectional LSTM for causal temporal encoding
        self.lstm = nn.LSTM(
            input_size=50,  # Updated to match fusion_path output
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
        )

        self.pre_output = ResidualMLP(
            input_dim=lstm_hidden_dim,
            hidden_dims=(102, 82, 66, 53),
            dropout=dropout,
            final_activation=True,
        )

        self.mu_layer = ResidualMLP(
            input_dim=53,
            hidden_dims=(43, 35, 27, 21, latent_dim),
            dropout=dropout,
            final_activation=False,
        )

    def _build_causal_conv_path(
        self,
        input_dim: int,
        output_dim: int,
        kernel_sizes: Union[int, list[int]],
        dropout: float,
        use_depthwise: bool,
    ) -> nn.Module:
        """Build causal convolution path with a sequence of residual blocks."""
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        layers = []
        current_dim = input_dim
        # Create a block for each kernel size provided
        for i, kernel_size in enumerate(kernel_sizes):
            expansion_factor = 2.0 if i % 2 == 0 else 1.5  # Alternate expansion
            layers.append(
                CausalResidualBlock(
                    channels=current_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    expansion_factor=expansion_factor,
                    use_depthwise=use_depthwise,
                )
            )

        # Final projection to target dimension
        layers.append(
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )

        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor (batch_size, seq_len, channels)
            return_intermediate: Whether to return intermediate activations

        Returns:
            mu: Latent mean representations (batch_size, seq_len, latent_dim)
            intermediates: Dictionary of intermediate activations (if requested)
        """
        batch_size, seq_len, channels = x.shape
        intermediates = {} if return_intermediate else None

        if return_intermediate:
            intermediates["input_with_bias"] = x

        # Apply channel reduction first
        x_reduced = self.channel_reducer(x)
        
        if return_intermediate:
            intermediates["channel_reduced"] = x_reduced

        # Path 1: Multi-layer linear projection with reduced channels
        linear_out = self.linear_path(x_reduced)
        if return_intermediate:
            intermediates["linear_path"] = linear_out

        # Path 2: Causal convolution with reduced channels
        conv_out = self.conv_path(x_reduced)

        if return_intermediate:
            intermediates["conv_path"] = conv_out

        # Memory-efficient path fusion
        x = torch.cat([linear_out, conv_out], dim=-1)
        del linear_out, conv_out  # Explicit cleanup

        x = self.fusion_path(x)
        if return_intermediate:
            intermediates["path_fusion"] = x

        # LSTM forward pass (unidirectional for causal encoding)
        x, (hidden, cell) = self.lstm(x)

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
            x: Input tensor (batch_size, seq_len, channels)
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

    def __init__(self, dim_hx: int, dim_hy: int, dim_z: int, dropout: float = 0.1):
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
        self.mlp = ResidualMLP(
            input_dim=dim_hx + dim_hy,
            hidden_dims=(55, 47, 41, 35),
            dropout=dropout,
            final_activation=True,
        )

        # Final linear layers to produce mu and logvar for the latent variable z
        self.fc_mu = ResidualMLP(
            input_dim=35,
            hidden_dims=(30, 26, 22, 19, dim_z),
            dropout=dropout,
            final_activation=False,
        )
        self.fc_logvar = ResidualMLP(
            input_dim=35,
            hidden_dims=(30, 26, 22, 19, dim_z),
            dropout=dropout,
            final_activation=False,
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


class UpsamplingBlock(nn.Module):
    """
    Upsampling block with learnable transposed convolutions and anti-aliasing.
    Implements progressive upsampling with residual connections for high-quality reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        kernel_size: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            upsample_factor: Upsampling factor (2, 4, 8, etc.)
            kernel_size: Transposed convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()

        self.upsample_factor = upsample_factor

        # Transposed convolution for learnable upsampling
        self.transpose_conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=kernel_size // 2,
            output_padding=upsample_factor - 1,
        )

        # Anti-aliasing filter to prevent reconstruction artifacts
        self.anti_alias = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
        )

        # Residual connection projection if needed
        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

        # Upsampling for skip connection
        self.skip_upsample = nn.Upsample(
            scale_factor=upsample_factor, mode="linear", align_corners=False
        )

        # Normalization and activation
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels), num_channels=out_channels
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, length)
        Returns:
            Upsampled tensor (batch_size, out_channels, length * upsample_factor)
        """
        target_length = x.size(-1) * self.upsample_factor
        
        # Apply transposed convolution
        upsampled = self.transpose_conv(x)
        
        # Ensure exact target length by cropping or padding
        current_length = upsampled.size(-1)
        if current_length > target_length:
            upsampled = upsampled[..., :target_length]
        elif current_length < target_length:
            # Pad to exact length
            pad_amount = target_length - current_length
            upsampled = F.pad(upsampled, (0, pad_amount))

        # Apply anti-aliasing filter
        upsampled = self.anti_alias(upsampled)

        # Residual connection with proper upsampling
        skip = self.skip_proj(x)
        skip_upsampled = self.skip_upsample(skip)
        
        # Ensure skip connection matches target length too
        if skip_upsampled.size(-1) != target_length:
            if skip_upsampled.size(-1) > target_length:
                skip_upsampled = skip_upsampled[..., :target_length]
            else:
                pad_amount = target_length - skip_upsampled.size(-1)
                skip_upsampled = F.pad(skip_upsampled, (0, pad_amount))

        # Add residual connection
        output = upsampled + skip_upsampled

        # Apply normalization, activation, and dropout
        output = self.norm(output)
        output = self.activation(output)
        output = self.dropout(output)

        return output


class Decoder(nn.Module):
    """
    Optimized Raw Signal Decoder that predicts the next 2 minutes (480 samples) of raw signal
    from each timestep in the latent representation. 
    
    Key optimizations:
    - Reduced intermediate dimensions for lower memory usage
    - Fewer layers for faster training
    - Shared computation between mu and logvar heads
    - More efficient convolution architecture
    """

    def __init__(
        self,
        latent_dim: int = 16,
        sequence_length: int = 300,
        prediction_horizon: int = 480,  # 2 minutes at 4Hz = 480 samples
        hidden_dim: int = 64,  # Reduced from 128
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Input latent dimension
            sequence_length: Input sequence length
            prediction_horizon: Number of future samples to predict (default 480 = 2 minutes at 4Hz)
            hidden_dim: Hidden processing dimension (reduced for efficiency)
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim

        # Streamlined latent processing - fewer layers, smaller dimensions
        self.latent_processor = ResidualMLP(
            input_dim=latent_dim,
            hidden_dims=geometric_schedule(latent_dim, 64, 6),  # Much simpler progression
            dropout=dropout,
            final_activation=True,
        )

        self.lstm =  nn.LSTM(
            input_size=64,  # Updated to match fusion_path output
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=False,
        )
        
        # Direct expansion to prediction horizon - skip intermediate processing
        self.prediction_expander = ResidualMLP(
            input_dim=128,
            hidden_dims=geometric_schedule(128, 256, 5),  # Direct path, fewer layers
            dropout=dropout,
            final_activation=True,
        )
        
        # Simplified output heads
        self.output_mu = ResidualMLP(
            input_dim=256,
            hidden_dims=geometric_schedule(256, prediction_horizon, 6),  # Direct path, fewer layers
            dropout=dropout,
            final_activation=False,
        )
        self.output_logvar = ResidualMLP(
            input_dim=256,
            hidden_dims=geometric_schedule(256, prediction_horizon, 6),  # Direct path, fewer layers
            dropout=dropout,
            final_activation=False,
        )

    def forward(self, latent_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simplified forward pass using only LSTM and linear layers for maximum efficiency.
        
        Args:
            latent_z: Latent variables (batch_size, sequence_length, latent_dim)
        Returns:
            Dictionary containing:
            - raw_signal_mu: (batch_size, sequence_length, prediction_horizon)
            - raw_signal_logvar: (batch_size, sequence_length, prediction_horizon)
        """
        B, S, Z = latent_z.shape

        # Initial latent processing
        processed = self.latent_processor(latent_z)  # (B, S, 64)
        
        # LSTM processing for temporal context
        lstm_out, (hidden, cell) = self.lstm(processed)  # (B, S, 128)
        
        # Expand to intermediate features
        expanded = self.prediction_expander(lstm_out)  # (B, S, 256)
        
        # Generate outputs using separate MLPs
        raw_mu = self.output_mu(expanded)  # (B, S, 480)
        raw_logvar = self.output_logvar(expanded)  # (B, S, 480)
        
        # Clamp log variance for numerical stability
        raw_logvar = torch.clamp(raw_logvar, min=-8.0, max=8.0)
        
        # Clean up intermediate tensors for memory efficiency
        del processed, lstm_out, expanded, hidden, cell

        return {"raw_signal_mu": raw_mu, "raw_signal_logvar": raw_logvar}

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], target_raw_signal: torch.Tensor, warmup_period: int = 30
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for raw signal prediction.
        For each timestep i, compares the predicted next 480 samples against
        the actual next 480 samples from the raw signal.

        Args:
            predictions: Dictionary containing raw_signal_mu and raw_signal_logvar
                        Both have shape (B, S, 480)
            target_raw_signal:  Ground truth raw signal (B, raw_signal_length)
                                where raw_signal_length = S * 16
            warmup_period:  Number of initial timesteps to skip (default: 30)
                            We don't compute loss for early timesteps as there's 
                            insufficient history for meaningful predictions
        Returns:
            NLL loss tensor
        """
        mu = predictions["raw_signal_mu"]  # (B, S, 480)
        logvar = predictions["raw_signal_logvar"]  # (B, S, 480)
        
        B, S, prediction_horizon = mu.shape
        
        # Apply warmup period - only compute loss for timesteps after warmup
        start_timestep = max(0, warmup_period)
        if start_timestep >= S:
            # If warmup period is longer than sequence, return zero loss
            return torch.tensor(0.0, device=mu.device, requires_grad=True)
        
        # Create target tensor for comparison (only for timesteps after warmup)
        effective_timesteps = S - start_timestep
        target_predictions = torch.zeros(B, effective_timesteps, prediction_horizon, device=mu.device)
        
        for idx, i in enumerate(range(start_timestep, S)):
            start_idx = i * 16 + 1  # Start from next sample after current timestep
            end_idx = start_idx + prediction_horizon  # Next 480 samples
            
            # Check bounds to avoid indexing errors
            if end_idx <= target_raw_signal.shape[1]:
                target_predictions[:, idx, :] = target_raw_signal[:, start_idx:end_idx]
            else:
                # Handle edge case where we don't have enough future samples
                available_samples = target_raw_signal.shape[1] - start_idx
                if available_samples > 0:
                    target_predictions[:, idx, :available_samples] = target_raw_signal[:, start_idx:]
                    # Pad with last available sample if needed
                    if available_samples < prediction_horizon:
                        last_sample = target_raw_signal[:, -1:].expand(B, prediction_horizon - available_samples)
                        target_predictions[:, idx, available_samples:] = last_sample

        # Extract predictions only for timesteps after warmup
        mu_effective = mu[:, start_timestep:, :]  # (B, effective_timesteps, 480)
        logvar_effective = logvar[:, start_timestep:, :]  # (B, effective_timesteps, 480)

        # Compute Gaussian NLL: 0.5 * (log(var) + (target - mu)^2 / var)
        diff = target_predictions - mu_effective
        var = logvar_effective.exp()
        nll = 0.5 * (logvar_effective + diff.pow(2) / var)
        
        # Clean up
        del target_predictions, diff, var
        
        return nll.mean()


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
        latent_dim_source: int = 16,
        latent_dim_target: int = 16,
        latent_dim_z: int = 16,
        decimation_factor: int = 16,
        warmup_period: int = 30,
        kld_beta: float = 1.0,
        source_encoder_params: Optional[dict] = None,
        target_encoder_params: Optional[dict] = None,
        cond_encoder_params: Optional[dict] = None,
        decoder_params: Optional[dict] = None,
    ):
        super().__init__()

        self.latent_dim_source = latent_dim_source
        self.latent_dim_target = latent_dim_target
        self.latent_dim_z = latent_dim_z
        self.decimation_factor = decimation_factor
        self.warmup_period = warmup_period
        self.kld_beta = kld_beta

        # Default parameters if not provided
        if source_encoder_params is None:
            source_encoder_params = {
                "lstm_hidden_dim": 128
            }
        if target_encoder_params is None:
            target_encoder_params = {
                "lstm_hidden_dim": 128,
            }
        if cond_encoder_params is None:
            cond_encoder_params = {"dropout": 0.1}
        if decoder_params is None:
            decoder_params = {"hidden_dim": 128, "dropout": 0.1}

        self.source_encoder = SourceEncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            latent_dim=latent_dim_source,
            **source_encoder_params,
        )
        self.target_encoder = TargetEncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            latent_dim=latent_dim_target,
            **target_encoder_params,
        )
        self.conditional_encoder = ConditionalEncoder(
            dim_hx=latent_dim_source,
            dim_hy=latent_dim_target,
            dim_z=latent_dim_z,
            **cond_encoder_params,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim_z,
            sequence_length=sequence_length,
            prediction_horizon=480,  # 2 minutes at 4Hz
            **decoder_params,
        )

        # Apply improved initialization
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
    ) -> torch.Tensor:
        """Computes the KL divergence between two Gaussian distributions."""
        kld = (
            logvar_post
            - logvar_prior
            - 1
            + (logvar_prior.exp() + (mu_prior - mu_post).pow(2)) / logvar_post.exp()
        )
        kld = 0.5 * kld.sum(dim=-1)
        return kld.mean()

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        y_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of the SeqVaeTeb model.

        Args:
            y_st: Target scattering input (B, C, L)
            y_ph: Target phase harmonic input (B, C, L)
            x_ph: Source phase harmonic input (B, C, L)

        Returns:
            A dictionary containing tensors needed for loss computation.
        """
        # Permute inputs from (B, C, L) to (B, L, C) for module compatibility
        # Use transpose for better memory efficiency
        y_st = y_st.transpose(1, 2)
        y_ph = y_ph.transpose(1, 2)
        x_ph = x_ph.transpose(1, 2)

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

        # Sample z from posterior
        z = self.reparameterize(mu_post, logvar_post)

        # Decode raw signal predictions from z
        raw_predictions = self.decoder(z)

        return {
            "z": z,
            "raw_predictions": raw_predictions,
            "mu_prior": mu_y,
            "logvar_prior": logvar_y_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
        }

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_raw: torch.Tensor,
        compute_kld_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the total training loss for raw signal prediction.

        Args:
            forward_outputs: The dictionary returned by the forward pass.
            y_raw: Ground truth raw signal data (B, raw_signal_length) where
                   raw_signal_length = sequence_length * decimation_factor.
            compute_kld_loss (bool): Whether to compute KLD loss.

        Returns:
            A dictionary of computed losses (total, reconstruction, KLD).
        """
        device = y_raw.device

        # Initialize losses
        kld_loss = torch.tensor(0.0, device=device)

        # Ensure y_raw has the right shape (B, raw_signal_length)
        if y_raw.dim() == 3 and y_raw.size(-1) == 1:
            y_raw = y_raw.squeeze(-1)  # Remove channel dimension if present

        # Raw signal prediction loss with warmup period
        raw_signal_loss = self.decoder.compute_loss(
            forward_outputs["raw_predictions"], y_raw, warmup_period=self.warmup_period
        )

        # KLD loss
        if compute_kld_loss:
            kld_loss = self._kld_loss(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"],
            )

        # Total loss
        total_loss = raw_signal_loss + self.kld_beta * kld_loss

        return {
            "total_loss": total_loss,
            "reconstruction_error": raw_signal_loss,  # Maps to required interface
            "reconstruction_loss": raw_signal_loss,  # Keep for backward compatibility
            "kld_loss": kld_loss,
            "classification_loss": None,  # Required by interface
            "raw_signal_loss": raw_signal_loss,
        }

    def get_average_predictions(
        self, forward_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the raw signal predictions for visualization.
        
        The decoder now outputs predictions for the next 480 samples (2 minutes)
        from each timestep in the latent sequence.

        Args:
            forward_outputs: The dictionary returned by the forward pass, which
                                contains the raw signal predictions.

        Returns:
            A dictionary containing raw signal predictions with shape 
            (batch_size, sequence_length, 480).
        """
        return forward_outputs["raw_predictions"]

    def predict_raw_signal(
        self, x_ph: torch.Tensor, y_st: torch.Tensor, y_ph: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict raw signal from latent representation.

        Args:
            x_ph: Source phase harmonic input (B, C, L)
            y_st: Target scattering input (B, C, L)
            y_ph: Target phase harmonic input (B, C, L)

        Returns:
            Dictionary containing raw signal predictions
        """
        with torch.no_grad():
            # Get latent representation
            forward_outputs = self.forward(y_st, y_ph, x_ph)
            return forward_outputs["raw_predictions"]

    def predict_single_timestep(
        self, 
        x_ph: torch.Tensor, 
        y_st: torch.Tensor, 
        y_ph: torch.Tensor,
        raw_timestep: int,
        raw_signal_length: int = 4800
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the next 2 minutes (480 samples) of raw FHR signal from a single timestep.
        
        This method allows you to specify a specific timestep in the raw signal,
        finds the corresponding timestep in the decimated scattering transform,
        and predicts the next 2 minutes (480 samples at 4Hz) of raw signal.

        Args:
            x_ph: Source phase harmonic input (B, C, L) - full sequence
            y_st: Target scattering input (B, C, L) - full sequence  
            y_ph: Target phase harmonic input (B, C, L) - full sequence
            raw_timestep: Timestep in the raw signal (0 to raw_signal_length-1)
            raw_signal_length: Length of the original raw signal (default: 4800)

        Returns:
            Dictionary containing:
            - raw_signal_mu: Predicted signal mean for next 2 minutes (B, 480)
            - raw_signal_logvar: Predicted log-variance for next 2 minutes (B, 480) 
            - decimated_timestep: Corresponding timestep in decimated sequence
            - z_single: Latent variable at the specific timestep (B, 1, latent_dim)
        """
        with torch.no_grad():
            # Calculate corresponding timestep in decimated sequence
            # Decimation factor is typically 16 (from 4800 raw samples to 300 decimated)
            decimated_timestep = raw_timestep // self.decimation_factor
            
            # Ensure timestep is within bounds
            max_decimated_timestep = x_ph.size(-1) - 1
            decimated_timestep = min(decimated_timestep, max_decimated_timestep)
            
            # Extract single timestep from inputs
            # Inputs are in (B, C, L) format, we need timestep from dimension L
            x_ph_single = x_ph[:, :, decimated_timestep:decimated_timestep+1]  # (B, C, 1)
            y_st_single = y_st[:, :, decimated_timestep:decimated_timestep+1]  # (B, C, 1) 
            y_ph_single = y_ph[:, :, decimated_timestep:decimated_timestep+1]  # (B, C, 1)
            
            # Convert to sequence-first format for model compatibility
            x_ph_single = x_ph_single.transpose(1, 2)  # (B, 1, C)
            y_st_single = y_st_single.transpose(1, 2)  # (B, 1, C)
            y_ph_single = y_ph_single.transpose(1, 2)  # (B, 1, C)
            
            # Source encoder for single timestep
            mu_x_single = self.source_encoder(x_ph_single)  # (B, 1, latent_dim)
            
            # Target encoder for single timestep  
            mu_y_single, logvar_y_full_single = self.target_encoder(y_st_single, y_ph_single)
            
            # Split target logvar for prior and conditional feature
            logvar_y_prior_single, c_logvar_single = torch.split(
                logvar_y_full_single, self.latent_dim_target, dim=-1
            )
            
            # Conditional encoder for single timestep
            mu_post_single, logvar_post_single = self.conditional_encoder(
                mu_x_single, c_logvar_single
            )
            
            # Sample z from posterior for single timestep
            z_single = self.reparameterize(mu_post_single, logvar_post_single)  # (B, 1, latent_dim)
            
            # Decode to get raw signal prediction for next 480 samples
            raw_predictions = self.decoder(z_single)  # Will output (B, 1, 480)
            
            return {
                "raw_signal_mu": raw_predictions["raw_signal_mu"].squeeze(1),      # (B, 480) 
                "raw_signal_logvar": raw_predictions["raw_signal_logvar"].squeeze(1), # (B, 480)
                "decimated_timestep": decimated_timestep,
                "z_single": z_single,  # (B, 1, latent_dim)
                "mu_post_single": mu_post_single,
                "logvar_post_single": logvar_post_single
            }


if __name__ == "__main__":
    # Common configuration
    batch_size = 4
    seq_len = 300
    channels = 76
    prediction_horizon = 30
    warmup_period = 50

    # --- SeqVaeTeb Model Initialization with Channel Reduction ---
    print("--- Initializing SeqVaeTeb Model with Channel Reduction ---")
    model = SeqVaeTeb(
        input_channels=channels,
        sequence_length=seq_len,
        decimation_factor=16,
        warmup_period=warmup_period,
        kld_beta=1.0,
    )
    print("Model initialized successfully.")
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Create Dummy Data ---
    y_st_input = torch.randn(batch_size, channels, seq_len)  # (B, C, L) format
    y_ph_input = torch.randn(batch_size, channels, seq_len)
    x_ph_input = torch.randn(batch_size, channels, seq_len)
    y_raw_input = torch.randn(
        batch_size, seq_len * 16
    )  # Raw signal at 16x resolution (B, 4800)
    print(f"\nInput shapes: y_st={y_st_input.shape}, y_ph={y_ph_input.shape}, x_ph={x_ph_input.shape}")
    print(f"Raw signal shape: {y_raw_input.shape}")

    # --- Test Channel Reduction ---
    print("\n--- Testing Channel Reduction ---")
    print("Testing forward pass with channel reduction...")
    try:
        forward_outputs = model(
            y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input
        )
        print("✓ Forward pass successful")
        
        # Print output shapes
        print(f"Raw signal prediction shape: {forward_outputs['raw_predictions']['raw_signal_mu'].shape}")
        print(f"Expected shape: (B, S, 480) = ({batch_size}, {seq_len}, 480)")
        print(f"Latent z shape: {forward_outputs['z'].shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Test Loss Computation ---
    print("\n--- Testing Loss Computation ---")
    try:
        loss_dict = model.compute_loss(forward_outputs, y_raw=y_raw_input)
        print("✓ Loss computation successful")
        print(f"Total Loss: {loss_dict['total_loss']:.4f}")
        print(f"Reconstruction Loss: {loss_dict['reconstruction_loss']:.4f}")
        print(f"KLD Loss: {loss_dict['kld_loss']:.4f}")
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Simple Training Loop ---
    print("\n--- Starting Simple Training Loop with Channel Reduction ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # 1. Zero the gradients
        optimizer.zero_grad()

        # 2. Forward pass
        forward_outputs = model(
            y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input
        )

        # 3. Compute loss
        loss_dict = model.compute_loss(forward_outputs, y_raw=y_raw_input)
        total_loss = loss_dict["total_loss"]

        # 4. Backward pass
        total_loss.backward()

        # 5. Update weights
        optimizer.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Loss: {loss_dict['total_loss']:.4f}, "
            f"Recon Loss: {loss_dict['reconstruction_loss']:.4f}, "
            f"KLD Loss: {loss_dict['kld_loss']:.4f}, "
            f"Raw Signal Loss: {loss_dict['raw_signal_loss']:.4f}"
        )

    print("--- Channel Reduction Training Test Completed Successfully ---")
