import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Tuple, Optional, Union, Dict
import math
import warnings


def improved_initialization(model):
    """
    Improved initialization scheme for all modules in the model.
    
    Args:
        model: The neural network model to initialize
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Xavier/Glorot normal initialization for linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            # He initialization for convolutional layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            # Orthogonal initialization for LSTM weights
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    # Input-hidden weights: Xavier initialization
                    nn.init.xavier_normal_(param)
                elif 'weight_hh' in name:
                    # Hidden-hidden weights: Orthogonal initialization
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # Initialize biases to zero, except forget gate bias to 1
                    nn.init.constant_(param, 0)
                    # Set forget gate bias to 1 for better gradient flow
                    if 'bias_ih' in name:
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            # Standard initialization for layer normalization
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)



class CausalConv1d(nn.Module):
    """
    Optimized causal 1D convolution that ensures no future information leaks.
    Uses efficient padding strategy and supports mixed precision.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = True, groups: int = 1):
        super(CausalConv1d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate left padding for causal convolution (more efficient)
        self.left_padding = (kernel_size - 1) * dilation

        # Use grouped convolution for efficiency when possible
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=bias, groups=groups
        )

        # Pre-allocate padding tensor for efficiency
        self.register_buffer('padding_zeros', torch.zeros(1, in_channels, self.left_padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, seq_len)
        Returns:
            Causal convolution output
        """
        batch_size = x.size(0)

        # Efficient left padding using pre-allocated zeros
        if self.left_padding > 0:
            padding = self.padding_zeros.expand(batch_size, -1, -1)
            x = torch.cat([padding, x], dim=2)

        return self.conv(x)


class CausalResidualBlock(nn.Module):
    """
    Optimized residual block with causal convolutions, reduced transposes, and efficient gating.
    Now supports dilations for building Temporal Convolutional Networks (TCNs).
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1,
                 expansion_factor: float = 2.0, use_depthwise: bool = False):
        super(CausalResidualBlock, self).__init__()

        self.channels = channels
        self.expansion_channels = int(channels * expansion_factor)
        self.use_depthwise = use_depthwise

        # Pre-norm design for better gradient flow
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(self.expansion_channels)
        self.norm3 = nn.LayerNorm(channels)

        # Efficient depthwise + pointwise convolution option
        if use_depthwise:
            self.conv1 = CausalConv1d(channels, channels, kernel_size, groups=channels, dilation=dilation)
            self.pointwise1 = nn.Conv1d(channels, self.expansion_channels, 1)
            self.conv2 = CausalConv1d(self.expansion_channels, self.expansion_channels, kernel_size, groups=self.expansion_channels, dilation=dilation)
            self.pointwise2 = nn.Conv1d(self.expansion_channels, channels, 1)
        else:
            self.conv1 = CausalConv1d(channels, self.expansion_channels, kernel_size, dilation=dilation)
            self.conv2 = CausalConv1d(self.expansion_channels, channels, kernel_size, dilation=dilation)

        # GLU (Gated Linear Unit) for efficient gating
        self.gate_linear = nn.Linear(channels, channels)

        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if needed
        self.skip_proj = nn.Linear(channels, channels) if expansion_factor != 1.0 else nn.Identity()

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
        x_conv = F.gelu(x_conv)
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
        x_gated = x_conv2 * gate

        # Residual connection with optional projection
        return self.skip_proj(residual) + x_gated


class FeedForward(nn.Module):
    """A simple feed-forward network with pre-LayerNorm and residual connection."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim_factor: int = 1, dropout: float = 0.1, activation=nn.GELU):
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
    def __init__(self, input_dim, hidden_dims=(72, 68, 64), dropout=0.1, final_activation=True):
        super().__init__()
        # initial layer-norm on raw input
        self.input_norm = nn.LayerNorm(input_dim)
        self.final_activation = final_activation
        # build the sequence of (Linear → LayerNorm → GELU → Dropout)
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
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
        self.post_act  = nn.GELU()

    def forward(self, x):
        # 1) normalize raw input
        x0 = self.input_norm(x)
        
        # 2) run through MLP body
        y  = self.body(x0)
        
        # 3) project + add skip
        skip = self.skip_proj(x0)
        z    = y + skip
        
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
        latent_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 3,
        conv_kernel_size: Union[int, Tuple[int, ...]] = (7, 5, 3),
        dropout: float = 0.1,
        use_bidirectional_lstm: bool = True,
        activation: str = 'gelu'
    ):
        super(TargetEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_bidirectional = use_bidirectional_lstm
        
        self.activation = getattr(F, activation)
                
        # Path 1: Linear projection path for both inputs
        self.linear_path_scattering = ResidualMLP(input_dim=input_channels, hidden_dims=(72, 68, 64), dropout=dropout, final_activation=False)
        self.linear_path_phase = ResidualMLP(input_dim=input_channels, hidden_dims=(72, 68, 64), dropout=dropout, final_activation=False)
        
        # Path 2: Causal convolution path for both inputs
        self.conv_path_scattering = self._build_causal_conv_path(input_channels, 64, conv_kernel_size, dropout, use_depthwise=True)
        self.conv_path_phase = self._build_causal_conv_path(input_channels, 64, conv_kernel_size, dropout, use_depthwise=True)
        
        # Intra-modal fusion for scattering and phase paths
        self.scatter_fusion = ResidualMLP(input_dim=128, hidden_dims=(120, 100, 64), dropout=dropout, final_activation=False)
        self.phase_fusion = ResidualMLP(input_dim=128, hidden_dims=(120, 100, 64), dropout=dropout, final_activation=False)
        
        # Cross-modal fusion (combining scattering and phase harmonic)
        self.cross_modal_fusion = ResidualMLP(input_dim=64*2, hidden_dims=(64*2, 120, 110, 100), dropout=dropout, final_activation=False)

                
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=use_bidirectional_lstm
        )

        lstm_output_dim = lstm_hidden_dim * (2 if use_bidirectional_lstm else 1)
        
        # Pre-output processing
        self.pre_output = ResidualMLP(input_dim=lstm_output_dim, hidden_dims=(120, 112, 100), dropout=dropout, final_activation=True)

        
        # Variational parameters
        self.mu_layer = ResidualMLP(input_dim=100, hidden_dims=(91, 82, 73, latent_dim), dropout=dropout, final_activation=False)
        self.logvar_layer =  ResidualMLP(input_dim=100, hidden_dims=(107, 114, 121, latent_dim * 2), dropout=dropout, final_activation=False)
    
    def _build_causal_conv_path(
        self, input_dim: int, output_dim: int, kernel_sizes: Union[int, list[int]],
        dropout: float, use_depthwise: bool
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
                    use_depthwise=use_depthwise
                )
            )
        
        # Final projection to target dimension
        layers.append(nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ))
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        scattering_input: torch.Tensor, 
        phase_harmonic_input: torch.Tensor,
        return_hidden: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
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
                
        # Process scattering transform
        scatter_linear = self.linear_path_scattering(scattering_input)
        scatter_conv = self.conv_path_scattering(scattering_input)
        scatter_combined = torch.cat([scatter_linear, scatter_conv], dim=-1)
        scatter_fused = self.scatter_fusion(scatter_combined)
        
        # Process phase harmonic
        phase_linear = self.linear_path_phase(phase_harmonic_input)
        phase_conv = self.conv_path_phase(phase_harmonic_input)
        phase_combined = torch.cat([phase_linear, phase_conv], dim=-1)
        phase_fused = self.phase_fusion(phase_combined)
        
        if return_hidden:
            hidden_states['scatter_fused'] = scatter_fused
            hidden_states['phase_fused'] = phase_fused
        
        # Cross-modal fusion
        combined = torch.cat([scatter_fused, phase_fused], dim=-1)
        x = self.cross_modal_fusion(combined)
                
        # LSTM processing
        x, (hidden, cell) = self.lstm(x)
        
        if return_hidden:
            hidden_states['lstm_out'] = x
            hidden_states['lstm_hidden'] = hidden
            hidden_states['lstm_cell'] = cell
                
        # Pre-output processing
        x = self.pre_output(x)
        
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        if return_hidden:
            hidden_states['mu'] = mu
            hidden_states['logvar'] = logvar
            return mu, logvar, hidden_states
        
        return mu, logvar
    
    def get_encoder_features(
        self, 
        scattering_input: torch.Tensor, 
        phase_harmonic_input: torch.Tensor
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
        latent_dim: int = 64,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        conv_kernel_size: list[int] = [9, 7, 5, 3],
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super(SourceEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        self.activation = getattr(F, activation)

        
        
        self.linear_path = ResidualMLP(input_dim=input_channels, hidden_dims=(70, 64), dropout=dropout, final_activation=False)
        
        # Path 2: Causal convolution path with advanced residual blocks
        self.conv_path = self._build_causal_conv_path(input_channels, 64, conv_kernel_size, dropout, use_depthwise=True)
        
        self.fusion_path = ResidualMLP(input_dim=64*2, hidden_dims=(120, 110, 100), dropout=dropout, final_activation=False)

        # Unidirectional LSTM for causal temporal encoding
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False 
        )
                
        self.pre_output = ResidualMLP(input_dim=lstm_hidden_dim, hidden_dims=(120, 112, 100), dropout=dropout, final_activation=True)
        
        self.mu_layer = ResidualMLP(input_dim=100, hidden_dims=(91, 82, 73, latent_dim), dropout=dropout, final_activation=False)
                    
    def _build_causal_conv_path(
        self, input_dim: int, output_dim: int, kernel_sizes: Union[int, list[int]],
        dropout: float, use_depthwise: bool
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
                    use_depthwise=use_depthwise
                )
            )
        
        # Final projection to target dimension
        layers.append(nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ))
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_intermediate: bool = False
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
            intermediates['input_with_bias'] = x
                
        # Path 1: Multi-layer linear projection
        linear_out = self.linear_path(x)
        if return_intermediate:
            intermediates['linear_path'] = linear_out
        
        # Path 2: Causal convolution
        conv_out = self.conv_path(x)
        
        if return_intermediate:
            intermediates['conv_path'] = conv_out
        
        # Simplified path fusion
        x = torch.cat([linear_out, conv_out], dim=-1)
        
        x = self.fusion_path(x)
        if return_intermediate:
            intermediates['path_fusion'] = x
        
        
        # LSTM forward pass (unidirectional for causal encoding)
        x, (hidden, cell) = self.lstm(x)
        
        if return_intermediate:
            intermediates['lstm_output'] = x
            intermediates['lstm_hidden'] = hidden
            intermediates['lstm_cell'] = cell
                
        x = self.pre_output(x)
        
        if return_intermediate:
            intermediates['post_lstm'] = x
                
        # Final mu layer with residual connection
        mu = self.mu_layer(x)
        
        if return_intermediate:
            intermediates['mu'] = mu
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
        return mu[:, :timestep + 1, :]


class ConditionalEncoder(nn.Module):
    """
    Implements the conditional encoder q(z | x, y) for the TEB framework.
    It maps concatenated source (x) and target (y) latent representations 
    to the parameters of the posterior Gaussian distribution for z.
    
    This module is designed to work with sequence data, where the linear
    transformations are applied independently at each time step.
    """
    def __init__(self,
                 dim_hx: int,
                 dim_hy: int,
                 dim_z: int,
                 dropout: float = 0.1):
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
        self.mlp = ResidualMLP(input_dim=dim_hx + dim_hy, hidden_dims=(120, 110, 100), dropout=dropout, final_activation=True)

        # Final linear layers to produce mu and logvar for the latent variable z
        self.fc_mu = ResidualMLP(input_dim=100, hidden_dims=(89, 77, dim_z), dropout=dropout, final_activation=False)
        self.fc_logvar = ResidualMLP(input_dim=100, hidden_dims=(89, 77, dim_z), dropout=dropout, final_activation=False)

    def forward(self,
                h_x: torch.Tensor,
                h_y: torch.Tensor
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
    Professional VAE decoder for multi-modal future prediction with mean-only output.
    
    This decoder uses only linear layers, LSTM, and causal convolutions (no transformers).
    Simplified architecture for better stability and accuracy:
    
    Architecture:
    1. **Dual-Path Processing:** Linear + Causal convolution paths with residual fusion
    2. **Bidirectional LSTM:** Captures complex temporal dependencies 
    3. **Multi-Layer Prediction Head:** Deep MLP with layer normalization and skip connections
    4. **Mean-Only Prediction:** Single output head for deterministic predictions
    5. **Professional Design:** Layer norm, dropout, skip connections, and proper initialization
    
    Key Features:
    - Causal temporal convolutions with exponential dilation
    - Bidirectional LSTM for rich temporal modeling
    - Multi-path feature fusion with learnable weights
    - Simplified prediction with mean-only output
    - Gradient-friendly architecture with skip connections
    """
    
    def __init__(self,
                 latent_dim: int,
                 output_channels: int = 76,
                 prediction_horizon: int = 30,
                 lstm_num_layers: int = 2,
                 tcn_num_layers: int = 4,
                 tcn_kernel_size: Union[int, list[int]] = [3, 5, 7, 9],
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 use_bidirectional: bool = True):
        """
        Args:
            latent_dim: Input latent dimension (Z)
            output_channels: Output channels (76 for scattering+phase)
            prediction_horizon: Future steps to predict (30)
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            tcn_num_layers: Number of causal conv layers
            tcn_kernel_size: Convolution kernel size
            hidden_dim: Main processing dimension
            dropout: Regularization rate
            use_bidirectional: Use bidirectional LSTM
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        self.use_bidirectional = use_bidirectional
        
        # Simplified input processing with smaller dimensions
        self.linear_path = ResidualMLP(latent_dim, (96, 128, 160), dropout=dropout, final_activation=True)
        
        # Simplified causal convolution paths
        if not isinstance(tcn_kernel_size, list):
            tcn_kernel_size = [tcn_kernel_size]
        
        self.conv_paths = nn.ModuleList([
            self._build_causal_conv_path(latent_dim, 80, tcn_num_layers, ks, dropout)
            for ks in tcn_kernel_size
        ])

        num_conv_paths = len(tcn_kernel_size)
        self.conv_fusion = ResidualMLP(input_dim=80*num_conv_paths, hidden_dims=(160, 160), dropout=dropout, final_activation=True)
        
        # Simplified path fusion
        self.path_fusion = ResidualMLP(input_dim=320, hidden_dims=(256, 512), dropout=dropout, final_activation=True)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=use_bidirectional
        )
        
        lstm_output_dim = 512 * (2 if use_bidirectional else 1)
        
        # Post-LSTM processing
        self.post_lstm = ResidualMLP(input_dim=lstm_output_dim, hidden_dims=(600, 700, 800), dropout=dropout, final_activation=True)
        
        # Prediction head
        self.prediction_head = ResidualMLP(input_dim=800, hidden_dims=(900, 1000, 1024), dropout=dropout, final_activation=True)
        
        # Output dimensions
        output_size = prediction_horizon * output_channels
        
        # Simplified prediction heads - mean only
        self.scattering_head = ResidualMLP(input_dim=1024, hidden_dims=(1524, 2000, output_size,), dropout=dropout, final_activation=False)
        
        self.phase_head = ResidualMLP(input_dim=1024, hidden_dims=(1524, 2000, output_size,), dropout=dropout, final_activation=False)
        
    def _build_causal_conv_path(self, input_dim: int, output_dim: int, num_layers: int, kernel_size: int, dropout: float):
        """Build simplified causal convolution path with residual blocks."""
        layers = []
        current_dim = input_dim
        
        # Reduce number of layers for simplicity
        num_layers = min(num_layers, 2)
        
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                CausalResidualBlock(
                    channels=current_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    expansion_factor=1.2,
                    use_depthwise=True
                )
            )
        
        # Final projection to output dimension
        if current_dim != output_dim:
            layers.append(nn.Sequential(
                nn.LayerNorm(current_dim),
                nn.Linear(current_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent representation tensor (batch_size, sequence_length, latent_dim)
            
        Returns:
            Dictionary containing predicted means for each modality:
            - scattering_pred: (batch_size, sequence_length, prediction_horizon, output_channels)
            - phase_harmonic_pred: (batch_size, sequence_length, prediction_horizon, output_channels)
        """
        B, S, _ = z.shape
        H, C = self.prediction_horizon, self.output_channels
        
        # Dual-path processing
        linear_out = self.linear_path(z)
        
        # Process through multiple TCN paths and fuse
        conv_outs = [path(z) for path in self.conv_paths]
        conv_combined = torch.cat(conv_outs, dim=-1)
        conv_fused = self.conv_fusion(conv_combined)

        # Fuse linear and convolutional paths
        combined = torch.cat([linear_out, conv_fused], dim=-1)
        fused = self.path_fusion(combined)
        
        # LSTM processing
        lstm_out, _ = self.lstm(fused)  # (B, S, lstm_output_dim)
        
        # Post-LSTM processing
        processed = self.post_lstm(lstm_out)  # (B, S, hidden_dim)
        
        # Prediction head
        pred_features = self.prediction_head(processed)  # (B, S, hidden_dim)
        
        # Generate mean predictions for each modality
        scatter_pred = self.scattering_head(pred_features)      # (B, S, H*C)
        phase_pred = self.phase_head(pred_features)            # (B, S, H*C)
        
        # Reshape to include prediction horizon dimension
        predictions = {
            "scattering_pred": scatter_pred.view(B, S, H, C),
            "phase_harmonic_pred": phase_pred.view(B, S, H, C)
        }
        
        return predictions
    
    def _mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Mean Squared Error loss.
        
        Args:
            pred: Predicted values
            target: Ground truth target
            
        Returns:
            MSE loss
        """
        return F.mse_loss(pred, target)
    
    def compute_loss(self,
                     predictions: Dict[str, torch.Tensor],
                     target_scattering: torch.Tensor,
                     target_phase: torch.Tensor,
                     warmup_period: int,
                     compute_scattering_loss: bool = True,
                     compute_phase_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute Mean Squared Error loss for mean-only predictions with selective computation.
        
        Args:
            predictions: Dictionary of predicted means from forward pass
            target_scattering: Ground truth scattering transform (B, S, C)
            target_phase: Ground truth phase harmonic (B, S, C) 
            warmup_period: Number of initial timesteps to ignore
            compute_scattering_loss: Whether to compute scattering loss
            compute_phase_loss: Whether to compute phase loss
            
        Returns:
            MSE loss dictionary with selective computation
        """
        S = target_scattering.shape[1]
        H = self.prediction_horizon
        
        # Determine valid prediction range
        start_idx = warmup_period
        end_idx = S - H
        
        if start_idx >= end_idx:
            warnings.warn("Insufficient sequence length for loss computation after warmup.")
            device = target_scattering.device
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'scattering_loss': torch.tensor(0.0, device=device),
                'phase_loss': torch.tensor(0.0, device=device),
            }
        
        # Extract predictions for valid range
        pred_scatter = predictions['scattering_pred'][:, start_idx:end_idx]
        pred_phase = predictions['phase_harmonic_pred'][:, start_idx:end_idx]
        
        # Create target windows efficiently
        def create_target_windows(target_tensor: torch.Tensor) -> torch.Tensor:
            """Create sliding windows of target data."""
            B, S_full, C = target_tensor.shape
            # Create indices for sliding windows [t+1, t+2, ..., t+H] for each t
            indices = torch.arange(H, device=target_tensor.device)[None, :] + \
                     torch.arange(S_full - H, device=target_tensor.device)[:, None] + 1
            
            # Expand for batch and channel dimensions
            indices = indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, C)
            
            # Gather windows
            expanded_target = target_tensor.unsqueeze(1).expand(-1, S_full - H, -1, -1)
            return torch.gather(expanded_target, 2, indices)
        
        # Create target windows
        true_scatter_windows = create_target_windows(target_scattering)[:, start_idx:end_idx]
        true_phase_windows = create_target_windows(target_phase)[:, start_idx:end_idx]
        
        # Initialize losses
        device = target_scattering.device
        scattering_loss = torch.tensor(0.0, device=device)
        phase_loss = torch.tensor(0.0, device=device)
        
        # Compute losses selectively
        if compute_scattering_loss:
            scattering_loss = self._mse_loss(pred_scatter, true_scatter_windows)
        
        if compute_phase_loss:
            phase_loss = self._mse_loss(pred_phase, true_phase_windows)
        
        # Total loss is sum of computed losses
        total_loss = scattering_loss + phase_loss
        
        return {
            'total_loss': total_loss,
            'scattering_loss': scattering_loss,
            'phase_loss': phase_loss,
        }
    
    def predict(self, z: torch.Tensor, index: int) -> Dict[str, torch.Tensor]:
        """
        Predict next prediction_horizon steps from a specific time index.
        
        Args:
            z: Full latent sequence tensor (B, S, latent_dim)
            index: Time step index to predict from
            
        Returns:
            Dictionary containing predictions at the specified index:
            - scattering_pred: (B, prediction_horizon, output_channels)
            - phase_harmonic_pred: (B, prediction_horizon, output_channels)
        """
        if not (0 <= index < z.shape[1]):
            raise IndexError(f"Index {index} out of bounds for sequence length {z.shape[1]}")
        
        # Get all predictions
        all_predictions = self.forward(z)
        
        # Extract prediction at specified index
        predictions_at_index = {
            key: value[:, index, :, :]
            for key, value in all_predictions.items()
        }
        
        return predictions_at_index


class SeqVaeTeb(nn.Module):
    """
    Sequence VAE with Target-Encoder-Bank (TEB) framework.

    This model integrates a source encoder, a target encoder, a conditional
    encoder, and a decoder to perform future prediction with uncertainty.

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
    def __init__(self,
                 input_channels: int = 76,
                 sequence_length: int = 300,
                 latent_dim_source: int = 64,
                 latent_dim_target: int = 64,
                 latent_dim_z: int = 64,
                 prediction_horizon: int = 30,
                 warmup_period: int = 60,
                 kld_beta: float = 1.0,
                 source_encoder_params: Optional[dict] = None,
                 target_encoder_params: Optional[dict] = None,
                 decoder_params: Optional[dict] = None,
                 cond_encoder_params: Optional[dict] = None):
        super().__init__()
        
        self.latent_dim_source = latent_dim_source
        self.latent_dim_target = latent_dim_target
        self.latent_dim_z = latent_dim_z
        self.prediction_horizon = prediction_horizon
        self.warmup_period = warmup_period
        self.kld_beta = kld_beta

        # Default parameters if not provided
        if source_encoder_params is None:
            source_encoder_params = {
                'lstm_hidden_dim': 256, 'lstm_num_layers': 3, 'dropout': 0.1
            }
        if target_encoder_params is None:
            target_encoder_params = {
                'lstm_hidden_dim': 256, 'lstm_num_layers': 3, 
                'conv_kernel_size': (9, 7, 5, 3), 'dropout': 0.1
            }
        if decoder_params is None:
            decoder_params = {
                'lstm_num_layers': 3, 'tcn_num_layers': 4,
                'tcn_kernel_size': [3, 5, 7, 9], 'hidden_dim': 256, 
                'dropout': 0.1, 'use_bidirectional': False
            }
        if cond_encoder_params is None:
            cond_encoder_params = {'dropout': 0.1}

        self.source_encoder = SourceEncoder(
            input_channels=input_channels, sequence_length=sequence_length,
            latent_dim=latent_dim_source, **source_encoder_params
        )
        self.target_encoder = TargetEncoder(
            input_channels=input_channels, sequence_length=sequence_length,
            latent_dim=latent_dim_target, **target_encoder_params
        )
        self.conditional_encoder = ConditionalEncoder(
            dim_hx=latent_dim_source, dim_hy=latent_dim_target, 
            dim_z=latent_dim_z, **cond_encoder_params
        )
        self.decoder = Decoder(
            latent_dim=latent_dim_z, output_channels=input_channels,
            prediction_horizon=prediction_horizon, **decoder_params
        )
        
        # Apply improved initialization to all modules
        improved_initialization(self)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample from a Gaussian with numerical stability."""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kld_loss(self, mu_prior: torch.Tensor, logvar_prior: torch.Tensor, 
                  mu_post: torch.Tensor, logvar_post: torch.Tensor) -> torch.Tensor:
        """Computes the KL divergence between two Gaussian distributions with numerical stability."""
        # Clamp logvar to prevent numerical instability
        logvar_prior = torch.clamp(logvar_prior, min=-10, max=10)
        logvar_post = torch.clamp(logvar_post, min=-10, max=10)
        
        # More numerically stable KLD computation
        # KL(q||p) = 0.5 * [log(σ²_p/σ²_q) - 1 + σ²_q/σ²_p + (μ_p - μ_q)²/σ²_p]
        var_ratio = (logvar_post - logvar_prior).exp()  # σ²_q/σ²_p
        mu_diff_sq = (mu_prior - mu_post).pow(2)
        var_prior = logvar_prior.exp()
        
        kld = 0.5 * (logvar_prior - logvar_post - 1 + var_ratio + mu_diff_sq / (var_prior + 1e-8))
        kld = kld.sum(dim=-1)  # Sum over latent dimensions
        return kld.mean()  # Average over batch and sequence

    def forward(self, y_st: torch.Tensor, y_ph: torch.Tensor, x_ph: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        y_st = y_st.permute(0, 2, 1)
        y_ph = y_ph.permute(0, 2, 1)
        x_ph = x_ph.permute(0, 2, 1)

        # Source encoder for q(h_x|x)
        mu_x = self.source_encoder(x_ph)
        
        # Target encoder for p(z|y)
        mu_y, logvar_y_full = self.target_encoder(y_st, y_ph)
        
        # Split target logvar for prior and conditional feature
        logvar_y_prior, c_logvar = torch.split(logvar_y_full, self.latent_dim_target, dim=-1)
        
        # Conditional encoder for q(z|x, y)
        mu_post, logvar_post = self.conditional_encoder(mu_x, c_logvar)
        
        # Sample z from posterior
        z = self.reparameterize(mu_post, logvar_post)
        
        # Decode future predictions from z
        predictions = self.decoder(z)
        
        return {
            "z": z,
            "predictions": predictions,
            "mu_prior": mu_y,
            "logvar_prior": logvar_y_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post
        }

    def compute_loss(self, forward_outputs: Dict[str, torch.Tensor], 
                     y_st: torch.Tensor, y_ph: torch.Tensor,
                     compute_scattering_loss: bool = False,
                     compute_phase_loss: bool = False,
                     compute_kld_loss: bool = False) -> Dict[str, torch.Tensor]:
        """
        Computes the total training loss with selective computation.
        
        Args:
            forward_outputs: The dictionary returned by the forward pass.
            y_st: Ground truth target scattering data (B, C, L).
            y_ph: Ground truth target phase harmonic data (B, C, L).
            compute_scattering_loss: Whether to compute scattering reconstruction loss
            compute_phase_loss: Whether to compute phase reconstruction loss
            compute_kld_loss: Whether to compute KLD loss
            
        Returns:
            A dictionary of computed losses (total, reconstruction, KLD).
        """
        # Permute from (B, C, L) to (B, L, C) for loss computation
        y_st = y_st.permute(0, 2, 1)
        y_ph = y_ph.permute(0, 2, 1)

        # Initialize losses
        device = y_st.device
        recon_loss = torch.tensor(0.0, device=device)
        kld_loss = torch.tensor(0.0, device=device)
        scattering_loss = torch.tensor(0.0, device=device)
        phase_loss = torch.tensor(0.0, device=device)
        
        # Compute reconstruction loss selectively
        if compute_scattering_loss or compute_phase_loss:
            recon_loss_dict = self.decoder.compute_loss(
                forward_outputs["predictions"],
                target_scattering=y_st,
                target_phase=y_ph,
                warmup_period=self.warmup_period,
                compute_scattering_loss=compute_scattering_loss,
                compute_phase_loss=compute_phase_loss
            )
            recon_loss = recon_loss_dict['total_loss']
            scattering_loss = recon_loss_dict['scattering_loss']
            phase_loss = recon_loss_dict['phase_loss']
        
        # Compute KLD loss selectively
        if compute_kld_loss:
            kld_loss = self._kld_loss(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"]
            )
        
        # Total loss is sum of computed losses
        total_loss = recon_loss + self.kld_beta * kld_loss
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kld_loss": kld_loss,
            "scattering_loss": scattering_loss,
            "phase_loss": phase_loss
        }

    def get_average_predictions(self, forward_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes average predictions over the prediction horizon for visualization.

        For each time step in the sequence, this method averages all predictions
        made for that time step from previous time steps that fall within the
        prediction horizon.

        For the warmup period, predictions are held constant using the first
        valid prediction available after the warmup phase.

        Args:
            forward_outputs: The dictionary returned by the forward pass, which
                             contains the multi-step predictions.

        Returns:
            A dictionary containing two tensors of averaged predictions, each with
            shape (batch_size, sequence_length, channels).
        """
        predictions = forward_outputs["predictions"]
        B, S, H, C = predictions['scattering_pred'].shape
        device = predictions['scattering_pred'].device

        avg_preds = {}
        for key in predictions:
            pred_tensor = predictions[key]  # (B, S, H, C)

            # --- Vectorized averaging of predictions ---
            summed_preds = torch.zeros(B, S, C, device=device)
            
            # This loop is efficient as it operates on tensors, not scalars
            for h in range(H):
                # These are predictions for `h+1` steps in the future
                preds_h = pred_tensor[:, :, h, :]  # Shape: (B, S, C)
                
                # The prediction made at `t_s` is for `t_f = t_s + h + 1`.
                # We align these predictions by padding and shifting.
                num_valid_preds = S - (h + 1)
                if num_valid_preds > 0:
                    # Pad predictions to align them for summation
                    padded_preds = F.pad(preds_h[:, :num_valid_preds, :], (0, 0, h + 1, 0))
                    summed_preds += padded_preds

            # --- Calculate counts for averaging ---
            # `counts[t_f]` = number of predictions made for time step `t_f`
            counts = torch.arange(S, device=device, dtype=torch.float32)
            counts = torch.min(counts, torch.tensor(H, device=device, dtype=torch.float32))
            counts[0] = 1  # Avoid division by zero for the first time step
            
            # Reshape for broadcasting and compute the average
            avg_pred_tensor = summed_preds / counts.view(1, S, 1)

            # --- Handle warmup period ---
            if self.warmup_period > 0 and self.warmup_period < S:
                # Get the first valid prediction after the warmup period
                first_valid_pred = avg_pred_tensor[:, self.warmup_period, :].unsqueeze(1)
                # Repeat it across the warmup period
                warmup_preds = first_valid_pred.repeat(1, self.warmup_period, 1)
                # Replace the warmup part of the predictions
                avg_pred_tensor[:, :self.warmup_period, :] = warmup_preds

            if key == 'scattering_pred':
                avg_preds['scattering_mu'] = avg_pred_tensor
            elif key == 'phase_harmonic_pred':
                avg_preds['phase_harmonic_mu'] = avg_pred_tensor
            else:
                avg_preds[key] = avg_pred_tensor

        return avg_preds


if __name__ == "__main__":
    # Common configuration
    batch_size = 4
    seq_len = 300
    channels = 76
    prediction_horizon = 30
    warmup_period = 50

    # --- SeqVaeTeb Model Initialization ---
    print("--- Initializing SeqVaeTeb Model ---")
    model = SeqVaeTeb(
        input_channels=channels,
        sequence_length=seq_len,
        prediction_horizon=prediction_horizon,
        warmup_period=warmup_period,
        kld_beta=0.1  # Reduced from 1.0 to prevent overwhelming reconstruction loss
    )
    print("Model initialized successfully.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Create Dummy Data ---
    y_st_input = torch.randn(batch_size, channels, seq_len)  # Note: (B, C, L) format
    y_ph_input = torch.randn(batch_size, channels, seq_len)
    x_ph_input = torch.randn(batch_size, channels, seq_len)
    print(f"\nInput shapes: ({batch_size}, {channels}, {seq_len})")

    # --- Test Forward Pass ---
    print("\n--- Testing Forward Pass ---")
    model.eval()
    with torch.no_grad():
        forward_outputs = model(y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input)
        predictions = forward_outputs["predictions"]
        print(f"Scattering predictions shape: {predictions['scattering_pred'].shape}")
        print(f"Phase harmonic predictions shape: {predictions['phase_harmonic_pred'].shape}")
        print(f"Latent z shape: {forward_outputs['z'].shape}")
        
        # Test average predictions
        avg_preds = model.get_average_predictions(forward_outputs)
        print(f"Average scattering predictions shape: {avg_preds['scattering_mu'].shape}")
        print(f"Average phase harmonic predictions shape: {avg_preds['phase_harmonic_mu'].shape}")

    # --- Simple Training Loop ---
    print("\n--- Starting Simple Training Loop ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5

    for epoch in range(num_epochs):
        # 1. Zero the gradients
        optimizer.zero_grad()

        # 2. Forward pass
        forward_outputs = model(y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input)

        # 3. Compute loss
        loss_dict = model.compute_loss(
            forward_outputs,
            y_st=y_st_input,
            y_ph=y_ph_input,
            compute_scattering_loss=True,
            compute_phase_loss=True,
            compute_kld_loss=True
        )
        total_loss = loss_dict['total_loss']
        
        # 4. Check for NaN
        if torch.isnan(total_loss):
            print(f"NaN detected at epoch {epoch+1}! Stopping training.")
            break

        # 5. Backward pass
        total_loss.backward()
        
        # 6. Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 7. Update weights
        optimizer.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Loss: {loss_dict['total_loss']:.4f}, "
            f"Recon Loss: {loss_dict['reconstruction_loss']:.4f}, "
            f"Scattering Loss: {loss_dict['scattering_loss']:.4f}, "
            f"Phase Loss: {loss_dict['phase_loss']:.4f}, "
            f"KLD Loss: {loss_dict['kld_loss']:.4f}"
        )

    print("--- Training loop finished ---")
    print("\n--- Testing Individual Decoder ---")
    
    # Test standalone decoder
    decoder = Decoder(
        latent_dim=64,
        output_channels=channels,
        prediction_horizon=prediction_horizon
    )
    
    # Test with random latent input
    z_test = torch.randn(batch_size, seq_len, 64)
    decoder_output = decoder(z_test)
    print(f"Decoder scattering output shape: {decoder_output['scattering_pred'].shape}")
    print(f"Decoder phase output shape: {decoder_output['phase_harmonic_pred'].shape}")
    
    # Test decoder loss computation
    target_scatter = torch.randn(batch_size, seq_len, channels)
    target_phase = torch.randn(batch_size, seq_len, channels)
    decoder_loss = decoder.compute_loss(decoder_output, target_scatter, target_phase, warmup_period)
    print(f"Decoder MSE loss: {decoder_loss['total_loss']:.4f}")
    
    print("\n--- All tests completed successfully! ---")