import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any, Sequence

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

try:
    from torch._dynamo.eval_frame import OptimizedModule as _TorchOptimizedModule
except Exception:
    _TorchOptimizedModule = tuple()

DEFAULT_COMPILE_ATTEMPTS: Tuple[Dict[str, Any], ...] = (
    {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": False,
        "dynamic": True,
    },
    {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
        "options": {"triton.cudagraphs": False},
    },
)

def is_compiled_module(module: nn.Module) -> bool:
    """Return True if `module` is already wrapped by torch.compile."""
    if module is None:
        return False
    if hasattr(module, "_orig_mod"):
        return True
    try:
        return isinstance(module, _TorchOptimizedModule) if _TorchOptimizedModule else False
    except TypeError:
        return False

def ensure_compiled_module(
    module: nn.Module,
    *,
    compile_flag: bool = True,
    module_name: str = "module",
    attempts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[nn.Module, bool]:
    """Wrap `module` with torch.compile using the provided attempts."""
    if module is None:
        return module, False
    if not compile_flag:
        return module, is_compiled_module(module)
    if not hasattr(torch, "compile"):
        log.warning(f"[{module_name}] torch.compile unavailable in this PyTorch build; running in eager mode")
        return module, False
    if is_compiled_module(module):
        return module, True

    compile_attempts = tuple(attempts) if attempts is not None else DEFAULT_COMPILE_ATTEMPTS
    for opts in compile_attempts:
        try:
            compiled = torch.compile(module, **opts)
            setattr(compiled, "_compile_options", opts)
            log.info(f"[{module_name}] Compiled with torch.compile options={opts}")
            return compiled, True
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[{module_name}] torch.compile failed with options={opts}: {exc}")

    log.warning(f"[{module_name}] Falling back to eager mode after all torch.compile attempts failed")
    return module, False

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

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
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
        bias: bool = False
    ):
        super(CausalMultiChannelConvBlock, self).__init__()
        
        self.up_sampling = up_sampling
        self.up_sample_scale = up_sample_scale
        self.filter_size = filter_size
        self.activation = activation
        self.dilation = dilation
        self.stride = stride
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
            padding=0,  # We handle padding manually
            dilation=dilation,
            stride=stride
        )
        # Activation function
        self.act_fn = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Cin, L)
        Returns:
            (B, Cout, L') with strict causality
        """
        # Apply upsampling first
        if self.up_sampling:
            x = F.interpolate(
                x, 
                scale_factor=self.up_sample_scale, 
                mode='linear', 
                align_corners=False
            )
        # Apply pre-normalization
        x = self.pre_norm(x)  # (B, Cin, L or upsampled L)
        x = self.act_fn(x)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        output = self.conv(x)  # (B, Cout, L')
        return output

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
    """
    Residual MLP operating per time step with optional skip connection.

    Shapes
    - Input:  x  (B, L, Cin)
    - Output: y  (B, L, Cout) where Cout = hidden_dims[-1]
    """
    def __init__(
        self, input_dim, hidden_dims=(72, 68, 64), final_activation=True, activation=nn.GELU, use_skip_connection=True
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.final_activation = final_activation
        self.activation = activation
        self.use_skip_connection = use_skip_connection
        # Sequence of (Linear → LayerNorm → activation → Dropout)
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(hidden_dims)):
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

        final_dim = hidden_dims[-1]
        if self.use_skip_connection:
            if input_dim != final_dim:
                self.skip_proj = nn.Linear(input_dim, final_dim)
            else:
                self.skip_proj = nn.Identity()
        else:
            self.skip_proj = None

        self.final_act = self.activation() if final_activation else None

    def forward(self, x):
        # x: (B, L, Cin)
        x0 = self.input_norm(x)          # (B, L, Cin)

        y = self.body(x0)                # (B, L, Cout)

        if self.final_activation:
            y = self.final_act(y)       # (B, L, Cout)

        if self.use_skip_connection:
            skip = self.skip_proj(x0)   # (B, L, Cout)
            z = y + skip               # (B, L, Cout)
        else:
            z = y

        return z                         # (B, L, Cout)



class TargetEncoder(nn.Module):
    """
    Encodes the target signal (y) to produce the parameters of the prior distribution p(z|y).

    The target signal is composed of two parts: scattering transform features (y_st) and 
    phase harmonic features (y_ph). These are processed through parallel MLP and causal convolution
    stacks, fused, and then passed through an LSTM to produce the final representations.

    The encoder outputs the mean and a composite log-variance vector. This vector is
    later split into the log-variance for the prior and a conditioning
    feature used by the ConditionalEncoder.

    Shapes
    - Inputs:
        y_st: (B, T, 43)
        y_ph: (B, T, 44)
    - Outputs:
        mu:             (B, T, D)
        prior_logvar:   (B, T, D)
        cond_features:  (B, T, D)  # used by ConditionalEncoder
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
                activation=nn.GELU
                )

        self.mlp_phase = ResidualMLP(
            input_dim=44,
            hidden_dims=geometric_schedule(44, 16, 4),
            final_activation=False,
            use_skip_connection=True,
            activation=nn.GELU
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
            activation=nn.GELU,
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
            activation=nn.GELU
        )

        self.mu_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.GELU
        )
        
        self.prior_logvar_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.GELU
        )
        
        self.conditioning_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.GELU
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

        scatter_linear = self.mlp_scattering(scattering_input)      # (B, T, 16)
        phase_linear = self.mlp_phase(phase_harmonic_input)          # (B, T, 16)
        
        if return_hidden:
            hidden_states["scattering_reduced"] = scatter_linear
            hidden_states["phase_reduced"] = phase_linear

        x_scatter = scatter_linear.transpose(1, 2)  # (B, 16, T)
        
        scatter_conv_1 = self.conv_scattering_1(x_scatter)
        scatter_conv_2 = self.conv_scattering_2(scatter_conv_1)
        skip_1_norm = self.scatter_skip_norm_1(scatter_conv_1)
        scatter_conv_2 = scatter_conv_2 + skip_1_norm
        
        scatter_conv_3 = self.conv_scattering_3(scatter_conv_2)
        skip_2_norm = self.scatter_skip_norm_2(scatter_conv_2)
        scatter_conv_3 = scatter_conv_3 + skip_2_norm

        scatter_conv = scatter_conv_3.transpose(1, 2).contiguous()  # (B, T, 16)
        scatter_conv = self.scatter_fused_norm(scatter_conv)        # (B, T, 16)
        del scatter_linear, x_scatter, scatter_conv_1, scatter_conv_2, scatter_conv_3, skip_1_norm, skip_2_norm

        x_phase = phase_linear.transpose(1, 2)  # (B, 16, T)
        
        phase_conv_1 = self.conv_phase_1(x_phase)
        
        phase_conv_2 = self.conv_phase_2(phase_conv_1)
        skip_1_norm = self.phase_skip_norm_1(phase_conv_1)
        phase_conv_2 = phase_conv_2 + skip_1_norm
        
        phase_conv_3 = self.conv_phase_3(phase_conv_2)
        skip_2_norm = self.phase_skip_norm_2(phase_conv_2)
        phase_conv_3 = phase_conv_3 + skip_2_norm
        
        phase_conv = phase_conv_3.transpose(1, 2).contiguous()  # (B, T, 16)
        phase_conv = self.phase_fused_norm(phase_conv)          # (B, T, 16)
        del phase_linear, x_phase, phase_conv_1, phase_conv_2, phase_conv_3, skip_1_norm, skip_2_norm

        combined = torch.cat([scatter_conv, phase_conv], dim=-1)    # (B, T, 32)
        del scatter_conv, phase_conv
        x = self.cross_modal_fusion(combined)
        del combined

        x, (hidden, cell) = self.lstm(x)                           # (B, T, lstm_hidden_dim*(1 or 2))
        x = self.lstm_norm(x)                                      # (B, T, lstm_hidden_dim*(1 or 2))

        if return_hidden:
            hidden_states["lstm_out"] = x
            hidden_states["lstm_hidden"] = hidden
            hidden_states["lstm_cell"] = cell
        else:
            del hidden, cell

        x = self.pre_output(x)                                      # (B, T, 32)

        mu = self.mu_layer(x)                                       # (B, T, D)
        prior_logvar = self.prior_logvar_layer(x)                   # (B, T, D)
        conditioning_features = self.conditioning_layer(x)          # (B, T, D)

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

    Shapes
    - Input:  x_ph (B, T, 130)
    - Output: mu_x (B, T, D)
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
            activation=nn.GELU
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
            activation=nn.GELU
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.pre_output = ResidualMLP(
            input_dim=lstm_hidden_dim,
            hidden_dims=geometric_schedule(lstm_hidden_dim, 32, 4),
            final_activation=True,
            activation=nn.GELU
        )

        self.mu_layer = ResidualMLP(
            input_dim=32,
            hidden_dims=geometric_schedule(32, 16, 4),
            final_activation=False,
            activation=nn.GELU
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
        x_linear = self.mlp(x)                        # (B, T, 32)
        
        if return_intermediate:
            intermediates["channel_reduced"] = x_linear

        x_conv = x_linear.transpose(1, 2)  # (B, 32, T)
        conv_1 = self.conv_1(x_conv)

        conv_2 = self.conv_2(conv_1)
        skip_1_norm = self.source_skip_norm_1(conv_1)
        conv_2 = conv_2 + skip_1_norm
        
        conv_3 = self.conv_3(conv_2)
        skip_2_norm = self.source_skip_norm_2(conv_2)
        conv_3 = conv_3 + skip_2_norm
        
        conv_out = conv_3.transpose(1, 2).contiguous()  # (B, T, 32)
        
        if return_intermediate:
            intermediates["conv_path"] = conv_out

        x = self.fused_norm(conv_out)                 # (B, T, 32)
        x = self.linear_after_conv(x)                 # (B, T, 32)
        del x_linear, x_conv, conv_1, conv_2, conv_3, conv_out, skip_1_norm, skip_2_norm  # Explicit cleanup

        x, (hidden, cell) = self.lstm(x)              # (B, T, lstm_hidden_dim)
        x = self.lstm_norm(x)                         # (B, T, lstm_hidden_dim)

        if return_intermediate:
            intermediates["lstm_output"] = x
            intermediates["lstm_hidden"] = hidden
            intermediates["lstm_cell"] = cell
        else:
            del hidden, cell

        x = self.pre_output(x)                        # (B, T, 32)

        if return_intermediate:
            intermediates["post_lstm"] = x

        mu = self.mu_layer(x)                         # (B, T, D)

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
            hidden_dims=geometric_schedule(dim_hx + dim_hy, 20, 4),
            final_activation=True,
            use_skip_connection=True, 
            activation=nn.GELU,
        )

        self.fc_mu = ResidualMLP(
            input_dim=20,
            hidden_dims=geometric_schedule(20, 16, 4),
            final_activation=False,
            use_skip_connection=False, 
            activation=nn.GELU,
        )
        self.fc_logvar = ResidualMLP(
            input_dim=20,
            hidden_dims=geometric_schedule(20, 16, 4),
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
        del h_combined
        mu = self.fc_mu(h_merged)
        logvar = self.fc_logvar(h_merged)
        del h_merged
        return mu, logvar


class Decoder(nn.Module):
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

        self.feature_expansion_1 = ResidualMLP(
                input_dim=latent_dim,
                hidden_dims=geometric_schedule(latent_dim, 50, 5),
                final_activation=True,
                use_skip_connection=True, 
                activation=nn.GELU,
                )

        self.feature_expansion_2 = ResidualMLP(
                input_dim=50,
                hidden_dims=geometric_schedule(50, 87, 5),
                final_activation=True,
                activation=nn.GELU,
                use_skip_connection=True
                )

        self.skip_con_exp = nn.Linear(latent_dim, 87, bias=False)
        self.feature_expansion_layer_norm = nn.LayerNorm(87)
        
        self.pre_linear = ResidualMLP(
            input_dim=87,
            hidden_dims=geometric_schedule(87, 128, 3), 
            final_activation=False,
            activation=nn.GELU,
            use_skip_connection=True
        )

        self.upsample_1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm_1 = nn.GroupNorm(num_groups=8, num_channels=64)

        self.upsample_2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) 
        self.norm_2 = nn.GroupNorm(num_groups=8, num_channels=32)

        self.upsample_3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.norm_3 = nn.GroupNorm(num_groups=4, num_channels=16)

        self.upsample_4 = nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1)
        self.norm_4 = nn.GroupNorm(num_groups=2, num_channels=8)


        self.refine_conv = nn.Conv1d(8, 4, kernel_size=7, padding=3)
        self.final_conv = nn.Conv1d(4, 1, kernel_size=5, padding=2)

        self.signal_mu = nn.Conv1d(1, 1, kernel_size=1)
        self.signal_logvar = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, latent_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Progressive upsampling forward pass with strict information bottleneck preservation.
        
        This implementation forces all reconstruction to flow through the 32D latent bottleneck,
        preventing any information shortcuts. Research-backed architecture ensures optimal
        latent representation learning.
        
        Args:
            latent_z: Latent variables (B, L, D)
        Returns:
            Tuple containing:
            - linear_output: Auxiliary features (B, L, 87)
            - raw_signal_mu: Raw signal reconstruction mean (B, 16L)
            - raw_signal_logvar: Raw signal reconstruction log variance (B, 16L)
        """
        L = latent_z.size(1)
        z_expanded = self.feature_expansion_1(latent_z)
        z_expanded = self.feature_expansion_2(z_expanded)
        z_expanded = self.feature_expansion_layer_norm(z_expanded + self.skip_con_exp(latent_z)) # (B, L, 87)
        z_expanded_pre = self.pre_linear(z_expanded)  # (B, L, 128)
        z_conv = z_expanded_pre.transpose(1, 2)       # (B, 128, L) for conv operations
        del z_expanded_pre

        x1 = F.gelu(self.norm_1(self.upsample_1(z_conv)))      # (B, 64, 2L)
        del z_conv

        x2 = F.gelu(self.norm_2(self.upsample_2(x1)))          # (B, 32, 4L)  
        del x1 

        x3 = F.gelu(self.norm_3(self.upsample_3(x2)))          # (B, 16, 8L)
        del x2 

        x4 = F.gelu(self.norm_4(self.upsample_4(x3)))          # (B, 8, 16L)
        del x3

        refined = F.gelu(self.refine_conv(x4))                  # (B, 4, 16L)
        del x4
        features = self.final_conv(refined)                     # (B, 1, 16L)
        del refined

        mu = self.signal_mu(features).squeeze(1)                # (B, 16L)
        logvar = self.signal_logvar(features).squeeze(1)        # (B, 16L)
        del features

        logvar = torch.clamp(logvar, min=-10, max=10)

        return z_expanded, mu, logvar

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

        if compute_st_mse and linear_output.shape[-1] == 87 and target_fhr_st.shape[-1] == 43 and target_fhr_ph.shape[-1] == 44:
            stacked_target = torch.cat([target_fhr_st, target_fhr_ph], dim=-1)  # (B, S, 87)
            mse_loss = F.mse_loss(linear_output, stacked_target)
        else:
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if target_raw_signal.dim() == 3 and target_raw_signal.size(-1) == 1:
            target_raw_signal = target_raw_signal.squeeze(-1)  # Remove channel dimension if present

        steps = max(0, linear_output.size(1) - 1)
        if steps > 0:
            stride = max(1, raw_mu_predicted.size(1) // linear_output.size(1))
            idx = stride * (torch.arange(steps, device=device, dtype=torch.long) + 1)
            gather_idx = idx.unsqueeze(0).expand(raw_mu_predicted.size(0), -1)
            mu_next = torch.gather(raw_mu_predicted, 1, gather_idx)
            logvar_next = torch.gather(raw_logvar_predicted, 1, gather_idx)
            target_next = torch.gather(target_raw_signal, 1, gather_idx)
            var_next = logvar_next.exp()
            nll_loss = 0.5 * (logvar_next + (target_next - mu_next) ** 2 / var_next)
            nll_loss = nll_loss.mean()
        else:
            nll_loss = torch.tensor(0.0, device=device)

        return {
            'mse_loss': mse_loss,
            'nll_loss': nll_loss,
            'total_decoder_loss': mse_loss + nll_loss
        }


class LatentForecaster(nn.Module):
    """
    Simple autoregressive forecaster on latent space.

    - Encodes the past context z_{t-Lc+1:t} with a GRU encoder
    - Decodes future H steps autoregressively with a GRU decoder
    - Each step predicts a D-dimensional latent vector

    Shapes:
        context: (B, Lc, D)
        returns: (B, H, D)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = ResidualMLP(hidden_dim, geometric_schedule(hidden_dim, latent_dim, 4), final_activation=False)

    def forward(self, z_context: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Args:
            z_context: (B, Lc, D)
            horizon: number of future steps to predict (H)
        Returns:
            z_future: (B, H, D)
        """
        B, Lc, D = z_context.shape

        # Encode context
        _, h_n = self.encoder(z_context)  # h_n: (num_layers, B, hidden_dim)

        # Initialize decoder with last context latent as first input
        dec_input = z_context[:, -1:, :]  # (B, 1, D)
        h = h_n

        outputs = []
        for _ in range(horizon):
            out, h = self.decoder(dec_input, h)   # out: (B, 1, hidden_dim)
            z_hat = self.proj(out)                # (B, 1, D)
            outputs.append(z_hat)
            # Autoregressive: feed back the predicted latent
            dec_input = z_hat

        z_future = torch.cat(outputs, dim=1)  # (B, H, D)
        return z_future


class SeqVaeCore(nn.Module):
    """
    Core sequential VAE module (encoders + decoder) with reconstruction losses.
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
        self.decoder = Decoder(latent_dim=latent_dim_z, sequence_length=sequence_length)

        if init_weights:
            initialization(self)

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu_x = self.source_encoder(x_ph)
        mu_y, logvar_y_prior, c_logvar = self.target_encoder(y_st, y_ph)

        mu_post, logvar_post = self.conditional_encoder(mu_x, c_logvar)
        mu_post = mu_post + mu_y
        z = self.reparameterize(mu_post, logvar_post)

        linear_output, mu_pr, logvar_pr = self.decoder(z)

        if self.sequence_length > 1:
            next_idx = self._next_step_indices(mu_pr.device)
            gather_idx = next_idx.unsqueeze(0).expand(mu_pr.size(0), -1)
            mu_next = torch.gather(mu_pr, 1, gather_idx)
            logvar_next = torch.gather(logvar_pr, 1, gather_idx)
        else:
            next_idx = torch.zeros(0, device=mu_pr.device, dtype=torch.long)
            mu_next = mu_pr.new_zeros(mu_pr.size(0), 0)
            logvar_next = logvar_pr.new_zeros(logvar_pr.size(0), 0)

        return {
            "z": z,
            "linear_output": linear_output,
            "mu_pr": mu_pr,
            "logvar_pr": logvar_pr,
            "mu_next": mu_next,
            "logvar_next": logvar_next,
            "next_step_indices": next_idx,
            "mu_prior": mu_y,
            "logvar_prior": logvar_y_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
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

    def compute_reconstruction_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        y_raw: torch.Tensor,
        *,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if y_raw.dim() == 3 and y_raw.size(-1) == 1:
            y_raw = y_raw.squeeze(-1)

        device = y_raw.device

        mse_loss = torch.tensor(0.0, device=device)
        linear_output = forward_outputs.get("linear_output")
        if (
            linear_output is not None
            and linear_output.shape[-1] == (y_st.shape[-1] + y_ph.shape[-1])
        ):
            stacked_target = torch.cat([y_st, y_ph], dim=-1)
            mse_loss = F.mse_loss(linear_output, stacked_target)

        mu_next = forward_outputs.get("mu_next")
        logvar_next = forward_outputs.get("logvar_next")
        indices = forward_outputs.get("next_step_indices")

        if indices is None:
            indices = self._next_step_indices(device)
        if indices.device != device:
            indices = indices.to(device)
        if mu_next is not None and mu_next.numel() > 0:
            gather_idx = indices.unsqueeze(0).expand(y_raw.size(0), -1)
            target_next = torch.gather(y_raw, 1, gather_idx)
            var_next = logvar_next.exp()
            next_step_nll = 0.5 * (logvar_next + (target_next - mu_next) ** 2 / var_next)
            nll_loss = next_step_nll.mean()
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
            "reconstruction_loss": total_decoder,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "kld_loss": kld_loss,
            "total_loss": total_loss,
            "total_decoder_loss": total_decoder,
        }


class SeqVaeTeb(nn.Module):
    """
    Complete SeqVAE model with an optional latent forecaster head.

    The reconstruction path lives in `SeqVaeCore`, which can be saved on its own via
    `vae_state_dict()`, while `state_dict(include_forecaster=False)` omits forecasting
    weights for backwards compatibility with older checkpoints.
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
        # Forecasting params
        context_len: int = 75,
        horizon_len: int = 30,
        forecaster_hidden_dim: int = 128,
        forecaster_layers: int = 1,
        forecaster_dropout: float = 0.0,
        *,
        use_latent_forecaster: bool = True,
        latent_forecaster: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.latent_dim_source = latent_dim_source
        self.latent_dim_target = latent_dim_target
        self.latent_dim_z = latent_dim_z
        self.decimation_factor = decimation_factor
        self.warmup_period = warmup_period
        self.context_len = context_len
        self.horizon_len = horizon_len

        self.core = SeqVaeCore(
            sequence_length=sequence_length,
            latent_dim_source=latent_dim_source,
            latent_dim_target=latent_dim_target,
            latent_dim_z=latent_dim_z,
            decimation_factor=decimation_factor,
            warmup_period=warmup_period,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            init_weights=False,
        )
        initialization(self.core)

        if latent_forecaster is not None:
            self.latent_forecaster = latent_forecaster
        elif use_latent_forecaster:
            self.latent_forecaster = LatentForecaster(
                latent_dim=latent_dim_z,
                hidden_dim=forecaster_hidden_dim,
                num_layers=forecaster_layers,
                dropout=forecaster_dropout,
            )
            initialization(self.latent_forecaster)
        else:
            self.latent_forecaster = None

        self.use_latent_forecaster = self.latent_forecaster is not None

    @property
    def source_encoder(self) -> SourceEncoder:
        return self.core.source_encoder

    @property
    def target_encoder(self) -> TargetEncoder:
        return self.core.target_encoder

    @property
    def conditional_encoder(self) -> ConditionalEncoder:
        return self.core.conditional_encoder

    @property
    def decoder(self) -> Decoder:
        return self.core.decoder

    def has_forecaster(self) -> bool:
        return self.latent_forecaster is not None

    def _require_forecaster(self) -> None:
        if self.latent_forecaster is None:
            raise RuntimeError("Latent forecaster is disabled for this SeqVaeTeb instance.")

    @staticmethod
    def _is_latent_forecaster_key(name: str) -> bool:
        return "latent_forecaster" in name.split(".")

    def state_dict(
        self,
        destination: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        *,
        include_forecaster: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return model weights, optionally skipping the latent forecaster block."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if include_forecaster or self.latent_forecaster is None:
            return state

        keys_to_drop = [k for k in list(state.keys()) if self._is_latent_forecaster_key(k)]
        for key in keys_to_drop:
            state.pop(key)
        return state

    def vae_state_dict(
        self,
        destination: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Convenience helper that exposes only the VAE core parameters."""
        return self.core.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        *,
        load_forecaster: Optional[bool] = None,
    ):
        """
        Load weights with explicit control over whether latent forecaster parameters are expected.
        """
        has_module_forecaster = self.latent_forecaster is not None
        has_forecaster_weights = any(self._is_latent_forecaster_key(k) for k in state_dict.keys())

        if load_forecaster is None:
            load_forecaster = has_module_forecaster and has_forecaster_weights

        if load_forecaster and not has_module_forecaster:
            raise RuntimeError("Cannot load latent forecaster weights: forecaster disabled on this model.")
        if load_forecaster and not has_forecaster_weights:
            raise RuntimeError("Requested latent forecaster weights but none found in the provided state_dict.")

        filtered_state = (
            {k: v for k, v in state_dict.items() if not self._is_latent_forecaster_key(k)}
            if not load_forecaster
            else state_dict
        )

        incompatible = super().load_state_dict(filtered_state, strict=False)
        missing_keys = getattr(incompatible, "missing_keys", []) if hasattr(incompatible, "missing_keys") else incompatible[0]
        unexpected_keys = getattr(incompatible, "unexpected_keys", []) if hasattr(incompatible, "unexpected_keys") else incompatible[1]

        missing_keys = [k for k in missing_keys if not self._is_latent_forecaster_key(k)]
        unexpected_keys = [k for k in unexpected_keys if not self._is_latent_forecaster_key(k)]

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error loading state_dict: missing keys {missing_keys}, unexpected keys {unexpected_keys}"
            )

        incompatible_type = type(incompatible)
        return incompatible_type(missing_keys, unexpected_keys)


class SeqVaeNoForecast(SeqVaeTeb):
    """
    Thin wrapper around SeqVaeTeb that guarantees the latent forecaster is disabled.
    Useful for lightweight training checkpoints while retaining interface compatibility.
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
        context_len: int = 75,
        horizon_len: int = 30,
        forecaster_hidden_dim: int = 128,
        forecaster_layers: int = 1,
        forecaster_dropout: float = 0.0,
    ):
        super().__init__(
            sequence_length=sequence_length,
            latent_dim_source=latent_dim_source,
            latent_dim_target=latent_dim_target,
            latent_dim_z=latent_dim_z,
            decimation_factor=decimation_factor,
            warmup_period=warmup_period,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            context_len=context_len,
            horizon_len=horizon_len,
            forecaster_hidden_dim=forecaster_hidden_dim,
            forecaster_layers=forecaster_layers,
            forecaster_dropout=forecaster_dropout,
            use_latent_forecaster=False,
            latent_forecaster=None,
        )

    @classmethod
    def from_legacy_checkpoint(
        cls,
        ckpt_path: str,
        *,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        compile_model: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SeqVaeTeb":
        """Instantiate SeqVaeTeb and load weights from a pre-forecaster checkpoint.

        Older checkpoints only contain the VAE components (source/target/conditional
        encoders plus decoder). This helper loads those weights while leaving the
        newly added latent forecaster randomly initialized so fine-tuning can
        continue seamlessly.
        """
        init_kwargs = init_kwargs or {}
        model = cls(**init_kwargs)

        ckpt = torch.load(ckpt_path, map_location=map_location)
        sd = ckpt.get("state_dict", ckpt)

        def _normalize_key(k: str) -> str:
            prefixes = (
                "model.",
                "module.",
                "_orig_mod.",
                "seqvae_model.",
                "vae_model.",
            )
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            return k

        legacy_prefixes = (
            "source_encoder.",
            "target_encoder.",
            "conditional_encoder.",
            "decoder.",
        )
        core_prefixes = tuple(f"core.{p}" for p in legacy_prefixes)

        current_sd = model.state_dict()
        filtered_sd: Dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            norm_key = _normalize_key(key)
            candidate_keys = []
            if norm_key.startswith(core_prefixes):
                candidate_keys.append(norm_key)
            if norm_key.startswith(legacy_prefixes):
                candidate_keys.append(f"core.{norm_key}")
            for target_key in candidate_keys:
                if target_key in current_sd:
                    filtered_sd[target_key] = value
                    break

        if not filtered_sd:
            raise ValueError(
                f"No legacy SeqVaeTeb parameters found in checkpoint {ckpt_path}"
            )

        shape_mismatches = []
        for key, value in filtered_sd.items():
            current_shape = current_sd[key].shape
            if current_shape != value.shape:
                shape_mismatches.append(
                    f"{key}: current {tuple(current_shape)} vs checkpoint {tuple(value.shape)}"
                )
        if shape_mismatches:
            raise ValueError(
                "SeqVaeTeb architecture mismatch when loading legacy checkpoint:\n"
                + "\n".join(shape_mismatches)
            )

        incompatible = model.load_state_dict(filtered_sd, strict=False)
        try:
            missing_keys = getattr(incompatible, "missing_keys", [])
            unexpected_keys = getattr(incompatible, "unexpected_keys", [])
        except Exception:
            try:
                missing_keys, unexpected_keys = incompatible
            except Exception:
                missing_keys, unexpected_keys = [], []

        legacy_key_prefixes = legacy_prefixes + core_prefixes
        legacy_missing = [k for k in missing_keys if k.startswith(legacy_key_prefixes)]
        new_module_missing = [k for k in missing_keys if not k.startswith(legacy_key_prefixes)]

        if legacy_missing:
            log.warning(
                f"[SeqVaeTeb] Missing legacy keys when loading checkpoint: {legacy_missing}"
            )
        if unexpected_keys:
            log.warning(
                f"[SeqVaeTeb] Unexpected keys ignored from checkpoint: {unexpected_keys}"
            )
        if new_module_missing:
            preview = new_module_missing[:5]
            suffix = " ..." if len(new_module_missing) > 5 else ""
            log.info(
                f"[SeqVaeTeb] Leaving newly introduced parameters uninitialized: {preview}{suffix}"
            )

        if strict and (legacy_missing or unexpected_keys):
            raise RuntimeError(
                "Strict legacy load failed due to missing or unexpected parameters"
            )

        log.info(
            f"Loaded {len(filtered_sd)} legacy parameters into SeqVaeTeb from {ckpt_path}"
        )

        if compile_model:
            primary_attempt = {
                "mode": compile_mode,
                "fullgraph": False,
                "dynamic": True,
            }
            attempt_chain = [primary_attempt]
            for default_opts in DEFAULT_COMPILE_ATTEMPTS:
                if default_opts != primary_attempt:
                    attempt_chain.append(default_opts)
            model, _ = ensure_compiled_module(
                model,
                module_name="SeqVaeTeb legacy load",
                attempts=attempt_chain,
            )

        return model

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full VAE forward pass via the core module."""
        return self.core.forward(y_st=y_st, y_ph=y_ph, x_ph=x_ph)

    # ------------------------
    # Encoding utilities
    # ------------------------
    def encode_only(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        sample_z: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encoder-only pass delegated to the core module."""
        return self.core.encode_only(y_st=y_st, y_ph=y_ph, x_ph=x_ph, sample_z=sample_z)

    # ------------------------
    # Forecasting helpers
    # ------------------------
    @staticmethod
    def anchor_range(T: int, Lc: int, H: int) -> torch.Tensor:
        """Return valid anchor indices t in [Lc-1, T-1-H] as (N,)."""
        start = max(0, Lc - 1)
        end = max(start - 1, T - 1 - H)  # inclusive
        if end < start:
            return torch.zeros(0, dtype=torch.long)
        return torch.arange(start, end + 1, dtype=torch.long)

    @staticmethod
    def raw_window_from_anchor(t: int, stride: int = 16, H: int = 30) -> Tuple[int, int]:
        """Given decimated anchor t, return raw [start, end) indices for the H-step forecast (length H*stride)."""
        start = stride * (t + 1)
        end = start + stride * H
        return start, end

    def _gather_context(self, z_seq: torch.Tensor, anchors: torch.Tensor, Lc: int) -> torch.Tensor:
        """
        Gather contexts for anchors from z sequence.

        Args:
            z_seq: (B, T, D)
            anchors: (N,)
        Returns:
            contexts: (B, N, Lc, D)
        """
        B, T, D = z_seq.shape
        N = anchors.numel()
        contexts = []
        for t in anchors.tolist():
            ctx = z_seq[:, t - Lc + 1 : t + 1, :]
            contexts.append(ctx.unsqueeze(1))  # (B,1,Lc,D)
        return torch.cat(contexts, dim=1) if contexts else z_seq.new_zeros(B, 0, Lc, D)

    def forecast(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        use_posterior_mean: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forecast future raw FHR windows at the given anchors.

        Shapes
            y_*: (B, T, ...)
            anchors: (N,) — if None, computed from (T, Lc, H)
            returns:
            anchors: (N,)
            z_future: (B, N, H, D)
            mu_future/logvar_future: (B, N, W) with W = 16*H
            enc: dict with full-sequence priors/posteriors
        """
        self._require_forecaster()

        B, T, _ = y_st.shape
        H = self.horizon_len
        Lc = self.context_len

        enc = self.encode_only(y_st, y_ph, x_ph, sample_z=not use_posterior_mean)
        z_seq = enc["z"] if not use_posterior_mean else enc["mu_post"]  # (B,T,D)

        if anchors is None:
            anchors = self.anchor_range(T, Lc, H).to(z_seq.device)
        if anchors.numel() == 0:
            return {
                "anchors": anchors,
                "z_future": z_seq.new_zeros(B, 0, H, self.latent_dim_z),
                "mu_future": z_seq.new_zeros(B, 0, H * self.decimation_factor),
                "logvar_future": z_seq.new_zeros(B, 0, H * self.decimation_factor),
                "enc": enc,
            }

        # Gather contexts: (B,N,Lc,D) -> flatten anchors for forecaster
        contexts = self._gather_context(z_seq, anchors, Lc)  # (B,N,Lc,D)
        B, N, Lc, D = contexts.shape
        contexts_flat = contexts.reshape(B * N, Lc, D)

        # Predict future latents per anchor
        z_future_flat = self.latent_forecaster(contexts_flat, horizon=H)  # (B*N, H, D)
        z_future = z_future_flat.reshape(B, N, H, D)

        # Decode per-anchor future latents into raw windows
        z_future_flat_for_dec = z_future_flat  # (B*N, H, D)
        _, mu_flat, logvar_flat = self.decoder(z_future_flat_for_dec)  # mu/logvar: (B*N, 16*H)
        mu_future = mu_flat.reshape(B, N, -1)
        logvar_future = torch.clamp(logvar_flat.reshape(B, N, -1), min=-10, max=10)

        return {
            "anchors": anchors,
            "z_future": z_future,
            "mu_future": mu_future,
            "logvar_future": logvar_future,
            "enc": enc,
        }

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick delegated to the core module."""
        return self.core.reparameterize(mu, logvar)

    def _kld_loss(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        reduce_mean: bool = True,
    ) -> torch.Tensor:
        """Wrapper over the core KL helper."""
        return self.core._kld_loss(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
            reduce_mean=reduce_mean,
        )

    @staticmethod
    def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Static redirect to the core Gaussian NLL helper."""
        return SeqVaeCore.gaussian_nll(mu, logvar, target, mask)

    def compute_forecast_loss(
        self,
        forecast_outputs: Dict[str, torch.Tensor],
        y_raw: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        include_kld: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forecasting loss.

        - Gaussian NLL averaged over predicted windows at N anchors
        - Optional KL(q||p) over full encoded sequence

        Shapes
            mu_future/logvar_future in forecast_outputs: (B, N, W) with W = 16*H
            y_raw: (B, R=4800)
            anchors: (N,)
        """
        self._require_forecaster()

        mu_future = forecast_outputs["mu_future"]  # (B,N,480)
        logvar_future = forecast_outputs["logvar_future"]  # (B,N,480)
        enc = forecast_outputs["enc"]
        if anchors is None:
            anchors = forecast_outputs["anchors"]

        B, N, W = mu_future.shape
        H = self.horizon_len
        stride = self.decimation_factor

        if N == 0:
            forecast_nll = torch.tensor(0.0, device=y_raw.device)
        else:
            window_len = mu_future.size(-1)
            anchor_positions = stride * (anchors.to(mu_future.device).long() + 1)
            offsets = torch.arange(window_len, device=mu_future.device, dtype=torch.long)
            gather_idx = anchor_positions[:, None] + offsets[None, :]  # (N, W)
            gather_idx = gather_idx.unsqueeze(0).expand(B, -1, -1)
            target_windows = torch.gather(
                y_raw.unsqueeze(1).expand(-1, N, -1),
                2,
                gather_idx
            )

            forecast_nll = self.gaussian_nll(
                mu_future.reshape(B * N, window_len),
                logvar_future.reshape(B * N, window_len),
                target_windows.reshape(B * N, window_len),
            )

        kld_loss = torch.tensor(0.0, device=y_raw.device)
        if include_kld:
            kld_loss = self._kld_loss(
                mu_prior=enc["mu_prior"],
                logvar_prior=enc["logvar_prior"],
                mu_post=enc["mu_post"],
                logvar_post=enc["logvar_post"],
                reduce_mean=True,
            )

        total = forecast_nll + beta * kld_loss
        return {
            "forecast_nll": forecast_nll,
            "kld_loss": kld_loss,
            "total_loss": total,
        }

    def aggregate_forecasts_to_canvas(
        self,
        mu_future: torch.Tensor,
        anchors: torch.Tensor,
        total_len: int = 4800,
        stride: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Place per-anchor windows on a raw-length canvas with NaNs in gaps.

        Args
            mu_future: (B, N, W) where W = 16*H
            anchors: (N,)
            total_len: raw timeline length (default 4800)
            stride: samples per decimated step (default 16)
        Returns
            stacked: (B, N, total_len) with NaNs where not covered
            mean:    (B, total_len) nanmean over anchors
        """
        B, N, W = mu_future.shape
        if N == 0:
            canvas = mu_future.new_full((B, 0, total_len), float('nan'))
            return canvas, canvas.new_full((B, total_len), float('nan'))

        device = mu_future.device
        anchor_positions = stride * (anchors.to(device).long() + 1)
        offsets = torch.arange(W, device=device, dtype=torch.long)
        index = anchor_positions[:, None] + offsets[None, :]  # (N, W)

        valid_mask = (index >= 0) & (index < total_len)
        index = index.clamp_(min=0, max=total_len - 1)

        expanded_index = index.unsqueeze(0).expand(B, -1, -1)  # (B, N, W)
        expanded_mask = valid_mask.unsqueeze(0).expand(B, -1, -1)

        canvas = mu_future.new_full((B, N, total_len), float('nan'))
        row_indices = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2).expand(B, N, W)
        anchor_indices = torch.arange(N, device=device).unsqueeze(0).unsqueeze(2).expand(B, N, W)

        flat_canvas = canvas.reshape(B * N, total_len)
        flat_rows = (row_indices * N + anchor_indices).reshape(B * N, W)
        flat_cols = expanded_index.reshape(B * N, W)
        flat_mask = expanded_mask.reshape(B * N, W)
        flat_values = mu_future.reshape(B * N, W)

        flat_rows = flat_rows.reshape(-1).long()
        flat_cols = flat_cols.reshape(-1).long()
        flat_values = flat_values.reshape(-1)
        flat_mask = flat_mask.reshape(-1)

        if flat_mask.any():
            flat_canvas[flat_rows[flat_mask], flat_cols[flat_mask]] = flat_values[flat_mask]

        canvas = flat_canvas.reshape(B, N, total_len)
        mean = torch.nanmean(canvas, dim=1)
        return canvas, mean

    # ------------------------
    # Convenience: full forecast + aggregation + metrics
    # ------------------------
    def forecast_full(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        use_posterior_mean: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs forecast and returns aggregated mean/std on the raw timeline.

        Returns keys
            anchors:      (N,)
            mu_future:    (B, N, 16*H)
            logvar_future:(B, N, 16*H)
            canvas_mu:    (B, N, 4800)
            mean_mu:      (B, 4800)
            canvas_var:   (B, N, 4800)
            std_mu:       (B, 4800)
            enc:          dict with priors/posteriors
        """
        out = self.forecast(y_st, y_ph, x_ph, anchors=anchors, use_posterior_mean=use_posterior_mean)
        anchors = out["anchors"]
        mu_future = out["mu_future"]
        logvar_future = out["logvar_future"]
        enc = out["enc"]

        canvas_mu, mean_mu = self.aggregate_forecasts_to_canvas(
            mu_future, anchors, total_len=4800, stride=self.decimation_factor
        )
        var_future = logvar_future.exp()
        canvas_var, mean_var = self.aggregate_forecasts_to_canvas(
            var_future, anchors, total_len=4800, stride=self.decimation_factor
        )
        std_mu = mean_var.clamp_min(1e-8).sqrt()
        return {
            "anchors": anchors,
            "mu_future": mu_future,
            "logvar_future": logvar_future,
            "canvas_mu": canvas_mu,
            "mean_mu": mean_mu,
            "canvas_var": canvas_var,
            "std_mu": std_mu,
            "enc": enc,
        }

    @staticmethod
    def _masked_corrcoef(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pearson correlation over masked elements per batch.
        a, b: (B, L), mask: (B, L) boolean
        Returns (B,) correlations; NaN if fewer than 2 valid points.
        """
        a = a.clone()
        b = b.clone()
        mask = mask.bool()
        # Replace invalid with 0 for sums, track counts
        a_masked = a.masked_fill(~mask, 0.0)
        b_masked = b.masked_fill(~mask, 0.0)
        n = mask.sum(dim=1).clamp_min(1)
        # Means
        mean_a = a_masked.sum(dim=1) / n
        mean_b = b_masked.sum(dim=1) / n
        # Centered
        a_c = a_masked - mean_a.unsqueeze(1)
        b_c = b_masked - mean_b.unsqueeze(1)
        a_c = a_c.masked_fill(~mask, 0.0)
        b_c = b_c.masked_fill(~mask, 0.0)
        # Covariance and variances
        cov = (a_c * b_c).sum(dim=1) / n.clamp_min(2)
        var_a = (a_c.pow(2)).sum(dim=1) / n.clamp_min(2)
        var_b = (b_c.pow(2)).sum(dim=1) / n.clamp_min(2)
        denom = (var_a.clamp_min(1e-12) * var_b.clamp_min(1e-12)).sqrt()
        corr = cov / denom
        # If fewer than 2 points, set NaN
        corr = torch.where(n >= 2, corr, torch.full_like(corr, float('nan')))
        return corr

    def evaluate_forecast_batch(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        y_raw: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        use_posterior_mean: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forecasts and computes per-sample metrics on the aggregated prediction over the raw timeline.

        Returns:
            - mean_mu: (B,4800), std_mu: (B,4800), mask: (B,4800) valid-coverage mask
            - mse: (B,), mae: (B,), corr: (B,) over covered region
            - anchors, enc (for further analysis)
        """
        out = self.forecast_full(y_st, y_ph, x_ph, anchors=anchors, use_posterior_mean=use_posterior_mean)
        mean_mu = out["mean_mu"]  # (B,4800)
        std_mu = out["std_mu"]    # (B,4800)
        anchors = out["anchors"]
        enc = out["enc"]

        # Coverage mask
        mask = ~torch.isnan(mean_mu)
        # Replace NaNs for arithmetic
        pred = mean_mu.masked_fill(~mask, 0.0)
        gt = y_raw.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp_min(1)
        # MSE and MAE over covered points
        mse = ((pred - gt).pow(2).sum(dim=1) / denom)
        mae = ((pred - gt).abs().sum(dim=1) / denom)
        # Correlation over covered points
        corr = self._masked_corrcoef(pred, gt, mask)

        return {
            "mean_mu": mean_mu,
            "std_mu": std_mu,
            "mask": mask,
            "mse": mse,
            "mae": mae,
            "corr": corr,
            "anchors": anchors,
            "enc": enc,
        }

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor, 
        y_raw: torch.Tensor,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
        predictive_weight: float = 0.0,
        predictive_horizon: int = 1,
        latent_consistency_weight: float = 0.0,
        predictive_context_len: Optional[int] = None,
        predictive_max_anchors: Optional[int] = None,
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
            predictive_weight (float): Weight for auxiliary raw-forecast NLL (0 disables).
            predictive_horizon (int): Number of decimated steps to forecast (H >= 1).
            latent_consistency_weight (float): Weight for latent forecast MSE (0 disables).
            predictive_context_len (Optional[int]): Context length for latent forecaster; defaults to
                ``self.context_len`` when ``None``.

        Returns:
            A dictionary of computed losses.
        """
        needs_predictive = (predictive_weight > 0.0 or latent_consistency_weight > 0.0)
        if needs_predictive:
            if predictive_horizon < 1:
                raise ValueError("predictive_horizon must be >= 1 when predictive losses are enabled")
            self._require_forecaster()

        base_losses = self.core.compute_reconstruction_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            y_raw=y_raw,
            compute_kld_loss=compute_kld_loss,
            beta=beta,
        )
        base_total = base_losses["total_loss"]

        device = forward_outputs["mu_pr"].device
        predictive_loss = torch.tensor(0.0, device=device)
        latent_consistency_loss = torch.tensor(0.0, device=device)

        y_raw_eval = y_raw
        if y_raw_eval.dim() == 3 and y_raw_eval.size(-1) == 1:
            y_raw_eval = y_raw_eval.squeeze(-1)

        if needs_predictive:
            mu_post = forward_outputs["mu_post"]
            _, T, D = mu_post.shape
            stride = self.decimation_factor
            context_len = predictive_context_len or self.context_len

            anchors = self.anchor_range(T, context_len, predictive_horizon)
            if anchors.numel() > 0:
                anchors = anchors.to(mu_post.device)
                if predictive_max_anchors is not None and predictive_max_anchors > 0:
                    max_anchors = min(int(predictive_max_anchors), anchors.numel())
                    if max_anchors < anchors.numel():
                        perm = torch.randperm(anchors.numel(), device=anchors.device)
                        anchors = anchors[perm[:max_anchors]]

                contexts = self._gather_context(mu_post, anchors, context_len)
                B_ctx, N, Lc, _ = contexts.shape
                contexts_flat = contexts.reshape(B_ctx * N, Lc, D)

                z_future_pred = self.latent_forecaster(contexts_flat, horizon=predictive_horizon)
                z_future_pred = z_future_pred.reshape(B_ctx, N, predictive_horizon, D)

                future_offsets = torch.arange(1, predictive_horizon + 1, device=mu_post.device, dtype=torch.long)
                step_indices = anchors.long().unsqueeze(1) + future_offsets.unsqueeze(0)
                gather_idx = step_indices.unsqueeze(0).unsqueeze(-1).expand(B_ctx, -1, -1, D)
                expanded_mu_post = mu_post.unsqueeze(1).expand(-1, N, -1, -1)
                future_targets = torch.gather(expanded_mu_post, 2, gather_idx).detach()

                if latent_consistency_weight > 0.0:
                    latent_consistency_loss = F.mse_loss(z_future_pred, future_targets)

                if predictive_weight > 0.0:
                    z_future_flat = z_future_pred.reshape(B_ctx * N, predictive_horizon, D)
                    _, mu_flat, logvar_flat = self.decoder(z_future_flat)
                    mu_future = mu_flat.reshape(B_ctx, N, -1)
                    logvar_future = torch.clamp(logvar_flat.reshape(B_ctx, N, -1), min=-10, max=10)

                    window_len = mu_future.size(-1)
                    start_positions = stride * (anchors.long() + 1)
                    raw_offsets = torch.arange(window_len, device=device, dtype=torch.long)
                    raw_indices = start_positions[:, None] + raw_offsets[None, :]
                    raw_indices = raw_indices.unsqueeze(0).expand(B_ctx, -1, -1)

                    target_windows = torch.gather(
                        y_raw_eval.unsqueeze(1).expand(-1, N, -1),
                        2,
                        raw_indices.to(y_raw_eval.device),
                    )

                    predictive_loss = self.gaussian_nll(
                        mu_future.reshape(B_ctx * N, window_len),
                        logvar_future.reshape(B_ctx * N, window_len),
                        target_windows.reshape(B_ctx * N, window_len),
                    )

        total_loss = (
            base_total
            + predictive_weight * predictive_loss
            + latent_consistency_weight * latent_consistency_loss
        )

        base_losses["base_total_loss"] = base_total
        base_losses["predictive_loss"] = predictive_loss
        base_losses["latent_consistency_loss"] = latent_consistency_loss
        base_losses["total_loss"] = total_loss
        base_losses["classification_loss"] = None  # Required by interface
        return base_losses

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
        """Load pretrained SeqVaeTeb weights (Lightning/DataParallel/torch.compile safe).

        Notes:
        - Lightning checkpoints typically store weights under 'state_dict' and prefix with
            'model.'; nested modules may appear as 'model.vae_model.' etc.
        - DataParallel adds 'module.' and torch.compile can add an '_orig_mod.' wrapper.
        - This method strips these known prefixes and filters keys to the bare `SeqVaeTeb`.
        - Shape mismatches raise with details; other discrepancies are logged as warnings.
        """
        try:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            # Prefer Lightning's 'state_dict' if present; otherwise treat the whole object as a state dict
            sd = ckpt.get("state_dict", ckpt)

            # Normalize key prefixes from common wrappers
            def _normalize_key(k: str) -> str:
                prefixes = (
                    "model.",
                    "module.",
                    "_orig_mod.",
                    "seqvae_model.",
                    "vae_model.",
                )
                changed = True
                while changed:
                    changed = False
                    for p in prefixes:
                        if k.startswith(p):
                            k = k[len(p):]
                            changed = True
                return k

            # Get current model state_dict to check compatibility
            current_sd = self.vae_model.state_dict()
            expected_keys = set(current_sd.keys())
            
            # Strip prefixes and filter only keys that belong to the VAE module
            new_sd = {}
            for k, v in sd.items():
                nk = _normalize_key(k)
                if nk in expected_keys:
                    new_sd[nk] = v

            # Check for shape mismatches in existing keys
            shape_mismatches = []
            for k, v in new_sd.items():
                if k in current_sd and current_sd[k].shape != v.shape:
                    shape_mismatches.append(f"{k}: current {current_sd[k].shape} vs checkpoint {v.shape}")
            
            if shape_mismatches:
                raise ValueError(f"VAE architecture mismatch - parameter shape conflicts:\n" + 
                                "\n".join(shape_mismatches[:5]) + 
                                (f"\n... and {len(shape_mismatches)-5} more" if len(shape_mismatches) > 5 else ""))

            # Load with strict=False to get missing/unexpected keys info
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

            # Log any remaining discrepancies (should be minimal after normalization)
            if missing_keys:
                log.warning(f"[SeqVaeTebClassifier] Missing keys when loading VAE: {missing_keys}")
            if unexpected_keys:
                log.warning(f"[SeqVaeTebClassifier] Unexpected keys when loading VAE: {unexpected_keys}")

            log.info(f"Successfully loaded all VAE parameters from {pretrained_path}")

        except Exception as e:
            log.error(f"Failed to load pretrained VAE from '{pretrained_path}': {e}")
            raise RuntimeError(f"Cannot load incompatible VAE checkpoint. Please use a checkpoint that matches "
                                f"the current VAE architecture, or train from scratch.") from e

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
            
        Returns:wZA
            vae_outputs: Full VAE outputs (if return_all_outputs=True)
        """
        # Set VAE to eval mode if frozen
        if self.freeze_vae:
            self.vae_model.eval()

        # Forward pass through VAE
        with torch.set_grad_enabled(not self.freeze_vae):
            vae_outputs = self.vae_model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)

        latent_z = vae_outputs['z']  # (batch, 300, latent_dim_z)

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

    # ---------------------- Loading utilities ----------------------
    @classmethod
    def from_lightning_checkpoint(
        cls,
        ckpt_path: str,
        *,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        compile_model: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SeqVaeTebClassifier":
        """
        Construct a SeqVaeTebClassifier and load weights from a Lightning checkpoint
        produced by LightSeqVaeTebClassifier.

        The Lightning checkpoint's state_dict is under the 'model.' prefix since the
        LightningModule wraps the classifier as self.model. This method strips that
        prefix and loads into a bare SeqVaeTebClassifier.

        Args:
            ckpt_path: Path to the .ckpt file saved by Lightning
            map_location: torch.load map_location
            strict: Whether to enforce exact key matching
            compile_model: If True, wraps the returned model with torch.compile
            compile_mode: torch.compile mode (default: 'max-autotune-no-cudagraphs')
            init_kwargs: Keyword args to instantiate the classifier (e.g., num_classes,...)

        Returns:
            Loaded SeqVaeTebClassifier instance (optionally compiled)
        """
        init_kwargs = init_kwargs or {}
        model = cls(**init_kwargs)

        ckpt = torch.load(ckpt_path, map_location=map_location)
        sd = ckpt.get("state_dict", ckpt)

        # Normalize common wrapper prefixes and filter to classifier keys
        def _normalize_key(k: str) -> str:
            prefixes = (
                "model.",
                "module.",
                "_orig_mod.",
            )
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            return k

        expected_keys = set(model.state_dict().keys())
        new_sd = {(_normalize_key(k)): v for k, v in sd.items() if _normalize_key(k) in expected_keys}

        incompatible = model.load_state_dict(new_sd, strict=strict)
        try:
            missing_keys = getattr(incompatible, "missing_keys", [])
            unexpected_keys = getattr(incompatible, "unexpected_keys", [])
        except Exception:
            try:
                missing_keys, unexpected_keys = incompatible
            except Exception:
                missing_keys, unexpected_keys = [], []

        if missing_keys:
            log.warning(f"[SeqVaeTebClassifier] Missing keys when loading classifier: {missing_keys}")
        if unexpected_keys:
            log.warning(f"[SeqVaeTebClassifier] Unexpected keys when loading classifier: {unexpected_keys}")

        if compile_model:
            primary_attempt = {
                "mode": compile_mode,
                "fullgraph": False,
                "dynamic": True,
            }
            attempt_chain = [primary_attempt]
            for default_opts in DEFAULT_COMPILE_ATTEMPTS:
                if default_opts != primary_attempt:
                    attempt_chain.append(default_opts)
            model, _ = ensure_compiled_module(
                model,
                module_name="SeqVaeTebClassifier load",
                attempts=attempt_chain,
            )

        return model


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
