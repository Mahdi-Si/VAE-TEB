# VAE TEB Model Architecture Documentation

## Overview

The VAE TEB (Variational Autoencoder with Target-Encoder-Bank) model is a sophisticated sequence-to-sequence prediction framework designed for time series forecasting with uncertainty quantification. The model combines multiple modalities (scattering transform and phase harmonic features) using a dual-encoder architecture with conditional latent variable modeling.

## Core Architecture Components

### 1. Initialization Function
**Function**: `initialization(model)`
- **Purpose**: Applies state-of-the-art initialization schemes to all model components
- **Usage**: Called automatically during model instantiation to ensure proper gradient flow

### 2. Causal Convolution Components

#### CausalConv1d
- **Purpose**: Implements causal 1D convolution ensuring no future information leakage
- **Key Features**: Efficient left padding, mixed precision support, grouped convolutions
- **Input Shape**: (batch_size, channels, seq_len)
- **Output Shape**: (batch_size, out_channels, seq_len)

#### CausalResidualBlock
- **Purpose**: Building block for temporal convolutional networks with residual connections
- **Features**: Pre-norm design, GLU gating, dilated convolutions for TCN
- **Architecture**: Pre-LayerNorm → Causal Conv → Activation → Residual Connection
- **Input/Output Shape**: (batch_size, seq_len, channels)

### 3. Feed-Forward Components

#### FeedForward
- **Purpose**: Simple MLP with pre-LayerNorm and residual connections
- **Architecture**: LayerNorm → Linear → Activation → Dropout
- **Activation**: Configurable (default: GELU)

#### ResidualMLP
- **Purpose**: Multi-layer MLP with skip connections and normalization
- **Architecture**: Input LayerNorm → (Linear → LayerNorm → GELU → Dropout)* → Skip Connection → Post-norm
- **Features**: Configurable hidden dimensions, optional final activation

### 4. Encoder Modules

#### SourceEncoder
- **Purpose**: Processes source phase harmonic input x_ph to generate latent representation μ_x
- **Architecture**:
  1. Dual-path processing: Linear projection + Causal convolution paths
  2. Path fusion with ResidualMLP
  3. Unidirectional LSTM for causal temporal encoding
  4. Multi-layer processing to output μ_x
- **Input**: x_ph (batch_size, seq_len, 76)
- **Output**: μ_x (batch_size, seq_len, latent_dim_source)
- **Key Features**: Causal design, no future information leakage

#### TargetEncoder
- **Purpose**: Processes target inputs (y_st, y_ph) to model prior p(z|y)
- **Architecture**:
  1. Dual-modal processing: Separate paths for scattering and phase harmonic
  2. Each modality: Linear + Causal conv paths → Intra-modal fusion
  3. Cross-modal fusion combining both modalities
  4. Bidirectional LSTM for full sequence context
  5. Variational outputs: μ_y and logvar_y
- **Inputs**: 
  - y_st: scattering transform (batch_size, seq_len, 76)
  - y_ph: phase harmonic (batch_size, seq_len, 76)
- **Outputs**:
  - μ_y: (batch_size, seq_len, latent_dim_target)
  - logvar_y: (batch_size, seq_len, 2*latent_dim_target)
- **Key Features**: Bidirectional processing, dual-modal fusion

#### ConditionalEncoder
- **Purpose**: Implements conditional encoder q(z|x,y) for TEB framework
- **Architecture**: MLP fusion of source and target representations
- **Inputs**:
  - h_x: source representation (batch_size, seq_len, dim_hx)
  - h_y: target representation (batch_size, seq_len, dim_hy)
- **Outputs**:
  - μ_post: posterior mean (batch_size, seq_len, dim_z)
  - logvar_post: posterior log-variance (batch_size, seq_len, dim_z)

### 5. Decoder Module

#### Decoder
- **Purpose**: Generates future predictions from latent variables
- **Architecture**:
  1. Dual-path processing: Linear + Multiple causal conv paths
  2. Path fusion with ResidualMLP
  3. Bidirectional LSTM for temporal modeling
  4. Multi-layer prediction head
  5. Separate output heads for each modality
- **Input**: z (batch_size, seq_len, latent_dim_z)
- **Outputs**:
  - scattering_pred: (batch_size, seq_len, prediction_horizon, 76)
  - phase_harmonic_pred: (batch_size, seq_len, prediction_horizon, 76)
- **Key Features**: Multi-step prediction, modality-specific heads

### 6. Main Model

#### SeqVaeTeb
- **Purpose**: Integrates all components into complete TEB framework
- **Architecture Flow**:
  1. Source encoder: x_ph → μ_x
  2. Target encoder: (y_st, y_ph) → (μ_y, logvar_y)
  3. Split logvar_y into prior and conditional features
  4. Conditional encoder: (μ_x, conditional_features) → (μ_post, logvar_post)
  5. Sample z from posterior using reparameterization trick
  6. Decoder: z → future predictions
- **Loss Components**:
  - Reconstruction loss: MSE between predictions and targets
  - KL divergence: Between prior p(z|y) and posterior q(z|x,y)

## Design Rules and Conventions

### Input/Output Requirements
- **Input Format**: All encoder inputs must be (Batch_size, Channel, Sequence_length)
- **Internal Processing**: Models convert to (Batch_size, Sequence_length, Channel) internally
- **Output Format**: Decoder outputs must be (Batch_size, Channel, Sequence_length) or compatible
- **Loss Dictionary**: Must contain exactly these keys:
  - `reconstruction_error` (can be MSE, NLL, or other reconstruction loss)
  - `kld_loss` (KL divergence between prior and posterior)
  - `classification_loss` (set to None if not implemented)

### Architectural Conventions

#### 1. Normalization Strategy
- **Pre-norm design**: Apply LayerNorm before transformations
- **Consistent normalization**: Use LayerNorm throughout (not BatchNorm)
- **Gradient-friendly**: Prevents vanishing gradients in deep networks

#### 2. Activation Functions
- **Primary activation**: GELU for better performance than ReLU
- **Gating mechanisms**: Sigmoid for gates, GLU-style activations
- **Output layers**: No activation for final prediction heads

#### 3. Residual Connections
- **Skip connections**: Mandatory for deep networks (>2 layers)
- **Dimension matching**: Use projection layers when input/output dims differ
- **ResidualMLP pattern**: Standardized residual MLP implementation

#### 4. Regularization
- **Dropout**: Apply consistently (default 0.1)
- **Gradient clipping**: Max norm 1.0 to prevent exploding gradients
- **Weight initialization**: Use improved_initialization() function

#### 5. Temporal Processing
- **Causal convolutions**: For source encoder (no future leakage)
- **Bidirectional LSTM**: For target encoder and decoder (full context)
- **Sequence-first**: Internal processing in (B, S, C) format

### Implementation Guidelines

#### 1. Module Structure
```python
class NewModule(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define components using ResidualMLP when possible
        
    def forward(self, x):
        # Convert input format if needed: (B, C, L) → (B, L, C)
        # Process through components
        # Return appropriate format
```

#### 2. Loss Computation
- **Selective computation**: Allow enabling/disabling specific loss components
- **Numerical stability**: Clamp log-variance values (-10, 10)
- **Warmup handling**: Ignore initial timesteps in sequence loss

#### 3. Prediction Interface
- **Multi-step output**: Decoder produces (B, S, H, C) predictions
- **Average predictions**: Provide method for single-step averaged predictions
- **Timestep indexing**: Support prediction at specific time indices

#### 4. Memory Efficiency
- **Gradient checkpointing**: For long sequences
- **Mixed precision**: Support for automatic mixed precision training
- **Efficient convolutions**: Use depthwise separable convolutions when beneficial

### Testing Requirements

#### 1. Shape Compatibility
- Test with various sequence lengths and batch sizes
- Verify input/output shape transformations
- Check gradient flow through all components

#### 2. Numerical Stability
- Test for NaN/Inf values in loss computation
- Verify gradient clipping effectiveness
- Check initialization scheme impact

#### 3. Training Stability
- Monitor loss convergence
- Test with different learning rates
- Verify KL annealing if implemented

### Extension Guidelines

#### 1. Adding New Encoders
- Maintain causal constraints where appropriate
- Apply initialization()

#### 2. Modifying Decoder
- Preserve multi-step prediction capability
- Maintain separate heads for different modalities
- Support selective loss computation

#### 3. Custom Loss Functions
- Integrate with existing loss dictionary structure
- Support selective computation flags
- Maintain numerical stability practices
- Document warmup period handling

### Current Implementation Details

#### Loss Dictionary Mapping
The current VAE TEB model outputs the following loss dictionary:
```python
{
    "total_loss": total_loss,
    "reconstruction_loss": recon_loss,  # Maps to reconstruction_error
    "kld_loss": kld_loss,              # Direct mapping
    "scattering_loss": scattering_loss,
    "phase_loss": phase_loss
}
```

**Note**: To maintain compatibility with the required interface, the model should map:
- `reconstruction_loss` → `reconstruction_error`
- `kld_loss` → `kld_loss` (direct)
- `classification_loss` → `None` (not implemented in base VAE)

This architecture provides a robust foundation for time series prediction with uncertainty quantification, suitable for various sequence modeling tasks while maintaining compatibility with existing training pipelines and evaluation metrics.