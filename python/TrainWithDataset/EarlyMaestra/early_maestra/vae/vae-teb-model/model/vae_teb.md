# TEB-VAE Model with Raw Signal Decoder

## Overview
The TEB-VAE (Target-Encoder-Bank Variational Autoencoder) is a sophisticated deep learning architecture designed for raw temporal signal reconstruction from latent representations. The model has been redesigned to focus exclusively on raw signal prediction with 16x temporal upsampling, implementing advanced uncertainty quantification through negative log-likelihood loss.

## Architecture Components

### 1. SeqVaeTeb (Main Model)
**Purpose**: Orchestrates the entire TEB framework for raw signal reconstruction with uncertainty quantification.

**Key Features**:
- Processes 76-channel input signals (scattering transform and phase harmonic features)
- Reconstructs raw temporal signals at 16x resolution using progressive upsampling
- Implements variational inference with KL divergence regularization
- Provides uncertainty quantification through predicted log-variance

**Flow**:
1. Source encoder processes phase harmonic input (x_ph) → mu_x
2. Target encoder processes scattering transform (y_st) and phase harmonic (y_ph) → mu_y, logvar_y
3. Conditional encoder combines mu_x and conditional features → posterior distribution
4. Raw signal decoder reconstructs temporal signal from sampled latent z with 16x upsampling

### 2. SourceEncoder
**Purpose**: Encodes source phase harmonic signals into latent representations using causal processing with channel reduction.

**Architecture**:
- **Channel reduction**: 76 → 48 channels using attention-based ChannelReductionBlock
- Dual-path processing: Linear projection + Causal convolution paths (with reduced channels)
- Path fusion with ResidualMLP for feature combination
- Unidirectional LSTM (256 hidden units, 3 layers) for causal temporal encoding
- Multi-layer processing with advanced normalization
- Output: 64-dimensional latent means (mu_x)

**Key Innovations**:
- **Intelligent channel reduction** with learned attention weights
- Causal convolutions prevent future information leakage
- Depthwise separable convolutions for computational efficiency
- Pre-norm design for superior gradient flow
- Multiple kernel sizes [9,7,5,3] for multi-scale feature extraction
- **37% reduction in encoder complexity** while maintaining feature quality

### 3. TargetEncoder
**Purpose**: Encodes target signals (scattering + phase harmonic) into variational parameters for prior modeling with channel reduction.

**Architecture**:
- **Channel reduction**: Separate 76 → 48 channel reduction for each modality using attention-based blocks
- Dual-modal processing with separate paths for scattering and phase harmonic inputs (with reduced channels)
- Intra-modal fusion within each modality using ResidualMLP
- Cross-modal fusion layer combining both modalities
- Bidirectional LSTM (128 hidden units, 3 layers) for rich temporal modeling
- Outputs: mu_y (64-dim) and logvar_y (128-dim split into prior and conditional features)

**Advanced Features**:
- **Modality-specific channel attention** learns optimal feature subsets for each input type
- Separate linear and causal convolution paths for each modality (with reduced channels)
- Advanced fusion strategy with residual connections
- Bidirectional LSTM captures forward and backward temporal dependencies
- Larger logvar output enables flexible posterior modeling
- **Reduced intermediate dimensions** (32 channels per path) improve computational efficiency

### 4. ConditionalEncoder
**Purpose**: Implements q(z|x,y) - the posterior distribution conditioned on both source and target inputs.

**Architecture**:
- ResidualMLP taking concatenated source representation and conditional features
- Multi-layer processing with dropout regularization
- Outputs posterior parameters (mu_post, logvar_post) for latent variable z
- Enables the TEB framework's conditional modeling capability

### 5. Decoder (Raw Signal Decoder)
**Purpose**: Reconstructs raw temporal signals from latent variables with 16x upsampling and uncertainty quantification.

**Architecture**:
- **Latent Processing**: ResidualMLP for feature extraction from latent variables
- **Progressive Upsampling**: 
  - Stage 1: 1x → 4x upsampling using learnable transposed convolutions
  - Stage 2: 4x → 16x upsampling with refined temporal reconstruction
- **Signal Refinement**: Multiple CausalResidualBlock layers for high-quality reconstruction
- **Uncertainty Quantification**: Separate heads for mean and log-variance prediction

**Key Design Features**:
- **UpsamplingBlock**: Learnable transposed convolutions with anti-aliasing filters
- **Residual Connections**: Skip connections preserve features during upsampling
- **Causal Constraints**: Maintains temporal causality during signal generation
- **NLL Loss**: Gaussian negative log-likelihood for uncertainty-aware training

## Advanced Components

### ChannelReductionBlock
- **Intelligent channel compression** from 76 to 48 channels
- **Channel attention mechanism** learns importance weights for each channel
- Depthwise separable convolutions for efficient processing
- Pointwise convolution for optimal channel combinations
- Pre-norm design with LayerNorm and GELU activation
- **Reduces model complexity by ~37%** while preserving essential features

### UpsamplingBlock
- Learnable transposed convolutions for intelligent upsampling
- Anti-aliasing filters prevent reconstruction artifacts
- Residual connections with proper dimension matching
- Group normalization and GELU activation for stable training

### CausalConv1d
- Custom causal convolution preventing future information leakage
- Efficient left-padding strategy for memory optimization
- Support for grouped convolutions and mixed precision
- Pre-allocated padding tensors for computational efficiency

### CausalResidualBlock
- Residual blocks with causal convolutions and dilation support
- **Channel contraction support** for dimensionality reduction
- GLU-style gating mechanisms for effective feature selection
- Pre-norm design for superior gradient flow
- Support for depthwise separable convolutions

### ResidualMLP
- Modular MLP with residual connections and layer normalization
- Flexible hidden dimensions with configurable activation
- Input normalization and post-processing capabilities
- Dropout regularization for generalization

### Initialization Function
- State-of-the-art weight initialization for all model components
- Xavier/Glorot initialization for linear and convolution layers
- Orthogonal initialization for LSTM weights with forget gate bias
- Proper LayerNorm initialization for training stability

## Training Methodology

### Loss Components
1. **Raw Signal Reconstruction Loss**: Gaussian negative log-likelihood between predicted and true raw signals
2. **KL Divergence**: Regularizes posterior q(z|x,y) to stay close to prior p(z|y)
3. **Uncertainty Quantification**: Log-variance predictions enable confidence estimation

### Key Training Features
- Numerical stability measures (logvar clamping [-10, 10])
- Improved initialization scheme for all model components
- Gradient clipping compatibility through architecture design
- Progressive upsampling strategy for artifact-free reconstruction

### Loss Interface Compliance
```python
{
    "total_loss": combined_loss,
    "reconstruction_error": raw_signal_nll,  # Required interface
    "reconstruction_loss": raw_signal_nll,   # Backward compatibility
    "kld_loss": kl_divergence,
    "classification_loss": None,             # Required interface
    "raw_signal_loss": raw_signal_nll
}
```


## Model Innovations

### 1. Raw Signal Focus
- Exclusive focus on raw temporal signal reconstruction
- 16x temporal upsampling (T=16 decimation factor compensation)
- Direct time-domain prediction with uncertainty quantification

### 2. Progressive Upsampling Strategy
- Two-stage upsampling: 1x→4x→16x for artifact-free reconstruction
- Learnable interpolation superior to simple upsampling methods
- Anti-aliasing filters prevent high-frequency artifacts

### 3. Advanced Uncertainty Quantification
- Predicts both signal mean and log-variance
- Enables negative log-likelihood loss computation
- Provides confidence measures for each predicted sample

### 4. TEB Framework Implementation
- Separates prior p(z|y) from posterior q(z|x,y) modeling
- Enables conditional generation with uncertainty quantification
- Maintains causal constraints for real-time applicability

### 5. Multi-Scale Feature Processing with Channel Reduction
- Multiple convolution kernel sizes capture different temporal scales
- **Channel reduction blocks** compress 76-channel inputs to 48 channels using attention-based selection
- Dual-path processing in encoders for comprehensive feature extraction
- Residual connections preserve information flow
- Learned channel correlations reduce computational overhead while preserving essential features

## Technical Specifications

- **Input Channels**: 76 (scattering transform + phase harmonic features)
- **Reduced Channels**: 48 (after channel reduction blocks)
- **Sequence Length**: 300 timesteps at input resolution
- **Output Resolution**: 4800 timesteps (300 × 16) raw signal samples
- **Decimation Factor**: 16 (compensated by decoder upsampling)
- **Latent Dimensions**: 64 (source), 64 (target), 64 (conditional)
- **Upsampling Strategy**: Progressive 1x→4x→16x
- **Loss Type**: Gaussian negative log-likelihood + KL divergence
- **Channel Reduction**: 37% complexity reduction with attention-based selection
- **Total Parameters**: ~1.4M (estimated, reduced with channel compression)

## Input/Output Specifications

### Input Format
- **x_ph**: Source phase harmonic (batch_size, 76, 300)
- **y_st**: Target scattering transform (batch_size, 76, 300)  
- **y_ph**: Target phase harmonic (batch_size, 76, 300)
- **y_raw**: Ground truth raw signal (batch_size, 4800, 1)

### Output Format
- **raw_signal_mu**: Predicted signal mean (batch_size, 4800, 1)
- **raw_signal_logvar**: Predicted log-variance (batch_size, 4800, 1)

## Use Cases

1. **Raw Signal Reconstruction**: High-fidelity temporal signal generation from compressed representations
2. **Uncertainty-Aware Prediction**: Confidence-weighted signal forecasting
3. **Signal Compression**: Efficient encoding/decoding of temporal signals
4. **Real-Time Processing**: Causal architecture supports streaming applications
5. **Multi-Modal Feature Fusion**: Combining scattering and phase harmonic information

## Key Advantages

- **High-Quality Reconstruction**: Progressive upsampling with learnable filters
- **Uncertainty Quantification**: Built-in confidence estimation
- **Causal Design**: Real-time processing capability
- **Efficient Architecture**: Focused on single objective with optimized components
- **Numerical Stability**: Robust training with proper initialization and clamping
- **Interface Compliance**: Standardized loss dictionary for integration

## Training Example

```python
model = SeqVaeTeb(
    input_channels=76,
    sequence_length=300,
    decimation_factor=16,
    latent_dim_z=64,
    kld_beta=1.0
)

# Forward pass
forward_outputs = model(y_st, y_ph, x_ph)

# Compute loss
loss_dict = model.compute_loss(forward_outputs, y_raw)
total_loss = loss_dict['total_loss']

# Prediction
raw_predictions = model.predict_raw_signal(x_ph, y_st, y_ph)
```

This architecture provides a robust foundation for raw signal reconstruction while maintaining compatibility with the TEB framework, enabling high-quality temporal signal generation with built-in uncertainty quantification.