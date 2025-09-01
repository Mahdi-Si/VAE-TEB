# Raw Signal Decoder Architecture Documentation

## Overview

The Raw Signal Decoder is an extension of the VAE TEB (Variational Autoencoder with Target-Encoder-Bank) model architecture designed to decode latent representations back to raw temporal signals. Unlike the standard decoder that predicts scattering transform and phase harmonic features, this decoder reconstructs the original time-domain signal, requiring sophisticated upsampling and temporal reconstruction capabilities.

## Key Design Challenge: Temporal Decimation Mapping

### Scattering Transform Parameters
- **T=16**: Decimation factor (temporal downsampling)
- **J=11**: Number of octaves in frequency decomposition  
- **Q=1**: Number of wavelets per octave
- **Signal mapping**: Latent time index `t=i` maps to raw signal time `t=i*16`

This means the decoder must upsample by a factor of 16x to reconstruct the original temporal resolution.

## Core Architecture Components

### 1. Raw Signal Decoder Module

#### RawSignalDecoder
- **Purpose**: Reconstructs raw temporal signal from latent variables with 16x upsampling
- **Architecture**:
  1. **Latent Processing**: Multi-path latent feature extraction
  2. **Temporal Upsampling**: Progressive upsampling with learnable interpolation
  3. **Causal Reconstruction**: Maintains causal constraints during signal generation
  4. **Signal Refinement**: Multi-scale refinement for high-quality reconstruction

### 2. Upsampling Strategy

#### Progressive Upsampling Path
- **Stage 1**: 1x → 4x upsampling using transposed convolutions
- **Stage 2**: 4x → 16x upsampling with refined temporal reconstruction
- **Rationale**: Progressive upsampling prevents artifacts and maintains temporal coherence

#### Learnable Interpolation
- **Method**: Combination of learned transposed convolutions and adaptive interpolation
- **Benefits**: Better reconstruction quality than simple interpolation
- **Implementation**: Custom upsampling blocks with residual connections

### 3. Architecture Flow

```
Latent Input (B, S, Z) → Temporal Processing → Progressive Upsampling → Raw Signal (B, S*16, 1)
    ↓                        ↓                       ↓                      ↓
Multi-path extraction → Causal convolutions → 1x→4x→16x upsample → Signal refinement
```

## Design Rules and Conventions

### Input/Output Requirements
- **Input Format**: Latent variables (Batch_size, Sequence_length, Latent_dim)
- **Output Format**: Raw signal (Batch_size, Sequence_length*16, 1)
- **Temporal Mapping**: Each latent timestep reconstructs 16 raw signal samples
- **Causality**: Maintain causal constraints - latent[t] only reconstructs signal[t*16:(t+1)*16]

### Architectural Conventions

#### 1. Upsampling Design
- **Progressive strategy**: Multi-stage upsampling (1x→4x→16x)
- **Transposed convolutions**: Learnable upsampling with proper stride and padding
- **Anti-aliasing**: Smoothing filters to prevent reconstruction artifacts
- **Temporal alignment**: Ensure proper phase alignment across upsampling stages
- **Uncertainty**: We want to also predict the logvar of uncertainty in the prediction, which would be same shape tensor as prediction and we can use it to calculate negative loglikelihood loss. 

#### 2. Temporal Processing
- **Causal convolutions**: For maintaining temporal causality
- **Dilated convolutions**: For multi-scale temporal context
- **Residual connections**: For gradient flow and feature preservation
- **Layer normalization**: Applied consistently throughout

#### 3. Signal Quality Optimization
- **Smoothness regularization**: Prevent high-frequency artifacts
- **Amplitude preservation**: Maintain signal energy across reconstruction

### Implementation Guidelines

#### 1. Upsampling Block Structure
```python
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor):
        super().__init__()
        # Transposed convolution for learnable upsampling
        # Anti-aliasing filter for artifact prevention
        # Residual connection for feature preservation
        # Layer normalization for stable training
        
    def forward(self, x):
        # Apply transposed convolution
        # Apply anti-aliasing filter
        # Add residual connection
        # Apply normalization
```

#### 2. Causal Reconstruction
- **No future leakage**: Each output sample only depends on current and past latent states
- **Streaming capability**: Support for real-time reconstruction
- **Memory efficiency**: Minimal buffering requirements

#### 3. Loss Computation
- **Selective computation**: Allow enabling/disabling specific loss components
- **Temporal weighting**: Optional weighting for different time horizons
- **Signal-to-noise ratio**: Include SNR metrics for quality assessment
- **Negative loglikelihood loss**: Since we also want to predict the negative logvar or uncertainty of the predictions we can calculate the NLL loss.

### Specific Design Considerations

#### 1. Temporal Upsampling Architecture
```python
class RawSignalDecoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, decimation_factor=16):
        super().__init__()
        
        # Latent processing path
        self.latent_processor = ResidualMLP(...)
        
        # Progressive upsampling
        self.upsample_4x = UpsamplingBlock(in_channels, mid_channels, 4)
        self.upsample_16x = UpsamplingBlock(mid_channels, 1, 4)  # Total 16x
        
        # Signal refinement
        self.signal_refiner = CausalResidualBlock(...)
        
    def forward(self, latent_z):
        # Process latent features
        # Apply progressive upsampling
        # Refine reconstructed signal
        # Return raw signal (B, S*16, 1)
```

#### 2. Loss Function Design
- **Primary loss**: MSE between reconstructed and target raw signals
- **Smoothness loss**: L2 penalty on signal derivatives for artifact reduction
- **Causality verification**: Ensure no future information leakage during training

#### 3. Training Considerations
- **Progressive training**: Start with lower upsampling factors, gradually increase
- **Learning rate scheduling**: Separate schedules for upsampling vs processing components
- **Gradient clipping**: Prevent instability during high-resolution reconstruction
- **Memory management**: Efficient handling of 16x longer sequences

### Integration with Existing VAE TEB

#### 1. Model Extension
- **Preserve existing architecture**: Keep current encoder-decoder structure intact
- **Add raw signal branch**: Parallel decoder for raw signal reconstruction
- **Shared latent space**: Use same latent variables z for both decoders
- **Optional training**: Allow training with/without raw signal reconstruction

#### 2. Loss Dictionary Extension
```python
{
    "total_loss": total_loss,
    "reconstruction_loss": recon_loss,      # Original ST/PH reconstruction
    "raw_signal_loss": raw_signal_loss,     # New raw signal reconstruction
    "kld_loss": kld_loss,
    "spectral_loss": spectral_loss          # Optional frequency domain loss
}
```

#### 3. Inference Modes
- **Scattering prediction**: Original mode predicting ST/PH features
- **Raw signal prediction**: New mode predicting raw temporal signal
- **Dual prediction**: Both modes simultaneously for comprehensive analysis

### Testing Requirements

#### 1. Temporal Alignment Verification
- **Phase coherence**: Verify proper temporal alignment across upsampling
- **Causality testing**: Ensure no future information leakage
- **Signal continuity**: Check for discontinuities at upsampling boundaries

#### 2. Reconstruction Quality
- **Signal-to-noise ratio**: Measure reconstruction quality
- **Spectral fidelity**: Compare frequency content of reconstructed vs original
- **Temporal dynamics**: Verify preservation of temporal patterns

#### 3. Computational Efficiency
- **Memory usage**: Monitor memory consumption with 16x longer sequences
- **Training stability**: Ensure stable convergence with progressive upsampling
- **Inference speed**: Benchmark reconstruction time for real-time applications

### Usage Examples

#### 1. Training with Raw Signal Reconstruction
```python
# Enable raw signal decoder during training
model = SeqVaeTebWithRawDecoder(
    enable_raw_signal_decoder=True,
    decimation_factor=16,
    raw_signal_weight=1.0
)

# Training loop includes raw signal loss
loss_dict = model(x_ph, y_st, y_ph, y_raw)
# loss_dict contains both original and raw signal losses
```

#### 2. Inference for Raw Signal Prediction
```python
# Predict raw signal from latent representation
with torch.no_grad():
    latent_z = model.encode(x_ph, y_st, y_ph)
    reconstructed_signal = model.raw_signal_decoder(latent_z)
    # reconstructed_signal: (B, S*16, 1)
```

## Extension Guidelines

### 1. Adding Multi-Resolution Support
- Support different decimation factors beyond 16x
- Configurable upsampling strategies
- Adaptive loss weighting based on resolution

### 2. Real-Time Processing
- Streaming-friendly architecture
- Minimal latency upsampling
- Causal-only processing for online applications

### 3. Multi-Signal Reconstruction
- Support for multi-channel raw signals
- Cross-channel dependency modeling
- Coherent reconstruction across channels

This architecture provides a robust foundation for raw signal reconstruction while maintaining compatibility with the existing VAE TEB framework, enabling both traditional scattering transform prediction and direct temporal signal generation from the same latent representation.