# SeqVAE-TEB: Sequential Variational Autoencoder with Transfer Entropy Bottleneck

## Overview

The SeqVAE-TEB model implements the Transfer Entropy Bottleneck (TEB) framework for learning directed information flow between time series. It extends the classical Information Bottleneck principle to the conditional, directed time-series setting, forcing the model to retain only the information from a source signal $X$ that is truly predictive of a target signal $Y$'s future.

## Mathematical Foundation

### Transfer Entropy
The core concept is **transfer entropy**, which measures directed information flow:

$$\text{TE}_{X \to Y} = I(X_{\text{past}}; Y_{\text{future}} \mid Y_{\text{past}})$$

This quantifies how much knowing the past of $X$ reduces uncertainty about the future of $Y$, beyond what $Y$'s own past already provides.

### TEB Objective
The TEB framework uses a variational approach with three key components:

1. **Encoder**: $q_\phi(z \mid X_{1:t}, Y_{1:t})$ - Posterior distribution
2. **Conditional Prior**: $r_\psi(z \mid Y_{1:t}) \approx p(z \mid Y_{1:t})$ - Context prior
3. **Decoder**: $p_\theta(Y_{\text{raw}} \mid z)$ - Reconstruction model

The TEB loss function is:

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[-\log p_\theta(Y_{\text{raw}} \mid z)]}_{\text{Reconstruction Loss}} + \beta \underbrace{\text{KL}(q_\phi(z \mid X_{1:t}, Y_{1:t}) \| r_\psi(z \mid Y_{1:t}))}_{\text{Information Bottleneck}}$$

## Model Architecture

### 1. Source Encoder

**Purpose**: Encodes source signal $X$ to produce deterministic representation $h_x$.

**Input**: $X_{\text{ph}} \in \mathbb{R}^{B \times 300 \times 130}$ (cross-phase features)

**Architecture**:
```
X_ph → ResidualMLP(130→32) → CausalConv1D Stack → LSTM(32→128) → ResidualMLP → h_x ∈ ℝ^{B×300×16}
```

**Mathematical Formulation**:
$$h_x = \text{SourceEncoder}(X_{\text{ph}}) = f_{\text{source}}(X_{\text{ph}})$$

### 2. Target Encoder

**Purpose**: Encodes target signal $Y$ to produce prior distribution parameters and conditioning features.

**Inputs**: 
- $Y_{\text{st}} \in \mathbb{R}^{B \times 300 \times 43}$ (scattering transform features)
- $Y_{\text{ph}} \in \mathbb{R}^{B \times 300 \times 44}$ (phase harmonic features)

**Architecture**:
```
Y_st → ResidualMLP(43→16) → CausalConv1D Stack
                                                  → Cross-Modal Fusion → LSTM → Outputs
Y_ph → ResidualMLP(44→16) → CausalConv1D Stack
```

**Outputs**:
- $\mu_{\text{prior}} \in \mathbb{R}^{B \times 300 \times 16}$: Prior mean
- $\log\sigma^2_{\text{prior}} \in \mathbb{R}^{B \times 300 \times 16}$: Prior log-variance  
- $c_y \in \mathbb{R}^{B \times 300 \times 16}$: Conditioning features

**Mathematical Formulation**:
$$(\mu_{\text{prior}}, \log\sigma^2_{\text{prior}}, c_y) = \text{TargetEncoder}(Y_{\text{st}}, Y_{\text{ph}})$$

$$p(z \mid Y_{1:t}) = \mathcal{N}(\mu_{\text{prior}}, \sigma^2_{\text{prior}})$$

### 3. Conditional Encoder

**Purpose**: Produces posterior distribution $q(z \mid X, Y)$ by combining source and target information.

**Inputs**: 
- $h_x \in \mathbb{R}^{B \times 300 \times 16}$ (from Source Encoder)
- $c_y \in \mathbb{R}^{B \times 300 \times 16}$ (from Target Encoder)

**Architecture**:
```
[h_x; c_y] → ResidualMLP(32→20) → {μ_post, σ²_post} heads
```

**Mathematical Formulation**:
$$h_{\text{combined}} = [h_x; c_y] \in \mathbb{R}^{B \times 300 \times 32}$$

$$(\mu_{\text{post}}, \log\sigma^2_{\text{post}}) = \text{ConditionalEncoder}(h_{\text{combined}})$$

**Key Implementation Detail**:
```python
mu_post = mu_post + mu_prior  # Authors' implementation
```

$$q(z \mid X_{1:t}, Y_{1:t}) = \mathcal{N}(\mu_{\text{post}} + \mu_{\text{prior}}, \sigma^2_{\text{post}})$$

### 4. Decoder

**Purpose**: Progressive upsampling decoder that reconstructs the raw signal from latent representations.

**Input**: $z \in \mathbb{R}^{B \times 300 \times 16}$ (sampled latent variables)

**Architecture**:
```
z → Feature Expansion → ConvTranspose1D Progressive Upsampling → Signal Reconstruction
  (16→87)              (300→600→1200→2400→4800)              (μ_recon, σ²_recon)
```

**Upsampling Chain**:
$$300 \xrightarrow{\times 2} 600 \xrightarrow{\times 2} 1200 \xrightarrow{\times 2} 2400 \xrightarrow{\times 2} 4800$$

**Outputs**:
- Linear features: $z_{\text{expanded}} \in \mathbb{R}^{B \times 300 \times 87}$
- Signal mean: $\mu_{\text{recon}} \in \mathbb{R}^{B \times 4800}$
- Signal log-variance: $\log\sigma^2_{\text{recon}} \in \mathbb{R}^{B \times 4800}$

**Mathematical Formulation**:
$$(z_{\text{expanded}}, \mu_{\text{recon}}, \log\sigma^2_{\text{recon}}) = \text{Decoder}(z)$$

$$p(Y_{\text{raw}} \mid z) = \mathcal{N}(\mu_{\text{recon}}, \sigma^2_{\text{recon}})$$

## Reparameterization Trick

The model uses the standard VAE reparameterization trick for gradient flow:

$$z = \mu_{\text{post}} + \epsilon \cdot \sigma_{\text{post}}$$

where $\epsilon \sim \mathcal{N}(0, I)$.

**Implementation**:
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

## Loss Function Components

### 1. Reconstruction Loss

**MSE Loss** for linear features:
$$\mathcal{L}_{\text{MSE}} = \|\text{linear\_output} - [Y_{\text{st}}; Y_{\text{ph}}]\|^2$$

**Negative Log-Likelihood Loss** for raw signal:
$$\mathcal{L}_{\text{NLL}} = \mathbb{E}\left[\frac{1}{2}\left(\log\sigma^2_{\text{recon}} + \frac{(Y_{\text{raw}} - \mu_{\text{recon}})^2}{\sigma^2_{\text{recon}}}\right)\right]$$

### 2. KL Divergence Loss (Information Bottleneck)

The core TEB component - KL divergence between posterior and prior:

$$\mathcal{L}_{\text{KL}} = \text{KL}(q(z \mid X, Y) \| p(z \mid Y))$$

**Analytical Form**:
$$\mathcal{L}_{\text{KL}} = \frac{1}{2}\left[\log\sigma^2_{\text{prior}} - \log\sigma^2_{\text{post}} - 1 + \frac{\sigma^2_{\text{post}} + (\mu_{\text{post}} - \mu_{\text{prior}})^2}{\sigma^2_{\text{prior}}}\right]$$

### 3. Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{NLL}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

where $\beta$ controls the strength of the information bottleneck.

## Information-Theoretic Interpretation

### What the Model Learns

1. **Maximizes**: $I(z; Y_{\text{future}} \mid Y_{\text{past}})$ through reconstruction loss
2. **Minimizes**: $I(z; X_{\text{past}} \mid Y_{\text{past}})$ through KL regularization

### Transfer Entropy Measurement

The model can measure transfer entropy as:

$$\hat{\text{TE}}_{X \to Y} = \mathbb{E}[\text{KL}(q(z \mid X, Y) \| p(z \mid Y))]$$

This provides a quantitative measure of directed information flow from $X$ to $Y$.

## Key Implementation Features

### 1. Causal Convolutions
All convolutions use causal padding to ensure no future information leaks:
```python
self.left_padding = (kernel_size - 1) * dilation
x = F.pad(x, (self.left_padding, 0))  # Left padding only
```

### 2. Normalization Strategy
- **LayerNorm**: Used in MLP components for stable training
- **GroupNorm**: Used in convolutional layers for better performance with small batch sizes
- **Pre-normalization**: Applied before activations for improved gradient flow

### 3. Memory Optimization
Explicit tensor deletion for memory efficiency:
```python
del intermediate_tensor  # Clean up to prevent memory leaks
```

### 4. Numerical Stability
Log-variance clamping to prevent numerical instabilities:
```python
logvar = torch.clamp(logvar, min=-10, max=10)
```

## Model Variants

The codebase also includes `SeqVaeTebClassifier` which:
1. Uses pre-trained SeqVaeTeb for feature extraction
2. Adds an InceptionTime classifier on latent representations
3. Supports both frozen and fine-tuned training modes

## Applications

This model is particularly useful for:
- **Physiological signal analysis** (FHR monitoring)
- **Causal discovery** in time series
- **Transfer learning** with information-theoretic regularization
- **Directed connectivity analysis** in complex systems

## Summary

The SeqVAE-TEB model successfully implements the Transfer Entropy Bottleneck framework by:

1. Learning compressed representations that preserve directed information flow
2. Using a three-encoder architecture (source, target, conditional) 
3. Enforcing information bottleneck through KL regularization
4. Providing quantitative transfer entropy measurements
5. Supporting both reconstruction and classification tasks

The model architecture ensures that the latent space $z$ contains only the information from $X$ that is genuinely predictive of $Y$'s future, making it a powerful tool for causal time series analysis.