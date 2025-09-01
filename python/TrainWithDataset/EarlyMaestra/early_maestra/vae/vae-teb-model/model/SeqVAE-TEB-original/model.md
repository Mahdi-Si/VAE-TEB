# SeqVaeTeb: Transfer Entropy Bottleneck VAE

This document describes **SeqVaeTeb**, a sequence-based Variational Autoencoder (VAE) that uses the Transfer Entropy Bottleneck (TEB) principle. It learns a compressed latent representation of a target signal (Fetal Heart Rate, FHR) while incorporating a minimal, necessary amount of information from a source signal (Uterine Pressure, UP).

- **Reference**: [Transfer Entropy Bottleneck (TEB)](https://arxiv.org/pdf/2211.16607)
- **Dataset**: See `documents/dataset.md` for details on features and normalization.

## 1. Core Concept

The model learns a latent sequence **z** for the target FHR signal. The goal is to make **z** expressive enough to reconstruct the FHR accurately, while constraining the information it draws from the source UP signal. This information flow is measured and penalized using a proxy for transfer entropy, implemented as a Kullback-Leibler (KL) divergence.

**Key Functions:**
- Encodes FHR and UP features into latent distributions.
- Reconstructs the raw FHR signal from the latent representation **z**.
- Provides a direct method to measure the transfer entropy from the UP source to the FHR latent representation.

## 2. Probabilistic & Mathematical Formulation

The model defines a prior distribution conditioned on the target **y** and a posterior distribution conditioned on both the target **y** and the source **x**.

- **Target (FHR) Features**: $\mathbf{y}_t = [\mathbf{y}^{st}_t, \mathbf{y}^{ph}_t] \in \mathbb{R}^{87}$
- **Source (UP) Features**: $\mathbf{x}_t \in \mathbb{R}^{130}$
- **Latent Variable**: $\mathbf{z}_t \in \mathbb{R}^{d}$ (latent dimension `d=32`)
- **Raw FHR Signal**: $\mathbf{r} \in \mathbb{R}^{4800}$

### 2.1. Encoders

**Target Encoder ($f_t$)**
The target encoder processes both scattering transform and phase harmonic features of the FHR signal to define the parameters of the prior distribution $p(\mathbf{z}_t | \mathbf{y}_t)$.

$$ (\boldsymbol{\mu}^{y}_t, \mathbf{v}_t) = f_t(\mathbf{y}^{st}_t, \mathbf{y}^{ph}_t) $$

The architecture uses dual-path processing:
- Scattering features (43D): ResidualMLP → CausalMultiChannelConvBlock stack
- Phase features (44D): ResidualMLP → CausalMultiChannelConvBlock stack  
- Cross-modal fusion → LSTM (bidirectional optional) → Two output heads

- $\boldsymbol{\mu}^{y}_t \in \mathbb{R}^{d}$ is the prior mean.
- The `TargetEncoder` produces a single tensor `logvar_y_full` of shape `(B, T, 2d)`. This tensor is then split into two separate tensors:
  - $\log\boldsymbol{\sigma}^{2,y}_t \in \mathbb{R}^{d}$: The log-variance for the prior distribution $p(\mathbf{z}_t | \mathbf{y}_t)$.
  - $\mathbf{c}_t \in \mathbb{R}^{d}$: A conditioning feature that is passed to the `ConditionalEncoder` to help model the posterior.

The **prior distribution** is a diagonal Gaussian:
$$ p(\mathbf{z}_t | \mathbf{y}_t) = \mathcal{N}(\mathbf{z}_t | \boldsymbol{\mu}^{y}_t, \text{diag}(\boldsymbol{\sigma}^{2,y}_t)) $$

**Source Encoder ($f_s$)**
The source encoder creates a deterministic representation **h** from the source features **x**. The encoder uses ResidualMLP blocks with causal convolutions and LSTM processing:
$$ \mathbf{h}^x_t = f_s(\mathbf{x}_t) \in \mathbb{R}^{d} $$
*Note: In the code, this is named `mu_x` for consistency, but it is a deterministic encoding, not a distribution's mean.*

The architecture follows: Input (130D) → ResidualMLP → CausalMultiChannelConvBlock stack → LSTM (unidirectional) → Final ResidualMLP → Output (32D).

**Conditional Encoder ($f_c$)**
The conditional encoder models the **posterior distribution** $q(\mathbf{z}_t | \mathbf{x}_t, \mathbf{y}_t)$ by combining the source representation $\mathbf{h}^x_t$ and the target-derived conditioning feature $\mathbf{c}_t$.

$$ (\tilde{\boldsymbol{\mu}}^{post}_t, \log\boldsymbol{\sigma}^{2,post}_t) = f_c([\mathbf{h}^x_t, \mathbf{c}_t]) $$

The final posterior mean is shifted by the prior mean to center it:
$$ \boldsymbol{\mu}^{post}_t = \tilde{\boldsymbol{\mu}}^{post}_t + \boldsymbol{\mu}^{y}_t $$

The posterior is also a diagonal Gaussian:
$$ q(\mathbf{z}_t | \mathbf{x}_t, \mathbf{y}_t) = \mathcal{N}(\mathbf{z}_t | \boldsymbol{\mu}^{post}_t, \text{diag}(\boldsymbol{\sigma}^{2,post}_t)) $$

### 2.2. Reparameterization

To enable gradient-based training, we sample from the posterior using the reparameterization trick:
$$ \mathbf{z}_t = \boldsymbol{\mu}^{post}_t + \boldsymbol{\sigma}^{post}_t \odot \boldsymbol{\epsilon}_t, \quad \text{where} \; \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) $$

### 2.3. Decoder ($f_d$)

The decoder reconstructs the raw FHR signal from the full latent sequence $\mathbf{z}_{1:T}$. The architecture uses ResidualMLP blocks followed by MultiChannelConvBlock layers with upsampling for signal reconstruction. It has two prediction heads:

1.  **Raw Signal Reconstruction**: Predicts the mean and log-variance of the raw FHR signal using separate ResidualMLP heads.
    $$ (\boldsymbol{\mu}^{raw}, \log\boldsymbol{\sigma}^{2,raw}) = f_{d,raw}(\mathbf{z}_{1:T}) $$
    The likelihood is a Gaussian distribution over the 4800-sample window:
    $$ p(\mathbf{r} | \mathbf{z}_{1:T}) = \mathcal{N}(\mathbf{r} | \boldsymbol{\mu}^{raw}, \text{diag}(\boldsymbol{\sigma}^{2,raw})) $$

2.  **Auxiliary Feature Reconstruction**: An intermediate linear head predicts the concatenated target features (scattering + phase harmonic) to stabilize training.
    $$ \widehat{\mathbf{y}}_t = f_{d,aux}(\mathbf{z}_t) \in \mathbb{R}^{87} $$
    where 87 = 43 (scattering) + 44 (phase harmonic) dimensions.

## 3. Loss Function

The total loss is a combination of reconstruction error and the KL divergence penalty.

$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} $$

### 3.1. Reconstruction Loss

The reconstruction loss has two components:

1.  **NLL Loss** for the raw signal:
    $$ \mathcal{L}_{\text{NLL}} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{2} \left( \log \sigma^{2,raw}_n + \frac{(r_n - \mu^{raw}_n)^2}{\sigma^{2,raw}_n} \right) $$

2.  **MSE Loss** for the auxiliary feature head:
    $$ \mathcal{L}_{\text{MSE}} = \frac{1}{T} \sum_{t=1}^{T} \|\widehat{\mathbf{y}}_t - \mathbf{y}_t\|_2^2 $$

So, $\mathcal{L}_{\text{recon}} = \mathcal{L}_{\text{NLL}} + \mathcal{L}_{\text{MSE}}$.

### 3.2. KL Divergence (Transfer Entropy Proxy)

The KL divergence between the posterior and the prior serves as the TEB penalty. It quantifies the information the source **x** adds to the latent representation **z**.

$$ \mathcal{L}_{\text{KL}} = \frac{1}{T} \sum_{t=1}^{T} \text{KL}[q(\mathbf{z}_t | \mathbf{x}_t, \mathbf{y}_t) ||| p(\mathbf{z}_t | \mathbf{y}_t)] $$

For diagonal Gaussians, this is:
$$ \text{KL}_t = \frac{1}{2} \sum_{i=1}^{d} \left[ \log \sigma^{2,y}_{t,i} - \log \sigma^{2,post}_{t,i} - 1 + \frac{\sigma^{2,post}_{t,i} + (\mu^{post}_{t,i} - \mu^{y}_{t,i})^2}{\sigma^{2,y}_{t,i}} \right] $$

The hyperparameter **β** controls the strength of the information bottleneck.

## 4. Architecture & I/O

| Module | Input(s) | Output(s) | Description |
|---|---|---|---|
| **`SourceEncoder`** | `x_ph`: (B, 300, 130) | `mu_x`: (B, 300, 32) | ResidualMLP → CausalMultiChannelConvBlock stack → LSTM → ResidualMLP |
| **`TargetEncoder`** | `y_st`: (B, 300, 43)<br>`y_ph`: (B, 300, 44) | `mu_y`: (B, 300, 32)<br>`logvar_y_full`: (B, 300, 64) | Dual-path ResidualMLPs & CausalMultiChannelConvBlocks → Cross-modal fusion → LSTM → Two ResidualMLP heads |
| **`ConditionalEncoder`**| `h_x`: (B, 300, 32)<br>`c_logvar`: (B, 300, 32) | `mu_post_shift`: (B, 300, 32)<br>`logvar_post`: (B, 300, 32) | ResidualMLP merger → Two ResidualMLP heads for posterior parameters |
| **`Decoder`** | `z`: (B, 300, 32) | `linear_output`: (B, 300, 87)<br>`mu_raw`: (B, 4800)<br>`logvar_raw`: (B, 4800) | ResidualMLP sequence → MultiChannelConvBlock stack with upsampling → Two ResidualMLP heads |

## 5. Usage

### Training Loop

```python
import torch
from model.vae_teb_model import SeqVaeTeb
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

# 1. Initialize model and optimizer
model = SeqVaeTeb(
    input_channels=76,
    sequence_length=300, 
    latent_dim_source=32,
    latent_dim_target=32,
    latent_dim_z=32,
    decimation_factor=16,
    warmup_period=30
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. Load data
train_loader = create_optimized_dataloader(...)

# 3. Training step
for batch in train_loader:
    y_st, y_ph, x_ph = batch['fhr_st'], batch['fhr_ph'], batch['fhr_up_ph']
    y_raw = batch['fhr'][:, :4800]

    # Forward pass
    outputs = model(y_st, y_ph, x_ph)
    
    # Compute loss
    losses = model.compute_loss(
        forward_outputs=outputs,
        y_st=y_st,
        y_ph=y_ph, 
        y_raw=y_raw,
        compute_kld_loss=True,
        beta=1.0
    )
    
    # Backward pass and optimization
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()
```

### Measuring Transfer Entropy

The KL divergence can be measured directly, providing the transfer entropy proxy.

```python
# Get TE per latent dimension at each timestep
te_tensor = model.measure_transfer_entropy(y_st, y_ph, x_ph, reduce_mean=False)
# shape: (batch, sequence_len, latent_dim)

# Get a single scalar value for the average TE
te_scalar = model.measure_transfer_entropy(y_st, y_ph, x_ph, reduce_mean=True)
```
This is useful for analysis, such as studying how time-shifting the source signal `x_ph` affects the information flow into `z`.

## 6. Key Implementation Details

### 6.1. ResidualMLP Architecture
The model extensively uses `ResidualMLP` blocks with:
- Input LayerNorm for gradient stability
- Configurable skip connections with dimension matching
- Optional final activation before residual connection
- Geometric scheduling for hidden layer dimensions

### 6.2. Causal Convolutions
All convolutions use `CausalMultiChannelConvBlock` to ensure:
- No future information leakage (left-padding only)
- Batch normalization and configurable activation functions
- Optional upsampling with linear interpolation
- Dilation support for larger receptive fields

### 6.3. Memory Optimizations
The implementation includes several optimizations:
- Explicit memory cleanup with `del` statements
- Contiguous tensor operations to reduce memory fragmentation
- Clamped log-variance values to prevent numerical instability (`torch.clamp(logvar, min=-10, max=10)`)
- Efficient tensor transpose operations

### 6.4. Advanced Initialization
The model uses `initialization()` function with:
- Xavier/Glorot initialization for linear and convolutional layers
- Orthogonal initialization for LSTM weights
- Forget gate bias initialization to 1.0 for better gradient flow

### 6.5. Model Variants
The codebase includes `SeqVaeTebClassifier` for classification tasks:
- Can load pretrained VAE weights
- Supports freezing/unfreezing VAE parameters
- Integrates with FHR Inception Time classifier
- Enables end-to-end fine-tuning or feature extraction modes
