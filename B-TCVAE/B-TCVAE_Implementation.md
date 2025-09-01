# $\beta$-TCVAE Implementation Details

## Overview

This document provides a comprehensive analysis of the $\beta$-TCVAE (Beta Total Correlation Variational Autoencoder) implementation, based on the code structure found in the `beta-tcvae` folder. The implementation demonstrates how the theoretical concepts described in the B-TCVAE paper are translated into working code.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Implementation Files](#core-implementation-files)
3. [VAE Model Architecture](#vae-model-architecture)
4. [ELBO Decomposition Implementation](#elbo-decomposition-implementation)
5. [Minibatch Weighted Sampling (MWS)](#minibatch-weighted-sampling-mws)
6. [Distribution Classes](#distribution-classes)
7. [Dataset Handling](#dataset-handling)
8. [MIG Metric Implementation](#mig-metric-implementation)
9. [Training Pipeline](#training-pipeline)
10. [Key Implementation Features](#key-implementation-features)
11. [Usage Examples](#usage-examples)

---

## Architecture Overview

The $\beta$-TCVAE implementation follows a modular structure with the following key components:

```
beta-tcvae/
├── vae_quant.py              # Main VAE model and training loop
├── elbo_decomposition.py     # ELBO decomposition and analysis
├── disentanglement_metrics.py # MIG metric computation
├── lib/
│   ├── dist.py              # Distribution classes (Normal, Laplace, Bernoulli)
│   ├── datasets.py          # Dataset loaders (dSprites, 3D Faces)
│   ├── utils.py             # Utility functions
│   └── functions.py         # Custom functions (e.g., STHeaviside)
└── metric_helpers/
    ├── mi_metric.py         # MIG computation helpers
    └── loader.py            # Model loading utilities
```

---

## Core Implementation Files

### 1. `vae_quant.py` - Main VAE Implementation

This is the central file containing:
- **VAE model class** with encoder/decoder architectures
- **ELBO computation** with $\beta$-TCVAE modifications
- **Training loop** with $\beta$ and $\lambda$ annealing
- **Visualization** and checkpointing

**Key Features:**
- Supports both MLP and Convolutional architectures
- Implements Minibatch Weighted Sampling (MWS)
- Handles $\beta$-VAE, $\beta$-TCVAE, and MSS variants
- Integrated visualization with Visdom

### 2. `elbo_decomposition.py` - ELBO Analysis

Implements the theoretical ELBO decomposition:
- **Index-Code MI** estimation
- **Total Correlation** computation
- **Dimension-wise KL** calculation
- **Entropy estimation** for analysis

### 3. `disentanglement_metrics.py` - MIG Evaluation

Computes the Mutual Information Gap (MIG) metric:
- **Dataset-specific implementations** for dSprites and 3D Faces
- **Conditional entropy estimation** for each factor
- **Marginal entropy computation**

---

## VAE Model Architecture

### Model Class Definition

```python
class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), 
                 q_dist=dist.Normal(), include_mutinfo=True, tcvae=False, 
                 conv=False, mss=False):
```

**Key Parameters:**
- `z_dim`: Latent space dimensionality $D$
- `tcvae`: Enable $\beta$-TCVAE mode (vs. standard $\beta$-VAE)
- `include_mutinfo`: Include mutual information term
- `conv`: Use convolutional vs. MLP architecture
- `mss`: Use Minibatch Stratified Sampling instead of MWS

### Architecture Options

#### 1. MLP Architecture

Implements the encoder $q_\phi(z|x)$ and decoder $p_\theta(x|z)$ using fully connected layers:

```python
class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        self.fc1 = nn.Linear(4096, 1200)  # 64x64 flattened input
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)
        self.act = nn.ReLU(inplace=True)

class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200), nn.Tanh(),
            nn.Linear(1200, 1200), nn.Tanh(),
            nn.Linear(1200, 1200), nn.Tanh(),
            nn.Linear(1200, 4096)  # 64x64 output
        )
```

**Mathematical Representation:**
- **Encoder**: $[\mu(x), \log\sigma(x)] = f_\phi(x)$ where $f_\phi$ is the MLP
- **Decoder**: $\hat{x} = g_\theta(z)$ where $g_\theta$ is the decoder MLP

#### 2. Convolutional Architecture

For image data, implements hierarchical feature extraction:

```python
class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)    # 64×32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)   # 32×16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)   # 16×8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)   # 8×4
        self.conv5 = nn.Conv2d(64, 512, 4)        # 4×1
        self.conv_z = nn.Conv2d(512, output_dim, 1)
```

**Usage Guidelines:**
- **MLP**: Used for dSprites (binary images)
- **Convolutional**: Used for 3D Faces (grayscale/color images)

---

## ELBO Decomposition Implementation

### Core ELBO Method

The main `elbo()` method in the VAE class implements the $\beta$-TCVAE objective based on the ELBO decomposition:

$$\mathbb{E}_{p(n)} [\text{KL}(q(z|n) \| p(z))] = I_q(z;n) + \text{TC}(z) + \sum_j \text{KL}(q(z_j) \| p(z_j))$$

```python
def elbo(self, x, dataset_size):
    batch_size = x.size(0)
    # Standard VAE terms
    logpx = self.x_dist.log_density(x, params=x_params).sum(1)
    logpz = self.prior_dist.log_density(zs, params=prior_params).sum(1)
    logqz_condx = self.q_dist.log_density(zs, params=z_params).sum(1)
    
    # Standard ELBO: log p(x,z) - log q(z|x)
    elbo = logpx + logpz - logqz_condx
    
    # β-TCVAE decomposition
    if self.tcvae:
        # Compute log q(z) using MWS
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )
        
        # Minibatch Weighted Sampling estimators
        logqz = (logsumexp(_logqz.sum(2), dim=1) - 
                math.log(batch_size * dataset_size))
        logqz_prodmarginals = (logsumexp(_logqz, dim=1) - 
                              math.log(batch_size * dataset_size)).sum(1)
        
        # β-TCVAE objective with explicit decomposition
        modified_elbo = logpx - \
            (logqz_condx - logqz) - \                    # Index-Code MI
            self.beta * (logqz - logqz_prodmarginals) - \ # Total Correlation
            (1 - self.lamb) * (logqz_prodmarginals - logpz) # Dimension-wise KL
```

### Term-by-Term Breakdown

The $\beta$-TCVAE objective decomposes the KL term into three interpretable components:

#### 1. **Index-Code Mutual Information**: $I_q(z;n) = \text{KL}(q(z,n) \| q(z)p(n))$

**Mathematical Formula:**
$$I_q(z;n) = \mathbb{E}_{q(z,n)} \left[ \log \frac{q(z|n)}{q(z)} \right]$$

**Implementation:** `(logqz_condx - logqz)`
- **Measures**: Information content of latent codes about specific data samples
- **Implementation**: Difference between conditional $\log q(z|x_n)$ and marginal $\log q(z)$ log-densities
- **Role**: Controls informativeness vs. compression trade-off

#### 2. **Total Correlation (TC)**: $\text{TC}(z) = \text{KL}(q(z) \| \prod_j q(z_j))$

**Mathematical Formula:**
$$\text{TC}(z) = \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{\prod_j q(z_j)} \right] = \mathbb{E}_{q(z)} \left[ \log q(z) - \sum_j \log q(z_j) \right]$$

**Implementation:** `self.beta * (logqz - logqz_prodmarginals)`
- **Measures**: Statistical dependence between latent dimensions
- **Implementation**: Difference between joint $\log q(z)$ and product of marginals $\log \prod_j q(z_j)$
- **Role**: **Primary disentanglement driver** - forces independence between latent factors

#### 3. **Dimension-wise KL**: $\sum_j \text{KL}(q(z_j) \| p(z_j))$

**Mathematical Formula:**
$$\sum_j \text{KL}(q(z_j) \| p(z_j)) = \sum_j \mathbb{E}_{q(z_j)} \left[ \log \frac{q(z_j)}{p(z_j)} \right]$$

**Implementation:** `(1 - self.lamb) * (logqz_prodmarginals - logpz)`
- **Measures**: Deviation of each latent dimension from its prior
- **Implementation**: Sum of KL divergences $\text{KL}(q(z_j) \| p(z_j))$ for each latent dimension
- **Role**: Complexity penalty following minimum description length principle

---

## Minibatch Weighted Sampling (MWS)

### The Challenge

Computing the aggregated posterior requires evaluating:
$$q(z) = \mathbb{E}_{p(n)}[q(z|n)] = \frac{1}{N}\sum_{n=1}^N q(z|x_n)$$

This requires access to the entire dataset, which is intractable for large datasets with computational complexity $O(N^2)$.

### MWS Solution

The implementation uses an importance sampling approach to approximate $q(z)$ using only a minibatch:

```python
# Compute log densities for all pairs in minibatch
_logqz = self.q_dist.log_density(
    zs.view(batch_size, 1, self.z_dim),                    # [B, 1, D]
    z_params.view(1, batch_size, self.z_dim, nparams)      # [1, B, D, P]
)
# Result: [B, B, D] - density of each z under each encoder

# MWS estimator with logsumexp trick
logqz = logsumexp(_logqz.sum(2), dim=1) - math.log(batch_size * dataset_size)
logqz_prodmarginals = logsumexp(_logqz, dim=1).sum(1) - math.log(batch_size * dataset_size)
```

### Mathematical Foundation

**MWS Estimator for Joint Distribution:**
$$\log q(z) \approx \log\left(\frac{1}{NM}\right) + \text{logsumexp}_j\left(\log q(z|x_j)\right)$$

**MWS Estimator for Product of Marginals:**
$$\log \prod_d q(z_d) \approx \sum_d \left[ \log\left(\frac{1}{NM}\right) + \text{logsumexp}_j\left(\log q(z_d|x_j)\right) \right]$$

Where:
- $N$: Total dataset size
- $M$: Minibatch size  
- $\frac{1}{NM}$: Normalization factor accounting for dataset size and minibatch approximation

**Key Properties:**
- **Biased but consistent** estimator: $\mathbb{E}[\hat{q}(z)] \leq q(z)$ (Jensen's inequality)
- **$O(M^2 D)$** computational complexity per minibatch
- **No additional hyperparameters** required
- **Numerically stable** via logsumexp trick

---

## Distribution Classes

### Normal Distribution

The `Normal` class in `lib/dist.py` implements a reparameterizable Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$:

**Reparameterization Trick:**
$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

```python
class Normal(nn.Module):
    def sample(self, size=None, params=None):
        mu, logsigma = self._check_inputs(size, params)
        std_z = Variable(torch.randn(mu.size()).type_as(mu.data))
        sample = std_z * torch.exp(logsigma) + mu  # z = μ + σε
        return sample
    
    def log_density(self, sample, params=None):
        mu, logsigma = self._check_inputs(None, params)
        c = self.normalization.type_as(sample.data)
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logsigma + c)
```

**Log-Density Formula:**
$$\log \mathcal{N}(z|\mu, \sigma^2) = -\frac{1}{2}\left[ \frac{(z-\mu)^2}{\sigma^2} + \log(2\pi\sigma^2) \right]$$

**Key Features:**
- **Reparameterization trick**: Enables gradient flow through sampling
- **Log-space operations**: Numerical stability for variance parameters $\log \sigma$
- **Analytical KL**: Closed-form KL divergence computation

### Other Distributions

#### Laplace Distribution

**Probability Density:**
$$p(z|\mu, b) = \frac{1}{2b} \exp\left(-\frac{|z-\mu|}{b}\right)$$

- Alternative prior/posterior choice with heavier tails than Normal
- Reparameterizable via inverse CDF sampling

#### Bernoulli Distribution  

**Probability Mass:**
$$p(x|\theta) = \theta^x (1-\theta)^{1-x}$$

- Used for binary output data (dSprites dataset)
- Implements straight-through gradient estimation
- Uses Gumbel-softmax for reparameterization

---

## Dataset Handling

### dSprites Dataset

```python
class Shapes(object):
    def __init__(self, dataset_zip=None):
        loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.dataset_zip = np.load(loc, encoding='latin1')
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()
    
    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        return x
```

**Dataset Properties:**
- **737,280 images** ($3 \times 6 \times 40 \times 32 \times 32$)
- **Binary $64 \times 64$ images**
- **5 factors of variation**: 
  - $v_1$: shape (3 values: square, ellipse, heart)
  - $v_2$: scale (6 values: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
  - $v_3$: orientation (40 values: $0°, 9°, 18°, \ldots, 351°$)
  - $v_4$: position X (32 values)
  - $v_5$: position Y (32 values)

### 3D Faces Dataset

```python
class Faces(Dataset):
    LOC = 'data/basel_face_renders.pth'
    
    def __init__(self):
        return super(Faces, self).__init__(self.LOC)
```

**Dataset Properties:**
- **127,050 images** ($50 \times 21 \times 11 \times 11$)
- **Grayscale $64 \times 64$ images**
- **3 main factors of variation**:
  - $v_1$: azimuth (21 values: face rotation left-right)
  - $v_2$: elevation (11 values: face rotation up-down)  
  - $v_3$: lighting (11 values: lighting direction)
- **Additional factor**: identity (50 different people)

---

## MIG Metric Implementation

### Overview

The Mutual Information Gap (MIG) metric measures disentanglement quality by computing the gap between the highest and second-highest mutual information values for each factor:

$$\text{MIG} = \frac{1}{K} \sum_{k=1}^K \frac{1}{H(v_k)} \left( I_n(z_{j^{(k)}}; v_k) - \max_{j \neq j^{(k)}} I_n(z_j; v_k) \right)$$

Where:
- $K$: Number of ground truth factors
- $v_k$: Ground truth factor $k$
- $z_j$: Latent dimension $j$
- $j^{(k)} = \arg\max_j I_n(z_j; v_k)$: Most informative latent for factor $k$
- $H(v_k)$: Entropy of factor $k$
- $I_n(z_j; v_k)$: Empirical mutual information

### Implementation Structure

```python
def mutual_info_metric_shapes(vae, shapes_dataset):
    # 1. Encode all data to get latent parameters
    qz_params = torch.Tensor(N, K, nparams)
    for xs in dataset_loader:
        qz_params[n:n + batch_size] = vae.encoder.forward(xs)
    
    # 2. Reshape for factor structure: [3, 6, 40, 32, 32, K, nparams]
    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)
    qz_samples = vae.q_dist.sample(params=qz_params)
    
    # 3. Compute marginal entropies H(Z_j)
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams)
    )
    
    # 4. Compute conditional entropies H(Z_j|V_k) for each factor
    cond_entropies = torch.zeros(4, K)  # 4 factors (excluding shape)
    
    # Scale factor: H(Z_j|V_scale)
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        cond_entropies[0] += estimate_entropies(...) / 6
    
    # Similar for orientation, posX, posY...
    
    # 5. Compute MIG
    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
```

### MIG Computation

The core MIG computation uses the mutual information formula:
$$I(Z_j; V_k) = H(Z_j) - H(Z_j|V_k)$$

```python
def compute_metric_shapes(marginal_entropies, cond_entropies):
    factor_entropies = [6, 40, 32, 32]  # log(num_values) for each factor
    
    # Mutual Information = H(Z_j) - H(Z_j|V_k)
    mutual_infos = marginal_entropies[None] - cond_entropies
    
    # Sort to get highest and second-highest MI for each factor
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    
    # Normalize by factor entropy: I(Z_j; V_k) / H(V_k)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    
    # MIG = mean gap across factors
    metric = torch.mean(mi_normed[:, 0] - mi_normed[:, 1])
    return metric
```

**Key Steps:**
1. **Marginal entropy**: $H(Z_j)$ for each latent dimension $j$
2. **Conditional entropy**: $H(Z_j|V_k)$ for each factor $k$  
3. **Mutual information**: $I(Z_j; V_k) = H(Z_j) - H(Z_j|V_k)$
4. **Gap computation**: Difference between highest and second-highest MI
5. **Normalization**: Divide by factor entropy $H(V_k) = \log(|\text{values of } V_k|)$ for comparability

---

## Training Pipeline

### Command Line Interface

```bash
# Train β-TCVAE on dSprites
python vae_quant.py --dataset shapes --beta 6 --tcvae

# Train with convolutional architecture on 3D faces
python vae_quant.py --dataset faces --beta 4 --tcvae --conv

# Evaluate MIG score
python disentanglement_metrics.py --checkpt model_checkpoint.pth
```

### Training Loop Structure

The training optimizes the $\beta$-TCVAE objective:
$$\mathcal{L}_{\beta\text{-TC}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \alpha I_q(z;n) - \beta \cdot \text{TC}(z) - \gamma \sum_j \text{KL}(q(z_j) \| p(z_j))$$

```python
def main():
    # 1. Setup
    train_loader = setup_data_loaders(args)
    vae = VAE(z_dim=args.latent_dim, tcvae=args.tcvae, ...)
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # 2. Training loop
    for epoch in range(args.num_epochs):
        for i, x in enumerate(train_loader):
            # Anneal hyperparameters β(t) and λ(t)
            anneal_kl(args, vae, iteration)
            
            # Forward pass: compute -ELBO
            obj, elbo = vae.elbo(x, dataset_size)
            
            # Backward pass: minimize -ELBO
            obj.mean().mul(-1).backward()
            optimizer.step()
            
            # Logging and visualization
            if iteration % args.log_freq == 0:
                display_samples(vae, x, vis)
                save_checkpoint(...)
    
    # 3. Final evaluation
    elbo_decomposition(vae, dataset_loader)
```

### Hyperparameter Annealing

The implementation uses careful annealing schedules for stable training:

```python
def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500
    
    # λ annealing: 1 → 0 (reduce mutual info penalty)
    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)
    
    # β annealing: 0 → target (gradually increase TC penalty)
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)
```

**Annealing Strategy:**
- **$\beta$ annealing**: $\beta(t): 0 \to \beta_{\text{target}}$ - Gradually increase TC penalty to avoid posterior collapse
- **$\lambda$ annealing**: $\lambda(t): 1 \to 0$ - Reduce mutual information penalty over time
- **Dataset-specific**: Different warmup periods for different datasets

---

## Key Implementation Features

### 1. **Modular Design**
- **Pluggable distributions**: Easy to swap Normal/Laplace/Flow priors
- **Architecture flexibility**: MLP vs. Convolutional encoders/decoders
- **Sampling methods**: MWS vs. MSS implementations

### 2. **Numerical Stability**
- **Logsumexp trick**: Prevents overflow in importance sampling
  $$\text{logsumexp}(x_1, \ldots, x_n) = \max_i x_i + \log\sum_i \exp(x_i - \max_i x_i)$$
- **Log-space operations**: Variance parameters stored as $\log \sigma$
- **Gradient clipping**: Prevents exploding gradients

### 3. **Efficient Implementation**
- **Vectorized operations**: Batch computation of log-densities
- **GPU acceleration**: CUDA support throughout
- **Memory management**: Chunked processing for large datasets

### 4. **Comprehensive Evaluation**
- **ELBO decomposition**: Full analysis of all terms
- **MIG computation**: Automatic disentanglement evaluation
- **Visualization**: Latent walks and reconstructions

### 5. **Reproducibility**
- **Checkpoint saving**: Model state and hyperparameters
- **Deterministic evaluation**: Fixed seeds for consistent results
- **Configuration management**: Command-line argument parsing

---

## Usage Examples

### Basic Training

```python
# Initialize model
vae = VAE(z_dim=10, tcvae=True, beta=6, conv=False)

# Load data
train_loader = setup_data_loaders(args)

# Train: minimize -ELBO
for x in train_loader:
    obj, elbo = vae.elbo(x, len(train_loader.dataset))
    loss = -obj.mean()
    loss.backward()
    optimizer.step()
```

### Evaluation

```python
# Load trained model
checkpoint = torch.load('model.pth')
vae.load_state_dict(checkpoint['state_dict'])

# Compute MIG
metric, marginals, conditionals = mutual_info_metric_shapes(vae, dataset)
print(f'MIG Score: {metric:.3f}')

# ELBO decomposition
logpx, dependence, information, dimwise_kl, _, _, _ = elbo_decomposition(vae, loader)
print(f'Total Correlation: {dependence:.3f}')
```

### Custom Architecture

```python
class CustomEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Custom architecture here
        
    def forward(self, x):
        # Custom forward pass
        return z_params  # [batch_size, z_dim, 2] for (μ, log σ)

# Use custom architecture
vae = VAE(z_dim=10, tcvae=True)
vae.encoder = CustomEncoder(10 * 2)  # 2 params per dimension (μ, log σ)
```

---

## Comparison with Standard $\beta$-VAE

### Key Differences

| Aspect | $\beta$-VAE | $\beta$-TCVAE |
|--------|-------|---------|
| **Objective** | $\log p(x\|z) - \beta \cdot \text{KL}(q(z\|x) \\\| p(z))$ | $\log p(x\|z) - I_q(z;n) - \beta \cdot \text{TC}(z) - \gamma \sum_j \text{KL}(q(z_j) \\\| p(z_j))$ |
| **TC Penalty** | Implicit in KL term | **Explicit $\beta \cdot \text{TC}(z)$ term** |
| **Theoretical Understanding** | Limited | **Complete ELBO decomposition** |
| **Hyperparameter Control** | Only $\beta$ | Independent control: $\alpha$, $\beta$, $\gamma$ |
| **Disentanglement** | Good | **Better and more consistent** |

### Implementation Differences

The key implementation difference is in lines 247-257 of `vae_quant.py`:

**$\beta$-VAE objective:**
$$\mathcal{L}_{\beta} = \log p(x|z) - \beta \cdot \text{KL}(q(z|x) \| p(z))$$

**$\beta$-TCVAE objective:**
$$\mathcal{L}_{\beta\text{-TC}} = \log p(x|z) - I_q(z;n) - \beta \cdot \text{TC}(z) - (1-\lambda) \sum_j \text{KL}(q(z_j) \| p(z_j))$$

```python
# β-VAE mode
if not self.tcvae:
    modified_elbo = logpx - self.beta * (logqz_condx - logpz)

# β-TCVAE mode  
else:
    modified_elbo = logpx - \
        (logqz_condx - logqz) - \                        # I_q(z;n)
        self.beta * (logqz - logqz_prodmarginals) - \    # β·TC(z)
        (1 - self.lamb) * (logqz_prodmarginals - logpz)  # γ·∑KL(q(z_j)||p(z_j))
```

---

## Conclusion

The $\beta$-TCVAE implementation provides a clean, modular, and theoretically grounded approach to learning disentangled representations. Key strengths include:

1. **Theoretical Foundation**: Direct implementation of ELBO decomposition
2. **Practical Efficiency**: MWS enables scalable TC estimation with $O(M^2)$ complexity
3. **Comprehensive Evaluation**: Built-in MIG metric computation
4. **Flexibility**: Support for multiple architectures and distributions
5. **Reproducibility**: Robust training and evaluation pipelines

The implementation serves as both a research tool and a reference implementation for the $\beta$-TCVAE method, demonstrating how theoretical insights can be translated into practical deep learning code.