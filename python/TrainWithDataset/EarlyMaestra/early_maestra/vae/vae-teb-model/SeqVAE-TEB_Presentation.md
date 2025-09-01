# SeqVAE-TEB: Advanced Sequential Variational Autoencoder with Transfer Entropy Bottleneck and β-TCVAE Integration

## A Comprehensive Technical Presentation


## 1. Introduction & Motivation

### 1.1 The Clinical Challenge

**Problem Statement:** Understanding the complex relationship between maternal uterine pressure (UP) and fetal heart rate (FHR) patterns during pregnancy monitoring.

**Key Questions:**
- How much information flows from maternal signals to fetal responses?
- Can we isolate the essential predictive factors while filtering noise?
- What are the independent physiological factors governing FHR patterns?

### 1.2 Technical Details

Our solution combines three cutting-edge machine learning approaches:

```
Transfer Entropy Bottleneck (TEB) + Sequential VAE + β-Total Correlation VAE
                ↓
        SeqVAE-TEB Model
```

**Core Components:**
- **Information-theoretic control** of source-to-target information flow
- **Disentangled representations** for interpretable physiological factors
- **Progressive upsampling decoder** for strict information bottleneck preservation
- **Three distinct loss computation modes** for different research objectives

---

## 2. Theoretical Foundation

### 2.1 Transfer Entropy Bottleneck (TEB)

The TEB principle aims to learn latent representations that are:
- **Minimally informative** about the source signal $\mathbf{x}$ (UP)
- **Maximally predictive** of the target signal $\mathbf{y}$ (FHR)

**Mathematical Formulation:**

$$\text{Transfer Entropy} \approx \text{KL}[q(\mathbf{z}|\mathbf{x},\mathbf{y}) \| p(\mathbf{z}|\mathbf{y})]$$

Where:
- $q(\mathbf{z}|\mathbf{x},\mathbf{y})$: Posterior distribution (conditional encoder)
- $p(\mathbf{z}|\mathbf{y})$: Prior distribution (target encoder)
- Lower KL divergence → Less information transfer from source

### 2.2 β-Total Correlation VAE (β-TCVAE)

**ELBO Decomposition:**
The standard VAE KL term can be decomposed into three interpretable components:

$$\mathbb{E}_{p(n)} [\text{KL}(q(\mathbf{z}|n) \| p(\mathbf{z}))] = \underbrace{I_q(\mathbf{z};n)}_{\text{Index-Code MI}} + \underbrace{\text{TC}(\mathbf{z})}_{\text{Total Correlation}} + \underbrace{\sum_j \text{KL}(q(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}$$

**Key Insight:** Penalizing **Total Correlation** drives disentanglement:
$$\text{TC}(\mathbf{z}) = \text{KL}(q(\mathbf{z}) \| \prod_j q(z_j))$$

### 2.3 Sequential Modeling

**Temporal Dependencies:**
- FHR patterns exhibit complex temporal correlations
- Causal relationships must be preserved (no future information leakage)
- Multi-scale patterns: beat-to-beat variations to long-term trends

---

## 3. Model Architecture Overview

### 3.1 High-Level Architecture

```
Source Signal (UP)     Target Signal (FHR)
      ↓                        ↓
[SourceEncoder]         [TargetEncoder]
      ↓                        ↓
    h_x                   μ_y, σ²_y, c_y
      ↓                        ↓
      └──── [ConditionalEncoder] ────┘
                   ↓
              μ_post, σ²_post
                   ↓
              [Reparameterization]
                   ↓
                   z
                   ↓
               [Decoder]
                   ↓
          Reconstructed Signals
```

## 4. Loss Function Framework

### 4.1 Three Computation Modes

Our implementation supports three distinct loss computation approaches:

#### 4.1.1 Standard TEB Mode

**Objective:** Minimize transfer entropy while maintaining reconstruction quality

$$\mathcal{L}_{TEB} = \mathcal{L}_{recon} + \beta \cdot \text{KL}[q(\mathbf{z}|\mathbf{x},\mathbf{y}) \| p(\mathbf{z}|\mathbf{y})]$$

**Characteristics:**
- Conditional learned prior $p(\mathbf{z}|\mathbf{y})$
- Direct transfer entropy measurement
- Single hyperparameter $\beta$

#### 4.1.2 Full β-TCVAE Mode (Complete Disentanglement)

**Objective:** Maximum disentanglement with standard normal prior

$$\mathcal{L}_{\beta\text{-TCVAE}} = \mathcal{L}_{recon} + \alpha \cdot I_q(\mathbf{z};n) + \beta \cdot \text{TC}(\mathbf{z}) + \gamma \cdot \sum_j \text{KL}(q(z_j) \| p(z_j))$$

**Characteristics:**
- Fixed standard normal prior $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$
- Unconditional generation capability
- Three-way hyperparameter control

#### 5.1.3 Hybrid β-TCVAE Mode (TEB + Disentanglement)

**Objective:** Balanced transfer entropy measurement with disentanglement

$$\mathcal{L}_{Hybrid} = \mathcal{L}_{recon} + \beta \cdot \text{KL}[q(\mathbf{z}|\mathbf{x},\mathbf{y}) \| p(\mathbf{z}|\mathbf{y})] + \gamma \cdot \text{TC}(\mathbf{z})$$

**Characteristics:**
- Preserves conditional prior for direct TEB measurement
- Adds TC penalty for disentanglement
- Two-parameter tuning

### 4.2 Reconstruction Loss Components

**Dual Reconstruction Tasks:**

1. **Auxiliary Feature Reconstruction (MSE):**
   $$\mathcal{L}_{MSE} = \frac{1}{T} \sum_{t=1}^{T} \|\widehat{\mathbf{y}}_t - \mathbf{y}_t\|_2^2$$

2. **Raw Signal Reconstruction (NLL):**
   $$\mathcal{L}_{NLL} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{2} \left( \log \sigma^{2,raw}_n + \frac{(r_n - \mu^{raw}_n)^2}{\sigma^{2,raw}_n} \right)$$

---

## 6. β-TCVAE Integration

### 6.1 Minibatch Weighted Sampling (MWS)

**Challenge:** Computing aggregated posterior $q(\mathbf{z}) = \frac{1}{N}\sum_{n=1}^N q(\mathbf{z}|\mathbf{x}_n, \mathbf{y}_n)$ requires full dataset access.

**Solution:** MWS provides tractable approximation:

$$\log q(\mathbf{z}_i) \approx \log \left[ \frac{1}{NM} \sum_{j=1}^M q(\mathbf{z}_i|\mathbf{x}_j, \mathbf{y}_j) \right]$$

**Algorithm Overview:**
```
FOR each sample z_i in minibatch:
    FOR each encoder q(z|x_j, y_j) in minibatch:
        compute density q(z_i | x_j, y_j)
    END FOR
    log_q_z[i] = logsumexp(all_densities) - log(N*M)
END FOR
```

### 6.2 Total Correlation Computation

**Mathematical Definition:**
$$\text{TC}(\mathbf{z}) = \mathbb{E}_{q(\mathbf{z})} \left[ \log \frac{q(\mathbf{z})}{\prod_j q(z_j)} \right]$$

**Implementation Strategy:**
```
Step 1: Compute joint density log q(z) using MWS
Step 2: Compute marginal densities log q(z_j) for each dimension
Step 3: TC = mean(log_q_z - sum(log_q_z_marginals))
```

### 6.3 Computational Complexity Analysis

**Standard TEB:** $O(BTD)$
- $B$: Batch size
- $T$: Sequence length  
- $D$: Latent dimension

**β-TCVAE with MWS:** $O(B^2T^2D)$
- Quadratic scaling due to pairwise density computations
- Memory bottleneck for large batches

---

---

## Mathematical Notation Reference

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}_t$ | Source signal (UP features) at time $t$ |
| $\mathbf{y}_t$ | Target signal (FHR features) at time $t$ |
| $\mathbf{z}_t$ | Latent variable at time $t$ |
| $q(\mathbf{z}\|\mathbf{x},\mathbf{y})$ | Posterior distribution (conditional encoder) |
| $p(\mathbf{z}\|\mathbf{y})$ | Prior distribution (target encoder) |
| $\text{TC}(\mathbf{z})$ | Total Correlation |
| $I_q(\mathbf{z};n)$ | Index-Code Mutual Information |
| $\mathcal{L}_{recon}$ | Reconstruction loss |
| $\beta, \alpha, \gamma$ | Loss weighting hyperparameters |
