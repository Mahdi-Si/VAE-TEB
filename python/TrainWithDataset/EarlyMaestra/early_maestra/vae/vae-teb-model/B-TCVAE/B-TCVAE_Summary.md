# $\beta$-TCVAE: Isolating Sources of Disentanglement in Variational Autoencoders

## A Comprehensive Tutorial on Total Correlation Penalty for Disentangled Representations

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background and Motivation](#background-and-motivation)
3. [Mathematical Framework](#mathematical-framework)
4. [The $\beta$-TCVAE Method](#the-beta-tcvae-method)
5. [Mutual Information Gap Metric](#mutual-information-gap-metric)
6. [Multi-Channel Signal Considerations](#multi-channel-signal-considerations)
7. [Experimental Results](#experimental-results)
8. [Practical Implementation](#practical-implementation)
9. [Conclusion](#conclusion)

---

## Introduction

The $\beta$-TCVAE (Beta Total Correlation Variational Autoencoder) represents a significant advancement in learning disentangled representations without supervision. This work provides both theoretical insights into why $\beta$-VAE works and introduces a more effective training methodology.

### Key Contributions

1. **ELBO Decomposition**: A novel decomposition of the Evidence Lower BOund (ELBO) that explains the success of $\beta$-VAE
2. **$\beta$-TCVAE Algorithm**: A plug-in replacement for $\beta$-VAE with improved disentanglement properties
3. **Minibatch Weighted Sampling**: A training method that enables arbitrary weighting of ELBO terms without hyperparameters
4. **MIG Metric**: A classifier-free, information-theoretic disentanglement metric

---

## Background and Motivation

### The Problem of Disentangled Representations

**Disentangled representations** are latent encodings where each dimension captures a distinct, interpretable factor of variation in the data. Such representations are valuable because they:

- Contain semantically meaningful information
- Are more generalizable and robust
- Facilitate downstream tasks
- Enable controllable generation

### Variational Autoencoders (VAEs)

The standard VAE optimizes the Evidence Lower BOund (ELBO):

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{p(x)} \left[ \mathbb{E}_{q(z|x)} [\log p(x|z)] - \text{KL}(q(z|x) \| p(z)) \right]$$

Where:
- $x$: observed data
- $z$: latent variables
- $q(z|x)$: encoder (inference network)
- $p(x|z)$: decoder (generative network)
- $p(z)$: prior distribution (typically $\mathcal{N}(0, I)$)

### $\beta$-VAE: The Predecessor

The $\beta$-VAE modifies the ELBO with a penalty parameter $\beta > 1$:

$$\mathcal{L}_{\beta} = \frac{1}{N} \sum_{n=1}^N \left( \mathbb{E}_{q(z|x_n)} [\log p(x_n|z)] - \beta \cdot \text{KL}(q(z|x_n) \| p(z)) \right)$$

While effective, $\beta$-VAE lacks theoretical justification for why penalizing the KL term leads to disentanglement.

---

## Mathematical Framework

### ELBO Total Correlation Decomposition

The core theoretical contribution is decomposing the KL term in the ELBO. Let's define:
- $n$: data index, uniformly distributed over $\{1, 2, ..., N\}$
- $q(z|n) = q(z|x_n)$: encoder for sample $n$
- $q(z,n) = q(z|n)p(n) = q(z|n) \cdot \frac{1}{N}$: joint distribution
- $q(z) = \sum_{n=1}^N q(z|n)p(n)$: **aggregated posterior**

#### The Decomposition

$$\mathbb{E}_{p(n)} [\text{KL}(q(z|n) \| p(z))] = \underbrace{\text{KL}(q(z,n) \| q(z)p(n))}_{\circled{1} \text{ Index-Code MI}} + \underbrace{\text{KL}(q(z) \| \prod_j q(z_j))}_{\circled{2} \text{ Total Correlation}} + \underbrace{\sum_j \text{KL}(q(z_j) \| p(z_j))}_{\circled{3} \text{ Dimension-wise KL}}$$

#### Term Analysis

**① Index-Code Mutual Information (MI)**
$$I_q(z;n) = \text{KL}(q(z,n) \| q(z)p(n))$$
- Measures mutual information between latent variables and data indices
- Higher values indicate more informative latent codes
- Previously identified in literature but role in disentanglement was unclear

**② Total Correlation (TC)**
$$\text{TC}(z) = \text{KL}(q(z) \| \prod_j q(z_j)) = \text{KL}\left(\prod_j q(z_j|z_{\neg j}) \| \prod_j q(z_j)\right)$$
- Measures statistical dependence between latent dimensions
- **Key insight**: This term drives disentanglement in $\beta$-VAE
- Lower TC $\rightarrow$ more independent latent factors $\rightarrow$ better disentanglement

**③ Dimension-wise KL**
$$\sum_j \text{KL}(q(z_j) \| p(z_j))$$
- Prevents individual dimensions from deviating too far from prior
- Acts as complexity penalty following minimum description length principle

### Proof of Decomposition

Starting from the KL term:

$$\begin{aligned}
&\mathbb{E}_{p(n)} [\text{KL}(q(z|n) \| p(z))] \\
&= \mathbb{E}_{p(n)} \left[ \mathbb{E}_{q(z|n)} [\log q(z|n) - \log p(z)] \right] \\
&= \mathbb{E}_{q(z,n)} [\log q(z|n) - \log p(z) + \log q(z) - \log q(z) + \log \prod_j q(z_j) - \log \prod_j q(z_j)] \\
&= \mathbb{E}_{q(z,n)} \left[ \log \frac{q(z|n)}{q(z)} \right] + \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{\prod_j q(z_j)} \right] + \mathbb{E}_{q(z)} \left[ \sum_j \log \frac{q(z_j)}{p(z_j)} \right] \\
&= \text{KL}(q(z,n) \| q(z)p(n)) + \text{KL}(q(z) \| \prod_j q(z_j)) + \sum_j \text{KL}(q(z_j) \| p(z_j))
\end{aligned}$$

---

## The $\beta$-TCVAE Method

### Understanding the Weighted Objective

The $\beta$-TCVAE method builds directly on the ELBO decomposition to create a more principled approach to learning disentangled representations. Let's break down what we're trying to achieve and how each component works.

#### The Complete $\beta$-TCVAE Objective

$$\mathcal{L}_{\beta-\text{TC}} = \mathbb{E}_{q(z|n)p(n)}[\log p(x|z)] - \alpha I_q(z;n) - \beta \cdot \text{KL}(q(z) \| \prod_j q(z_j)) - \gamma \sum_j \text{KL}(q(z_j) \| p(z_j))$$

This objective allows us to **independently control** each component of the ELBO decomposition. Let's understand what each term does:

#### Term-by-Term Analysis

**① Reconstruction Term: $\mathbb{E}_{q(z|n)p(n)}[\log p(x|z)]$**
- **What it measures**: How well the decoder reconstructs input data from latent codes
- **What we're calculating**: Expected log-likelihood of data given latent representations
- **Role in disentanglement**: Ensures latent codes contain enough information to reconstruct the original data
- **Intuition**: "Can we get back our original data from the latent representation?"

**② Index-Code Mutual Information: $\alpha I_q(z;n)$**
- **What it measures**: Mutual information between latent variables $z$ and data indices $n$
- **Mathematical form**: $I_q(z;n) = \text{KL}(q(z,n) \| q(z)p(n))$
- **What we're calculating**: How much information the latent codes contain about which specific data sample they came from
- **Role in disentanglement**: Controls information content of latent variables
- **Intuition**: "How much do the latent codes tell us about which specific training example we're looking at?"

**Physical Interpretation**: 
- **High $I_q(z;n)$**: Latent codes are very informative about individual data points
- **Low $I_q(z;n)$**: Latent codes capture general patterns rather than specific details
- **$\alpha$ controls the trade-off**: Higher $\alpha$ → more compression, lower $\alpha$ → more detailed representation

**③ Total Correlation (TC): $\beta \cdot \text{KL}(q(z) \| \prod_j q(z_j))$**
- **What it measures**: Statistical dependence between different dimensions of the latent space
- **Mathematical form**: $\text{TC}(z) = \text{KL}(q(z) \| \prod_j q(z_j))$
- **What we're calculating**: How much the joint distribution $q(z)$ deviates from the product of marginals $\prod_j q(z_j)$
- **Role in disentanglement**: **This is the key term!** Forces latent dimensions to be statistically independent
- **Intuition**: "How much do different latent dimensions depend on each other?"

**Why TC is Crucial for Disentanglement**:
```
If z = [z₁, z₂, z₃] represents [color, shape, size]:
- Good disentanglement: z₁, z₂, z₃ are independent
  → TC ≈ 0 → q(z₁, z₂, z₃) ≈ q(z₁)q(z₂)q(z₃)
- Bad disentanglement: z₁ and z₂ are correlated  
  → TC > 0 → changing color affects shape representation
```

**④ Dimension-wise KL: $\gamma \sum_j \text{KL}(q(z_j) \| p(z_j))$**
- **What it measures**: How much each latent dimension deviates from its prior
- **What we're calculating**: Sum of KL divergences between each marginal posterior $q(z_j)$ and prior $p(z_j)$
- **Role in disentanglement**: Complexity penalty preventing latent dimensions from becoming too complex
- **Intuition**: "How much does each latent dimension differ from what we expected (the prior)?"

#### Parameter Selection Strategy

**Default Choice**: $\alpha = \gamma = 1$, tune only $\beta$

**Why this works**:
1. **Maintains proper ELBO**: With $\alpha = \gamma = 1$, the objective remains a valid lower bound on $\log p(x)$
2. **Focuses on the key term**: Empirical evidence shows TC penalty ($\beta$) is most important for disentanglement
3. **Reduces hyperparameter space**: Only need to tune one parameter instead of three
4. **Theoretical justification**: The ELBO decomposition reveals TC as the primary driver of disentanglement

### The Core Challenge: Computing $q(z)$

#### Why Computing $q(z)$ is Difficult

The **aggregated posterior** $q(z) = \mathbb{E}_{p(n)}[q(z|n)] = \frac{1}{N}\sum_{n=1}^N q(z|n)$ requires:
- Access to **entire dataset** during training
- Computing densities for **all training samples**
- **Intractable** for large datasets

**Conceptual Problem**: 
```
To compute TC = KL(q(z) || ∏ⱼq(zⱼ)), we need:
1. q(z) = mixture of all encoder distributions
2. q(zⱼ) = marginal of the mixture
3. Both require full dataset access!
```

#### Minibatch Weighted Sampling (MWS): The Elegant Solution

The key insight is to use **importance sampling** with clever reweighting.

**The MWS Estimator**:
$$\mathbb{E}_{q(z)}[\log q(z)] \approx \frac{1}{M} \sum_{i=1}^M \log \left[ \frac{1}{NM} \sum_{j=1}^M q(z(n_i)|n_j) \right]$$

**Step-by-Step Breakdown**:

1. **Sample minibatch**: $\{n_1, n_2, ..., n_M\}$ from training data
2. **Encode each sample**: $z(n_i) \sim q(z|n_i)$ for each $n_i$
3. **For each encoded $z(n_i)$**: 
   - Compute its density under **all** encoders in the minibatch
   - $q(z(n_i)|n_j)$ = density of $z(n_i)$ under encoder $q(z|n_j)$
4. **Weight and average**: Use the $\frac{1}{NM}$ factor to approximate the full dataset mixture

**Intuitive Explanation**:
```python
# What we want (intractable):
q_z = (1/N) * sum([q(z|n_i) for all n_i in dataset])

# What MWS does (tractable):
# For each z(n_i) from minibatch:
density_estimates = [q(z(n_i)|n_j) for n_j in minibatch]
log_q_z_approx = log((1/NM) * sum(density_estimates))
```

**Why the $\frac{1}{NM}$ Factor?**
- $\frac{1}{N}$: Accounts for the fact that $q(z) = \frac{1}{N}\sum_{n=1}^N q(z|n)$
- $\frac{1}{M}$: Monte Carlo approximation using only $M$ samples instead of full dataset

#### Theoretical Properties of MWS

**① Biased Estimator**:
- Due to Jensen's inequality: $\mathbb{E}[\log f(X)] \leq \log \mathbb{E}[f(X)]$
- **Lower bound**: $\mathbb{E}[\text{MWS estimate}] \leq \mathbb{E}_{q(z)}[\log q(z)]$
- **Practical impact**: Slightly underestimates $\log q(z)$, leading to conservative TC estimates

**② Computational Complexity**: 
- **$O(M^2 \cdot D)$** per minibatch where $D$ is latent dimension
- For each of $M$ samples, compute density under $M$ encoders
- **Scalable**: Much better than $O(N^2)$ for full dataset

**③ No Additional Hyperparameters**:
- Unlike FactorVAE which needs auxiliary discriminator networks
- Unlike methods requiring inner optimization loops
- **Simple implementation**: Drop-in replacement for standard VAE training

#### Practical Implementation Details

**Computing Gaussian Densities**:
For Gaussian encoders $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$:

$$q(z(n_i)|n_j) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(z(n_i) - \mu_j)^2}{2\sigma_j^2}\right)$$

**Numerical Stability**:
- Work in **log space**: $\log q(z(n_i)|n_j) = -\frac{1}{2}\log(2\pi\sigma_j^2) - \frac{(z(n_i) - \mu_j)^2}{2\sigma_j^2}$
- Use **logsumexp trick**: $\log\sum_j \exp(a_j) = \max_j(a_j) + \log\sum_j \exp(a_j - \max_j(a_j))$

**Memory Considerations**:
- Store $\mu$ and $\log\sigma$ for all samples in minibatch
- Compute pairwise densities efficiently using broadcasting
- Typical minibatch size: 64-128 (balance between approximation quality and memory)

#### Alternative: Minibatch Stratified Sampling (MSS)

For completeness, the paper also proposes MSS as an **unbiased estimator**:

$$f(z,n^*,\hat{B}_M) = \frac{1}{N}q(z|n^*) + \frac{1}{M}\sum_{m=1}^{M-1} q(z|n_m) + \frac{N-M}{NM} q(z|n_M)$$

**Key Properties**:
- **Unbiased**: $\mathbb{E}[f(z,n^*,\hat{B}_M)] = q(z)$ exactly
- **More complex**: Requires careful sampling without replacement  
- **Empirically similar**: Performance comparable to MWS in practice
- **Less popular**: MWS is simpler and more commonly used

**When to Use Each**:
- **MWS**: Standard choice, simple implementation, good empirical results
- **MSS**: When theoretical guarantees about unbiasedness are crucial

---

## Mutual Information Gap Metric

### Motivation for New Metric

Previous disentanglement metrics suffer from:
- **Hyperparameter sensitivity**: Results vary with classifier parameters
- **Lack of axis-alignment detection**: Don't enforce one-to-one factor-dimension correspondence
- **Dataset dependence**: Require specific data distributions

### Understanding Ground Truth Factors

Before diving into the MIG metric, we need to understand what **ground truth factors** are and why they're crucial for evaluating disentanglement.

#### What Are Ground Truth Factors?

**Ground truth factors** $\{v_k\}_{k=1}^K$ are the **true underlying sources of variation** that generate the observed data. These represent the **semantically meaningful attributes** that we ideally want our latent variables to capture independently.

**Mathematical Representation**:
- Each factor $v_k$ is a **discrete or continuous variable** that controls a specific aspect of data generation
- The **generative process** can be written as: $x \sim p(x|v_1, v_2, ..., v_K)$
- **Factor independence**: Ideally, factors are **statistically independent**: $p(v_1, ..., v_K) = \prod_{k=1}^K p(v_k)$

#### Ground Truth Factors by Dataset

Let's examine specific examples from the paper:

**① dSprites Dataset**
```
Image: 64×64 binary images of 2D shapes
Ground truth factors:
- v₁: Shape type (3 values: square, ellipse, heart)
- v₂: Scale (6 values: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
- v₃: Rotation (40 values: 0°, 9°, 18°, ..., 351°)
- v₄: Position X (32 values: equally spaced across width)
- v₅: Position Y (32 values: equally spaced across height)

Total combinations: 3 × 6 × 40 × 32 × 32 = 737,280 images
```

**② 3D Faces Dataset**
```
Image: 64×64 color images of synthetic faces  
Ground truth factors:
- v₁: Azimuth (21 values: face rotation left-right)
- v₂: Elevation (11 values: face rotation up-down)  
- v₃: Lighting (11 values: lighting direction)

Additional factors (treated as noise):
- Identity (50 different people)
Total: 21 × 11 × 11 × 50 = 127,050 images
```

**③ CelebA Dataset (Real Data)**
```
Image: 64×64 color images of celebrity faces
Ground truth factors (labeled attributes):
- v₁: Male/Female (binary)
- v₂: Smiling/Not smiling (binary)  
- v₃: Wearing eyeglasses (binary)
- v₄: Bald/Not bald (binary)
- v₅: Mustache/No mustache (binary)
- ... (40 binary attributes total)

Challenge: Real data, factors may be correlated
Example: Male faces more likely to have mustaches
```

**④ 3D Chairs Dataset**
```
Image: 64×64 color images of chairs
Ground truth factors:
- v₁: Azimuth angle (continuous: 0° to 360°)
- v₂: Chair type/style (discrete: different chair models)

Discovered factors (not labeled):
- Chair size, leg style, backrest type, material, swivel capability
```

#### Why Ground Truth Factors Matter

**1. Disentanglement Definition**:
Perfect disentanglement means each latent variable $z_j$ corresponds to exactly one ground truth factor $v_k$:
```
Ideal mapping:
z₁ ↔ v₁ (shape)
z₂ ↔ v₂ (scale) 
z₃ ↔ v₃ (rotation)
z₄ ↔ v₄ (position X)
z₅ ↔ v₅ (position Y)
```

**2. Evaluation Necessity**:
Without ground truth, we cannot:
- Measure disentanglement quality objectively
- Compare different methods quantitatively
- Understand what semantic concepts the model learned

**3. Challenges with Real Data**:
- **Factor correlation**: Real factors may not be independent (e.g., age ↔ hair color)
- **Incomplete labeling**: Not all relevant factors may be known
- **Subjective factors**: What constitutes a "factor" can be debatable

#### Factor Types and Properties

**Discrete vs. Continuous Factors**:
```python
# Discrete factors (categorical)
shape = {0: 'square', 1: 'ellipse', 2: 'heart'}
gender = {0: 'female', 1: 'male'}

# Continuous factors (numerical)  
rotation = [0°, 9°, 18°, ..., 351°]  # quantized continuous
scale = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # quantized continuous
```

**Factor Hierarchy**:
Some factors may be **hierarchical** or **compositional**:
- **Primary factors**: Shape, color, size
- **Secondary factors**: Texture, material properties
- **Interaction factors**: How lighting affects appearance

### MIG Definition and Formulation

Now we can understand the MIG metric in context. For ground truth factors $\{v_k\}_{k=1}^K$ and learned latent variables $\{z_j\}_{j=1}^J$:

$$\text{MIG} = \frac{1}{K} \sum_{k=1}^K \frac{1}{H(v_k)} \left( I_n(z_{j^{(k)}}; v_k) - \max_{j \neq j^{(k)}} I_n(z_j; v_k) \right)$$

#### Component Analysis

**Core Components**:
- $j^{(k)} = \arg\max_j I_n(z_j; v_k)$: **Most informative latent variable** for factor $k$
- $H(v_k) = \mathbb{E}_{p(v_k)}[-\log p(v_k)]$: **Entropy of ground truth factor** $k$
- $I_n(z_j; v_k)$: **Empirical mutual information** between latent $j$ and factor $k$

**The Gap Term**: $I_n(z_{j^{(k)}}; v_k) - \max_{j \neq j^{(k)}} I_n(z_j; v_k)$

This measures the **difference** between:
1. **Highest MI**: Best latent variable for factor $k$
2. **Second highest MI**: Second-best latent variable for factor $k$

#### What MIG Measures

**Perfect Disentanglement** (MIG = 1):
```
For each factor vₖ:
- One latent variable has high MI: I(z₁; v₁) = H(v₁) 
- All others have zero MI: I(zⱼ; v₁) = 0 for j ≠ 1
- Gap = H(v₁) - 0 = H(v₁)
- Normalized gap = H(v₁)/H(v₁) = 1
```

**Complete Entanglement** (MIG = 0):
```
For each factor vₖ:  
- Multiple latent variables have similar MI
- Gap ≈ 0 (top two MIs are similar)
- No clear axis-alignment
```

**Partial Disentanglement** (0 < MIG < 1):
```
For each factor vₖ:
- One latent is clearly best, but others also informative
- Gap > 0 but less than maximum possible
- Some axis-alignment but not perfect
```

#### MIG Advantages Over Previous Metrics

**1. Axis-Alignment Detection**:
- **Previous metrics**: Could miss that multiple latents encode same factor
- **MIG**: Explicitly penalizes multiple informative latents via gap term

**2. Normalization**:
- **Previous metrics**: Not normalized, hard to compare across factors
- **MIG**: Divided by $H(v_k)$ makes it comparable across different factor types

**3. Information-Theoretic Foundation**:
- **Previous metrics**: Based on classifier accuracy (indirect)
- **MIG**: Based on mutual information (direct measure of information sharing)

#### Concrete Example: dSprites

Let's work through a concrete example:

**Setup**:
```
Ground truth factors: v₁=shape, v₂=scale, v₃=rotation, v₄=posX, v₅=posY
Learned latents: z₁, z₂, z₃, z₄, z₅ (5-dimensional)
Factor entropies: H(shape)=log(3), H(scale)=log(6), H(rotation)=log(40), etc.
```

**Mutual Information Matrix** (example values):
```
       v₁(shape) v₂(scale) v₃(rotation) v₄(posX) v₅(posY)
z₁        0.95      0.05       0.02       0.01     0.03
z₂        0.10      0.92       0.08       0.05     0.02  
z₃        0.05      0.15       0.88       0.12     0.06
z₄        0.02      0.08       0.10       0.90     0.15
z₅        0.01      0.03       0.05       0.12     0.85
```

**MIG Calculation**:
```python
# For v₁ (shape):
j^(1) = argmax([0.95, 0.10, 0.05, 0.02, 0.01]) = 1  # z₁ is most informative
I_max = 0.95  # I(z₁; shape)  
I_second = max([0.10, 0.05, 0.02, 0.01]) = 0.10  # I(z₂; shape)
gap₁ = (0.95 - 0.10) / H(shape) = 0.85 / log(3) ≈ 0.77

# For v₂ (scale):
gap₂ = (0.92 - 0.15) / H(scale) = 0.77 / log(6) ≈ 0.43

# Continue for all factors...
# MIG = average of all normalized gaps
```

### Mutual Information Estimation

$$I_n(z_j; v_k) = \mathbb{E}_{q(z_j, v_k)} \left[ \log \sum_{n \in \mathcal{X}_{v_k}} q(z_j|n)p(n|v_k) \right] + H(z_j)$$

Where $\mathcal{X}_{v_k}$ is the support of $p(n|v_k)$.

### Properties of MIG

1. **Bounded**: $0 \leq \text{MIG} \leq 1$
2. **Axis-alignment**: Gap term penalizes multiple variables encoding same factor
3. **Compactness**: Encourages single variable per factor
4. **General**: Works with any latent distribution (continuous, discrete, multimodal)
5. **Classifier-free**: Direct information-theoretic computation

---

## Multi-Channel Signal Considerations

### Extension to Multi-Channel Inputs

When dealing with multi-channel signals (e.g., RGB images, multi-sensor data, spectrograms), several considerations arise:

#### 1. Channel-Specific Factors

Multi-channel inputs may have:
- **Shared factors**: Affect all channels (e.g., object pose in RGB channels)
- **Channel-specific factors**: Affect subset of channels (e.g., lighting in RGB, frequency bands in audio)

#### 2. Modified Architecture

For input $x \in \mathbb{R}^{H \times W \times C}$ with $C$ channels:

**Encoder Architecture**:
$$\begin{aligned}
h_c &= f_c(x_c) \quad \forall c \in \{1,...,C\} \quad \text{(channel-specific features)} \\
h_{\text{shared}} &= g(\text{concat}(h_1, ..., h_C)) \quad \text{(shared representation)} \\
\mu, \log\sigma^2 &= \text{MLP}(h_{\text{shared}})
\end{aligned}$$

#### 3. Total Correlation Across Channels

The TC decomposition remains valid, but interpretations change:

$$\text{TC}(z) = \text{KL}(q(z) \| \prod_j q(z_j))$$

For multi-channel data, we expect:
- **Lower TC for shared factors**: Should be represented consistently across channels
- **Higher TC for channel-specific factors**: May require coupled latent dimensions

#### 4. Multi-Modal MIG

Extend MIG to handle channel-specific ground truth:

$$\text{MIG}_{\text{multi}} = \frac{1}{K} \sum_{k=1}^K w_k \frac{1}{H(v_k)} \left( I_n(z_{j^{(k)}}; v_k) - \max_{j \neq j^{(k)}} I_n(z_j; v_k) \right)$$

Where $w_k$ weights factors by their cross-channel relevance.

#### 5. Training Considerations

**Data Augmentation**: 
- Channel dropout during training
- Cross-channel consistency losses

**Modified Objective**:
$$\mathcal{L}_{\text{multi}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] + \lambda \sum_{c,c'} \text{KL}(q(z|x_c) \| q(z|x_{c'})) - \beta \cdot \text{TC}(z)$$

The cross-channel consistency term encourages similar representations across channels.

---

## Experimental Results

### Datasets and Setup

**Synthetic Datasets**:
- **dSprites**: 2D shapes with factors (scale, rotation, posX, posY, shape)
- **3D Faces**: Synthetic faces with (azimuth, elevation, lighting)

**Real Datasets**:
- **CelebA**: Celebrity faces with labeled attributes
- **3D Chairs**: Chair images with pose variations

### Quantitative Comparisons

#### MIG Scores (Higher is Better)

| Method | dSprites | 3D Faces | CelebA |
|--------|----------|----------|---------|
| $\beta$-VAE | $0.35 \pm 0.12$ | $0.28 \pm 0.15$ | - |
| $\beta$-TCVAE | **$0.45 \pm 0.08$** | **$0.38 \pm 0.10$** | - |
| InfoGAN | $0.18 \pm 0.20$ | $0.15 \pm 0.18$ | - |
| FactorVAE | $0.44 \pm 0.09$ | $0.36 \pm 0.12$ | - |

#### ELBO vs. Disentanglement Trade-off

$\beta$-TCVAE consistently achieves:
- **Higher disentanglement** for same ELBO values
- **Better ELBO** for same disentanglement levels
- **More stable training** across different $\beta$ values

### Qualitative Results

#### CelebA Discoveries
$\beta$-TCVAE discovered **15 interpretable attributes** without supervision:
- Baldness, Gender, Mustache, Face width
- Azimuth, Skin color, Eye shadow
- Enhanced extrapolation (e.g., "bald females")

#### 3D Chairs
Discovered **6 semantic factors**:
- Azimuth, Size, Leg style, Backrest
- Material, Swivel capability

### Correlation Analysis

Strong negative correlation between Total Correlation and MIG score:
- **$\beta$-TCVAE**: $r = -0.85$ (dSprites), $r = -0.82$ (3D Faces)  
- **$\beta$-VAE**: $r = -0.64$ (dSprites), $r = -0.71$ (3D Faces)

This confirms the hypothesis that minimizing TC improves disentanglement.

---

## Practical Implementation

### Algorithm: $\beta$-TCVAE Training

```python
def beta_tcvae_loss(x_batch, encoder, decoder, beta=6):
    """
    Compute β-TCVAE loss with minibatch weighted sampling
    
    Args:
        x_batch: Input batch [B, ...]
        encoder: q(z|x) network
        decoder: p(x|z) network  
        beta: Total correlation penalty weight
    """
    B = x_batch.size(0)
    
    # Encode
    mu, logvar = encoder(x_batch)
    z = reparameterize(mu, logvar)  # [B, latent_dim]
    
    # Decode
    x_recon = decoder(z)
    recon_loss = F.mse_loss(x_recon, x_batch, reduction='sum')
    
    # KL terms
    # 1. Index-Code MI (standard KL)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 2. Total Correlation (minibatch weighted sampling)
    log_qz = gaussian_log_density(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))
    log_qz_i = gaussian_log_density(z, mu, logvar)
    log_prod_qzi = gaussian_log_density(z.unsqueeze(2), mu.unsqueeze(0).unsqueeze(2), 
                                       logvar.unsqueeze(0).unsqueeze(2)).sum(2)
    
    # MWS estimator for log q(z)
    log_qz = torch.logsumexp(log_qz, dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(log_prod_qzi, dim=1, keepdim=False)
    
    mi_loss = (log_qz_i - log_qz).mean()
    tc_loss = (log_qz - log_prod_qzi).mean()
    
    total_loss = recon_loss + mi_loss + beta * tc_loss + kl_div
    
    return total_loss, recon_loss, mi_loss, tc_loss, kl_div

def gaussian_log_density(samples, mu, logvar):
    """Compute log density of samples under Gaussian distribution"""
    normalization = -0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((samples - mu) ** 2 * inv_var)
    return log_density.sum(dim=-1)

def reparameterize(mu, logvar):
    """Reparameterization trick"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### Hyperparameter Guidelines

**$\beta$ Selection**:
- Start with $\beta = 4-6$ for most datasets
- Higher $\beta$ for datasets requiring stronger disentanglement
- Monitor MIG score and ELBO trade-off

**Architecture**:
- **Latent dimension**: 10-50 (depends on data complexity)
- **Batch size**: 64-128 (affects MWS quality)
- **Learning rate**: $1 \times 10^{-4}$ with Adam optimizer

**Training**:
- 100-500 epochs depending on dataset size
- Early stopping based on validation MIG score
- Learning rate scheduling can help stability

### MIG Computation

```python
def compute_mig(latent_codes, factors):
    """
    Compute Mutual Information Gap
    
    Args:
        latent_codes: [N, latent_dim] encoded representations
        factors: [N, factor_dim] ground truth factors
    """
    num_factors = factors.shape[1]
    num_latents = latent_codes.shape[1]
    
    # Compute mutual information matrix
    mi_matrix = np.zeros((num_latents, num_factors))
    
    for i in range(num_latents):
        for j in range(num_factors):
            mi_matrix[i, j] = mutual_info_regression(
                latent_codes[:, i:i+1], factors[:, j]
            )
    
    # Normalize by factor entropy
    factor_entropy = np.array([
        entropy_discrete(factors[:, j]) for j in range(num_factors)
    ])
    mi_matrix_norm = mi_matrix / factor_entropy[np.newaxis, :]
    
    # Compute MIG
    mig_scores = []
    for j in range(num_factors):
        sorted_mi = np.sort(mi_matrix_norm[:, j])[::-1]
        if len(sorted_mi) > 1:
            mig_scores.append(sorted_mi[0] - sorted_mi[1])
        else:
            mig_scores.append(sorted_mi[0])
    
    return np.mean(mig_scores)
```

---

## Conclusion

### Key Insights

1. **Theoretical Foundation**: The ELBO decomposition reveals that **Total Correlation** is the key component driving disentanglement in $\beta$-VAE

2. **Practical Algorithm**: $\beta$-TCVAE provides a principled way to achieve better disentanglement while maintaining computational efficiency

3. **Evaluation Metric**: MIG offers a more robust, classifier-free approach to measuring disentanglement quality

4. **Multi-Channel Extension**: The framework naturally extends to multi-channel signals with appropriate architectural modifications

### Advantages of $\beta$-TCVAE

- **No additional hyperparameters** compared to $\beta$-VAE
- **Better disentanglement-reconstruction trade-off**
- **More stable training** across different initializations
- **Plug-in replacement** for existing $\beta$-VAE implementations
- **Theoretical grounding** for why the method works

### Limitations and Future Work

1. **Biased Estimator**: MWS provides biased estimates of TC
2. **Scalability**: $O(M^2)$ computation per minibatch
3. **Prior Assumptions**: Still requires factorial prior
4. **Evaluation**: MIG requires known ground truth factors

### Research Directions

1. **Improved Estimators**: Develop unbiased, scalable TC estimators
2. **Flexible Priors**: Extend to non-factorial priors using normalizing flows
3. **Unsupervised Metrics**: Develop disentanglement metrics without ground truth
4. **Hierarchical Factors**: Handle factors of variation at different scales
5. **Multi-Modal Learning**: Apply to datasets with multiple input modalities

### Practical Impact

$\beta$-TCVAE has become a standard baseline for disentanglement research, providing:
- **Reproducible results** across different implementations
- **Clear theoretical understanding** of disentanglement mechanisms  
- **Practical tool** for controllable generation tasks
- **Foundation** for subsequent advances in disentangled representation learning

The work bridges the gap between empirical success and theoretical understanding, establishing Total Correlation minimization as a fundamental principle for learning disentangled representations.