# $\beta$-TCVAE Implementation Details

---

### 1. `elbo_decomposition.py` - ELBO Analysis

Implements the theoretical ELBO decomposition:
- **Index-Code MI** estimation
- **Total Correlation** computation
- **Dimension-wise KL** calculation
- **Entropy estimation** for analysis

### 2. `disentanglement_metrics.py` - MIG Evaluation

Computes the Mutual Information Gap (MIG) metric:
- **Dataset-specific implementations** for dSprites and 3D Faces
- **Conditional entropy estimation** for each factor
- **Marginal entropy computation**

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