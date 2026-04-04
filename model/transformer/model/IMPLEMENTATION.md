# Causal Multimodal Forecasting Transformer -- Implementation Reference

This document is a complete reference for the PyTorch implementation in `model/transformer/model/`. It describes every public class, function, input, output, tensor shape, configuration option, loss term, and usage pattern. It is intended to be self-contained so that a coding agent can integrate, train, test, or extend this model without reading the source code.

For the mathematical specification and design rationale, see `model/transformer/model.md` (one level up).

---

## 1. Overview

The model is a **strictly causal dual-branch forecasting architecture** for self-supervised pretraining on fetal heart rate (FHR) and uterine pressure (UP) scattering-transform sequences. It learns three types of representation:

1. **Intrinsic FHR state** -- from FHR history alone.
2. **Causal multimodal predictive state** -- from FHR + UP history.
3. **Transfer entropy (TE) latent** -- the incremental predictive contribution of UP beyond FHR's own past.

The model has two operating modes:
- **Training mode**: given anchor indices, produces multi-horizon FHR forecasts from three heads (self-only, fused, TE-residual) plus TE latent parameters.
- **Inference mode**: produces a fixed-size window-level embedding for downstream classification.

---

## 2. File Structure

```
model/transformer/
    model.md                # Mathematical specification (design document)
    model/                  # Python package (all implementation code)
        __init__.py         # Public API: TransformerConfig, CausalMultimodalTransformer,
                            #             CausalTransformerLoss, sample_anchors
        config.py           # TransformerConfig dataclass
        layers.py           # CausalConv1d, CausalConvBlock, CausalSelfAttention,
                            # CausalCrossAttention, FeedForward, AttentionPool
        stems.py            # CausalStem
        encoder.py          # CausalTransformerBlock, CausalTransformerEncoder,
                            # CausalCrossAttentionFusion
        heads.py            # ForecastHead, TELatentModule, WindowRepresentationExport
        model.py            # CausalMultimodalTransformer, CausalTransformerLoss,
                            #   sample_anchors(), _init_weights()
        IMPLEMENTATION.md   # This file
```

---

## 3. Imports

```python
from model.transformer.model import (
    TransformerConfig,
    CausalMultimodalTransformer,
    CausalTransformerLoss,
    sample_anchors,
    validate_anchor_indices,
)
```

---

## 4. Configuration -- `TransformerConfig`

A `@dataclass` containing every hyperparameter. All fields have defaults matching the spec, so the model can be instantiated with zero arguments.

### 4.1 All Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `d_model` | `int` | `192` | Backbone hidden dimension. All internal representations have this width. |
| `d_f` | `int` | `43` | Number of FHR scattering-transform input channels. |
| `d_u` | `int` | `43` | Number of UP scattering-transform input channels. |
| `d_z` | `int` | `16` | TE latent dimension. |
| `n_heads` | `int` | `4` | Number of attention heads. `d_model` must be divisible by `n_heads`. |
| `dropout` | `float` | `0.1` | Dropout probability used throughout the model. |
| `stem_num_blocks` | `int` | `3` | Number of residual causal-conv blocks per stem. |
| `stem_kernels` | `Tuple[int, ...]` | `(3, 5, 5)` | Kernel sizes for each stem block. Length must equal `stem_num_blocks`. |
| `stem_dilations` | `Tuple[int, ...]` | `(1, 2, 4)` | Dilation factors for each stem block. Length must equal `stem_num_blocks`. |
| `stem_expansion` | `int` | `2` | Pointwise expansion ratio in stem conv blocks. |
| `fhr_encoder_layers` | `int` | `4` | Number of causal transformer blocks in the FHR-only encoder. |
| `up_encoder_layers` | `int` | `4` | Number of causal transformer blocks in the UP-only encoder. |
| `fused_encoder_layers` | `int` | `4` | Number of causal transformer blocks in the fused encoder. |
| `ff_expansion` | `int` | `4` | Feed-forward expansion ratio in transformer blocks. |
| `seq_len` | `int` | `300` | Effective sequence length T (after 1-min trimming from each end). |
| `ctx_len` | `int` | `30` | Local context length L_ctx for anchor-based attention pooling. |
| `guard_gap` | `int` | `4` | Guard gap g between anchor time and prediction target start (~16 seconds). |
| `horizons` | `Tuple[int, ...]` | `(8, 15, 30)` | Prediction horizons in time steps (~32s, ~60s, ~120s). |
| `horizon_weights` | `Tuple[float, ...]` | `(1.0, 1.5, 2.0)` | Loss weights per horizon. Longer horizons weighted more. |
| `num_anchors` | `int` | `4` | Number of anchors K sampled per window during training. |
| `anchor_uniform_ratio` | `float` | `0.5` | Fraction of anchors drawn uniformly; rest are activity-biased. |
| `lambda_fus` | `float` | `1.0` | Weight for the main fused forecasting loss. |
| `lambda_delta` | `float` | `0.5` | Weight for the dynamics (temporal difference) loss. |
| `lambda_self` | `float` | `0.25` | Weight for the self-only baseline loss. |
| `lambda_te` | `float` | `0.25` | Weight for the TE residual loss. |
| `huber_delta` | `float` | `1.0` | Threshold delta for the Huber loss function. |
| `gradient_checkpointing` | `bool` | `False` | Enable gradient checkpointing in encoder blocks (saves memory, slower). |

### 4.2 Derived Properties

| Property | Value (defaults) | Formula |
|----------|-----------------|---------|
| `d_head` | `48` | `d_model // n_heads` |
| `max_horizon` | `30` | `max(horizons)` |
| `valid_anchor_start` | `29` | `ctx_len - 1` |
| `valid_anchor_end` | `265` | `seq_len - guard_gap - max_horizon - 1` |

### 4.3 Validation

`__post_init__` checks:
- `d_model % n_heads == 0`
- `len(stem_kernels) == stem_num_blocks`
- `len(stem_dilations) == stem_num_blocks`
- `len(horizons) == len(horizon_weights)`
- `valid_anchor_end - valid_anchor_start + 1 > 0` (at least one valid anchor exists)
- `num_anchors <= valid_range` (can sample without replacement)
- `anchor_uniform_ratio in [0, 1]`

### 4.4 Example

```python
# Default config
config = TransformerConfig()

# Custom config
config = TransformerConfig(
    d_model=256,
    n_heads=8,
    fhr_encoder_layers=6,
    dropout=0.15,
    num_anchors=3,
)
```

---

## 5. Model -- `CausalMultimodalTransformer`

### 5.1 Constructor

```python
model = CausalMultimodalTransformer(config)
# OR with keyword arguments:
model = CausalMultimodalTransformer(d_model=256, n_heads=8)
```

The constructor builds all submodules and applies Xavier weight initialization.

### 5.2 Architecture Data Flow

```
Y (B, 300, 43) ─> fhr_stem ─> F (B, 300, 192) ─> fhr_encoder ─> H_F (B, 300, 192) ──┐
                                                                                        │
U (B, 300, 43) ─> up_stem  ─> S (B, 300, 192) ─> up_encoder  ─> H_U (B, 300, 192) ──┤
                                                                                        │
                           fusion(Q=H_F, KV=H_U) ─> H_tilde (B, 300, 192)              │
                                      │                                                 │
                                fused_encoder ─> H_FU (B, 300, 192)                     │
                                      │                                                 │
                  ┌───────────────────┴───────────────────┐                             │
            [Training Mode]                        [Inference Mode]                     │
                  │                                       │                             │
       sample anchors (B, K)                    WindowRepresentationExport              │
                  │                                       │                             │
    gather context windows                         e_win (B, 1568)                     │
    at each anchor position                                                             │
                  │                                                                     │
         AttentionPool ──> s_F, s_U, s_FU  (B*K, 192 each)  <──────────────────────────┘
                  │
    ┌─────────────┼──────────────────┐
    │             │                  │
self_head     fused_head      TE module + te_head
    │             │                  │
Y_hat_self   Y_hat_fus     z_te ─> R_hat ─> Y_hat_te = sg(Y_hat_self) + R_hat
```

### 5.3 Submodules

| Attribute | Class | Purpose |
|-----------|-------|---------|
| `fhr_stem` | `CausalStem` | FHR causal conv stem (d_f=43 -> d=192) |
| `up_stem` | `CausalStem` | UP causal conv stem (d_u=43 -> d=192) |
| `fhr_encoder` | `CausalTransformerEncoder` | 4-layer FHR-only causal encoder |
| `up_encoder` | `CausalTransformerEncoder` | 4-layer UP-only causal encoder |
| `fusion` | `CausalCrossAttentionFusion` | Cross-attention + gated residual fusion |
| `fused_encoder` | `CausalTransformerEncoder` | 4-layer fused causal encoder |
| `pool_f` | `AttentionPool` | Pools FHR context windows at anchors |
| `pool_u` | `AttentionPool` | Pools UP context windows at anchors |
| `pool_fu` | `AttentionPool` | Pools fused context windows at anchors |
| `self_head` | `ForecastHead` | Self-only FHR forecast (in_dim=192) |
| `fused_head` | `ForecastHead` | Fused multimodal forecast (in_dim=192) |
| `te_head` | `ForecastHead` | TE residual forecast (in_dim=192+16=208) |
| `te_module` | `TELatentModule` | Posterior/prior/reparameterize for TE latent |
| `window_export` | `WindowRepresentationExport` | Inference-mode embedding export |

### 5.4 Forward Pass -- Training Mode

```python
model.train()
outputs = model(Y, U, anchor_indices=anchors)
```

**Inputs:**

| Argument | Shape | Description |
|----------|-------|-------------|
| `Y` | `(B, T, d_f)` = `(B, 300, 43)` | FHR scattering-transform features (normalized). |
| `U` | `(B, T, d_u)` = `(B, 300, 43)` | UP scattering-transform features (normalized). |
| `anchor_indices` | `(B, K)` e.g. `(B, 4)` | Sampled anchor positions. Values in `[29, 265]`. LongTensor. |

**Output dictionary:**

| Key | Shape | Description |
|-----|-------|-------------|
| `Y_hat_self` | `Dict[int, (B*K, h, 43)]` | Self-only FHR forecasts per horizon. Keys: `{8, 15, 30}`. |
| `Y_hat_fus` | `Dict[int, (B*K, h, 43)]` | Fused multimodal FHR forecasts per horizon. |
| `Y_hat_te` | `Dict[int, (B*K, h, 43)]` | TE-augmented forecasts: `sg(Y_hat_self[h]) + R_hat[h]`. |
| `R_hat` | `Dict[int, (B*K, h, 43)]` | Raw TE residual predictions per horizon. |
| `mu_post` | `(B*K, 16)` | Posterior mean of the TE latent. |
| `logvar_post` | `(B*K, 16)` | Posterior log-variance of the TE latent. |
| `mu_prior` | `(B*K, 16)` | Prior mean of the TE latent. |
| `logvar_prior` | `(B*K, 16)` | Prior log-variance of the TE latent. |
| `anchor_indices` | `(B, K)` | Passed through for loss computation. |
| `H_F` | `(B, 300, 192)` | FHR encoder states (for inspection/export). |
| `H_FU` | `(B, 300, 192)` | Fused encoder states (for inspection/export). |

**Note:** The first dimension of forecast/latent outputs is `B*K` (batch * anchors flattened), not `(B, K, ...)`.

### 5.5 Forward Pass -- Inference Mode

```python
model.eval()
with torch.no_grad():
    outputs = model(Y, U)  # anchor_indices=None
```

**Output dictionary:**

| Key | Shape | Description |
|-----|-------|-------------|
| `e_win` | `(B, 1568)` | Window-level embedding for downstream classification. |

The embedding dimension `1568 = 8 * d_model + 2 * d_z = 8 * 192 + 2 * 16` is also available as `model.window_export.output_dim`.

**Embedding structure** (concatenated in this order):

| Component | Dim | Source | Description |
|-----------|-----|--------|-------------|
| `e_f_attn` | 192 | `AttentionPool(H_F)` | Attention-weighted FHR summary |
| `e_f_max` | 192 | `max(H_F, dim=T)` | Max-pooled FHR summary |
| `e_fu_attn` | 192 | `AttentionPool(H_FU)` | Attention-weighted fused summary |
| `e_fu_max` | 192 | `max(H_FU, dim=T)` | Max-pooled fused summary |
| `q_1` | 192 | `mean(H_FU[:, 0:75, :])` | 1st quarter mean (earliest ~5 min) |
| `q_2` | 192 | `mean(H_FU[:, 75:150, :])` | 2nd quarter mean |
| `q_3` | 192 | `mean(H_FU[:, 150:225, :])` | 3rd quarter mean |
| `q_4` | 192 | `mean(H_FU[:, 225:300, :])` | 4th quarter mean (latest ~5 min) |
| `e_te_mean` | 16 | `mean(mu_post)` over grid | Mean TE posterior means |
| `e_te_max` | 16 | `max(mu_post)` over grid | Max TE posterior means |

In inference mode, the TE posterior means are computed on a fixed anchor grid (every 15 steps from index 29 to 265, giving 16 anchors).

---

## 6. Anchor Sampling -- `sample_anchors()`

```python
anchors = sample_anchors(Y, U, config, training=True)
```

| Argument | Type | Description |
|----------|------|-------------|
| `Y` | `Tensor (B, T, d_f)` | FHR features (used for activity scoring). |
| `U` | `Tensor (B, T, d_u)` | UP features (used for activity scoring). |
| `config` | `TransformerConfig` | Configuration. |
| `training` | `bool` | Whether to sample stochastically or use a fixed grid. |

**Returns:**

| Mode | Shape | Description |
|------|-------|-------------|
| `training=True` | `(B, K)` e.g. `(B, 4)` | Stochastically sampled anchors. Values in `[29, 265]`. |
| `training=False` | `(B, K_grid)` e.g. `(B, 16)` | Fixed grid: `[30, 45, 60, ..., 255]`. Same for all batch elements. |

**Training sampling strategy:**
- Activity score at each valid position t: `s_t = |U_t|_1 + |delta_U_t|_1 + |delta_Y_t|_1`
- Mixed distribution: `p(t) = 0.5 * uniform + 0.5 * (s_t / sum(s))`
- K anchors sampled without replacement per batch element.

This function is standalone (not a model method) so it can be called before the forward pass:

```python
anchors = sample_anchors(Y, U, config, training=model.training)
outputs = model(Y, U, anchor_indices=anchors)
```

---

## 7. Loss -- `CausalTransformerLoss`

### 7.1 Constructor

```python
loss_fn = CausalTransformerLoss(config)
```

### 7.2 Forward

```python
losses = loss_fn(outputs, Y)
```

| Argument | Type | Description |
|----------|------|-------------|
| `outputs` | `dict` | Output from `model.forward()` in training mode. |
| `Y` | `Tensor (B, T, d_f)` | FHR input features. Used to extract future targets. |

**Returns a dict:**

| Key | Type | Description |
|-----|------|-------------|
| `total_loss` | `Tensor (scalar)` | `lambda_fus * L_fus + lambda_delta * L_delta + lambda_self * L_self + lambda_te * L_te`. **Does NOT include L_kl** (see below). |
| `L_fus` | `Tensor (scalar)` | Horizon-weighted Huber loss on fused forecasts vs. actual future FHR. Main training signal. |
| `L_delta` | `Tensor (scalar)` | Horizon-weighted Huber loss on temporal differences of fused forecasts. Preserves dynamics. |
| `L_self` | `Tensor (scalar)` | Horizon-weighted Huber loss on self-only forecasts. Trains the FHR-only baseline. |
| `L_te` | `Tensor (scalar)` | Horizon-weighted Huber loss on TE residuals vs. `(target - sg(Y_hat_self))`. |
| `L_kl` | `Tensor (scalar)` | `KL(posterior || prior)` for the TE latent. Closed-form diagonal Gaussian KL. |

### 7.3 Why `L_kl` Is Separate

The KL loss uses a time-dependent weight `beta(t)` with warmup scheduling. Since this is controlled by the training loop, not the model, `L_kl` is returned separately. The training loop should compute:

```python
final_loss = losses["total_loss"] + beta * losses["L_kl"]
final_loss.backward()
```

### 7.4 Loss Details

**Target extraction:** For anchor `a` and horizon `h`, the target is `Y[:, a+g+1 : a+g+1+h, :]` where `g=4` is the guard gap.

**Huber loss:** Uses `F.huber_loss(pred, target, delta=config.huber_delta)`. Robust to heavy-tailed feature distributions.

**Horizon weighting:** Each horizon's loss is multiplied by its weight (`w_8=1.0, w_15=1.5, w_30=2.0`), then the total is normalized by the sum of weights.

**Stop-gradient:** The TE residual target uses `sg(Y_hat_self)` -- gradients from `L_te` do not flow back through the self-only head. This forces the TE branch to explain only what UP adds beyond FHR.

---

## 8. Complete Training Example

```python
import torch
from model.transformer.model import (
    TransformerConfig,
    CausalMultimodalTransformer,
    CausalTransformerLoss,
    sample_anchors,
)

# 1. Setup
config = TransformerConfig()
model = CausalMultimodalTransformer(config).cuda()
loss_fn = CausalTransformerLoss(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. Training step
model.train()
Y = batch["fhr_st"].cuda()      # (B, 300, 43) -- normalized FHR scattering
U = batch["up_st"].cuda()        # (B, 300, 43) -- normalized UP scattering

# 2a. Sample anchors
anchors = sample_anchors(Y, U, config, training=True)  # (B, 4)

# 2b. Forward pass
outputs = model(Y, U, anchor_indices=anchors)

# 2c. Compute loss
losses = loss_fn(outputs, Y)

# 2d. Apply KL with beta warmup
beta = compute_beta(step)  # e.g. linear warmup to 1e-4
total_loss = losses["total_loss"] + beta * losses["L_kl"]

# 2e. Backward + step
optimizer.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

# 2f. Log metrics
print(f"L_fus={losses['L_fus']:.4f}  L_self={losses['L_self']:.4f}  "
      f"L_kl={losses['L_kl']:.4f}")
```

---

## 9. Complete Inference Example

```python
model.eval()
embeddings = []

with torch.no_grad():
    for batch in dataloader:
        Y = batch["fhr_st"].cuda()   # (B, 300, 43)
        U = batch["up_st"].cuda()    # (B, 300, 43)
        outputs = model(Y, U)        # inference mode (no anchors)
        embeddings.append(outputs["e_win"])  # (B, 1568)

all_embeddings = torch.cat(embeddings, dim=0)  # (N, 1568)
```

---

## 10. Staged Training Schedule

The spec prescribes a 3-stage training schedule (spec §19):

### Stage 1: Deterministic warm start

Train with `L_fus + lambda_delta * L_delta + lambda_self * L_self` only. Set `config.lambda_te = 0` and `beta = 0`.

Purpose: stabilize stems and encoders, learn good intrinsic and fused predictive states.

### Stage 2: Activate TE residual head

Set `config.lambda_te = 0.25` but keep `beta` near zero.

Purpose: make the TE head useful before applying the KL bottleneck.

### Stage 3: KL warmup

Gradually increase `beta` from 0 to `beta_max` (e.g. `1e-4` to `1e-3`):

```python
beta = beta_max * min(1.0, step / warmup_steps)
```

Purpose: force the TE latent to become compact and source-incremental.

**Alternative:** Control stages by modifying `config.lambda_te` and `beta` at runtime. The loss module reads `config.lambda_*` from `self.config`, so changing them on the config object takes effect immediately.

---

## 11. Component Details

### 11.1 CausalStem (`stems.py`)

Each stem (FHR and UP) has:
1. `Linear(in_dim -> d_model)` -- channel projection
2. 3 x `CausalConvBlock` with increasing receptive fields:
   - Block 0: kernel=3, dilation=1 (receptive field = 3 steps)
   - Block 1: kernel=5, dilation=2 (receptive field += 9)
   - Block 2: kernel=5, dilation=4 (receptive field += 17)

Total receptive field: ~29 steps (~116 seconds), covering ~2 minutes of local context.

Each `CausalConvBlock`:
```
x -> LayerNorm -> transpose -> DWConv_causal(groups=d) -> transpose -> Linear(d->2d) -> GELU -> Linear(2d->d) -> Dropout -> + x
```

### 11.2 CausalTransformerEncoder (`encoder.py`)

Stack of N `CausalTransformerBlock` modules + final `LayerNorm`.

Each block:
```
x = x + CausalSelfAttention(x)      # pre-norm MHA with is_causal=True
x = x + FeedForward(x)              # pre-norm FFN
```

Self-attention uses a fused QKV projection for efficiency and `F.scaled_dot_product_attention(is_causal=True)` which auto-selects FlashAttention.

### 11.3 CausalCrossAttentionFusion (`encoder.py`)

Implements spec §9: gated cross-attention fusion.

```
C = CausalCrossAttention(Q=H_F, KV=H_U)   # causal cross-attention
G = sigmoid(Linear([H_F | C]))              # gate
H_tilde = H_F + G * C                       # gated residual
```

The gate learns per-timestep how much UP context should influence the fused state.

### 11.4 ForecastHead (`heads.py`)

Contains a separate MLP for each horizon h in `{8, 15, 30}`:
```
Linear(in_dim -> 4*in_dim) -> GELU -> Dropout -> Linear(4*in_dim -> h*d_f) -> reshape to (h, d_f)
```

Three instances:
- `self_head`: in_dim = 192 (from s_F)
- `fused_head`: in_dim = 192 (from s_FU)
- `te_head`: in_dim = 208 (from [s_F | z_te])

### 11.5 TELatentModule (`heads.py`)

**Posterior** (sees FHR + UP): `[s_F | s_U]` (384) -> LayerNorm -> MLP -> (mu, logvar) each (16,)

**Prior** (sees FHR only): `s_F` (192) -> LayerNorm -> MLP -> (mu0, logvar0) each (16,)

**Reparameterization**: `z = mu + exp(0.5 * logvar) * epsilon` during training; `z = mu` during eval.

**KL divergence**: `TELatentModule.kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior)` -- static method, closed-form.

### 11.6 WindowRepresentationExport (`heads.py`)

Produces a 1568-dim embedding by pooling encoder states. See Section 5.5 above for the full breakdown.

---

## 12. Input Data Format

The model expects **normalized** scattering-transform features from `CombinedHDF5Dataset` with `trim_minutes=1.0` and stats-based normalization.

| Field | HDF5 storage | After trim & transpose | Description |
|-------|-------------|----------------------|-------------|
| `fhr_st` | `(43, 330)` | `(300, 43)` | FHR scattering coefficients. 43 channels: order-0 + first-order wavelets. |
| `up_st` | `(43, 330)` | `(300, 43)` | UP scattering coefficients. Same structure as fhr_st. |

**Normalization pipeline** (applied at load time by `CombinedHDF5Dataset`):
1. `fhr_st` channel 0: standardize directly. Channels 1-42: `log(clamp(x, 0) + 1e-6)` then standardize.
2. `up_st`: same transform as `fhr_st`.
3. Standardization: `(x - mean) / (std + eps)` using per-channel stats from training split.

**Important:** `Y` corresponds to `fhr_st` and `U` corresponds to `up_st` in the model's `forward()`.

---

## 13. Weight Initialization

Applied automatically in the constructor via `model.apply(_init_weights)`:

| Module type | Weight init | Bias init |
|-------------|------------|-----------|
| `nn.Linear` | Xavier uniform | Zeros |
| `nn.Conv1d` | Xavier uniform | Zeros |
| `nn.LayerNorm` | Ones | Zeros |

---

## 14. Estimated Parameter Count

With default config (`d_model=192`, 4+4+4 encoder layers):

| Component | Approximate params |
|-----------|--------------------|
| FHR stem | ~230K |
| UP stem | ~230K |
| FHR encoder (4 layers) | ~1.15M |
| UP encoder (4 layers) | ~1.15M |
| Cross-attention fusion | ~220K |
| Fused encoder (4 layers) | ~1.15M |
| Attention pools (3x) | ~115K |
| Forecast heads (3x) | ~350K |
| TE latent module | ~12K |
| Window export | ~75K |
| **Total** | **~4.7M** |

---

## 15. Key Design Decisions

1. **Strictly causal**: No bidirectional attention, no masked reconstruction. All information flows from past to future only.

2. **Three heads are not redundant**: Self-only provides the TE baseline, fused is the main representation learner, TE residual captures only what UP adds beyond FHR.

3. **Stop-gradient on self prediction**: `Y_hat_te = sg(Y_hat_self) + R_hat`. The TE head gradient does not update the self-only pathway.

4. **Anchor sampling is external**: `sample_anchors()` is a standalone function, not a model method. The model receives `anchor_indices` as input. This keeps the model deterministic for a given set of anchors. When supplying custom anchors, call `validate_anchor_indices(anchors, config)` beforehand — this runs on CPU and is safe outside `torch.compile` graphs.

5. **No positional encoding**: Causal stems with increasing dilation encode local temporal structure. Causal attention masks provide temporal ordering. Can be added later if needed.

6. **FlashAttention**: All attention layers use `F.scaled_dot_product_attention(is_causal=True)`, which automatically selects the fastest available backend (FlashAttention, memory-efficient, or math).

7. **KL loss returned separately**: The training loop controls `beta(t)` warmup. The loss module's `total_loss` includes only the four forecast-related terms.

---

## 16. Integration with Existing Codebase

### Lightning wrapper

The model is designed to be wrapped in `LightningModelBase` from `train/pl_model_base.py`:

```python
from train.pl_model_base import LightningModelBase

class LitCausalTransformer(LightningModelBase):
    def __init__(self, config, beta_max=1e-4, warmup_steps=5000, **kwargs):
        super().__init__(**kwargs)
        self.model = CausalMultimodalTransformer(config)
        self.loss_fn = CausalTransformerLoss(config)
        self.config = config
        self.beta_max = beta_max
        self.warmup_steps = warmup_steps

    def compute_loss_and_metrics(self, batch, batch_idx, stage):
        Y = batch["fhr_st"]
        U = batch["up_st"]
        anchors = sample_anchors(Y, U, self.config, training=(stage == "train"))
        outputs = self.model(Y, U, anchor_indices=anchors)
        losses = self.loss_fn(outputs, Y)
        beta = self.beta_max * min(1.0, self.global_step / self.warmup_steps)
        total = losses["total_loss"] + beta * losses["L_kl"]
        metrics = {k: v.detach() for k, v in losses.items()}
        metrics["beta"] = beta
        return total, metrics
```

### Checkpoint loading

Use `train/graph_models_utils.py`:

```python
from train.graph_models_utils import load_checkpoint_strict
model = CausalMultimodalTransformer(config)
load_checkpoint_strict(model, "path/to/checkpoint.ckpt")
```

### DataLoader

```python
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

loader = create_optimized_dataloader(
    hdf5_files=["fold_1/train/acidosis_cs.hdf5", ...],
    batch_size=32,
    stats_path="fold_1/train/stats.hdf5",
    trim_minutes=1.0,
    load_fields=["fhr_st", "up_st", "target", "weight"],
    normalize_fields=["fhr_st", "up_st"],
)
```

---

## 17. Verification Checklist

When testing this model, verify:

1. **Shape test**: `model(Y, U, anchors)` with `Y, U = (4, 300, 43)` and `anchors = (4, 4)` produces outputs with correct shapes (see Section 5.4 table).
2. **Inference shape**: `model(Y, U)` produces `e_win` of shape `(4, 1568)`.
3. **Causality**: Perturb `Y[:, t+1:, :]`; verify `H_F[:, :t+1, :]` is unchanged.
4. **Gradient flow**: After `total_loss.backward()`, all parameters have non-None gradients.
5. **Loss sanity**: All loss components are finite and non-negative.
6. **Anchor range**: All sampled anchors are in `[config.valid_anchor_start, config.valid_anchor_end]`.
7. **torch.compile compatibility**: `compiled = torch.compile(model); compiled(Y, U, anchors)` runs without error.
