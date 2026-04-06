# Latent Representation and Downstream Classification

This document describes the learned representations produced by the Causal Multimodal Forecasting Transformer and how to use them for downstream classification tasks.

---

## 1. What the model produces

The model operates on a single 20-minute window of FHR and UP scattering-transform features (`fhr_st` and `up_st`, each shape `(300, 43)`). Internally it builds several representations at different levels of abstraction:

### Per-timestep representations (dense, full temporal resolution)

| Representation | Shape | Description |
|---------------|-------|-------------|
| `H_F[t]` | `(300, 192)` | FHR-only causal encoder output at each time step. Captures intrinsic fetal dynamics. |
| `H_U[t]` | `(300, 192)` | UP-only causal encoder output. Captures uterine contraction dynamics. |
| `H_FU[t]` | `(300, 192)` | Fused causal encoder output. Contains the full multimodal predictive state at each step. |

These exist at all 300 time steps and carry the bulk of the model's learned knowledge. They are fully deterministic (no sampling, no KL).

### Per-anchor representations (sparse, at sampled time points)

| Representation | Shape | Description |
|---------------|-------|-------------|
| `s_F` | `(K, 192)` | Attention-pooled FHR context at each anchor. |
| `s_U` | `(K, 192)` | Attention-pooled UP context at each anchor. |
| `s_FU` | `(K, 192)` | Attention-pooled fused context at each anchor. |
| `z_te` | `(K, 16)` | Sampled TE latent at each anchor. The only stochastic variable in the model. |
| `mu_post` | `(K, 16)` | Posterior mean of `z_te` (deterministic at inference). |

K = 4 anchors during training, ~16 on a fixed grid during inference.

### Window-level embedding (pooled, single vector per 20-min segment)

| Representation | Shape | Description |
|---------------|-------|-------------|
| `e_win` | `(1568,)` | Fixed-size summary of the entire 20-minute window. |

The 1568 dimensions decompose as:

| Component | Dims | Source | What it captures |
|-----------|------|--------|-----------------|
| `e_F` (attn_pool + max_pool of H_F) | 0--383 (384) | FHR encoder | Intrinsic fetal state: baseline, variability, autonomous decelerations |
| `e_FU` (attn_pool + max_pool + 4 quarter means of H_FU) | 384--1535 (1152) | Fused encoder | Full multimodal state, including temporal progression via quarter pools |
| `e_TE` (mean + max of posterior means) | 1536--1567 (32) | TE latent | UP-to-FHR coupling strength: how much UP predicts FHR beyond self-history |

---

## 2. Is this model a VAE?

**Partially.** It has a VAE-style component (the TE branch with posterior, prior, reparameterization, and KL divergence), but it is **not** a global VAE. The TE latent is:

- **Local** -- defined per anchor point, not per window or per time step
- **Auxiliary** -- it exists to give the transfer entropy concept meaning, not to bottleneck the whole representation
- **Small** (d_z = 16) -- intentionally compact so it captures only source-incremental information
- **Conditional** -- the KL is between `q(z|FHR,UP)` and `r(z|FHR)`, not against a standard normal

The main representation power lives in the 192-dim encoder states (`H_F` and `H_FU`), which are trained by the forecasting losses without any information bottleneck.

---

## 3. Classification approaches

### 3.1 Per-window classification

One `e_win` per window, one prediction per window.

```
e_win (1568) -> MLP -> healthy / unhealthy
```

**Pros:** Simple, fast, no sequence modeling needed.
**Cons:** Ignores the temporal trajectory across hours of monitoring. A baby has ~18 windows over 6 hours. One bad window among 18 good ones means something different than 18 bad windows.

### 3.2 Per-baby classification using e_win (recommended starting point)

Each baby (GUID) has `N_i` windows at different epochs (times before delivery). The classifier sees the full sequence of window embeddings:

```
Window 1 (6h before delivery):   e_win_1  (1568)
Window 2 (5h 40min before):      e_win_2  (1568)
...
Window N (20min before):          e_win_N  (1568)
         |
         v
Add time-to-delivery embedding:  [e_win_j | pi(tau_j)]   per window
         |
         v
Temporal aggregation:  attention pooling + max pooling across windows
         |
         v
Baby-level vector:  h_i
         |
         v
Classifier head:  MLP -> healthy / unhealthy / severity
```

**Why attention + max pooling:**
- Attention pooling gives a smooth summary, weighting the most clinically relevant windows.
- Max pooling preserves rare severe events that might be diluted by averaging.
- Time-to-delivery embedding tells the classifier whether a concerning pattern at 6 hours vs 30 minutes before delivery has different significance.

**This is already implemented** in the project for the existing VAE model. `SignalSequenceDataset` in `hdf5_dataset/guid_hdf5_dataset.py` groups segments by GUID and provides `delta_t` and `segment_indices` for temporal modeling. `TemporalVaeClassifier` in `model/vae_teb_prediction/guid_classifier/temporal_classification_model.py` aggregates with LSTM + attention.

### 3.3 Per-baby classification using H_FU (maximum temporal resolution)

Instead of pooling each window to a 1568-dim vector, keep the full per-timestep representation:

```
Per window:  H_FU (300, 192) -> segment encoder (LSTM/attention) -> segment embedding (e.g. 256)
Per baby:    [seg_emb_1, ..., seg_emb_N] -> temporal LSTM -> baby embedding -> classifier
```

**Pros:** Preserves within-window temporal dynamics (e.g. exact deceleration timing, recovery patterns) that `e_win`'s pooling discards.
**Cons:** More complex classifier, more compute, harder to debug.

---

## 4. Choosing the right representation

| Representation | Shape per window | Temporal resolution | Complexity | Best for |
|---------------|-----------------|--------------------|-----------|---------| 
| `e_win` | (1568,) | None (fully pooled) | Low | First experiments, bag-of-windows, per-window labels |
| `H_FU` full sequence | (300, 192) | Full (~4 sec per step) | High | Temporal classifiers that need within-window dynamics |
| `H_FU` + `H_F` + TE means | (300, 192) + (300, 192) + (K, 16) | Full + sparse anchors | Highest | Maximum information, complex classifier |

**Recommendation:** Start with `e_win` (1568). It captures all three conceptual pieces (intrinsic FHR, fused multimodal, TE coupling) in a fixed-size vector. If classification performance is insufficient, upgrade to `H_FU` (300, 192) fed into a temporal classifier for finer resolution.

---

## 5. Extracting representations from a trained model

### Extract e_win (inference mode)

```python
from model.transformer.model import CausalMultimodalTransformer, TransformerConfig

config = TransformerConfig()
model = CausalMultimodalTransformer(config)
# ... load checkpoint ...

model.eval()
with torch.no_grad():
    outputs = model(Y, U)            # anchor_indices=None -> inference mode
    e_win = outputs["e_win"]          # (B, 1568)
```

### Extract H_F, H_FU, and TE means (training mode forward)

```python
from model.transformer.model import sample_anchors

model.eval()
with torch.no_grad():
    anchors = sample_anchors(Y, U, config, training=False)   # fixed grid
    outputs = model(Y, U, anchor_indices=anchors)
    H_F = outputs["H_F"]             # (B, 300, 192)
    H_FU = outputs["H_FU"]           # (B, 300, 192)
    mu_post = outputs["mu_post"]     # (B*K, 16) -> reshape to (B, K, 16)
```

### Freeze the pretrained encoder for classification

```python
# Freeze all pretrained weights
for param in model.parameters():
    param.requires_grad = False

# Add a trainable classifier head on top
classifier = nn.Sequential(
    nn.Linear(1568, 256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 2),      # binary: healthy vs unhealthy
)
```

---

## 6. Per-baby aggregation (from model.md section 21)

For infant `i` with `N_i` windows, define the augmented embedding:

```
e_tilde_{i,j} = [e_win_{i,j} | pi(tau_{i,j})]
```

where `pi(tau)` is a learned time-to-delivery embedding.

Attention pooling:

```
b_i = sum_j alpha_{i,j} * e_tilde_{i,j}
alpha_{i,j} = softmax(v^T tanh(W e_tilde_{i,j}))
```

Max pooling:

```
m_i = max_j e_tilde_{i,j}
```

Baby-level input to classifier:

```
h_i = [b_i | m_i]
```

This aggregation is the right one because:
- Attention gives a smooth summary, focusing on the most informative windows.
- Max pooling preserves rare severe windows that attention might smooth over.
- Together they capture both the typical state and the worst-case state of the fetus.
