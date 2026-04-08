# Causal Multimodal Transformer — Testing & Analysis Pipeline

## Context

We have trained a Causal Multimodal Forecasting Transformer that learns:
- **Intrinsic FHR dynamics** via a self-only encoder pathway (H_F)
- **Fused FHR+UP dynamics** via cross-attention fusion with gating (H_FU)
- **Transfer entropy coupling** (TE) — the incremental predictive contribution of UP beyond FHR's own past

The model produces three forecast branches at anchor points, a TE latent space (d_z=16), and a 1568-dim window embedding for downstream classification. It was trained on scattering-transform features from fetal heart rate (FHR) and uterine pressure (UP) signals across three clinical outcome classes: HEALTHY, ACIDOSIS, and HIE.

This document specifies a comprehensive testing and analysis pipeline to evaluate and visualize every aspect of the trained model across the three classes.

---

## Input Specification

The pipeline accepts **3 lists of HDF5 dataset paths**, one per clinical class:

```python
# Full 3-class usage:
class_data_paths = {
    "healthy":  ["path/to/healthy_no_bg_no_cs.hdf5", "path/to/healthy_no_bg_cs.hdf5", ...],
    "acidosis": ["path/to/acidosis_cs.hdf5", "path/to/acidosis_no_cs.hdf5"],
    "hie":      ["path/to/hie_cs.hdf5", "path/to/hie_no_cs.hdf5"],
}

# Single-class usage (also valid):
class_data_paths = {
    "acidosis": ["path/to/acidosis_cs.hdf5", "path/to/acidosis_no_cs.hdf5"],
}
```

Keys are arbitrary class names (not restricted to "healthy"/"acidosis"/"hie"). You may pass 1, 2, or 3+ classes. Each HDF5 file contains segments with fields: `fhr_st` (B,300,43), `up_st` (B,300,43), `fhr` (raw), `up` (raw), `guid`, `epoch`, `cs_label`, `bg_label`.

All analyses run **per class** and then **cross-class** for comparison (when ≥2 classes provided).

### Flexible class count behaviour

| # Classes | Categories 1-5 (per-class) | Category 6 (cross-class) |
|-----------|---------------------------|--------------------------|
| 1 | Full analysis for that class | Histograms (single-color), summary table, subgroup breakdown. **Skip**: statistical tests, effect sizes, ROC/confusion (need ≥2 classes). |
| 2 | Full analysis per class | Everything runs. Pairwise tests = single comparison. |
| 3+ | Full analysis per class | Everything runs. Full pairwise matrix. |

The pipeline must never crash due to class count. Cross-class analyses that require ≥2 classes log a warning and return empty results when only 1 class is provided.

---

## Model Outputs Available for Analysis

### Training-mode forward (with anchors)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `Y_hat_self[h]` | (B*K, h, 43) | Self-only forecast at horizon h |
| `Y_hat_fus[h]` | (B*K, h, 43) | Fused forecast at horizon h |
| `Y_hat_te[h]` | (B*K, h, 43) | TE-augmented forecast = sg(self) + R_hat |
| `R_hat[h]` | (B*K, h, 43) | Pure TE residual (UP's incremental contribution) |
| `mu_post` | (B*K, 16) | Posterior mean (from [s_f \| s_u]) |
| `logvar_post` | (B*K, 16) | Posterior log-variance |
| `mu_prior` | (B*K, 16) | Prior mean (from s_f only) |
| `logvar_prior` | (B*K, 16) | Prior log-variance |
| `H_F` | (B, 300, 192) | FHR-only encoder states |
| `H_FU` | (B, 300, 192) | Fused encoder states |

### Extracted intermediates (partial forward)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `H_U` | (B, 300, 192) | UP encoder states |
| `gate` | (B, 300, 192) | Gating activation (sigmoid output, 0=ignore UP, 1=full UP) |
| `context` | (B, 300, 192) | Cross-attention output before gating |

### Inference-mode forward (no anchors)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `e_win` | (B, 1568) | Window embedding = [e_F(384) \| e_FU(1152) \| e_TE(32)] |

### Loss components

| Loss | Description |
|------|-------------|
| `L_fus` | Huber loss on fused forecasts (horizon-weighted) |
| `L_delta` | Huber loss on temporal differences of fused forecasts |
| `L_self` | Huber loss on self-only forecasts |
| `L_te` | Huber loss on TE residuals vs (target - sg(self)) |
| `L_kl` | KL(posterior \|\| prior) on TE latents |

---

## Complete Analysis Inventory

### Category 1: Per-Sample Diagnostics

Extended version of the training diagnostic callback. Each sample produces a multi-row figure matching the publication style.

| Fig ID | Title | Description | Data Used |
|--------|-------|-------------|-----------|
| 1.1 | Raw Signals | Twin-axis: FHR (bpm) + UP (mmHg) vs time | `fhr`, `up` raw signals |
| 1.2 | FHR Scattering Heatmap | bwr heatmap (43 channels x 300 steps) | `Y` (fhr_st) |
| 1.3 | UP Scattering Heatmap | bwr heatmap (43 channels x 300 steps) | `U` (up_st) |
| 1.4 | H_F Encoder States | bwr heatmap (192 dims x 300 steps) | `H_F` |
| 1.5 | H_U Encoder States | bwr heatmap (192 dims x 300 steps) | `H_U` |
| 1.6 | H_FU Fused Encoder States | bwr heatmap (192 dims x 300 steps) | `H_FU` |
| 1.7 | PCA of H_F | Top 3 principal components with variance explained | `H_F` |
| 1.8 | PCA of H_FU | Top 3 principal components with variance explained | `H_FU` |
| 1.9 | Fusion Contribution | L2 dist + relative change (H_FU - H_F) with anchors | `H_F`, `H_FU`, anchors |
| 1.10 | Gate Activation | Mean +/- SD gate over time, anchors marked | `gate` |
| 1.11 | Forecast at All Horizons | GT vs self vs fused vs TE, one subplot per h={8,15,30} | All Y_hat, targets |
| 1.12 | MAE by Horizon | Grouped bar: 3 heads x 3 horizons | All Y_hat, targets |
| 1.13 | Per-Channel MAE Heatmap | 43 channels x 3 horizons for fused head | `Y_hat_fus`, targets |
| 1.14 | TE Latent: Posterior vs Prior | Split heatmap (16 dims x K anchors) | `mu_post`, `mu_prior` |
| 1.15 | KL per Anchor | Bar chart with beta annotation | KL computation |
| 1.16 | TE Residual Magnitude | Bar chart: \|\|R_hat\|\| per horizon per anchor | `R_hat` |
| 1.17 | Window Embedding Components | Stacked magnitude bars: e_F, e_FU, e_TE | `e_win` |

**Output:** `per_sample_diagnostics/{class}/{guid}_{epoch}.pdf` (10 samples per class)

---

### Category 2: Forecasting Performance

Aggregate forecasting metrics across the full test set, comparing all 3 heads, 3 horizons, and 3 classes.

| Fig ID | Title | Description |
|--------|-------|-------------|
| 2.1 | MAE Distributions | 9 histograms: 3 heads x 3 horizons, all samples pooled |
| 2.2 | MAE Box Plots by Class | Grouped box: x=horizon, groups=heads, panels=class |
| 2.3 | Fused vs Self-Only Scatter | x=MAE_self, y=MAE_fused, color=class, diagonal line |
| 2.4 | TE vs Self-Only Scatter | x=MAE_self, y=MAE_te, color=class |
| 2.5 | Improvement Ratio Distribution | Histograms of (MAE_self - MAE_fused)/MAE_self per class |
| 2.6 | Per-Channel Error Heatmap | x=channel(0..42), y=horizon, value=MAE, panels=head |
| 2.7 | Error vs Anchor Position | MAE vs anchor index, lines per head, panels per horizon |
| 2.8 | Error vs Time-to-Delivery | MAE vs hours-before-birth, lines per class |
| 2.9 | Loss Decomposition | Stacked bars: L_fus, L_delta, L_self, L_te, L_kl per class |
| 2.10 | Head Comparison Summary | Radar chart: 3 heads across metrics per class |

**Output:** `forecasting/{plots + forecast_metrics.csv + summary.json}` (CSV has full segment metadata: guid, epoch, cs_label, bg_label, subgroup, tlo_seconds)

---

### Category 3: TE / Coupling Analysis

Analyses specific to the transfer entropy latent and the UP->FHR coupling mechanism.

| Fig ID | Title | Description |
|--------|-------|-------------|
| 3.1 | KL Distribution per Class | Histograms of mean-per-sample KL, separated by class |
| 3.2 | Per-Dimension KL | Bar chart: mean KL per latent dim (16 bars), grouped by class |
| 3.3 | KL vs Anchor Position | Line: mean KL at each anchor position, per class |
| 3.4 | Posterior vs Prior Scatter | Per-dim scatter: mu_post vs mu_prior, colored by class |
| 3.5 | TE Residual Magnitude | Distribution of \|\|R_hat\|\| per horizon per class |
| 3.6 | TE Latent PCA/UMAP | 2D/3D projection of mu_post across samples, colored by class |
| 3.7 | Posterior Variance per Class | Distribution of exp(logvar_post), per class |
| 3.8 | TE Latent Correlation Matrix | Heatmap of correlations between 16 dims, per class |
| 3.9 | KL Information Content | Sorted per-dimension KL cumulative plot (information content) |

**Output:** `te_coupling/{plots + te_anchor_data.csv + te_segment_data.csv}`

---

### Category 4: Representation & Embedding Analysis

Analysis of the learned representations: encoder states, gate activations, fusion contributions, and window embeddings.

| Fig ID | Title | Description |
|--------|-------|-------------|
| 4.1 | e_win PCA (2D + 3D) | PCA of full 1568-dim embedding, colored by class |
| 4.2 | e_F PCA | PCA of FHR component (384-dim), colored by class |
| 4.3 | e_FU PCA | PCA of fused component (1152-dim), colored by class |
| 4.4 | e_TE PCA | PCA of TE component (32-dim), colored by class |
| 4.5 | e_win UMAP (2D + 3D) | UMAP of full embedding, colored by class |
| 4.6 | e_win t-SNE | t-SNE of full embedding, colored by class |
| 4.7 | Embedding Norm Distributions | Violin: \|e_F\|, \|e_FU\|, \|e_TE\| per class |
| 4.8 | Fusion Contribution per Class | Histogram of mean \|\|H_FU - H_F\|\| per sample per class |
| 4.9 | Gate Activation Distribution | Histogram of mean gate per sample per class |
| 4.10 | Gate Temporal Profile per Class | Mean gate over time (x=step), lines per class, +/- SEM |
| 4.11 | Linear Separability | Bar: LDA/LogReg accuracy on e_win, e_F, e_FU, e_TE |
| 4.12 | Clustering Quality | Silhouette, Davies-Bouldin, Calinski-Harabasz scores |
| 4.13 | Explained Variance Spectrum | Cumulative PCA variance for e_win, e_F, e_FU, e_TE |

**Output:** `representation/{plots + embeddings.npz + embedding_metadata.csv}`

---

### Category 5: Temporal / Trajectory Analysis

Per-patient (GUID) trajectory analysis over time-to-delivery. Requires GUID-based DataLoader where each batch = one patient's epochs sorted by time.

| Fig ID | Title | Description |
|--------|-------|-------------|
| 5.1 | e_win Trajectory per GUID | PCA projection as trajectory colored by time-to-delivery |
| 5.2 | KL Trajectory per GUID | Line: mean KL vs hours-before-delivery per patient |
| 5.3 | Class-Mean KL Trajectory | Averaged KL with confidence bands, per class |
| 5.4 | Gate Trajectory per GUID | Mean gate activation vs time-to-delivery |
| 5.5 | Class-Mean Gate Trajectory | Averaged gate with confidence bands, per class |
| 5.6 | Forecast Quality Trajectory | MAE (fused, h=30) vs time-to-delivery per class |
| 5.7 | Embedding Drift Rate | Rate of e_win change between consecutive epochs, by class |
| 5.8 | TE Residual Trajectory | \|\|R_hat\|\| vs time-to-delivery per class |
| 5.9 | Changepoint Detection | Changepoints on KL and e_win trajectories |
| 5.10 | 3D Trajectory Visualization | 3D PCA of e_win with time as trajectory parameter |
| 5.11 | Cross-Class Trajectory Comparison | Frechet distance / MMD at matched time bins |

**Output:** `trajectory/{per_guid_plots/ + class_summary_plots/ + trajectory_data.csv}`

---

### 

Statistical tests and classification experiments comparing the three outcome classes.

| Fig ID | Title | Description |
|--------|-------|-------------|
| 6.1 | Metric Summary Table | LaTeX table: mean +/- std of all metrics per class |
| 6.2 | Per-Class MAE Comparison | Multi-panel bar with error bars, per horizon |
| 6.3 | Statistical Significance | Kruskal-Wallis + pairwise Mann-Whitney for all metrics |
| 6.4 | Effect Size Heatmap | Cohen's d for all pairwise class comparisons |
| 6.5 | ROC Curves | Logistic regression on e_win, e_F, e_FU, e_TE (HEALTHY vs each) |
| 6.6 | Confusion Matrices | Best embedding-based classifier |
| 6.7 | Per-Subgroup Breakdown | All metrics broken down by 8 subgroups (if labels available) |
| 6.8 | Cross-Class MAE Histograms | Overlaid histograms (3 classes on same axes), one subplot per head x horizon (3x3 grid) |
| 6.9 | Cross-Class VAF Histograms | VAF = 1 - var(target-pred)/var(target) per head, overlaid by class (3 subplots) |
| 6.10 | Cross-Class SNR Histograms | SNR = 10*log10(var(target)/var(error)) per head, overlaid by class (3 subplots) |
| 6.11 | Cross-Class MSE Histograms | MSE per head x horizon, overlaid by class (3x3 grid) |
| 6.12 | Cross-Class Loss Component Histograms | Per-sample L_fus, L_delta, L_self, L_te, L_kl, overlaid by class (5 subplots) |
| 6.13 | Cross-Class KL Histograms | Per-sample KL divergence overlaid by class (complement to 3.1 which shows separated panels) |

**Output:** `cross_class/{plots + statistical_tests.csv + classification_results.json}`

---

### Category 7: Dataset Statistics

Characterization of the test data before any model inference.

| Fig ID | Title | Description |
|--------|-------|-------------|
| 7.1 | Sample Count per Class | Bar chart of total epochs per class |
| 7.2 | GUIDs per Class | Bar chart of unique patients per class |
| 7.3 | Epochs per GUID Distribution | Histograms per class |
| 7.4 | Time-to-Delivery Distribution | Histograms of epoch values per class |
| 7.5 | ST Coefficient Statistics | Per-channel mean/std of FHR-ST and UP-ST per class |

**Output:** `dataset_stats/{plots + metadata.csv + summary.json}`

---

## Module Architecture

```
model/transformer/tr_testing/
    __init__.py                       # Public API exports
    base.py                           # TransformerTestRunner dataclass
    collectors.py                     # Data collection (iterate loader, extract tensors)
    metrics.py                        # Pure metric computation functions
    style.py                          # Publication style (extracted from plotting_callback.py)
    visualizers.py                    # All matplotlib plotting functions
    run_tests.py                      # Main entry point + CLI
    
    analyses/
        __init__.py                   # run_all_analyses() + exports
        per_sample_diagnostics.py     # Category 1
        forecasting.py                # Category 2
        te_coupling.py                # Category 3
        representation.py             # Category 4
        trajectory.py                 # Category 5
        cross_class.py                # Category 6
        dataset_stats.py              # Category 7
```

---

## Module Specifications

### `style.py` — Publication Style Constants

Extracted from `model/transformer/training/plotting_callback.py` lines 30-139.

```python
# Color palette
COLOR_BLUE       = "#3F72AF"
COLOR_ORANGE     = "#FFB200"
COLOR_GREEN      = "#59e467"
COLOR_SKY        = "#00ADB5"
COLOR_PURPLE     = "#7553c4"
COLOR_VERMILLION = "#EB5B00"
COLOR_GRAY       = "#393E46"
COLOR_BLACK      = "#1c2025"
COLOR_LIGHT_GRAY = "#EEEEEE"
COLOR_SAGE       = "#9DC08B"

SAVE_DPI = 600  # Higher than training (300) for publication quality

# Class-specific colors (consistent across all plots)
# Default mapping for known classes; unknown keys get assigned from the
# remaining palette automatically so the pipeline works with any class names.
CLASS_COLORS_DEFAULT = {
    "healthy":  COLOR_BLUE,
    "acidosis": COLOR_ORANGE,
    "hie":      COLOR_VERMILLION,
}
_PALETTE = [COLOR_BLUE, COLOR_ORANGE, COLOR_VERMILLION, COLOR_GREEN,
            COLOR_SKY, COLOR_PURPLE, COLOR_SAGE]

def get_class_colors(class_names: list[str]) -> dict[str, str]:
    """Return a color mapping for any set of class names."""
    # Uses defaults where available, assigns from palette for unknowns

def apply_publication_style() -> None
def style_axes(ax, *, grid="major") -> None
def add_colorbar(fig, mappable, ax, *, label=None)
def heatmap(ax, data, *, cmap="bwr", origin="upper", title="", ...)
```

All style functions are identical to `plotting_callback.py` to ensure visual consistency.

---

### `base.py` — TransformerTestRunner

```python
@dataclass
class TransformerTestRunner:
    model: CausalMultimodalTransformer
    config: TransformerConfig
    device: torch.device
    output_dir: Path
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, output_dir, device=None) -> "TransformerTestRunner"
        # Loads model using train.graph_models_utils (per CLAUDE.md rules)
        # Extracts TransformerConfig from checkpoint hparams
    
    @contextmanager
    def inference_mode(self)
        # Sets eval + torch.inference_mode()
    
    def iter_batches(self, loader, max_samples=None)
        # Yields batches with fhr_st, up_st, fhr, up on device; guid, epoch on CPU
    
    def forward_with_anchors(self, Y, U)
        # Training-mode forward with eval-grid anchors
        # Returns: outputs dict (all 3 heads, TE latents, H_F, H_FU)
    
    def forward_for_embedding(self, Y, U)
        # Inference-mode forward (no anchors)
        # Returns: e_win (B, 1568)
    
    def extract_intermediates(self, Y, U)
        # Partial forward: stems + encoders + fusion + gate
        # Returns: H_F, H_U, H_FU, gate, context (like plotting_callback lines 636-643)
    
    def ensure_dir(self, subdir) -> Path
```

---

### `collectors.py` — Data Collection

Each collector streams through the dataloader batch-by-batch, accumulating results.
All collectors preserve **full segment identification** in every row so that the final
CSVs are self-contained and traceable back to the exact 20-minute segment.

#### Common metadata columns (present in every CSV)

| Column | Source | Description |
|--------|--------|-------------|
| `guid` | `batch.guid` | Patient/recording unique identifier |
| `epoch` | `batch.epoch` | Domain start in seconds before delivery (negative = before birth). This is the min_domain_start of the 20-min segment. |
| `epoch_minutes` | computed | `epoch / 60` for readability |
| `epoch_hours` | computed | `epoch / 3600` for readability |
| `class_label` | argument | "healthy", "acidosis", or "hie" (from the loader key) |
| `cs_label` | `batch.cs_label` | Caesarean section label (0 or 1) |
| `bg_label` | `batch.bg_label` | Blood gas label (0 or 1) |
| `subgroup` | derived | Derived from source HDF5 filename (e.g. "acidosis_cs") |
| `tlo_seconds` | `batch.time_from_labor_onset` | Time from labour onset in seconds (NaN if unavailable) |

#### Collector functions

```python
def collect_forecast_metrics(runner, loader, class_label, max_samples=None) -> pd.DataFrame
    # One row per (segment, head, horizon).
    # Columns: [common metadata] + head, horizon, mae, mse, vaf, snr, huber_loss
    #   mae: Mean Absolute Error averaged across eval-grid anchors
    #   mse: Mean Squared Error averaged across eval-grid anchors
    #   vaf: Variance Accounted For = 1 - var(target - pred) / var(target)
    #   snr: Signal-to-Noise Ratio = 10 * log10(var(target) / var(error))
    #   huber_loss: Huber loss (delta=1.0) averaged across anchors
    # All metrics averaged across all eval-grid anchors within the segment.

def collect_loss_components(runner, loader, class_label, max_samples=None) -> pd.DataFrame
    # One row per segment. Per-sample loss decomposition.
    # Columns: [common metadata] + L_fus, L_delta, L_self, L_te, L_kl, total_loss

def collect_te_latent_data(runner, loader, class_label, max_samples=None) -> tuple[pd.DataFrame, pd.DataFrame]
    # Returns TWO DataFrames:
    #
    # 1) ANCHOR-LEVEL DataFrame (te_anchor_data):
    #    One row per (segment, anchor). For fine-grained analysis of TE
    #    variation within the 20-minute window.
    #    Columns: [common metadata] + anchor_idx, anchor_timestep,
    #             kl_total, kl_dim_0..15,
    #             mu_post_0..15, mu_prior_0..15,
    #             logvar_post_0..15, logvar_prior_0..15,
    #             residual_norm_h8, residual_norm_h15, residual_norm_h30
    #
    # 2) SEGMENT-LEVEL DataFrame (te_segment_data):
    #    One row per 20-minute segment. Aggregated across all eval-grid
    #    anchors (~16 per segment) for trajectory/comparison analysis.
    #    Columns: [common metadata] +
    #             n_anchors,
    #             kl_mean, kl_max, kl_min, kl_std,
    #             kl_dim_mean_0..15, kl_dim_max_0..15, kl_dim_min_0..15,
    #             mu_post_mean_0..15, mu_post_max_0..15, mu_post_min_0..15,
    #             mu_prior_mean_0..15, mu_prior_max_0..15, mu_prior_min_0..15,
    #             logvar_post_mean_0..15,
    #             logvar_prior_mean_0..15,
    #             residual_norm_mean_h8, residual_norm_max_h8, residual_norm_min_h8,
    #             residual_norm_mean_h15, residual_norm_max_h15, residual_norm_min_h15,
    #             residual_norm_mean_h30, residual_norm_max_h30, residual_norm_min_h30

def collect_embeddings(runner, loader, class_label, max_samples=None) -> dict
    # Returns: {
    #   "e_win": (N, 1568), "e_F": (N, 384), "e_FU": (N, 1152), "e_TE": (N, 32),
    #   "metadata": DataFrame with [common metadata]
    # }

def collect_gate_and_fusion(runner, loader, class_label, max_samples=None) -> pd.DataFrame
    # One row per segment.
    # Columns: [common metadata] + mean_gate, std_gate, min_gate, max_gate,
    #          gate_temporal_profile (saved as 300 columns: gate_t000..gate_t299),
    #          mean_fusion_dist, max_fusion_dist, relative_fusion_mean

def collect_full_sample_data(runner, loader, class_label, n_samples=10) -> list[dict]
    # Full forward + intermediates for selected samples.
    # Returns list of dicts with ALL tensors needed for per-sample diagnostics
    # plus all common metadata fields.
```

#### Output CSV files

| File | Level | Description |
|------|-------|-------------|
| `forecast_metrics.csv` | segment x head x horizon | MAE/MSE/VAF/SNR/Huber for every head and horizon |
| `loss_components.csv` | segment | Per-sample L_fus, L_delta, L_self, L_te, L_kl |
| `te_anchor_data.csv` | segment x anchor | Per-anchor TE latent values (fine-grained) |
| `te_segment_data.csv` | segment | Aggregated TE stats (mean/max/min/std across anchors) |
| `embedding_metadata.csv` | segment | Metadata for each embedding vector in embeddings.npz |
| `gate_fusion_data.csv` | segment | Gate and fusion contribution stats per segment |

Every CSV row can be uniquely identified by `(guid, epoch)` and traced to its exact 20-minute segment, class, subgroup, and temporal position relative to delivery and labour onset.

---

### `metrics.py` — Pure Computation Functions

```python
def compute_per_anchor_mae(Y_hat, Y, anchor_indices, config) -> dict
    # {head_name: {horizon: (B*K,) array of MAE}}

def compute_per_anchor_mse(Y_hat, Y, anchor_indices, config) -> dict
    # {head_name: {horizon: (B*K,) array of MSE}}

def compute_per_anchor_vaf(Y_hat, Y, anchor_indices, config) -> dict
    # VAF = 1 - var(target - pred) / var(target), per anchor
    # {head_name: {horizon: (B*K,) array of VAF}}

def compute_per_anchor_snr(Y_hat, Y, anchor_indices, config) -> dict
    # SNR = 10 * log10(var(target) / var(target - pred)), per anchor
    # {head_name: {horizon: (B*K,) array of SNR in dB}}

def compute_kl_per_anchor(mu_post, logvar_post, mu_prior, logvar_prior) -> Tensor
    # (B*K,) total KL per anchor

def compute_kl_per_dimension(mu_post, logvar_post, mu_prior, logvar_prior) -> Tensor
    # (B*K, d_z) per-dimension KL

def compute_te_residual_norm(R_hat) -> dict
    # {horizon: (B*K,) L2 norm of residual}

def compute_fusion_contribution(H_FU, H_F) -> tuple[Tensor, Tensor]
    # (B, T) L2 distance and (B, T) relative change

def compute_gate_statistics(gate) -> dict
    # {mean: (B,), std: (B,), min: (B,), max: (B,), temporal_profile: (B, T)}

def compute_loss_components(outputs, Y, anchor_indices, config, beta=0.0) -> dict
    # Per-sample loss decomposition using CausalTransformerLoss
    # {L_fus: (B,), L_delta: (B,), L_self: (B,), L_te: (B,), L_kl: (B,), total: (B,)}

def extract_targets(Y, anchor_indices, horizon, guard_gap) -> Tensor
    # (B*K, h, d_f) ground truth targets for each anchor
```

---

### `visualizers.py` — All Plot Functions

Every function: takes numpy/DataFrame data + output path, saves figure, returns path. Uses `style.py` for all styling.

```python
# Category 1: Per-sample
def plot_sample_diagnostic(sample_data, output_path, config) -> Path

# Category 2: Forecasting
def plot_mae_histograms(metrics_df, output_dir) -> Path
def plot_mae_boxplots_by_class(metrics_df, output_dir) -> Path
def plot_head_comparison_scatter(metrics_df, output_dir, head_x, head_y) -> Path
def plot_improvement_distribution(metrics_df, output_dir) -> Path
def plot_channel_error_heatmap(channel_errors, output_dir) -> Path
def plot_error_vs_anchor(metrics_df, output_dir) -> Path
def plot_error_vs_time(metrics_df, output_dir) -> Path
def plot_loss_decomposition(loss_data, output_dir) -> Path
def plot_head_radar(metrics_df, output_dir) -> Path

# Category 3: TE coupling
def plot_kl_distributions(te_df, output_dir) -> Path
def plot_kl_per_dimension(te_df, output_dir) -> Path
def plot_kl_vs_anchor(te_df, output_dir) -> Path
def plot_posterior_vs_prior(te_df, output_dir) -> Path
def plot_te_residual_analysis(te_df, output_dir) -> Path
def plot_te_latent_projection(te_data, labels, output_dir, method="pca") -> Path
def plot_posterior_variance(te_df, output_dir) -> Path
def plot_te_correlation_matrix(te_df, output_dir) -> Path

# Category 4: Representation
def plot_embedding_projection(embeddings, labels, output_dir, method, component) -> Path
def plot_embedding_norms(embeddings, labels, output_dir) -> Path
def plot_fusion_distribution(fusion_df, output_dir) -> Path
def plot_gate_distribution(gate_df, output_dir) -> Path
def plot_gate_temporal_profile(gate_df, output_dir) -> Path
def plot_linear_separability(scores, output_dir) -> Path
def plot_clustering_quality(scores, output_dir) -> Path
def plot_variance_spectrum(embeddings, output_dir) -> Path

# Category 5: Trajectory
def plot_guid_trajectory(guid_data, output_dir) -> Path
def plot_class_mean_trajectory(trajectory_data, output_dir, metric) -> Path
def plot_embedding_drift(drift_data, output_dir) -> Path
def plot_changepoints(trajectory_data, output_dir) -> Path
def plot_3d_trajectory(trajectory_data, output_dir) -> Path
def plot_trajectory_comparison(class_trajectories, output_dir) -> Path

# Category 6: Cross-class
def plot_metric_summary_table(results, output_dir) -> Path
def plot_class_mae_comparison(metrics_df, output_dir) -> Path
def plot_significance_heatmap(test_results, output_dir) -> Path
def plot_effect_size_heatmap(effect_sizes, output_dir) -> Path
def plot_roc_curves(classification_results, output_dir) -> Path
def plot_confusion_matrices(classification_results, output_dir) -> Path
def plot_cross_class_mae_histograms(metrics_df, output_dir) -> Path       # 6.8: 3x3 grid (head x horizon)
def plot_cross_class_vaf_histograms(metrics_df, output_dir) -> Path       # 6.9: 3 subplots (one per head)
def plot_cross_class_snr_histograms(metrics_df, output_dir) -> Path       # 6.10: 3 subplots (one per head)
def plot_cross_class_mse_histograms(metrics_df, output_dir) -> Path       # 6.11: 3x3 grid (head x horizon)
def plot_cross_class_loss_histograms(loss_df, output_dir) -> Path         # 6.12: 5 subplots (one per loss component)
def plot_cross_class_kl_histograms(te_df, output_dir) -> Path             # 6.13: single overlaid histogram

# Category 7: Dataset stats
def plot_dataset_overview(stats, output_dir) -> Path
def plot_time_distribution(stats, output_dir) -> Path
def plot_st_coefficient_stats(stats, output_dir) -> Path
```

---

### `analyses/` — High-Level Analysis Runners

Each module follows this pattern:

```python
def run_<category>_analysis(
    runner: TransformerTestRunner,
    loaders: dict[str, DataLoader],  # {"healthy": ..., "acidosis": ..., "hie": ...}
    output_dir: Path,
    max_samples: int = None,
    **kwargs,
) -> dict[str, Any]:
    """Run <category> analysis for all classes.
    
    Returns dict with summary statistics and paths to saved figures.
    """
```

#### `analyses/__init__.py`

```python
def run_all_analyses(
    runner: TransformerTestRunner,
    class_loaders: dict[str, DataLoader],
    guid_loaders: dict[str, DataLoader],   # For trajectory analysis
    output_dir: Path,
    max_samples: int = None,
    skip_trajectory: bool = False,
    skip_per_sample: bool = False,
    n_diagnostic_samples: int = 10,
) -> dict[str, Any]
```

Execution order:
1. `run_dataset_stats_analysis` — no model needed
2. `run_forecasting_analysis` — per class, then merge
3. `run_te_coupling_analysis` — per class, then merge
4. `run_representation_analysis` — all classes together
5. `run_trajectory_analysis` — per class, GUID loaders
6. `run_cross_class_analysis` — uses merged results from 2-5; gracefully skips statistical tests / ROC / confusion when only 1 class provided (logs warning, still produces histograms and summary table)
7. `run_per_sample_diagnostics` — selected samples from each class

---

### `run_tests.py` — Main Entry Point

```python
def run_full_test_pipeline(
    checkpoint_path: str,
    class_data_paths: dict[str, list[str]],
    output_dir: str = None,
    stats_path: str = None,
    device: str = None,
    max_samples: int = None,
    batch_size: int = 64,
    num_workers: int = 4,
    skip_trajectory: bool = False,
    skip_per_sample: bool = False,
    n_diagnostic_samples: int = 10,
    min_epochs_per_guid: int = 5,
) -> dict[str, Any]:
    """Run the complete testing and analysis pipeline.
    
    Args:
        checkpoint_path: Path to trained model .ckpt file.
        class_data_paths: Dict mapping class names to lists of HDF5 file paths.
            Keys must be "healthy", "acidosis", "hie".
        output_dir: Where to save results. Defaults to timestamped dir.
        stats_path: Path to normalization statistics .pt file.
        device: "cuda", "cpu", or None (auto-detect).
        max_samples: Limit per class (None = all).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        skip_trajectory: Skip trajectory analysis (requires GUID loaders).
        skip_per_sample: Skip per-sample diagnostics (slow).
        n_diagnostic_samples: Samples per class for per-sample diagnostics.
        min_epochs_per_guid: Minimum epochs for GUID trajectory analysis.
    
    Returns:
        Summary dict with all results and figure paths.
    """
```

Flow:
1. Create `TransformerTestRunner.from_checkpoint(checkpoint_path, ...)`
2. Create 3 standard DataLoaders (one per class) with `create_optimized_dataloader`
3. Create 3 GUID DataLoaders (one per class) with `build_guid_filtered_dataloader`
4. Call `run_all_analyses(runner, class_loaders, guid_loaders, output_dir, ...)`
5. Save `test_summary.json`
6. Return results dict

Also provides:

```python
def quick_test(checkpoint_path, class_data_paths, ...) -> dict:
    """Fast validation with limited samples, no trajectory, no per-sample."""
```

---

## Output Directory Structure

```
{output_dir}/
    test_summary.json
    
    dataset_stats/
        sample_counts.pdf
        guids_per_class.pdf
        epochs_per_guid.pdf
        time_distribution.pdf
        st_coefficient_stats.pdf
        metadata.csv
        summary.json
    
    forecasting/
        mae_histograms.pdf
        mae_boxplots_by_class.pdf
        fused_vs_self_scatter.pdf
        te_vs_self_scatter.pdf
        improvement_distribution.pdf
        channel_error_heatmap.pdf
        error_vs_anchor.pdf
        error_vs_time.pdf
        loss_decomposition.pdf
        head_radar.pdf
        forecast_metrics.csv             # MAE, MSE, VAF, SNR, Huber per segment x head x horizon
        loss_components.csv              # Per-sample L_fus, L_delta, L_self, L_te, L_kl
        summary.json
    
    te_coupling/
        kl_distributions.pdf
        kl_per_dimension.pdf
        kl_vs_anchor.pdf
        posterior_vs_prior.pdf
        te_residual_analysis.pdf
        te_latent_pca.pdf
        te_latent_umap.pdf
        posterior_variance.pdf
        te_correlation_matrix.pdf
        te_anchor_data.csv          # Per-anchor level (segment x anchor)
        te_segment_data.csv         # Per-segment level (aggregated across anchors)
    
    representation/
        e_win_pca_2d.pdf
        e_win_pca_3d.pdf
        e_F_pca.pdf
        e_FU_pca.pdf
        e_TE_pca.pdf
        e_win_umap_2d.pdf
        e_win_umap_3d.pdf
        e_win_tsne.pdf
        embedding_norms.pdf
        fusion_distribution.pdf
        gate_distribution.pdf
        gate_temporal_profile.pdf
        linear_separability.pdf
        clustering_quality.pdf
        variance_spectrum.pdf
        embeddings.npz
        embedding_metadata.csv
    
    trajectory/
        per_guid/
            {guid}_trajectory.pdf
            {guid}_kl_trajectory.pdf
            ...
        class_mean_kl.pdf
        class_mean_gate.pdf
        forecast_quality_trajectory.pdf
        embedding_drift.pdf
        te_residual_trajectory.pdf
        changepoints/
            ...
        trajectory_3d/
            ...
        trajectory_comparison.pdf
        trajectory_data.csv
    
    cross_class/
        metric_summary_table.pdf
        class_mae_comparison.pdf
        significance_heatmap.pdf
        effect_size_heatmap.pdf
        roc_curves.pdf
        confusion_matrices.pdf
        subgroup_breakdown.pdf
        cross_class_mae_histograms.pdf   # 6.8: overlaid 3-class MAE (3x3 grid)
        cross_class_vaf_histograms.pdf   # 6.9: overlaid 3-class VAF (3 subplots)
        cross_class_snr_histograms.pdf   # 6.10: overlaid 3-class SNR (3 subplots)
        cross_class_mse_histograms.pdf   # 6.11: overlaid 3-class MSE (3x3 grid)
        cross_class_loss_histograms.pdf  # 6.12: overlaid 3-class loss components (5 subplots)
        cross_class_kl_histograms.pdf    # 6.13: overlaid 3-class KL divergence
        statistical_tests.csv
        classification_results.json
    
    per_sample_diagnostics/
        healthy/
            {guid}_{epoch}.pdf
            ...
        acidosis/
            {guid}_{epoch}.pdf
            ...
        hie/
            {guid}_{epoch}.pdf
            ...
```

---

## Implementation Order

### Phase 1: Foundation (no dependencies)
1. `style.py` — extract from plotting_callback.py
2. `metrics.py` — pure functions
3. `__init__.py` — package init

### Phase 2: Core Infrastructure
4. `base.py` — TransformerTestRunner (depends on: style, model package)
5. `collectors.py` — data collection (depends on: base, metrics)

### Phase 3: Visualization
6. `visualizers.py` — all plotting (depends on: style)

### Phase 4: Analysis Modules
7. `analyses/__init__.py`
8. `analyses/dataset_stats.py` — no model needed
9. `analyses/per_sample_diagnostics.py`
10. `analyses/forecasting.py`
11. `analyses/te_coupling.py`
12. `analyses/representation.py`
13. `analyses/trajectory.py`
14. `analyses/cross_class.py`

### Phase 5: Entry Point
15. `run_tests.py` — main orchestrator

---

## Key Implementation Notes

1. **Gate extraction requires partial forward** (not returned by `model.forward()`):
   ```python
   F_out = model.fhr_stem(Y)
   S_out = model.up_stem(U)
   H_F = model.fhr_encoder(F_out)
   H_U = model.up_encoder(S_out)
   context = model.fusion.cross_attn(target=H_F, source=H_U)
   gate = torch.sigmoid(model.fusion.gate_proj(torch.cat([H_F, context], dim=-1)))
   ```

2. **Memory management**: H_F, H_U, H_FU are (B, 300, 192). For aggregate analyses, stream batch-by-batch and keep only summary stats. Save full tensors only for per-sample diagnostics.

3. **Checkpoint loading**: Use `train.graph_models_utils.load_checkpoint_strict` per CLAUDE.md rules.

4. **Eval anchor grid**: Use `sample_anchors(Y, U, config, training=False)` which returns a fixed grid every 15 steps for deterministic evaluation.

5. **GUID-based DataLoader**: Required for trajectory analysis. Use `hdf5_dataset.hdf5_dataset.build_guid_filtered_dataloader()` where each batch = one patient's complete epochs.

6. **Three-class input**: Each class gets its own DataLoader. Per-class results are computed independently, then merged for cross-class analysis with a `class_label` column.

---

## Verification Plan

1. **Smoke test**: Run `quick_test()` on a small subset (10 samples per class) to verify all modules load and produce output without errors
2. **Style verification**: Compare a per-sample diagnostic figure side-by-side with a training callback figure to confirm visual consistency
3. **Metric sanity**: Verify that test set losses match the validation loss reported during training (within tolerance)
4. **Embedding sanity**: Check that e_win dimensionality is 1568 and component boundaries (384, 1152, 32) match
5. **Full pipeline**: Run on full test set with all analyses enabled
