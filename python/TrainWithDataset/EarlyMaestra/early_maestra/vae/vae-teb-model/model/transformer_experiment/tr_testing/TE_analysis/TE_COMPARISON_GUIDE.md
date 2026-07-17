# Model TE vs Empirical TE Comparison Pipeline

## Purpose

This module compares the **model-based Transfer Entropy** (learned by the
Causal Multimodal Forecasting Transformer) with **empirical Transfer Entropy**
(computed externally via IDTxl on raw FHR/UP time series).  The goal is to
validate whether the transformer's learned TE latent space captures the same
UP-to-FHR coupling that classical statistical TE estimation measures.

### What is compared

| Side | Source | Quantity | Interpretation |
|------|--------|----------|----------------|
| **Model** | `te_segment_data.csv` | KL(posterior ‖ prior) per anchor, aggregated to segment level | Variational upper bound on *I(U_past ; Y_future \| Y_past)* — extra information the latent captures when UP is available vs FHR alone |
| **Model** | `te_segment_data.csv` | TE residual norm (per horizon) | L2 magnitude of the predicted incremental UP contribution |
| **Model** | `te_segment_data.csv` | TE forecast gain (`MSE_self − MSE_te`) | Actual prediction error improvement from adding the TE latent — how much knowing UP reduces forecast error |
| **Model** | `te_segment_data.csv` | TE relative gain (`1 − MSE_te / MSE_self`) | Fractional improvement in forecast accuracy from UP — comparable to variance explained |
| **Empirical** | `te_record_epoch.csv` | `ite_valid` (IDTxl) | Classical conditional mutual information UP→FHR |

All model quantities approximate the same theoretical object — *I(U_past ; Y_future | Y_past)* — but in different spaces:

- **KL divergence**: measures information gain in *latent* space (variational bound)
- **TE forecast gain**: measures prediction improvement in *output* space (operational TE)
- **Residual norm**: measures the magnitude of the UP contribution in *output* space

The empirical `ite_valid` estimates the same quantity through non-parametric density estimation on the raw signals.  The forecast-based gains are the most directly comparable to `ite_valid` because both measure how much UP *actually helps* predict FHR.

---

## Data

### `te_segment_data.csv` — Model TE (1 500 rows, 65 GUIDs)

Produced by `collect_te_latent_data()` in the testing pipeline.  One row per
20-minute segment.

Key columns:

| Column | Description |
|--------|-------------|
| `guid` | Patient identifier |
| `epoch` | Segment start in seconds relative to delivery (negative) |
| `class_label` | `healthy`, `acidosis`, or `hie` |
| `kl_mean` | Mean KL divergence across all 16 latent dims and K anchors |
| `kl_max` | Max KL across anchors |
| `kl_dim_mean_{0..15}` | Per-dimension KL mean (16 columns) |
| `mu_post_mean_{0..15}` | Posterior mean per dimension |
| `mu_prior_mean_{0..15}` | Conditional prior mean per dimension |
| `residual_norm_mean_h{8,15,30}` | Mean L2 norm of TE residual per horizon |
| `te_forecast_gain_mean_h{8,15,30}` | Mean MSE improvement from TE latent per horizon (`MSE_self − MSE_te`) |
| `te_forecast_gain_mean` | Cross-horizon mean forecast gain |
| `te_relative_gain_mean_h{8,15,30}` | Relative forecast improvement (`1 − MSE_te / MSE_self`) per horizon |
| `te_relative_gain_mean` | Cross-horizon mean relative gain |
| `mse_self_mean_h{8,15,30}` | Mean MSE of self-only (FHR-only) forecast per horizon |
| `mse_te_mean_h{8,15,30}` | Mean MSE of TE-augmented forecast per horizon |

### `te_record_epoch.csv` — Empirical TE (9 507 rows, 273 GUIDs)

Produced by IDTxl.  One row per 20-minute epoch.  Header has a `#` prefix
on the first column name (handled automatically by the loader).

Key columns:

| Column | Description |
|--------|-------------|
| `tracing_guid` | Patient identifier (renamed to `guid` after loading) |
| `domain_start` | Epoch start in seconds relative to delivery |
| **`ite_valid`** | **ITE after significance filtering — the only empirical measure used for comparison** |
| `omnibus_te` | Omnibus TE statistic (informational only, not used) |
| `source_te_max` | Maximum TE across lags (informational only, not used) |
| `ite` | Raw instantaneous TE (informational only, not used) |
| `ite_valid_pc` | Fraction of valid samples (informational only, not used) |

### GUID Overlap

Only **3 GUIDs** appear in both datasets, all from the acidosis class:

| GUID (truncated) | Model segments | Empirical epochs | Matched pairs |
|-------------------|---------------|-----------------|---------------|
| `02DBBF70...` | 25 | 50 | 20 |
| `F891A9EB...` | 30 | 26 | 16 |
| `F943BB03...` | 13 | 24 | 7 |
| **Total** | — | — | **43** |

---

## Pipeline Architecture

```
te_data_loader.py          te_comparison_analysis.py       te_comparison_visualizations.py
─────────────────          ──────────────────────────      ──────────────────────────────
load_model_te_data()       compute_correlation_matrix()    plot_data_quality_summary()
load_empirical_te_data()   permutation_test_correlation()  plot_correlation_heatmap()
fuzzy_time_match()    ──>  compute_within_guid_corr()  ──> plot_per_dimension_kl_heatmap()
compute_data_quality()     concordance_analysis()          plot_scatter_matrix()
                           trend_agreement_analysis()      plot_trend_agreement()
                           per_dimension_analysis()        plot_leave_one_out()
                           dtw_trajectory_similarity()     plot_bootstrap_distribution()
                           leave_one_guid_out_sens()       plot_permutation_null()
                           cluster_aware_bootstrap()       plot_mutual_information_comparison()
                           mutual_information_knn()        plot_dtw_trajectories()
                           run_full_comparison()           plot_model_guid_trajectories()
                                                           plot_empirical_guid_trajectories()
                                                           generate_summary_table()

                           te_comparison_runner.py
                           ──────────────────────
                           run_te_comparison()    ← single entry point
                           __main__               ← standalone execution
```

### Running

```bash
# From project root (use the Python environment with torch/pandas/scipy)
python -m model.transformer.tr_testing.TE_analysis.te_comparison_runner

# Or from Python
from model.transformer.tr_testing.TE_analysis import run_te_comparison
results = run_te_comparison(
    model_csv_path="model/transformer/tr_testing/TE_analysis/te_segment_data.csv",
    empirical_csv_path="model/transformer/tr_testing/TE_analysis/te_record_epoch.csv",
    output_dir="model/transformer/tr_testing/TE_analysis/results",
)
```

Output is written to `TE_analysis/results/`.

---

## Files

| File | Role |
|------|------|
| `te_data_loader.py` | Load CSVs, normalise GUIDs, fuzzy nearest-neighbour time matching, data quality report |
| `te_comparison_analysis.py` | 10 statistical analyses (all non-parametric, valid at small N) |
| `te_comparison_visualizations.py` | 13 publication-quality plots (follows `tr_testing/style.py`) |
| `te_comparison_runner.py` | Orchestrator — wires load → analyse → plot → export |
| `__init__.py` | Exports `run_te_comparison` |

---

## Time Matching

The previous VAE pipeline used **grid-rounding** (snap both sides to the
nearest 1 200 s boundary).  This implementation uses **fuzzy nearest-neighbour
matching**:

1. For each common GUID, enumerate all (model-segment, empirical-epoch)
   candidate pairs where |epoch − domain_start| ≤ 600 s (10 minutes).
2. Sort candidates by absolute time gap (smallest first).
3. Greedily assign pairs: each model row and each empirical row is used at
   most once (1-to-1 constraint).
4. The resulting `time_gap_seconds` column records the actual gap for each
   matched pair.

This avoids losing matches due to the systematic −1 s offset in IDTxl exports
and is transparent about match quality (the gap distribution is reported and
plotted).

---

## Statistical Analyses

All methods are chosen for validity at small sample sizes (N ≈ 43, 3 clusters).

### 1. Spearman / Kendall Correlation Matrix

Rank correlation between every model measure (5) and the single empirical
measure (`ite_valid`), producing a 5 × 1 vector.  Both Spearman ρ and
Kendall τ-b are computed.  Kendall τ-b handles ties properly and is more
robust at small N.

### 2. Permutation Tests

For each measure pair, 10 000 random permutations of the empirical values build
a null distribution.  The **two-sided exact p-value** is the fraction of
permuted correlations with |ρ| ≥ |ρ_observed|.  This avoids reliance on
parametric p-value approximations that break at N = 43.

### 3. Within-GUID Temporal Correlations

For each GUID with ≥ 5 matched pairs (2 of 3 qualify), Spearman ρ is computed
on the **time-ordered** values.  This tests whether the model and empirical TE
co-vary within a single patient's recording — the strongest form of evidence
for correspondence.

### 4. Concordance Analysis (Kendall τ-b)

Counts concordant, discordant, and tied pairs.  The **concordance index**
(C-index = P(concordant) / P(concordant + discordant)) gives the probability
that a randomly chosen pair of observations has the same relative ordering in
both measures.  C = 0.5 is chance.

### 5. Trend Agreement

For each pair of consecutive matched time points within a GUID, the first
differences (Δ_model, Δ_empirical) are computed.  **Sign agreement rate** is
the fraction where both increase or both decrease.  Tested against 50% chance
via a binomial test.  This is more robust than continuous correlation for small,
noisy samples.

### 6. Per-Dimension Analysis

Correlates each of the 16 individual KL dimensions (`kl_dim_mean_{0..15}`)
with `ite_valid`.  Identifies **which latent dimensions carry coupling
information**.  Uses permutation p-values (5 000 permutations per dimension).

### 7. Dynamic Time Warping (DTW)

Uses the **full per-GUID trajectories** from each dataset (not just the
matched time points), since DTW naturally handles different-length sequences.
For example, GUID `02DBBF70` uses all 25 model segments and all 50 empirical
epochs.  Both trajectories are z-scored, then DTW computes the optimal
temporal alignment and returns the cumulative distance.  Falls back to
Euclidean distance if `dtw-python` / `tslearn` are not installed.  The
**normalised DTW** (distance / max trajectory length) enables cross-GUID
comparison.

### 8. Leave-One-GUID-Out Sensitivity

With only 3 GUIDs, each patient has outsized influence.  The pooled Spearman ρ
is recomputed with each GUID removed in turn.  The **most influential GUID**
is the one whose removal changes the correlation the most.  If removing a
single GUID flips the correlation sign, the finding is not robust.

### 9. Cluster-Aware Block Bootstrap

Standard bootstrap assumes IID observations — incorrect here because
observations within the same patient are temporally correlated.  Block
bootstrap resamples at the **GUID level**: each of 5 000 iterations draws
N_GUIDs with replacement and includes all matched pairs for the selected GUIDs.

With 3 GUIDs, there are only 3³ = 27 unique resample combinations, so the
bootstrap distribution is coarse.  The resulting **wide confidence intervals**
honestly reflect the uncertainty from having only 3 patients.

### 10. Mutual Information (KSG Estimator)

Captures **non-linear** associations that rank correlations cannot.  Uses the
Kraskov–Stoegbauer–Grassberger k-nearest-neighbour estimator (via
`sklearn.feature_selection.mutual_info_regression` when available, binned
fallback otherwise).  Compared against |Spearman ρ| to detect non-linear
coupling.

---

## Figures

All plots use the project's publication style (`tr_testing/style.py`): serif
fonts, DPI 600, four-spine axes, and the standard colour palette.

### `data_quality_summary.pdf`

Three-panel infographic: GUID overlap counts (left), matching statistics table
(centre), per-GUID matched-pair bar chart annotated with mean time gap (right).
This is the first figure a reader should look at to understand the data
constraints.

### `correlation_heatmap.pdf`

Diverging blue–white–red heatmap of Spearman ρ between 5 model measures (rows)
and `ite_valid` (single column).  Cells are annotated with ρ values and
significance stars (\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001).

### `per_dimension_kl_heatmap.pdf`

Same format as above but with 16 rows (one per KL latent dimension) and
`ite_valid` as the single column.  Reveals which dimensions carry coupling
information.  Annotated with permutation-based significance stars.

### `scatter_matrix.pdf`

5 × 1 column of scatter plots.  Each panel plots one model measure (y) against
`ite_valid` (x).  Points are **coloured by GUID** (3 colours) so the reader
can visually assess whether a pattern holds across patients or is driven by
one.  Each panel shows Spearman ρ and permutation p-value, with a linear
regression trend line.

### `trend_agreement.pdf`

Per-GUID step plot showing sign agreement of consecutive temporal derivatives.
Green upward triangles = both measures moved in the same direction; red
downward triangles = opposite directions.  Summary bar at bottom reports the
overall agreement rate and binomial p-value.

### `leave_one_out.pdf`

Bar chart with 4 bars: the full-data Spearman ρ plus one bar for each
leave-one-GUID-out recomputation.  The most influential GUID is highlighted
in red.  Shows whether the signal is robust or driven by a single patient.

### `bootstrap_distribution.pdf`

Histogram of 5 000 cluster-aware (block) bootstrap Spearman ρ samples.  The
observed value is marked with a red vertical line, the 95% CI is shaded in
green, and the null reference (ρ = 0) is shown as a dashed grey line.

### `permutation_null.pdf`

Histogram of the permutation null distribution (10 000 permutations) for the
primary measure pair (kl\_mean vs ite\_valid).  The observed correlation is
marked with a red line; the tail beyond |observed| is shaded red to visualise
the p-value area.

### `mi_vs_correlation.pdf`

Grouped bar chart comparing normalised MI estimates with |Spearman ρ| for each
model–empirical measure pair.  Cases where MI is notably higher than |ρ|
suggest non-linear coupling that rank correlation misses.

### `dtw_trajectories.pdf`

Per-GUID z-scored trajectories (same as trajectory overlay) annotated with the
DTW (or Euclidean) distance and normalised distance.  Shows temporal shape
similarity independently of absolute scale.

### `model_guid_trajectories.pdf`

10-panel figure (5 × 2 grid) showing **model KL divergence over time** for 10
diverse GUIDs.  The 3 overlap GUIDs appear first with a **z-scored overlay**
of both model `kl_mean` and empirical `ite_valid` on the same axis (star
markers, `[OVERLAP]` tag showing count from each side).  The remaining 7
non-overlap GUIDs show raw `kl_mean` values, balanced across classes (acidosis
in orange, healthy in blue, HIE in red).  This merges the trajectory overlay
concept into a broader population context.

### `empirical_guid_trajectories.pdf`

10-panel figure (5 × 2 grid) showing **empirical TE (ite_valid) over time** for
10 GUIDs.  The 3 overlap GUIDs appear first with a **z-scored overlay** of
both empirical `ite_valid` and model `kl_mean` (star markers, `[OVERLAP]`
tag).  The remaining 7 non-overlap GUIDs show raw `ite_valid` values (cyan),
selected as the most data-rich recordings.  The empirical data spans much
longer windows (up to −70 hours) compared to the model's −12 hours.

### `summary_table.pdf`

Publication-ready table rendered as a matplotlib figure.  Rows = each
model–empirical measure pair.  Columns = Spearman ρ, Kendall τ, permutation p,
MI, trend agreement %, concordance index.

---

## Improvements Over Previous Implementation

The previous VAE pipeline (`model/vae_teb_prediction/testing/TE_Calculated/`)
had several limitations that this module addresses:

| Aspect | Previous (VAE) | Current (Transformer) |
|--------|----------------|----------------------|
| Time matching | Grid-rounding to 1 200 s boundaries | Fuzzy nearest-neighbour (±600 s) with 1-to-1 constraint |
| Measures compared | Single scalar KLD vs ite\_valid | 5 model measures × ite\_valid |
| Per-dimension | Not analysed | 16-dim KL heatmap reveals which dims carry coupling |
| P-values | Parametric only (unreliable at small N) | Permutation tests (exact p-values) |
| Bootstrap | IID (violates patient nesting) | GUID-level block bootstrap |
| Fisher z on Spearman | Applied (mathematically incorrect) | Not used |
| Trajectory plots | Dual-axis (visually misleading) | Z-scored single-axis overlay |
| Robustness check | None | Leave-one-GUID-out sensitivity |
| Data quality | Not reported | Comprehensive diagnostic report + infographic |
| Non-linear detection | None | Mutual information (KSG estimator) |
| Temporal shape | None | DTW distance |
| Trend direction | None | Sign-of-derivative agreement + binomial test |

---

## Output Files

Running the pipeline produces the following in `results/`:

| File | Content |
|------|---------|
| `merged_data.csv` | All 43 matched pairs with columns from both sources |
| `data_quality.json` | GUID overlap, coverage stats, per-GUID matching details |
| `summary_statistics.json` | All numerical analysis results |
| `correlation_spearman.csv` | 5 × 1 Spearman ρ (model measures vs ite\_valid) |
| `correlation_kendall.csv` | 5 × 1 Kendall τ |
| `pvalue_spearman.csv` | Parametric p-values for Spearman |
| `pvalue_kendall.csv` | Parametric p-values for Kendall |
| `per_dimension_analysis.csv` | 16 dims × ite\_valid: ρ, p-values, mean/std KL |
| `within_guid_correlations.csv` | Per-GUID Spearman ρ for primary pair |
| 13 × `.pdf` | All figures described above |

---

## Dependencies

**Required** (standard scientific stack): `numpy`, `pandas`, `scipy`,
`matplotlib`, `loguru`

**Optional** (graceful fallback):

- `dtw-python` or `tslearn` — for true DTW distance; falls back to Euclidean
- `scikit-learn` — for KSG mutual information estimator; falls back to binned MI

---

## Limitations

1. **Only 3 overlapping GUIDs** (all acidosis class).  Results cannot be
   generalised to healthy or HIE patients.
2. **43 matched pairs** still limits statistical power.  None of the correlations
   reached conventional significance (p < 0.05).
3. Block bootstrap CIs are inherently coarse with 3 GUIDs (only 27 unique
   resamples).  The wide CIs honestly reflect this limitation.
4. The third GUID (`F943BB03...`) now contributes 7 matched pairs, but it still
   remains a relatively small cluster compared with the other two GUIDs.
5. Only `ite_valid` is used from the empirical CSV.  Other columns
   (`omnibus_te`, `source_te_max`, `ite`) are informational but not used
   in the comparison, as `ite_valid` is the significance-filtered measure
   and the most reliable estimate of directed transfer entropy.
