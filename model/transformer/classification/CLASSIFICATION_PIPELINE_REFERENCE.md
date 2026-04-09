# Transformer Classification Pipeline — LLM Reference

This is the authoritative reference for the transformer-based classification pipeline. It describes every module, class, function, config key, tensor shape, and CLI command. Use this document to understand, modify, or extend the pipeline.

---

## 1. Architecture Overview

A pretrained `CausalMultimodalTransformer` (unsupervised forecasting model) produces 416-dim segment embeddings from 20-minute CTG recordings. A **time-aware GRU classifier** processes these embeddings in chronological order with time-decay gating to produce per-segment binary predictions (healthy vs unhealthy).

### Data Flow

```
Raw HDF5 (fhr_st, up_st per segment)
    |
    v
CausalMultimodalTransformer (frozen or fine-tunable)
    |
    v   Per segment:
    |   H_F (300, 192) -> pool -> s_F (192)
    |   H_FU (300, 192) -> pool -> s_FU (192)
    |   mu_post (K, 16) -> mean -> mean_TE (16)
    |                    -> std  -> std_TE (16)
    v
Segment embedding e = [s_F | s_FU | mean_TE | std_TE] = 416 dims
    |
    v   Per GUID (chronological sequence of segments):
    |   e_j (416) + delta_e_j (416) + time_embed_j (32) = 864
    |   -> input_proj (864 -> 256)
    |   -> time-decay GRU loop (hidden 256)
    |   -> output [h_j | x_j] (512)
    |   -> binary head (512 -> 1 or 2)
    v
Per-segment prediction: P(unhealthy)
```

---

## 2. File Structure

All files are in `model/transformer/classification/`.

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports, version |
| `config_classification.yaml` | Single source of truth for all settings |
| `classification_model.py` | `TimeFeatureEncoder` + `TimeAwareGRUClassifier` |
| `precompute_embeddings.py` | Pre-compute 416-dim embeddings to HDF5 |
| `classification_trainer.py` | `PlClassifier` (Lightning) + `GraphModelClassificationTrainer` + `train_fold()` |
| `kfold_classification_trainer.py` | Parallel k-fold orchestration with MLflow |
| `classification_model.md` | Mathematical architecture specification |

---

## 3. Segment Embedding (416-dim)

### Composition

```
e = [s_F(192) | s_FU(192) | mean_TE(16) | std_TE(16)]
```

| Component | Dim | Source | What it captures |
|-----------|-----|--------|------------------|
| `s_F` | 192 | `pool(H_F)` over 300 steps | Intrinsic fetal heart rate state |
| `s_FU` | 192 | `pool(H_FU)` over 300 steps | Multimodal state (FHR + UP coupling) |
| `mean_TE` | 16 | `mean(mu_post)` over K anchors | Average UP->FHR coupling level |
| `std_TE` | 16 | `std(mu_post)` over K anchors | Coupling variability (erratic = bad sign) |

### Pooling Modes

- **`pooling="mean"`** (default): `s_F = H_F.mean(dim=1)`. Deterministic, used for precomputation.
- **`pooling="attention"`**: `s_F = AttentionPool(H_F)`. Trainable, on-the-fly only.

### Dense TE Anchor Grid

The TE posterior is evaluated at a dense grid of anchor positions:

```python
grid = torch.arange(valid_anchor_start, valid_anchor_end + 1, anchor_step)
# anchor_step=5 (default): ~47 anchors
# anchor_step=15: ~16 anchors (matches inference mode)
```

H_F and H_FU are **anchor-independent** (full-sequence encoder outputs). Only the TE component depends on anchor positions.

---

## 4. Time Feature Encoding

### Raw 6-dim Feature Vector

For each segment j:

```
r_j = [tau, log(1+tau), delta_tau, log(1+delta_tau), delta, m]
```

| Feature | Formula | What it captures |
|---------|---------|------------------|
| `tau` | `TLO / 60` (minutes) | Absolute progression through labour |
| `log(1+tau)` | `log1p(tau.clamp(min=0))` | Compressed progression scale |
| `delta_tau` | `delta_t / 60` (minutes) | Time since previous segment |
| `log(1+delta_tau)` | `log1p(delta_tau.clamp(min=0))` | Compressed gap scale |
| `delta` | `delta_tau - 20` | Gap deviation from nominal 20 min |
| `m` | `(delta_tau > 22).float()` | Missingness indicator (0 or 1) |

NaN TLO is replaced with 0; negative TLO is clamped to 0 for log features.

### Time Embedding MLP

```
Linear(6, 32) -> GELU -> Linear(32, 32) -> (B, S_max, 32)
```

---

## 5. Time-Decay GRU

### Mathematical Formulation

At each segment step j:

```
1. gamma_j = exp(-softplus(W_gamma * time_embed_j + b_gamma))   in (0, 1)^256
2. h_tilde = gamma_j * h_{j-1}                                   decay previous state
3. h_j = GRUCell(x_j, h_tilde)                                   standard GRU update
4. h_j = h_j * mask_j                                            zero out padding
```

- **Small/nominal gap**: gamma near 1, minimal decay (state persists)
- **Large gap**: gamma near 0, strong decay (state nearly reset)
- The decay gate uses the **32-dim time embedding** (not raw features), giving it access to TLO, gap, deviation, and missingness.

### Output Feature

```
o_j = [h_j(256) | x_j(256)] = 512 dims
```

Concatenating the recurrent hidden state (long-term context) with the projected token (current-segment evidence).

---

## 6. Forward Pass — Full Tensor Shape Trace

```
Input batch:
  fhr_st                         (B, S_max, 300, 43)
  up_st                          (B, S_max, 300, 43)
  time_from_labor_onset          (B, S_max)  seconds, may have NaN
  delta_t                        (B, S_max)  seconds
  mask                           (B, S_max)  bool
  target                         (B, S_max, 300)  for loss only

Step 1: Segment embeddings       (B, S_max, 416)
Step 2: Segment deltas            (B, S_max, 416)   delta_e_j = e_j - e_{j-1}
Step 3: Time embedding            (B, S_max, 32)
Step 4: Concatenation             (B, S_max, 864)   [e | delta_e | t]
Step 5: Input projection          (B, S_max, 256)   Linear(864,256)+LN+GELU+Drop
Step 6: Time-decay gates          (B, S_max, 256)   exp(-softplus(Linear(32,256)))
Step 7: GRU hidden states         (B, S_max, 256)   custom loop with decay
Step 8: Output feature            (B, S_max, 512)   [h | x]
Step 9: Prediction head           (B, S_max, 1) BCE  or  (B, S_max, 2) CE

Output:
  logits    (B, S_max) or (B, S_max, 2)
  probs     (B, S_max)                     sigmoid (BCE) or softmax[:,1] (CE)
  preds     (B, S_max)                     {0, 1}
  mask      (B, S_max)                     passed through
```

---

## 7. Loss Function

### Two Modes

| Setting | Head | Loss | Class Imbalance |
|---------|------|------|-----------------|
| `loss.type: "bce"` | `Linear(512, 1)` | `BCEWithLogitsLoss` | `pos_weight` scalar |
| `loss.type: "ce"` | `Linear(512, 2)` | `CrossEntropyLoss` | `class_weights` tensor (2,) |

### Label Extraction

```python
seg_labels = target.max(dim=-1).values  # (B, S_max)
binary_labels = (seg_labels > 1).float()  # HEALTHY=1 -> 0, ACIDOSIS=2/HIE=3 -> 1
```

Padding segments (all target=0) get label=0 but are masked out.

### Class Balance Options

```yaml
loss:
  class_balance:
    enabled: true       # false = no weighting at all
    method: "auto"      # auto-compute from training data
    # method: "manual"  # use manual values below
    manual_pos_weight: 1.0       # BCE only
    manual_class_weights: [1.0, 1.0]  # CE only
  label_smoothing: 0.0  # 0 = off, e.g. 0.05 for mild smoothing
```

Auto computation: `pos_weight = N_healthy / N_unhealthy` (for BCE), inverse-frequency weights (for CE).

---

## 8. Freeze / Unfreeze Strategy

### Three Modes

| Mode | Transformer | Classifier Head | When to Use |
|------|-------------|-----------------|-------------|
| `"frozen"` | Frozen all epochs | Trainable | Default. Fast training, good baseline. |
| `"trainable"` | Unfrozen from epoch 0 | Trainable | Full fine-tuning. Needs lower transformer LR. |
| `"phased"` | Frozen until epoch N, then unfrozen | Trainable | Best of both: train head first, then fine-tune. |

### Config

```yaml
freeze_strategy:
  mode: "frozen"              # "frozen" | "trainable" | "phased"
  unfreeze_after_epoch: 50    # phased mode only
  transformer_lr: 1.0e-5     # separate LR for transformer (trainable/phased)
```

### How Phased Works

1. Epochs 0 to N-1: transformer frozen, only classifier head trains at `lr=1e-3`.
2. At epoch N: `PlClassifier.on_train_epoch_start()` calls `model.unfreeze_transformer()`, adds transformer params to optimizer with `transformer_lr=1e-5`.
3. Epochs N+: both transformer and classifier train with discriminative learning rates.

### Gradient Control

Inside `_encode_transformer_chunked()`, a context manager controls gradient flow:

```python
needs_grad = (strategy == "trainable") or (strategy == "phased" and transformer.training)
ctx = torch.enable_grad() if needs_grad else torch.no_grad()
```

When frozen, `transformer.eval()` also disables dropout and fixes batchnorm.

---

## 9. Precomputed Embeddings

### Why Precompute?

The transformer is large. Precomputing 416-dim embeddings avoids the expensive forward pass during training. Storage is compact: 416 floats/segment = 1.66 KB. 100K segments ~ 166 MB.

### HDF5 Schema

```
precomputed_fold_{fold_id}_{partition}.hdf5
|-- attrs:
|   |-- transformer_checkpoint_hash  (SHA-256, 64 chars)
|   |-- transformer_checkpoint_path  (str)
|   |-- d_embedding                  (416)
|   |-- anchor_step                  (5)
|   |-- pooling                      ("mean")
|   |-- creation_timestamp           (ISO-8601)
|   |-- fold_id                      (int)
|   |-- partition                    ("train"|"val"|"test")
|   |-- total_segments               (int)
|   |-- total_guids                  (int)
|-- guids/
    |-- GUID_001/
    |   |-- embeddings               (S_1, 416) float32, gzip
    |   |-- epochs                   (S_1,)     float64
    |-- GUID_002/
    |   |-- embeddings               (S_2, 416) float32, gzip
    |   |-- epochs                   (S_2,)     float64
    ...
```

### Cache Invalidation

SHA-256 hash of the transformer checkpoint is stored in the HDF5. At training time, `PrecomputedEmbeddingDataset` computes the hash of the provided checkpoint and compares. Mismatch raises `ValueError`.

### PrecomputedEmbeddingDataset

Wraps `SignalSequenceDataset` and injects `embeddings_precomputed` (S_i, 416) into each sample. The existing `sequence_collate_fn` automatically pads it to `(B, S_max, 416)`. `TimeAwareGRUClassifier.forward()` detects the key and skips transformer encoding.

### CLI

```bash
python -m model.transformer.classification.precompute_embeddings \
    --config model/transformer/classification/config_classification.yaml \
    --fold_ids 1 2 3 \
    --output_dir /data/precomputed_embeddings \
    --device cuda:0 \
    --chunk_size 16
```

---

## 10. Training Pipeline

### Single-Fold Training

```bash
python -m model.transformer.classification.classification_trainer
```

**`train_fold(fold_id, config, gpu_id)`** does:

1. Seed random generators (42 + fold_id)
2. Load fold datasets via `get_fold_datasets()`
3. Create dataloaders (3 modes):
   - **Precomputed**: `PrecomputedEmbeddingDataset` + `LengthBucketSampler`
   - **Bucketed**: `SignalSequenceDataset` + `LengthBucketSampler`
   - **Standard**: `SignalSequenceDataset` + random shuffle
4. Estimate class balance weights (if `loss.class_balance.enabled`)
5. Write fold-specific config to `fold_{fold_id}/config.yaml`
6. Create `GraphModelClassificationTrainer`, load transformer, build model
7. Train with Lightning (callbacks: ModelCheckpoint, EarlyStopping, LossPlot)
8. Resolve best checkpoint metrics via validation pass
9. Save `fold_results.json`
10. Cleanup dataloaders and caches

### K-Fold Parallel Training

```bash
python -m model.transformer.classification.kfold_classification_trainer
```

**`run_kfold_classification_parallel()`** does:

1. Create parent MLflow run (if enabled)
2. Spawn one subprocess per fold using `ProcessPoolExecutor` with `'spawn'` context
3. GPU assignment: round-robin `gpu_ids[job_idx % len(gpu_ids)]`
4. Each subprocess sets `CUDA_VISIBLE_DEVICES` before any CUDA init
5. Timeout: `fold_timeout_hours` (default 6h). Hung folds are cancelled.
6. Cross-fold summary: mean +/- std of val_loss, val_accuracy
7. Save `kfold_summary.json`

---

## 11. Lightning Module: PlClassifier

Inherits from `LightningModelBase`. Key overrides:

- **Bypasses `torch.compile`** due to dynamic control flow (custom GRU loop, chunked encoding, conditional branches).
- **`training_step`**: Logs with `on_step=False` for epoch-averaged metrics (bucket sampler makes per-step values noisy).
- **`configure_optimizers`**: AdamW with separate param groups (classifier vs transformer LR) + MultiStepLR scheduler.
- **`on_train_epoch_start`**: Implements phased unfreezing by adding transformer params to optimizer mid-training.

---

## 12. Config Reference

### `config_classification.yaml` — Complete Key Listing

```
general_config.tag                              str     Experiment tag
general_config.cuda_devices                     [int]   GPU IDs
general_config.max_parallel_folds               int     Max parallel fold processes
general_config.plot_frequency                   int     Loss plot update every N epochs
general_config.lr                               float   Classifier learning rate
general_config.lr_milestone                     [int]   Epochs for 0.1x LR decay
general_config.epochs                           int     Max training epochs
general_config.accumulate_grad_batches          int     Gradient accumulation
general_config.batch_size.train                 int     GUIDs per training batch
general_config.batch_size.test                  int     GUIDs per validation batch
general_config.folders_config.out_dir_base      str     Root output directory

model_config.transformer_checkpoint             str     Path to .ckpt file
model_config.transformer_chunk_size             int     Segments per forward chunk
model_config.freeze_strategy.mode               str     "frozen"|"trainable"|"phased"
model_config.freeze_strategy.unfreeze_after_epoch int   Epoch to unfreeze (phased)
model_config.freeze_strategy.transformer_lr     float   Transformer LR after unfreeze
model_config.segment_embedding.d_embedding      int     416
model_config.segment_embedding.anchor_step      int     TE grid step (5=dense, 15=sparse)
model_config.segment_embedding.pooling          str     "mean"|"attention"
model_config.time_features.raw_dim              int     6 (fixed)
model_config.time_features.embed_dim            int     Time MLP output dim (32)
model_config.time_features.nominal_gap_minutes  float   Expected gap (20.0)
model_config.time_features.gap_threshold_minutes float  Missingness threshold (22.0)
model_config.classifier.input_proj_dim          int     Projection dim (256)
model_config.classifier.gru_hidden_dim          int     GRU hidden dim (256)
model_config.classifier.dropout                 float   Dropout rate (0.1)
model_config.loss.type                          str     "bce"|"ce"
model_config.loss.class_balance.enabled         bool    Enable class weighting
model_config.loss.class_balance.method          str     "auto"|"manual"
model_config.loss.class_balance.manual_pos_weight float Manual BCE weight
model_config.loss.class_balance.manual_class_weights [float] Manual CE weights
model_config.loss.label_smoothing               float   0=off
model_config.precompute_embeddings              bool    Use cached HDF5
model_config.precomputed_dir                    str     HDF5 cache directory
model_config.evaluation.target_fpr              float   Target FPR (0.2)
model_config.evaluation.exclude_last_minutes    float   Exclude near-birth bins (30.0)
model_config.evaluation.decision_time_hours     float   Decision time (1.0)
model_config.evaluation.max_gap_multiplier      float   Max gap for epoch filling (null)
model_config.evaluation.run_after_training      bool    Run eval after each fold

dataset_config.kfold_base_path                  str     K-fold dataset root
dataset_config.test_mode                        str     null|"holdout"|"augmented"
dataset_config.num_folds                        int     Total folds (10)
dataset_config.fold_ids                         [int]   Subset or null=all
dataset_config.stat_path                        str     Normalization stats HDF5
dataset_config.dataloader_config.num_workers    int     DataLoader workers
dataset_config.dataloader_config.prefetch_factor int    Prefetch per worker
dataset_config.dataloader_config.normalize_fields [str] Fields to normalize
dataset_config.dataloader_config.segment_duration float Grid slot seconds (1200)
dataset_config.dataloader_config.guid_cache_size int    GUID cache capacity (128)
dataset_config.dataloader_config.pin_memory     bool    Pin tensors
dataset_config.dataloader_config.dataset_kwargs.load_fields [str] HDF5 fields
dataset_config.dataloader_config.dataset_kwargs.epoch_min float Min epoch
dataset_config.dataloader_config.dataset_kwargs.trim_minutes float Trim minutes
dataset_config.bucket_sampler.enabled           bool    Use bucket sampling
dataset_config.bucket_sampler.bucket_ranges     [[int]] Segment count ranges
dataset_config.bucket_sampler.shuffle           bool    Shuffle within buckets

advanced_config.trainer.precision               str     "32-true"|"16-mixed"|"bf16-mixed"
advanced_config.trainer.gradient_clip_val        float   Max gradient norm
advanced_config.trainer.gradient_clip_algorithm  str     "norm"|"value"
advanced_config.trainer.deterministic            bool    Deterministic mode
advanced_config.trainer.benchmark                bool    cuDNN benchmark
advanced_config.callbacks.early_stopping.enabled bool    Enable early stopping
advanced_config.callbacks.early_stopping.patience int    Epochs without improvement
advanced_config.callbacks.early_stopping.monitor str     Metric to watch
advanced_config.callbacks.model_checkpoint.save_top_k int Keep best K checkpoints
advanced_config.callbacks.model_checkpoint.monitor str   Metric to sort by
advanced_config.tracking.mlflow.enabled          bool    Enable MLflow
advanced_config.tracking.mlflow.tracking_uri     str     MLflow server URI
advanced_config.tracking.mlflow.experiment_name  str     Experiment name
advanced_config.tracking.mlflow.tags             dict    Key-value tags
```

---

## 13. Class and Function Reference

### classification_model.py

**`TimeFeatureEncoder(embed_dim=32, nominal_gap_minutes=20.0, gap_threshold_minutes=22.0)`**
- `forward(time_from_labor_onset, delta_t, mask) -> Tensor (B, S_max, embed_dim)`

**`TimeAwareGRUClassifier(transformer_model, d_embedding=416, time_embed_dim=32, input_proj_dim=256, gru_hidden_dim=256, dropout=0.1, loss_type="bce", pos_weight=None, class_weights=None, label_smoothing=0.0, transformer_chunk_size=16, freeze_strategy="frozen", pooling="mean", anchor_step=5, nominal_gap_minutes=20.0, gap_threshold_minutes=22.0)`**
- `forward(batch) -> Dict[logits, probs, preds, mask]`
- `compute_loss(outputs, batch) -> Dict[loss, accuracy, class_0_acc, class_1_acc]`
- `freeze_transformer() -> None`
- `unfreeze_transformer() -> None`

### classification_trainer.py

**`PlClassifier(base_model, lr=1e-3, lr_milestones=None, weight_decay=1e-4, freeze_mode="frozen", unfreeze_after_epoch=50, transformer_lr=1e-5)`**
- `training_step(batch, batch_idx) -> Tensor`
- `compute_loss_and_metrics(batch, batch_idx, stage) -> Tuple[Tensor, Dict]`
- `configure_optimizers() -> Dict`
- `on_train_epoch_start() -> None` (phased unfreezing)

**`estimate_class_balance(dataset, loss_type="bce") -> Dict[pos_weight, class_weights]`**

**`resolve_best_checkpoint_metrics(trainer, pl_model, val_dl, ckpt_cb) -> Dict`**

**`GraphModelClassificationTrainer(config_file_path=None)`**
- `create_model(pos_weight=None, class_weights=None) -> None`
- `train_model(train_dl, val_dl) -> pl.Trainer`

**`train_fold(fold_id, config, gpu_id=0) -> Tuple[str, GraphModelClassificationTrainer]`**

### precompute_embeddings.py

**`precompute_fold_embeddings(fold_id, kfold_base_path, transformer_checkpoint, output_dir, config, device="cuda:0", chunk_size=16) -> Dict[str, str]`**

**`PrecomputedEmbeddingDataset(precomputed_path, transformer_checkpoint=None, segment_duration=1200.0, guid_cache_size=128, **dataset_kwargs)`**
- `__getitem__(idx) -> Dict` (injects `embeddings_precomputed`)
- `guid_lengths -> List[int]`
- `get_guid_list() -> List[str]`
- `estimate_class_weights(num_classes=2) -> Tuple`

**`create_precomputed_embedding_dataloader(precomputed_path, hdf5_files, batch_size=8, ...) -> Tuple[DataLoader, PrecomputedEmbeddingDataset]`**

### kfold_classification_trainer.py

**`create_parent_mlflow_run(config_data, experiment_tag, extra_tags=None) -> Tuple[client, run_id]`**

**`train_single_fold_classification(fold_id, gpu_id, base_config_path, kfold_base_path, output_base_dir, transformer_checkpoint, parent_run_id=None, **kwargs) -> Dict`**

**`run_kfold_classification_parallel(num_folds, gpu_ids, base_config_path, kfold_base_path, output_base_dir, transformer_checkpoint, max_parallel=None, fold_ids=None, sequential=False, fold_timeout_hours=6.0, **kwargs) -> List[Dict]`**

### evaluate_transformer_classifier.py

**`create_transformer_model_from_config(config, device="cuda:0") -> nn.Module`**

**`run_transformer_inference(model, dataloader, device="cuda:0") -> pd.DataFrame`**
- Output columns: `guid, epoch, target, binary_target, predicted_class, prob_class_0, prob_class_1, cs_label, bg_label, tlo_hours`

**`evaluate_single_fold_transformer(fold_dir, config, device, target_fpr=0.2, exclude_last_minutes=30.0, decision_time_hours=1.0, max_gap_multiplier=None, regenerate_predictions=False) -> Dict`**

**`aggregate_transformer_results(output_base_dir, fold_ids=None, fold_results=None, exclude_last_minutes=30.0) -> Dict`**

**`main(output_base_dir, config_path, target_fpr, device, ...) -> Dict`** — CLI entry point

### evaluation_plots.py

**`plot_metric_curves(metrics_df, metric_type, output_dir, title_suffix="")`** — 5 plot variants
**`plot_metric_comparison(metrics_dict, output_dir, title_suffix="")`** — 1x3 comparison
**`plot_subgroup_analysis(subgroup_metrics, metric_type, output_dir, title_suffix="")`** — 4 subgroup plots
**`plot_roc_curve(roc_data, output_path, title_suffix="", threshold=None)`** — single ROC
**`plot_aggregated_roc(all_roc_data, output_dir, n_folds, title_suffix="")`** — k-fold ROC overlay
**`plot_dataset_statistics(df, time_bins, output_dir, title_suffix="")`** — dataset overview
**`plot_aggregated_metrics(metric_type, all_fold_dfs, output_dir, n_folds, title_suffix="")`** — cross-fold metrics

---

## 14. Reused Components (External)

| Component | Location | Used For |
|-----------|----------|----------|
| `CausalMultimodalTransformer` | `model/transformer/model/model.py` | Pretrained encoder |
| `TransformerConfig` | `model/transformer/model/config.py` | Encoder config dataclass |
| `AttentionPool` | `model/transformer/model/layers.py` | Optional trainable pooling |
| `TransformerTestRunner._extract_config()` | `model/transformer/tr_testing/base.py` | Extract config from checkpoint |
| `load_checkpoint_strict()` | `train/graph_models_utils.py` | Load model weights |
| `GraphModelBase` | `train/graph_model_base.py` | Trainer base class |
| `LightningModelBase` | `train/pl_model_base.py` | Lightning module base |
| `LengthBucketSampler` | `model/vae_teb_prediction/guid_classifier/length_bucket_sampler.py` | Efficient variable-length batching |
| `create_bucketed_sequence_dataloader()` | Same file | DataLoader factory |
| `SignalSequenceDataset` | `hdf5_dataset/guid_hdf5_dataset.py` | GUID-grouped data loading |
| `sequence_collate_fn` | Same file | Pad variable-length sequences |
| `get_fold_datasets()` | `model/vae_teb_prediction/kfold_classifier_trainer.py` | Resolve fold file paths |
| `LossPlotCallback`, `MetricsLoggingCallback`, `HyperparameterLoggingCallback` | `train/callbacks.py` | Training callbacks |

---

## 15. Output Structure

After k-fold training + evaluation, the output directory looks like:

```
classification_results/
|-- kfold_summary.json
|-- aggregated_evaluation_results.json
|-- aggregated_plots/
|   |-- aggregated_roc_curves.png
|   |-- instantaneous/  (5 aggregated metric plots)
|   |-- committed_cumulative/
|   |-- committed_overall/
|-- fold_1/
|   |-- config.yaml
|   |-- fold_results.json
|   |-- checkpoints/
|   |   |-- cls-model-epoch=XX.ckpt
|   |-- train_results/
|   |   |-- loss_plot.png
|   |-- evaluation/
|       |-- validation_predictions_raw.csv
|       |-- validation_predictions_clinical.csv
|       |-- test_predictions_raw.csv
|       |-- test_predictions_clinical.csv
|       |-- threshold_info.json
|       |-- roc_curve.png
|       |-- roc_curve_committed_cumulative.png
|       |-- roc_data.csv
|       |-- validation_evaluation/
|       |   |-- three_metric_types/
|       |       |-- instantaneous/ (5 plots + subgroups/)
|       |       |-- committed_cumulative/
|       |       |-- committed_overall/
|       |       |-- comparison/
|       |       |-- dataset_stats/
|       |       |-- metrics_summary.json
|       |       |-- thresholds.json
|       |-- three_metric_types/  (same structure, for TEST data)
|-- fold_2/
|   |-- ...
```

### fold_results.json Schema (after evaluation)

```json
{
    "fold_id": 1,
    "training_time_minutes": 45.2,
    "best_val_loss_training": 0.4123,
    "best_val_accuracy_training": 0.8234,
    "best_checkpoint_path": ".../cls-model-epoch=42.ckpt",
    "status": "success"
}
```
