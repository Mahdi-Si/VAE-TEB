**SeqVaeTebClassifier: Training & Testing Pipeline (cls_implementation.md)**

This document explains how to implement a classification pipeline using `SeqVaeTebClassifier` built on top of the pretrained `SeqVaeTeb` representation. It covers dataset usage, model wiring, checkpoint loading with compiled models, PyTorch Lightning integration, training, and evaluation.

The guide maps to and extends these files:
- `model/vae_teb_model.py` (base VAE + classifier definition)
- `model/pytorch_lightning_modules.py` (VAE Lightning module; pattern to replicate for classifier)
- `model/graph_model_train.py` (VAE trainer; pattern to replicate for classifier)
- `model/graph_model_train_cls.py` (classification trainer; to be finalized per this doc)
- `hdf5_dataset/hdf5_dataset.py` (dataset + dataloader)
- `model/inception_time.py` (Inception‑Time classifier backbone)


**1) Dataset & Labels**

- Input fields (sequence-first already):
  - `fhr_st`: `(seq_len=360 or 300, channels=43)`
  - `fhr_ph`: `(seq_len, 44)`
  - `fhr_up_ph`: `(seq_len, 130)`
  - `fhr`: `(len_signal=4800)`; optional if adding auxiliary VAE loss during classifier training
  - `target`: `(seq_len,)` per-timestep class id masked by `weight` (see `model/dataset.md`)
- Class encoding: default mapping used during dataset creation is `1=HEALTHY, 2=ACIDOSIS, 3=HIE`.
- Extracting a sample’s label from `target`:
  - Recommended: `label = int(target.max())` (since invalid timesteps are 0 and valid are `c`)
  - More robust alternative: majority vote over `target[target>0]` with tie→highest class.

Tip: Ensure dataloader returns tensors already transposed to `(batch, seq, channels)` as implemented in `CombinedHDF5Dataset.__getitem__`.

Note: We want to do k-fold cross-validation. we have the following structure for our saved HDF5 files for each fold and we repeat this structure for each fold:

```
/data1/fetal-heart-tracing/HDF5_Datasets/last_12_hours_Q4/k_fold_cross_validation_dataset/
.
├── fold_1
│   ├── test
│   │   ├── acidosis_cs.hdf5
│   │   ├── acidosis_no_cs.hdf5
│   │   ├── healthy_bg_cs.hdf5
│   │   ├── healthy_bg_no_cs.hdf5
│   │   ├── healthy_no_bg_cs.hdf5
│   │   ├── healthy_no_bg_no_cs.hdf5
│   │   ├── hie_cs.hdf5
│   │   └── hie_no_cs.hdf5
│   ├── train
│   │   ├── acidosis_cs.hdf5
│   │   ├── acidosis_no_cs.hdf5
│   │   ├── healthy_bg_cs.hdf5
│   │   ├── healthy_bg_no_cs.hdf5
│   │   ├── healthy_no_bg_cs.hdf5
│   │   ├── healthy_no_bg_no_cs.hdf5
│   │   ├── hie_cs.hdf5
│   │   └── hie_no_cs.hdf5
│   └── val
│       ├── acidosis_cs.hdf5
│       ├── acidosis_no_cs.hdf5
│       ├── healthy_bg_cs.hdf5
│       ├── healthy_bg_no_cs.hdf5
│       ├── healthy_no_bg_cs.hdf5
│       ├── healthy_no_bg_no_cs.hdf5
│       ├── hie_cs.hdf5
│       └── hie_no_cs.hdf5
├── fold_k
│   ....

```
**2) Model Stack**

- `SeqVaeTeb` encodes (`fhr_st`, `fhr_ph`, `fhr_up_ph`) to latent `z` and also has a decoder for reconstruction. See `model/vae_teb_model.py:1177` for `SeqVaeTebClassifier`.
- `SeqVaeTebClassifier` wraps a base `SeqVaeTeb` and routes the latent `z` through `FHRInceptionTimeClassifier` to produce logits.
- Default shapes:
  - Latent `z`: `(batch, 300, latent_dim_z)` where `latent_dim_z` is 16 by default in `SeqVaeTeb`.
  - Classifier input size must match `latent_dim_z`.

Freezing vs fine-tuning:
- Feature extractor (VAE) is typically frozen during classifier training for stability and speed.
- Optional end-to-end fine‑tuning can be enabled to improve accuracy after the classifier head converges.


**3) Checkpoint Loading (Lightning + torch.compile)**

You may have trained `SeqVaeTeb` under PyTorch Lightning and with `torch.compile`. Key points to load its weights into `SeqVaeTebClassifier.vae_model` safely:

- Lightning checkpoints contain a `state_dict` where VAE parameters are usually prefixed by `model.` because the LightningModule wraps the model as `self.model` (see `model/pytorch_lightning_modules.py:604`).
- When loading directly into a bare `SeqVaeTeb`, strip the `model.` prefix from keys.
- torch.compile does not affect state_dict contents; compile after constructing/loading the model, not before saving.

Robust load function for the classifier (recommended implementation):

```python
def load_pretrained_vae_from_lightning_ckpt(vae: nn.Module, lightning_ckpt_path: str) -> None:
    ckpt = torch.load(lightning_ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # strip leading 'model.' if present
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("model."):
            nk = k[len("model."):]
        new_sd[nk] = v

    missing, unexpected = vae.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[SeqVaeTebClassifier] Missing keys: {missing}")
    if unexpected:
        print(f"[SeqVaeTebClassifier] Unexpected keys: {unexpected}")
```

Where to compile:
- Instantiate `SeqVaeTeb()` → load state_dict (as above) → optionally wrap with `torch.compile`. Do not compile the temporary loader model used to read the checkpoint; compile only the final model you’ll actually run.

Alternative (Lightning-native) load path:
- If you prefer Lightning’s `load_from_checkpoint`, create a `LightSeqVaeTeb` with a constructed `SeqVaeTeb()` and pass it to `load_from_checkpoint(..., seqvae_teb_model=instance, strict=False)`, then set `classifier.vae_model = lightning_module.model`. This preserves exact parity with how the checkpoint was saved. See the pattern in `model/graph_model_train.py:284-384`.


**4) Classifier Module**

`SeqVaeTebClassifier` exists in `model/vae_teb_model.py:1177`. Key behaviors:
- Initializes `self.vae_model = SeqVaeTeb()` and `self.classifier = FHRInceptionTimeClassifier(input_size=latent_dim_z, num_classes=...)`.
- `freeze_vae_parameters()` marks all VAE params `requires_grad=False`.
- `extract_latent_features(...)` runs the VAE and returns `z`. If frozen, uses `torch.set_grad_enabled(False)` and sets `vae_model.eval()`.
- `forward(...)` returns logits, probabilities, predictions; accepts optional labels to compute `nn.CrossEntropyLoss`.
- `compute_loss(...)` can combine classification loss with an optional VAE reconstruction loss (weighted by `vae_loss_weight`).

Performance note:
- The current `extract_latent_features` invokes the full VAE forward (encoders + decoder). For speed when frozen, consider adding a light `encode_to_z(y_st, y_ph, x_ph)` method on `SeqVaeTeb` to compute only `z` without decoder.


**5) Lightning Module for Classification**

Create `LightSeqVaeTebClassifier` modeled on `LightSeqVaeTeb` but for classification. Recommended responsibilities:
- Inputs from batch: `y_st = batch.fhr_st`, `y_ph = batch.fhr_ph`, `x_ph = batch.fhr_up_ph`, `target = batch.target`.
- Label extraction: prefer `labels = target.max(dim=1).values.long()`; or a majority vote over nonzero timesteps.
- Loss: `nn.CrossEntropyLoss` with optional class weights for imbalance.
- Metrics: accuracy, AUROC (binary/multiclass), F1. Log both on `train/` and `val/`.
- Optional: add `vae_loss_weight > 0` to include frozen VAE reconstruction loss; requires `batch.fhr` present.
- Optimizer: `AdamW` + optional milestone scheduler, mirroring `LightSeqVaeTeb.configure_optimizers`.

Skeleton:

```python
class LightSeqVaeTebClassifier(L.LightningModule):
    def __init__(self, model: SeqVaeTebClassifier, lr=1e-4, lr_milestones=None, 
                 class_weights=None, vae_loss_weight=0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # saves lr, etc.
        self.model = model
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # define metrics here (torchmetrics)

    def _extract_labels(self, target):
        # target: (batch, seq_len)
        return target.max(dim=1).values.long()

    def training_step(self, batch, batch_idx):
        y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
        labels = self._extract_labels(batch.target)

        # main loss
        out = self.model(y_st, y_ph, x_ph, labels=labels)
        loss = out["classification_loss"]

        # optional VAE loss
        if self.hparams.vae_loss_weight > 0 and hasattr(batch, "fhr"):
            vae_losses = self.model.vae_model.compute_loss(
                forward_outputs=self.model.vae_model(y_st, y_ph, x_ph),
                y_st=y_st, y_ph=y_ph, y_raw=batch.fhr, compute_kld_loss=True, beta=1.0)
            loss = loss + self.hparams.vae_loss_weight * vae_losses["total_loss"]

        # log loss + metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # compute + log metrics (accuracy, f1, auroc)
        return loss

    def validation_step(self, batch, batch_idx):
        y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
        labels = self._extract_labels(batch.target)
        out = self.model(y_st, y_ph, x_ph, labels=labels)
        self.log("val/loss", out["classification_loss"], on_epoch=True, prog_bar=True)
        # log metrics
        return out["classification_loss"]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        if self.hparams.lr_milestones:
            from torch.optim.lr_scheduler import MultiStepLR
            sch = MultiStepLR(opt, milestones=self.hparams.lr_milestones, gamma=0.1)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        return opt
```


**6) Training Script (`graph_model_train_cls.py`)**

Mirror the structure of `model/graph_model_train.py` but for classification:
- Dataloader: `create_optimized_dataloader(...)` with fields including `target` and optionally `fhr` (if using VAE loss).
- Model creation:
  - Build `SeqVaeTebClassifier(freeze_vae=True, num_classes=..., pretrained_vae_path=...)`.
  - Load VAE weights using the robust helper in Section 3.
  - If desired, compile the classifier’s `vae_model` after loading the state_dict:
    `classifier.vae_model = torch.compile(classifier.vae_model, mode='max-autotune-no-cudagraphs', dynamic=True)`.
- Lightning module: instantiate `LightSeqVaeTebClassifier` with optimizer settings and optional `class_weights`.
- Callbacks: `ModelCheckpoint(monitor='val/loss', mode='min')`, `EarlyStopping`, plus plotting/metrics callbacks as needed.
- Trainer: follow the DDP/precision/compile settings from config; ensure `deterministic` and TF32 flags align with your environment.

Checkpoint best practices:
- In VAE training, also export a pure model state for easier reuse later:
  ```python
  torch.save({"model_state_dict": lightning_module.model.state_dict()}, "seqvae_model_state.pt")
  ```
- For classification, save both Lightning checkpoint and an export of the classifier head if needed:
  ```python
  torch.save({
      "classifier_head": classifier.classifier.state_dict(),
      "vae_frozen": True,
  }, "seqvae_classifier_head.pt")
  ```


**7) Testing & Evaluation**

Add `graph_model_test_cls.py` (or extend an existing test script) to evaluate on held‑out data:
- Load the same dataloader with the val/test split.
- Restore `LightSeqVaeTebClassifier.load_from_checkpoint` for the last/best checkpoint.
- Compute metrics: accuracy, balanced accuracy, AUROC, AUPRC, F1, confusion matrix. Log per‑class metrics.
- For imbalanced datasets, report macro/micro averages.

Minimal evaluation snippet:
```python
ckpt = ".../checkpoints/best.ckpt"
lm = LightSeqVaeTebClassifier.load_from_checkpoint(ckpt, model=SeqVaeTebClassifier(...), strict=False)
lm.eval().to(device)
with torch.no_grad():
    for batch in dl:
        y_st, y_ph, x_ph = batch.fhr_st.to(device), batch.fhr_ph.to(device), batch.fhr_up_ph.to(device)
        labels = batch.target.max(dim=1).values.long().to(device)
        out = lm.model(y_st, y_ph, x_ph, labels=labels)
        # accumulate metrics
```


**8) Config Suggestions**

Extend `model/config.yaml` with classification settings:
- `classifier_config`:
  - `num_classes: 2|3`
  - `filters: 32`
  - `depth: 6`
  - `dropout: 0.2`
  - `use_attention: true`
  - `freeze_vae: true`
  - `pretrained_vae_ckpt: path/to/vae.ckpt or .pt`
  - `vae_loss_weight: 0.0`
  - `class_weights: [..]` or compute from train split
- `training_config`:
  - `epochs`, `lr`, `lr_milestones`, `batch_size`, `precision`, `devices`, `strategy: ddp_find_unused_parameters_false`
  - `compile_model: true|false`


**9) Common Pitfalls**

- Prefix mismatch when loading Lightning checkpoints: strip `model.` or use Lightning `load_from_checkpoint` and take `.model`.
- torch.compile loading order: instantiate → load state_dict → then compile. Do not attempt to load a compiled GraphModule from disk.
- Dataset label extraction: `target` is masked; do not treat zeros as a valid class.
- Freezing VAE: set `requires_grad=False` and also consider `vae_model.eval()` during training to disable dropout/BN updates.
- DDP/AMP: ensure operations in frozen VAE do not require grads; wrap feature extraction in `torch.set_grad_enabled(False)` for speed.


**10) Migration Checklist**

- Implement `LightSeqVaeTebClassifier` (Section 5) alongside `LightSeqVaeTeb`.
- Finalize `graph_model_train_cls.py` to:
  - Build dataloaders with `target` present
  - Construct `SeqVaeTebClassifier`, load VAE checkpoint, optionally compile the VAE
  - Create `LightSeqVaeTebClassifier` and configure callbacks
  - Train and save checkpoints
- (Optional) Add `graph_model_test_cls.py` for standalone evaluation and metrics export.
- Add a pure VAE `model_state_dict` export to your VAE training for future reuse.


**11) File Pointers (for context)**

- Base model and classifier definition: `model/vae_teb_model.py:1177`
- Inception‑Time classifier: `model/inception_time.py:185`
- Lightning VAE module pattern: `model/pytorch_lightning_modules.py:568`
- Lightning VAE checkpoint/compile pattern: `model/graph_model_train.py:284`
- Dataset shapes and semantics: `model/dataset.md:1`
- Dataset loader transposes to (seq, ch): `hdf5_dataset/hdf5_dataset.py:706`

