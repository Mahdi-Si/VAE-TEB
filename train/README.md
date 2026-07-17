## Training Framework

Reference guide for anyone wiring up a training run on the shared `train/` stack. `GraphModelBase` is the experiment driver (config parse, run directories, determinism, loguru sinks, MLflow, `Trainer` construction); `LightningModelBase` is the `LightningModule` you subclass to define a loss. You implement three methods total — `create_model`, `train_model`, `compute_loss_and_metrics` — and inherit config validation, seeding, per-rank run logs, MLflow provenance, an AdamW + `MultiStepLR` (optionally warmup) stack, a checkpoint class stamp, and an opt-in loss-spike circuit breaker.

File map:

| File | What it is |
| --- | --- |
| `train/graph_model_base.py` | `GraphModelBase` (ABC) — the experiment driver you subclass |
| `train/pl_model_base.py` | `LightningModelBase` (ABC) — the `LightningModule` you subclass |
| `train/callbacks.py` | 4 model-agnostic callbacks |
| `train/data_module.py` | `GraphDataModule` — opt-in `LightningDataModule` over the HDF5 loaders |
| `train/graph_models_utils.py` | `load_checkpoint_torch`, `load_checkpoint_strict`, `denormalize_signal_data` |
| `train/config.yaml` | The shipped reference config |
| `train/test_utils.py` | Test doubles (`make_graph_model`, `TinyLightningModel`, fake MLflow/Trainer/Strategy) |
| `train/tests/` | 17 test modules + `conftest.py` |
| `utils/mlflow_utils.py` | Out of tree: the `log_artifact_to_mlflow` rank-0 upload seam, shared by `train/` and the plotters |
| `utils/seqvae_plot_callbacks.py` | Out of tree, listed here because §5 sends you there: the SeqVAE-coupled plotting callbacks |

---

### 1. Quickstart

Copy `train/config.yaml`, edit the required keys, write one file with two subclasses, run it.

```yaml
# my_config.yaml — the keys that must exist or construction fails.
general_config:
  tag: my_experiment          # required
  cuda_devices: [0]           # required, must be a list
  epochs: 200                 # required
  lr: 1.0e-4                  # required
  lr_milestone: [50, 120]     # KeyError in __init__ if missing (NOT in validate_config)
  plot_frequency: 5           # KeyError in __init__ if missing (NOT in validate_config)
  batch_size: {train: 32, test: 64}            # required, dict
  folders_config: {out_dir_base: /data/runs}   # required, dict; out_dir_base KeyErrors if missing
advanced_config:              # required, dict
  trainer:                    # required, dict; may be empty -> code defaults
    compile: false            # see the local-dev note below
```

```python
# my_trainer.py
from lightning.pytorch.callbacks import ModelCheckpoint

from train.graph_model_base import GraphModelBase
from train.pl_model_base import LightningModelBase


class MyModule(LightningModelBase):
    def compute_loss_and_metrics(self, batch, batch_idx, stage):
        out = self.model(batch["x"])                      # compiled handle
        loss = self.orig_model.compute_loss(out, batch)   # eager module for non-forward helpers
        return loss, {"total_loss": loss}                 # -> logged as train/total_loss, val/total_loss


class MyTrainer(GraphModelBase):
    def create_model(self):
        base = MyRawModule(**self.config["model_config"])
        trainer_cfg = self.config["advanced_config"]["trainer"]
        self.pl_model = MyModule(
            base,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            compile_model=bool(trainer_cfg.get("compile", True)),
            spike_breaker=self.config["advanced_config"].get("spike_breaker"),
        )
        self.apply_config_hyperparameters({"lr": self.lr}, self.pl_model)

    def train_model(self, train_loader, validation_loader):
        callbacks = [ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor="val/total_loss", mode="min", save_top_k=3,
            filename="model-{epoch:02d}",
        )]
        trainer = self.build_trainer(callbacks, model=self.pl_model)
        trainer.fit(self.pl_model, train_loader, validation_loader)
        return trainer


def main():
    gm = MyTrainer(config_file_path="my_config.yaml")
    gm.setup_config()                    # validate -> seed -> mkdirs -> loguru -> MLflow
    train_loader, val_loader = build_my_dataloaders(gm.config)
    gm.create_model()
    gm.train_model(train_loader, val_loader)


if __name__ == "__main__":
    main()
```

```bash
python -m model.my_pkg.my_trainer

# DDP under torchrun — export the stamp so ranks 1..N-1 share rank 0's run dir.
# Rank 0 has LOCAL_RANK=0 and therefore regenerates the stamp rather than inheriting
# it (see the non-zero-LOCAL_RANK gate in §4). The two agree only within the same
# minute, so export immediately before launching.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=4 -m model.my_pkg.my_trainer
```

Two things bite on a first local run:

- **The call order is load-bearing and nothing chains it for you.** `create_model()` before `setup_config()` means no seeding, no log sinks, no output dirs, and `self.mlflow_logger is None` — so `build_trainer` silently drops the MLflow callback.
- **`compile: true` (the shipped value) cannot work on the Windows dev box.** Triton is Linux-only, so the first forward raises `torch._inductor.exc.TritonMissing: Cannot find a working triton installation`. The shipped `true` targets the Linux prod box; set `compile: false` locally.

Note the `ModelCheckpoint` `filename` template: Lightning auto-prefixes each placeholder with its own name, so `"model-epoch={epoch:02d}"` renders as `model-epoch=epoch=00.ckpt`. Write `"model-{epoch:02d}"`.

---

### 2. The two classes you implement

#### 2.1 `GraphModelBase` — the experiment driver

`GraphModelBase(ABC)` cannot be instantiated. Constructor is `__init__(self, config_file_path=None)`; the `None` default points at `<train dir>/seqvae_configs/config_args.yaml`, which does not exist in this repo — always pass an explicit path.

| You must implement | Signature | Notes |
| --- | --- | --- |
| `create_model` | `create_model(self)` | Build the raw `nn.Module`, optionally load weights, wrap in your `LightningModelBase`. Return value is not consumed — store it (convention: `self.pl_model`). |
| `train_model` | `train_model(self, train_loader, validation_loader)` | Build your callbacks, call `build_trainer`, call `trainer.fit`. |

Only those two are abstract. Two stale signposts to ignore: `create_model`'s docstring tells you to call `self.configure_trainable_params`, but that method is entirely commented out and calling it raises `AttributeError` — freeze inline or use the `freeze_model(model)` staticmethod. And `train_model`'s `NotImplementedError` message names `train_base_model()`, which does not exist.

| You get for free | What it does |
| --- | --- |
| `setup_config()` | `validate_config` -> `configure_determinism` -> mkdirs -> loguru sinks -> log config dump -> reset CUDA memory stats -> MLflow logger + run metadata + system metrics monitor + `atexit` log upload |
| `validate_config()` | Required-key and known-key type checks; raises `ValueError` naming the key |
| `configure_determinism()` | `seed_everything(seed, workers=True)` + cuDNN/TF32/matmul flags |
| `build_trainer(callbacks, model=None)` | `Trainer(**self._build_trainer_kwargs(callbacks, model=model))` |
| `select_ddp_strategy(num_devices, config, model=None)` | Override hook; returns `"ddp"` when `num_devices > 1` else `"auto"` |
| `set_cuda_devices(device_list=None)` | Overrides `self.cuda_devices`; `None` -> `[0]`. Must precede `build_trainer` |
| `freeze_model(model)` | staticmethod, in-place `requires_grad = False` |
| `apply_config_hyperparameters(hparams_dict, lightning_module)` | Force-`setattr`s each non-`None` entry onto `.hparams` so config beats checkpoint-restored values |
| `upload_run_logs()` | Uploads `full.log` + `run.jsonl` to MLflow artifact path `logs`; idempotent, auto-registered with `atexit` |

Attributes set by `__init__` and available to you: `config`, `experiment_tag`, `cuda_devices`, `output_base_dir`, `base_folder`, `train_results_dir`, `test_results_dir`, `model_checkpoint_dir`, `aux_dir`, `tensorboard_dir`, `plot_every_epoch`, `epochs_num`, `lr`, `lr_milestones`, `batch_size_train`, `batch_size_test`, `accumulate_grad_batches`, `clip = 10`, `mlflow_logger = None`, `lightning_loggers = []`.

`self.clip = 10` is assigned and never read — clipping is `advanced_config.trainer.gradient_clip_val`, which is passed through with no default (omit it and you get `None` = no clipping).

`apply_config_hyperparameters` skips `None` values, so an explicitly-null config key will not clear a checkpoint-restored hparam — it leaves the old value. The "no overrides were applied" log only fires when every value was `None`.

Importing `train.graph_model_base` has side effects you cannot avoid: `KMP_DUPLICATE_LIB_OK=TRUE`, `TF_ENABLE_ONEDNN_OPTS=0`, `PYDEVD_USE_CYTHON=NO`, `CUDA_LAUNCH_BLOCKING=0`, `matplotlib.use('Agg')`; `plt.ion()` runs in `__init__`.

#### 2.2 `LightningModelBase` — the module

```python
from typing import Tuple

import torch
from torch import nn

from train.pl_model_base import LightningModelBase, MetricDict


class MyVaeModule(LightningModelBase):
    prog_bar_metrics = ("total_loss", "kld_loss")     # matched on the suffix after the last '/'
    sync_dist_stages = ("val", "test")                # train metrics are rank-local by default

    def __init__(self, base_model: nn.Module, *, kld_beta: float = 1.0, **kwargs) -> None:
        super().__init__(base_model, **kwargs)
        self.save_hyperparameters("kld_beta")         # the base ignores only `base_model`

    def compute_loss_and_metrics(self, batch, batch_idx: int, stage: str) -> Tuple[torch.Tensor, MetricDict]:
        outputs = self.model(batch["x"])
        loss_dict = self.orig_model.compute_loss(outputs, batch, beta=self.hparams.kld_beta)
        total_loss = loss_dict["total_loss"]
        return total_loss, {
            "total_loss": total_loss,
            "kld_loss": loss_dict.get("kld_loss"),    # None values are skipped
            "main_loss": loss_dict.get("reconstruction_loss"),
        }

    def _on_train_epoch_start_hook(self) -> None:      # override THIS, not on_train_epoch_start
        self.hparams.kld_beta = min(1.0, 0.1 * (self.current_epoch + 1))
```

Constructor: `__init__(self, base_model, *, lr=1e-4, lr_milestones=None, lr_gamma=0.1, lr_warmup_epochs=0, weight_decay=1e-4, module_name=None, compile_model=True, spike_breaker=None)`. `base_model` is the only positional arg; everything after the bare `*` is keyword-only. All kwargs except `base_model` land in `self.hparams` via `save_hyperparameters(ignore=['base_model'])`.

**Always go through this constructor.** If you need eager, pass `compile_model=False` — do **not** bypass with a grandparent `pl.LightningModule.__init__(self)` call. That hack predates the flag and silently drops every base `__init__` side effect: `save_hyperparameters`, `self._orig_model` (so `orig_model` and the `model_class` checkpoint stamp both break), `self.model`, `self._wrapper_name`, and the spike-breaker counters. Because `hparams` access in the base is defensive, you get silent fallback defaults rather than an error. Six live consumers still use the bypass (`trainer_lag_attn_v1.py:121`, `pl_module_synth.py:124`, `pl_module_v2.py:213`, two `temporal_classifier_trainer.py:93`, `classification_trainer.py:99`), plus `StandInConsumer` in `train/test_utils.py:66` — which does it deliberately, to pin the unmigrated behavior in a test.

**Automatic optimization is assumed, not merely defaulted to.** `training_step` returns a loss and nothing sets `automatic_optimization = False`. Setting it in a subclass makes Lightning reject `Trainer(gradient_clip_val>0)` (shipped as `0.5`) and `accumulate_grad_batches != 1` outright, forcing you to hand-roll the clip, the accumulation boundary, the scheduler step, and the spike breaker — whose skip path is a zero-gradient step that *relies* on the framework stepping the optimizer for you (§8). It also breaks DDP: under plain `'ddp'` (`find_unused_parameters=False`, what `select_ddp_strategy` returns) `DDPStrategy.pre_backward` calls `prepare_for_backward` on every `manual_backward`, so a second backward that does not touch every parameter raises. To add an auxiliary term (e.g. a permutation KL), fuse it into the single main forward and reuse the tensors that forward already computed, so one forward and one backward mark each parameter ready exactly once.

| You must implement | Notes |
| --- | --- |
| `compute_loss_and_metrics(batch, batch_idx, stage)` | Returns `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`; `stage` is `'train'`, `'val'` or `'test'` |

| You get for free | Notes |
| --- | --- |
| `training_step` / `validation_step` / `test_step` | Thin wrappers over `_dispatch_stage_step`; do not override |
| `forward` | `self.model(*args, **kwargs)` |
| Metric logging | Stage-prefixed, `on_step=True, on_epoch=True, logger=True`, detached |
| `configure_optimizers` | `list(configure_param_groups())` -> overview log -> `build_optimizer` -> `build_lr_scheduler` |
| `on_save_checkpoint` | Stamps `checkpoint["model_class"]` with the eager class name |
| Spike breaker | Train-stage only, off unless `spike_breaker["enabled"]` |

| Override seam | Default |
| --- | --- |
| `configure_param_groups()` | `self._trainable_parameters()` — flat `requires_grad` list. Return AdamW group dicts for differential LR; omitted keys fall back to base `lr`/`weight_decay` |
| `build_optimizer(trainable_params)` | `AdamW(lr=hparams.lr, weight_decay=hparams.weight_decay, eps=1e-8, betas=(0.9, 0.95))` — note betas are $(0.9, 0.95)$, not torch's $(0.9, 0.999)$ |
| `build_lr_scheduler(optimizer)` | See below |
| `_on_train_epoch_start_hook()` | Empty |
| `prog_bar_metrics` | `("total_loss",)` |
| `sync_dist_stages` | `("val", "test")` |

Scheduler branches, driven by three hparams read via `getattr`:

- no `lr_milestones` and `lr_warmup_epochs <= 0` -> returns `None`, `configure_optimizers` returns the bare optimizer.
- `lr_warmup_epochs <= 0` with milestones -> `MultiStepLR(optimizer, milestones, gamma)`.
- `lr_warmup_epochs > 0` -> `SequentialLR([LinearLR(start_factor=0.1, end_factor=1.0, total_iters=warmup), MultiStepLR(shifted_milestones, gamma)], milestones=[warmup])`, where `shifted_milestones = [max(0, m - warmup) for m in milestones]` because `SequentialLR` restarts the second scheduler's epoch counter at the switch. You still specify milestones in **absolute** epochs; a milestone earlier than the warmup collapses to epoch 0 of the decay phase and fires immediately at the switch.

When a scheduler exists it is returned as `{"scheduler": ..., "interval": "epoch", "frequency": 1}` — epoch-wise stepping. Override `build_lr_scheduler` for a per-step dict.

Metric-key gotchas:

- A key containing `/` bypasses stage prefixing entirely: `name = raw_name if "/" in raw_name else f"{stage}/{raw_name}"`. Returning `"val/foo"` from a train-stage call logs it under `val/foo` and can poison a `ModelCheckpoint` monitor.
- Non-tensor values are coerced: `float`/`int` become tensors; **anything else silently becomes `torch.tensor(0.0)`**. A stray string logs as a clean $0.0$ rather than raising.
- Train metrics are not `sync_dist`-reduced, so `train/total_loss` in `callback_metrics` is rank-local under DDP.
- `_log_learning_rate` reads `param_groups[0]` only and logs the unprefixed key `lr`. With a differential-LR override the logged `lr` is just the first group.

`configure_optimizers` materialises the param groups with `list(...)` on purpose: an override returning a lazy generator (e.g. `self.parameters()`) would be exhausted by `_log_parameter_overview` before the optimizer sees it, leaving the optimizer with zero parameters and no error. If you call `build_optimizer` yourself, materialize first.

`hparams` access is defensive throughout (`getattr(self.hparams, "lr", 1e-4)`, `self.hparams.get("spike_breaker", None)`): a subclass that skips `super().__init__()` or its own `save_hyperparameters` silently gets the fallback defaults instead of an error.

---

### 3. `config.yaml` reference

Validation is deliberately asymmetric and is **not** a schema. Only 9 keys are required; only 27 optional-but-known keys are type-checked (plus the 6 required keys whose container type is checked); unknown keys and the dead `advanced_config.memory` block only warn, so unmigrated consumer configs keep loading. The entire `model_config` and `dataset_config` blocks are unvalidated by the framework.

Type checking treats `bool` as distinct from `int`: `bool` requires a real bool, `int` rejects bools, `float` accepts int/float but rejects bool. So `precision: 32` fails (must be `str`) and `multiplier: 5` passes.

`validate_config()` runs from `setup_config()` — **after** `__init__` already hard-indexed the config. See the (*) rows below.

#### 3.1 `general_config`

| Key | Shipped | Code default | Validated | Controls |
| --- | --- | --- | --- | --- |
| `tag` | `core_model_with_l8` | — | required (presence only) | `self.experiment_tag`; run folder `{stamp}-{tag}`; MLflow experiment_name fallback |
| `seed` | `42` | `42` | optional, strict `int` | `seed_everything(seed, workers=True)` |
| `cuda_devices` | `[0,1,2,3,4,5,6]` | — | required, `list` | `Trainer(devices=...)`, world size, DDP strategy, `sync_batchnorm` gate, `world_size` tag |
| `lr` | `0.001` | — | required (presence only) | `self.lr` |
| `lr_milestone` (*) | `[500]` | — | none — bare index in `__init__` | `self.lr_milestones` |
| `checkpoint_frequency` | `1` | — | none | **dead** — nothing under `train/` reads it, and no warning fires (the unknown-key scan only covers `advanced_config`) |
| `plot_frequency` (*) | `1` | — | none — bare index in `__init__` | `self.plot_every_epoch` |
| `epochs` | `5000` | — | required (presence only) | `Trainer(max_epochs=...)` |
| `accumulate_grad_batches` | `1` | `1` | none | `Trainer(accumulate_grad_batches=...)` |
| `batch_size` | dict | — | required, `dict` | container |
| `batch_size.train` | `210` | `32` (DataModule) | none | train loader batch size |
| `batch_size.test` | `210` | `32` (DataModule) | none | val **and** test loader batch size |
| `folders_config` | dict | — | required, `dict` | container |
| `folders_config.out_dir_base` (*) | `/data/deid/isilon/MS_model/seq_vae_teb_results/new_structure` | — | none — bare index in `__init__` | root of all run output; `os.path.normpath`-ed |

(*) Read with a bare `[...]` index in `__init__`, which runs before `validate_config`. Missing -> raw `KeyError` from the constructor, never the friendly `ValueError`.

#### 3.2 `model_config`

Not validated, and not consumed by the framework — consumer-owned. The framework's only interaction is flattening this block into MLflow params (`_log_run_metadata_to_mlflow`). The shipped block is the reference consumer's:

| Key | Shipped | Consumer default |
| --- | --- | --- |
| `VAE_model.enable_forecaster` | `true` | `scattering_forecast_weight > 0.0` |
| `VAE_model.freeze_core_model` | `false` | `False` |
| `VAE_model.scattering_forecast_weight` | `1` | `latent_nll_weight`, then `0.0` |
| `VAE_model.horizon_len` | `30` | `30` |
| `VAE_model.beta_schedule` | `constant` | `'linear'` |
| `VAE_model.beta_start` | `0.0` | `0.0` |
| `VAE_model.beta_end` | `1.0` | `6.0` |
| `VAE_model.beta_anneal_epochs` | `50` | `50` |
| `VAE_model.beta_cycle_len` | `1000` | `1000` |
| `VAE_model.beta_const_val` | `0.005` | `kld_beta` |
| `VAE_model.kld_beta` | `0.005` | — (bare index; required for that consumer) |
| `VAE_model.forecaster_hidden_dim` | `128` | `128` |
| `VAE_model.forecaster_layers` | `2` | `2` |
| `VAE_model.forecaster_dropout` | `0.1` | `0.1` |
| `VAE_model.forecaster_min_logvar` | `-7.0` | `-7.0` — lower clamp on predicted $\log\sigma^2$ |
| `VAE_model.forecaster_max_logvar` | `4.0` | `4.0` |
| `VAE_model.predictive_horizon` | `30` | falls back to `horizon_len`; floored at 1 |
| `VAE_model.predictive_max_anchors` | `0` | `0` = every valid timestep; floored at 0 |
| `VAE_model.log_forecast_metrics` | `true` | `True` |
| `VAE_model.scattering_discount_gamma` | `1.0` | `latent_discount_gamma`, then `1.0` |
| `base_model_checkpoint` | `null` | relative paths resolve against the config file's dir |
| `legacy_seqvae_checkpoint` | a `.ckpt` path | same resolution; its presence is how that consumer detects the training stage |

`enable_forecaster: false` silently overrides three keys in that consumer: `scattering_forecast_weight` is zeroed, `log_forecast_metrics` is forced `False`, and a latent/scattering `monitor_metric` is rewritten to `val/total_loss`.

#### 3.3 `dataset_config`

| Key | Shipped | Code default | Controls |
| --- | --- | --- | --- |
| `vae_train_datasets` | 2 HDF5 paths | `[]` | shuffled train loader; MLflow tag (truncated to 500 chars) |
| `vae_test_datasets` | 1 HDF5 path | `[]` | **both** val and test loaders, unshuffled; MLflow tag |
| `stat_path` | `.../stats.hdf5` | `None` | passed as `stats_path=` to `create_optimized_dataloader`. Note the key is `stat_path`, the param is `stats_path` — a typo'd `stats_path:` in YAML silently yields `None` and disables normalization |
| `dataloader_config.num_workers` | `0` | `4` | worker processes; `0` also disables prefetch, spawn context and `persistent_workers` |
| `dataloader_config.prefetch_factor` | — | `2` | prefetch |
| `dataloader_config.normalize_fields` | `['fhr','up','fhr_st','fhr_ph','fhr_up_ph']` | `None` | `None` means normalize **all** fields that have stats, not none |
| `dataloader_config.dataset_kwargs` | see below | `{}` | copied then splatted into `CombinedHDF5Dataset`; a typo'd member is a `TypeError` from the dataset constructor at first loader build |
| `…dataset_kwargs.load_fields` | `['fhr','up','fhr_st','fhr_ph','fhr_up_ph','epoch','guid']` | `None` = every field | |
| `…dataset_kwargs.allowed_guids` | `null` | `None` | GUID allowlist |
| `…dataset_kwargs.cs_label` | `null` | `None` | filter |
| `…dataset_kwargs.bg_label` | `null` | `None` | filter |
| `…dataset_kwargs.epoch_min` / `epoch_max` | `null` | `None` | bounds on the `epoch` field |
| `…dataset_kwargs.label` | `null` | `None` | filter |
| `…dataset_kwargs.cache_size` | `20000` | `2000` | in-memory sample cache |
| `…dataset_kwargs.trim_minutes` | `2.0` | `None` | trims `int(4*60*trim_minutes)` samples per end of raw signals, that `//16` for decimated fields. Must match the stats file's `trim_minutes` or normalization is wrong — mismatch only emits a `warnings.warn` |

Over-restrictive filters raise `ValueError("No samples match the specified filters.")` at dataset construction, not at first batch.

#### 3.4 `advanced_config.trainer`

Required as a `dict` (may be empty). Keys outside `{precision, gradient_clip_val, gradient_clip_algorithm, sync_batchnorm, deterministic, benchmark, compile, log_every_n_steps, num_sanity_val_steps, use_distributed_sampler, profiler}` warn and are ignored.

**This table describes `build_trainer` — and no shipped trainer calls it yet.** `grep build_trainer model/` returns zero call sites: every consumer hand-rolls its own `pl.Trainer(**kwargs)` block (`model/vae_teb_prediction/model/model_raw/trainer_raw_v4.py:530-548`, `model/vae_teb_small/trainer.py:115-133`, `model/vae_teb_prediction/model/trainer_lag_attn_v3.py:785-790` all carry near-identical copies). Editing these keys against an unmigrated consumer does nothing. Check whether the trainer you are touching calls `build_trainer` before trusting this table. Where the hand-rolled copies diverge at runtime:

- `enable_progress_bar` is hardcoded `True`, so a redirected log fills with carriage-return repaints.
- `profiler` is always attached as `SimpleProfiler(dirpath=train_results_dir)` rather than being opt-out.
- `log_every_n_steps: 1`, `num_sanity_val_steps: 0`, `use_distributed_sampler: True` are hardcoded, not read from config.
- `benchmark` is passed through **unreconciled**, so `deterministic: true` + `benchmark: true` silently restores cuDNN autotuning — the one thing `build_trainer`'s `benchmark and not deterministic` exists to prevent.
- Neither `LearningRateMonitor` nor `MLflowRunLoggingCallback` is attached: no `lr` metric, no architecture or final-model MLflow logging.

| Key | Shipped | Code default | Validated | Controls |
| --- | --- | --- | --- | --- |
| `precision` | `"bf16-mixed"` | `'32-true'` | `str` | `Trainer(precision=...)`; also an MLflow tag. bf16-mixed avoids the fp16 GradScaler loss-scale collapse on Ada/Ampere |
| `gradient_clip_val` | `0.5` | `None` (no clipping) | no | `Trainer(gradient_clip_val=...)` |
| `gradient_clip_algorithm` | `"norm"` | `'norm'` | no | `'norm'` \| `'value'` |
| `sync_batchnorm` | `true` | `False` | `bool` | ANDed with `len(cuda_devices) > 1` — SyncBatchNorm's forward needs an initialized process group |
| `deterministic` | `false` | `False` | `bool` | `Trainer(deterministic=...)` **and** `configure_determinism` |
| `benchmark` | `true` | `True` | `bool` | cuDNN autotuning; forced `False` whenever `deterministic` is true |
| `compile` | `true` | — | `bool` | **`_build_trainer_kwargs` never reads it.** Compilation comes from the module's `compile_model` ctor arg |
| `log_every_n_steps` | `1` | `1` | `int` | `Trainer(log_every_n_steps=...)` |
| `num_sanity_val_steps` | `0` | `0` | `int` | `Trainer(num_sanity_val_steps=...)` |
| `use_distributed_sampler` | `true` | `True` | `bool` | keep `true` when using `GraphDataModule` — it relies on Lightning owning the sampler |
| `profiler` | `simple` | `'simple'` | no | `'simple'`/`True` -> `SimpleProfiler(dirpath=train_results_dir)`; `'advanced'` -> `AdvancedProfiler`; `None`/`False`/`'none'`/`'off'`/`''` -> omit the kwarg entirely; anything else warns and disables |

Two config values silently lose to runtime reconciliation, and the YAML will not reflect what the `Trainer` got: `sync_batchnorm: true` becomes `False` on a single device, and `benchmark: true` becomes `False` whenever `deterministic: true`.

#### 3.5 `advanced_config.logging`

Run logs only — metrics go to MLflow. `compression="zip"`, `serialize=False`, `backtrace=True` are hardcoded.

| Key | Shipped | Code default | Validated | Controls |
| --- | --- | --- | --- | --- |
| `console_level` | `INFO` | `'INFO'` | `str` | console sink (rank 0 only) |
| `file_level` | `INFO` | `'INFO'` | `str` | `full.log` sink |
| `json_log` | `true` | `True` | `bool` | writes `run.jsonl` (one object per line, so a killed run still leaves parseable lines) |
| `console_diagnose` | `false` | `False` | `bool` | loguru `diagnose` for the **console sink only**. The file and JSON sinks are hardcoded `diagnose=False` and cannot be enabled from config: diagnose renders every local in a traceback, and a `training_step` frame holds the input batch |
| `rotation` | `"100 MB"` | `'100 MB'` | `str` | loguru rotation size |
| `retention` | `"14 days"` | `'14 days'` | `str` | loguru retention |

#### 3.6 `advanced_config.spike_breaker`

Entirely optional — deleting the block is fine (breaker off, no raise). If present, members are strictly typed. Not consumed by `GraphModelBase`; pass it into your module as the `spike_breaker` ctor kwarg.

| Key | Shipped / default | Validated | Controls |
| --- | --- | --- | --- |
| `enabled` | `false` | `bool` | master switch |
| `multiplier` | `5.0` | `float` | spike when $\ell > \mathrm{multiplier}\cdot\max(\mathrm{EMA}, \mathrm{ema\_floor})$ |
| `ema_decay` | `0.02` | `float` | $\mathrm{EMA} \leftarrow m\,\ell + (1-m)\,\mathrm{EMA}$, accepted batches only |
| `ema_floor` | `0.0` | `float` | floor on the comparison base only |
| `warmup_batches` | `100` | `int` | never skip during the first N batches |
| `max_consecutive_skips` | `25` | `int` | force-accept + hard EMA re-seed after N consecutive skips; `<= 0` disables the escape hatch |
| `comparison_metric` | `total_loss` | `str` | `total_loss` (the returned loss) or `main_loss` (`metrics["main_loss"]`, silently falling back to the returned loss when absent) |

#### 3.7 `advanced_config.tracking.mlflow`

| Key | Shipped | Code default | Validated | Controls |
| --- | --- | --- | --- | --- |
| `enabled` | `false` | `False` | `bool` | master switch; when off `mlflow_logger` is `None`, `lightning_loggers` is `[]`, and the Trainer gets `logger=True` |
| `tracking_uri` | `"http://localhost:5000"` | — | `str` | must start `http://`, `https://`, `file://`, `databricks://` or be a local path whose parent exists; otherwise it is dropped with a warning and MLflow falls back to its default |
| `experiment_name` | `seqvae-teb` | `general_config.tag` | no | a soft-deleted experiment of this name is auto-restored before the run is created |
| `run_name` | `null` | `self.base_folder` (`{stamp}-{tag}`) | no | run name |
| `artifact_location` | `null` | MLflow default | no | `MLFlowLogger(artifact_location=...)` |
| `log_model` | `true` | `False` | `bool` | passed to `MLflowRunLoggingCallback(log_model=...)`: logs + registers the final eager model. **Also the fallback for `log_checkpoints`** |
| `log_checkpoints` | `false` | falls back to `log_model` | `bool` or `str` | Lightning's own checkpoint-weight artifacts; threaded into `MLFlowLogger(log_model=...)` with no `bool()` coercion, so `"all"` survives |
| `log_config_artifact` | `true` | `True` | `bool` | uploads the resolved YAML as artifact `config/resolved_config.yaml` |
| `tags` | `{project: teb_vae, stage: dev}` | `None` | no | merged onto the run; an empty dict becomes `None` |

`_init_mlflow_logger` also passes `save_dir=self.train_results_dir` to `MLFlowLogger` — not config-driven, and it is what anchors a file-based or fallback MLflow store inside the run directory.

Deleting `log_checkpoints: false` while leaving `log_model: true` silently turns checkpoint uploading back on and double-stores weights.

#### 3.8 `advanced_config.callbacks`

| Key | Shipped | Code default | Read by |
| --- | --- | --- | --- |
| `early_stopping.enabled` | `false` | `False` | `_build_trainer_kwargs` |
| `early_stopping.monitor` | `"val/total_loss"` | `'val/total_loss'` | `_build_trainer_kwargs` |
| `early_stopping.patience` | `30` | `30` | `_build_trainer_kwargs` |
| `early_stopping.min_delta` | `0.0001` | `0.0` | `_build_trainer_kwargs` |
| `early_stopping.mode` | `"min"` | `'min'` | `_build_trainer_kwargs` |
| `model_checkpoint.monitor` | `"val/total_loss"` | — | consumer-owned; consumers hardcode the monitor string, so changing this key does nothing |
| `model_checkpoint.save_top_k` | `3` | `2` (consumer) | consumer-owned |
| `model_checkpoint.mode` | `"min"` | — | consumer-owned |
| `loss_plotting.max_history_size` | `1000` | `1000` (ctor) | **dead** — nothing in `train/` reads it |
| `reconstruction_plotting.{enabled, plot_frequency, num_examples}` | `false`, `10`, `3` | — | **dead** — toggling `enabled: false` disables nothing |
| `comprehensive_plotting.{enabled, plot_frequency}` | `true`, `10` | — | **dead** |

`advanced_config.memory` is a denylisted dead block: present-but-ignored, warns "memory monitoring was removed and is no longer wired".

Code defaults contradict shipped values for several keys — deleting the key does **not** keep current behavior: `beta_schedule` (`linear` vs shipped `constant`), `beta_end` (`6.0` vs `1.0`), `early_stopping.min_delta` (`0.0` vs `0.0001`), `model_checkpoint.save_top_k` (`2` vs `3`), `num_workers` (`4` vs `0`), `precision` (`'32-true'` vs `'bf16-mixed'`).

---

### 4. Run lifecycle

`setup_config()`, in order:

1. `validate_config()` — required keys, then known-key types, then dead-key and unknown-key warnings.
2. `configure_determinism()` — `seed_everything(seed, workers=True)`, then per-run backend flags.
3. `os.makedirs` for four directories.
4. `setup_logging(...)` — loguru sinks; `self._log_paths` gets `.text_log` / `.json_log`.
5. Log the pretty config dump.
6. Reset CUDA memory stats.
7. `_init_mlflow_logger()` -> `_log_run_metadata_to_mlflow()` -> `_start_system_metrics_monitor()` -> `_register_run_log_upload()`.

`validate_config()` runs **before** `setup_logging`, so its unknown-key warnings go to loguru's default stderr sink and may never land in `full.log`.

#### On disk

Everything under `<out_dir_base>/<YYYY-MM-DD--[HH-MM]>-<tag>/`:

| Path | Created by `setup_config`? | Contents |
| --- | --- | --- |
| `train_results/` | yes | `full.log`, `run.jsonl`, the profiler output, `default_root_dir` for the Trainer, most callback output |
| `test_results/` | yes | yours |
| `model_checkpoints/` | yes | whatever your own `ModelCheckpoint(dirpath=self.model_checkpoint_dir)` writes |
| `aux_tests/` (`self.aux_dir`) | **no** | writing there without `os.makedirs` raises `FileNotFoundError` |
| `tensorboard_log/` (`self.tensorboard_dir`) | **no** | same |

Run directories are stamped at **minute** resolution. Two runs with the same tag started in the same minute share one directory and interleave checkpoints and plots.

Under DDP, rank 0 keeps `full.log`; other ranks get a `.rank{N}` suffix. `resolve_global_rank` tries `RANK`, then `LOCAL_RANK`, then `SLURM_PROCID`, then `JSM_NAMESPACE_RANK`; a malformed value is skipped and the next variable is tried, with 0 only as the final fallback.

#### MLflow

Off by default. `_init_mlflow_logger` logs the basic hparams `tag`, `lr`, `epochs`, `batch_size_train`, `batch_size_test`, `run_directory` as soon as the connectivity probe succeeds. On the run-creating process, `_log_run_metadata_to_mlflow()` then writes:

- Artifact `config/resolved_config.yaml` (when `log_config_artifact`).
- All of `general_config` + `model_config` flattened to dotted params — nested dicts become dotted keys, non-scalar leaves are `json.dumps`-ed, every value is stringified and truncated to 500 chars (MLflow's param cap).
- Provenance tags: `git_commit`, `git_dirty` (best-effort; silently omitted if git is absent), `torch_version`, `lightning_version`, `mlflow_version`, `cuda_version` (or `'cpu'`), `host`, `world_size` (`len(cuda_devices)`), `precision`, plus `dataset_config.vae_train_datasets` / `vae_test_datasets` truncated to 500 chars.

The two write paths are independent: the basic hparams survive even if the metadata call fails, and vice versa.

Every write is wrapped in try/except and only warns on failure. `_init_mlflow_logger` forces the lazy `MLFlowLogger.experiment` property so a broken tracking server fails at startup rather than silently dropping metrics; `self.mlflow_logger` is only assigned after connectivity is confirmed. An explicit `SystemMetricsMonitor(run_id=...)` is started because Lightning's `MLFlowLogger` never enters a fluent run.

MLflow failure is **fail-closed and non-fatal**: it logs an error plus "Training will continue WITHOUT MLflow logging" and sets `mlflow_logger=None`, `lightning_loggers=[]` — which also silently removes the run-logging callback and the log upload. If you depend on MLflow, assert `graph_model.mlflow_logger is not None` after `setup_config()` yourself. An invalid `tracking_uri` warns and falls back rather than raising, so a typo'd server silently sends the run to a local `./mlruns`.

Log upload runs via `atexit`, which does **not** survive `SIGKILL`/OOM-kill. The local `full.log` + `run.jsonl` are authoritative, not the MLflow copies. Note `upload_run_logs` is intentionally **not** rank-0-guarded (per-rank filenames), while the `log_artifact_to_mlflow` seam **is** (duplicate cross-rank writes).

#### DDP and `TEB_RUN_STAMP`

`_resolve_run_stamp()` writes `TEB_RUN_STAMP` into `os.environ` in the parent so Lightning's `os.environ.copy()` subprocess launcher gives every rank the same output directory. A child inherits the stamp only when `TEB_RUN_STAMP` is set **and** `LOCAL_RANK` is set to something other than `"0"` — deliberately, because Lightning sets `LOCAL_RANK=0` in the parent once `fit` starts, so inheriting on any set `LOCAL_RANK` would make a second run in the same process reuse the first run's directory.

`lean-limit`: this covers Lightning's `ddp`/`ddp_spawn` launchers. Under `torchrun` every rank launches independently — the job script must export `TEB_RUN_STAMP` itself, and even then rank 0 (`LOCAL_RANK=0`) regenerates its own stamp rather than inheriting. They agree only because both render the same minute-resolution string from the same clock, so a launch straddling a minute boundary still splits rank 0 into a sibling directory.

#### What `build_trainer` gives the Trainer

Kwargs: `max_epochs`, `default_root_dir=train_results_dir`, `accumulate_grad_batches`, `precision`, `gradient_clip_val`, `gradient_clip_algorithm`, `enable_checkpointing=True` (hardcoded), `log_every_n_steps`, `num_sanity_val_steps`, `use_distributed_sampler`, `logger=self.lightning_loggers or True`, `sync_batchnorm`, `benchmark`, `deterministic`, `enable_progress_bar=sys.stdout.isatty()`, plus `profiler` when enabled. GPU: `accelerator='gpu'`, `devices=self.cuda_devices`, `strategy=self.select_ddp_strategy(...)`. CPU: `accelerator='cpu'`, `devices=1`, no strategy key.

`enable_progress_bar = sys.stdout.isatty()` means no bar under nohup/SLURM/CI. That is intentional — `TQDMProgressBar` writes to stdout, and megabytes of carriage-return spam in a redirected log is worse than no bar.

Callbacks appended on top of the list you pass:

1. `EarlyStopping` — only when `advanced_config.callbacks.early_stopping.enabled`.
2. `LearningRateMonitor(logging_interval='epoch')` — always.
3. `MLflowRunLoggingCallback(mlflow_logger, experiment_tag, log_model=...)` — only when `self.mlflow_logger is not None` (lazily imported from `train.callbacks` so importing `graph_model_base` never pulls the plotting stack).

`enable_checkpointing=True` is hardcoded, so if you pass no `ModelCheckpoint` Lightning adds a default one writing into `default_root_dir` (= `train_results_dir`), **not** `model_checkpoint_dir`.

`select_ddp_strategy` is an override hook — override it wholesale if your model needs `DDPStrategy(find_unused_parameters=True)`.

**Watch the name: `select_ddp_strategy` vs `_select_ddp_strategy`.** The base hook is the un-prefixed `select_ddp_strategy(self, num_devices, config, model=None)` (`graph_model_base.py:388`), and `_build_trainer_kwargs` looks up **only** that name (`:499`). Every SeqVAE consumer separately defines an underscore-prefixed staticmethod of the same purpose but a model-specific signature, and calls it itself while hand-rolling its `Trainer`: `trainer_lag_attn_v1.py:807` (`likelihood, sigma_obs, curriculum_enabled`), `trainer_lag_attn_v3.py:590` (adds `head_structured_latent`, `freeze_unused_attn_proj`; `trainer_raw_v4` inherits it), `pl_module_v2.py:1228`. Carrying `_select_ddp_strategy` over when you move onto `build_trainer` is a **silent no-op** — the underscore version is never looked up, and you fall back to the base default. A model needing `ddp_find_unused_parameters_true` (dead logvar heads, structured latent, curriculum) must override the un-prefixed name. This fails only on a multi-GPU box, as a DDP reducer error; a single-GPU dev run cannot catch it.

---

### 5. Callbacks

`train/callbacks.py` exports 4 callbacks, all model-agnostic — they read only `trainer.callback_metrics` and `nn.Module` APIs. Only `MLflowRunLoggingCallback` is wired automatically. Every plotting callback and every `ModelCheckpoint` must be constructed by hand in your `train_model()` and passed to `build_trainer`.

`log_artifact_to_mlflow(mlflow_logger, path, trainer)` — in `utils/mlflow_utils.py` — is the single shared upload seam: no-op when the logger is `None`, the trainer is `None`, or `not trainer.is_global_zero`; exceptions are logged as a warning and swallowed so plotting can never kill training. It sits in `utils/` rather than here because the SeqVAE plotters need it too, and `utils/` may not import `train/` (see the layering note below).

| Callback | Fires on | Writes | MLflow upload? |
| --- | --- | --- | --- |
| `LossPlotCallback(output_dir, plot_frequency=5, max_history_size=1000, *, metric_filters=None, hyperparam_keys=None, mlflow_logger=None)` | `on_validation_epoch_end` | `loss_plot_epoch.html`, `hyperparameters_evolution.html` (Plotly) | yes |
| `HyperparameterLoggingCallback(tracked_keys=None, *, output_dir, plot_frequency=10, mlflow_logger=None)` | `on_train_epoch_end` | `hyperparameters.html` | yes |
| `MetricsLoggingCallback(tracked_metrics=None)` | `on_validation_epoch_end` | nothing — in-memory only, read via `.as_dict()` | no |
| `MLflowRunLoggingCallback(mlflow_logger, experiment_tag, log_model=False)` | `on_fit_start` / `on_fit_end` | `model/model_architecture.txt`, param `model_class`, metrics `params_total`/`params_trainable`; at fit end `mlflow.pytorch.log_model(...)` | yes |

**The SeqVAE plotters live elsewhere.** `ReconstructionPlotCallback`, `PlottingCallBack` and `PlottingAvgPredCallBack` are hard-coupled to the SeqVAE model and its 4 Hz FHR/UP batch contract, so they live in `utils/seqvae_plot_callbacks.py`, not here. That module's docstring documents exactly what they demand of a model and the limitations they carry. Keeping this file agnostic is the point: if you are adding a callback that names a batch field or an output key, it does not belong in `train/callbacks.py`.

**Layering.** Dependencies run one way — `utils/` <- `train/` <- `model/`. `utils/` holds leaf helpers and imports no other first-party package; `train/` may use `utils/` but never `model/`; consumers use both. `train/tests/test_layering.py` enforces this by walking imports (including lazy ones inside functions), so an upward import fails a test rather than passing review. This is why the artifact seam lives in `utils/mlflow_utils.py`: both `train/callbacks.py` and the SeqVAE plotters need it, and the plotters cannot reach up into `train/` to get it.

Defaults worth knowing:

- `LossPlotCallback.metric_filters` = `("train/*", "val/*")`; it skips names ending `_step`, names in `hyperparam_keys`, and the literal `epoch`. `hyperparam_keys` = `("hyperparams/beta", "hyperparams/lr", "hyperparams/kld_beta", "lr", "learning_rate", "kld_beta", "beta")`.
- `MetricsLoggingCallback.tracked_metrics` = `("train/total_loss", "val/total_loss", "kld_beta", "learning_rate")`.
- `HyperparameterLoggingCallback.tracked_keys` = `("hyperparams/beta", "kld_beta", "hyperparams/lr", "lr", "learning_rate")`; it falls back to `trainer.optimizers[0].param_groups[0]['lr']` when an `lr`-containing key is missing from `callback_metrics`.

Gotchas:

- **Inconsistent kwarg naming, now across two modules.** `LossPlotCallback`/`HyperparameterLoggingCallback` here take `plot_frequency`, as does `ReconstructionPlotCallback` in `utils/seqvae_plot_callbacks.py` — but `PlottingCallBack`/`PlottingAvgPredCallBack` in that same module take `plot_every_epoch`. Wrong keyword = `TypeError`.
- `HyperparameterLoggingCallback`'s `output_dir` is **keyword-only and required**: `HyperparameterLoggingCallback(keys, out_dir)` raises `TypeError`.
- **Rank behavior differs per callback.** `LossPlotCallback` accumulates history on all ranks and only guards plotting. `HyperparameterLoggingCallback` returns before recording anything off rank 0 — its history is empty there. `MetricsLoggingCallback` has no rank guard at all, so any downstream CSV writer needs its own `is_global_zero` guard.
- `LossPlotCallback.plot_hyperparameters` only ever renders the two literal keys `hyperparams/beta` and `hyperparams/lr`. Logging beta as plain `beta` or `kld_beta` gets it tracked but never plotted, and the figure is skipped with no warning.
- These four are Plotly-only; **importing `train.callbacks` no longer pulls matplotlib or forces the `Agg` backend.** Nothing should depend on that former side effect — `train.graph_model_base` sets `Agg` itself at import (`graph_model_base.py:35`), as does `utils/seqvae_plot_callbacks.py`.
- `MLflowRunLoggingCallback.on_fit_end` registers into the model registry under `experiment_tag`, so every run with the same tag creates a new registry version. It always logs the **eager** module via `getattr(pl_module, "orig_model", pl_module)`, so the state dict carries no `_orig_mod.` prefix and reloads with `mlflow.pytorch.load_model`. No signature is logged.

---

### 6. Data

`GraphDataModule(config)` is opt-in and not wired into any entry point. Its only constructor arg is the whole resolved config dict; it mutates nothing and reads exactly three things: `config['dataset_config']`, `dataset_config['dataloader_config']`, `config['general_config']['batch_size']`.

It implements only `train_dataloader` / `val_dataloader` / `test_dataloader` — no `prepare_data`, no `setup`, no `predict_dataloader`. `val` and `test` read the identical `vae_test_datasets` list; there is no separate test split and no in-process randomized split.

It deliberately builds **plain** loaders — `create_optimized_dataloader` is called at its default `world_size=1`, so no `DistributedSampler` is created. Lightning's `use_distributed_sampler` (default `true`) injects the sampler and calls `set_epoch` each epoch. Keep that config key `true`. Conversely, passing `world_size > 1` to `create_optimized_dataloader` directly builds its own `DistributedSampler` with `drop_last=True`, which would collide with Lightning and drop the eval tail.

`_make_loader` maps: `num_workers` (default 4), `stats_path=dataset_config['stat_path']`, `normalize_fields` (default `None`), `prefetch_factor` (default 2), plus a defensive copy of `dataset_kwargs`.

#### Batch contract

A batch is an `AttributeDict` (dict subclass with attribute access — `batch['fhr']` and `batch.fhr` both work) produced by `default_collate` via `attribute_dict_collate`. The underlying loader uses `drop_last=False`, `multiprocessing_context='spawn'`, `persistent_workers=True` when `num_workers > 0`.

| Key | Batch shape / type | Note |
| --- | --- | --- |
| `fhr`, `up` | `(B, len_signal)` float32 | raw 4 Hz |
| `fhr_st` | `(B, len_sequence, 43)` float32 | **already transposed** to (seq, channels) in `__getitem__` |
| `fhr_ph` | `(B, len_sequence, 44)` float32 | same |
| `fhr_up_ph` | `(B, len_sequence, n_cross_phase_channels)` | same |
| `up_st`, `up_ph` | `(B, len_sequence, C)` | same |
| `target`, `weight` | `(B, len_sequence)` | |
| `epoch` | `(B,)` float32 | |
| `guid` | `list[str]` | decoded to `str`, **not a tensor** |
| `cs_label`, `bg_label` | Python `bool` | **not tensors** |
| `time_from_labor_onset` | float32 | |
| `source_file`, `source_file_basename` | `list[str]` | **injected regardless of `load_fields`** |
| `source_file_index` | int | injected regardless |

On-disk HDF5 layout is `(N, C, T)`; `__getitem__` transposes `fhr_st`/`fhr_ph`/`fhr_up_ph`/`up_st`/`up_ph` to `(seq, channels)`. **Do not permute again in the model.**

Because `guid`, `source_file` and `source_file_basename` are non-tensor entries, anything doing `{k: v.to(device) for k, v in batch.items()}` will crash.

Only `fhr`, `up`, `fhr_st`, `fhr_ph`, `fhr_up_ph`, `up_st`, `up_ph` are normalized when a stats file is loaded.

Other operational notes: the dataset's sample cache is **FIFO, not LRU** (it evicts `next(iter(self._cache))` despite `_access_count` bookkeeping) and returns cached entries **by reference**, so in-place mutation of a batch element corrupts the cache. `pin_memory=True` is the dataset default and every sample tensor is pinned in `__getitem__`; combined with `cache_size: 20000` and `spawn` + `persistent_workers`, each worker keeps its own independent pinned cache. `GraphDataModule` constructs a brand-new `CombinedHDF5Dataset` on every `*_dataloader()` call, re-scanning and re-filtering the HDF5 index — Lightning calls these once per fit, but calling them manually is not free.

---

### 7. Checkpoints

#### The stamp

`LightningModelBase.on_save_checkpoint` sets `checkpoint["model_class"] = type(self._orig_model).__name__` — deliberately the **eager** class, so no `_orig_mod.` compile prefix appears. It mutates in place and returns `None`. An override that does not call `super().on_save_checkpoint(checkpoint)` drops the stamp that downstream tooling relies on.

**Nothing under `train/` reads the stamp back** — grepping here for the reader finds nothing. The only enforcement point lives in the consumer tree: `check_model_class(ckpt, active_cls_name)` at `model/vae_teb_prediction/model/vae_teb_lag_attn_trfr.py:224` raises `ValueError` when a checkpoint's stored `model_class` disagrees with the active class, and only warns when the field is absent (pre-guard checkpoints). The three `graph_models_utils` loaders all read tensor state and never inspect the class name. (`train/callbacks.py` does log a `model_class` MLflow param, but from `type(module).__name__` — not from the checkpoint.)

#### Saving

The framework creates no `ModelCheckpoint`. Build your own with `dirpath=self.model_checkpoint_dir` and `monitor="val/total_loss"` (the stage-prefixed key). `advanced_config.callbacks.model_checkpoint` is consumer-owned; consumers hardcode the monitor, so changing that key silently does nothing.

#### Loading

`GraphModelBase` has **no** resume support: no `ckpt_path` handling, no `resume_from_checkpoint`. Loading weights is your `create_model()`'s job; resume means passing `ckpt_path=` to your own `trainer.fit(...)`. (`graph_model_base.py` carries two dead imports that suggest otherwise — `load_checkpoint_torch` on line 5 and `DistributedDataParallel as DDP` on line 2 are both imported and never used.)

`train/graph_models_utils.py` exposes exactly three public functions:

```python
from train.graph_models_utils import (
    load_checkpoint_strict,   # prefer this
    load_checkpoint_torch,    # legacy
    denormalize_signal_data,
)

loaded = load_checkpoint_strict(model, "epoch=42.ckpt", map_location="cpu")
if loaded is None:
    raise RuntimeError("checkpoint did not align with any discovered module")
```

`load_checkpoint_strict(model, checkpoint, *, map_location='cpu', module_attr_names=None) -> Optional[nn.Module]`: `checkpoint` may be a path, a wrapper object, or a raw state dict. It BFS-discovers nested modules via `model, module, _orig_mod, wrapped_module, pytorch_model, lightning_module, base_model, network, net` (plus any `module_attr_names`) and loads into the first candidate whose keys, shapes and dtypes align exactly, with `strict=True`. Returns the original model on success, `None` on any misalignment. It calls `torch.load(..., weights_only=False)` explicitly.

`load_checkpoint_torch(model, checkpoint_path)`: two positional args, no defaults, always returns the same `model` mutated in place. Pipeline: `torch.load(..., map_location="cpu")` -> `_extract_state_dict` -> `OrderedDict` -> `_normalize_checkpoint_state_dict` -> `_clean_state_dict` -> `load_state_dict(cleaned, strict=False)`, then loguru warnings on missing/unexpected keys.

Three reasons to prefer the strict variant:

- `load_checkpoint_torch` calls `torch.load` **without** `weights_only=False`. On torch >= 2.6 (this project pins 2.7.1) `weights_only` defaults to `True`, so it raises `UnpicklingError` on any Lightning `.ckpt` that pickles non-tensor objects.
- It uses `strict=False` and only *logs* — a checkpoint matching nothing returns the model happily with random weights, with no return-value or exception signal.
- It does `OrderedDict(state_dict)` with no `None` check, so an unrecognized layout raises a bare `TypeError` from the `OrderedDict` constructor.

Both loaders share the same sanitising pipeline, and the `_orig_mod.` entries in its prefix lists are what make compiled and eager checkpoints interoperate: `_normalize_checkpoint_state_dict` rewrites `model._orig_mod.` -> `model.` and `_orig_mod.` -> `` (plus the `_orig_model.` / `model.model.` variants), then `_clean_state_dict` strips the compile prefixes alongside the DDP/wrapper ones. That pre-empts the classic `torch.compile` failure where a checkpoint saved from a compiled module will not load into an eager one — and it is why a checkpoint trained on the prod box under `compile_model=True` still loads on the Windows dev box under `compile_model=False`. The guarantee covers only these two helpers: Lightning's own `ckpt_path=` resume does no prefix rewriting, so toggling `compile_model` across a resume is **not** covered.

Also note `_clean_state_dict` strips prefixes in a **while-loop until no prefix matches**, and the list includes the very generic `'model.'`, `'module.'`, `'net.'`, `'network.'`. A model with a legitimate top-level submodule named `net`/`network`/`module`/`model` silently loses that level, and `model.model.model.x` collapses all the way to `x`. Aggressive by design for this repo's wrappers; not safe for arbitrary architectures.

`denormalize_signal_data(normalized_data, field_name, normalization_stats)` supports only `field_name` `'fhr'` or `'up'` — anything else, or a missing stats entry, logs a warning and returns the input unchanged. The transform is `normalized * (std + 1e-8) + mean`, preferring `mean_tensor`/`std_tensor` and falling back to scalar `mean`/`std`.

#### `orig_model` outside Lightning

`self.model` is `torch.compile(base_model)` when `compile_model=True` (the default); `self._orig_model` always holds the uncompiled module and is exposed read-only as `self.orig_model`. Use `self.orig_model` — not `self.model` — for `state_dict` export, model-registry logging, and any non-forward helper method, because the compile wrapper prefixes keys with `_orig_mod.`.

---

### 8. Operational knobs and gotchas

#### Determinism

`configure_determinism()` reads `general_config.seed` (default $42$) and `advanced_config.trainer.deterministic` (default `false`) — note the two live in different blocks. It calls `seed_everything(seed, workers=True)` then sets, per run and never at import:

| Flag | `deterministic: true` | `deterministic: false` |
| --- | --- | --- |
| `cudnn.benchmark` | `False` | `True` |
| `cudnn.deterministic` | `True` | `False` |
| `cudnn.allow_tf32` | `False` | `True` |
| `torch.set_float32_matmul_precision` | `"highest"` | `"high"` |

`benchmark` is additionally reconciled as `benchmark and not deterministic` in the Trainer kwargs, so a `Trainer` cannot re-enable cuDNN autotuning behind `configure_determinism`'s back. These are process-global mutations.

#### `torch.compile`

Driven by the module's `compile_model` ctor kwarg, **not** by `advanced_config.trainer.compile` (validated, listed as known, never read). `torch.compile` is lazy — the backend only runs on first forward — so either path is cheap to construct.

Pass `compile_model=False` when compile is incompatible with your model. Two distinct reasons this comes up here:

- **On the Windows dev box, always.** Triton is Linux-only, so the default `compile_model=True` raises `torch._inductor.exc.TritonMissing` on the first forward. The shipped `compile: true` targets the Linux prod box.
- **With activation checkpointing, on any platform** — this, not triton, is why the lag-attention models run eager on the *Linux* prod box. When a module wraps an inner call in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` (e.g. `LagCrossAttention.forward` under `attention_grad_checkpoint`) and the outer module is compiled, AOT autograd's `min_cut_rematerialization_partition` asserts `Node <name> was invalid, but is output` on the first backward: forward-only nodes inside the checkpointed region get marked as backward-graph outputs across the partition cut. Disabling `torch._dynamo.config.optimize_ddp` does not fix it — the assert reappears on `constant_pad_nd_1` from the lag memory builder's `F.pad`. Any architecture relying on activation checkpointing to fit in memory (the lag models need it for the ~900 MB lag memory bank at $B=64$) must run eager.

`trainer_lag_attn_v1.py:88-109` and `model_experiment/synthetic/pl_module_synth.py:16-21` carry the full diagnosis, but they still use the older hack of bypassing `LightningModelBase.__init__` to reach the grandparent `LightningModule.__init__`. Use the ctor kwarg instead — see §2.2.

#### Spike breaker

Fires inside `_dispatch_stage_step` and **only** on the train stage — fabricating a val loss would corrupt `ModelCheckpoint` selection. Config is read from `self.hparams.get("spike_breaker", None)` per step, so it can be mutated at runtime.

Mechanics:

- `is_spike_local = loss_value > multiplier * max(ema_before, ema_floor)`, evaluated only after `_spike_batches_seen > warmup_batches` and once the EMA is seeded; otherwise `False`.
- A non-finite value forces a spike regardless of warmup, checking **both** the watched value and the actually-returned loss.
- EMA updates only on accepted batches, seeded to `loss_value` on the first accepted batch.
- DDP: the skip flag is reduced with `ReduceOp.MAX` (skip if **any** rank skips); the force-accept flag with `ReduceOp.MIN` (force-accept only if **every** rank agrees). No-op at `world_size <= 1`.
- Escape hatch: `forced_local = (not is_nonfinite) and max_consecutive_skips > 0 and self._spike_consecutive >= max_consecutive_skips`. On force-accept the EMA is **hard re-seeded** to the current loss and the run counter resets. So persistent NaN produces an unbounded skip run by design.
- On a skip, poisoned metrics are overwritten with the finite EMA before logging (`metrics["total_loss"] = safe_loss`, and `metrics["main_loss"]` in `main_loss` mode), so a NaN never reaches the logger and poisons the epoch aggregate.
- Diagnostics injected into the metrics dict: `spike_ema_loss` (instantaneous EMA, `0.0` if unseeded) and `spike_skipped` (`1.0`/`0.0`, whose epoch mean is the skip rate) — logged as `train/spike_ema_loss` and `train/spike_skipped`. Cumulative `_spike_skips_total` / `_spike_forced_accepts_total` are instance attributes and deliberately not logged (an `on_epoch` mean of a running total is meaningless).

Three things that bite:

1. **A skip is a zero-gradient step, not a true skip.** The returned loss is `zero_grad_loss * 0.0`, accumulating `torch.nan_to_num(param).sum()` over every `requires_grad` parameter so DDP's already-armed reducer still fires a hook per parameter; returning `None` would desynchronise it. Under automatic optimization the optimizer still steps, so AdamW's decoupled weight decay and carried momentum still nudge weights on a "skipped" batch. Asserting `out is None` for a skipped batch will fail.
2. **`ema_floor: 0.0` plus a collapsing loss is a deadlock.** The EMA drags toward zero, the `loss > multiplier * EMA` threshold becomes un-clearable, and the run 100%-skips forever; under DDP the MAX-reduce makes one rank's collapse permanent for all. Set a non-zero `ema_floor` when the loss can collapse (e.g. gaussian NLL variance collapse). `max_consecutive_skips <= 0` disables the only escape.
3. **Breaker state does not survive a resume.** `_spike_ema_loss`, `_spike_consecutive`, `_spike_batches_seen` and the totals are plain instance attributes — not buffers, not written by `on_save_checkpoint`. A resumed run restarts warmup and the EMA from scratch, so the breaker is blind for the first `warmup_batches` (default $100$) batches.

#### Cross-cutting gotchas

- **Dataloaders are yours.** The base never touches them. `use_distributed_sampler=True` is the default, but nothing checks that your loaders are compatible with it.
- **`self.cuda_devices` drives everything DDP** — `devices`, `select_ddp_strategy(len(cuda_devices), ...)`, `sync_batchnorm` gating, and the `world_size` tag. A single-element list gives `strategy='auto'` (no DDP).
- **Unknown/misspelled config keys never fail the run.** `advanced_config.trainer.gradient_clipping` or a typo'd block produces a loguru warning and is silently ignored — and because `validate_config` runs before `setup_logging`, that warning may not even land in `full.log`.
- **`self.hparams.get(...)` hides config typos.** A key that `apply_config_hyperparameters` failed to set silently returns the default rather than raising.

---

### 9. Testing

```bash
# Full framework suite (~9s, no GPU, no MLflow server needed):
.venv/Scripts/python.exe -m pytest train/tests -q
#   -> 103 passed, 1 skipped

# Show why the one test skipped (there is no -m selector and no --runslow flag):
.venv/Scripts/python.exe -m pytest train/tests -q -rs

# The only server-dependent test, opt-in:
MLFLOW_SMOKE_URI=http://<host>:<port> \
  .venv/Scripts/python.exe -m pytest train/tests/test_mlflow_smoke.py -q

# Regression gate after ANY change to the base classes or config.yaml:
.venv/Scripts/python.exe -m pytest model/vae_teb_prediction/model/tests -q
```

There is no `pytest.ini`/`setup.cfg`/`tox.ini` and no `[tool.pytest]` section anywhere — nothing to register, no markers. `cwd` does not matter: `conftest.py` resolves the repo root from `__file__` (it removes the repo root from `sys.path` then re-inserts it at index 0, and eagerly imports `utils` to pin its `__path__` so a near-empty sibling package cannot shadow it). It exposes one fixture, `config_path`, pointing at the real shipped `train/config.yaml`.

What the suite guarantees: clean imports; compile is opt-out and compiles exactly once; the `model_class` stamp survives compile and a `super()`-calling override; an unmigrated-style consumer still trains a real CPU epoch; `_build_trainer_kwargs` reproduces the consumer trainer configs (precision, clipping, profiler on/off, sync_batchnorm gating, strategy selection); `validate_config` fails fast on missing/mistyped keys and only warns on unknown/dead ones; all four determinism flags flip together; the `configure_param_groups` seam handles flat lists, group dicts and generators; the warmup scheduler decays at the **absolute** milestone; the spike breaker is train-only, zero-gradient, EMA-clean, floor-clamped, escape-hatched and MAX/MIN-reduced; the artifact seam is rank-guarded and swallows errors; MLflow flattening/provenance/run-id-binding/`"all"`-passthrough all hold with zero server contact; per-rank log isolation and forced `diagnose=False` on disk sinks; `GraphDataModule` passes no `world_size`.

Writing a test for your own trainer. These belong under `train/tests/` — the `config_path` fixture lives in `train/tests/conftest.py` and pytest does not share fixtures across sibling directories, so a test placed next to your own trainer must define it locally:

```python
import torch
from train.test_utils import (
    FakeMLflowLogger, FakeStrategy, FakeTrainer, TinyLightningModel, make_graph_model,
)


def test_my_trainer_kwarg(config_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gm = make_graph_model(config_path, **{"advanced_config.trainer.num_sanity_val_steps": 2})
    kw = gm._build_trainer_kwargs([])       # assert on the dict, never a real Trainer
    assert kw["num_sanity_val_steps"] == 2


def test_anything_calling_setup_config(config_path, tmp_path):
    gm = make_graph_model(config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)})
    gm.setup_config()
```

```python
# Outside train/tests/, define the fixture yourself:
@pytest.fixture
def config_path():
    return Path(REPO_ROOT) / "train" / "config.yaml"
```

`make_graph_model(config_path, **dotted_overrides)` builds a concrete `GraphModelBase` subclass with both abstract methods stubbed, so you can call `validate_config` / `configure_determinism` / `_build_trainer_kwargs` with no training run. Also available: `TinyLightningModel` (a concrete `LightningModelBase` with an MSE loss over a 4->8->4 MLP), `StandInConsumer` (bypasses `LightningModelBase.__init__` on purpose), `FakeMLflowLogger` / `FakeMLflowExperiment` (records every client call as `(method, run_id, payload)` in `.calls`), `FakeTrainer`, `FakeStrategy` (records `reduce_op` into `.reduce_calls`).

Testing gotchas:

- **Never build a real `lightning.Trainer` from the shipped config.** It names 7 CUDA devices while the dev box has one GPU; `Trainer` validates against hardware and fails. Assert on `_build_trainer_kwargs([])`.
- **`setup_config()` in a test needs `out_dir_base` overridden** to `tmp_path`, or it tries to `mkdir` under the prod path. `__init__` never writes, which is why most tests skip the override. `make_graph_model` special-cases this one override to re-derive all five run subdirectories.
- **Dotted overrides land in `gm.config`, but `__init__` caches some values as attributes.** Only `cuda_devices`, `epochs` and `accumulate_grad_batches` are re-synced. Any other cached general-config attribute you override silently does not take effect — extend the re-sync block in `test_utils.py`.
- **`TinyLightningModel` defaults to `compile_model=False`**, inverting `LightningModelBase`'s `True`. Do not infer the production default from the helper.
- **`test_determinism.py` mutates process-global torch backend flags and never restores them.** Set the flags you depend on inside your own test.
- **The loguru reset fixture is local to `test_logging.py`.** A logging test in another file must reset loguru itself — on Windows `logger.remove()` is also what releases file handles so `tmp_path` can be unlinked.
- **A green `train/tests` run is not evidence a DDP change is safe.** Multi-GPU behavior is deliberately out of scope — every DDP test uses `FakeStrategy`. The `sync_batchnorm` gate, sampler sharding and spike-breaker skip-sync are only exercised on the real box.

---

### 10. Checklists

- **Authoring a new model** — `train/MODEL_MIGRATION_GUIDE.md` §3, ending in a checklist. §1 there defines the layering (`utils/` <- `train/` <- `model/`) and where each kind of code goes.
- **Migrating an existing consumer** — `train/MODEL_MIGRATION_GUIDE.md` §4–§8.
- **The regression gate** after any base-class or `config.yaml` change: `pytest train/tests -q`, then `pytest model/vae_teb_prediction/model/tests -q`.
- **Multi-GPU smoke** is manual on the 8xA6000 box: bf16 loss sanity, disjoint sampler sharding, spike-breaker skip sync, per-variant `build_trainer`, MLflow record.

> Two checklist files, `train/tests/RUN_MODEL_TESTS.md` and `train/tests/PROD_SMOKE_CHECKLIST.md`, were referenced here and in the file map above but were never written. The four bullets above are what they were meant to say; the gap is tracked in `train/MODEL_MIGRATION_GUIDE.md` §8.
