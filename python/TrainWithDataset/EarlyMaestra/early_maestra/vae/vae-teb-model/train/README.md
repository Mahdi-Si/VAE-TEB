## Unified Lightning + Native PyTorch Workflow

Fast notes for anyone wiring up training or inference across the mixed PyTorch + Lightning stack. Grammar sacrificed for clarity.

### Pipeline TL;DR
- edit `train/config.yaml` (general/model/dataset/advanced blocks) then point `GraphModelBase` subclass at it
- call `setup_config()` once -> logging + output folders land under `out_dir_base/timestamp-tag`
- subclass implements `create_model()` -> instantiates raw `nn.Module` (e.g., `SeqVaeTeb`), freezes pieces via `configure_trainable_params()`, optionally loads a checkpoint with `graph_models_utils.load_checkpoint_torch`
- wrap raw model in `model.pytorch_lightning_modules.LightSeqVaeTeb` (extends `train/pl_model_base.LightningModelBase`) for Lightning runs or keep `orig_model` for plain PyTorch loops
- build dataloaders from `hdf5_dataset` utilities, launch `lightning.Trainer` with callbacks from `train/callbacks.py`
- after training: `LightningModelBase.orig_model` or the saved `*_orig_state_dict.pt` keeps compatibility with non-Lightning inference scripts

### GraphModelBase (`train/graph_model_base.py`)
- `__init__(config_file_path=None)`: load YAML config, stash hyperparams (epochs, lr, milestone, batch sizes, grad accumulate), derive timestamped output dirs, setup CUDA device list, track plotting cadence.
- `setup_config()`: make results/checkpoint dirs, initialize Loguru sinks via `utils.custom_logger.setup_logging`, reset CUDA memory stats when available.
- `create_model()` (abstract): instantiate raw `nn.Module`, run `configure_trainable_params()`, wrap into a Lightning module; concrete class decides how to merge config knobs with model constructors.
- `configure_trainable_params(model)`: shared place to freeze/thaw weights before optimizer sees them; helper `freeze_model()` toggles `requires_grad=False` everywhere.
- `set_cuda_devices(device_list=None)`: override GPU topology from config when multi-node experiments need manual overrides.
- `apply_config_hyperparameters(hparams_dict, lightning_module)`: force Lightning `hparams` (lr, milestone list, beta schedule, horizon, etc.) to match YAML even after checkpoint restores; logs shared when overrides applied.
- `train_model(train_loader, validation_loader)`: abstract entry point; subclass typically spins up Lightning `Trainer` with the pl module + callbacks.
- extra runtime hygiene: `plot_every_epoch`, `clip=10`, `accumulate_grad_batches`, constant Matplotlib/torch backend tweaks to keep deterministic flags centralized.

### LightningModelBase (`train/pl_model_base.py`)
- constructor saves hyperparams (lr, milestones, weight decay, optional name), keeps both compiled model (`self.model = torch.compile(base_model)`) and unwrapped copy (`self.orig_model`); compiled path accelerates training, raw path stays portable.
- `forward(*args, **kwargs)`: thin pass-through into compiled core; PyTorch-only scripts call `orig_model` methods directly.
- `training_step` / `validation_step` / `test_step`: all route through `_dispatch_stage_step()` so every stage gets shared logging + metric formatting; implementers only supply `compute_loss_and_metrics(batch, batch_idx, stage)`.
- `compute_loss_and_metrics` contract: return `(loss_tensor, metrics_dict)` per stage; metrics auto-prefixed with `train/`, `val/`, or `test/` and optional sync across ranks defined by `sync_dist_stages`.
- Optimizer stack: `configure_optimizers()` grabs filtered params via `_trainable_parameters()`, logs totals with `_log_parameter_overview()`, builds default `AdamW` in `build_optimizer()` and optional `MultiStepLR` from `build_lr_scheduler()` (milestones + gamma from `hparams`).
- Diagnostics: `_log_learning_rate()` pushes LR telemetry every epoch; `_log_metrics()` handles tqdm vs logger routing using `prog_bar_metrics`; `_as_tensor()` guards scalar conversion.
- Hooks: `_on_train_epoch_start_hook()` gives subclasses an easy override for beta schedules or reset logic, while `on_save_checkpoint()` mirrors native weights to `<wrapper>_epoch=XXXX_orig_state_dict.pt` so raw PyTorch checkpoints always exist beside Lightning `.ckpt` files.
- Compatibility helpers: `_trainable_parameters()` respects any manual freezing done inside `GraphModelBase`, `_should_log_on_prog_bar()` keeps the progress bar tight, `.orig_model` ensures experimentation notebooks can keep using the plain module API.

### Callback Suite (`train/callbacks.py`)
- helper utilities `_resolve_validation_dataloader`, `_first_validation_batch`, `_metric_to_float` abstract dataloader discovery + tensor-to-float conversion so callbacks stay agnostic to trainer/datamodule wiring.
- `LossPlotCallback`: captures `trainer.callback_metrics` every validation epoch, filters via glob patterns (default `train/*`, `val/*`), tracks hyperparams like `hyperparams/beta` and `lr`, trims history, and writes Plotly HTML (`loss_plot_epoch.html`, `hyperparameters_evolution.html`) optionally uploading via MLflow.
- `HyperparameterLoggingCallback`: lightweight tracker for selected metric keys (defaults to beta + LR variants), `on_train_epoch_end` only runs on global zero, `plot_hyperparameters()` dumps one Plotly figure if requested post-run.
- `MetricsLoggingCallback`: simple buffer for arbitrary metric names (default total losses + lr/beta) evaluated on validation end; stored history is retrievable via `.as_dict()` for notebooks or CSV exports.

### Mixing Lightning and Native PyTorch
- to continue fine-tuning outside Lightning, grab `trained_lightning_module.orig_model`, put it in eval/train mode, and reuse plain PyTorch optimizers or inference scripts; weights always stay aligned because `on_save_checkpoint()` keeps raw `state_dict` dumps.
- when importing historical `.pt` or `.ckpt` files, prefer `graph_models_utils.load_checkpoint_torch()` to sanitize prefixes (`model._orig_mod.*`, DDP shards, etc.) before calling `.load_state_dict()` on either Lightning wrappers or raw modules.
- inference services can skip Lightning entirely: call `GraphModelBase.create_model()` (or manually instantiate `SeqVaeTeb`), load weights with the util above, run forward passes -- no need to bring in Trainer or callbacks.

### Suggested Usage Patterns
- development loop: edit config -> run `python -m model.graph_model_train` -> inspect artifacts under `train_results/` and callback-generated HTML -> iterate.
- research notebooks: instantiate the same config + GraphModelBase subclass, call `create_model()` to get Lightning module, but call `.orig_model` when you only need the plain PyTorch VAE for ablation scripts.
- production eval: restore Lightning checkpoint via `LightSeqVaeTeb.load_from_checkpoint(...)`, immediately call `apply_config_hyperparameters()` with the current YAML block to ensure lr/beta/horizon align with deployment expectations, then either call `trainer.test(...)` or export the raw module.
