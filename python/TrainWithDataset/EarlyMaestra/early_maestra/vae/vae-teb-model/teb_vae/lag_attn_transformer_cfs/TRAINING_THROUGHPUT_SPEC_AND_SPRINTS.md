# Training throughput of lag_attn_transformer_cfs - Spec and Roadmap

Status: IN_PROGRESS
Last updated: 2026-09-23
Baseline: repository root `C:\Users\mahdi\Desktop\teb_vae_model`, revision `2a2bc78` (2026-09-23), working tree dirty with untracked documents and fixtures, none of them touched by this work. Inspected: `teb_vae/lag_attn_transformer_cfs/` (trainer, task, nets, configs, tests), the shared driver, task and objective it inherits (`teb_vae/lag_attn_rws/`, `teb_vae/lag_attn_cfs/`, `teb_vae/lag_attn_fs/nets/feature_target.py`, `teb_vae/lag_attn/nets/`), the framework (`train/`), the loader (`hdf5_dataset/hdf5_dataset.py`), the shard writer (`hdf5_dataset/new_pipeline/create_new_pipeline.py`), `requirements.txt`, and the installed Lightning 2.6.1 logger connector.
Profile: full

## How to use this document

This file is the single source of truth for this work. Read it once top to bottom, then
work from Resume and the Todo checklist.

1. Authority: requirements say what must be true; task blocks say what to change and how
   to verify it; the Todo checklist is the only place task state lives. Task blocks hold
   evidence, never a status.
2. Pick work: start with `Next task` in Resume. A task is executable only when every task
   in its `Depends on` is checked and it has no `Blocker:` line.
3. Re-ground before coding: confirm the paths and symbols in `Files affected` still exist
   and follow `Re-ground` in Resume.
4. Implement the task as one coherent change. Keep sprint labels, task IDs, and this
   file's name out of production code, comments, and user-facing text.
5. Validate with the task's `Validation` command from its stated cwd. A command written
   here is not a run. Record what actually ran and its actual result.
6. Record evidence in the task block, one line per run, newest last:
   `Evidence: YYYY-MM-DD | <command or inspection> | pass|fail|partial | <revision or artifact>`
   Check the task `[x]` only when acceptance criteria hold and every required check
   passed. Otherwise leave it unchecked with `[in_progress]`, or `[blocked]` plus a
   `Blocker:` line naming the resolution.
7. After each task: overwrite the Resume fields, append one dated line to Change log,
   update `Last updated`, and update `Status` if the document state changed.
8. Never rename or renumber an ID. New work gets a fresh ID. A moved task keeps its ID.
   A replaced task stays with `Disposition:` and its replacement linked.
9. If the code contradicts this document, fix the document, log the change, then continue.
   Do not silently work around it.
10. After every edit to this file run
    `node C:/Users/mahdi/.claude/skills/spec-and-sprints/scripts/validate-roadmap.mjs teb_vae/lag_attn_transformer_cfs/TRAINING_THROUGHPUT_SPEC_AND_SPRINTS.md --repo .`
    from the repository root and fix errors before continuing. `--next` prints the next
    executable task. The validator is a dependency-free Node script; copy it into
    `scripts/` if this repository is checked out on a machine without that path.

Repository rules that bind every task here (from `CLAUDE.md` and `AGENTS.md`): Google-style
docstrings with LaTeX math (`$ $` inline), every runnable module also runnable from an IDE's
Run button with values edited inside the file, the smallest test selection that covers the
change, no commits unless asked, and no downstream compensation of the stored UP shift. One
further house rule from the cell's own documents: geometry in comments, docstrings and guides
is stated symbolically ($H$, $S$, $F$, $A_{\max}$, $C_{\mathrm{keep}}$) or read from the model,
never as a literal.

## Context and scope

### The problem

The production run of this cell (`configs/default.yaml`, six or seven A6000 ranks at batch
$B = 128$) is a multi-day fit: the shipped budget is 5000 epochs with early stopping at 50
validation epochs, the last production CSV in `output/lag_attn_transformer_cfs_results/`
holds 936 epochs, and `RESULTS.md` still carries an empty "Wall clock per epoch" row. The
owner asked where the time goes and which changes are small, safe and worth making before
the next run.

This document records the answer as measured on 2026-09-23 and turns it into two sprints of
changes plus a list of deferred levers with their reconsideration triggers. Nothing below
alters what the model computes: the sprints are constrained to be bitwise neutral on the
objective and on every reported metric, and the levers that are not neutral (bf16 autocast,
`torch.compile`, a coarser validation) are deferred behind the verification protocol the
configuration already prescribes for `compile`.

### How the training loop runs today

Entry point `teb_vae/lag_attn_transformer_cfs/trainer.py:main` delegates to
`teb_vae/lag_attn_rws/trainer.py:main`, which resolves the config, builds the driver, runs
the pre-flight refusals and calls `create_model` and `train_model`. Data comes through
`train/data_module.py:GraphDataModule._make_loader` and
`hdf5_dataset/hdf5_dataset.py:create_optimized_dataloader`: a map-style
`CombinedHDF5Dataset` whose `__getitem__` reads one sample at a time from the HDF5 shards
(`f[name][sample_idx]` per field), normalises it with the stats file, and transposes the
coefficient blocks to $(T, C)$; the `DataLoader` uses spawn workers, persistent workers,
loader-level pinning, and Lightning injects the `DistributedSampler` with a per-epoch
reshuffle. Every step therefore reads $B$ random samples per rank.

The Lightning step is `train/pl_model_base.py:LightningModelBase._dispatch_stage_step` ->
`teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask.compute_loss_and_metrics` ->
`teb_vae/lag_attn_rws/task.py:SeqVaeLagAttnRwsTask.compute_loss_and_metrics`. The input
builder resolves the anchor geometry from the stage: on `train` it derives one tile phase
$\varphi_b \in [0, S)$ per sample by hashing `guid`, `epoch` (the segment's start), the
training epoch and the seed (`SeqVaeLagAttnCfsTask.anchor_phase`); on `val` and `test` it
decodes the dense range at stride 1. The net's forward
(`teb_vae/lag_attn_cfs/nets/causal_inputs.py:CausalWarmupInputs.forward`) builds the anchor
index, runs the target adapter and encoder, the source adapter (the shipped
`lag_kv_source: adapter` builds no source encoder), the prior head with its detached
source-null clock, the lag cross-attention, the posterior head, and invokes the shared
decoder twice on the latents gathered at the anchors. The objective
(`teb_vae/lag_attn_rws/nets/losses.py:compute_loss`, reached through
`teb_vae/lag_attn_cfs/nets/causal_feature_target.py:CausalFeatureForecastTarget.compute_loss`
and `teb_vae/lag_attn_fs/nets/feature_target.py:FeatureForecastTarget.compute_loss`)
gathers the target block at the same anchors, builds the forecast and KL masks
(`teb_vae/lag_attn_rws/nets/raw_masks.py`), scores both branches and emits about sixty
metrics. On validation the task additionally runs the permutation control
(`teb_vae/lag_attn_rws/nets/controls.py:perm_forward_outputs`, a third decoder invocation
plus a second `compute_loss`) and the source-null KL (`controls.source_null_kld`).

The train step's anchor count is $A_{\max} = \lceil (\mathrm{ceiling} - F) / S \rceil$ per
sample; the validation step's is the dense $\mathrm{ceiling} - F$. At the shipped leaves
($H = 30$, $S = 15$, $F = 134$, $T = 300$, stored clock) that is 10 tiles against 136 dense
anchors, a factor of 13.6 on every anchor-axis tensor and on the decoder's work.

### What was measured, and how

All numbers below were measured on 2026-09-23 on a Windows 11 laptop with an RTX 4080 Laptop
GPU (12 GB), 20 logical cores, the project `.venv` (torch 2.7.1+cu128, lightning 2.6.1,
h5py 3.14.0). The model was built through the real driver
(`LagAttnTrfCfsTrainer.create_model()` on a leaf config inheriting `configs/default.yaml`,
with the committed fixture `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5` and
its stats file substituted for the shard paths so the warm-up budget and the scored horizon
resolve to the production values: 4.16 M parameters, $C_{\mathrm{keep}} = 76$, $L = 38$).
Batches were synthetic Gaussian tensors at the production geometry ($T = 300$, $c_y = 80$,
$c_u = 46$), at $B = 64$ unless stated (this card cannot hold $B = 128$ without thrashing its
allocator). The measurement scripts are session artefacts and are not in the repository;
Sprint 1 re-measures on the production box with the shipped `SimpleProfiler`, whose report
every run already writes at the end of `train_results/full.log`.

Three instruments were used: CUDA-event timing of the task's `compute_loss_and_metrics`
plus backward and optimizer step; `torch.cuda.set_sync_debug_mode` with a stack walk to
attribute every host-device synchronisation to a repository line; and `torch.profiler` for
kernel time. Two real Lightning fits through the shipped entry point ran on synthetic
shards (4096 samples, the fixture's exact schema and attributes) written in the production
chunk layout and in a per-sample layout, with six spawn workers, so the loader and the
trainer's own overheads appear in the profiler report exactly as they would on the box.

| Quantity | Measured (this laptop, $B = 64$) |
| --- | --- |
| Train step: forward, loss, backward, AdamW; cuDNN autotune on as the trainer sets it | 178 ms (199 ms with autotune off) |
| Validation step: dense forward, loss, permutation control, source-null KL | 695 ms, 3.5x a train step |
| Of which the two dense decoder invocations alone | ~300 ms |
| Target adapter + encoder, fwd + bwd, isolated | 65 ms |
| Two decoder invocations at the training tile, fwd + bwd, isolated | 51 ms |
| Lag attention, fwd + bwd, isolated | 30 ms |
| Posterior head, fwd + bwd, isolated | 18 ms |
| Prior head with clock, source adapter | 8 ms, 5 ms |
| Kernel launches per train step; GPU busy share | 7255; about 80 % |
| Host-device synchronisations per train step, per validation step | 89, 50 |
| Peak memory, train / val | 4.9 GiB / 3.0 GiB |
| Train step and validation step at $B = 128$ on this 12 GB card | 465 ms, 1440 ms (9.7 GiB peak, allocator near its limit) |
| Random single-sample read, production layout vs per-sample chunks, one process | 18.9 ms vs 2.9 ms |
| 128-sample batch through the shipped `DataLoader`, six spawn workers, production vs per-sample chunks | 432 ms vs 73 ms |
| Real fit, one epoch of 64 steps, production vs per-sample chunks | 32.8 s vs 23.0 s |
| Steady-state step inside the real fit on per-sample shards | 226 ms |
| Same, with the batch's `epoch` field kept on the host | 212 ms |
| Same, with train metrics logged epoch-only (one epoch, see F3) | 209 ms |
| Same, with `prog_bar` removed from the three metrics, nothing else | 233 to 241 ms (no gain) |
| bf16 autocast: train step, validation step, peak memory | 1.22x, 1.26x, 4.9 -> 3.5 GiB |
| bf16 drift on one fixed batch: `pred_gap`, `total_loss` | -0.037 nats (of -1.91), +1.21 nats (of 6387) |
| AdamW step, foreach (shipped) vs `fused=True` | 3.0 ms vs 2.3 ms |
| `on_after_backward` guard loop | 2.6 ms, 337 launches |

### Findings

**F1. The shard layout makes the loader the first bottleneck.** The writer
`hdf5_dataset/new_pipeline/create_new_pipeline.py:create_initial_hdf5` stores every dataset
in 32-sample chunks with LZF (`chunk_n = 32`, line 821; every `create_dataset` call at lines
851 to 997 passes `chunks=(chunk_n, ...)` and `compression="lzf"`); the committed fixture
shows the same layout. The loader reads one random sample per `__getitem__`, so each of the
seven per-sample fields decompresses a whole 32-sample chunk to serve one sample: about
6.8 MB of decompression for about 150 KB of data. The 128 MB per-file chunk cache the loader
opens (`CombinedHDF5Dataset._open_handle`, `rdcc_nbytes`) holds about twenty chunk sets, so
under a shuffled `DistributedSampler` over hundreds of thousands of samples it almost never
hits. Measured: 18.9 ms per sample in one process on the production layout against 2.9 ms
with `chunks=(1, ...)` and no compression, 3.3 ms with per-sample chunks and LZF, and 5.7 ms
with 32-sample chunks uncompressed (the chunk, not the codec, is the cost). Through the
shipped `DataLoader` with six spawn workers a 128-sample batch took 432 ms on the production
layout and 73 ms on the per-sample one; a real fit of the cell went from 32.8 s to 23.0 s
per 64-step epoch on the same GPU with nothing else changed, and the main process's own
`training_step` and `backward` actions shrank too because six workers decompressing flat out
had been competing with it for CPU. Scaled to the box: at $B = 128$ with the shipped eight
workers per rank the production layout needs about $128 \times 19 / 8 \approx 300$ ms of
worker time per step per rank, about a GPU step, so the loader is at parity with the GPU or
already the limiter, and any GPU-side gain is wasted until it moves. An existing shard can be
converted offline: a plain h5py copy with per-sample chunks kept every root and per-dataset
attribute and the data bit-identical (verified NaN-aware on a synthetic shard) and read at
4.4 ms per sample. One-dimensional datasets (`epoch`, `guid`, the labels) must keep a large
chunk: the index build reads them whole and one-element chunks would turn that into hundreds
of thousands of B-tree lookups.

**F2. Validation costs a quarter of every epoch, by construction.** A validation batch costs
3.1x to 3.5x a train step. The two decoder invocations run at the dense anchor count (a
factor of $(\mathrm{ceiling} - F) / A_{\max}$, 13.6 at the shipped leaves, on the decoder's
$(B \cdot A) \times H$ token rows in `teb_vae/lag_attn/nets/decoders.py:HorizonDecoderCore.decode`)
and were ~300 ms of the 695 ms; the permutation control re-decodes the full branch a third
time and re-scores it (about a fifth); the objective's four extra scoring passes for the gap
readouts (`FeatureForecastTarget._forecast_gaps_from_mask`,
`CausalFeatureForecastTarget._gap_by_kept_channel`) and the source-null KL take the rest.
With about 506,000 training windows per epoch (`DIAGNOSIS.md`, section 4) and 41,381
held-out windows (`EVAL_DIAGNOSIS_2026-09-08.md`), six ranks at $B = 128$ run about 659 train
steps and 54 validation batches per rank per epoch, so validation is about 180
train-step-equivalents, roughly 27 % of the epoch. The dense validation decode is a design
decision the cell documents (a single phase would be phase-biased), so the only cut that
changes no metric's definition is a cadence on the permutation control, which is a readout
whose columns already tolerate absent steps. `check_val_every_n_epoch` is not available:
`train/callbacks.py:MetricsHistoryCsvCallback._write` numbers rows assuming validation ran
every epoch.

**F3. The step carries dozens of host-device synchronisations, and the trainer adds one.**
A synchronising call makes the CPU wait until the GPU has drained everything queued so far,
which throws away the run-ahead that hides Python and launch overhead on a model that issues
7255 kernels per step. At $B = 64$ a train step synchronises 89 times and a validation step
50 times, at these sites (line numbers on the baseline):

| Site | Count per train step | Mechanism |
| --- | --- | --- |
| `teb_vae/lag_attn_cfs/task.py:573` `_as_float`, from `anchor_phase` | $B$ (64) | `batch.epoch` is a CUDA tensor after Lightning's transfer; the hash loop calls `.item()` once per sample |
| `teb_vae/lag_attn_rws/nets/raw_masks.py:95` and `:108` `_validate_anchors` | 10 | two `bool(tensor.any())` refusals per mask build; masks are built five times per train step (objective, `SeqVaeLagAttnCfsTask._mu_gap_rms`, `_resolved_forecast_gaps`) and ten times per validation step |
| `teb_vae/lag_attn_rws/nets/losses.py:403` `masked_raw_likelihood` | 2 | `float(cell_mask.sum())` for the per-element rescaling |
| `losses.py:454`, `:455` `masked_source_kl` | 2 | `bool(support.any())` and boolean-mask indexing |
| `losses.py:1019`, `:1020`, `:1026`, `:1028` diagnostics | 4 | `bool(support.any())` and three boolean-mask indexings |
| `losses.py:1072` to `:1078` weight echoes | 5 | `torch.tensor(float(x), device=cuda)` is a blocking pageable copy |
| `teb_vae/lag_attn_cfs/nets/causal_inputs.py:734` and `:743` `_build_anchor_index` | 2 | pageable `.to(device)` of the phase, then `bool(...any())` on the device copy |
| `teb_vae/lag_attn/nets/controls.py:173` `make_derangement` | val only | the permutation draw |

Lightning adds one per step that no configuration key removes: the logger connector's
`on_batch_end` (`lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py`)
converts every metric logged with `prog_bar=True` to a Python scalar
(`result.py`, `convert_tensors_to_scalars`) whether or not a bar is displayed, so the three
`prog_bar_metrics` of `SeqVaeLagAttnRwsTask` force a drain each step. Measured inside the real
fit: keeping `epoch` on the host took the steady-state step from 226 ms to 212 ms; removing
`prog_bar` alone changed nothing while the other syncs remained; logging the train metrics
epoch-only measured 209 ms on one epoch but is **not** a free change, because
`callback_metrics` holds a train metric's last-step value only because it is logged on step,
and `MetricsLoggingCallback`, `MetricsHistoryCsvCallback` and `LossPlotCallback` read the
`train/` columns from it at validation end, before the train epoch is reduced
(`SeqVaeLagAttnRwsTask.on_before_optimizer_step` documents exactly this for `grad_norm`, and
`DIAGNOSIS.md` section 3 relies on it). Switching train logging to epoch-only would empty or
lag those columns by an epoch, which is why it is deferred rather than planned.

**F4. GPU-side profile.** Kernel time per train step splits as: matrix multiplies about 18 %,
elementwise multiplies 11 % (LayerScale, gates, masks, weights), the target encoder's
memory-efficient attention 15 % forward and backward, the lag attention's `unfold_backward`
6 % (the `unfold` windows in `teb_vae/lag_attn/nets/attention.py:LagCrossAttention._attend`
are copied by the einsum, $B \cdot T \cdot d_{\mathrm{model}} \cdot L$ floats twice, 712 MiB
per forward at $B = 64$ and kept for backward), convolution backward 6 %, layer-norm backward
6 %, the optimizer 3 ms. Isolated module costs are in the table above; the target encoder is
the largest single item and the decoder pair the second. The GPU is busy about 80 % of the
unprofiled step, so the synchronisations of F3 and Python overhead bound the gain of F3's
fixes at roughly a fifth of the step; the loader (F1) and validation (F2) are larger.

**F5. Precision and compilation are real but not neutral.** bf16 autocast gave 1.22x on the
train step and 1.26x on validation and cut peak memory 28 %, but on one fixed batch it moved
`pred_gap` by 0.037 nats, the order of the quantity itself, and `total_loss` by 1.2 nats:
exactly the non-neutrality `compile_model_requested` in
`teb_vae/lag_attn_transformer_rws/trainer.py` warns about for compilation, so the same
one-batch comparison protocol applies. Two guards would be required for bf16 to be sound:
`teb_vae/lag_attn_transformer_rws/nets/blocks.py:RMSNorm.forward` computes its mean of
squares in the input dtype and should upcast, and `entmax15` in
`LagCrossAttention._attend` should receive float32 scores. `torch.compile` could not be
timed here (no Triton in the Windows venv); the forward has graph breaks at the two
`bool(...any())` checks in `_build_anchor_index` and at the prior clock's eval toggling in
`CausalWarmupInputs._prior_clock`, and CUDA graphs are out until the syncs of F3 are gone.
`tests/test_config_load.py` asserts `precision == "32-true"` and `compile is False` on the
shipped config, so either belongs in a sweep arm, not the default.

**F6. What is already right, and what is not a lever.** DDP ships with
`broadcast_buffers=False`, `gradient_as_bucket_view=True` and `find_unused_parameters` derived
from the likelihood (`teb_vae/lag_attn_rws/trainer.py:LagAttnRwsTrainer.ddp_kwargs`); TF32
matmuls and cuDNN autotune are on (`train/graph_model_base.py:GraphModelBase.configure_determinism`,
worth about 10 % against autotune off); the profiler is on. The batch size is held at 128 for
gradient-noise comparability with the sibling cells, so the memory headroom of a 48 GB card
is not a free speedup. `AdamW(fused=True)` and gating the `on_after_backward` guard loop are
each worth about 1 % and are recorded as deferred. `default.yaml` ships seven `cuda_devices`;
the box has six, so that leaf is edited per run, as today.

### Scope

Goals: make the shards cheap to read (Sprint 1), remove the synchronisations that are
bitwise-neutral to remove and prove the neutrality (Sprint 2), and measure both on the box
with the instrument the run already carries. Non-goals: changing the model, the objective,
the validation definition, the batch size, precision or compilation; those are deferred
levers with triggers below. Build acceptance is the sprint Definitions of Done. Post-launch
outcome: the "Distributed smoke, memory and throughput" table in `RESULTS.md` filled with
before-and-after production numbers, and a headline run whose profiler report shows the
loader no longer stalling the step.

## Requirements and acceptance

- FR-001 [required]: A re-chunk tool converts an existing shard to one sample per chunk for every dataset of two or more dimensions (`fhr`, `up`, the four coefficient blocks, `target`, `weight`), keeps the existing chunking of one-dimensional datasets, copies the root and per-dataset attributes verbatim, refuses to overwrite its output, and is runnable from the command line and from an IDE Run button by editing constants in the file. Acceptance: on the committed causal fixture the output's 2-D and 3-D datasets report `chunks[0] == 1`, every attribute and every sample compare equal (NaN-aware), `read_causal_warmup` resolves identically from either file, and `CombinedHDF5Dataset` serves identical tensors from both at `trim_minutes=1.0` with the fixture stats.
- FR-002 [required]: New shards written by `create_initial_hdf5` use one sample per chunk for datasets of two or more dimensions and keep `chunk_n` for one-dimensional ones. Acceptance: a shard written by the writer under the existing pipeline tests reports those chunk shapes; the pipeline tests pass.
- FR-003 [required]: The production box is measured before and after re-chunking with the shipped profiler on one epoch of the shipped config, and the rows `train_dataloader_next`, `run_training_batch`, the strategy `training_step` and `backward`, `run_training_epoch`, `validation_step` and the first-step peak-memory log line are recorded, dated, in the `RESULTS.md` throughput table. Acceptance: the table's production column is filled for both states, and after re-chunking `train_dataloader_next` is no longer the larger of it and `run_training_batch`.
- FR-004 [required]: The tile phase is derived and range-checked on the host, the batch's `epoch` field is never moved to the device, and the phase reaches the model through a pinned, non-blocking copy, so a train step performs no synchronisation in `anchor_phase` or in `_build_anchor_index`. Acceptance: the phases equal today's bitwise (the existing phase tests hold); a CUDA-only test attributes no synchronising operation to those two functions; the refusal of a phase outside $[0, S)$ still raises.
- FR-005 [required]: The bitwise-evidence script `scripts/print_objective_metrics.py` covers the two causal cells so a before-and-after byte diff of the complete metric dictionary exists for every Sprint 2 change. Acceptance: the script prints the full metric dictionary of `SeqVaeLagAttnCfs` and `SeqVaeLagAttnTrfCfs` on a fixed stub batch for a tiled train-stage forward and a dense validation-stage forward, plus the anchor phases, at full precision, and its output on the baseline tree is captured before any Sprint 2 code change.
- FR-006 [required]: The objective's bitwise-neutral synchronisation sites are removed: the anchor set is validated once per step rather than at every mask build, the per-element count in `masked_raw_likelihood` stays a device tensor, the five weight echoes in `compute_loss` are built without a host-to-device copy, and the existing refusals for an externally supplied anchor set still raise. Acceptance: the FR-005 script's output is byte-identical before and after; the CUDA-only sync test counts exactly one `_validate_anchors` call per train step (its two refusals are device reads by nature and are the one validation the step keeps) and attributes no synchronising operation to `masked_raw_likelihood` or the echo lines; `test_raw_masks.py` passes unchanged.
- FR-007 [required]: After Sprint 2 the box is re-measured with the FR-003 protocol and the steady-state step time recorded beside the Sprint 1 numbers. Acceptance: a second dated row set in the `RESULTS.md` table.
- NFR-001 [required]: Numerical neutrality of every Sprint 2 change, measured as a byte-identical output of the FR-005 script and `torch.equal` anchor phases on the stub batch; the cell's default test selection and the slow smoke suite pass.
- FR-008 [deferred]: Sync-free boolean-index diagnostics (`masked_source_kl`, the log-variance diagnostics in `compute_loss`) cannot be made bitwise neutral (a masked sum changes summation order); reconsider after FR-007 if the residual six syncs per step measure above a few percent.
- FR-009 [deferred]: A cadence on the validation permutation control (run every $k$-th epoch) would cut about a fifth of validation cost; it changes how often three readout columns exist and needs the owner's decision.
- FR-010 [deferred]: Epoch-only train metric logging (measured about 8 % on the step) changes the meaning of the `train/` CSV columns from one-step samples to epoch means and needs the three epoch-end callbacks to collect train values at `on_train_epoch_end`; reconsider if that semantic change is wanted.
- FR-011 [deferred]: A bf16-mixed sweep arm with the RMSNorm and entmax float32 guards and the one-batch `pred_gap` comparison the config prescribes for compilation; reconsider after FR-007 shows the GPU step is the limiter.
- FR-012 [deferred]: A `torch.compile` sweep arm; reconsider after FR-004 and FR-006 remove the graph breaks in the forward and only with Triton available on the box.
- FR-013 [deferred]: A banded $T \times T$ formulation of the lag attention (4x less memory than the unfolded windows, tensor-core friendly); numerics change at float rounding, so it is a modelling decision.
- FR-014 [deferred]: `AdamW(fused=True)` and gating the `on_after_backward` guard loop on a single host read; about 1 % each.
- FR-015 [deferred]: Raising `dataloader_config.num_workers` per rank as an interim if the shards cannot be re-chunked before the next run; the leaf is compared across the square by `tests/test_config_load.py`, so it moves in all three defaults or through the allow-list.

## Codebase map

| Path | Role | Relevance |
| --- | --- | --- |
| hdf5_dataset/new_pipeline/create_new_pipeline.py:create_initial_hdf5 | Creates every dataset of a new shard with `chunks=(chunk_n, ...)`, `chunk_n = 32`, LZF | change point (FR-002) |
| hdf5_dataset/hdf5_dataset.py:CombinedHDF5Dataset.__getitem__ | One HDF5 read per field per sample, trim, normalise, transpose | reads the layout; unchanged |
| hdf5_dataset/hdf5_dataset.py:CombinedHDF5Dataset._open_handle | Opens shards with a 128 MB chunk cache per file | unchanged; cache becomes moot after FR-001 |
| hdf5_dataset/hdf5_dataset.py:read_causal_warmup | Resolves the warm-up budget from shard attributes | acceptance oracle for FR-001 |
| hdf5_dataset/hdf5_dataset.py:create_optimized_dataloader | Spawn, persistent workers, loader-level pinning, collate | unchanged |
| hdf5_dataset/rechunk_hdf5.py | Offline re-chunk tool | (new) FR-001 |
| hdf5_dataset/tests/test_rechunk_hdf5.py | Fixture round-trip test of the tool | (new) FR-001 |
| hdf5_dataset/tests/test_causal_loader.py | Loader contract on causal shards, fixtures written per test | reuse fixtures; must keep passing |
| hdf5_dataset/tests/test_causal_pipeline.py | Pipeline tests around the writer | must keep passing after FR-002; implementer checks whether it asserts chunk shapes |
| train/data_module.py:GraphDataModule._make_loader | Config to loader kwargs | unchanged |
| train/pl_model_base.py:LightningModelBase._dispatch_stage_step | Step dispatch, spike breaker, metric logging | unchanged (FR-010 deferred) |
| train/pl_model_base.py:LightningModelBase.on_after_backward | Per-parameter guard loop | unchanged (FR-014 deferred) |
| train/graph_model_base.py:GraphModelBase._build_profiler | `profiler: simple` writes the report FR-003 reads | reuse |
| train/graph_model_base.py:GraphModelBase.configure_determinism | TF32 and cuDNN autotune | reuse; must not change |
| train/callbacks.py:MetricsHistoryCsvCallback._write | Numbers CSV rows assuming validation every epoch | must not change; why `check_val_every_n_epoch` is out |
| teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask.anchor_phase | Hashes guid, epoch, training epoch, seed into $\varphi_b$; `.item()` per sample on a device tensor | change point (FR-004) |
| teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask.resolve_anchor_geometry | Stage to $(\varphi, S)$ | reuse |
| teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask._mu_gap_rms | Rebuilds both masks per step | passes the validation flag (FR-006) |
| teb_vae/lag_attn_cfs/nets/causal_inputs.py:CausalWarmupInputs._build_anchor_index | Builds $\mathcal A(\varphi)$; moves the phase to the device then checks its range | change point (FR-004) |
| teb_vae/lag_attn_cfs/nets/causal_feature_target.py:CausalFeatureForecastTarget._resolved_forecast_gaps | Rebuilds the forecast mask for the gap splits | passes the validation flag (FR-006) |
| teb_vae/lag_attn_rws/nets/raw_masks.py:_validate_anchors | Two `bool(any())` refusals | change point (FR-006): keep, make skippable by the model's own callers |
| teb_vae/lag_attn_rws/nets/raw_masks.py:forecast_mask | Forecast mask, dense or gathered | gains an opt-out of validation |
| teb_vae/lag_attn_rws/nets/raw_masks.py:kl_mask | KL support, scattered back to $(B, T)$ | gains the same opt-out |
| teb_vae/lag_attn_rws/nets/losses.py:masked_raw_likelihood | Per-element count via `float(cell_mask.sum())` | change point (FR-006) |
| teb_vae/lag_attn_rws/nets/losses.py:compute_loss | Objective; builds masks; five scalar echoes | change point (FR-006) |
| teb_vae/lag_attn_rws/nets/losses.py:masked_source_kl | KL with boolean-index diagnostics | unchanged (FR-008 deferred) |
| teb_vae/lag_attn_rws/nets/controls.py:source_null_kld | Validation readout; builds both masks | passes the validation flag (FR-006) |
| teb_vae/lag_attn_rws/task.py:SeqVaeLagAttnRwsTask.compute_loss_and_metrics | Forward, objective, permutation control | unchanged |
| teb_vae/lag_attn_rws/task.py:SeqVaeLagAttnRwsTask.on_train_batch_end | Logs first-step peak CUDA memory | the memory line FR-003 records |
| teb_vae/lag_attn_rws/plotting.py:LagAttnRwsPlotCallback._generate_plots | Calls `transfer_batch_to_device` on the task | must keep working with the FR-004 override |
| teb_vae/lag_attn_rws/tests/test_raw_masks.py | Refusals for out-of-range, repeated and float anchors | must keep passing (FR-006) |
| teb_vae/lag_attn_cfs/tests/test_task.py | Phase keyed per segment, rotates with the epoch, seed moves it, no RNG draw | must keep passing (FR-004) |
| teb_vae/lag_attn_cfs/tests/conftest.py:make_stub_batch | Stub batch with `guid` and `epoch` | reuse in FR-005 script and tests |
| teb_vae/lag_attn_transformer_cfs/tests/test_objective.py | Bitwise pins of the objective on this cell | must keep passing (NFR-001) |
| teb_vae/lag_attn_transformer_cfs/tests/test_train_smoke.py | Slow: tiny fit; every metric reaches the logger, no all-NaN column | Sprint 2 integration check |
| teb_vae/lag_attn_transformer_cfs/tests/test_config_load.py | Pins `precision`, `compile`, and parity of shared leaves | why FR-011, FR-012, FR-015 are arms or deferred |
| teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py | CUDA-only sync attribution test | (new) FR-004, FR-006 |
| teb_vae/lag_attn_transformer_cfs/configs/default.yaml | Shipped leaves: `num_workers: 8`, `prefetch_factor: 4`, `profiler: simple`, `precision: "32-true"`, `compile: false` | read by FR-003; unchanged |
| teb_vae/lag_attn_transformer_cfs/RESULTS.md | "Distributed smoke, memory and throughput" table with empty rows | record for FR-003 and FR-007 |
| scripts/print_objective_metrics.py | Bitwise metric dump of the fs and rws cells for before-and-after diffs | extend to the causal cells (FR-005) |
| teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5 | Committed causal shard, 32-sample LZF chunks | input of the FR-001 test |
| teb_vae/lag_attn/tests/fixtures/tiny_stats_causal_int.hdf5 | Its stats file | input of the FR-001 test |

Call chain (training step): Lightning `training_step` -> `train/pl_model_base.py:LightningModelBase._dispatch_stage_step` -> `teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask.compute_loss_and_metrics` -> `teb_vae/lag_attn_rws/task.py:SeqVaeLagAttnRwsTask.compute_loss_and_metrics` -> `SeqVaeLagAttnCfsTask._build_forward_inputs` -> `SeqVaeLagAttnCfsTask.anchor_phase` -> `teb_vae/lag_attn_cfs/nets/causal_inputs.py:CausalWarmupInputs.forward` -> `_build_anchor_index` -> `teb_vae/lag_attn_cfs/nets/causal_feature_target.py:CausalFeatureForecastTarget.compute_loss` -> `teb_vae/lag_attn_fs/nets/feature_target.py:FeatureForecastTarget.compute_loss` -> `teb_vae/lag_attn_rws/nets/losses.py:compute_loss` -> `teb_vae/lag_attn_rws/nets/raw_masks.py:forecast_mask` and `kl_mask` -> `_validate_anchors`.
Call chain (data): `teb_vae/lag_attn_rws/trainer.py:main` -> `train/data_module.py:GraphDataModule.train_dataloader` -> `hdf5_dataset/hdf5_dataset.py:create_optimized_dataloader` -> `CombinedHDF5Dataset.__getitem__` -> `_open_handle` -> h5py chunk read.
Conventions: tests run with `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"` from the repository root (`DESIGN.md`, "How to run the tests"); the `slow` marker is registered in `teb_vae/lag_attn_transformer_cfs/tests/conftest.py`; on a Linux box the interpreter is `.venv/bin/python`. No lint or type gate is configured. Every runnable module carries a Run-button entry (`CLAUDE.md`). Documents in this cell are as-built records amended with dated sections rather than rewritten.
Sources: `CLAUDE.md`, `AGENTS.md`, `requirements.txt` (torch 2.7.1+cu128, lightning 2.6.1, h5py 3.14.0, pytest 9.0.3), `DESIGN.md`, `DIAGNOSIS.md`, `EVAL_DIAGNOSIS_2026-09-08.md`, `RESULTS.md`, the inspected modules and tests above, and the installed Lightning logger connector.

## Approach

Sprint 1 changes no training code. A new `hdf5_dataset/rechunk_hdf5.py` copies a shard
dataset by dataset with `chunks=(1, *shape[1:])` for datasets of two or more dimensions,
the source's own chunking for one-dimensional ones, attributes copied verbatim, in
row blocks so memory stays bounded; the same rule lands in `create_initial_hdf5` for new
builds. Compression stays a parameter: the tool defaults to none (measured fastest), the
writer keeps its LZF (per-sample LZF measured within 16 % of uncompressed and 19 % smaller).
The loader, its chunk cache, the sampler and the config are untouched. The box measurement
uses the profiler report the run already writes; no new instrumentation.

Sprint 2 removes the synchronisations that can be removed bitwise-neutrally and proves it
with the repository's own convention for such changes: a byte diff of the complete metric
dictionary printed by `scripts/print_objective_metrics.py`, extended first to the two
causal cells. The phase becomes a host quantity end to end: `SeqVaeLagAttnCfsTask`
overrides `transfer_batch_to_device` to leave `epoch` on the host (every other field moves
as today; `guid` is a list and never moved), `anchor_phase` returns a pinned tensor when CUDA
is available, and `_build_anchor_index` range-checks the host tensor before a non-blocking
copy, keeping the device path for an externally supplied device phase. The anchor-set
validation stays where it is and keeps its refusals, but `forecast_mask` and `kl_mask` gain
a `validate` keyword defaulting to `True`; the objective validates once at its first mask
build and passes `validate=False` to the other four builds of the same index, and the task's
readouts and the source-null control do the same for the index they read off the forward
dictionary. The per-element count stays a 0-dim tensor (integer counts are exact in float32,
so the division is bitwise the same) and the five echoes use `torch.full`. Everything else in
F3 is deferred with its reason. Alternatives rejected: validating on a host twin of the anchor
index (more state for the same effect), `torch._assert_async` (a device-side abort with no
message), and dropping the validation (the eval and attribution paths hand the masks their
own indices).

Data, contracts and failure behaviour: the re-chunked shard is byte-for-byte the same data
under the same names and attributes, so every resolver, the stats pairing and the loader
contract are unchanged; the tool refuses to overwrite and leaves a partial output only on a
crash, which the acceptance check would catch. The `validate` keyword defaults to the current
behaviour, so external callers are unchanged. Delivery: re-chunk the shards on the box in a
new directory, repoint the four `dataset_config` paths, run the FR-003 epoch, then switch
the headline run; rollback is repointing the paths.

## Decisions and risks

| Item | State | Evidence / rationale | Resolution / owner | Affected work |
| --- | --- | --- | --- | --- |
| Per-sample chunks for datasets of two or more dimensions; 1-D datasets keep their chunking | resolved | F1 measurements; one-element chunks on 1-D datasets would make the whole-column index reads in `_build_index` chunk-count bound | none | FR-001, FR-002 |
| Compression: tool defaults to none, writer keeps LZF | assumed | 2.9 ms vs 3.3 ms per sample; 19 % file size; disk headroom on the box unknown | owner confirms disk budget; flip the tool's default if disk-bound | S1-T01, S1-T02 |
| The box numbers are extrapolated from this laptop until measured | assumed | A6000 vs laptop 4080 and server vs laptop cores differ | S1-T03 measures | FR-003 |
| Re-chunk offline rather than rebuild the shards | resolved | verified identical attributes and data on a synthetic shard; rebuilding costs a pipeline run | owner picks per shard set | S1-T01 |
| Epoch-only train logging stays deferred | resolved | `callback_metrics` holds train values at validation end only because they are logged on step (installed Lightning `logger_connector.py`, `result.py`) | FR-010 trigger | none |
| Permutation-control cadence, bf16, compile, banded attention, batch size | deferred | each changes a definition, a numeric result or a comparability decision | owner decides after FR-007 | FR-009, FR-011, FR-012, FR-013 |
| `num_workers` interim raise | deferred | leaf compared across the square by `tests/test_config_load.py` | only if the shards cannot be re-chunked before the next run | FR-015 |
| The CUDA-only sync test may skip on CI without a GPU | assumed | `torch.cuda.set_sync_debug_mode` is a prototype feature and needs CUDA | test skips without CUDA and states so | S2-T02, S2-T03 |

## Validation strategy

Existing coverage reused: `hdf5_dataset/tests/test_causal_loader.py` (the loader contract
on causal shards and the budget reader), `hdf5_dataset/tests/test_causal_pipeline.py` (the
writer), `teb_vae/lag_attn_cfs/tests/test_task.py` (the phase is per segment, rotates with
the epoch, moves with the seed, draws no random number), `teb_vae/lag_attn_rws/tests/test_raw_masks.py`
(anchor refusals and mask semantics), `teb_vae/lag_attn_transformer_cfs/tests/test_objective.py`
(bitwise pins of the objective on this cell), `teb_vae/lag_attn_transformer_cfs/tests/test_train_smoke.py`
(slow; a real tiny fit through the shipped entry point, every metric reaches the logger, no
all-NaN column), `train/tests/test_metrics_history.py` (CSV numbering).

New tests, each for a failure nothing above catches: (1) `hdf5_dataset/tests/test_rechunk_hdf5.py`,
one round trip of the committed causal fixture through the tool: chunk shapes, attributes,
NaN-aware data equality, identical `read_causal_warmup` and identical loader samples, which
catches a dropped attribute or a mis-chunked 1-D dataset. (2) `teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py`,
CUDA-only: runs one train-stage `compute_loss_and_metrics` on the tiny model under
`torch.cuda.set_sync_debug_mode(1)`, captures the synchronisation warnings, walks each
warning's stack, and asserts none originates in `anchor_phase`, `_build_anchor_index`,
`_validate_anchors`, `masked_raw_likelihood` or the echo lines of `compute_loss`; this
catches a re-introduced `.item()` or `bool(any())` at those sites without pinning a total
count that the deferred sites and future torch versions would move. A `meta`-device test of
the `transfer_batch_to_device` override (every tensor moved, `epoch` still on the host) runs
without CUDA.

Evidence rather than tests: the FR-005 script's output captured on the baseline tree
(`before.txt`) and diffed byte for byte after each Sprint 2 task; the FR-003 and FR-007
profiler rows on the box. Commands run from the repository root with the project `.venv`;
fast tests after every task, the slow smoke suite at each sprint boundary, box measurements
when the operator schedules them.

## Sprint overview

| Sprint | Goal | Usable outcome | Dependencies | Detail |
| --- | --- | --- | --- | --- |
| Sprint 1 | Shards readable at per-sample granularity, measured on the box | The loader stops being the larger of loader and step in the profiler report; a dated throughput row in RESULTS.md | none | Detailed |
| Sprint 2 | Bitwise-neutral removal of the step's avoidable synchronisations | Fewer syncs per step with byte-identical metrics; a second dated row in RESULTS.md | Sprint 1 (for the measurement protocol only) | Detailed |

## Sprint 1: Data path

Goal: Serve one sample per chunk read, for existing and new shards, and measure the effect where it matters.
Demo: Re-chunk the committed fixture into a temporary directory, load a sample from both files and see identical tensors; on the box, compare the `train_dataloader_next` and `run_training_batch` rows of one epoch before and after.
Definition of Done: FR-001, FR-002 and FR-003 acceptance hold with evidence; `hdf5_dataset/tests` passes; the RESULTS.md table holds both production rows.
Dependencies: none

#### S1-T01: Re-chunk tool for existing shards

Requirements: FR-001
Depends on: none
Description: Add `hdf5_dataset/rechunk_hdf5.py` with a function `rechunk(src, dst, *, compression=None, rows_per_pass=1024)` that opens `src` read-only and `dst` with `libver="latest"` (as the writer does), copies the root attributes, then for every dataset creates the destination with the same shape, dtype and the source's own `maxshape` (so the file stays appendable and a 1-D chunk wider than the row count stays legal), `chunks=(1, *shape[1:])` when `ndim >= 2` and the source's own `chunks` otherwise, the requested compression for numeric datasets and none for the object (`guid`) dataset, copies the dataset's attributes, and copies rows in blocks of `rows_per_pass`. Refuse an existing `dst`. Provide an `argparse` entry and module constants `SRC`, `DST`, `COMPRESSION` so the Run button converts one file with no command line. Google-style docstrings; state in the module docstring why 1-D datasets keep their chunking (whole-column index reads) and that the loader, the resolvers and the stats pairing read nothing but names, shapes and attributes, which the copy preserves. Do not import from `create_new_pipeline.py` (it builds a filter bank at import).
Acceptance criteria:
- Running the tool on `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5` produces a file whose `fhr`, `up`, `fhr_st`, `fhr_ph`, `up_st`, `up_ph`, `target` and `weight` datasets report `chunks[0] == 1` and whose 1-D datasets keep the source chunking.
- Root and per-dataset attributes compare equal with `numpy.array_equal`; every dataset compares equal NaN-aware; `guid` strings compare equal.
- `read_causal_warmup([dst], 1.0)` equals `read_causal_warmup([src], 1.0)` field by field, and `CombinedHDF5Dataset` built on each file with the fixture stats and `trim_minutes=1.0` returns `torch.equal` tensors for every field of every sample.
- An existing destination path is refused with a message naming it.
Files affected:
- hdf5_dataset/rechunk_hdf5.py (new) - the tool, CLI and Run-button entry.
- hdf5_dataset/tests/test_rechunk_hdf5.py (new) - the fixture round trip described in the acceptance criteria, using `tmp_path`.
Validation: `.venv/Scripts/python.exe -m pytest hdf5_dataset/tests/test_rechunk_hdf5.py hdf5_dataset/tests/test_causal_loader.py -q`; cwd repository root; prerequisites project `.venv`; expect all cases pass.
Test rationale: No existing test exercises a converted shard. One round-trip test on the committed fixture catches a dropped attribute, a mis-chunked dataset, a changed dtype or a changed value; the loader test file is run beside it because the loader is the consumer.
Runtime: fast/local
Evidence: 2026-09-23 | `.venv/Scripts/python.exe -m pytest hdf5_dataset/tests/test_rechunk_hdf5.py hdf5_dataset/tests/test_causal_loader.py -q` | pass | 67 passed on the 2a2bc78 working tree; tool at hdf5_dataset/rechunk_hdf5.py, test at hdf5_dataset/tests/test_rechunk_hdf5.py

#### S1-T02: Per-sample chunks in the shard writer

Requirements: FR-002
Depends on: none
Description: In `hdf5_dataset/new_pipeline/create_new_pipeline.py:create_initial_hdf5`, give every dataset of two or more dimensions `chunks=(1, *shape)` (the sample-axis chunk of one) and leave the one-dimensional datasets on `(chunk_n,)`. Keep `compression="lzf"` as it is. Update the function's docstring and the comment beside `chunk_n` to state the rule and its reason (one random sample per read; whole-column reads on 1-D datasets). Check whether `hdf5_dataset/tests/test_causal_pipeline.py` asserts chunk shapes and, if it does not, add one assertion on a written coefficient block to the most local existing test rather than a new file.
Acceptance criteria:
- A shard written through `create_initial_hdf5` reports `chunks[0] == 1` on every dataset of two or more dimensions and `chunks == (chunk_n,)` on the 1-D ones.
- `hdf5_dataset/tests/test_causal_pipeline.py` passes.
Files affected:
- hdf5_dataset/new_pipeline/create_new_pipeline.py:create_initial_hdf5 - existing; chunk rule per dataset rank.
- hdf5_dataset/tests/test_causal_pipeline.py - existing; one chunk-shape assertion if none exists.
Validation: `.venv/Scripts/python.exe -m pytest hdf5_dataset/tests/test_causal_pipeline.py -q`; cwd repository root; prerequisites project `.venv` with the vendored `kymatio`; expect pass.
Test rationale: The writer's existing tests cover its contents; the one new assertion catches a future regression to sample-batched chunks, which no current test would see.
Runtime: fast/local
Evidence: 2026-09-23 | `.venv/Scripts/python.exe -m pytest hdf5_dataset/tests/test_causal_pipeline.py -q` | pass | 41 passed; the writer chunks every dataset of two or more dimensions one sample deep, the 1-D ones stay at `chunk_n`; no test asserted chunk shapes, one assertion added to `test_the_written_causal_shard_has_the_causal_schema`

#### S1-T03: Measure the loader and the step on the production box, before and after

Requirements: FR-003
Depends on: S1-T01
Description: Operator task with a written protocol. On the box, write a leaf config outside the repository with `base:` pointing at the absolute path of `configs/default.yaml`, `general_config.epochs: 1`, `general_config.plot_frequency: 1000`, a distinct `general_config.tag`, `advanced_config.tracking.mlflow.enabled: false`, and `general_config.cuda_devices` matching the box; launch it exactly as the trainer's module docstring shows (`TEB_RUN_STAMP` exported, `torchrun --nproc_per_node` equal to the device count). Run it once on the current shards and once on shards converted with S1-T01 into a new directory (repoint the four `dataset_config` paths in the leaf). From each run's `train_results/full.log` take the FIT Profiler Report rows `run_training_epoch`, `run_training_batch`, the strategy `training_step` and `backward`, `train_dataloader_next`, `validation_step`, `val_next`, and the `peak CUDA memory after the first training step` line, and write them, dated and labelled by shard layout, into the "Distributed smoke, memory and throughput" table of `RESULTS.md` as a dated amendment (the cell's documents are amended, not rewritten). Note the worker count and batch size beside the numbers.
Acceptance criteria:
- Both runs complete one epoch and one validation pass; both profiler reports are captured.
- The RESULTS.md table's production column carries the rows for both layouts with a date.
- On the re-chunked shards `train_dataloader_next` is smaller than `run_training_batch`.
Files affected:
- teb_vae/lag_attn_transformer_cfs/RESULTS.md - existing; fill the throughput table by dated amendment.
- teb_vae/lag_attn_transformer_cfs/configs/default.yaml - existing; read only, the leaf inherits it.
Validation: inspection of the two `train_results/full.log` reports and of the RESULTS.md diff; cwd repository root on the box; prerequisites the box's `.venv`, the shards and the stats file at the configured paths, a re-chunked copy of the shards; expect the acceptance rows present.
Test rationale: A measurement, not a code change; the evidence is the two profiler reports.
Runtime: slow/external
Evidence: none

## Sprint 2: Step overhead

Goal: Remove every synchronisation in the train step that can be removed without moving a number, and prove that no number moved.
Demo: The FR-005 script prints byte-identical output before and after; the CUDA-only sync test passes; the box's steady-state step time is recorded beside Sprint 1's.
Definition of Done: FR-004, FR-005, FR-006, FR-007 and NFR-001 acceptance hold with evidence; the cell's default test selection and slow smoke suite pass; `teb_vae/lag_attn_cfs/tests` and `teb_vae/lag_attn_rws/tests/test_raw_masks.py` pass.
Dependencies: Sprint 1 (S1-T03 supplies the measurement protocol and the baseline numbers)

#### S2-T01: Extend the bitwise metric dump to the causal cells

Requirements: FR-005
Depends on: none
Description: Extend `scripts/print_objective_metrics.py` (or add `scripts/print_causal_objective_metrics.py` beside it if the import of the fs and rws conftests and the cfs conftest cannot coexist cleanly) so it also builds `SeqVaeLagAttnCfs` and `SeqVaeLagAttnTrfCfs` at the causal cells' test kwargs and stub batch (`teb_vae/lag_attn_cfs/tests/conftest.py:make_stub_batch` and the task factory in that conftest), runs, at a fixed seed on the CPU, one tiled train-stage step (phase from `anchor_phase`, the configured stride) and one dense validation-stage step through the task's own `compute_loss_and_metrics`, and prints every metric of both at full precision together with the anchor phases and the anchor index. Keep the script's contract: no arguments, runnable from the Run button, output meant for a byte diff. Immediately after this task lands and before any other Sprint 2 change, run it and keep the output as `before.txt` outside the repository; every later task diffs against it.
Acceptance criteria:
- The script prints the complete metric dictionaries of both causal cells for both stages, plus the phases and the anchor index, deterministically across two runs on one tree.
- The reference output is captured on the tree before S2-T02 starts and its location is recorded in this task's evidence.
Files affected:
- scripts/print_objective_metrics.py - existing; add the two causal cells and both stages.
- teb_vae/lag_attn_cfs/tests/conftest.py:make_stub_batch - existing; reused, unchanged.
Validation: `.venv/Scripts/python.exe scripts/print_objective_metrics.py > before.txt` twice and `diff` the two outputs; cwd repository root; prerequisites project `.venv`; expect no difference.
Test rationale: The script is the evidence instrument, not production code; determinism across two runs is the only property it needs and the diff establishes it.
Runtime: fast/local
Evidence: 2026-09-23 | `.venv/Scripts/python.exe scripts/print_objective_metrics.py` run twice and diffed | pass | identical 680-line outputs on the 2a2bc78 working tree before any Sprint 2 change; both causal cells at both stages with phases and anchor index; reference kept outside the repository in the session scratchpad as `before_baseline.txt`

#### S2-T02: Derive and validate the tile phase on the host

Requirements: FR-004, NFR-001
Depends on: S2-T01
Description: In `teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask`, override `transfer_batch_to_device(batch, device, dataloader_idx)` to remove `epoch` from the mapping, delegate to the inherited transfer for everything else, and put the host `epoch` back, so `anchor_phase` reads host values (its `_as_float` then costs no device round trip). In `anchor_phase`, build the phase tensor on the host as today and return it pinned (`pin_memory()`) when CUDA is available, otherwise unchanged. In `teb_vae/lag_attn_cfs/nets/causal_inputs.py:CausalWarmupInputs._build_anchor_index`, when the phase arrives as a host tensor, perform the range refusal on that host tensor and then move it with `non_blocking=True`; when it arrives already on the device, keep the current path. Update the docstrings that describe where the phase is built and moved (the task's `anchor_phase` and the net's `_build_anchor_index` both discuss it). The plotting callback calls the same override, so no change there.
Acceptance criteria:
- `teb_vae/lag_attn_cfs/tests/test_task.py` passes with `transfer_batch_to_device` added to its declared-member set (the set is asserted by name, so the override has to be listed) and the FR-005 script's phases and metrics are byte-identical to `before.txt`.
- A `meta`-device call of the override moves every tensor field and leaves `epoch` on the host.
- A phase outside $[0, S)$ is still refused by name, from a host tensor and from a device tensor.
- Under CUDA, the sync attribution test finds no synchronising operation in `anchor_phase` or `_build_anchor_index` during a train step.
Files affected:
- teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask.anchor_phase - existing; pinned host tensor.
- teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask - existing; add the `transfer_batch_to_device` override.
- teb_vae/lag_attn_cfs/nets/causal_inputs.py:CausalWarmupInputs._build_anchor_index - existing; host-side range check, non-blocking move.
- teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py (new) - CUDA-only attribution test and the `meta`-device override test.
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests/test_task.py teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py teb_vae/lag_attn_transformer_cfs/tests/test_task.py -q` and `.venv/Scripts/python.exe scripts/print_objective_metrics.py > after.txt && diff before.txt after.txt`; cwd repository root; prerequisites project `.venv`, `before.txt` from S2-T01, CUDA for the attribution test (it skips otherwise); expect all pass and an empty diff.
Test rationale: Existing phase tests cover the value; the new attribution test catches a re-introduced device read at these two sites; the `meta` test catches the override moving or dropping the wrong field.
Runtime: fast/local
Evidence: 2026-09-23 | `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests/test_task.py teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py teb_vae/lag_attn_transformer_cfs/tests/test_task.py -q` | partial | 84 passed, 1 failed: the attribution test's `_validate_anchors` count was 5, which is S2-T03's change; `anchor_phase`, `_as_float` and `_build_anchor_index` attributed no synchronisation; the meta-device transfer test and both range refusals passed
- 2026-09-23 | `scripts/print_objective_metrics.py` diffed against the reference | pass | empty diff
- 2026-09-23 | same pytest selection after S2-T03 | pass | 3 sync-test cases pass; see S2-T03

#### S2-T03: Validate the anchor set once per step and build constants without host copies

Requirements: FR-006, NFR-001
Depends on: S2-T02
Description: In `teb_vae/lag_attn_rws/nets/raw_masks.py`, add a keyword `validate: bool = True` to `forecast_mask` and `kl_mask` that skips `_validate_anchors` when false, and say in both docstrings that only a caller holding an index the model itself built may pass false. In `teb_vae/lag_attn_rws/nets/losses.py:compute_loss`, keep validation on the first `forecast_mask` call and pass `validate=False` to `kl_mask`; in `teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask._mu_gap_rms`, `teb_vae/lag_attn_cfs/nets/causal_feature_target.py:CausalFeatureForecastTarget._resolved_forecast_gaps` and `teb_vae/lag_attn_rws/nets/controls.py:source_null_kld`, pass `validate=False` for the index read off `forward_outputs`, with a one-line comment naming the objective's own validation as the one that ran. In `masked_raw_likelihood`, keep `elements` a 0-dim tensor (`cell_mask.sum().clamp_min(1.0)`) instead of a Python float and divide by it; in `compute_loss`, build the five echoes (`kld_beta`, `beta_prior`, `lambda_ms`, `lambda_deriv`, `lambda_boundary`) with `torch.full((), value, device=device, dtype=dtype)`. Leave `masked_source_kl` and the boolean-index diagnostics untouched (FR-008). Check the eval and attribution callers of `forecast_mask` and `kl_mask` (`grep` under `teb_vae/`) and leave them on the default.
Acceptance criteria:
- `teb_vae/lag_attn_rws/tests/test_raw_masks.py` passes unchanged; the refusals for an out-of-range, repeated or float anchor set still raise from `forecast_mask` and `kl_mask` at the default.
- The FR-005 script's output is byte-identical to `before.txt`.
- Under CUDA, the attribution test finds no synchronising operation in `masked_raw_likelihood` or the echo lines of `compute_loss` during a train step, and `_validate_anchors` runs exactly once per train step (the two device reads of that one call are its refusals and stay).
- `teb_vae/lag_attn_transformer_cfs/tests/test_objective.py` and `test_perm_control.py` pass.
Files affected:
- teb_vae/lag_attn_rws/nets/raw_masks.py:forecast_mask - existing; `validate` keyword.
- teb_vae/lag_attn_rws/nets/raw_masks.py:kl_mask - existing; `validate` keyword.
- teb_vae/lag_attn_rws/nets/losses.py:compute_loss - existing; validate once, `torch.full` echoes.
- teb_vae/lag_attn_rws/nets/losses.py:masked_raw_likelihood - existing; tensor element count.
- teb_vae/lag_attn_cfs/task.py:SeqVaeLagAttnCfsTask._mu_gap_rms - existing; pass `validate=False`.
- teb_vae/lag_attn_cfs/nets/causal_feature_target.py:CausalFeatureForecastTarget._resolved_forecast_gaps - existing; pass `validate=False`.
- teb_vae/lag_attn_rws/nets/controls.py:source_null_kld - existing; pass `validate=False`.
- teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py (new) - created by S2-T02; extend the attributed site list and count `_validate_anchors` calls.
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests/test_raw_masks.py teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"` and `.venv/Scripts/python.exe scripts/print_objective_metrics.py > after.txt && diff before.txt after.txt`; cwd repository root; prerequisites project `.venv`, `before.txt`; expect all pass and an empty diff.
Test rationale: The refusal tests already pin the guard; the attribution test and the byte diff are what catch a skipped validation that should have run or a moved number, which no existing test would see.
Runtime: fast/local
Evidence: 2026-09-23 | `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests/test_raw_masks.py teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"` | partial | 707 passed, 2 skipped, 2 failed: `test_docs.py::test_every_companion_document_the_design_defers_to_exists` (pre-existing, DESIGN.md names a document not in the tree, untouched here) and the sync test attributing the kept validation's two refusals to `_validate_anchors`, which contradicted "runs once per step"; acceptance amended to the count, see Change log
- 2026-09-23 | `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests/test_sync_free_step.py -q` | pass | 3 passed: no synchronisation in `anchor_phase`, `_as_float`, `_build_anchor_index`, `masked_raw_likelihood` or the echo lines; `_validate_anchors` called once
- 2026-09-23 | `scripts/print_objective_metrics.py` diffed against the reference | pass | empty diff
- 2026-09-23 | `teb_vae/lag_attn_cfs/tests -m "not slow"` and `teb_vae/lag_attn_rws/tests -m "not slow"` launched in the background | partial | both runs were stopped by the session for low system memory before reporting; rerun on the development machine from a terminal

#### S2-T04: Re-measure on the box and record

Requirements: FR-007
Depends on: S1-T03, S2-T02, S2-T03
Description: Repeat the S1-T03 protocol on the re-chunked shards with the Sprint 2 tree, and add the rows as a second dated amendment to the RESULTS.md table, labelled with the revision. Run the slow smoke suite once on the development machine before the box run.
Acceptance criteria:
- `teb_vae/lag_attn_transformer_cfs/tests/test_train_smoke.py` passes on the Sprint 2 tree.
- The RESULTS.md table carries the Sprint 2 rows beside Sprint 1's, with the revision and date.
Files affected:
- teb_vae/lag_attn_transformer_cfs/RESULTS.md - existing; dated amendment.
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests/test_train_smoke.py -q -m slow` (cwd repository root, project `.venv`) then the S1-T03 protocol on the box; expect the smoke suite to pass and the rows present.
Test rationale: The smoke suite is the integration check that every metric still reaches the logger after the Sprint 2 changes; the box run is the measurement.
Runtime: slow/external
Evidence: 2026-09-23 | `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests/test_train_smoke.py -q -m slow` in the background | partial | stopped by the session for low system memory before reporting; rerun from a terminal before the box run

## Change log

- 2026-09-23 | planner | Created roadmap from the 2026-09-23 measurements; two detailed sprints, seven required requirements, eight deferred levers with triggers; baseline 2a2bc78.
- 2026-09-23 | implementer | S1-T01, S1-T02, S2-T01, S2-T02, S2-T03 done on the 2a2bc78 working tree. Three document corrections: the re-chunk tool keeps the source `maxshape` (a fixed-shape 1-D dataset cannot carry a chunk wider than its rows); the cfs `test_task.py` member set gains `transfer_batch_to_device`; FR-006 and S2-T03 acceptance now pin one `_validate_anchors` call per step rather than a sync-free `_validate_anchors`, since its refusals are device reads. `test_docs.py` fails on a pre-existing missing companion document. S1-T03 and S2-T04 are operator measurements on the box.

## Resume

Current sprint: Sprint 1
Next task: S1-T03
Blockers: none
Last evidence: 2026-09-23 S2-T03 sync test 3 passed, metric dump byte-identical to the reference
Re-ground: S1-T03 and S2-T04 are operator runs on the production box; both code sprints are landed and validated here. Before S1-T03, re-chunk the box's shards with `hdf5_dataset/rechunk_hdf5.py` into a new directory and repoint the four `dataset_config` paths in the leaf config. S2-T04 can run in the same session as S1-T03's after-state since the Sprint 2 tree is what the box will check out.

## Todo checklist

### Sprint 1: Data path
- [x] S1-T01: Re-chunk tool for existing shards [done]
- [x] S1-T02: Per-sample chunks in the shard writer [done]
- [ ] S1-T03: Measure the loader and the step on the production box, before and after [ready]

### Sprint 2: Step overhead
- [x] S2-T01: Extend the bitwise metric dump to the causal cells [done]
- [x] S2-T02: Derive and validate the tile phase on the host [done]
- [x] S2-T03: Validate the anchor set once per step and build constants without host copies [done]
- [ ] S2-T04: Re-measure on the box and record [planned]
