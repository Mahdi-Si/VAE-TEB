# `lag_attn_transformer_patch` — design record

The approach and the specification are in `IMPLEMENTATION_PLAN.md` (Parts A and B). This page
records how the package is built, what the cell deliberately does not do, and how to run it.

## 1. What the model is

`SeqVaeLagAttnTrfPatch` is the CFS cell (`lag_attn_transformer_cfs`) with a different
representation. Each stream is cut into non-overlapping 4 s patches of 16 raw samples, one token per
patch. The forecast target is the next 30 per-patch summaries, `[level, variability]`. The encoders,
prior, lag attention, posterior, decoder, objective and controls are imported unchanged.

```
SeqVaeLagAttnTrfPatch -> PatchStreamInputs -> CausalWarmupInputs -> PatchSummaryTarget
                      -> SeqVaeLagAttnTrfRws -> nn.Module
```

| File | Supplies |
|---|---|
| `nets/patching.py` | `patchify`, `patch_summaries` (the one definition of the target) and `PatchEmbedding` |
| `nets/patch_inputs.py` | `PatchStreamInputs`: both adapters are `PatchEmbedding`; the CFS tiled-anchor forward; no-op warm-up hooks |
| `nets/patch_target.py` | `PatchSummaryTarget`: decoder width 2, `compute_loss`, `_build_forecast_target`, the persistence input, and the AR(1) parameter (bound from CFT) |
| `nets/model.py` | The constructor, and nothing else |
| `task.py` | `SeqVaeLagAttnTrfPatchTask(SeqVaeLagAttnTrfCrwsTask)`: `_build_forward_inputs`, `_added_metrics` and the four diagnostic-page seams |
| `trainer.py` | `LagAttnTrfPatchTrainer`: seed handoff, tracked metrics, startup sentence, pre-flight |
| `summary_stats.py` | Prints `target_summary_loc`, `target_summary_scale` and `variability_eps` for a config's training shards |
| `lag_recovery_check.py` | The planted-delay check (P5-01) |
| `notes/CONTRACT.md` | The co-mixin contract map (P0-01) that the mixins were built against |
| `eval/` | The evaluation: the shared CFS pipeline bound through `eval/binding.py`, plus fourteen ported or raw-signal analyses (`eval/EVAL.md`) |

**Token layout.** `patchify` returns `(B, T, 33)` = `[value (16), delta (16), m_t − 1]`.
$m_t$ is the minimum of the per-sample mask over the patch. A fully valid stream has 0 in the last
channel, so an all-zero stream means "valid and flat". That is the null the source-null control
feeds. An invalid token never reaches a norm layer as an exact zero vector. `PatchEmbedding`
replaces each invalid token with one learned `missing` vector through `torch.where`, with no Python
branch on a tensor value.

**Target.** `level` is the patch mean. `variability` is $\log(\mathrm{rms}(d) + \varepsilon)$,
where $d$ is the 15 first differences *inside* the patch. The affine $(s - \mathrm{loc}) /
\mathrm{scale}$ follows. The loss target and the persistence input both go through
`PatchSummaryTarget._standardized_summaries`. The target of anchor $a$ at step $\tau$ is token
$a + 1 + \tau$; `_build_forecast_target` is that gather, and both `compute_loss` and the shared
evaluation call it. The mask is the shared objective's own, built from `weight`.

**Geometry** (shipped): $T = 300$, $R = 16$, $H = 30$, $F = 30$, $S = 15$. $A_{\max} = 16$ in
training and 240 dense on val/test. The lag window is `max_lag` = 37. The scored block is
$H \cdot C = 60$ cells per anchor.

## 2. Implementation details

The IDs C1–C14 refer to the decisions in `IMPLEMENTATION_PLAN.md` §C.3. The ones that change
behaviour or structure are:

- **The task parent (C6).** The task subclasses `SeqVaeLagAttnTrfCrwsTask`. That parent already
  carries the seed, the `_stage` class default that `VaeSource` relies on, the stage-setting
  `compute_loss_and_metrics`, the bound anchor-phase members and the LR ramp. Only the members that
  name the input layout and the page seams are written here.
- **The trainer (C7, C8, C10).** `create_model` hands the task `general_config.seed`; the
  conv-Transformer parent never does, so without it the anchor phase would key on 0.
  `TRACKED_METRICS` adds `train/anchors_per_sample`, `val/anchors_per_sample` and
  `val/kld_source_null`. The pre-flight is one short refusal plus five imported guards; the e2e
  refusal reads a module global and cannot take another key list.
- **Persistence (C2).** `_check_persistence_target` is a no-op, so `sweep_persistence.yaml`
  constructs.
- **Horizon weight (C5).** The loss passes `getattr(self, "horizon_weight", None)`. It is `None`
  at the shipped `null`, and a config that sets a half-life is honoured rather than ignored.
- **Summary constants (C12).** `default.yaml` ships the identity values. See §3.
- **Removed keys (C8, C13).** The pre-flight refuses every key in `REMOVED_MODEL_KEYS` and every key
  that starts with `causal_`, `target_weight_` or `target_phase_fast_`. The signature sweep would
  otherwise drop each one in silence.

Smaller details:

- **`horizon_film` defaults to `True`.** With `False` the horizon core cannot construct, because it
  hardcodes per-block FiLM.
- **The constructor validates the new keys.** `loc` and `scale` must have 2 entries, `scale` must be
  above 0, `variability_eps` must be above 0, and `source_validity` must be `finite` or
  `fhr_weight`. With $\varepsilon = 0$ a flat patch would give a target of $-\infty$.
- **The affine uses Python floats per channel**, not a tensor, so a step does no host-to-device
  copy.
- **`nets/__init__.py` exports nothing**, as in the sibling packages.
- **`planted.yaml`** carries its own summary constants from `summary_stats.py` on the planted shard,
  because the tiny shard is a different, synthetic signal. It sets `max_lag: 60`, so the informative
  band $[15, 44]$ lies inside the lag window. It also carries the three leaves of
  `lag_attn_transformer_cfs/configs/planted.yaml`: `anchor_stride: 1`, `source_dropout: null` and
  `forecast_ar_residual: false` (C14).
- **Tests.** Eleven files in `tests/`, plus `teb_vae/classifier/tests/test_patch_source.py`:

  | File | What it checks |
  |---|---|
  | `test_patching.py` | The token layout and validity channel, the summary target against NumPy, and the `missing` swap |
  | `test_model.py` | Raw causality, source purity, one decoder invoked twice, zero KL at init, the lag-map identity, DDP reachability, no tensor-value branch |
  | `test_objective.py` | The target and persistence gathers, AR(1) at $\phi = 0$ against the factorised score, invalid patches not scored |
  | `test_task.py` | The patch streams and anchor geometry per stage, and the source-null control |
  | `test_config_load.py` | Every key reaches the model or task, the removed keys are refused, and each config differs from its reference only in its declared leaves |
  | `test_train_smoke.py` (`slow`) | A `tiny.yaml` fit is finite and trains every `PatchEmbedding` tensor |
  | `test_lag_recovery.py` (`slow`) | `lag_recovery_check.py` runs and reports finite numbers |
  | `test_diagnostic_page.py` | The plotting callback writes the patch page |
  | `test_eval_binding.py`, `test_eval_raw.py`, `test_eval_run.py` | The eval registry and geometry keys, the raw forward and raw IG, and one end-to-end eval run |

## 3. Provisional values the user must set before a production run

| Leaf | Shipped | How to set it |
|---|---|---|
| `target_summary_loc`, `target_summary_scale`, `variability_eps` | `[0, 0]`, `[1, 1]`, `0.01` | `python teb_vae/lag_attn_transformer_patch/summary_stats.py --config teb_vae/lag_attn_transformer_patch/configs/default.yaml` on the production machine, then paste the printed YAML |
| `gradient_clip_val`, `additive_margin` | 280, 175 | The CFS values scaled by 60/2280. Re-derive them with the procedure in `lag_attn_transformer_e2e/DESIGN.md` §12 |

The shards in `default.yaml` are the CFS paths. Any build works, because only `fhr`, `up` and
`weight` are read. For a comparison against CFS, use the same shard folds and the same stats file.

## 4. Deliberate limitations

> lean-limit: the run-level input-budget figure is not drawn. A patch token has no per-channel
> warm-up, so the task's `input_budget_figure` returns `None`. The per-epoch diagnostic page is drawn:
> the task's page seams hand the plotting callback the evaluation's patch page
> (`eval/analyses/samples.py`), so the training page and the evaluation page are one figure.

> lean-limit: the evaluation is a binding, not a fork. `eval/` runs the shared CFS pipeline
> (`teb_vae/lag_attn_cfs/eval`) on this model and adds the raw-signal analyses. Its own limits (the
> shared figures' lag-axis label, the rarely trained `missing` token, climatology under the identity
> summary constants) are listed in `eval/EVAL.md`.

> lean-limit: the UP validity rule is `finite`. Only non-finite UP samples are invalid; `weight`
> describes FHR. A disconnected UP transducer that writes zeros or a flat run reads as valid UP.
> `sweep_source_validity_fhr.yaml` selects the other rule. UP dropout, robust UP scaling and a
> flat-run detector are out of scope for v1 (plan A.6).

> lean-limit: the classifier's `kld_excess` key is not supported. Its gate looks for a forward
> parameter named `u_stream`; this forward names it `u_patch` on purpose, so the key is refused
> cleanly with `NotImplementedError` instead of being handed the anchor phase. `mu_prior`,
> `delta_mu` and `kld_per_t` load (`teb_vae/classifier/tests/test_patch_source.py`).

> lean-limit: the occlusion control (`controls.occluded_forward_outputs`) zeroes source rows, which
> this representation reads as "valid and flat", not as `missing`. The patch eval's `occlusion`
> edits raw UP itself and calls the control with `occlusion=None`.

> lean-limit: the committed fixture shards have no FHR gap and no non-finite UP sample, so a fit on
> them never trains `missing`. The train smoke test plants a gap in a temporary copy of the shard.

## 5. Running it

From the repository root:

```bash
# Target constants for the production shards (paste the output into configs/default.yaml).
python teb_vae/lag_attn_transformer_patch/summary_stats.py \
    --config teb_vae/lag_attn_transformer_patch/configs/default.yaml

# Production, 7 ranks. The rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_patch.trainer \
    --config teb_vae/lag_attn_transformer_patch/configs/default.yaml

# An arm: the same command with configs/sweep_<name>.yaml.

# Local smoke: one epoch, one device, the committed 4-sample shard.
python -m teb_vae.lag_attn_transformer_patch.trainer \
    --config teb_vae/lag_attn_transformer_patch/configs/tiny.yaml

# The planted-delay check: fits planted.yaml and prints the band mass and the lag argmax.
PYTHONPATH=. python teb_vae/lag_attn_transformer_patch/lag_recovery_check.py

# Evaluation (eval/EVAL.md): a production checkpoint, then the planted instrument's.
python -m teb_vae.lag_attn_transformer_patch.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_patch.eval.run \
    --checkpoint output/teb_vae_trf_patch_planted/<run>/model_checkpoints/<name>.ckpt \
    --overrides teb_vae/lag_attn_transformer_patch/eval/configs/planted_overrides.yaml

# Tests.
python -m pytest teb_vae/lag_attn_transformer_patch/tests -q -m "not slow"
python -m pytest teb_vae/lag_attn_transformer_patch/tests -q -m slow
python -m pytest teb_vae/classifier/tests/test_patch_source.py -q
```

Each entry point also runs from the IDE's Run button with no command line. `trainer.py` reads the
config path from `RUN_CONFIG`. `summary_stats.py`, `lag_recovery_check.py`, `eval/run.py` and
`eval/verify.py` read their values from a `RUN_ARGS` dictionary at the bottom of the file. A
command-line flag wins over the value in the file.
