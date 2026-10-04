# `lag_attn_transformer_patch` — the as-built design record

Written 2026-10-03. The approach and the specification are in `IMPLEMENTATION_PLAN.md` (Parts A and
B). This page records what was built, what moved from that plan and why, what the cell deliberately
does not do, and how to run it. Measurements are in `RESULTS.md`.

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
| `nets/patch_target.py` | `PatchSummaryTarget`: decoder width 2, `compute_loss`, the persistence input, and the AR(1) parameter (bound from CFT) |
| `nets/model.py` | The constructor, and nothing else |
| `task.py` | `SeqVaeLagAttnTrfPatchTask(SeqVaeLagAttnTrfCrwsTask)`: `_build_forward_inputs` and `_added_metrics` |
| `trainer.py` | `LagAttnTrfPatchTrainer`: seed handoff, tracked metrics, startup sentence, pre-flight |
| `summary_stats.py` | Prints `target_summary_loc`, `target_summary_scale` and `variability_eps` for a config's training shards |
| `lag_recovery_check.py` | The planted-delay instrument (P5-01) |
| `notes/CONTRACT.md` | The co-mixin contract map (P0-01) that the mixins were built against |
| `eval/` | The evaluation: the shared CFS pipeline bound through `eval/binding.py`, plus fourteen ported or raw-signal analyses (`eval/EVAL.md`) |

**Token layout.** `patchify` returns `(B, T, 33)` = `[value (16), delta (16), m_t − 1]`.
`m_t` is the minimum of the per-sample mask over the patch. A fully valid stream has 0 in the last
channel, so an all-zero stream means "valid and flat". That is the null the source-null control
feeds. An invalid token never reaches a norm layer as an exact zero vector. `PatchEmbedding`
replaces each invalid token with one learned `missing` vector through `torch.where`, with no Python
branch on a tensor value.

**Target.** `level` is the patch mean. `variability` is `log(rms(d) + eps)`, where `d` is the 15
first differences *inside* the patch. The affine `(s − loc) / scale` follows. The loss target and the
persistence input both go through `PatchSummaryTarget._standardized_summaries`. The target of anchor
`a` at step `τ` is token `a + 1 + τ`. The mask is the shared objective's own, built from `weight`.

**Geometry** (shipped): T = 300, R = 16, H = 30, F = 30, S = 15. `A_max` = 16 in training and 240
dense on val/test. The lag window is `max_lag` = 37. The scored block is H·C = 60 cells per anchor.

## 2. What moved from the plan, and why

Plan §C.3 records C1–C13 with the decision taken for each. The ones that change behaviour or
structure are:

- **The task parent (C6).** The task subclasses `SeqVaeLagAttnTrfCrwsTask` instead of binding four
  CFS members onto a `SeqVaeLagAttnTrfRwsTask`. That parent already carries the seed, the
  `_stage` class default that `VaeSource` relies on, the stage-setting `compute_loss_and_metrics`,
  the bound anchor-phase members and the LR ramp. Only the two members that name the input layout
  are written here.
- **The trainer (C7, C8, C10).** `create_model` hands the task `general_config.seed`; the
  conv-Transformer parent never does, so the anchor phase would key on 0. `TRACKED_METRICS` adds
  `anchors_per_sample` and `val/kld_source_null`. The pre-flight is one short refusal plus five
  imported guards; the e2e refusal reads a module global and could not be adapted.
- **Persistence (C2).** `_check_persistence_target` is a no-op, so `sweep_persistence.yaml`
  constructs.
- **Horizon weight (C5).** The loss passes `getattr(self, "horizon_weight", None)`. It is `None`
  at the shipped `null`, and a config that sets a half-life is honoured rather than ignored.
- **Summary constants (C12).** `default.yaml` ships the identity values. See §3.
- **Removed keys (C13).** `target_phase_fast_*`, `target_delays` and `source_delays` are refused by
  name, beside the B.5 list.

Smaller deviations, each found while building:

- **`horizon_film` defaults to `True`.** With `False` the horizon core cannot construct, because it
  hardcodes per-block FiLM. `SeqVaeLagAttnTrfCrws()` with defaults fails the same way today.
- **The constructor validates the new keys.** `loc` and `scale` must have 2 entries, `scale` must be
  above 0, `variability_eps` must be above 0, and `source_validity` must be `finite` or
  `fhr_weight`. With `eps = 0` a flat patch would give a target of `−inf`.
- **The affine uses Python floats per channel**, not a tensor, so a step does no host-to-device
  copy.
- **`nets/__init__.py` exports nothing**, as in the sibling packages.
- **`planted.yaml`** carries its own summary constants, measured on the planted shard, because the
  tiny shard is a different, synthetic signal. It also carries the CFS instrument's three leaves
  (`anchor_stride: 1`, `source_dropout: null`, `forecast_ar_residual: false`), so the two cells'
  planted readings are comparable (C14, the user's decision).
- **Tests.** The user asked for few tests, each able to fail. P4-01 … P4-08 were folded into 7
  files with 18 test functions in this package, plus one classifier test. The agents checked the
  invariant, objective, task and config tests by injecting the regression each names and seeing
  it fail. The fast gate runs in about 4 s, the slow pair in about 7 s.

## 3. Provisional values the user must set before a production run

| Leaf | Shipped | How to set it |
|---|---|---|
| `target_summary_loc`, `target_summary_scale`, `variability_eps` | `[0, 0]`, `[1, 1]`, `0.01` | `python teb_vae/lag_attn_transformer_patch/summary_stats.py --config teb_vae/lag_attn_transformer_patch/configs/default.yaml` on the production machine, then paste the printed YAML |
| `gradient_clip_val`, `additive_margin` | 280, 175 | The CFS values scaled by 60/2280. Re-derive from the first production run with `lag_attn_transformer_e2e/DESIGN.md` §12 |

The shards in `default.yaml` are the CFS paths. Any build works, because only `fhr`, `up` and
`weight` are read. For a comparison against CFS, use the same shard folds and the same stats file.

## 4. Deliberate limitations

> lean-limit: no forecast page. The plotting callback is disabled in every config. The inherited
> CRWS seams (`forecast_rows`, `input_stream_panels`) draw `(B, A, H, R)` raw rows and would
> misdraw a `(B, A, H, 2)` forecast. Add a summary-row page when one is wanted.

> lean-limit: the evaluation is a binding, not a fork. `eval/` runs the shared CFS pipeline
> (`teb_vae/lag_attn_cfs/eval`) on this model and adds the raw-signal analyses. Its own limits (the
> shared figures' lag-axis label, the rarely trained `missing` token, climatology under the identity
> summary constants) are listed in `eval/EVAL.md`.

> lean-limit: the UP validity rule is `finite`. Only non-finite UP samples are invalid; `weight`
> describes FHR. A disconnected UP transducer that writes zeros or a flat run reads as valid UP.
> `sweep_source_validity_fhr.yaml` measures the other choice. UP dropout, robust UP scaling and a
> flat-run detector are out of scope for v1 (plan A.6).

> lean-limit: the classifier's `kld_excess` key is not supported. Its gate looks for a forward
> parameter named `u_stream`; this forward names it `u_patch` on purpose, so the key is refused
> cleanly with `NotImplementedError` instead of being handed the anchor phase. `mu_prior`,
> `delta_mu` and `kld_per_t` load (`teb_vae/classifier/tests/test_patch_source.py`).

> lean-limit: the occlusion control (`controls.occluded_forward_outputs`) zeroes source rows, which
> this representation reads as "valid and flat", not as `missing`. Its only caller is the CFS
> evaluation; the patch eval's `occlusion` edits raw UP itself and calls it with `occlusion=None`.

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

# Local smoke: one epoch, one device, the committed 4-sample shard (about 6 s).
python -m teb_vae.lag_attn_transformer_patch.trainer \
    --config teb_vae/lag_attn_transformer_patch/configs/tiny.yaml

# The planted-delay instrument (about 35 s at 40 epochs on one GPU).
PYTHONPATH=. python teb_vae/lag_attn_transformer_patch/lag_recovery_check.py

# Evaluation (eval/EVAL.md): a production checkpoint, then the planted instrument's (about 9 min).
python -m teb_vae.lag_attn_transformer_patch.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_patch.eval.run \
    --checkpoint output/teb_vae_trf_patch_planted/<run>/model_checkpoints/<name>.ckpt \
    --overrides teb_vae/lag_attn_transformer_patch/eval/configs/planted_overrides.yaml

# Tests.
.venv/bin/python -m pytest teb_vae/lag_attn_transformer_patch/tests -q -m "not slow"
.venv/bin/python -m pytest teb_vae/lag_attn_transformer_patch/tests -q -m slow
.venv/bin/python -m pytest teb_vae/classifier/tests/test_patch_source.py -q
```
